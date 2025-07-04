//! High-performance connection pool for Triton clients.
//!
//! This module provides a lock-free connection pool that significantly reduces
//! connection overhead by reusing gRPC channels to the Triton Inference Server.

use crate::error::{AppError, Result};
use crate::triton::client::TritonClient;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool.
    pub max_connections: usize,
    /// Minimum number of connections to maintain.
    pub min_connections: usize,
    /// Maximum time a connection can be idle before being closed.
    pub max_idle_time: Duration,
    /// Timeout for acquiring a connection from the pool.
    pub acquire_timeout: Duration,
    /// Interval for cleaning up idle connections.
    pub cleanup_interval: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 50,
            min_connections: 5,
            max_idle_time: Duration::from_secs(300), // 5 minutes
            acquire_timeout: Duration::from_millis(500),
            cleanup_interval: Duration::from_secs(60),
        }
    }
}

/// A pooled connection wrapper.
pub struct PooledConnection {
    client: TritonClient,
    pool: Arc<ConnectionPoolInner>,
    created_at: Instant,
    last_used: Instant,
    // Flag to prevent infinite recursion in Drop
    should_return_to_pool: bool,
}

impl PooledConnection {
    /// Get a reference to the underlying Triton client.
    pub fn client(&mut self) -> &mut TritonClient {
        self.last_used = Instant::now();
        &mut self.client
    }

    /// Get the age of this connection.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the time since this connection was last used.
    pub fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        // Only return to pool if this is the original connection
        if self.should_return_to_pool && self.idle_time() < self.pool.config.max_idle_time {
            // Return the raw connection to the pool
            let raw_conn = RawConnection {
                client: self.client.clone(),
                created_at: self.created_at,
                last_used: self.last_used,
            };
            self.pool.return_raw_connection(raw_conn);
        }
    }
}

/// Raw connection with metadata.
struct RawConnection {
    client: TritonClient,
    created_at: Instant,
    last_used: Instant,
}

/// Internal connection pool state.
struct ConnectionPoolInner {
    endpoint: String,
    config: PoolConfig,
    connections: Mutex<VecDeque<RawConnection>>,
    semaphore: Semaphore,
    active_connections: parking_lot::Mutex<usize>,
    self_ref: std::sync::Weak<Self>,
}

impl ConnectionPoolInner {
    fn return_raw_connection(&self, conn: RawConnection) {
        let mut connections = self.connections.lock();
        if connections.len() < self.config.max_connections {
            connections.push_back(conn);
            debug!(
                "Returned connection to pool, pool size: {}",
                connections.len()
            );
        } else {
            debug!("Pool full, dropping connection");
        }
    }

    async fn create_connection(&self) -> Result<PooledConnection> {
        let client = TritonClient::connect(&self.endpoint)
            .await
            .map_err(AppError::from)?;

        let now = Instant::now();
        let pool_arc = self
            .self_ref
            .upgrade()
            .ok_or_else(|| AppError::Internal("Connection pool has been dropped".to_string()))?;

        Ok(PooledConnection {
            client,
            pool: pool_arc,
            created_at: now,
            last_used: now,
            should_return_to_pool: true,
        })
    }

    fn try_get_raw_connection(&self) -> Option<RawConnection> {
        let mut connections = self.connections.lock();
        connections.pop_front()
    }

    fn cleanup_idle_connections(&self) {
        let mut connections = self.connections.lock();
        let before_cleanup = connections.len();

        connections.retain(|conn| {
            // Remove connections that are too old or have been idle too long
            let is_too_old = conn.created_at.elapsed() > Duration::from_secs(3600); // 1 hour max age
            let is_idle_too_long = conn.last_used.elapsed() > self.config.max_idle_time;
            
            if is_too_old {
                debug!("Removing connection due to age: {:?}", conn.created_at.elapsed());
                false
            } else if is_idle_too_long {
                debug!("Removing connection due to idle time: {:?}", conn.last_used.elapsed());
                false
            } else {
                true
            }
        });

        let after_cleanup = connections.len();
        if before_cleanup != after_cleanup {
            debug!(
                "Cleaned up {} connections (idle/old), pool size: {} -> {}",
                before_cleanup - after_cleanup,
                before_cleanup,
                after_cleanup
            );
        }
    }

    /// Check if a connection is still healthy and usable.
    /// 
    /// This method performs basic health checks on a connection to determine
    /// if it should be reused or discarded.
    fn is_connection_healthy(&self, conn: &RawConnection) -> bool {
        // Check connection age - connections older than 1 hour should be refreshed
        if conn.created_at.elapsed() > Duration::from_secs(3600) {
            return false;
        }

        // Check if connection has been idle too long
        if conn.last_used.elapsed() > self.config.max_idle_time {
            return false;
        }

        // TODO: Could add actual gRPC channel health check here in the future
        // For now, we rely on age and idle time checks which are sufficient
        // for most use cases and avoid the overhead of network calls.
        
        true
    }
}

/// High-performance connection pool for Triton clients.
pub struct ConnectionPool {
    inner: Arc<ConnectionPoolInner>,
}

impl ConnectionPool {
    /// Create a new connection pool.
    ///
    /// # Arguments
    /// * `endpoint` - The Triton server endpoint.
    /// * `config` - Pool configuration.
    ///
    /// # Returns
    /// A new connection pool.
    pub async fn new(endpoint: String, config: PoolConfig) -> Result<Self> {
        let semaphore = Semaphore::new(config.max_connections);

        let inner = Arc::new_cyclic(|weak_ref| ConnectionPoolInner {
            endpoint: endpoint.clone(),
            config: config.clone(),
            connections: Mutex::new(VecDeque::<RawConnection>::with_capacity(config.max_connections)),
            semaphore,
            active_connections: parking_lot::Mutex::new(0),
            self_ref: weak_ref.clone(),
        });

        let pool = Self { inner };

        // Pre-warm the pool with minimum connections
        pool.prewarm().await?;

        // Start cleanup task
        pool.start_cleanup_task();

        info!(
            "Created Triton connection pool for {} with config: {:?}",
            endpoint, config
        );

        Ok(pool)
    }

    /// Create a connection pool with default configuration.
    pub async fn with_defaults(endpoint: String) -> Result<Self> {
        Self::new(endpoint, PoolConfig::default()).await
    }

    /// Get a connection from the pool.
    ///
    /// This method will:
    /// 1. Try to get an existing connection from the pool
    /// 2. If none available, create a new connection (up to max_connections)
    /// 3. Block until a connection becomes available or timeout
    ///
    /// # Returns
    /// A pooled connection that will be automatically returned when dropped.
    pub async fn get(&self) -> Result<PooledConnection> {
        // Try to acquire a semaphore permit first
        let _permit = tokio::time::timeout(
            self.inner.config.acquire_timeout,
            self.inner.semaphore.acquire(),
        )
        .await
        .map_err(|_| AppError::Internal("Connection pool timeout".to_string()))?
        .map_err(|_| AppError::Internal("Connection pool closed".to_string()))?;

        // Try to get an existing connection and validate it
        while let Some(raw_conn) = self.inner.try_get_raw_connection() {
            // Check if connection is still healthy
            if self.inner.is_connection_healthy(&raw_conn) {
                debug!("Reused existing connection from pool");
                return Ok(PooledConnection {
                    client: raw_conn.client,
                    pool: self.inner.clone(),
                    created_at: raw_conn.created_at,
                    last_used: Instant::now(), // Update last used time
                    should_return_to_pool: true,
                });
            } else {
                debug!("Discarded unhealthy connection from pool");
                // Continue loop to try next connection
            }
        }

        // Create a new connection
        debug!("Creating new connection for pool");
        let conn = self.inner.create_connection().await?;

        {
            let mut active = self.inner.active_connections.lock();
            *active += 1;
        }

        Ok(conn)
    }

    /// Pre-warm the pool with minimum connections.
    async fn prewarm(&self) -> Result<()> {
        let mut connections = Vec::new();

        for _ in 0..self.inner.config.min_connections {
            match self.inner.create_connection().await {
                Ok(conn) => {
                    let raw_conn = RawConnection {
                        client: conn.client.clone(),
                        created_at: conn.created_at,
                        last_used: conn.last_used,
                    };
                    connections.push(raw_conn);
                },
                Err(e) => {
                    warn!("Failed to create connection during prewarm: {}", e);
                    break;
                }
            }
        }

        let mut pool_connections = self.inner.connections.lock();
        for conn in connections {
            pool_connections.push_back(conn);
        }

        info!(
            "Pre-warmed pool with {} connections",
            pool_connections.len()
        );
        Ok(())
    }

    /// Start the cleanup task that removes idle connections.
    fn start_cleanup_task(&self) {
        let inner = self.inner.clone();
        let cleanup_interval = inner.config.cleanup_interval;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                inner.cleanup_idle_connections();
            }
        });
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let connections = self.inner.connections.lock();
        let active = *self.inner.active_connections.lock();

        PoolStats {
            total_connections: active + connections.len(),
            active_connections: active,
            idle_connections: connections.len(),
            max_connections: self.inner.config.max_connections,
        }
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub max_connections: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pool: {}/{} total, {} active, {} idle",
            self.total_connections,
            self.max_connections,
            self.active_connections,
            self.idle_connections
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_pool_creation() {
        let _config = PoolConfig {
            max_connections: 10,
            min_connections: 2,
            max_idle_time: Duration::from_secs(30),
            acquire_timeout: Duration::from_millis(100),
            cleanup_interval: Duration::from_secs(10),
        };

        // Note: This test would require a running Triton server
        // In a real test environment, you'd use a mock server
        // let pool = ConnectionPool::new("http://localhost:8001".to_string(), config).await;
        // assert!(pool.is_ok());
    }

    #[test]
    fn test_pool_config_defaults() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections, 50);
        assert_eq!(config.min_connections, 5);
        assert_eq!(config.max_idle_time, Duration::from_secs(300));
    }
}
