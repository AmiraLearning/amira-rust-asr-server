//! Graceful shutdown handling for production reliability.
//!
//! Provides utilities to handle shutdown signals gracefully, allowing active
//! requests to complete and resources to be cleaned up properly.

use std::time::Duration;
use tokio::signal;
use tokio::sync::broadcast;
// Temporary tracing macros while resolving external dependencies
macro_rules! debug { ($($tt:tt)*) => {}; }
macro_rules! error { ($($tt:tt)*) => {}; }
macro_rules! info { ($($tt:tt)*) => {}; }
macro_rules! warn { ($($tt:tt)*) => {}; }

/// Handle for managing graceful shutdown.
#[derive(Clone)]
pub struct GracefulShutdown {
    /// Sender for shutdown signal.
    shutdown_tx: broadcast::Sender<()>,
}

impl GracefulShutdown {
    /// Create a new graceful shutdown handler.
    pub fn new() -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self { shutdown_tx }
    }

    /// Get a receiver for shutdown signals.
    pub fn subscribe(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }

    /// Trigger graceful shutdown.
    pub fn shutdown(&self) {
        info!("Initiating graceful shutdown");
        if let Err(e) = self.shutdown_tx.send(()) {
            warn!("Failed to send shutdown signal: {}", e);
        }
    }

    /// Wait for shutdown signals (SIGINT, SIGTERM).
    pub async fn wait_for_signal(&self) {
        let shutdown_tx = self.shutdown_tx.clone();

        tokio::spawn(async move {
            let ctrl_c = async {
                if let Err(e) = signal::ctrl_c().await {
                    error!("Failed to install Ctrl+C handler: {}", e);
                    // If we can't install Ctrl+C handler, we'll rely on SIGTERM only
                    return;
                }
            };

            #[cfg(unix)]
            let terminate = async {
                match signal::unix::signal(signal::unix::SignalKind::terminate()) {
                    Ok(mut stream) => {
                        stream.recv().await;
                    }
                    Err(e) => {
                        error!("Failed to install SIGTERM handler: {}", e);
                        // If we can't install signal handlers, we'll wait indefinitely
                        // This is not ideal but prevents panic
                        std::future::pending::<()>().await;
                    }
                }
            };

            #[cfg(not(unix))]
            let terminate = std::future::pending::<()>();

            tokio::select! {
                _ = ctrl_c => {
                    info!("Received Ctrl+C signal");
                },
                _ = terminate => {
                    info!("Received SIGTERM signal");
                }
            }

            info!("Shutdown signal received, initiating graceful shutdown");
            if let Err(e) = shutdown_tx.send(()) {
                error!("Failed to broadcast shutdown signal: {}", e);
            }
        });
    }
}

impl Default for GracefulShutdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for graceful shutdown of various components.
pub struct ShutdownGuard {
    name: String,
    shutdown_rx: broadcast::Receiver<()>,
}

impl ShutdownGuard {
    /// Create a new shutdown guard for a component.
    pub fn new(name: impl Into<String>, shutdown_handler: &GracefulShutdown) -> Self {
        Self {
            name: name.into(),
            shutdown_rx: shutdown_handler.subscribe(),
        }
    }

    /// Wait for shutdown signal or until the guard is dropped.
    pub async fn wait_for_shutdown(&mut self) {
        match self.shutdown_rx.recv().await {
            Ok(()) => {
                info!("Component '{}' received shutdown signal", self.name);
            }
            Err(broadcast::error::RecvError::Closed) => {
                debug!("Shutdown channel closed for component '{}'", self.name);
            }
            Err(broadcast::error::RecvError::Lagged(_)) => {
                warn!("Component '{}' lagged behind shutdown signal", self.name);
            }
        }
    }

    /// Wait for shutdown with a timeout.
    pub async fn wait_for_shutdown_with_timeout(&mut self, timeout: Duration) -> bool {
        tokio::select! {
            result = self.shutdown_rx.recv() => {
                match result {
                    Ok(()) => {
                        info!("Component '{}' received shutdown signal", self.name);
                        true
                    }
                    Err(e) => {
                        warn!("Component '{}' shutdown error: {}", self.name, e);
                        true
                    }
                }
            }
            _ = tokio::time::sleep(timeout) => {
                warn!("Component '{}' shutdown timed out after {:?}", self.name, timeout);
                false
            }
        }
    }
}

/// Helper function to perform graceful shutdown with active connection tracking.
pub async fn shutdown_with_grace<F, Fut>(
    shutdown_handler: &GracefulShutdown,
    active_connections: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    cleanup_fn: F,
    grace_period: Duration,
) where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut shutdown_guard = ShutdownGuard::new("main_server", shutdown_handler);

    // Wait for shutdown signal
    shutdown_guard.wait_for_shutdown().await;

    info!("Graceful shutdown initiated");

    // Wait for active connections to finish
    let start = std::time::Instant::now();
    while active_connections.load(std::sync::atomic::Ordering::Relaxed) > 0 {
        if start.elapsed() >= grace_period {
            warn!(
                "Grace period ({:?}) exceeded with {} active connections remaining",
                grace_period,
                active_connections.load(std::sync::atomic::Ordering::Relaxed)
            );
            break;
        }

        debug!(
            "Waiting for {} active connections to finish...",
            active_connections.load(std::sync::atomic::Ordering::Relaxed)
        );

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Perform cleanup
    cleanup_fn().await;

    info!("Graceful shutdown completed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn test_graceful_shutdown_creation() {
        let shutdown = GracefulShutdown::new();
        let mut guard = ShutdownGuard::new("test", &shutdown);

        // Test that we can create subscribers
        let _rx1 = shutdown.subscribe();
        let _rx2 = shutdown.subscribe();

        // Trigger shutdown
        shutdown.shutdown();

        // Guard should receive shutdown signal
        let result = timeout(Duration::from_millis(100), guard.wait_for_shutdown()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_shutdown_guard_timeout() {
        let shutdown = GracefulShutdown::new();
        let mut guard = ShutdownGuard::new("test", &shutdown);

        // Should timeout since no shutdown signal is sent
        let received = guard
            .wait_for_shutdown_with_timeout(Duration::from_millis(50))
            .await;
        assert!(!received);
    }

    #[tokio::test]
    async fn test_shutdown_with_grace() {
        let shutdown = GracefulShutdown::new();
        let active_connections = Arc::new(AtomicUsize::new(2));
        let cleanup_called = Arc::new(AtomicUsize::new(0));

        let active_connections_clone = Arc::clone(&active_connections);
        let cleanup_called_clone = Arc::clone(&cleanup_called);

        // Simulate connections finishing during grace period
        tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            active_connections_clone.store(1, Ordering::Relaxed);
            sleep(Duration::from_millis(10)).await;
            active_connections_clone.store(0, Ordering::Relaxed);
        });

        // Start shutdown process
        let shutdown_clone = shutdown.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(5)).await;
            shutdown_clone.shutdown();
        });

        shutdown_with_grace(
            &shutdown,
            active_connections,
            || async move {
                cleanup_called_clone.store(1, Ordering::Relaxed);
            },
            Duration::from_millis(100),
        )
        .await;

        assert_eq!(cleanup_called.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let shutdown = GracefulShutdown::new();

        let mut guard1 = ShutdownGuard::new("test1", &shutdown);
        let mut guard2 = ShutdownGuard::new("test2", &shutdown);

        // Trigger shutdown
        shutdown.shutdown();

        // Both guards should receive the signal
        let result1 = timeout(Duration::from_millis(100), guard1.wait_for_shutdown()).await;
        let result2 = timeout(Duration::from_millis(100), guard2.wait_for_shutdown()).await;

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }
}
