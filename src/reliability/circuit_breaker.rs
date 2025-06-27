//! Circuit breaker implementation for Triton client calls.
//!
//! Provides fault tolerance by automatically failing fast when the Triton
//! server is unavailable or responding slowly, preventing cascading failures.

use crate::error::{AppError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{debug, error, warn};

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed - requests are allowed through.
    Closed,
    /// Circuit is open - requests fail fast.
    Open,
    /// Circuit is half-open - limited requests are allowed to test recovery.
    HalfOpen,
}

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit.
    pub failure_threshold: u64,
    /// Time to wait before transitioning from open to half-open.
    pub recovery_timeout: Duration,
    /// Timeout for individual requests.
    pub request_timeout: Duration,
    /// Number of successful requests needed to close the circuit from half-open.
    pub success_threshold: u64,
    /// Window size for tracking failures.
    pub window_size: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(10),
            success_threshold: 3,
            window_size: Duration::from_secs(60),
        }
    }
}

/// Metrics tracked by the circuit breaker.
#[derive(Debug, Default)]
pub struct CircuitBreakerMetrics {
    /// Total number of requests processed.
    pub total_requests: AtomicU64,
    /// Number of successful requests.
    pub successful_requests: AtomicU64,
    /// Number of failed requests.
    pub failed_requests: AtomicU64,
    /// Number of requests rejected due to open circuit.
    pub rejected_requests: AtomicU64,
    /// Number of times the circuit has opened.
    pub circuit_opens: AtomicU64,
    /// Number of times the circuit has closed.
    pub circuit_closes: AtomicU64,
}

impl CircuitBreakerMetrics {
    /// Get the current failure rate as a percentage.
    pub fn failure_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let failed = self.failed_requests.load(Ordering::Relaxed);
        (failed as f64 / total as f64) * 100.0
    }

    /// Get the current success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        100.0 - self.failure_rate()
    }
}

/// Circuit breaker for protecting against cascading failures.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// Current state of the circuit.
    state: parking_lot::RwLock<CircuitState>,
    /// Configuration settings.
    config: CircuitBreakerConfig,
    /// Metrics and counters.
    metrics: Arc<CircuitBreakerMetrics>,
    /// Timestamp of the last state change.
    last_state_change: parking_lot::RwLock<Instant>,
    /// Recent failure timestamps for sliding window.
    recent_failures: parking_lot::RwLock<Vec<Instant>>,
    /// Recent success count in half-open state.
    half_open_successes: AtomicU64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given configuration.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: parking_lot::RwLock::new(CircuitState::Closed),
            config,
            metrics: Arc::new(CircuitBreakerMetrics::default()),
            last_state_change: parking_lot::RwLock::new(Instant::now()),
            recent_failures: parking_lot::RwLock::new(Vec::new()),
            half_open_successes: AtomicU64::new(0),
        }
    }

    /// Create a circuit breaker with default configuration.
    pub fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }

    /// Get the current state of the circuit breaker.
    pub fn state(&self) -> CircuitState {
        *self.state.read()
    }

    /// Get a reference to the metrics.
    pub fn metrics(&self) -> Arc<CircuitBreakerMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Execute a function with circuit breaker protection.
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = std::result::Result<T, E>>,
        E: Into<AppError>,
    {
        // Check if we should allow the request
        if !self.should_allow_request() {
            self.metrics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            return Err(AppError::ServiceUnavailable(
                "Circuit breaker is open - service is unavailable".to_string(),
            ));
        }

        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

        // Execute the operation with timeout
        let result = timeout(self.config.request_timeout, operation).await;

        match result {
            Ok(Ok(value)) => {
                // Success - record it and potentially close circuit
                self.record_success().await;
                Ok(value)
            }
            Ok(Err(e)) => {
                // Operation failed - record failure and potentially open circuit
                self.record_failure().await;
                Err(e.into())
            }
            Err(_) => {
                // Timeout - treat as failure
                self.record_failure().await;
                Err(AppError::Timeout(format!(
                    "Operation timed out after {:?}",
                    self.config.request_timeout
                )))
            }
        }
    }

    /// Check if a request should be allowed through the circuit breaker.
    fn should_allow_request(&self) -> bool {
        let current_state = *self.state.read();

        match current_state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                let last_change = *self.last_state_change.read();
                if last_change.elapsed() >= self.config.recovery_timeout {
                    self.transition_to_half_open();
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful operation.
    async fn record_success(&self) {
        self.metrics
            .successful_requests
            .fetch_add(1, Ordering::Relaxed);

        let current_state = *self.state.read();
        if current_state == CircuitState::HalfOpen {
            let successes = self.half_open_successes.fetch_add(1, Ordering::Relaxed) + 1;
            if successes >= self.config.success_threshold {
                self.transition_to_closed();
            }
        }

        debug!(
            "Circuit breaker recorded success, state: {:?}",
            current_state
        );
    }

    /// Record a failed operation.
    async fn record_failure(&self) {
        self.metrics.failed_requests.fetch_add(1, Ordering::Relaxed);

        // Add failure to sliding window
        {
            let mut recent_failures = self.recent_failures.write();
            let now = Instant::now();
            recent_failures.push(now);

            // Remove old failures outside the window
            recent_failures.retain(|&failure_time| {
                now.duration_since(failure_time) <= self.config.window_size
            });
        }

        // Check if we should open the circuit
        let current_state = *self.state.read();
        match current_state {
            CircuitState::Closed => {
                let failure_count = self.recent_failures.read().len() as u64;
                if failure_count >= self.config.failure_threshold {
                    self.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state opens the circuit
                self.transition_to_open();
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }

        warn!(
            "Circuit breaker recorded failure, state: {:?}",
            current_state
        );
    }

    /// Transition the circuit breaker to the open state.
    fn transition_to_open(&self) {
        {
            let mut state = self.state.write();
            if *state != CircuitState::Open {
                *state = CircuitState::Open;
                *self.last_state_change.write() = Instant::now();
                self.metrics.circuit_opens.fetch_add(1, Ordering::Relaxed);
                error!("Circuit breaker opened due to failures");
            }
        }
    }

    /// Transition the circuit breaker to the half-open state.
    fn transition_to_half_open(&self) {
        {
            let mut state = self.state.write();
            if *state != CircuitState::HalfOpen {
                *state = CircuitState::HalfOpen;
                *self.last_state_change.write() = Instant::now();
                self.half_open_successes.store(0, Ordering::Relaxed);
                warn!("Circuit breaker transitioned to half-open for recovery testing");
            }
        }
    }

    /// Transition the circuit breaker to the closed state.
    fn transition_to_closed(&self) {
        {
            let mut state = self.state.write();
            if *state != CircuitState::Closed {
                *state = CircuitState::Closed;
                *self.last_state_change.write() = Instant::now();
                self.metrics.circuit_closes.fetch_add(1, Ordering::Relaxed);

                // Clear failure history when closing
                self.recent_failures.write().clear();

                debug!("Circuit breaker closed after successful recovery");
            }
        }
    }

    /// Force the circuit breaker to a specific state (for testing).
    #[cfg(test)]
    pub fn force_state(&self, new_state: CircuitState) {
        *self.state.write() = new_state;
        *self.last_state_change.write() = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_circuit_breaker_success() {
        let circuit_breaker = CircuitBreaker::default();

        let result = circuit_breaker
            .call(async { Ok::<i32, AppError>(42) })
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(circuit_breaker.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_failure() {
        let circuit_breaker = CircuitBreaker::default();

        let result = circuit_breaker
            .call(async { Err::<i32, AppError>(AppError::Internal("test error".to_string())) })
            .await;

        assert!(result.is_err());
        assert_eq!(circuit_breaker.state(), CircuitState::Closed); // Still closed after one failure
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let circuit_breaker = CircuitBreaker::new(config);

        // First failure
        let _ = circuit_breaker
            .call(async { Err::<i32, AppError>(AppError::Internal("test error".to_string())) })
            .await;
        assert_eq!(circuit_breaker.state(), CircuitState::Closed);

        // Second failure should open the circuit
        let _ = circuit_breaker
            .call(async { Err::<i32, AppError>(AppError::Internal("test error".to_string())) })
            .await;
        assert_eq!(circuit_breaker.state(), CircuitState::Open);

        // Next request should be rejected
        let result = circuit_breaker
            .call(async { Ok::<i32, AppError>(42) })
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AppError::ServiceUnavailable(_)
        ));
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(10),
            success_threshold: 1,
            ..Default::default()
        };
        let circuit_breaker = CircuitBreaker::new(config);

        // Cause failure to open circuit
        let _ = circuit_breaker
            .call(async { Err::<i32, AppError>(AppError::Internal("test error".to_string())) })
            .await;
        assert_eq!(circuit_breaker.state(), CircuitState::Open);

        // Wait for recovery timeout
        sleep(Duration::from_millis(15)).await;

        // Next request should transition to half-open and succeed
        let result = circuit_breaker
            .call(async { Ok::<i32, AppError>(42) })
            .await;
        assert!(result.is_ok());
        assert_eq!(circuit_breaker.state(), CircuitState::Closed);
    }
}
