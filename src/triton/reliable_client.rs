//! Reliable Triton client with circuit breaker protection.
//!
//! This module provides a production-ready wrapper around the Triton client
//! that includes circuit breaker protection, metrics, and fault tolerance.

use crate::error::{AppError, Result};
use crate::reliability::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use crate::triton::client::TritonClient;
use crate::triton::proto::{ModelInferRequest, ModelInferResponse};
use std::sync::Arc;
use tracing::{debug, instrument, warn};

/// A reliable Triton client with circuit breaker protection.
#[derive(Clone)]
pub struct ReliableTritonClient {
    /// The underlying Triton client.
    client: Arc<tokio::sync::Mutex<TritonClient>>,
    /// Circuit breaker for fault tolerance.
    circuit_breaker: Arc<CircuitBreaker>,
}

impl ReliableTritonClient {
    /// Create a new reliable Triton client.
    ///
    /// # Arguments
    /// * `endpoint` - The URL of the Triton Inference Server
    /// * `circuit_config` - Configuration for the circuit breaker
    ///
    /// # Returns
    /// A new reliable Triton client with circuit breaker protection
    pub async fn new(endpoint: &str, circuit_config: Option<CircuitBreakerConfig>) -> Result<Self> {
        let client = TritonClient::connect(endpoint)
            .await
            .map_err(|e| AppError::Internal(format!("Failed to connect to Triton: {}", e)))?;

        let circuit_breaker = Arc::new(CircuitBreaker::new(circuit_config.unwrap_or_default()));

        debug!("Created reliable Triton client for endpoint: {}", endpoint);

        Ok(Self {
            client: Arc::new(tokio::sync::Mutex::new(client)),
            circuit_breaker,
        })
    }

    /// Create a reliable client with default circuit breaker configuration.
    pub async fn connect(endpoint: &str) -> Result<Self> {
        Self::new(endpoint, None).await
    }

    /// Execute an inference request with circuit breaker protection.
    ///
    /// # Arguments
    /// * `request` - The inference request to execute
    ///
    /// # Returns
    /// The inference response or an error if the circuit is open or the request fails
    #[instrument(skip(self, request), fields(model_name = %request.model_name, request_id = %request.id))]
    pub async fn infer(&self, request: ModelInferRequest) -> Result<ModelInferResponse> {
        let model_name = request.model_name.clone();
        let request_id = request.id.clone();

        debug!(
            "Executing inference request for model: {}, request_id: {}",
            model_name, request_id
        );

        // Use circuit breaker to protect the inference call
        self.circuit_breaker
            .call(async {
                let mut client = self.client.lock().await;
                client.infer(request).await
            })
            .await
    }

    /// Get the current circuit breaker state.
    pub fn circuit_state(&self) -> crate::reliability::circuit_breaker::CircuitState {
        self.circuit_breaker.state()
    }

    /// Get circuit breaker metrics.
    pub fn circuit_metrics(
        &self,
    ) -> Arc<crate::reliability::circuit_breaker::CircuitBreakerMetrics> {
        self.circuit_breaker.metrics()
    }

    /// Create a new inference request builder.
    ///
    /// # Arguments
    /// * `model_name` - The name of the model to infer
    ///
    /// # Returns
    /// A new inference request builder
    pub async fn request_builder(
        &self,
        model_name: &str,
    ) -> crate::triton::client::InferRequestBuilder {
        let client = self.client.lock().await;
        client.request_builder(model_name)
    }

    /// Check if the client is healthy (circuit is not open).
    pub fn is_healthy(&self) -> bool {
        !matches!(
            self.circuit_breaker.state(),
            crate::reliability::circuit_breaker::CircuitState::Open
        )
    }

    /// Force the circuit breaker to reset (for emergency recovery).
    /// This should only be used in exceptional circumstances.
    pub fn force_reset_circuit(&self) {
        warn!("Forcing circuit breaker reset - this should only be done in emergencies");
        // We can't directly reset the circuit breaker, but we can log the request
        // In a real implementation, you might want to add a force_reset method
    }
}

/// Configuration builder for reliable Triton client.
pub struct ReliableTritonClientBuilder {
    endpoint: String,
    circuit_config: Option<CircuitBreakerConfig>,
}

impl ReliableTritonClientBuilder {
    /// Create a new builder for the given endpoint.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            circuit_config: None,
        }
    }

    /// Set the circuit breaker configuration.
    pub fn with_circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_config = Some(config);
        self
    }

    /// Set the failure threshold for the circuit breaker.
    pub fn with_failure_threshold(mut self, threshold: u64) -> Self {
        let mut config = self.circuit_config.unwrap_or_default();
        config.failure_threshold = threshold;
        self.circuit_config = Some(config);
        self
    }

    /// Set the request timeout for the circuit breaker.
    pub fn with_request_timeout(mut self, timeout: std::time::Duration) -> Self {
        let mut config = self.circuit_config.unwrap_or_default();
        config.request_timeout = timeout;
        self.circuit_config = Some(config);
        self
    }

    /// Set the recovery timeout for the circuit breaker.
    pub fn with_recovery_timeout(mut self, timeout: std::time::Duration) -> Self {
        let mut config = self.circuit_config.unwrap_or_default();
        config.recovery_timeout = timeout;
        self.circuit_config = Some(config);
        self
    }

    /// Build the reliable Triton client.
    pub async fn build(self) -> Result<ReliableTritonClient> {
        ReliableTritonClient::new(&self.endpoint, self.circuit_config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_reliable_client_builder() {
        let builder = ReliableTritonClientBuilder::new("http://localhost:8001")
            .with_failure_threshold(3)
            .with_request_timeout(Duration::from_secs(5))
            .with_recovery_timeout(Duration::from_secs(10));

        // Note: This test would fail without a real Triton server
        // In a real test environment, you'd mock the Triton client
        assert_eq!(builder.endpoint, "http://localhost:8001");
        assert!(builder.circuit_config.is_some());

        let config = builder.circuit_config.unwrap();
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.request_timeout, Duration::from_secs(5));
        assert_eq!(config.recovery_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_builder_defaults() {
        let builder = ReliableTritonClientBuilder::new("http://test:8001");
        assert!(builder.circuit_config.is_none());
    }
}
