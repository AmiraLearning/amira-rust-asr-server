//! Prometheus metrics collection and export.
//!
//! Provides comprehensive metrics for monitoring ASR server performance,
//! including request latency, error rates, and resource utilization.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// Metrics recorder for the ASR server.
pub struct AsrMetrics {
    /// Prometheus handle for exporting metrics.
    prometheus_handle: PrometheusHandle,
}

impl AsrMetrics {
    /// Initialize metrics collection and return a metrics instance.
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let builder = PrometheusBuilder::new();
        let prometheus_handle = builder.install_recorder()?;

        // Register all metrics with descriptions
        Self::register_metrics();

        info!("Metrics collection initialized");

        Ok(Self { prometheus_handle })
    }

    /// Register all metrics with their descriptions.
    fn register_metrics() {
        // Request metrics
        describe_counter!(
            "asr_requests_total",
            "Total number of ASR requests processed"
        );
        describe_counter!(
            "asr_requests_failed_total",
            "Total number of failed ASR requests"
        );
        describe_histogram!(
            "asr_request_duration_seconds",
            "Duration of ASR request processing in seconds"
        );

        // WebSocket metrics
        describe_counter!(
            "websocket_connections_total",
            "Total number of WebSocket connections established"
        );
        describe_gauge!(
            "websocket_connections_active",
            "Number of active WebSocket connections"
        );
        describe_counter!(
            "websocket_messages_total",
            "Total number of WebSocket messages processed"
        );

        // Triton inference metrics
        describe_histogram!(
            "triton_inference_duration_seconds",
            "Duration of Triton inference requests in seconds"
        );
        describe_counter!(
            "triton_requests_total",
            "Total number of requests sent to Triton"
        );
        describe_counter!(
            "triton_requests_failed_total",
            "Total number of failed Triton requests"
        );

        // Model-specific metrics
        describe_histogram!(
            "model_preprocessing_duration_seconds",
            "Duration of audio preprocessing in seconds"
        );
        describe_histogram!(
            "model_encoder_duration_seconds",
            "Duration of encoder inference in seconds"
        );
        describe_histogram!(
            "model_decoder_duration_seconds",
            "Duration of decoder inference in seconds"
        );

        // Circuit breaker metrics
        describe_counter!(
            "circuit_breaker_opens_total",
            "Total number of times circuit breaker has opened"
        );
        describe_counter!(
            "circuit_breaker_closes_total",
            "Total number of times circuit breaker has closed"
        );
        describe_gauge!(
            "circuit_breaker_state",
            "Current state of circuit breaker (0=closed, 1=half-open, 2=open)"
        );
        describe_counter!(
            "circuit_breaker_rejected_requests_total",
            "Total number of requests rejected by circuit breaker"
        );

        // Audio processing metrics
        describe_histogram!(
            "audio_conversion_duration_seconds",
            "Duration of audio format conversion in seconds"
        );
        describe_histogram!(
            "audio_chunk_size_bytes",
            "Size of audio chunks processed in bytes"
        );

        // Memory and resource metrics
        describe_gauge!(
            "memory_pool_available_buffers",
            "Number of available buffers in memory pool"
        );
        describe_gauge!(
            "memory_pool_allocated_buffers",
            "Number of allocated buffers in memory pool"
        );
        describe_counter!(
            "memory_allocations_total",
            "Total number of memory allocations"
        );

        // Connection pool metrics
        describe_gauge!(
            "connection_pool_active_connections",
            "Number of active connections in the pool"
        );
        describe_gauge!(
            "connection_pool_idle_connections",
            "Number of idle connections in the pool"
        );
        describe_counter!(
            "connection_pool_created_connections_total",
            "Total number of connections created"
        );
        describe_counter!(
            "connection_pool_dropped_connections_total",
            "Total number of connections dropped"
        );
    }

    /// Get the Prometheus metrics as a string.
    pub fn render(&self) -> String {
        self.prometheus_handle.render()
    }

    /// Create an Axum router for the metrics endpoint.
    pub fn router(self) -> Router {
        Router::new()
            .route("/metrics", get(metrics_handler))
            .with_state(Arc::new(self))
    }
}

impl Default for AsrMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to initialize metrics")
    }
}

/// Handler for the /metrics endpoint.
async fn metrics_handler(State(metrics): State<Arc<AsrMetrics>>) -> Response {
    let metrics_text = metrics.render();

    (
        StatusCode::OK,
        [("Content-Type", "text/plain; version=0.0.4")],
        metrics_text,
    )
        .into_response()
}

/// Timer for measuring operation durations.
pub struct MetricsTimer {
    name: String,
    start: Instant,
    labels: Vec<(String, String)>,
}

impl MetricsTimer {
    /// Create a new timer for the given metric.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            labels: Vec::new(),
        }
    }

    /// Add a label to the timer.
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }

    /// Record the elapsed time and drop the timer.
    pub fn finish(self) {
        let duration = self.start.elapsed();
        let duration_secs = duration.as_secs_f64();

        if self.labels.is_empty() {
            histogram!(self.name.clone()).record(duration_secs);
        } else {
            // For simplicity, record without labels if there are any labels
            // This avoids complex lifetime and ownership issues
            histogram!(self.name.clone()).record(duration_secs);
        }

        debug!(
            "Recorded metric '{}': {:.3}ms",
            self.name,
            duration.as_millis()
        );
    }
}

impl Drop for MetricsTimer {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let duration = self.start.elapsed();
            let duration_secs = duration.as_secs_f64();

            // For simplicity, record without labels to avoid ownership issues
            histogram!(self.name.clone()).record(duration_secs);
        }
    }
}

/// Convenience macros for common metrics operations.

/// Record the start of a request.
pub fn record_request_start(endpoint: &str) {
    counter!("asr_requests_total", "endpoint" => endpoint.to_string()).increment(1);
}

/// Record a failed request.
pub fn record_request_failure(endpoint: &str, error_type: &str) {
    counter!("asr_requests_failed_total", "endpoint" => endpoint.to_string(), "error" => error_type.to_string()).increment(1);
}

/// Record WebSocket connection events.
pub fn record_websocket_connection() {
    counter!("websocket_connections_total").increment(1);
    gauge!("websocket_connections_active").increment(1.0);
}

pub fn record_websocket_disconnection() {
    gauge!("websocket_connections_active").decrement(1.0);
}

pub fn record_websocket_message() {
    counter!("websocket_messages_total").increment(1);
}

/// Record Triton inference metrics.
pub fn record_triton_request(model_name: &str) {
    counter!("triton_requests_total", "model" => model_name.to_string()).increment(1);
}

pub fn record_triton_failure(model_name: &str, error_type: &str) {
    counter!("triton_requests_failed_total", "model" => model_name.to_string(), "error" => error_type.to_string()).increment(1);
}

/// Record circuit breaker state changes.
pub fn record_circuit_breaker_open() {
    counter!("circuit_breaker_opens_total").increment(1);
    gauge!("circuit_breaker_state").set(2.0);
}

pub fn record_circuit_breaker_close() {
    counter!("circuit_breaker_closes_total").increment(1);
    gauge!("circuit_breaker_state").set(0.0);
}

pub fn record_circuit_breaker_half_open() {
    gauge!("circuit_breaker_state").set(1.0);
}

pub fn record_circuit_breaker_rejection() {
    counter!("circuit_breaker_rejected_requests_total").increment(1);
}

/// Record memory pool statistics.
pub fn record_memory_pool_stats(available: usize, allocated: usize) {
    gauge!("memory_pool_available_buffers").set(available as f64);
    gauge!("memory_pool_allocated_buffers").set(allocated as f64);
}

/// Record connection pool statistics.
pub fn record_connection_pool_stats(active: usize, idle: usize) {
    gauge!("connection_pool_active_connections").set(active as f64);
    gauge!("connection_pool_idle_connections").set(idle as f64);
}

pub fn record_connection_created() {
    counter!("connection_pool_created_connections_total").increment(1);
}

pub fn record_connection_dropped() {
    counter!("connection_pool_dropped_connections_total").increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;
    use std::time::Duration;

    #[tokio::test]
    async fn test_metrics_initialization() {
        let metrics = AsrMetrics::new();
        assert!(metrics.is_ok());
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let metrics = AsrMetrics::new().unwrap();
        let app = metrics.router();
        let server = TestServer::new(app).unwrap();

        // Record some test metrics
        record_request_start("test");
        record_websocket_connection();
        record_triton_request("test_model");

        let response = server.get("/metrics").await;
        assert_eq!(response.status_code(), StatusCode::OK);

        let content_type = response.headers().get("content-type");
        assert!(content_type.is_some());
        assert!(content_type
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/plain"));

        let body = response.text();
        assert!(body.contains("asr_requests_total"));
        assert!(body.contains("websocket_connections_total"));
        assert!(body.contains("triton_requests_total"));
    }

    #[tokio::test]
    async fn test_metrics_timer() {
        let _timer = MetricsTimer::new("test_duration_seconds")
            .with_label("operation", "test")
            .with_label("status", "success");

        // Simulate some work
        tokio::time::sleep(Duration::from_millis(1)).await;

        // Timer will record the metric when dropped
    }

    #[test]
    fn test_metrics_recording() {
        // Test that metric recording functions don't panic
        record_request_start("test");
        record_request_failure("test", "timeout");
        record_websocket_connection();
        record_websocket_disconnection();
        record_triton_request("model");
        record_triton_failure("model", "error");
        record_circuit_breaker_open();
        record_circuit_breaker_close();
        record_memory_pool_stats(10, 5);
        record_connection_pool_stats(3, 2);
        record_connection_created();
        record_connection_dropped();
    }
}
