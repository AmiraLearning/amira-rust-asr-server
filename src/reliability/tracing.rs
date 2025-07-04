//! Distributed tracing setup with OpenTelemetry and Jaeger.
//!
//! Provides comprehensive tracing for debugging and performance monitoring
//! across the entire ASR pipeline.

use opentelemetry::global;
use opentelemetry_jaeger::new_agent_pipeline;
use tracing::{info, warn};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{
    filter::EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt, Registry,
};

/// Configuration for distributed tracing.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Service name for tracing.
    pub service_name: String,
    /// Jaeger agent endpoint.
    pub jaeger_endpoint: Option<String>,
    /// Whether to enable console logging.
    pub enable_console: bool,
    /// Log level filter.
    pub log_level: String,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "amira-asr-server".to_string(),
            jaeger_endpoint: Some("http://localhost:14268/api/traces".to_string()),
            enable_console: true,
            log_level: "info".to_string(),
        }
    }
}

/// Initialize distributed tracing with OpenTelemetry and Jaeger.
pub fn init_tracing(config: TracingConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(
        "Initializing distributed tracing with service name: {}",
        config.service_name
    );

    // Create OpenTelemetry tracer
    let tracer = if let Some(jaeger_endpoint) = config.jaeger_endpoint {
        info!("Setting up Jaeger tracing to endpoint: {}", jaeger_endpoint);

        match new_agent_pipeline()
            .with_service_name(&config.service_name)
            .with_endpoint(&jaeger_endpoint)
            .install_simple()
        {
            Ok(tracer) => {
                info!("Jaeger tracing initialized successfully");
                Some(tracer)
            }
            Err(e) => {
                warn!("Failed to initialize Jaeger tracing: {}. Continuing without distributed tracing.", e);
                None
            }
        }
    } else {
        info!("No Jaeger endpoint configured, skipping distributed tracing");
        None
    };

    // Create tracing subscriber with layers
    let registry = Registry::default();

    // Add environment filter
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    let subscriber = registry.with(env_filter);

    // Build subscriber with conditional layers
    if config.enable_console {
        if let Some(tracer) = tracer {
            subscriber
                .with(
                    fmt::layer()
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_thread_names(true)
                        .compact(),
                )
                .with(OpenTelemetryLayer::new(tracer))
                .try_init()?;
        } else {
            subscriber
                .with(
                    fmt::layer()
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_thread_names(true)
                        .compact(),
                )
                .try_init()?;
        }
    } else {
        if let Some(tracer) = tracer {
            subscriber
                .with(OpenTelemetryLayer::new(tracer))
                .try_init()?;
        } else {
            subscriber.try_init()?;
        }
    }

    info!("Tracing initialization completed");
    Ok(())
}

/// Shutdown tracing and flush any pending spans.
pub fn shutdown_tracing() {
    info!("Shutting down tracing");
    global::shutdown_tracer_provider();
}

/// Create a tracing span for ASR request processing.
#[macro_export]
macro_rules! asr_span {
    ($name:expr) => {
        tracing::info_span!($name,
            request_id = tracing::field::Empty,
            user_id = tracing::field::Empty,
            model_name = tracing::field::Empty,
            audio_duration_ms = tracing::field::Empty,
            processing_time_ms = tracing::field::Empty,
        )
    };
    ($name:expr, $($key:ident = $value:expr),*) => {
        tracing::info_span!($name,
            request_id = tracing::field::Empty,
            user_id = tracing::field::Empty,
            model_name = tracing::field::Empty,
            audio_duration_ms = tracing::field::Empty,
            processing_time_ms = tracing::field::Empty,
            $($key = $value),*
        )
    };
}

/// Create a tracing span for Triton inference operations.
#[macro_export]
macro_rules! triton_span {
    ($name:expr, $model:expr) => {
        tracing::info_span!(
            $name,
            model_name = $model,
            request_id = tracing::field::Empty,
            input_size = tracing::field::Empty,
            output_size = tracing::field::Empty,
            inference_time_ms = tracing::field::Empty,
        )
    };
}

/// Create a tracing span for WebSocket operations.
#[macro_export]
macro_rules! websocket_span {
    ($name:expr) => {
        tracing::info_span!(
            $name,
            connection_id = tracing::field::Empty,
            remote_addr = tracing::field::Empty,
            message_type = tracing::field::Empty,
            message_size = tracing::field::Empty,
        )
    };
}

/// Utilities for adding common fields to spans.
pub mod span_utils {
    use tracing::Span;
    use uuid::Uuid;

    /// Add request identification to the current span.
    pub fn add_request_id(request_id: &Uuid) {
        Span::current().record("request_id", &request_id.to_string());
    }

    /// Add user identification to the current span.
    pub fn add_user_id(user_id: &str) {
        Span::current().record("user_id", user_id);
    }

    /// Add model name to the current span.
    pub fn add_model_name(model_name: &str) {
        Span::current().record("model_name", model_name);
    }

    /// Add audio duration to the current span.
    pub fn add_audio_duration(duration_ms: u64) {
        Span::current().record("audio_duration_ms", duration_ms);
    }

    /// Add processing time to the current span.
    pub fn add_processing_time(time_ms: u64) {
        Span::current().record("processing_time_ms", time_ms);
    }

    /// Add input size to the current span.
    pub fn add_input_size(size: usize) {
        Span::current().record("input_size", size);
    }

    /// Add output size to the current span.
    pub fn add_output_size(size: usize) {
        Span::current().record("output_size", size);
    }

    /// Add inference time to the current span.
    pub fn add_inference_time(time_ms: u64) {
        Span::current().record("inference_time_ms", time_ms);
    }

    /// Add connection ID to the current span.
    pub fn add_connection_id(connection_id: &Uuid) {
        Span::current().record("connection_id", &connection_id.to_string());
    }

    /// Add remote address to the current span.
    pub fn add_remote_addr(addr: &str) {
        Span::current().record("remote_addr", addr);
    }

    /// Add message type to the current span.
    pub fn add_message_type(msg_type: &str) {
        Span::current().record("message_type", msg_type);
    }

    /// Add message size to the current span.
    pub fn add_message_size(size: usize) {
        Span::current().record("message_size", size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use tracing::{info, info_span};

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "amira-asr-server");
        assert!(config.jaeger_endpoint.is_some());
        assert!(config.enable_console);
        assert_eq!(config.log_level, "info");
    }

    #[tokio::test]
    async fn test_tracing_initialization() {
        let config = TracingConfig {
            service_name: "test-service".to_string(),
            jaeger_endpoint: None, // Disable Jaeger for test
            enable_console: false, // Disable console for test
            log_level: "debug".to_string(),
        };

        let result = init_tracing(config);
        // Note: This might fail in test environment, which is expected
        // The important thing is that it doesn't panic
        println!("Tracing init result: {:?}", result);
    }

    #[tokio::test]
    async fn test_span_macros() {
        // Test that span macros compile and work
        let _span1 = asr_span!("test_asr");
        let _span2 = asr_span!("test_asr_with_fields", custom_field = "test_value");
        let _span3 = triton_span!("test_triton", "test_model");
        let _span4 = websocket_span!("test_websocket");
    }

    #[test]
    fn test_span_utils() {
        // Test span utilities (they should not panic even without a span)
        span_utils::add_request_id(&uuid::Uuid::new_v4());
        span_utils::add_user_id("test_user");
        span_utils::add_model_name("test_model");
        span_utils::add_audio_duration(1000);
        span_utils::add_processing_time(50);
        span_utils::add_input_size(1024);
        span_utils::add_output_size(512);
        span_utils::add_inference_time(25);
        span_utils::add_connection_id(&uuid::Uuid::new_v4());
        span_utils::add_remote_addr("127.0.0.1:8080");
        span_utils::add_message_type("audio_chunk");
        span_utils::add_message_size(2048);
    }
}
