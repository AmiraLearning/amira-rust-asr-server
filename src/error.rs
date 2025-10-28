//! Domain-specific error types for the amira-rust-asr-server.
//!
//! This module provides a hierarchical error handling system using the `thiserror` crate
//! to define structured, typed errors with clear messages and proper error conversion.
//! Each domain has its own error type for better error handling and debugging.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::future::Future;
use std::io;
use std::time::Duration;
use thiserror::Error;
use tonic::Status as TonicStatus;

/// ASR-specific errors for audio processing and speech recognition.
#[derive(Debug, Error)]
pub enum AsrError {
    #[error("Audio processing failed: {0}")]
    AudioProcessing(#[from] AudioError),

    #[error("Model inference failed: {0}")]
    ModelInference(#[from] ModelError),

    #[error("Decoder state invalid: {0}")]
    DecoderState(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Vocabulary error: {0}")]
    Vocabulary(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

/// Audio processing errors.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Invalid sample rate: expected {expected}, got {actual}")]
    InvalidSampleRate { expected: u32, actual: u32 },

    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Buffer underrun: insufficient audio data")]
    BufferUnderrun,

    #[error("Buffer overflow: audio data too large")]
    BufferOverflow,

    #[error("SIMD processing error: {0}")]
    SimdProcessing(String),

    #[error("Windowing error: {0}")]
    Windowing(String),
}

/// Model inference errors.
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model not found: {model_name}")]
    NotFound { model_name: String },

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid output shape: expected {expected:?}, got {actual:?}")]
    InvalidOutputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Tensor conversion error: {0}")]
    TensorConversion(String),

    #[error("Preprocessing error: {0}")]
    Preprocessing(String),

    #[error("Postprocessing error: {0}")]
    Postprocessing(String),

    #[error("Model inference error: {0}")]
    Inference(String),
}

/// Triton Inference Server errors.
#[derive(Debug, Error)]
pub enum TritonError {
    #[error("Connection failed: {0}")]
    Connection(#[from] tonic::transport::Error),

    #[error("Inference timeout: {0}")]
    Timeout(#[from] tokio::time::error::Elapsed),

    #[error("Pool exhausted: {0}")]
    PoolExhausted(String),

    #[error("gRPC error: {0}")]
    Grpc(#[from] TonicStatus),

    #[error("Model server error: {0}")]
    ModelServer(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// Configuration errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },

    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Model configuration error: {0}")]
    ModelConfig(#[from] ModelError),
}

/// Server errors.
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Bind error: {0}")]
    Bind(#[from] std::io::Error),

    #[error("Request validation error: {0}")]
    RequestValidation(String),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("JSON serialization error: {0}")]
    JsonSerialization(#[from] serde_json::Error),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
}

/// Performance and reliability errors.
#[derive(Debug, Error)]
pub enum PerformanceError {
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("CPU affinity error: {0}")]
    CpuAffinity(String),

    #[error("NUMA error: {0}")]
    Numa(String),

    #[error("Circuit breaker open: {0}")]
    CircuitBreakerOpen(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}

/// CUDA-specific errors.
#[cfg(feature = "cuda")]
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA initialization failed: {0}")]
    Initialization(String),

    #[error("CUDA memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("CUDA kernel execution failed: {0}")]
    KernelExecution(String),

    #[error("CUDA device error: {0}")]
    Device(String),
}

/// Primary error type for the application, covering all possible error cases.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("ASR error: {0}")]
    Asr(#[from] AsrError),

    #[error("Triton error: {0}")]
    Triton(#[from] TritonError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Server error: {0}")]
    Server(#[from] ServerError),

    #[error("Performance error: {0}")]
    Performance(#[from] PerformanceError),

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),

    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("Triton inference error: {0}")]
    TritonInference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Implementation to convert AppError into an HTTP response for Axum.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            AppError::Asr(AsrError::AudioProcessing(AudioError::InvalidSampleRate { .. })) => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            AppError::Asr(AsrError::AudioProcessing(AudioError::InvalidFormat(_))) => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            AppError::Asr(AsrError::Configuration(_)) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            AppError::Triton(TritonError::Connection(_)) => {
                (StatusCode::BAD_GATEWAY, self.to_string())
            }
            AppError::Triton(TritonError::Timeout(_)) => {
                (StatusCode::REQUEST_TIMEOUT, self.to_string())
            }
            AppError::Triton(TritonError::PoolExhausted(_)) => {
                (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
            }
            AppError::Config(ConfigError::MissingField { .. }) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            AppError::Config(ConfigError::InvalidValue { .. }) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
            AppError::Server(ServerError::RequestValidation(_)) => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            AppError::Server(ServerError::ServiceUnavailable(_)) => {
                (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
            }
            AppError::Server(ServerError::RateLimitExceeded(_)) => {
                (StatusCode::TOO_MANY_REQUESTS, self.to_string())
            }
            AppError::Performance(PerformanceError::CircuitBreakerOpen(_)) => {
                (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
            }
            AppError::Performance(PerformanceError::ResourceExhausted(_)) => {
                (StatusCode::SERVICE_UNAVAILABLE, self.to_string())
            }
            #[cfg(feature = "cuda")]
            AppError::Cuda(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            AppError::Network(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            AppError::ConfigError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            AppError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::Timeout(_) => (StatusCode::REQUEST_TIMEOUT, self.to_string()),
            AppError::CapacityExceeded(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::TritonInference(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({
            "error": error_message,
            "error_type": self.error_type(),
        }));

        (status, body).into_response()
    }
}

impl AppError {
    /// Returns the error type as a string for client consumption.
    fn error_type(&self) -> &'static str {
        match self {
            AppError::Asr(AsrError::AudioProcessing(_)) => "audio_processing",
            AppError::Asr(AsrError::ModelInference(_)) => "model_inference",
            AppError::Asr(AsrError::DecoderState(_)) => "decoder_state",
            AppError::Asr(AsrError::Configuration(_)) => "asr_configuration",
            AppError::Asr(AsrError::Vocabulary(_)) => "vocabulary",
            AppError::Asr(AsrError::Pipeline(_)) => "pipeline",
            AppError::Triton(_) => "triton",
            AppError::Config(_) => "configuration",
            AppError::Server(_) => "server",
            AppError::Performance(_) => "performance",
            #[cfg(feature = "cuda")]
            AppError::Cuda(_) => "cuda",
            AppError::Io(_) => "io",
            AppError::Network(_) => "network",
            AppError::ConfigError(_) => "config_error",
            AppError::ServiceUnavailable(_) => "service_unavailable",
            AppError::Timeout(_) => "timeout",
            AppError::CapacityExceeded(_) => "capacity_exceeded",
            AppError::TritonInference(_) => "triton_inference",
            AppError::Internal(_) => "internal",
            AppError::InvalidInput(_) => "invalid_input",
        }
    }
}

/// Convenience type alias for Results with AppError.
pub type Result<T> = std::result::Result<T, AppError>;

/// Extension trait for adding context to errors.
pub trait ErrorContext<T> {
    /// Add context to the error.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    /// Add static context to the error.
    fn with_static_context(self, context: &'static str) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| AppError::Internal(format!("{}: {}", f(), e)))
    }

    fn with_static_context(self, context: &'static str) -> Result<T> {
        self.map_err(|e| AppError::Internal(format!("{}: {}", context, e)))
    }
}

/// Extension trait for converting domain errors to AppError.
pub trait IntoAppError<T> {
    /// Convert a domain error into an AppError.
    fn into_app_error(self) -> Result<T>;
}

impl<T> IntoAppError<T> for std::result::Result<T, AsrError> {
    fn into_app_error(self) -> Result<T> {
        self.map_err(AppError::Asr)
    }
}

impl<T> IntoAppError<T> for std::result::Result<T, TritonError> {
    fn into_app_error(self) -> Result<T> {
        self.map_err(AppError::Triton)
    }
}

impl<T> IntoAppError<T> for std::result::Result<T, ConfigError> {
    fn into_app_error(self) -> Result<T> {
        self.map_err(AppError::Config)
    }
}

impl<T> IntoAppError<T> for std::result::Result<T, ServerError> {
    fn into_app_error(self) -> Result<T> {
        self.map_err(AppError::Server)
    }
}

impl<T> IntoAppError<T> for std::result::Result<T, PerformanceError> {
    fn into_app_error(self) -> Result<T> {
        self.map_err(AppError::Performance)
    }
}

#[cfg(feature = "cuda")]
impl<T> IntoAppError<T> for std::result::Result<T, CudaError> {
    fn into_app_error(self) -> Result<T> {
        self.map_err(AppError::Cuda)
    }
}

/// Standardized async operation with timeout handling.
///
/// This function provides a consistent pattern for async operations that need
/// timeout handling, proper error conversion, and context information.
pub async fn with_timeout<T, E, F>(
    operation: F,
    timeout_duration: Duration,
    context: &'static str,
) -> Result<T>
where
    F: Future<Output = std::result::Result<T, E>>,
    E: std::error::Error + Send + Sync + 'static,
{
    match tokio::time::timeout(timeout_duration, operation).await {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => Err(AppError::Internal(format!("{}: {}", context, e))),
        Err(_) => Err(AppError::Timeout("Operation timed out".to_string())),
    }
}

/// Standardized async operation with timeout and custom error conversion.
///
/// This variant allows for custom error conversion from the operation's error type.
pub async fn with_timeout_and_convert<T, E, F, C>(
    operation: F,
    timeout_duration: Duration,
    _context: &'static str,
    error_converter: C,
) -> Result<T>
where
    F: Future<Output = std::result::Result<T, E>>,
    C: FnOnce(E) -> AppError,
{
    match tokio::time::timeout(timeout_duration, operation).await {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => Err(error_converter(e)),
        Err(_) => Err(AppError::Timeout("Operation timed out".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // Test HTTP status code mapping for various error types
    #[test]
    fn test_http_status_invalid_sample_rate() {
        let error = AppError::Asr(AsrError::AudioProcessing(AudioError::InvalidSampleRate {
            expected: 16000,
            actual: 8000,
        }));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_http_status_invalid_format() {
        let error = AppError::Asr(AsrError::AudioProcessing(AudioError::InvalidFormat(
            "PCM required".to_string(),
        )));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_http_status_asr_configuration() {
        let error = AppError::Asr(AsrError::Configuration("Invalid config".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_status_triton_connection() {
        let grpc_error = TonicStatus::unavailable("Connection refused");
        let error = AppError::Triton(TritonError::Grpc(grpc_error));

        let response = error.into_response();
        // Grpc errors fall through to default INTERNAL_SERVER_ERROR
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_http_status_triton_timeout() {
        // Create a timeout error by actually timing out an operation
        let operation = tokio::time::timeout(
            Duration::from_millis(1),
            tokio::time::sleep(Duration::from_millis(100)),
        )
        .await;

        let elapsed_error = operation.unwrap_err();
        let error = AppError::Triton(TritonError::Timeout(elapsed_error));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn test_http_status_pool_exhausted() {
        let error = AppError::Triton(TritonError::PoolExhausted("No connections".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_config_missing_field() {
        let error = AppError::Config(ConfigError::MissingField {
            field: "triton_endpoint".to_string(),
        });

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_status_config_invalid_value() {
        let error = AppError::Config(ConfigError::InvalidValue {
            field: "port".to_string(),
            value: "999999".to_string(),
        });

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_status_request_validation() {
        let error = AppError::Server(ServerError::RequestValidation(
            "Invalid audio buffer".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_http_status_service_unavailable() {
        let error = AppError::Server(ServerError::ServiceUnavailable("Overloaded".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_rate_limit_exceeded() {
        let error = AppError::Server(ServerError::RateLimitExceeded(
            "Too many requests".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn test_http_status_circuit_breaker() {
        let error = AppError::Performance(PerformanceError::CircuitBreakerOpen(
            "Circuit open".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_resource_exhausted() {
        let error = AppError::Performance(PerformanceError::ResourceExhausted(
            "Out of memory".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_network_error() {
        let error = AppError::Network("Connection failed".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn test_http_status_timeout() {
        let error = AppError::Timeout("Request timeout".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn test_http_status_capacity_exceeded() {
        let error = AppError::CapacityExceeded("Queue full".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_triton_inference() {
        let error = AppError::TritonInference("Model error".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn test_http_status_default_internal_server_error() {
        let error = AppError::Internal("Something went wrong".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Test error_type() method
    #[test]
    fn test_error_type_audio_processing() {
        let error = AppError::Asr(AsrError::AudioProcessing(AudioError::BufferUnderrun));
        assert_eq!(error.error_type(), "audio_processing");
    }

    #[test]
    fn test_error_type_model_inference() {
        let error = AppError::Asr(AsrError::ModelInference(ModelError::Inference(
            "Test".to_string(),
        )));
        assert_eq!(error.error_type(), "model_inference");
    }

    #[test]
    fn test_error_type_decoder_state() {
        let error = AppError::Asr(AsrError::DecoderState("Invalid state".to_string()));
        assert_eq!(error.error_type(), "decoder_state");
    }

    #[test]
    fn test_error_type_asr_configuration() {
        let error = AppError::Asr(AsrError::Configuration("Config error".to_string()));
        assert_eq!(error.error_type(), "asr_configuration");
    }

    #[test]
    fn test_error_type_vocabulary() {
        let error = AppError::Asr(AsrError::Vocabulary("Vocab error".to_string()));
        assert_eq!(error.error_type(), "vocabulary");
    }

    #[test]
    fn test_error_type_pipeline() {
        let error = AppError::Asr(AsrError::Pipeline("Pipeline error".to_string()));
        assert_eq!(error.error_type(), "pipeline");
    }

    #[test]
    fn test_error_type_triton() {
        let error = AppError::Triton(TritonError::ModelServer("Server error".to_string()));
        assert_eq!(error.error_type(), "triton");
    }

    #[test]
    fn test_error_type_configuration() {
        let error = AppError::Config(ConfigError::Validation("Validation error".to_string()));
        assert_eq!(error.error_type(), "configuration");
    }

    #[test]
    fn test_error_type_server() {
        let error = AppError::Server(ServerError::WebSocket("WS error".to_string()));
        assert_eq!(error.error_type(), "server");
    }

    #[test]
    fn test_error_type_performance() {
        let error = AppError::Performance(PerformanceError::Numa("NUMA error".to_string()));
        assert_eq!(error.error_type(), "performance");
    }

    #[test]
    fn test_error_type_io() {
        let error = AppError::Io(io::Error::new(io::ErrorKind::NotFound, "File not found"));
        assert_eq!(error.error_type(), "io");
    }

    #[test]
    fn test_error_type_network() {
        let error = AppError::Network("Network error".to_string());
        assert_eq!(error.error_type(), "network");
    }

    #[test]
    fn test_error_type_service_unavailable() {
        let error = AppError::ServiceUnavailable("Service down".to_string());
        assert_eq!(error.error_type(), "service_unavailable");
    }

    #[test]
    fn test_error_type_timeout() {
        let error = AppError::Timeout("Timeout".to_string());
        assert_eq!(error.error_type(), "timeout");
    }

    #[test]
    fn test_error_type_capacity_exceeded() {
        let error = AppError::CapacityExceeded("Queue full".to_string());
        assert_eq!(error.error_type(), "capacity_exceeded");
    }

    #[test]
    fn test_error_type_internal() {
        let error = AppError::Internal("Internal error".to_string());
        assert_eq!(error.error_type(), "internal");
    }

    #[test]
    fn test_error_type_invalid_input() {
        let error = AppError::InvalidInput("Bad input".to_string());
        assert_eq!(error.error_type(), "invalid_input");
    }

    // Test ErrorContext trait
    #[test]
    fn test_error_context_with_context() {
        let result: std::result::Result<(), io::Error> =
            Err(io::Error::new(io::ErrorKind::NotFound, "file.txt"));

        let app_result = result.with_context(|| "Failed to read file".to_string());

        assert!(app_result.is_err());
        let err_msg = format!("{:?}", app_result.unwrap_err());
        assert!(err_msg.contains("Failed to read file"));
        assert!(err_msg.contains("file.txt"));
    }

    #[test]
    fn test_error_context_with_static_context() {
        let result: std::result::Result<(), io::Error> = Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "access denied",
        ));

        let app_result = result.with_static_context("Unable to access resource");

        assert!(app_result.is_err());
        let err_msg = format!("{:?}", app_result.unwrap_err());
        assert!(err_msg.contains("Unable to access resource"));
        assert!(err_msg.contains("access denied"));
    }

    #[test]
    fn test_error_context_success() {
        let result: std::result::Result<i32, io::Error> = Ok(42);

        let app_result = result.with_context(|| "This should not be called".to_string());

        assert!(app_result.is_ok());
        assert_eq!(app_result.unwrap(), 42);
    }

    // Test IntoAppError trait
    #[test]
    fn test_into_app_error_asr() {
        let result: std::result::Result<(), AsrError> =
            Err(AsrError::Vocabulary("Missing token".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Asr(_)));
    }

    #[test]
    fn test_into_app_error_triton() {
        let result: std::result::Result<(), TritonError> =
            Err(TritonError::InvalidResponse("Bad response".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Triton(_)));
    }

    #[test]
    fn test_into_app_error_config() {
        let result: std::result::Result<(), ConfigError> =
            Err(ConfigError::Parse("Invalid TOML".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Config(_)));
    }

    #[test]
    fn test_into_app_error_server() {
        let result: std::result::Result<(), ServerError> =
            Err(ServerError::WebSocket("Connection closed".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Server(_)));
    }

    #[test]
    fn test_into_app_error_performance() {
        let result: std::result::Result<(), PerformanceError> = Err(PerformanceError::CpuAffinity(
            "Failed to set affinity".to_string(),
        ));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Performance(_)));
    }

    #[test]
    fn test_into_app_error_success() {
        let result: std::result::Result<String, AsrError> = Ok("success".to_string());

        let app_result = result.into_app_error();

        assert!(app_result.is_ok());
        assert_eq!(app_result.unwrap(), "success");
    }

    // Test timeout helpers
    #[tokio::test]
    async fn test_with_timeout_success() {
        let operation = async { Ok::<i32, io::Error>(42) };

        let result = with_timeout(operation, Duration::from_secs(1), "Test operation").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_timeout_error() {
        let operation =
            async { Err::<i32, io::Error>(io::Error::new(io::ErrorKind::NotFound, "Not found")) };

        let result = with_timeout(operation, Duration::from_secs(1), "Failed operation").await;

        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Failed operation"));
        assert!(err_msg.contains("Not found"));
    }

    #[tokio::test]
    async fn test_with_timeout_timeout() {
        let operation = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<i32, io::Error>(42)
        };

        let result = with_timeout(operation, Duration::from_millis(50), "Slow operation").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AppError::Timeout(_)));
        assert_eq!(format!("{}", err), "Timeout: Operation timed out");
    }

    #[tokio::test]
    async fn test_with_timeout_and_convert_success() {
        let operation = async { Ok::<String, io::Error>("hello".to_string()) };

        let result =
            with_timeout_and_convert(operation, Duration::from_secs(1), "Test conversion", |e| {
                AppError::Network(e.to_string())
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello");
    }

    #[tokio::test]
    async fn test_with_timeout_and_convert_error() {
        let operation = async {
            Err::<String, io::Error>(io::Error::new(io::ErrorKind::ConnectionRefused, "Refused"))
        };

        let result = with_timeout_and_convert(
            operation,
            Duration::from_secs(1),
            "Failed conversion",
            |e| AppError::Network(e.to_string()),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AppError::Network(_)));
    }

    #[tokio::test]
    async fn test_with_timeout_and_convert_timeout() {
        let operation = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<String, io::Error>("delayed".to_string())
        };

        let result = with_timeout_and_convert(
            operation,
            Duration::from_millis(50),
            "Timeout conversion",
            |e| AppError::Network(e.to_string()),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AppError::Timeout(_)));
    }

    // Test error message formatting
    #[test]
    fn test_audio_error_invalid_sample_rate_message() {
        let error = AudioError::InvalidSampleRate {
            expected: 16000,
            actual: 8000,
        };
        assert_eq!(
            format!("{}", error),
            "Invalid sample rate: expected 16000, got 8000"
        );
    }

    #[test]
    fn test_model_error_not_found_message() {
        let error = ModelError::NotFound {
            model_name: "encoder".to_string(),
        };
        assert_eq!(format!("{}", error), "Model not found: encoder");
    }

    #[test]
    fn test_config_error_missing_field_message() {
        let error = ConfigError::MissingField {
            field: "vocabulary_path".to_string(),
        };
        assert_eq!(
            format!("{}", error),
            "Missing required field: vocabulary_path"
        );
    }

    #[test]
    fn test_config_error_invalid_value_message() {
        let error = ConfigError::InvalidValue {
            field: "port".to_string(),
            value: "999999".to_string(),
        };
        assert_eq!(format!("{}", error), "Invalid value for port: 999999");
    }

    // Test CUDA errors when feature is enabled
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_error_http_status() {
        let error = AppError::Cuda(CudaError::Initialization("GPU not found".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_error_type() {
        let error = AppError::Cuda(CudaError::MemoryAllocation("Out of VRAM".to_string()));
        assert_eq!(error.error_type(), "cuda");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_into_app_error_cuda() {
        let result: std::result::Result<(), CudaError> =
            Err(CudaError::KernelExecution("Kernel failed".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Cuda(_)));
    }
}
