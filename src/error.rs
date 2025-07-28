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
    InvalidInputShape { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Invalid output shape: expected {expected:?}, got {actual:?}")]
    InvalidOutputShape { expected: Vec<usize>, actual: Vec<usize> },
    
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
