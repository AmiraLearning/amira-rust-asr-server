//! Custom error types for the amira-rust-asr-server.
//!
//! This module provides a centralized error handling system using the `thiserror` crate
//! to define structured, typed errors with clear messages and proper error conversion.

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

/// Primary error type for the application, covering all possible error cases.
#[derive(Debug, Error)]
pub enum AppError {
    /// Errors occurring during Triton Inference Server communication.
    #[error("Triton inference error: {0}")]
    TritonInference(#[from] TonicStatus),

    /// Errors occurring during model processing or prediction.
    #[error("Model error: {0}")]
    Model(String),

    /// Errors related to audio processing.
    #[error("Audio processing error: {0}")]
    Audio(String),

    /// Errors from invalid user input or requests.
    #[error("Invalid request: {0}")]
    Validation(String),

    /// Errors from invalid input data or parameters.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Errors from invalid configuration.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// CUDA-related errors.
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Errors from the underlying IO system.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Network-related errors.
    #[error("Network error: {0}")]
    Network(String),

    /// Service capacity limit reached.
    #[error("Service capacity exceeded: {0}")]
    CapacityExceeded(String),

    /// Timeouts in various operations.
    #[error("Operation timeout: {0}")]
    Timeout(String),

    /// Internal server errors.
    #[error("Internal server error: {0}")]
    Internal(String),

    /// Service is unavailable (e.g., circuit breaker is open).
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// JSON serialization/deserialization errors.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Implementation to convert AppError into an HTTP response for Axum.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            AppError::Validation(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            AppError::InvalidInput(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            AppError::ConfigError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            #[cfg(feature = "cuda")]
            AppError::CudaError(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            AppError::CapacityExceeded(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::Timeout(_) => (StatusCode::REQUEST_TIMEOUT, self.to_string()),
            AppError::Network(_) => (StatusCode::BAD_GATEWAY, self.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(json!({
            "error": error_message,
        }));

        (status, body).into_response()
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
        Err(_) => Err(AppError::Timeout(format!(
            "{}: operation timed out after {:?}",
            context, timeout_duration
        ))),
    }
}

/// Standardized async operation with timeout and custom error conversion.
///
/// This variant allows for custom error conversion from the operation's error type.
pub async fn with_timeout_and_convert<T, E, F, C>(
    operation: F,
    timeout_duration: Duration,
    context: &'static str,
    error_converter: C,
) -> Result<T>
where
    F: Future<Output = std::result::Result<T, E>>,
    C: FnOnce(E) -> AppError,
{
    match tokio::time::timeout(timeout_duration, operation).await {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(e)) => Err(error_converter(e)),
        Err(_) => Err(AppError::Timeout(format!(
            "{}: operation timed out after {:?}",
            context, timeout_duration
        ))),
    }
}
