//! Application-level error type and HTTP response mapping.
//!
//! This module defines the main `AppError` type that wraps all domain errors
//! and implements HTTP response conversion for Axum.

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

use super::types::*;

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
    pub fn error_type(&self) -> &'static str {
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
