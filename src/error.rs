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
use std::io;
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

    /// Errors from invalid configuration.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Errors from the underlying IO system.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

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
            AppError::CapacityExceeded(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            AppError::Timeout(_) => (StatusCode::REQUEST_TIMEOUT, self.to_string()),
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
