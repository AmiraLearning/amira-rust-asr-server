//! Server error types.
//!
//! This module contains error types for HTTP server operations,
//! WebSocket handling, and request validation.

use thiserror::Error;

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
