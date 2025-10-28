//! Configuration error types.
//!
//! This module contains error types for configuration validation
//! and loading errors.

use thiserror::Error;

use super::model::ModelError;

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
