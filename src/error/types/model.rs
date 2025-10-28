//! Model inference error types.
//!
//! This module contains error types for model operations,
//! including input/output validation and inference errors.

use thiserror::Error;

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
