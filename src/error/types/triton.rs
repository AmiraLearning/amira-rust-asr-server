//! Triton Inference Server error types.
//!
//! This module contains error types for Triton server operations,
//! including connection, inference, and gRPC errors.

use thiserror::Error;
use tonic::Status as TonicStatus;

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
