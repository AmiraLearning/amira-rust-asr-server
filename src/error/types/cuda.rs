//! CUDA-specific error types.
//!
//! This module contains error types for CUDA operations,
//! including initialization, memory allocation, and kernel execution.

use thiserror::Error;

/// CUDA-specific errors.
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
