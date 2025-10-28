//! CUDA error types and conversions
//!
//! This module provides error handling types for CUDA operations,
//! including FFI error codes and Rust-friendly error types.

use std::fmt;

/// Error codes that match the C implementation
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub enum CudaError {
    CudaSuccess = 0,
    CudaErrorInvalidValue = 1,
    CudaErrorOutOfMemory = 2,
    CudaErrorUnknown = 3,
    CudaErrorNotReady = 4,
}

/// Rust error type for CUDA operations
#[derive(Debug)]
pub enum CudaSharedMemoryError {
    InvalidValue,
    OutOfMemory,
    Unknown,
    NullPointer,
}

impl fmt::Display for CudaSharedMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaSharedMemoryError::InvalidValue => write!(f, "Invalid value"),
            CudaSharedMemoryError::OutOfMemory => write!(f, "Out of memory"),
            CudaSharedMemoryError::Unknown => write!(f, "Unknown error"),
            CudaSharedMemoryError::NullPointer => write!(f, "Null pointer"),
        }
    }
}

impl std::error::Error for CudaSharedMemoryError {}

impl CudaSharedMemoryError {
    /// Validate a device ID is within valid range
    ///
    /// Note: This requires CUDA to be available. Returns Ok if device count
    /// cannot be determined (to avoid breaking existing code).
    pub fn validate_device_id(device_id: i32) -> Result<(), Self> {
        if device_id < 0 {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        // Try to get device count - if this fails, we can't validate
        // but we shouldn't break existing code, so return Ok
        let count = crate::cuda::device_buffer::utils::device_count();
        if count > 0 && device_id >= count {
            Err(CudaSharedMemoryError::InvalidValue)
        } else {
            Ok(())
        }
    }
}

impl From<CudaError> for CudaSharedMemoryError {
    fn from(error: CudaError) -> Self {
        match error {
            // CudaSuccess should never be converted to an error, but handle it gracefully
            // rather than panicking to avoid potential DoS vectors
            CudaError::CudaSuccess => CudaSharedMemoryError::Unknown,
            CudaError::CudaErrorInvalidValue => CudaSharedMemoryError::InvalidValue,
            CudaError::CudaErrorOutOfMemory => CudaSharedMemoryError::OutOfMemory,
            CudaError::CudaErrorUnknown => CudaSharedMemoryError::Unknown,
            CudaError::CudaErrorNotReady => CudaSharedMemoryError::Unknown,
        }
    }
}
