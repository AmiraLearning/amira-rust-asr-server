//! Utility functions for CUDA operations
//!
//! This module provides helper functions for querying CUDA capabilities
//! and device information.

use crate::cuda::error::{CudaError, CudaSharedMemoryError};
use crate::cuda::ffi::get_cuda_device_count_ffi;
use std::os::raw::c_int;

/// Get the number of available CUDA devices
pub fn get_cuda_device_count() -> Result<i32, CudaSharedMemoryError> {
    let mut count: c_int = 0;
    let result = unsafe { get_cuda_device_count_ffi(&mut count) };

    if result != CudaError::CudaSuccess {
        return Err(result.into());
    }

    Ok(count)
}

/// Check if CUDA is available
pub fn is_cuda_available() -> bool {
    get_cuda_device_count().is_ok()
}
