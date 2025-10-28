//! FFI declarations for CUDA helper functions
//!
//! This module contains unsafe foreign function declarations for
//! interfacing with the CUDA C API.
//!
//! ## Safety
//!
//! All FFI functions must be called with valid pointers. The caller is responsible
//! for ensuring:
//! - Pointers are non-null when required
//! - Buffers have sufficient capacity for the requested operations
//! - Memory regions are properly aligned
//! - Concurrent access is properly synchronized

use std::os::raw::{c_char, c_int, c_void};

use crate::cuda::error::CudaError;

// FFI declarations for CUDA helper functions
#[allow(improper_ctypes)]  // Allow for compatibility with C API
extern "C" {
    pub(crate) fn get_cuda_device_count_ffi(count: *mut c_int) -> CudaError;
    pub(crate) fn CudaSharedMemoryRegionCreate(
        name: *const c_char,
        byte_size: usize,
        device_id: c_int,
        handle: *mut *mut c_void,
    ) -> CudaError;
    pub(crate) fn CudaSharedMemoryRegionDestroy(handle: *mut c_void) -> CudaError;
    pub(crate) fn GetRawHandle(handle: *mut c_void, raw_handle: *mut *mut c_char) -> CudaError;
    pub(crate) fn FreeRawHandle(raw_handle: *mut c_char) -> CudaError;
    pub(crate) fn WriteTestData(handle: *mut c_void, data: *const f32, element_count: usize) -> CudaError;
    pub(crate) fn ReadTestData(handle: *mut c_void, data: *mut f32, element_count: usize) -> CudaError;
    pub(crate) fn RegisterWithTritonServer(handle: *mut c_void) -> CudaError;
    #[allow(dead_code)]
    pub(crate) fn RunTritonInference(handle: *mut c_void) -> CudaError;
    pub(crate) fn RunTritonInferenceWithConfig(
        handle: *mut c_void,
        model_name: *const c_char,
        input_name: *const c_char,
        input_data_type: c_int,
        input_shape: *const i64,
        input_dims: usize,
        output_name: *const c_char,
        buffer_size: usize,
    ) -> CudaError;
    pub(crate) fn RunTritonInferenceWithOutputRegions(
        input_handle: *mut c_void,
        output_handle: *mut c_void,
        model_name: *const c_char,
        input_name: *const c_char,
        input_data_type: c_int,
        input_shape: *const i64,
        input_dims: usize,
        output_name: *const c_char,
        input_buffer_size: usize,
        output_buffer_size: usize,
    ) -> CudaError;
    pub(crate) fn cuda_region_device_ptr(handle: *mut c_void) -> *mut c_void;
    pub(crate) fn cuda_region_device_id(handle: *mut c_void) -> c_int;
    pub(crate) fn cuda_region_size(handle: *mut c_void) -> usize;
}

/// Safe wrapper functions with input validation
impl CudaError {
    /// Validate that a handle pointer is non-null
    #[inline]
    pub(crate) fn validate_handle(handle: *mut c_void) -> Result<(), CudaError> {
        if handle.is_null() {
            Err(CudaError::CudaErrorInvalidValue)
        } else {
            Ok(())
        }
    }

    /// Validate that a data pointer and size are valid
    #[inline]
    pub(crate) fn validate_buffer<T>(data: *const T, element_count: usize) -> Result<(), CudaError> {
        if data.is_null() && element_count > 0 {
            Err(CudaError::CudaErrorInvalidValue)
        } else {
            Ok(())
        }
    }
}
