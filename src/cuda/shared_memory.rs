//! Safe wrapper for CUDA shared memory regions with IPC support
//!
//! This module provides high-level abstractions for CUDA shared memory
//! specifically designed for Triton C-API integration with IPC handles.
//!
//! ## Memory Lifecycle Management
//!
//! **Reference Counting for Safe IPC Memory Sharing**
//!
//! When memory is shared via CUDA IPC with an external Triton server:
//! - Rust process creates memory and gets IPC handle
//! - External process (Triton) opens the memory via IPC handle
//! - **Problem:** If Rust frees memory while Triton still holds it → CRASH!
//!
//! **Solution:** Lease-based reference counting:
//! 1. Before using memory: `acquire_lease()` - increments counter
//! 2. After using memory: `release_lease()` - decrements counter
//! 3. On Drop: Wait for counter to reach 0 before freeing (with timeout)
//!
//! This ensures memory is only freed when all users have finished with it.

use crate::cuda::async_stream::AsyncCudaStream;
use crate::cuda::device_buffer::DeviceBuffer;
use crate::cuda::error::{CudaError, CudaSharedMemoryError};
use crate::cuda::ffi::*;
use crate::cuda::types::ModelConfig;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// RAII guard for memory leases
///
/// Automatically releases the lease when dropped, ensuring proper cleanup.
/// This prevents memory from being freed while it's still in use.
#[must_use = "MemoryLease must be held for the duration of memory access. Dropping it immediately defeats its purpose."]
pub struct MemoryLease {
    lease_counter: Arc<AtomicUsize>,
}

impl MemoryLease {
    /// Create a new memory lease (private - use acquire_lease())
    pub(crate) fn new(lease_counter: Arc<AtomicUsize>) -> Self {
        lease_counter.fetch_add(1, Ordering::SeqCst);
        Self { lease_counter }
    }
}

impl Drop for MemoryLease {
    fn drop(&mut self) {
        self.lease_counter.fetch_sub(1, Ordering::SeqCst);
    }
}

// Make it safe to send between threads
unsafe impl Send for MemoryLease {}
unsafe impl Sync for MemoryLease {}

/// Safe wrapper for CUDA shared memory region
///
/// This is a higher-level abstraction specifically for Triton C-API integration
/// with IPC handles, built on top of the lower-level DeviceBuffer.
///
/// ## Usage Example
///
/// ```ignore
/// // Create shared memory
/// let region = CudaSharedMemoryRegion::new("input", 1024, 0)?;
///
/// // Before inference
/// let lease = region.acquire_lease();
///
/// // Perform inference...
///
/// // After inference (automatic on drop)
/// drop(lease);
/// ```
pub struct CudaSharedMemoryRegion {
    handle: *mut c_void,
    /// Track if memory is shared via IPC (don't free if true)
    /// Uses AtomicBool for interior mutability
    is_ipc_shared: AtomicBool,
    /// Reference counter for active leases
    /// Shared via Arc so clones increment the same counter
    active_leases: Arc<AtomicUsize>,
}

impl CudaSharedMemoryRegion {
    /// Create a new CUDA shared memory region
    pub fn new(name: &str, size: usize, device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        // Validate device_id before attempting CUDA operations
        CudaSharedMemoryError::validate_device_id(device_id)?;

        let c_name = CString::new(name).map_err(|_| CudaSharedMemoryError::InvalidValue)?;

        let mut handle: *mut c_void = std::ptr::null_mut();

        let result =
            unsafe { CudaSharedMemoryRegionCreate(c_name.as_ptr(), size, device_id, &mut handle) };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        if handle.is_null() {
            return Err(CudaSharedMemoryError::NullPointer);
        }

        Ok(CudaSharedMemoryRegion {
            handle,
            is_ipc_shared: AtomicBool::new(false), // Will be set to true when IPC handle is obtained
            active_leases: Arc::new(AtomicUsize::new(0)), // No active leases initially
        })
    }

    /// Get the raw CUDA IPC handle
    ///
    /// **IMPORTANT:** Once this is called, the memory is marked as IPC-shared and will
    /// NOT be automatically freed on Drop to prevent cross-process use-after-free bugs.
    pub fn get_raw_handle(&self) -> Result<Vec<u8>, CudaSharedMemoryError> {
        let mut raw_handle: *mut i8 = std::ptr::null_mut();

        let result = unsafe { GetRawHandle(self.handle, &mut raw_handle) };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        if raw_handle.is_null() {
            return Err(CudaSharedMemoryError::NullPointer);
        }

        // Mark memory as IPC shared - it will not be freed on Drop
        self.is_ipc_shared.store(true, Ordering::Release);

        // SAFETY:
        // - CStr::from_ptr is safe because:
        //   1. raw_handle is guaranteed non-null (checked above)
        //   2. raw_handle points to a valid C string (null-terminated) from GetRawHandle
        //   3. The C string's lifetime is managed by the C code until we call FreeRawHandle
        // - FreeRawHandle is safe to call because:
        //   1. raw_handle is the pointer returned by GetRawHandle
        //   2. We call it exactly once before raw_handle goes out of scope
        let bytes = unsafe {
            let c_str = CStr::from_ptr(raw_handle);
            let bytes = c_str.to_bytes().to_vec();

            // Free the C-allocated memory
            let _ = FreeRawHandle(raw_handle);

            bytes
        };

        Ok(bytes)
    }

    /// Acquire a lease on this memory region
    ///
    /// The lease increments a reference counter and prevents the memory from being freed
    /// until the lease is dropped. This ensures safe cross-process memory sharing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let region = CudaSharedMemoryRegion::new("input", 1024, 0)?;
    ///
    /// {
    ///     let _lease = region.acquire_lease();
    ///     // Memory is safe to use...
    ///     // Perform inference...
    /// } // Lease dropped here, counter decremented
    /// ```
    pub fn acquire_lease(&self) -> MemoryLease {
        MemoryLease::new(Arc::clone(&self.active_leases))
    }

    /// Get the number of active leases on this memory region
    ///
    /// Useful for debugging and monitoring memory usage.
    pub fn active_lease_count(&self) -> usize {
        self.active_leases.load(Ordering::SeqCst)
    }

    /// Write f32 data to the region
    pub fn write_f32_data(&self, data: &[f32]) -> Result<(), CudaSharedMemoryError> {
        // Validate inputs before FFI call using existing validation functions
        CudaError::validate_handle(self.handle).map_err(CudaSharedMemoryError::from)?;
        CudaError::validate_buffer(data.as_ptr(), data.len()).map_err(CudaSharedMemoryError::from)?;

        let result = unsafe { WriteTestData(self.handle, data.as_ptr(), data.len()) };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }

    /// Enqueue async write f32 data to the region (non-blocking)
    pub fn enqueue_write_f32_data(
        &self,
        data: &[f32],
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        // Validate inputs before FFI call using existing validation functions
        CudaError::validate_handle(self.handle).map_err(CudaSharedMemoryError::from)?;
        CudaError::validate_buffer(data.as_ptr(), data.len()).map_err(CudaSharedMemoryError::from)?;

        // Create a device buffer from the shared memory region
        let device_buffer = unsafe { self.as_device_buffer::<f32>(data.len()) };

        // Enqueue copy to device using the stream (non-blocking)
        let mut mut_buffer = device_buffer;
        mut_buffer.enqueue_copy_from_host(data, stream)?;

        Ok(())
    }

    /// Asynchronously write f32 data to the region using a CUDA stream (blocks until complete)
    pub async fn write_f32_data_async(
        &self,
        data: &[f32],
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_write_f32_data(data, stream)?;
        stream.wait().await
    }

    /// Read f32 data from the region
    pub fn read_f32_data(&self, element_count: usize) -> Result<Vec<f32>, CudaSharedMemoryError> {
        // Validate handle before FFI call using existing validation function
        CudaError::validate_handle(self.handle).map_err(CudaSharedMemoryError::from)?;

        let mut data = vec![0.0f32; element_count];

        // Validate buffer after allocation using existing validation function
        CudaError::validate_buffer(data.as_ptr(), element_count).map_err(CudaSharedMemoryError::from)?;

        let result = unsafe { ReadTestData(self.handle, data.as_mut_ptr(), element_count) };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(data)
    }

    /// Enqueue async read f32 data from the region (non-blocking)
    pub fn enqueue_read_f32_data(
        &self,
        data: &mut [f32],
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        // Validate inputs before FFI call using existing validation functions
        CudaError::validate_handle(self.handle).map_err(CudaSharedMemoryError::from)?;
        CudaError::validate_buffer(data.as_ptr(), data.len()).map_err(CudaSharedMemoryError::from)?;

        // Create a device buffer from the shared memory region
        let device_buffer = unsafe { self.as_device_buffer::<f32>(data.len()) };

        // Enqueue copy from device to host using the stream (non-blocking)
        device_buffer.enqueue_copy_to_host(data, stream)?;

        Ok(())
    }

    /// Asynchronously read f32 data from the region using a CUDA stream (blocks until complete)
    pub async fn read_f32_data_async(
        &self,
        element_count: usize,
        stream: &AsyncCudaStream,
    ) -> Result<Vec<f32>, CudaSharedMemoryError> {
        let mut data = vec![0.0f32; element_count];
        self.enqueue_read_f32_data(&mut data, stream)?;
        stream.wait().await?;
        Ok(data)
    }

    /// Register with Triton server
    ///
    /// **IMPORTANT:** Once this is called, the memory is marked as IPC-shared and will
    /// NOT be automatically freed on Drop to prevent cross-process use-after-free bugs.
    pub fn register_with_triton_server(&self) -> Result<(), CudaSharedMemoryError> {
        let result = unsafe { RegisterWithTritonServer(self.handle) };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        // Mark memory as IPC shared - it will not be freed on Drop
        self.is_ipc_shared.store(true, Ordering::Release);

        Ok(())
    }

    /// Run inference with the specified model configuration
    ///
    /// Automatically acquires and releases a memory lease for the duration of the inference.
    pub fn run_inference_with_config(
        &self,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
    ) -> Result<(), CudaSharedMemoryError> {
        // Acquire lease for the duration of this inference
        let _lease = self.acquire_lease();
        let model_name =
            CString::new(config.name.as_str()).map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let input_name_c =
            CString::new(input_name).map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let output_name_c =
            CString::new(output_name).map_err(|_| CudaSharedMemoryError::InvalidValue)?;

        let input_spec = config
            .inputs
            .get(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let buffer_size = config
            .calculate_buffer_size(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let result = unsafe {
            RunTritonInferenceWithConfig(
                self.handle,
                model_name.as_ptr(),
                input_name_c.as_ptr(),
                input_spec.data_type.to_c_type(),
                input_spec.dims.as_ptr(),
                input_spec.dims.len(),
                output_name_c.as_ptr(),
                buffer_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }

    /// Enqueue stream-aware Triton inference (non-blocking)
    pub fn enqueue_inference_with_config(
        &self,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        // Record an event before inference for stream ordering
        let _pre_inference_event = stream.record_event()?;

        // Run the inference (this will be automatically ordered after previous operations on the stream)
        self.run_inference_with_config(config, input_name, output_name)?;

        // Record an event after inference for future synchronization
        let _post_inference_event = stream.record_event()?;

        Ok(())
    }

    /// Asynchronously run inference with the specified model configuration (blocks until complete)
    pub async fn run_inference_with_config_async(
        &self,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_inference_with_config(config, input_name, output_name, stream)?;
        stream.wait().await
    }

    /// Create a typed DeviceBuffer view of this shared memory region
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The memory region contains valid data of type T (or will be initialized before reading)
    /// - The capacity doesn't exceed the actual allocated size
    /// - The memory is properly aligned for type T (CUDA allocations are typically 256-byte aligned)
    /// - Type T is 'static and safe to use with CUDA (typically POD types like f32, i32, etc.)
    /// - No aliasing violations occur (only one mutable DeviceBuffer per memory region at a time)
    ///
    /// # Implementation Safety
    ///
    /// This function is safe internally because:
    /// - cuda_region_device_ptr returns a valid CUDA device pointer from the C API
    /// - We calculate max_capacity to prevent buffer overruns (region_bytes / elem_size)
    /// - We clamp the requested capacity to max_capacity
    /// - DeviceBuffer::from_raw_parts takes ownership of managing this memory view
    pub unsafe fn as_device_buffer<T: 'static>(&self, capacity: usize) -> DeviceBuffer<T> {
        // SAFETY: All FFI calls here return valid values for this handle:
        // - cuda_region_device_ptr: returns device memory pointer (valid for device operations)
        // - cuda_region_device_id: returns the CUDA device ID (always valid if region exists)
        // - cuda_region_size: returns the allocated size in bytes (always accurate)
        let ptr = cuda_region_device_ptr(self.handle) as *mut T;
        let device_id = cuda_region_device_id(self.handle);
        let region_bytes = cuda_region_size(self.handle);
        let elem_size = std::mem::size_of::<T>();
        let max_capacity = if elem_size == 0 {
            0
        } else {
            region_bytes / elem_size
        };
        let len = capacity.min(max_capacity);
        DeviceBuffer::from_raw_parts(ptr, len, device_id)
    }

    /// Run inference with separate input and output regions
    ///
    /// Automatically acquires and releases memory leases for both input and output regions
    /// for the duration of the inference.
    pub fn run_inference_with_output_regions(
        &self,
        output_region: &CudaSharedMemoryRegion,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
    ) -> Result<(), CudaSharedMemoryError> {
        // Acquire leases for both input and output regions
        let _input_lease = self.acquire_lease();
        let _output_lease = output_region.acquire_lease();
        let model_name =
            CString::new(config.name.as_str()).map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let input_name_c =
            CString::new(input_name).map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let output_name_c =
            CString::new(output_name).map_err(|_| CudaSharedMemoryError::InvalidValue)?;

        let input_spec = config
            .inputs
            .get(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let input_buffer_size = config
            .calculate_buffer_size(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let output_buffer_size = config
            .calculate_output_buffer_size(output_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let result = unsafe {
            RunTritonInferenceWithOutputRegions(
                self.handle,
                output_region.handle,
                model_name.as_ptr(),
                input_name_c.as_ptr(),
                input_spec.data_type.to_c_type(),
                input_spec.dims.as_ptr(),
                input_spec.dims.len(),
                output_name_c.as_ptr(),
                input_buffer_size,
                output_buffer_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }

    /// Enqueue stream-aware Triton inference with separate input and output regions (non-blocking)
    pub fn enqueue_inference_with_output_regions(
        &self,
        output_region: &CudaSharedMemoryRegion,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        // Record an event before inference for stream ordering
        let _pre_inference_event = stream.record_event()?;

        // Run the inference (this will be automatically ordered after previous operations on the stream)
        self.run_inference_with_output_regions(output_region, config, input_name, output_name)?;

        // Record an event after inference for future synchronization
        let _post_inference_event = stream.record_event()?;

        Ok(())
    }

    /// Asynchronously run inference with separate input and output regions (blocks until complete)
    pub async fn run_inference_with_output_regions_async(
        &self,
        output_region: &CudaSharedMemoryRegion,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
        stream: &AsyncCudaStream,
    ) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_inference_with_output_regions(
            output_region,
            config,
            input_name,
            output_name,
            stream,
        )?;
        stream.wait().await
    }
}

impl Drop for CudaSharedMemoryRegion {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Wait for all active leases to be released before freeing memory
            const LEASE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

            let start = Instant::now();
            let mut last_count = self.active_leases.load(Ordering::SeqCst);

            // Wait for leases to reach 0
            while last_count > 0 {
                if start.elapsed() > LEASE_WAIT_TIMEOUT {
                    eprintln!(
                        "WARNING: CUDA memory region {:p} still has {} active leases after {}s timeout",
                        self.handle, last_count, LEASE_WAIT_TIMEOUT.as_secs()
                    );

                    // SAFETY: Always leak memory when leases are still active, regardless of IPC status.
                    // This prevents both cross-process crashes (IPC case) and use-after-free within
                    // this process (non-IPC case). Memory leaks are safer than undefined behavior.
                    let shared_status = if self.is_ipc_shared.load(Ordering::Acquire) {
                        "IPC-shared"
                    } else {
                        "local"
                    };
                    eprintln!(
                        "    Memory is {} - leaking to prevent use-after-free",
                        shared_status
                    );
                    eprintln!("    This is safer than potentially causing undefined behavior");
                    return; // Intentional leak for safety
                }

                // Use spinloop with yield instead of blocking sleep to avoid blocking async executors
                // This is acceptable in Drop as it should be a rare case
                std::hint::spin_loop();
                std::thread::yield_now();

                last_count = self.active_leases.load(Ordering::SeqCst);
            }

            // All leases released - safe to free
            // Note: If last_count > 0, we would have returned early above to leak the memory
            let result = unsafe { CudaSharedMemoryRegionDestroy(self.handle) };
            if result != CudaError::CudaSuccess {
                eprintln!(
                    "Warning: Failed to destroy CUDA shared memory region: {:?}",
                    result
                );
            }
        }
    }
}

// Make it safe to send between threads
unsafe impl Send for CudaSharedMemoryRegion {}
unsafe impl Sync for CudaSharedMemoryRegion {}
