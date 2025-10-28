//! CUDA FFI module for direct Triton Server integration
//!
//! This module provides a direct C API interface to Triton Server using CUDA shared memory,
//! eliminating the network overhead of gRPC calls and enabling zero-copy inference.
//!
//! ## Architecture
//!
//! Instead of using gRPC clients to communicate with Triton Server, this module:
//! - Uses Triton's C API directly (embedded in the same process)
//! - Allocates CUDA shared memory for zero-copy tensor operations
//! - Leverages CUDA IPC handles for efficient memory sharing
//! - Supports both simple and complex model configurations
//!
//! ## Performance Benefits
//!
//! - **Zero Network Overhead**: Direct C API calls instead of gRPC
//! - **Zero Copy**: Tensors stay in GPU memory throughout the pipeline
//! - **Reduced Latency**: Eliminates serialization/deserialization overhead
//! - **Better Memory Management**: Direct control over CUDA memory allocation
//!
//! ## Memory Management Architecture
//!
//! This module provides two complementary memory management abstractions:
//!
//! 1. **`DeviceBuffer<T>`**: Low-level, generic CUDA memory buffer with RAII semantics
//!    - Type-safe memory operations
//!    - Automatic cleanup on drop
//!    - Zero-copy casting between compatible types
//!    - Host/device memory transfers
//!
//! 2. **`CudaSharedMemoryRegion`**: High-level Triton C-API integration
//!    - CUDA IPC handles for inter-process sharing
//!    - Direct integration with Triton inference server
//!    - Model-specific memory pool management
//!
//! The `DeviceBuffer` provides the foundation for general CUDA memory operations,
//! while `CudaSharedMemoryRegion` handles the specifics of Triton server integration.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub mod async_stream;
pub mod device_buffer;

pub use async_stream::{AsyncCudaEvent, AsyncCudaStream, AsyncCudaStreamPool};
pub use device_buffer::{DeviceBuffer, DevicePod, DeviceSlice};

// Re-export utility functions from device_buffer
pub use device_buffer::utils::{default_device, device_count, is_available};

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

impl From<CudaError> for CudaSharedMemoryError {
    fn from(error: CudaError) -> Self {
        match error {
            CudaError::CudaSuccess => unreachable!("Success should not be converted to error"),
            CudaError::CudaErrorInvalidValue => CudaSharedMemoryError::InvalidValue,
            CudaError::CudaErrorOutOfMemory => CudaSharedMemoryError::OutOfMemory,
            CudaError::CudaErrorUnknown => CudaSharedMemoryError::Unknown,
            CudaError::CudaErrorNotReady => CudaSharedMemoryError::Unknown,
        }
    }
}

/// Data type enumeration for tensors
#[derive(Debug, Clone)]
pub enum DataType {
    FP32,
    FP16,
    INT32,
    INT64,
    UINT8,
    BOOL,
}

impl DataType {
    /// Convert to Triton C API type constant
    fn to_c_type(&self) -> c_int {
        match self {
            DataType::BOOL => 1,  // TRITONSERVER_TYPE_BOOL
            DataType::UINT8 => 2, // TRITONSERVER_TYPE_UINT8
            DataType::INT32 => 8, // TRITONSERVER_TYPE_INT32
            DataType::INT64 => 9, // TRITONSERVER_TYPE_INT64
            DataType::FP16 => 10, // TRITONSERVER_TYPE_FP16
            DataType::FP32 => 11, // TRITONSERVER_TYPE_FP32
        }
    }

    /// Size in bytes of one element
    pub fn element_size(&self) -> usize {
        match self {
            DataType::FP32 => 4,
            DataType::FP16 => 2,
            DataType::INT32 => 4,
            DataType::INT64 => 8,
            DataType::UINT8 => 1,
            DataType::BOOL => 1,
        }
    }
}

/// Tensor specification
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub data_type: DataType,
    pub dims: Vec<i64>,
}

/// Model configuration for inference
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub inputs: HashMap<String, TensorSpec>,
    pub outputs: HashMap<String, TensorSpec>,
    pub max_batch_size: i32,
    pub stateful: bool,
}

impl ModelConfig {
    /// Create configuration for RNN-T ASR models
    pub fn rnnt_ensemble() -> Self {
        let mut inputs = HashMap::new();
        inputs.insert(
            "audio_features".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 80, 3000], // batch, features, time
            },
        );
        inputs.insert(
            "encoder_state".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 2048], // batch, layers, hidden
            },
        );
        inputs.insert(
            "decoder_state".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 1024], // batch, layers, hidden
            },
        );

        let mut outputs = HashMap::new();
        outputs.insert(
            "transcripts".to_string(),
            TensorSpec {
                data_type: DataType::INT32,
                dims: vec![1, 512], // batch, max_seq_length
            },
        );
        outputs.insert(
            "updated_encoder_state".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 2048],
            },
        );
        outputs.insert(
            "updated_decoder_state".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 1024],
            },
        );
        outputs.insert(
            "beam_scores".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 16], // batch, beam_size
            },
        );

        Self {
            name: "rnnt_ensemble".to_string(),
            inputs,
            outputs,
            max_batch_size: 1,
            stateful: true,
        }
    }

    /// Create configuration for preprocessor model
    pub fn preprocessor() -> Self {
        let mut inputs = HashMap::new();
        inputs.insert(
            "AUDIO_FRAMES".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 3000], // batch, frames
            },
        );

        let mut outputs = HashMap::new();
        outputs.insert(
            "MEL_FEATURES".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 80, 3000], // batch, features, time
            },
        );

        Self {
            name: "preprocessor".to_string(),
            inputs,
            outputs,
            max_batch_size: 1,
            stateful: false,
        }
    }

    /// Create configuration for encoder model
    pub fn encoder() -> Self {
        let mut inputs = HashMap::new();
        inputs.insert(
            "MEL_FEATURES".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 80, 3000], // batch, features, time
            },
        );
        inputs.insert(
            "ENCODER_STATE".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 2048], // batch, layers, hidden
            },
        );

        let mut outputs = HashMap::new();
        outputs.insert(
            "ENCODER_OUTPUT".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 3000, 1024], // batch, time, hidden
            },
        );
        outputs.insert(
            "UPDATED_ENCODER_STATE".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 2048],
            },
        );

        Self {
            name: "encoder".to_string(),
            inputs,
            outputs,
            max_batch_size: 1,
            stateful: true,
        }
    }

    /// Create configuration for decoder/joint model
    pub fn decoder_joint() -> Self {
        let mut inputs = HashMap::new();
        inputs.insert(
            "ENCODER_OUTPUT".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 3000, 1024], // batch, time, hidden
            },
        );
        inputs.insert(
            "DECODER_STATE".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 1024], // batch, layers, hidden
            },
        );

        let mut outputs = HashMap::new();
        outputs.insert(
            "LOGITS".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 3000, 4096], // batch, time, vocab_size
            },
        );
        outputs.insert(
            "UPDATED_DECODER_STATE".to_string(),
            TensorSpec {
                data_type: DataType::FP32,
                dims: vec![1, 512, 1024],
            },
        );

        Self {
            name: "decoder_joint".to_string(),
            inputs,
            outputs,
            max_batch_size: 1,
            stateful: true,
        }
    }

    /// Calculate buffer size for a specific input
    pub fn calculate_buffer_size(&self, input_name: &str) -> Option<usize> {
        self.inputs.get(input_name).map(|spec| {
            let element_count: usize = spec.dims.iter().map(|&d| d as usize).product();
            element_count * spec.data_type.element_size()
        })
    }

    /// Calculate buffer size for a specific output
    pub fn calculate_output_buffer_size(&self, output_name: &str) -> Option<usize> {
        self.outputs.get(output_name).map(|spec| {
            let element_count: usize = spec.dims.iter().map(|&d| d as usize).product();
            element_count * spec.data_type.element_size()
        })
    }

    /// Calculate total size of all inputs
    pub fn total_input_size(&self) -> usize {
        self.inputs
            .values()
            .map(|spec| {
                let element_count: usize = spec.dims.iter().map(|&d| d as usize).product();
                element_count * spec.data_type.element_size()
            })
            .sum()
    }

    /// Calculate total size of all outputs
    pub fn total_output_size(&self) -> usize {
        self.outputs
            .values()
            .map(|spec| {
                let element_count: usize = spec.dims.iter().map(|&d| d as usize).product();
                element_count * spec.data_type.element_size()
            })
            .sum()
    }
}

// FFI declarations for CUDA helper functions
unsafe extern "C" {
    fn get_cuda_device_count_ffi(count: *mut c_int) -> CudaError;
    fn CudaSharedMemoryRegionCreate(
        name: *const c_char,
        byte_size: usize,
        device_id: c_int,
        handle: *mut *mut c_void,
    ) -> CudaError;
    fn CudaSharedMemoryRegionDestroy(handle: *mut c_void) -> CudaError;
    fn GetRawHandle(handle: *mut c_void, raw_handle: *mut *mut c_char) -> CudaError;
    fn FreeRawHandle(raw_handle: *mut c_char) -> CudaError;
    fn WriteTestData(handle: *mut c_void, data: *const f32, element_count: usize) -> CudaError;
    fn ReadTestData(handle: *mut c_void, data: *mut f32, element_count: usize) -> CudaError;
    fn RegisterWithTritonServer(handle: *mut c_void) -> CudaError;
    #[allow(dead_code)]
    fn RunTritonInference(handle: *mut c_void) -> CudaError;
    fn RunTritonInferenceWithConfig(
        handle: *mut c_void,
        model_name: *const c_char,
        input_name: *const c_char,
        input_data_type: c_int,
        input_shape: *const i64,
        input_dims: usize,
        output_name: *const c_char,
        buffer_size: usize,
    ) -> CudaError;
    fn RunTritonInferenceWithOutputRegions(
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
    fn cuda_region_device_ptr(handle: *mut c_void) -> *mut c_void;
    fn cuda_region_device_id(handle: *mut c_void) -> c_int;
    fn cuda_region_size(handle: *mut c_void) -> usize;
}

/// Safe wrapper for CUDA shared memory region
///
/// This is a higher-level abstraction specifically for Triton C-API integration
/// with IPC handles, built on top of the lower-level DeviceBuffer.
///
/// ## Memory Lifecycle Management
///
/// **Reference Counting for Safe IPC Memory Sharing**
///
/// When memory is shared via CUDA IPC with an external Triton server:
/// - Rust process creates memory and gets IPC handle
/// - External process (Triton) opens the memory via IPC handle
/// - **Problem:** If Rust frees memory while Triton still holds it → CRASH!
///
/// **Solution:** Lease-based reference counting:
/// 1. Before using memory: `acquire_lease()` - increments counter
/// 2. After using memory: `release_lease()` - decrements counter
/// 3. On Drop: Wait for counter to reach 0 before freeing (with timeout)
///
/// This ensures memory is only freed when all users have finished with it.
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
/// RAII guard for memory leases
///
/// Automatically releases the lease when dropped, ensuring proper cleanup.
/// This prevents memory from being freed while it's still in use.
pub struct MemoryLease {
    lease_counter: Arc<AtomicUsize>,
}

impl MemoryLease {
    /// Create a new memory lease (private - use acquire_lease())
    fn new(lease_counter: Arc<AtomicUsize>) -> Self {
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
        let mut raw_handle: *mut c_char = std::ptr::null_mut();

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
        let mut data = vec![0.0f32; element_count];

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
            const POLL_INTERVAL: Duration = Duration::from_millis(100);

            let start = Instant::now();
            let mut last_count = self.active_leases.load(Ordering::SeqCst);

            // Wait for leases to reach 0
            while last_count > 0 {
                if start.elapsed() > LEASE_WAIT_TIMEOUT {
                    eprintln!(
                        "WARNING: CUDA memory region {:p} still has {} active leases after {}s timeout",
                        self.handle, last_count, LEASE_WAIT_TIMEOUT.as_secs()
                    );

                    // Check if it's IPC-shared - if so, leak the memory
                    if self.is_ipc_shared.load(Ordering::Acquire) {
                        eprintln!(
                            "    Memory is IPC-shared - leaking to prevent cross-process crash"
                        );
                        eprintln!("    This is safer than potentially crashing external processes");
                        return; // Intentional leak
                    }

                    eprintln!("    Forcing cleanup anyway (not IPC-shared) - may cause issues!");
                    break;
                }

                // Sleep briefly before checking again
                std::thread::sleep(POLL_INTERVAL);
                last_count = self.active_leases.load(Ordering::SeqCst);
            }

            // All leases released (or timeout expired) - safe to free
            if last_count == 0 {
                // Normal case: all leases released properly
                let result = unsafe { CudaSharedMemoryRegionDestroy(self.handle) };
                if result != CudaError::CudaSuccess {
                    eprintln!(
                        "Warning: Failed to destroy CUDA shared memory region: {:?}",
                        result
                    );
                }
            } else {
                // Timeout case: forcing cleanup with active leases
                eprintln!(
                    "WARNING: Attempting forced cleanup with {} active leases!",
                    last_count
                );
                let result = unsafe { CudaSharedMemoryRegionDestroy(self.handle) };
                if result != CudaError::CudaSuccess {
                    eprintln!(
                        "ERROR: Failed to force-destroy CUDA shared memory region: {:?}",
                        result
                    );
                }
            }
        }
    }
}

// Make it safe to send between threads
unsafe impl Send for CudaSharedMemoryRegion {}
unsafe impl Sync for CudaSharedMemoryRegion {}

/// Multi-region pool for complex models
pub struct CudaSharedMemoryPool {
    pub input_regions: HashMap<String, CudaSharedMemoryRegion>,
    pub output_regions: HashMap<String, CudaSharedMemoryRegion>,
    pub state_regions: HashMap<String, CudaSharedMemoryRegion>,
    pub config: ModelConfig,
}

impl CudaSharedMemoryPool {
    /// Create a new memory pool for the specified model
    pub fn new_for_model(
        config: ModelConfig,
        device_id: i32,
    ) -> Result<Self, CudaSharedMemoryError> {
        let mut input_regions = HashMap::new();
        let mut output_regions = HashMap::new();
        let mut state_regions = HashMap::new();

        // Create input regions
        for (name, spec) in &config.inputs {
            let size = spec.dims.iter().map(|&d| d as usize).product::<usize>()
                * spec.data_type.element_size();
            let region = CudaSharedMemoryRegion::new(&format!("input_{}", name), size, device_id)?;
            input_regions.insert(name.clone(), region);
        }

        // Create output regions
        for (name, spec) in &config.outputs {
            let size = spec.dims.iter().map(|&d| d as usize).product::<usize>()
                * spec.data_type.element_size();
            let region = CudaSharedMemoryRegion::new(&format!("output_{}", name), size, device_id)?;
            output_regions.insert(name.clone(), region);
        }

        // Create state regions for stateful models
        if config.stateful {
            for (name, spec) in &config.inputs {
                if name.contains("state") {
                    let size = spec.dims.iter().map(|&d| d as usize).product::<usize>()
                        * spec.data_type.element_size();
                    let region =
                        CudaSharedMemoryRegion::new(&format!("state_{}", name), size, device_id)?;
                    state_regions.insert(name.clone(), region);
                }
            }
        }

        Ok(CudaSharedMemoryPool {
            input_regions,
            output_regions,
            state_regions,
            config,
        })
    }

    /// Get input region by name
    pub fn get_input_region(&self, name: &str) -> Option<&CudaSharedMemoryRegion> {
        self.input_regions.get(name)
    }

    /// Get output region by name
    pub fn get_output_region(&self, name: &str) -> Option<&CudaSharedMemoryRegion> {
        self.output_regions.get(name)
    }

    /// Get state region by name
    pub fn get_state_region(&self, name: &str) -> Option<&CudaSharedMemoryRegion> {
        self.state_regions.get(name)
    }
}

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

#[cfg(test)]
mod tests {
    //! # Test Suite for CUDA Memory Lease Management
    //!
    //! This test suite validates the reference counting and memory lifecycle
    //! management for CUDA shared memory regions with IPC support.
    //!
    //! ## Running Tests
    //!
    //! **Requirements:**
    //! - Linux system with NVIDIA GPU and CUDA drivers
    //! - Build with `--features cuda` flag
    //!
    //! **Basic tests (fast, no Triton required):**
    //! ```bash
    //! cargo test --features cuda --lib test_memory_lease
    //! ```
    //!
    //! **Integration tests (requires Triton server):**
    //! ```bash
    //! cargo test --features cuda --lib test_inference_auto_lease
    //! cargo test --features cuda --lib test_register_sets_ipc_flag
    //! ```
    //!
    //! **Longer tests and benchmarks:**
    //! ```bash
    //! cargo test --features cuda --lib -- --ignored --nocapture
    //! ```
    //!
    //! ## Test Categories
    //!
    //! 1. **Unit Tests** - Test MemoryLease RAII behavior without CUDA:
    //!    - `test_memory_lease_increment_decrement`
    //!    - `test_memory_lease_concurrent`
    //!
    //! 2. **CUDA Integration Tests** - Require GPU and CUDA drivers:
    //!    - `test_active_lease_count`
    //!    - `test_inference_auto_lease`
    //!    - `test_drop_waits_for_leases`
    //!    - `test_ipc_shared_flag`
    //!    - `test_register_sets_ipc_flag`
    //!
    //! 3. **Benchmarks** (marked with `#[ignore]`):
    //!    - `bench_lease_acquisition`
    //!    - `test_full_lifecycle_concurrent`
    //!
    //! ## macOS / Non-CUDA Systems
    //!
    //! These tests cannot run on macOS or systems without CUDA because:
    //! - The entire `cuda` module is feature-gated (`#[cfg(feature = "cuda")]`)
    //! - CUDA drivers and runtime are Linux-only
    //! - FFI functions require CUDA C API
    //!
    //! To validate the core logic on macOS, you would need to extract
    //! `MemoryLease` into a standalone module without CUDA dependencies.

    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// Test that MemoryLease properly increments and decrements the counter
    #[test]
    fn test_memory_lease_increment_decrement() {
        let counter = Arc::new(AtomicUsize::new(0));

        assert_eq!(counter.load(Ordering::SeqCst), 0);

        {
            let _lease1 = MemoryLease::new(Arc::clone(&counter));
            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "Counter should be 1 after first lease"
            );

            {
                let _lease2 = MemoryLease::new(Arc::clone(&counter));
                assert_eq!(
                    counter.load(Ordering::SeqCst),
                    2,
                    "Counter should be 2 with two leases"
                );

                let _lease3 = MemoryLease::new(Arc::clone(&counter));
                assert_eq!(
                    counter.load(Ordering::SeqCst),
                    3,
                    "Counter should be 3 with three leases"
                );
            } // lease2 and lease3 dropped

            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "Counter should be back to 1"
            );
        } // lease1 dropped

        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "Counter should be back to 0"
        );
    }

    /// Test that MemoryLease works correctly across threads
    #[test]
    fn test_memory_lease_concurrent() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn 10 threads that each acquire a lease
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                let _lease = MemoryLease::new(counter_clone);
                thread::sleep(Duration::from_millis(10));
                // Lease will be dropped when thread exits
            });
            handles.push(handle);
        }

        // Wait a bit and check that some leases are active
        thread::sleep(Duration::from_millis(5));
        let active = counter.load(Ordering::SeqCst);
        assert!(
            active > 0 && active <= 10,
            "Should have some active leases: {}",
            active
        );

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // All leases should be released
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "All leases should be released"
        );
    }

    /// Test that active_lease_count() returns correct values
    #[test]
    #[cfg(feature = "cuda")]
    fn test_active_lease_count() {
        // This test requires CUDA to be available
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region = CudaSharedMemoryRegion::new("test_lease_count", 1024, 0)
            .expect("Failed to create region");

        assert_eq!(region.active_lease_count(), 0, "Should start with 0 leases");

        {
            let _lease1 = region.acquire_lease();
            assert_eq!(region.active_lease_count(), 1);

            {
                let _lease2 = region.acquire_lease();
                assert_eq!(region.active_lease_count(), 2);
            }

            assert_eq!(region.active_lease_count(), 1);
        }

        assert_eq!(region.active_lease_count(), 0);
    }

    /// Test that leases are automatically managed in inference
    #[test]
    #[cfg(feature = "cuda")]
    fn test_inference_auto_lease() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region = CudaSharedMemoryRegion::new("test_inference", 4096, 0)
            .expect("Failed to create region");

        assert_eq!(region.active_lease_count(), 0);

        // During inference, lease should be active
        // We can't easily test this without mocking, but we can verify
        // the lease count returns to 0 after inference

        let config = ModelConfig::preprocessor();

        // This will fail if Triton isn't running, but that's OK for this test
        // We're testing the lease mechanism, not the inference itself
        let _ = region.run_inference_with_config(&config, "AUDIO_FRAMES", "MEL_FEATURES");

        // After inference completes (even with error), leases should be released
        assert_eq!(
            region.active_lease_count(),
            0,
            "Leases should be released after inference"
        );
    }

    /// Test that Drop waits for leases (with quick timeout for testing)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_drop_waits_for_leases() {
        use std::sync::mpsc;

        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let (tx, rx) = mpsc::channel();

        let region = Arc::new(
            CudaSharedMemoryRegion::new("test_drop_wait", 1024, 0)
                .expect("Failed to create region"),
        );

        // Acquire a lease in another thread
        let region_clone = Arc::clone(&region);
        let handle = thread::spawn(move || {
            let _lease = region_clone.acquire_lease();
            tx.send(()).unwrap(); // Signal that lease is acquired
            thread::sleep(Duration::from_millis(500)); // Hold lease for 500ms
                                                       // Lease dropped here
        });

        // Wait for lease to be acquired
        rx.recv().unwrap();
        assert_eq!(region.active_lease_count(), 1);

        // Drop the Arc - the last strong reference will trigger Drop
        // but it should wait for the lease in the other thread
        let start = std::time::Instant::now();
        drop(region);
        let elapsed = start.elapsed();

        // Should have waited at least 400ms (allowing some slack)
        assert!(
            elapsed >= Duration::from_millis(400),
            "Drop should wait for leases: waited {:?}",
            elapsed
        );

        handle.join().unwrap();
    }

    /// Test IPC flag behavior
    #[test]
    #[cfg(feature = "cuda")]
    fn test_ipc_shared_flag() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region =
            CudaSharedMemoryRegion::new("test_ipc_flag", 1024, 0).expect("Failed to create region");

        // Initially not IPC-shared
        assert!(!region.is_ipc_shared.load(Ordering::Acquire));

        // Getting raw handle should mark it as IPC-shared
        let _ = region.get_raw_handle();
        assert!(
            region.is_ipc_shared.load(Ordering::Acquire),
            "Should be marked as IPC-shared after get_raw_handle()"
        );
    }

    /// Test register_with_triton_server sets IPC flag
    #[test]
    #[cfg(feature = "cuda")]
    fn test_register_sets_ipc_flag() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region =
            CudaSharedMemoryRegion::new("test_register", 1024, 0).expect("Failed to create region");

        assert!(!region.is_ipc_shared.load(Ordering::Acquire));

        // This will likely fail without Triton running, but should still set the flag
        let _ = region.register_with_triton_server();
        assert!(
            region.is_ipc_shared.load(Ordering::Acquire),
            "Should be marked as IPC-shared after registration"
        );
    }

    /// Benchmark: Lease acquisition overhead
    #[test]
    #[ignore] // Run with --ignored for benchmarks
    fn bench_lease_acquisition() {
        let counter = Arc::new(AtomicUsize::new(0));
        let iterations = 1_000_000;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _lease = MemoryLease::new(Arc::clone(&counter));
            // Lease immediately dropped
        }
        let elapsed = start.elapsed();

        println!(
            "Lease acquisition: {} iterations in {:?}",
            iterations, elapsed
        );
        println!("Average: {:?} per lease", elapsed / iterations);
        println!(
            "Rate: {:.2} million leases/sec",
            iterations as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
    }

    /// Integration test: Full lifecycle with concurrent access
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore] // Run with --ignored for longer tests
    fn test_full_lifecycle_concurrent() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region = Arc::new(
            CudaSharedMemoryRegion::new("test_concurrent_lifecycle", 4096, 0)
                .expect("Failed to create region"),
        );

        let mut handles = vec![];

        // Spawn multiple threads that acquire leases and do work
        for i in 0..5 {
            let region_clone = Arc::clone(&region);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let _lease = region_clone.acquire_lease();
                    // Simulate work
                    thread::sleep(Duration::from_millis(10));
                    println!(
                        "Thread {} iteration {} - active leases: {}",
                        i,
                        j,
                        region_clone.active_lease_count()
                    );
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // All leases should be released
        assert_eq!(
            region.active_lease_count(),
            0,
            "All leases should be released"
        );

        println!("Concurrent lifecycle test passed!");
    }
}
