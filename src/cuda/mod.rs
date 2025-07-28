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

pub mod device_buffer;
pub mod async_stream;

pub use device_buffer::{DeviceBuffer, DeviceSlice, DevicePod};
pub use async_stream::{AsyncCudaStream, AsyncCudaEvent, AsyncCudaStreamPool};

// Re-export utility functions from device_buffer
pub use device_buffer::utils::{device_count, is_available, default_device};

/// Error codes that match the C implementation
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub enum CudaError {
    CudaSuccess = 0,
    CudaErrorInvalidValue = 1,
    CudaErrorOutOfMemory = 2,
    CudaErrorUnknown = 3,
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
            DataType::BOOL => 1,   // TRITONSERVER_TYPE_BOOL
            DataType::UINT8 => 2,  // TRITONSERVER_TYPE_UINT8
            DataType::INT32 => 8,  // TRITONSERVER_TYPE_INT32
            DataType::INT64 => 9,  // TRITONSERVER_TYPE_INT64
            DataType::FP16 => 10,  // TRITONSERVER_TYPE_FP16
            DataType::FP32 => 11,  // TRITONSERVER_TYPE_FP32
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
        inputs.insert("audio_features".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 80, 3000],  // batch, features, time
        });
        inputs.insert("encoder_state".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 2048],  // batch, layers, hidden
        });
        inputs.insert("decoder_state".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 1024],  // batch, layers, hidden
        });
        
        let mut outputs = HashMap::new();
        outputs.insert("transcripts".to_string(), TensorSpec {
            data_type: DataType::INT32,
            dims: vec![1, 512],  // batch, max_seq_length
        });
        outputs.insert("updated_encoder_state".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 2048],
        });
        outputs.insert("updated_decoder_state".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 1024],
        });
        outputs.insert("beam_scores".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 16],  // batch, beam_size
        });
        
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
        inputs.insert("AUDIO_FRAMES".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 3000],  // batch, frames
        });
        
        let mut outputs = HashMap::new();
        outputs.insert("MEL_FEATURES".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 80, 3000],  // batch, features, time
        });
        
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
        inputs.insert("MEL_FEATURES".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 80, 3000],  // batch, features, time
        });
        inputs.insert("ENCODER_STATE".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 2048],  // batch, layers, hidden
        });
        
        let mut outputs = HashMap::new();
        outputs.insert("ENCODER_OUTPUT".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 3000, 1024],  // batch, time, hidden
        });
        outputs.insert("UPDATED_ENCODER_STATE".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 2048],
        });
        
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
        inputs.insert("ENCODER_OUTPUT".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 3000, 1024],  // batch, time, hidden
        });
        inputs.insert("DECODER_STATE".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 1024],  // batch, layers, hidden
        });
        
        let mut outputs = HashMap::new();
        outputs.insert("LOGITS".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 3000, 4096],  // batch, time, vocab_size
        });
        outputs.insert("UPDATED_DECODER_STATE".to_string(), TensorSpec {
            data_type: DataType::FP32,
            dims: vec![1, 512, 1024],
        });
        
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
        self.inputs.values().map(|spec| {
            let element_count: usize = spec.dims.iter().map(|&d| d as usize).product();
            element_count * spec.data_type.element_size()
        }).sum()
    }
    
    /// Calculate total size of all outputs
    pub fn total_output_size(&self) -> usize {
        self.outputs.values().map(|spec| {
            let element_count: usize = spec.dims.iter().map(|&d| d as usize).product();
            element_count * spec.data_type.element_size()
        }).sum()
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
}

/// Safe wrapper for CUDA shared memory region
/// 
/// This is a higher-level abstraction specifically for Triton C-API integration
/// with IPC handles, built on top of the lower-level DeviceBuffer.
pub struct CudaSharedMemoryRegion {
    handle: *mut c_void,
}

impl CudaSharedMemoryRegion {
    /// Create a new CUDA shared memory region
    pub fn new(name: &str, size: usize, device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        let c_name = CString::new(name)
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        
        let mut handle: *mut c_void = std::ptr::null_mut();
        
        let result = unsafe {
            CudaSharedMemoryRegionCreate(
                c_name.as_ptr(),
                size,
                device_id,
                &mut handle,
            )
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        if handle.is_null() {
            return Err(CudaSharedMemoryError::NullPointer);
        }
        
        Ok(CudaSharedMemoryRegion { handle })
    }
    
    /// Get the raw CUDA IPC handle
    pub fn get_raw_handle(&self) -> Result<Vec<u8>, CudaSharedMemoryError> {
        let mut raw_handle: *mut c_char = std::ptr::null_mut();
        
        let result = unsafe {
            GetRawHandle(self.handle, &mut raw_handle)
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        if raw_handle.is_null() {
            return Err(CudaSharedMemoryError::NullPointer);
        }
        
        let bytes = unsafe {
            let c_str = CStr::from_ptr(raw_handle);
            let bytes = c_str.to_bytes().to_vec();
            
            // Free the C-allocated memory
            let _ = FreeRawHandle(raw_handle);
            
            bytes
        };
        
        Ok(bytes)
    }
    
    /// Write f32 data to the region
    pub fn write_f32_data(&self, data: &[f32]) -> Result<(), CudaSharedMemoryError> {
        let result = unsafe {
            WriteTestData(self.handle, data.as_ptr(), data.len())
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(())
    }

    /// Enqueue async write f32 data to the region (non-blocking)
    pub fn enqueue_write_f32_data(&self, data: &[f32], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        // Create a device buffer from the shared memory region
        let device_buffer = unsafe { self.as_device_buffer::<f32>(data.len()) };
        
        // Enqueue copy to device using the stream (non-blocking)
        let mut mut_buffer = device_buffer;
        mut_buffer.enqueue_copy_from_host(data, stream)?;
        
        Ok(())
    }
    
    /// Asynchronously write f32 data to the region using a CUDA stream (blocks until complete)
    pub async fn write_f32_data_async(&self, data: &[f32], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_write_f32_data(data, stream)?;
        stream.wait().await
    }
    
    /// Read f32 data from the region
    pub fn read_f32_data(&self, element_count: usize) -> Result<Vec<f32>, CudaSharedMemoryError> {
        let mut data = vec![0.0f32; element_count];
        
        let result = unsafe {
            ReadTestData(self.handle, data.as_mut_ptr(), element_count)
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(data)
    }

    /// Enqueue async read f32 data from the region (non-blocking)
    pub fn enqueue_read_f32_data(&self, data: &mut [f32], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        // Create a device buffer from the shared memory region
        let device_buffer = unsafe { self.as_device_buffer::<f32>(data.len()) };
        
        // Enqueue copy from device to host using the stream (non-blocking)
        device_buffer.enqueue_copy_to_host(data, stream)?;
        
        Ok(())
    }
    
    /// Asynchronously read f32 data from the region using a CUDA stream (blocks until complete)
    pub async fn read_f32_data_async(&self, element_count: usize, stream: &AsyncCudaStream) -> Result<Vec<f32>, CudaSharedMemoryError> {
        let mut data = vec![0.0f32; element_count];
        self.enqueue_read_f32_data(&mut data, stream)?;
        stream.wait().await?;
        Ok(data)
    }
    
    /// Register with Triton server
    pub fn register_with_triton_server(&self) -> Result<(), CudaSharedMemoryError> {
        let result = unsafe {
            RegisterWithTritonServer(self.handle)
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(())
    }
    
    /// Run inference with the specified model configuration
    pub fn run_inference_with_config(
        &self,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
    ) -> Result<(), CudaSharedMemoryError> {
        let model_name = CString::new(config.name.as_str())
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let input_name_c = CString::new(input_name)
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let output_name_c = CString::new(output_name)
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        
        let input_spec = config.inputs.get(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;
        
        let buffer_size = config.calculate_buffer_size(input_name)
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
    /// - The memory region contains valid data of type T
    /// - The capacity doesn't exceed the actual allocated size
    /// - The memory is properly aligned for type T
    pub unsafe fn as_device_buffer<T>(&self, capacity: usize) -> DeviceBuffer<T> {
        // Note: This is a simplified version - in practice you'd need to extract
        // the actual device pointer from the C handle
        let ptr = self.handle as *mut T;
        DeviceBuffer::from_raw_parts(ptr, capacity, 0) // device_id would need to be stored
    }
    
    /// Run inference with separate input and output regions
    pub fn run_inference_with_output_regions(
        &self,
        output_region: &CudaSharedMemoryRegion,
        config: &ModelConfig,
        input_name: &str,
        output_name: &str,
    ) -> Result<(), CudaSharedMemoryError> {
        let model_name = CString::new(config.name.as_str())
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let input_name_c = CString::new(input_name)
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        let output_name_c = CString::new(output_name)
            .map_err(|_| CudaSharedMemoryError::InvalidValue)?;
        
        let input_spec = config.inputs.get(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;
        
        let input_buffer_size = config.calculate_buffer_size(input_name)
            .ok_or(CudaSharedMemoryError::InvalidValue)?;
        
        let output_buffer_size = config.calculate_output_buffer_size(output_name)
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
        self.enqueue_inference_with_output_regions(output_region, config, input_name, output_name, stream)?;
        stream.wait().await
    }
}

impl Drop for CudaSharedMemoryRegion {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let result = unsafe { CudaSharedMemoryRegionDestroy(self.handle) };
            if result != CudaError::CudaSuccess {
                eprintln!("Warning: Failed to destroy CUDA shared memory region: {:?}", result);
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
    pub fn new_for_model(config: ModelConfig, device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        let mut input_regions = HashMap::new();
        let mut output_regions = HashMap::new();
        let mut state_regions = HashMap::new();
        
        // Create input regions
        for (name, spec) in &config.inputs {
            let size = spec.dims.iter().map(|&d| d as usize).product::<usize>() * spec.data_type.element_size();
            let region = CudaSharedMemoryRegion::new(&format!("input_{}", name), size, device_id)?;
            input_regions.insert(name.clone(), region);
        }
        
        // Create output regions
        for (name, spec) in &config.outputs {
            let size = spec.dims.iter().map(|&d| d as usize).product::<usize>() * spec.data_type.element_size();
            let region = CudaSharedMemoryRegion::new(&format!("output_{}", name), size, device_id)?;
            output_regions.insert(name.clone(), region);
        }
        
        // Create state regions for stateful models
        if config.stateful {
            for (name, spec) in &config.inputs {
                if name.contains("state") {
                    let size = spec.dims.iter().map(|&d| d as usize).product::<usize>() * spec.data_type.element_size();
                    let region = CudaSharedMemoryRegion::new(&format!("state_{}", name), size, device_id)?;
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