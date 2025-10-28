//! Multi-region memory pool for complex models
//!
//! This module provides a high-level memory pool abstraction that manages
//! multiple CUDA shared memory regions for input, output, and state tensors.

use crate::cuda::error::CudaSharedMemoryError;
use crate::cuda::shared_memory::CudaSharedMemoryRegion;
use crate::cuda::types::ModelConfig;
use std::collections::HashMap;

/// Calculate buffer size with checked arithmetic to prevent integer overflow
fn calculate_buffer_size_checked(
    dims: &[i64],
    element_size: usize,
) -> Result<usize, CudaSharedMemoryError> {
    // Use try_fold with checked_mul to safely compute product of dimensions
    let element_count = dims
        .iter()
        .try_fold(1usize, |acc, &dim| {
            if dim < 0 {
                return None; // Negative dimensions are invalid
            }
            acc.checked_mul(dim as usize)
        })
        .ok_or(CudaSharedMemoryError::InvalidValue)?;

    // Check for overflow when multiplying by element size
    element_count
        .checked_mul(element_size)
        .ok_or(CudaSharedMemoryError::InvalidValue)
}

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
            let size = calculate_buffer_size_checked(&spec.dims, spec.data_type.element_size())?;
            let region = CudaSharedMemoryRegion::new(&format!("input_{}", name), size, device_id)?;
            input_regions.insert(name.clone(), region);
        }

        // Create output regions
        for (name, spec) in &config.outputs {
            let size = calculate_buffer_size_checked(&spec.dims, spec.data_type.element_size())?;
            let region = CudaSharedMemoryRegion::new(&format!("output_{}", name), size, device_id)?;
            output_regions.insert(name.clone(), region);
        }

        // Create state regions for stateful models
        if config.stateful {
            for (name, spec) in &config.inputs {
                if name.contains("state") {
                    let size = calculate_buffer_size_checked(&spec.dims, spec.data_type.element_size())?;
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
