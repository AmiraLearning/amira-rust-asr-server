//! CUDA data types and model configurations
//!
//! This module provides type definitions for tensors and model configurations
//! used in CUDA inference operations.

use std::collections::HashMap;
use std::os::raw::c_int;

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
    pub(crate) fn to_c_type(&self) -> c_int {
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
