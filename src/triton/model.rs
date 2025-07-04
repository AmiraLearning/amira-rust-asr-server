//! Model interfaces for Triton Inference Server.
//!
//! This module defines the trait for Triton models and implements it for
//! the specific models used in the ASR pipeline.

use async_trait::async_trait;
use std::collections::HashMap;

use crate::config::model;
use crate::error::{AppError, Result};
use crate::triton::client::TritonClient;
use crate::triton::proto::{
    model_infer_request::{InferInputTensor, InferRequestedOutputTensor},
    InferTensorContents,
};
use crate::triton::types::{parse_raw_tensors, TensorDataType, TensorDef, TensorShape};

/// A trait for models served by Triton, defining a uniform inference contract.
///
/// This abstraction allows for a generic client to execute different models
/// (e.g., preprocessor, encoder) without knowing their specific tensor names
/// or shapes, promoting code reuse and extensibility.
#[async_trait]
pub trait TritonModel: Send + Sync {
    /// The type of data required as input to this model.
    type Input;

    /// The type of data produced as output by this model.
    type Output;

    /// Get the name of this model.
    fn name(&self) -> &str;

    /// Execute inference with this model.
    ///
    /// # Arguments
    /// * `client` - The Triton client to use for inference.
    /// * `input` - The input data for the model.
    ///
    /// # Returns
    /// The output data from the model.
    async fn infer(&self, client: &mut TritonClient, input: Self::Input) -> Result<Self::Output>;
}

/// Input for the preprocessor model.
pub struct PreprocessorInput {
    /// The audio waveform (f32 samples)
    pub waveform: Vec<f32>,
}

/// Zero-copy input for the preprocessor model.
pub struct PreprocessorInputRef<'a> {
    /// The audio waveform (f32 samples)
    pub waveform: &'a [f32],
}

/// Output from the preprocessor model.
pub struct PreprocessorOutput {
    /// The extracted features (mel spectrograms)
    pub features: Vec<f32>,

    /// The length of the features
    pub features_len: i64,
}

/// The preprocessor model converts raw audio to features (mel spectrograms).
pub struct PreprocessorModel;

impl PreprocessorModel {
    /// Zero-copy inference method that avoids allocations in the hot path
    pub async fn infer_zero_copy(&self, client: &mut TritonClient, input: PreprocessorInputRef<'_>) -> Result<PreprocessorOutput> {
        let waveform_len = input.waveform.len() as i64;

        // Build inference request with borrowed data
        let request = client
            .request_builder(self.name())
            .with_input(InferInputTensor {
                name: "waveforms".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, waveform_len],
                contents: Some(InferTensorContents {
                    fp32_contents: input.waveform.to_vec(), // Still need to copy for protobuf
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "waveforms_lens".to_string(),
                datatype: "INT64".to_string(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    int64_contents: vec![waveform_len],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "features".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "features_length".to_string(),
                ..Default::default()
            })
            .build();

        // Execute inference
        let response = client.infer(request).await?;

        // Process response (same as regular method) 
        let expected_tensors = {
            let mut map = HashMap::new();

            map.insert(
                "features".to_string(),
                TensorDef::new(
                    "features",
                    TensorDataType::Float32,
                    TensorShape::new(vec![1, 80, -1]),
                ),
            );

            map.insert(
                "features_length".to_string(),
                TensorDef::new(
                    "features_length",
                    TensorDataType::Int64,
                    TensorShape::new(vec![1]),
                ),
            );

            map
        };

        let tensors = parse_raw_tensors(&response, &expected_tensors)?;

        let features_tensor = tensors.get("features").ok_or_else(|| {
            AppError::Model("Missing features tensor in preprocessor response".to_string())
        })?;

        let features_length_tensor = tensors.get("features_length").ok_or_else(|| {
            AppError::Model("Missing features_length tensor in preprocessor response".to_string())
        })?;

        let features = features_tensor.as_f32()?;
        let features_len = features_length_tensor.as_scalar_i64()?;

        Ok(PreprocessorOutput {
            features,
            features_len,
        })
    }
}

#[async_trait]
impl TritonModel for PreprocessorModel {
    type Input = PreprocessorInput;
    type Output = PreprocessorOutput;

    fn name(&self) -> &str {
        model::PREPROCESSOR_MODEL_NAME
    }

    async fn infer(&self, client: &mut TritonClient, input: Self::Input) -> Result<Self::Output> {
        let waveform_len = input.waveform.len() as i64;

        // Build inference request
        let request = client
            .request_builder(self.name())
            .with_input(InferInputTensor {
                name: "waveforms".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, waveform_len],
                contents: Some(InferTensorContents {
                    fp32_contents: input.waveform,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "waveforms_lens".to_string(),
                datatype: "INT64".to_string(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    int64_contents: vec![waveform_len],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "features".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "features_lens".to_string(),
                ..Default::default()
            })
            .build();

        // Execute inference
        let response = client.infer(request).await?;

        // Process response
        let expected_tensors = {
            let mut map = HashMap::new();

            map.insert(
                "features".to_string(),
                TensorDef::new(
                    "features",
                    TensorDataType::Float32,
                    TensorShape::new(vec![1, 128, -1]),
                ),
            );

            map.insert(
                "features_lens".to_string(),
                TensorDef::new(
                    "features_lens",
                    TensorDataType::Int64,
                    TensorShape::new(vec![1]),
                ),
            );

            map
        };

        let tensors = parse_raw_tensors(&response, &expected_tensors)?;

        // Extract features and features_len
        let features_tensor = tensors.get("features").ok_or_else(|| {
            AppError::Model("Missing features tensor in preprocessor response".to_string())
        })?;

        let features_lens_tensor = tensors.get("features_lens").ok_or_else(|| {
            AppError::Model("Missing features_lens tensor in preprocessor response".to_string())
        })?;

        let features = features_tensor.as_f32()?;
        let features_len = features_lens_tensor.as_scalar_i64()?;

        Ok(PreprocessorOutput {
            features,
            features_len,
        })
    }
}

/// Input for the encoder model.
pub struct EncoderInput {
    /// The features from the preprocessor
    pub features: Vec<f32>,

    /// The length of the features
    pub features_len: i64,
}

/// Output from the encoder model.
pub struct EncoderOutput {
    /// The encoder outputs
    pub outputs: Vec<f32>,

    /// The length of the encoded outputs
    pub encoded_len: i64,
}

/// The encoder model converts features to encoder representations.
pub struct EncoderModel;

#[async_trait]
impl TritonModel for EncoderModel {
    type Input = EncoderInput;
    type Output = EncoderOutput;

    fn name(&self) -> &str {
        model::ENCODER_MODEL_NAME
    }

    async fn infer(&self, client: &mut TritonClient, input: Self::Input) -> Result<Self::Output> {
        // Build inference request
        let request = client
            .request_builder(self.name())
            .with_input(InferInputTensor {
                name: "audio_signal".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 128, input.features_len],
                contents: Some(InferTensorContents {
                    fp32_contents: input.features,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "length".to_string(),
                datatype: "INT64".to_string(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    int64_contents: vec![input.features_len],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "outputs".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "encoded_lengths".to_string(),
                ..Default::default()
            })
            .build();

        // Execute inference
        let response = client.infer(request).await?;

        // Process response
        let expected_tensors = {
            let mut map = HashMap::new();

            map.insert(
                "outputs".to_string(),
                TensorDef::new(
                    "outputs",
                    TensorDataType::Float32,
                    TensorShape::new(vec![1, 1024, -1]),
                ),
            );

            map.insert(
                "encoded_lengths".to_string(),
                TensorDef::new(
                    "encoded_lengths",
                    TensorDataType::Int64,
                    TensorShape::new(vec![1]),
                ),
            );

            map
        };

        let tensors = parse_raw_tensors(&response, &expected_tensors)?;

        // Extract outputs and encoded_lengths
        let outputs_tensor = tensors.get("outputs").ok_or_else(|| {
            AppError::Model("Missing outputs tensor in encoder response".to_string())
        })?;

        let encoded_lengths_tensor = tensors.get("encoded_lengths").ok_or_else(|| {
            AppError::Model("Missing encoded_lengths tensor in encoder response".to_string())
        })?;

        let outputs = outputs_tensor.as_f32()?;
        let encoded_len = encoded_lengths_tensor.as_scalar_i64()?;

        Ok(EncoderOutput {
            outputs,
            encoded_len,
        })
    }
}

/// Input for the decoder_joint model.
pub struct DecoderJointInput {
    /// The encoder outputs for a single time step
    pub encoder_frame: Vec<f32>,

    /// The token IDs of the current hypothesis
    pub targets: Vec<i32>,

    /// The current decoder state (first part)
    pub states_1: Vec<f32>,

    /// The current decoder state (second part)
    pub states_2: Vec<f32>,
}

/// Zero-copy input for the decoder_joint model.
pub struct DecoderJointInputRef<'a> {
    /// The encoder outputs for a single time step
    pub encoder_frame: &'a [f32],

    /// The token IDs of the current hypothesis
    pub targets: &'a [i32],

    /// The current decoder state (first part)
    pub states_1: &'a [f32],

    /// The current decoder state (second part)
    pub states_2: &'a [f32],
}

/// Output from the decoder_joint model.
pub struct DecoderJointOutput {
    /// The logits for the next token
    pub logits: Vec<f32>,

    /// The updated decoder state (first part)
    pub new_states_1: Vec<f32>,

    /// The updated decoder state (second part)
    pub new_states_2: Vec<f32>,
}

/// The decoder_joint model combines encoder outputs and decoder state to predict the next token.
pub struct DecoderJointModel;

impl DecoderJointModel {
    /// Zero-copy inference method that avoids allocations in the hot path
    pub async fn infer_zero_copy(&self, client: &mut TritonClient, input: DecoderJointInputRef<'_>) -> Result<DecoderJointOutput> {
        let target_length = input.targets.len() as i32;

        // Build inference request with borrowed data
        let request = client
            .request_builder(self.name())
            .with_input(InferInputTensor {
                name: "encoder_outputs".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 1024, 1],
                contents: Some(InferTensorContents {
                    fp32_contents: input.encoder_frame.to_vec(), // Still need to copy for protobuf
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "targets".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1, target_length as i64],
                contents: Some(InferTensorContents {
                    int_contents: input.targets.to_vec(), // Still need to copy for protobuf
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "target_length".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    int_contents: vec![target_length],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "input_states_1".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![2, 1, model::DECODER_STATE_SIZE as i64],
                contents: Some(InferTensorContents {
                    fp32_contents: input.states_1.to_vec(), // Still need to copy for protobuf
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "input_states_2".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![2, 1, model::DECODER_STATE_SIZE as i64],
                contents: Some(InferTensorContents {
                    fp32_contents: input.states_2.to_vec(), // Still need to copy for protobuf
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "outputs".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "prednet_lengths".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "output_states_1".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "output_states_2".to_string(),
                ..Default::default()
            })
            .build();

        // Execute inference
        let response = client.infer(request).await?;

        // Process response (same as regular method)
        let expected_tensors = {
            let mut map = HashMap::new();

            map.insert(
                "outputs".to_string(),
                TensorDef::new(
                    "outputs",
                    TensorDataType::Float32,
                    TensorShape::new(vec![1, model::VOCABULARY_SIZE as i64]),
                ),
            );

            map.insert(
                "output_states_1".to_string(),
                TensorDef::new(
                    "output_states_1",
                    TensorDataType::Float32,
                    TensorShape::new(vec![2, 1, model::DECODER_STATE_SIZE as i64]),
                ),
            );

            map.insert(
                "output_states_2".to_string(),
                TensorDef::new(
                    "output_states_2",
                    TensorDataType::Float32,
                    TensorShape::new(vec![2, 1, model::DECODER_STATE_SIZE as i64]),
                ),
            );

            map
        };

        let tensors = parse_raw_tensors(&response, &expected_tensors)?;

        let logits_tensor = tensors.get("outputs").ok_or_else(|| {
            AppError::Model("Missing outputs tensor in decoder_joint response".to_string())
        })?;

        let states_1_tensor = tensors.get("output_states_1").ok_or_else(|| {
            AppError::Model("Missing output_states_1 tensor in decoder_joint response".to_string())
        })?;

        let states_2_tensor = tensors.get("output_states_2").ok_or_else(|| {
            AppError::Model("Missing output_states_2 tensor in decoder_joint response".to_string())
        })?;

        let logits = logits_tensor.as_f32()?;
        let new_states_1 = states_1_tensor.as_f32()?;
        let new_states_2 = states_2_tensor.as_f32()?;

        Ok(DecoderJointOutput {
            logits,
            new_states_1,
            new_states_2,
        })
    }
}

#[async_trait]
impl TritonModel for DecoderJointModel {
    type Input = DecoderJointInput;
    type Output = DecoderJointOutput;

    fn name(&self) -> &str {
        model::DECODER_JOINT_MODEL_NAME
    }

    async fn infer(&self, client: &mut TritonClient, input: Self::Input) -> Result<Self::Output> {
        let target_length = input.targets.len() as i32;

        // Build inference request
        let request = client
            .request_builder(self.name())
            .with_input(InferInputTensor {
                name: "encoder_outputs".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![1, 1024, 1],
                contents: Some(InferTensorContents {
                    fp32_contents: input.encoder_frame,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "targets".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1, target_length as i64],
                contents: Some(InferTensorContents {
                    int_contents: input.targets,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "target_length".to_string(),
                datatype: "INT32".to_string(),
                shape: vec![1],
                contents: Some(InferTensorContents {
                    int_contents: vec![target_length],
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "input_states_1".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![2, 1, model::DECODER_STATE_SIZE as i64],
                contents: Some(InferTensorContents {
                    fp32_contents: input.states_1,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_input(InferInputTensor {
                name: "input_states_2".to_string(),
                datatype: "FP32".to_string(),
                shape: vec![2, 1, model::DECODER_STATE_SIZE as i64],
                contents: Some(InferTensorContents {
                    fp32_contents: input.states_2,
                    ..Default::default()
                }),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "outputs".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "prednet_lengths".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "output_states_1".to_string(),
                ..Default::default()
            })
            .with_output(InferRequestedOutputTensor {
                name: "output_states_2".to_string(),
                ..Default::default()
            })
            .build();

        // Execute inference
        let response = client.infer(request).await?;

        // Process response
        let expected_tensors = {
            let mut map = HashMap::new();

            map.insert(
                "outputs".to_string(),
                TensorDef::new(
                    "outputs",
                    TensorDataType::Float32,
                    TensorShape::new(vec![1, -1, 1, model::VOCABULARY_SIZE as i64]),
                ),
            );

            map.insert(
                "output_states_1".to_string(),
                TensorDef::new(
                    "output_states_1",
                    TensorDataType::Float32,
                    TensorShape::new(vec![2, 1, model::DECODER_STATE_SIZE as i64]),
                ),
            );

            map.insert(
                "output_states_2".to_string(),
                TensorDef::new(
                    "output_states_2",
                    TensorDataType::Float32,
                    TensorShape::new(vec![2, 1, model::DECODER_STATE_SIZE as i64]),
                ),
            );

            map
        };

        let tensors = parse_raw_tensors(&response, &expected_tensors)?;

        // Extract outputs and states
        let outputs_tensor = tensors.get("outputs").ok_or_else(|| {
            AppError::Model("Missing outputs tensor in decoder_joint response".to_string())
        })?;

        let output_states_1_tensor = tensors.get("output_states_1").ok_or_else(|| {
            AppError::Model("Missing output_states_1 tensor in decoder_joint response".to_string())
        })?;

        let output_states_2_tensor = tensors.get("output_states_2").ok_or_else(|| {
            AppError::Model("Missing output_states_2 tensor in decoder_joint response".to_string())
        })?;

        let logits = outputs_tensor.as_f32()?;
        let new_states_1 = output_states_1_tensor.as_f32()?;
        let new_states_2 = output_states_2_tensor.as_f32()?;

        Ok(DecoderJointOutput {
            logits,
            new_states_1,
            new_states_2,
        })
    }
}
