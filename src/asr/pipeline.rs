//! ASR pipeline definition and implementation.
//!
//! This module defines the ASR pipeline interface and provides
//! a concrete implementation that uses Triton Inference Server.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::asr::decoder_optimized::greedy_decode;
use crate::asr::memory::global_pools;
use crate::asr::types::{DecoderState, Transcription, Vocabulary};
use crate::error::{AppError, Result};
use crate::triton::{
    ConnectionPool, DecoderJointInput, DecoderJointModel, EncoderInput, EncoderModel,
    PreprocessorInputRef, PreprocessorModel, TritonModel,
};

/// Defines the contract for an ASR processing pipeline.
#[async_trait]
pub trait AsrPipeline: Send + Sync {
    /// Process a chunk of audio in a streaming fashion.
    ///
    /// # Arguments
    /// * `audio_bytes` - The raw audio bytes (16-bit PCM)
    /// * `state` - The current decoder state
    ///
    /// # Returns
    /// The transcription result and updated decoder state
    async fn process_stream_chunk(
        &self,
        audio_bytes: &[u8],
        state: &mut DecoderState,
    ) -> Result<Transcription>;

    /// Process a complete audio buffer in a single batch operation.
    ///
    /// # Arguments
    /// * `audio_bytes` - The raw audio bytes (16-bit PCM)
    ///
    /// # Returns
    /// The transcription result
    async fn process_batch(&self, audio_bytes: &[u8]) -> Result<Transcription>;

    /// Process a chunk of audio samples directly (zero-copy for incremental processing).
    ///
    /// # Arguments  
    /// * `audio_samples` - The normalized audio samples (f32, -1.0 to 1.0)
    /// * `state` - The current decoder state
    ///
    /// # Returns
    /// The transcription result and updated decoder state
    async fn process_stream_samples(
        &self,
        audio_samples: &[f32],
        state: &mut DecoderState,
    ) -> Result<Transcription>;

    /// Process a complete audio sample buffer in a single batch operation (zero-copy).
    ///
    /// # Arguments
    /// * `audio_samples` - The normalized audio samples (f32, -1.0 to 1.0)
    ///
    /// # Returns
    /// The transcription result
    async fn process_batch_samples(&self, audio_samples: &[f32]) -> Result<Transcription>;
}

/// Decoding algorithm selection.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum DecodingAlgorithm {
    /// Greedy decoding (faster, less accurate)
    Greedy,
}

impl Default for DecodingAlgorithm {
    fn default() -> Self {
        Self::Greedy
    }
}

/// ASR pipeline implementation using Triton Inference Server.
pub struct TritonAsrPipeline {
    /// Connection pool for Triton clients
    connection_pool: Arc<ConnectionPool>,

    /// Vocabulary for token decoding
    vocabulary: Arc<Vocabulary>,

    /// Preprocessor model
    preprocessor: PreprocessorModel,

    /// Encoder model
    encoder: EncoderModel,

    /// Decoder and joint network model
    decoder_joint: DecoderJointModel,
}

impl TritonAsrPipeline {
    /// Create a new ASR pipeline with a connection pool.
    ///
    /// # Arguments
    /// * `connection_pool` - The Triton connection pool
    /// * `vocabulary` - The vocabulary for token decoding
    ///
    /// # Returns
    /// A new ASR pipeline with default greedy decoding
    pub fn new(connection_pool: Arc<ConnectionPool>, vocabulary: Arc<Vocabulary>) -> Self {
        Self {
            connection_pool,
            vocabulary,
            preprocessor: PreprocessorModel,
            encoder: EncoderModel,
            decoder_joint: DecoderJointModel,
        }
    }

    /// Convert audio bytes to normalized samples using optimized SIMD conversion.
    ///
    /// # Arguments
    /// * `audio_bytes` - The raw audio bytes (16-bit PCM)
    ///
    /// # Returns
    /// The normalized audio samples
    fn convert_audio(&self, audio_bytes: &[u8]) -> Result<Vec<f32>> {
        // Use lock-free memory pool for audio buffer
        let mut audio_buffer = global_pools().audio_buffers.get();
        audio_buffer.clear();

        // Use optimized conversion from performance_opts
        crate::performance_opts::audio::bytes_to_f32_optimized(audio_bytes, &mut audio_buffer);

        // Take ownership to return from pool
        audio_buffer
            .take()
            .map_err(|e| AppError::Internal(format!("Memory pool error: {}", e)))
    }

    /// Process audio with the full ASR pipeline.
    ///
    /// # Performance Notes
    ///
    /// This method is optimized for the inference hot path:
    /// - Reuses connection across all model inferences
    /// - Minimizes cloning of large state vectors
    /// - Only allocates Vec when required by model input APIs
    ///
    /// # Arguments
    /// * `waveform` - The audio waveform (f32 samples)
    /// * `initial_state` - The initial decoder state (taken by value to avoid cloning)
    ///
    /// # Returns
    /// The transcription result and updated decoder state
    #[allow(dead_code)]
    async fn process_audio_internal(
        &self,
        waveform: &[f32],
        initial_state: DecoderState,
    ) -> Result<(Transcription, DecoderState)> {
        info!("Starting ASR pipeline for {} samples", waveform.len());

        // Step 1: Preprocess audio to features
        // Get connection from pool and reuse throughout the pipeline
        let mut pooled_connection = self.connection_pool.get().await?;
        let preprocessor_input = PreprocessorInputRef { waveform };

        debug!("Calling preprocessor...");
        let preprocessor_output = {
            let mut connection = pooled_connection.client_mut().client_mut().await;
            self.preprocessor
                .infer_zero_copy(&mut *connection, preprocessor_input)
                .await?
        };
        info!(
            "Preprocessor complete: features_len={}",
            preprocessor_output.features_len
        );

        // Step 2: Encode features
        let encoder_input = EncoderInput {
            features: preprocessor_output.features,
            features_len: preprocessor_output.features_len,
        };

        debug!("Calling encoder...");
        let encoder_output = {
            let mut connection = pooled_connection.client_mut().client_mut().await;
            self.encoder.infer(&mut *connection, encoder_input).await?
        };
        info!(
            "Encoder complete: encoded_len={}",
            encoder_output.encoded_len
        );

        // Step 3: RNN-T Decoding
        let decoding_algorithm_name = "greedy";

        info!(
            "Starting RNN-T decoding with {} encoder frames using {}",
            encoder_output.encoded_len, decoding_algorithm_name
        );
        let decoder_state = initial_state;
        let decoder_joint_ref = &self.decoder_joint;

        // Create the decode step function that will be called by the decoder
        let client_for_decode = pooled_connection.client_clone();
        let decode_step = |encoder_frame: &[f32], targets: &[i32], state: DecoderState| {
            // Clone the client for this specific decode step
            let client = client_for_decode.clone();
            let decoder = decoder_joint_ref;

            // For now, we still need to clone for the async closure, but we avoid the double clone
            // This is still better than the original implementation
            let input = DecoderJointInput {
                encoder_frame: encoder_frame.to_vec(),
                targets: targets.to_vec(),
                states_1: state.states_1.clone(),
                states_2: state.states_2.clone(),
            };

            async move {
                let output = {
                    let mut client_guard = client.client_mut().await;
                    decoder.infer(&mut *client_guard, input).await?
                };

                let new_state = DecoderState {
                    states_1: output.new_states_1,
                    states_2: output.new_states_2,
                };

                Ok((output.logits, new_state))
            }
        };

        // Use greedy decoding
        let (tokens, final_state) = greedy_decode(
            &encoder_output.outputs,
            encoder_output.encoded_len,
            decoder_state,
            decode_step,
        )
        .await?;

        info!("RNN-T decoding complete: {} tokens generated", tokens.len());
        if !tokens.is_empty() {
            debug!("Generated tokens: {:?}", tokens);
        }

        // Step 4: Convert tokens to text
        let text = self.vocabulary.decode_tokens(&tokens);
        info!("Decoded text: '{}'", text);

        let transcription = Transcription {
            text,
            tokens,
            audio_length_samples: waveform.len(),
            features_length: preprocessor_output.features_len,
            encoded_length: encoder_output.encoded_len,
        };

        Ok((transcription, final_state))
    }

    /// Zero-copy optimized version that minimizes allocations in the hot path.
    /// Uses pre-allocated buffers and slice references instead of Vec allocations.
    async fn process_audio_zero_copy(
        &self,
        waveform: &[f32],
        initial_state: DecoderState,
    ) -> Result<(Transcription, DecoderState)> {
        info!(
            "Starting zero-copy ASR pipeline for {} samples",
            waveform.len()
        );

        // Get connection from pool and reuse throughout the pipeline
        let mut pooled_connection = self.connection_pool.get().await?;

        // Step 1: Preprocess audio to features using zero-copy
        let preprocessor_input = PreprocessorInputRef { waveform };

        debug!("Calling preprocessor (zero-copy)...");
        let preprocessor_output = {
            let mut connection = pooled_connection.client_mut().client_mut().await;
            self.preprocessor
                .infer_zero_copy(&mut *connection, preprocessor_input)
                .await?
        };
        info!(
            "Preprocessor complete: features_len={}",
            preprocessor_output.features_len
        );

        // Step 2: Encode features (still needs to own the data from preprocessor)
        let encoder_input = EncoderInput {
            features: preprocessor_output.features,
            features_len: preprocessor_output.features_len,
        };

        debug!("Calling encoder...");
        let encoder_output = {
            let mut connection = pooled_connection.client_mut().client_mut().await;
            self.encoder.infer(&mut *connection, encoder_input).await?
        };
        info!(
            "Encoder complete: encoded_len={}",
            encoder_output.encoded_len
        );

        // Step 3: RNN-T Decoding with zero-copy decode steps
        info!(
            "Starting zero-copy RNN-T decoding with {} encoder frames",
            encoder_output.encoded_len
        );
        let decoder_state = initial_state;
        let decoder_joint_ref = &self.decoder_joint;

        // Create the zero-copy decode step function
        let client_for_decode = pooled_connection.client_clone();
        let decode_step = |encoder_frame: &[f32], targets: &[i32], state: DecoderState| {
            let client = client_for_decode.clone();
            let decoder = decoder_joint_ref;

            // Still need to clone for async closure but this is more optimal than original
            let input = DecoderJointInput {
                encoder_frame: encoder_frame.to_vec(),
                targets: targets.to_vec(),
                states_1: state.states_1,
                states_2: state.states_2,
            };

            async move {
                let output = {
                    let mut client_guard = client.client_mut().await;
                    decoder.infer(&mut *client_guard, input).await?
                };

                let new_state = DecoderState {
                    states_1: output.new_states_1,
                    states_2: output.new_states_2,
                };

                Ok((output.logits, new_state))
            }
        };

        let (tokens, final_state) = greedy_decode(
            &encoder_output.outputs,
            encoder_output.encoded_len,
            decoder_state,
            decode_step,
        )
        .await?;

        info!(
            "Zero-copy RNN-T decoding complete: {} tokens generated",
            tokens.len()
        );
        if !tokens.is_empty() {
            debug!("Generated tokens: {:?}", tokens);
        }

        // Step 4: Convert tokens to text
        let text = self.vocabulary.decode_tokens(&tokens);
        info!("Decoded text: '{}'", text);

        let transcription = Transcription {
            text,
            tokens,
            audio_length_samples: waveform.len(),
            features_length: preprocessor_output.features_len,
            encoded_length: encoder_output.encoded_len,
        };

        Ok((transcription, final_state))
    }
}

#[async_trait]
impl AsrPipeline for TritonAsrPipeline {
    async fn process_stream_chunk(
        &self,
        audio_bytes: &[u8],
        state: &mut DecoderState,
    ) -> Result<Transcription> {
        let waveform = self.convert_audio(audio_bytes)?;

        // Take ownership of current state to avoid cloning
        let current_state = std::mem::replace(state, DecoderState::new());
        let (transcription, new_state) = self
            .process_audio_zero_copy(&waveform, current_state)
            .await?;

        // Update the state for the next chunk
        *state = new_state;

        Ok(transcription)
    }

    async fn process_batch(&self, audio_bytes: &[u8]) -> Result<Transcription> {
        let waveform = self.convert_audio(audio_bytes)?;

        // For batch processing, we always start with a fresh state
        let initial_state = DecoderState::new();

        let (transcription, _) = self
            .process_audio_zero_copy(&waveform, initial_state)
            .await?;

        Ok(transcription)
    }

    async fn process_stream_samples(
        &self,
        audio_samples: &[f32],
        state: &mut DecoderState,
    ) -> Result<Transcription> {
        // Take ownership of current state to avoid cloning
        let current_state = std::mem::replace(state, DecoderState::new());
        let (transcription, new_state) = self
            .process_audio_zero_copy(audio_samples, current_state)
            .await?;

        // Update the state for the next chunk
        *state = new_state;

        Ok(transcription)
    }

    async fn process_batch_samples(&self, audio_samples: &[f32]) -> Result<Transcription> {
        // For batch processing, we always start with a fresh state
        let initial_state = DecoderState::new();

        let (transcription, _) = self
            .process_audio_zero_copy(audio_samples, initial_state)
            .await?;

        Ok(transcription)
    }
}
