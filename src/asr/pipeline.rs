//! ASR pipeline definition and implementation.
//!
//! This module defines the ASR pipeline interface and provides
//! a concrete implementation that uses Triton Inference Server.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, info};

use crate::asr::audio::bytes_to_f32_samples;
use crate::asr::decoder::greedy_decode;
use crate::asr::types::{DecoderState, Transcription, Vocabulary};
use crate::error::Result;
use crate::triton::{
    DecoderJointInput, DecoderJointModel, EncoderInput, EncoderModel, PreprocessorInput,
    PreprocessorModel, TritonClient, TritonModel,
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
}

/// ASR pipeline implementation using Triton Inference Server.
pub struct TritonAsrPipeline {
    /// Client for communicating with Triton
    client: TritonClient,

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
    /// Create a new ASR pipeline.
    ///
    /// # Arguments
    /// * `client` - The Triton client
    /// * `vocabulary` - The vocabulary for token decoding
    ///
    /// # Returns
    /// A new ASR pipeline
    pub fn new(client: TritonClient, vocabulary: Arc<Vocabulary>) -> Self {
        Self {
            client,
            vocabulary,
            preprocessor: PreprocessorModel,
            encoder: EncoderModel,
            decoder_joint: DecoderJointModel,
        }
    }

    /// Convert audio bytes to normalized samples.
    ///
    /// # Arguments
    /// * `audio_bytes` - The raw audio bytes (16-bit PCM)
    ///
    /// # Returns
    /// The normalized audio samples
    fn convert_audio(&self, audio_bytes: &[u8]) -> Vec<f32> {
        bytes_to_f32_samples(audio_bytes)
    }

    /// Process audio with the full ASR pipeline.
    ///
    /// # Arguments
    /// * `waveform` - The audio waveform (f32 samples)
    /// * `initial_state` - The initial decoder state
    ///
    /// # Returns
    /// The transcription result and updated decoder state
    async fn process_audio_internal(
        &self,
        waveform: &[f32],
        initial_state: DecoderState,
    ) -> Result<(Transcription, DecoderState)> {
        info!("Starting ASR pipeline for {} samples", waveform.len());

        // Step 1: Preprocess audio to features
        let mut client = self.client.clone();
        let preprocessor_input = PreprocessorInput {
            waveform: waveform.to_vec(),
        };

        debug!("Calling preprocessor...");
        let preprocessor_output = self
            .preprocessor
            .infer(&mut client, preprocessor_input)
            .await?;
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
        let encoder_output = self.encoder.infer(&mut client, encoder_input).await?;
        info!(
            "Encoder complete: encoded_len={}",
            encoder_output.encoded_len
        );

        // Step 3: RNN-T Decoding
        info!(
            "Starting RNN-T decoding with {} encoder frames",
            encoder_output.encoded_len
        );
        let decoder_state = initial_state;
        let decoder_joint_ref = &self.decoder_joint;

        // Use a struct to hold the state for the decode function
        struct DecodeState<'a> {
            client: TritonClient,
            decoder: &'a DecoderJointModel,
        }

        let decode_state = DecodeState {
            client: client.clone(),
            decoder: decoder_joint_ref,
        };

        // Create the decode step function that will be called by greedy_decode
        let decode_step = |encoder_frame: &[f32], targets: &[i32], state: DecoderState| {
            // Clone what we need for the async block
            let mut client = decode_state.client.clone();
            let decoder = decode_state.decoder;
            let encoder_frame = encoder_frame.to_vec();
            let targets = targets.to_vec();

            async move {
                let input = DecoderJointInput {
                    encoder_frame,
                    targets,
                    states_1: state.states_1,
                    states_2: state.states_2,
                };

                let output = decoder.infer(&mut client, input).await?;

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
}

#[async_trait]
impl AsrPipeline for TritonAsrPipeline {
    async fn process_stream_chunk(
        &self,
        audio_bytes: &[u8],
        state: &mut DecoderState,
    ) -> Result<Transcription> {
        let waveform = self.convert_audio(audio_bytes);

        let (transcription, new_state) = self
            .process_audio_internal(&waveform, state.clone())
            .await?;

        // Update the state for the next chunk
        *state = new_state;

        Ok(transcription)
    }

    async fn process_batch(&self, audio_bytes: &[u8]) -> Result<Transcription> {
        let waveform = self.convert_audio(audio_bytes);

        // For batch processing, we always start with a fresh state
        let initial_state = DecoderState::new();

        let (transcription, _) = self
            .process_audio_internal(&waveform, initial_state)
            .await?;

        Ok(transcription)
    }
}
