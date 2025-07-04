//! Incremental ASR processing.
//!
//! This module provides the incremental processing functionality for ASR,
//! including overlapping audio chunks, logit accumulation, and transcript weaving.

use std::sync::Arc;

use crate::asr::audio::{audio_len, bytes_to_f32_samples, OverlappingAudioBuffer};
use crate::asr::types::{
    AccumulatedPredictions, DecoderState, SeqSlice, Transcription, Vocabulary, W2V_SAMPLE_RATE,
};
use crate::asr::weaving::{is_overlap_silence, weave_transcript_segs};
use crate::asr::{AsrPipeline, TritonAsrPipeline};
use crate::error::Result;

// use tracing::{debug, info};  // Temporarily disabled
macro_rules! debug { ($($tt:tt)*) => {}; }
macro_rules! info { ($($tt:tt)*) => {}; }

/// Default minimum alignment score for transcript weaving
const MIN_ALIGNMENT_SCORE: f32 = 0.01;

/// Function for converting logit indices to time
fn logit_index_to_time(idx: usize) -> f32 {
    idx as f32 * 6.0 / 299.0
}

/// Function for converting sample indices to logit indices
fn sample_index_to_logit_index(idx: usize) -> usize {
    ((idx as f32 * 299.0) / 96000.0) as usize
}

// Removed f32_samples_to_bytes - no longer needed with zero-copy processing

/// Incremental ASR processor for streaming audio.
///
/// This struct manages the state for incremental ASR processing,
/// including overlapping audio chunks and transcript accumulation.
pub struct IncrementalAsr {
    /// The ASR pipeline
    pipeline: Arc<TritonAsrPipeline>,

    /// Vocabulary for decoding
    _vocabulary: Arc<Vocabulary>,

    /// Audio buffer with overlap support
    audio_buffer: OverlappingAudioBuffer,

    /// Accumulated predictions
    accumulated: AccumulatedPredictions,

    /// Decoder state
    decoder_state: DecoderState,

    /// Chunk size in seconds
    chunk_size: f32,

    /// Leading context in seconds
    _leading_context: f32,

    /// Trailing context in seconds
    _trailing_context: f32,
}

impl IncrementalAsr {
    /// Create a new incremental ASR processor.
    ///
    /// # Arguments
    /// * `pipeline` - The ASR pipeline
    /// * `vocabulary` - Vocabulary for decoding
    /// * `chunk_size` - Size of processing chunks in seconds
    /// * `leading_context` - Leading context size in seconds
    /// * `trailing_context` - Trailing context size in seconds
    /// * `buffer_capacity` - Audio buffer capacity in seconds
    pub fn new(
        pipeline: Arc<TritonAsrPipeline>,
        vocabulary: Arc<Vocabulary>,
        chunk_size: f32,
        leading_context: f32,
        trailing_context: f32,
        buffer_capacity: f32,
    ) -> Self {
        let capacity = (buffer_capacity * W2V_SAMPLE_RATE as f32) as usize;

        Self {
            pipeline,
            _vocabulary: vocabulary,
            audio_buffer: OverlappingAudioBuffer::new(
                capacity,
                chunk_size,
                leading_context,
                trailing_context,
            ),
            accumulated: AccumulatedPredictions::new(),
            decoder_state: DecoderState::new(),
            chunk_size,
            _leading_context: leading_context,
            _trailing_context: trailing_context,
        }
    }

    /// Clear the state to start a new transcription.
    pub fn clear(&mut self) {
        self.audio_buffer.clear();
        self.accumulated.clear();
        self.decoder_state = DecoderState::new();
    }

    /// Process a chunk of audio bytes.
    ///
    /// # Arguments
    /// * `audio_bytes` - Raw audio bytes (16-bit PCM)
    ///
    /// # Returns
    /// The current transcription
    pub async fn process_chunk(&mut self, audio_bytes: &[u8]) -> Result<String> {
        // Convert to f32 samples
        let samples = bytes_to_f32_samples(audio_bytes);

        // Add to buffer
        self.audio_buffer.add_samples(&samples);

        // Update mean amplitude from the buffer
        self.accumulated.mean_amplitude = self.audio_buffer.mean_amplitude();

        // Process if we have enough data
        if !self.audio_buffer.is_empty() {
            self.process_buffered_audio().await?;
        }

        Ok(self.accumulated.transcript.clone())
    }

    /// Process the current audio buffer.
    ///
    /// # Returns
    /// Result indicating success or failure
    async fn process_buffered_audio(&mut self) -> Result<()> {
        let audio_window = self.audio_buffer.get_window();

        // Use zero-copy processing with f32 samples directly (no conversion to bytes)
        if self.accumulated.token_ids.is_empty() {
            let transcription = self
                .pipeline
                .process_stream_samples(audio_window, &mut self.decoder_state)
                .await?;

            self.accumulated.token_ids = transcription.tokens.clone();
            self.accumulated.transcript = transcription.text;

            return Ok(());
        }

        // Collect all overlapping windows first to avoid borrowing issues
        let windows: Vec<_> = self.audio_buffer.overlapping_windows().collect();

        // Process each window with zero-copy f32 processing
        for (source_slice, target_slice, overlap) in windows {
            let chunk = self.audio_buffer.get_slice(&source_slice);

            // Use zero-copy processing with f32 samples directly (eliminates conversion overhead)
            let transcription = self
                .pipeline
                .process_stream_samples(chunk, &mut self.decoder_state)
                .await?;

            // Accumulate the transcription
            self.accumulate_transcription(&transcription, &target_slice, overlap)
                .await?;
        }

        Ok(())
    }

    /// Accumulate a new transcription into the existing state.
    ///
    /// # Arguments
    /// * `transcription` - The new transcription
    /// * `target_slice` - The target slice for this transcription
    /// * `overlap` - The overlap ratio
    ///
    /// # Returns
    /// Result indicating success or failure
    async fn accumulate_transcription(
        &mut self,
        transcription: &Transcription,
        target_slice: &SeqSlice,
        overlap: f32,
    ) -> Result<()> {
        let segment_transcript = transcription.text.clone();
        info!("Segment transcript: {}", segment_transcript);

        // If first segment, just set it
        if self.accumulated.transcript.is_empty() {
            self.accumulated.transcript = segment_transcript;
            self.accumulated.token_ids = transcription.tokens.clone();
            return Ok(());
        }

        // Determine if the overlap is silence
        let chunk_size = (overlap * self.chunk_size * W2V_SAMPLE_RATE as f32) as usize;
        let is_silence = if chunk_size > 0 {
            let audio_window = self.audio_buffer.get_window();
            let overlap_start = audio_window.len().saturating_sub(chunk_size);
            let overlap_audio = &audio_window[overlap_start..];
            is_overlap_silence(overlap_audio, self.accumulated.mean_amplitude)
        } else {
            false
        };

        // Update accumulated transcript
        if is_silence {
            info!("Not attempting transcript weaving due to overlap silence");
            self.accumulated.transcript.push(' ');
            self.accumulated.transcript.push_str(&segment_transcript);
        } else {
            // Use transcript weaving
            self.accumulated.transcript = weave_transcript_segs(
                &self.accumulated.transcript,
                &segment_transcript,
                overlap,
                MIN_ALIGNMENT_SCORE,
            );
        }

        // Accumulate token IDs
        let logit_target_slice = target_slice.map(sample_index_to_logit_index);
        debug!(
            "Accumulating tokens to slice [{}, {}] (time: [{:.2}, {:.2}])",
            logit_target_slice.start,
            logit_target_slice.end,
            logit_index_to_time(logit_target_slice.start),
            logit_index_to_time(logit_target_slice.end),
        );

        // Ensure our accumulated tokens vector is large enough
        if self.accumulated.token_ids.len() < logit_target_slice.end {
            let old_len = self.accumulated.token_ids.len();
            self.accumulated.token_ids.resize(logit_target_slice.end, 0);
            debug!(
                "Resized token vector from {} to {}",
                old_len, logit_target_slice.end
            );
        }

        // Copy the new tokens into the appropriate slice
        let tokens_to_copy = std::cmp::min(transcription.tokens.len(), logit_target_slice.len());

        if tokens_to_copy > 0 && logit_target_slice.start < self.accumulated.token_ids.len() {
            let end_idx = std::cmp::min(
                logit_target_slice.start + tokens_to_copy,
                self.accumulated.token_ids.len(),
            );

            self.accumulated.token_ids[logit_target_slice.start..end_idx]
                .copy_from_slice(&transcription.tokens[..tokens_to_copy]);
        }

        Ok(())
    }

    /// Process a complete audio buffer in a single batch operation.
    ///
    /// # Arguments
    /// * `audio_bytes` - The raw audio bytes (16-bit PCM)
    ///
    /// # Returns
    /// The transcription result
    pub async fn process_batch(&mut self, audio_bytes: &[u8]) -> Result<Transcription> {
        // Clear any existing state
        self.clear();

        // Convert to f32 samples
        let samples = bytes_to_f32_samples(audio_bytes);
        let audio_len = samples.len();

        // If the audio is small enough, just process it directly
        if audio_len as f32 / W2V_SAMPLE_RATE as f32 <= self.chunk_size {
            return self.pipeline.process_batch(audio_bytes).await;
        }

        // Otherwise, process with overlapping chunks
        self.audio_buffer.add_samples(&samples);
        self.process_buffered_audio().await?;

        // Create transcription from accumulated state
        Ok(Transcription {
            text: self.accumulated.transcript.clone(),
            tokens: self.accumulated.token_ids.clone(),
            audio_length_samples: audio_len,
            features_length: 0, // We don't have this information here
            encoded_length: 0,  // We don't have this information here
        })
    }

    /// Get the current audio length in seconds.
    pub fn audio_length(&self) -> f32 {
        audio_len(self.audio_buffer.get_window())
    }
}
