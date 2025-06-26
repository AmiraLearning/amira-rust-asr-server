//! Audio processing utilities.
//!
//! This module contains utilities for working with audio data, including
//! conversion between formats and a high-performance ring buffer for
//! streaming audio.

use crate::asr::types::{SeqSlice, W2V_SAMPLE_RATE};
use crate::error::{AppError, Result};
use tracing::debug;

/// Convert raw audio bytes (16-bit PCM) to floating point samples.
///
/// # Arguments
/// * `audio_bytes` - Raw audio bytes in 16-bit PCM format
///
/// # Returns
/// Vector of f32 samples normalized to [-1.0, 1.0]
pub fn bytes_to_f32_samples(audio_bytes: &[u8]) -> Vec<f32> {
    audio_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Convert raw audio bytes (16-bit PCM) to floating point samples into an existing buffer.
/// This version avoids allocations by reusing the provided buffer and uses SIMD optimizations.
///
/// # Arguments
/// * `audio_bytes` - Raw audio bytes in 16-bit PCM format
/// * `output` - Output buffer to write samples into (will be cleared first)
pub fn bytes_to_f32_samples_into(audio_bytes: &[u8], output: &mut Vec<f32>) {
    crate::asr::simd::bytes_to_f32_optimized(audio_bytes, output);
}

/// Get the length of audio in seconds.
///
/// # Arguments
/// * `audio` - Audio samples
///
/// # Returns
/// Length in seconds
pub fn audio_len(audio: &[f32]) -> f32 {
    audio.len() as f32 / W2V_SAMPLE_RATE as f32
}

/// Calculate the mean amplitude of audio using SIMD optimizations.
///
/// # Arguments
/// * `audio` - Audio samples
///
/// # Returns
/// Mean amplitude value
pub fn calculate_mean_amplitude(audio: &[f32]) -> f32 {
    crate::asr::simd::mean_amplitude_optimized(audio)
}

/// Generate sequence windows with overlap.
///
/// # Arguments
/// * `total_len` - Total length of the sequence
/// * `window_size` - Size of each window
/// * `leading_context` - Size of leading context
/// * `trailing_context` - Size of trailing context
///
/// # Returns
/// Iterator of tuples containing (source_slice, target_slice, overlap_ratio)
pub fn window_sequence(
    total_len: usize,
    window_size: usize,
    leading_context: usize,
    trailing_context: usize,
) -> impl Iterator<Item = (SeqSlice, SeqSlice, f32)> {
    WindowSequenceIterator {
        total_len,
        window_size,
        leading_context,
        trailing_context,
        consumed: 0,
    }
}

/// Iterator for generating sequence windows with overlap.
struct WindowSequenceIterator {
    total_len: usize,
    window_size: usize,
    leading_context: usize,
    trailing_context: usize,
    consumed: usize,
}

impl Iterator for WindowSequenceIterator {
    type Item = (SeqSlice, SeqSlice, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed >= self.total_len {
            return None;
        }

        let start = self.consumed;
        let end = std::cmp::min(self.total_len, self.consumed + self.window_size);
        let offset = std::cmp::min(self.leading_context, self.consumed);
        let mut overlap = self.trailing_context + self.leading_context;

        if end < self.total_len {
            self.consumed = end - self.leading_context - self.trailing_context;
        } else {
            self.consumed = end;
            if end - start < self.window_size {
                let new_start = std::cmp::max(0, end - self.window_size);
                overlap += start - new_start;
            }
        }

        let source_slice = SeqSlice::new(start, end);
        let target_slice = SeqSlice::new(start + offset, end);
        let overlap_ratio = overlap as f32 / self.window_size as f32;

        debug!(
            "Audio window [{:.2}, {:.2}] seconds, overlap ratio: {:.2}",
            start as f32 / W2V_SAMPLE_RATE as f32,
            end as f32 / W2V_SAMPLE_RATE as f32,
            overlap_ratio
        );

        Some((source_slice, target_slice, overlap_ratio))
    }
}

/// Enhanced audio buffer for streaming with overlap support.
///
/// This structure allows efficient storage and management of audio data
/// with support for overlapping windows and audio energy analysis.
#[derive(Debug)]
pub struct OverlappingAudioBuffer {
    /// The underlying audio samples
    buffer: Vec<f32>,

    /// Current buffer length
    length: usize,

    /// Maximum capacity
    capacity: usize,

    /// Leading context size in samples
    leading_context: usize,

    /// Trailing context size in samples
    trailing_context: usize,

    /// Main chunk size in samples
    chunk_size: usize,

    /// Mean amplitude of processed audio
    mean_amplitude: f32,
}

impl OverlappingAudioBuffer {
    /// Create a new overlapping audio buffer.
    ///
    /// # Arguments
    /// * `capacity` - Maximum buffer capacity in samples
    /// * `chunk_size` - Size of processing chunks in seconds
    /// * `leading_context` - Leading context size in seconds
    /// * `trailing_context` - Trailing context size in seconds
    pub fn new(
        capacity: usize,
        chunk_size: f32,
        leading_context: f32,
        trailing_context: f32,
    ) -> Self {
        let chunk_samples = (chunk_size * W2V_SAMPLE_RATE as f32) as usize;
        let leading_samples = (leading_context * W2V_SAMPLE_RATE as f32) as usize;
        let trailing_samples = (trailing_context * W2V_SAMPLE_RATE as f32) as usize;

        Self {
            buffer: vec![0.0; capacity],
            length: 0,
            capacity,
            leading_context: leading_samples,
            trailing_context: trailing_samples,
            chunk_size: chunk_samples,
            mean_amplitude: 0.0,
        }
    }

    /// Add audio samples to the buffer.
    ///
    /// If the buffer would overflow, the oldest samples are discarded
    /// while preserving context.
    ///
    /// # Arguments
    /// * `samples` - Audio samples to add
    pub fn add_samples(&mut self, samples: &[f32]) {
        let samples_len = samples.len();

        // If samples exceed available space, we need to shift the buffer
        if self.length + samples_len > self.capacity {
            // Keep enough context from the old buffer
            let context_to_keep = std::cmp::min(self.leading_context, self.length);
            let start_idx = self.length - context_to_keep;

            // Shift buffer to make room
            if context_to_keep > 0 {
                self.buffer.copy_within(start_idx..self.length, 0);
            }

            self.length = context_to_keep;
        }

        // Add new samples
        let start_idx = self.length;
        let end_idx = start_idx + samples_len;

        if end_idx <= self.capacity {
            self.buffer[start_idx..end_idx].copy_from_slice(samples);
            self.length = end_idx;

            // Update mean amplitude
            if self.mean_amplitude == 0.0 {
                self.mean_amplitude = calculate_mean_amplitude(samples);
            } else {
                // Exponential moving average with alpha = 0.3
                let new_amplitude = calculate_mean_amplitude(samples);
                self.mean_amplitude = 0.7 * self.mean_amplitude + 0.3 * new_amplitude;
            }
        } else {
            debug!("Buffer capacity exceeded, truncating samples");
            let available = self.capacity - start_idx;
            self.buffer[start_idx..self.capacity].copy_from_slice(&samples[..available]);
            self.length = self.capacity;
        }
    }

    /// Get the current audio window for processing.
    ///
    /// Returns a slice of the buffer containing the current window.
    pub fn get_window(&self) -> &[f32] {
        &self.buffer[..self.length]
    }

    /// Get the mean amplitude of the audio.
    pub fn mean_amplitude(&self) -> f32 {
        self.mean_amplitude
    }

    /// Generate overlapping windows from the buffer.
    ///
    /// Each call returns a window of audio with appropriate context,
    /// and the overlap ratio for weaving transcripts.
    pub fn overlapping_windows(&self) -> impl Iterator<Item = (SeqSlice, SeqSlice, f32)> + '_ {
        window_sequence(
            self.length,
            self.chunk_size + self.leading_context + self.trailing_context,
            self.leading_context,
            self.trailing_context,
        )
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.length = 0;
        self.mean_amplitude = 0.0;
    }

    /// Get the audio from a specific slice.
    ///
    /// # Arguments
    /// * `slice` - The sequence slice to extract
    ///
    /// # Returns
    /// The audio samples for the slice
    pub fn get_slice(&self, slice: &SeqSlice) -> &[f32] {
        if slice.end <= self.length {
            &self.buffer[slice.start..slice.end]
        } else {
            debug!(
                "Slice end {} exceeds buffer length {}",
                slice.end, self.length
            );
            &self.buffer[slice.start..self.length]
        }
    }
}

/// A lock-free ring buffer for audio streaming.
///
/// This structure allows efficient storage and retrieval of audio data
/// in a streaming context, with support for circular read/write operations.
#[derive(Debug)]
pub struct AudioRingBuffer {
    /// The underlying byte buffer
    buffer: Vec<u8>,

    /// Current write position
    write_pos: usize,

    /// Current read position
    read_pos: usize,

    /// Total capacity of the buffer
    capacity: usize,
}

impl AudioRingBuffer {
    /// Create a new audio ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            write_pos: 0,
            read_pos: 0,
            capacity,
        }
    }

    /// Write data to the buffer.
    ///
    /// # Arguments
    /// * `data` - Data to write to the buffer
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if the buffer would overflow
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        let len = data.len();
        if len > self.available_write() {
            return Err(AppError::Audio("Buffer overflow".to_string()));
        }

        let end_pos = (self.write_pos + len) % self.capacity;

        if end_pos > self.write_pos {
            self.buffer[self.write_pos..end_pos].copy_from_slice(data);
        } else {
            let first_part = self.capacity - self.write_pos;
            self.buffer[self.write_pos..].copy_from_slice(&data[..first_part]);
            self.buffer[..end_pos].copy_from_slice(&data[first_part..]);
        }

        self.write_pos = end_pos;
        Ok(())
    }

    /// Read data from the buffer.
    ///
    /// # Arguments
    /// * `len` - Number of bytes to read
    ///
    /// # Returns
    /// The read data, or None if not enough data is available
    pub fn read(&mut self, len: usize) -> Option<Vec<u8>> {
        if len > self.available_read() {
            return None;
        }

        let mut result = vec![0u8; len];
        let end_pos = (self.read_pos + len) % self.capacity;

        if end_pos > self.read_pos {
            result.copy_from_slice(&self.buffer[self.read_pos..end_pos]);
        } else {
            let first_part = self.capacity - self.read_pos;
            result[..first_part].copy_from_slice(&self.buffer[self.read_pos..]);
            result[first_part..].copy_from_slice(&self.buffer[..end_pos]);
        }

        self.read_pos = end_pos;
        Some(result)
    }

    /// Get the number of bytes available to read.
    pub fn available_read(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        }
    }

    /// Get the number of bytes available to write.
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read() - 1
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
    }
}
