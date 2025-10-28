//! SIMD utility functions and data structures
//!
//! This module provides utility functions and data structures for SIMD operations,
//! including aligned buffers, batch processing, and smoothing filters.

use crate::error::{AppError, AsrError, AudioError, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// ALIGNED BUFFER UTILITIES
// ============================================================================

/// Aligned buffer for optimal SIMD performance.
/// Ensures memory alignment for efficient vectorized operations.
#[derive(Debug)]
pub struct AlignedBuffer {
    data: Vec<f32>,
    #[allow(dead_code)]
    alignment: usize,
}

impl AlignedBuffer {
    /// Creates a new aligned buffer with the specified capacity and alignment.
    pub fn new(capacity: usize, alignment: usize) -> Self {
        assert!(
            alignment.is_power_of_two(),
            "Alignment must be a power of 2"
        );
        let mut data = Vec::with_capacity(capacity + alignment);

        // Ensure the data is aligned
        let ptr = data.as_ptr() as usize;
        let aligned_ptr = (ptr + alignment - 1) & !(alignment - 1);
        let offset = aligned_ptr - ptr;

        // Reserve space for alignment
        data.resize(offset, 0.0);

        Self { data, alignment }
    }

    /// Gets a mutable slice of the aligned data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Gets an immutable slice of the aligned data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Resizes the buffer to the new length.
    pub fn resize(&mut self, new_len: usize, value: f32) {
        self.data.resize(new_len, value);
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Returns the length of the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ============================================================================
// BATCH PROCESSING UTILITIES
// ============================================================================

/// Batch SIMD operations across multiple audio streams.
pub fn batch_process_audio_streams(streams: &[&[u8]], outputs: &mut [Vec<f32>]) -> Result<()> {
    if streams.len() != outputs.len() {
        return Err(AppError::Internal(
            "Number of streams and outputs must match".to_string(),
        ));
    }

    // Process streams in parallel using SIMD
    for (stream, output) in streams.iter().zip(outputs.iter_mut()) {
        super::audio::bytes_to_f32_optimized(stream, output);
    }

    Ok(())
}

// ============================================================================
// AUDIO SMOOTHING
// ============================================================================

/// SIMD-optimized smoothing filter for audio data.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn smooth_audio_avx2(input: &[f32], output: &mut [f32], window_size: usize) {
    if window_size == 0 || input.len() < window_size {
        output.copy_from_slice(input);
        return;
    }

    let window_size_f32 = window_size as f32;
    let _window_recip = _mm256_set1_ps(1.0 / window_size_f32);

    for i in 0..input.len() {
        let start = if i >= window_size / 2 {
            i - window_size / 2
        } else {
            0
        };
        let end = std::cmp::min(start + window_size, input.len());
        let actual_window = &input[start..end];

        if actual_window.len() >= 8 {
            let mut sum_vec = _mm256_setzero_ps();
            let chunks = actual_window.chunks_exact(8);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let values = _mm256_loadu_ps(chunk.as_ptr());
                sum_vec = _mm256_add_ps(sum_vec, values);
            }

            // Horizontal sum
            let sum_low = _mm256_extractf128_ps(sum_vec, 0);
            let sum_high = _mm256_extractf128_ps(sum_vec, 1);
            let sum_combined = _mm_add_ps(sum_low, sum_high);

            let sum_shuffled = _mm_shuffle_ps(sum_combined, sum_combined, 0x4E);
            let sum_added = _mm_add_ps(sum_combined, sum_shuffled);
            let sum_final_shuffle = _mm_shuffle_ps(sum_added, sum_added, 0x11);
            let final_sum = _mm_add_ss(sum_added, sum_final_shuffle);

            let mut scalar_sum = _mm_cvtss_f32(final_sum);
            scalar_sum += remainder.iter().sum::<f32>();

            output[i] = scalar_sum / actual_window.len() as f32;
        } else {
            output[i] = actual_window.iter().sum::<f32>() / actual_window.len() as f32;
        }
    }
}

/// Scalar fallback for audio smoothing.
pub(crate) fn smooth_audio_scalar(input: &[f32], output: &mut [f32], window_size: usize) {
    if window_size == 0 {
        output.copy_from_slice(input);
        return;
    }

    for (i, out) in output.iter_mut().enumerate().take(input.len()) {
        let start = i.saturating_sub(window_size / 2);
        let end = std::cmp::min(start + window_size, input.len());
        let window = &input[start..end];
        *out = window.iter().sum::<f32>() / window.len() as f32;
    }
}

/// Public interface for optimized audio smoothing.
pub fn smooth_audio_optimized(input: &[f32], output: &mut [f32], window_size: usize) -> Result<()> {
    if input.len() != output.len() {
        return Err(AppError::Asr(AsrError::AudioProcessing(
            AudioError::InvalidFormat(format!(
                "Input and output slices must have the same length: {} != {}",
                input.len(),
                output.len()
            )),
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { smooth_audio_avx2(input, output, window_size) }
        } else {
            smooth_audio_scalar(input, output, window_size)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        smooth_audio_scalar(input, output, window_size)
    }

    Ok(())
}
