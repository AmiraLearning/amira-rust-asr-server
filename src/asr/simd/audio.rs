//! SIMD-optimized audio processing functions
//!
//! This module provides high-performance SIMD implementations for audio processing:
//! - Audio conversion (bytes to f32)
//! - Amplitude calculation
//! - Automatic SIMD selection based on CPU features

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// AUDIO CONVERSION FUNCTIONS
// ============================================================================

/// Safe SIMD-optimized audio conversion with bounds checking.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bytes_to_f32_safe_avx2(input: &[u8], output: &mut Vec<f32>) {
    output.clear();
    output.reserve(input.len() / 2);

    let scale = _mm256_set1_ps(1.0 / 32768.0);
    let chunks = input.chunks_exact(16); // 8 i16 samples
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Load 16 bytes as 8 i16 values
        let bytes = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

        // Convert to i32 for precision
        let i32_vals = _mm256_cvtepi16_epi32(bytes);

        // Convert to f32 and scale
        let f32_vals = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_vals), scale);

        // Store directly - safe method
        let old_len = output.len();
        output.resize(old_len + 8, 0.0);
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len), f32_vals);
    }

    // Handle remainder
    bytes_to_f32_scalar(remainder, output);
}

/// Original complex SIMD implementation (kept for reference).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn bytes_to_f32_avx2_complex(input: &[u8], output: &mut Vec<f32>) {
    output.clear();

    if input.len() < 32 {
        // Fall back to scalar for small inputs
        bytes_to_f32_scalar(input, output);
        return;
    }

    output.reserve(input.len() / 2);

    const SIMD_WIDTH: usize = 16; // 16 i16 values per AVX2 register
    let scale = _mm256_set1_ps(1.0 / 32768.0);

    let chunks = input.chunks_exact(32); // 16 i16 samples * 2 bytes each
    let remainder = chunks.remainder();

    for chunk in chunks {
        // Load 32 bytes (16 i16 samples)
        let bytes = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

        // Extract low and high 8 i16 values
        let i16_lo = _mm256_extracti128_si256(bytes, 0);
        let i16_hi = _mm256_extracti128_si256(bytes, 1);

        // Convert i16 to i32, then to f32 and normalize
        // i16_lo contains elements 0-7, i16_hi contains elements 8-15
        let i32_lo = _mm256_cvtepi16_epi32(i16_lo); // Convert elements 0-7
        let i32_hi = _mm256_cvtepi16_epi32(i16_hi); // Convert elements 8-15

        let f32_lo = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_lo), scale); // Elements 0-7
        let f32_hi = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_hi), scale); // Elements 8-15

        // Extend output vector - safe method (16 i16 -> 16 f32)
        let old_len = output.len();
        output.resize(old_len + 16, 0.0);

        // Store results (each vector contains 8 f32 values)
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len), f32_lo); // 0-7
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len + 8), f32_hi); // 8-15
    }

    // Handle remainder with scalar code
    if !remainder.is_empty() {
        bytes_to_f32_scalar(remainder, output);
    }
}

/// Scalar fallback for audio conversion.
pub(crate) fn bytes_to_f32_scalar(input: &[u8], output: &mut Vec<f32>) {
    for chunk in input.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        output.push(sample as f32 / 32768.0);
    }
}

/// Safe optimized audio conversion with automatic SIMD selection.
pub fn bytes_to_f32_safe_optimized(input: &[u8], output: &mut Vec<f32>) {
    // For small inputs, scalar is faster due to setup overhead
    if input.len() < 64 {
        bytes_to_f32_scalar(input, output);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Use runtime feature detection for safety
        if is_x86_feature_detected!("avx2") {
            unsafe { bytes_to_f32_safe_avx2(input, output) }
        } else {
            bytes_to_f32_scalar(input, output)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        bytes_to_f32_scalar(input, output)
    }
}

/// Legacy interface for backward compatibility.
pub fn bytes_to_f32_optimized(input: &[u8], output: &mut Vec<f32>) {
    bytes_to_f32_safe_optimized(input, output);
}

// ============================================================================
// AMPLITUDE CALCULATION
// ============================================================================

/// Safe SIMD-optimized mean amplitude calculation with bounds checking.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mean_amplitude_safe_avx2(samples: &[f32]) -> f32 {
    if samples.len() < 8 {
        return mean_amplitude_scalar(samples);
    }

    let mut sum_vec = _mm256_setzero_ps();
    let chunks = samples.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let values = _mm256_loadu_ps(chunk.as_ptr());
        let abs_values = _mm256_andnot_ps(_mm256_set1_ps(-0.0), values); // Fast abs using bit manipulation
        sum_vec = _mm256_add_ps(sum_vec, abs_values);
    }

    // Horizontal sum of the vector
    let sum_low = _mm256_extractf128_ps(sum_vec, 0);
    let sum_high = _mm256_extractf128_ps(sum_vec, 1);
    let sum_combined = _mm_add_ps(sum_low, sum_high);

    // Further reduce to scalar
    let sum_shuffled = _mm_shuffle_ps(sum_combined, sum_combined, 0x4E);
    let sum_added = _mm_add_ps(sum_combined, sum_shuffled);
    let sum_final_shuffle = _mm_shuffle_ps(sum_added, sum_added, 0x11);
    let final_sum = _mm_add_ss(sum_added, sum_final_shuffle);

    let mut scalar_sum = _mm_cvtss_f32(final_sum);

    // Add remainder
    scalar_sum += remainder.iter().map(|x| x.abs()).sum::<f32>();

    scalar_sum / samples.len() as f32
}

/// Scalar fallback for mean amplitude calculation.
pub(crate) fn mean_amplitude_scalar(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32
}

/// Safe optimized mean amplitude calculation with automatic SIMD selection.
pub fn mean_amplitude_safe_optimized(samples: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { mean_amplitude_safe_avx2(samples) }
        } else {
            mean_amplitude_scalar(samples)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        mean_amplitude_scalar(samples)
    }
}

/// Legacy interface for backward compatibility.
pub fn mean_amplitude_optimized(samples: &[f32]) -> f32 {
    mean_amplitude_safe_optimized(samples)
}
