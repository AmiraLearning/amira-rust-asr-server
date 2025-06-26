//! SIMD-optimized audio processing kernels.
//!
//! This module provides high-performance SIMD implementations for common
//! audio processing operations, achieving significant speedups over scalar code.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized audio conversion for x86_64 with AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bytes_to_f32_avx2(input: &[u8], output: &mut Vec<f32>) {
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
        
        // Convert to i32 (needed for precise conversion)
        let i32_lo_lo = _mm256_cvtepi16_epi32(i16_lo);
        let i32_lo_hi = _mm256_cvtepi16_epi32(_mm_shuffle_epi32(i16_lo, 0x4E));
        let i32_hi_lo = _mm256_cvtepi16_epi32(i16_hi);
        let i32_hi_hi = _mm256_cvtepi16_epi32(_mm_shuffle_epi32(i16_hi, 0x4E));
        
        // Convert to f32 and normalize
        let f32_0 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_lo_lo), scale);
        let f32_1 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_lo_hi), scale);
        let f32_2 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_hi_lo), scale);
        let f32_3 = _mm256_mul_ps(_mm256_cvtepi32_ps(i32_hi_hi), scale);
        
        // Extend output vector
        let old_len = output.len();
        output.set_len(old_len + 16);
        
        // Store results
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len), f32_0);
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len + 8), f32_1);
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len + 16), f32_2);
        _mm256_storeu_ps(output.as_mut_ptr().add(old_len + 24), f32_3);
    }
    
    // Handle remainder with scalar code
    if !remainder.is_empty() {
        bytes_to_f32_scalar(remainder, output);
    }
}

/// Scalar fallback for audio conversion.
fn bytes_to_f32_scalar(input: &[u8], output: &mut Vec<f32>) {
    for chunk in input.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        output.push(sample as f32 / 32768.0);
    }
}

/// SIMD-optimized mean amplitude calculation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mean_amplitude_avx2(samples: &[f32]) -> f32 {
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
fn mean_amplitude_scalar(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    
    samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32
}

/// Public interface for optimized audio conversion.
/// Automatically selects the best implementation based on available CPU features.
pub fn bytes_to_f32_optimized(input: &[u8], output: &mut Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { bytes_to_f32_avx2(input, output) }
        } else {
            bytes_to_f32_scalar(input, output)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        bytes_to_f32_scalar(input, output)
    }
}

/// Public interface for optimized mean amplitude calculation.
/// Automatically selects the best implementation based on available CPU features.
pub fn mean_amplitude_optimized(samples: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { mean_amplitude_avx2(samples) }
        } else {
            mean_amplitude_scalar(samples)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        mean_amplitude_scalar(samples)
    }
}

/// SIMD-optimized smoothing filter for audio data.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn smooth_audio_avx2(input: &[f32], output: &mut [f32], window_size: usize) {
    if window_size == 0 || input.len() < window_size {
        output.copy_from_slice(input);
        return;
    }
    
    let window_size_f32 = window_size as f32;
    let window_recip = _mm256_set1_ps(1.0 / window_size_f32);
    
    for i in 0..input.len() {
        let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
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

/// Public interface for optimized audio smoothing.
pub fn smooth_audio_optimized(input: &[f32], output: &mut [f32], window_size: usize) {
    if input.len() != output.len() {
        panic!("Input and output slices must have the same length");
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
}

/// Scalar fallback for audio smoothing.
fn smooth_audio_scalar(input: &[f32], output: &mut [f32], window_size: usize) {
    if window_size == 0 {
        output.copy_from_slice(input);
        return;
    }
    
    for i in 0..input.len() {
        let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
        let end = std::cmp::min(start + window_size, input.len());
        let window = &input[start..end];
        output[i] = window.iter().sum::<f32>() / window.len() as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bytes_to_f32_consistency() {
        let test_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        
        let mut scalar_result = Vec::new();
        bytes_to_f32_scalar(&test_data, &mut scalar_result);
        
        let mut simd_result = Vec::new();
        bytes_to_f32_optimized(&test_data, &mut simd_result);
        
        assert_eq!(scalar_result.len(), simd_result.len());
        
        for (i, (&scalar, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-6,
                "Mismatch at index {}: scalar={}, simd={}",
                i, scalar, simd
            );
        }
    }
    
    #[test]
    fn test_mean_amplitude_consistency() {
        let test_data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();
        
        let scalar_result = mean_amplitude_scalar(&test_data);
        let simd_result = mean_amplitude_optimized(&test_data);
        
        assert!(
            (scalar_result - simd_result).abs() < 1e-6,
            "Mean amplitude mismatch: scalar={}, simd={}",
            scalar_result, simd_result
        );
    }
    
    #[test]
    fn test_smoothing_consistency() {
        let test_data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut scalar_result = vec![0.0; test_data.len()];
        let mut simd_result = vec![0.0; test_data.len()];
        
        smooth_audio_scalar(&test_data, &mut scalar_result, 5);
        smooth_audio_optimized(&test_data, &mut simd_result, 5);
        
        for (i, (&scalar, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-6,
                "Smoothing mismatch at index {}: scalar={}, simd={}",
                i, scalar, simd
            );
        }
    }
}