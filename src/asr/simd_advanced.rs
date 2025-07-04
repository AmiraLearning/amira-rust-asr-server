//! Advanced SIMD kernels for inference operations.
//!
//! This module provides high-performance SIMD implementations for neural network
//! operations commonly used in ASR inference, including softmax, batch operations,
//! and optimized tensor manipulations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::error::{AppError, Result};
// Note: Using intrinsics directly instead of wide crate for broader compatibility

/// SIMD-optimized softmax implementation for logits processing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_avx512(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(AppError::Internal(
            "Input and output lengths must match for softmax".to_string(),
        ));
    }
    
    if input.is_empty() {
        return Ok(());
    }
    
    let len = input.len();
    
    // Step 1: Find maximum value for numerical stability
    let mut max_val = input[0];
    
    // SIMD maximum finding
    if len >= 16 {
        let mut max_vec = _mm512_set1_ps(input[0]);
        let chunks = input.chunks_exact(16);
        
        for chunk in chunks {
            let vals = _mm512_loadu_ps(chunk.as_ptr());
            max_vec = _mm512_max_ps(max_vec, vals);
        }
        
        // Horizontal maximum
        let max_array: [f32; 16] = std::mem::transmute(max_vec);
        max_val = max_array.iter().fold(max_val, |acc, &x| acc.max(x));
        
        // Handle remainder
        for &val in chunks.remainder() {
            max_val = max_val.max(val);
        }
    } else {
        // Scalar fallback for small inputs
        for &val in input {
            max_val = max_val.max(val);
        }
    }
    
    // Step 2: Compute exp(x - max) and sum
    let max_broadcast = _mm512_set1_ps(max_val);
    let mut sum = 0.0f32;
    
    if len >= 16 {
        let mut sum_vec = _mm512_setzero_ps();
        let chunks_in = input.chunks_exact(16);
        let chunks_out = output.chunks_exact_mut(16);
        
        for (chunk_in, chunk_out) in chunks_in.zip(chunks_out) {
            let vals = _mm512_loadu_ps(chunk_in.as_ptr());
            let shifted = _mm512_sub_ps(vals, max_broadcast);
            let exp_vals = _mm512_exp_ps(shifted);
            
            _mm512_storeu_ps(chunk_out.as_mut_ptr(), exp_vals);
            sum_vec = _mm512_add_ps(sum_vec, exp_vals);
        }
        
        // Horizontal sum
        let sum_array: [f32; 16] = std::mem::transmute(sum_vec);
        sum = sum_array.iter().sum();
        
        // Handle remainder
        let remainder_in = chunks_in.remainder();
        let remainder_out = &mut output[len - remainder_in.len()..];
        for (i, (&val, out)) in remainder_in.iter().zip(remainder_out.iter_mut()).enumerate() {
            let exp_val = (val - max_val).exp();
            *out = exp_val;
            sum += exp_val;
        }
    } else {
        // Scalar fallback
        for (i, &val) in input.iter().enumerate() {
            let exp_val = (val - max_val).exp();
            output[i] = exp_val;
            sum += exp_val;
        }
    }
    
    // Step 3: Normalize by sum
    if sum == 0.0 {
        return Err(AppError::Internal("Softmax sum is zero".to_string()));
    }
    
    let inv_sum = 1.0 / sum;
    let inv_sum_broadcast = _mm512_set1_ps(inv_sum);
    
    if len >= 16 {
        let chunks = output.chunks_exact_mut(16);
        let remainder_start = len - chunks.remainder().len();
        
        for chunk in chunks {
            let vals = _mm512_loadu_ps(chunk.as_ptr());
            let normalized = _mm512_mul_ps(vals, inv_sum_broadcast);
            _mm512_storeu_ps(chunk.as_mut_ptr(), normalized);
        }
        
        // Handle remainder
        for val in &mut output[remainder_start..] {
            *val *= inv_sum;
        }
    } else {
        // Scalar fallback
        for val in output {
            *val *= inv_sum;
        }
    }
    
    Ok(())
}

/// SIMD-optimized softmax using AVX2 for broader compatibility.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn softmax_avx2(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(AppError::Internal(
            "Input and output lengths must match for softmax".to_string(),
        ));
    }
    
    if input.is_empty() {
        return Ok(());
    }
    
    let len = input.len();
    
    // Step 1: Find maximum value
    let mut max_val = input[0];
    
    if len >= 8 {
        let mut max_vec = _mm256_set1_ps(input[0]);
        let chunks = input.chunks_exact(8);
        
        for chunk in chunks {
            let vals = _mm256_loadu_ps(chunk.as_ptr());
            max_vec = _mm256_max_ps(max_vec, vals);
        }
        
        // Horizontal maximum
        let max_array: [f32; 8] = std::mem::transmute(max_vec);
        max_val = max_array.iter().fold(max_val, |acc, &x| acc.max(x));
        
        for &val in chunks.remainder() {
            max_val = max_val.max(val);
        }
    } else {
        for &val in input {
            max_val = max_val.max(val);
        }
    }
    
    // Step 2: Compute exp and sum
    let max_broadcast = _mm256_set1_ps(max_val);
    let mut sum = 0.0f32;
    
    if len >= 8 {
        let mut sum_vec = _mm256_setzero_ps();
        let chunks_in = input.chunks_exact(8);
        let chunks_out = output.chunks_exact_mut(8);
        
        for (chunk_in, chunk_out) in chunks_in.zip(chunks_out) {
            let vals = _mm256_loadu_ps(chunk_in.as_ptr());
            let shifted = _mm256_sub_ps(vals, max_broadcast);
            
            // Manual exp approximation for AVX2 (since _mm256_exp_ps is not standard)
            let exp_vals = exp_approx_avx2(shifted);
            
            _mm256_storeu_ps(chunk_out.as_mut_ptr(), exp_vals);
            sum_vec = _mm256_add_ps(sum_vec, exp_vals);
        }
        
        // Horizontal sum
        let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
        sum = sum_array.iter().sum();
        
        // Handle remainder
        let remainder_in = chunks_in.remainder();
        let remainder_out = &mut output[len - remainder_in.len()..];
        for ((&val, out)) in remainder_in.iter().zip(remainder_out.iter_mut()) {
            let exp_val = (val - max_val).exp();
            *out = exp_val;
            sum += exp_val;
        }
    } else {
        for (i, &val) in input.iter().enumerate() {
            let exp_val = (val - max_val).exp();
            output[i] = exp_val;
            sum += exp_val;
        }
    }
    
    // Step 3: Normalize
    if sum == 0.0 {
        return Err(AppError::Internal("Softmax sum is zero".to_string()));
    }
    
    let inv_sum = 1.0 / sum;
    let inv_sum_broadcast = _mm256_set1_ps(inv_sum);
    
    if len >= 8 {
        for chunk in output.chunks_exact_mut(8) {
            let vals = _mm256_loadu_ps(chunk.as_ptr());
            let normalized = _mm256_mul_ps(vals, inv_sum_broadcast);
            _mm256_storeu_ps(chunk.as_mut_ptr(), normalized);
        }
        
        let remainder_start = output.len() - output.len() % 8;
        for val in &mut output[remainder_start..] {
            *val *= inv_sum;
        }
    } else {
        for val in output {
            *val *= inv_sum;
        }
    }
    
    Ok(())
}

/// Fast exponential approximation for AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn exp_approx_avx2(x: __m256) -> __m256 {
    // Fast exp approximation using polynomial
    // exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sixth = _mm256_set1_ps(1.0 / 6.0);
    
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    
    let term1 = x;
    let term2 = _mm256_mul_ps(x2, half);
    let term3 = _mm256_mul_ps(x3, sixth);
    
    let result = _mm256_add_ps(one, term1);
    let result = _mm256_add_ps(result, term2);
    _mm256_add_ps(result, term3)
}

/// Batch SIMD operations across multiple audio streams.
pub fn batch_process_audio_streams(
    streams: &[&[u8]], 
    outputs: &mut [Vec<f32>]
) -> Result<()> {
    if streams.len() != outputs.len() {
        return Err(AppError::Internal(
            "Number of streams and outputs must match".to_string(),
        ));
    }
    
    // Process streams in parallel using SIMD
    for (stream, output) in streams.iter().zip(outputs.iter_mut()) {
        crate::asr::simd::bytes_to_f32_optimized(stream, output);
    }
    
    Ok(())
}

/// SIMD-optimized batch normalization for tensor operations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn batch_normalize_avx2(
    input: &[f32],
    output: &mut [f32],
    mean: f32,
    variance: f32,
    epsilon: f32,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(AppError::Internal(
            "Input and output lengths must match".to_string(),
        ));
    }
    
    let inv_std = 1.0 / (variance + epsilon).sqrt();
    let mean_broadcast = _mm256_set1_ps(mean);
    let inv_std_broadcast = _mm256_set1_ps(inv_std);
    
    let len = input.len();
    
    if len >= 8 {
        let chunks_in = input.chunks_exact(8);
        let chunks_out = output.chunks_exact_mut(8);
        
        for (chunk_in, chunk_out) in chunks_in.zip(chunks_out) {
            let vals = _mm256_loadu_ps(chunk_in.as_ptr());
            let centered = _mm256_sub_ps(vals, mean_broadcast);
            let normalized = _mm256_mul_ps(centered, inv_std_broadcast);
            _mm256_storeu_ps(chunk_out.as_mut_ptr(), normalized);
        }
        
        // Handle remainder
        let remainder_in = chunks_in.remainder();
        let remainder_out = &mut output[len - remainder_in.len()..];
        for ((&val, out)) in remainder_in.iter().zip(remainder_out.iter_mut()) {
            *out = (val - mean) * inv_std;
        }
    } else {
        // Scalar fallback
        for (i, &val) in input.iter().enumerate() {
            output[i] = (val - mean) * inv_std;
        }
    }
    
    Ok(())
}

/// SIMD-optimized vector dot product.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(AppError::Internal(
            "Vector lengths must match for dot product".to_string(),
        ));
    }
    
    let len = a.len();
    let mut result = 0.0f32;
    
    if len >= 8 {
        let mut sum_vec = _mm256_setzero_ps();
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vals_a = _mm256_loadu_ps(chunk_a.as_ptr());
            let vals_b = _mm256_loadu_ps(chunk_b.as_ptr());
            let product = _mm256_mul_ps(vals_a, vals_b);
            sum_vec = _mm256_add_ps(sum_vec, product);
        }
        
        // Horizontal sum
        let sum_array: [f32; 8] = std::mem::transmute(sum_vec);
        result = sum_array.iter().sum();
        
        // Handle remainder
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        for ((&a_val, &b_val)) in remainder_a.iter().zip(remainder_b.iter()) {
            result += a_val * b_val;
        }
    } else {
        // Scalar fallback
        for ((&a_val, &b_val)) in a.iter().zip(b.iter()) {
            result += a_val * b_val;
        }
    }
    
    Ok(result)
}

/// Public interface functions that automatically select the best SIMD implementation.

/// Optimized softmax with automatic SIMD selection.
pub fn softmax_optimized(input: &[f32], output: &mut [f32]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { softmax_avx512(input, output) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { softmax_avx2(input, output) }
        } else {
            softmax_scalar(input, output)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        softmax_scalar(input, output)
    }
}

/// Scalar fallback softmax implementation.
fn softmax_scalar(input: &[f32], output: &mut [f32]) -> Result<()> {
    if input.len() != output.len() {
        return Err(AppError::Internal(
            "Input and output lengths must match for softmax".to_string(),
        ));
    }
    
    if input.is_empty() {
        return Ok(());
    }
    
    // Find maximum for numerical stability
    let max_val = input.iter().fold(input[0], |acc, &x| acc.max(x));
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for (i, &val) in input.iter().enumerate() {
        let exp_val = (val - max_val).exp();
        output[i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    if sum == 0.0 {
        return Err(AppError::Internal("Softmax sum is zero".to_string()));
    }
    
    let inv_sum = 1.0 / sum;
    for val in output {
        *val *= inv_sum;
    }
    
    Ok(())
}

/// Optimized batch normalization with automatic SIMD selection.
pub fn batch_normalize_optimized(
    input: &[f32],
    output: &mut [f32],
    mean: f32,
    variance: f32,
    epsilon: f32,
) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { batch_normalize_avx2(input, output, mean, variance, epsilon) }
        } else {
            batch_normalize_scalar(input, output, mean, variance, epsilon)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        batch_normalize_scalar(input, output, mean, variance, epsilon)
    }
}

/// Scalar batch normalization fallback.
fn batch_normalize_scalar(
    input: &[f32],
    output: &mut [f32],
    mean: f32,
    variance: f32,
    epsilon: f32,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(AppError::Internal(
            "Input and output lengths must match".to_string(),
        ));
    }
    
    let inv_std = 1.0 / (variance + epsilon).sqrt();
    
    for (i, &val) in input.iter().enumerate() {
        output[i] = (val - mean) * inv_std;
    }
    
    Ok(())
}

/// Optimized dot product with automatic SIMD selection.
pub fn dot_product_optimized(a: &[f32], b: &[f32]) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { dot_product_avx2(a, b) }
        } else {
            dot_product_scalar(a, b)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        dot_product_scalar(a, b)
    }
}

/// Scalar dot product fallback.
fn dot_product_scalar(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(AppError::Internal(
            "Vector lengths must match for dot product".to_string(),
        ));
    }
    
    let result = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_softmax_optimized() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        
        softmax_optimized(&input, &mut output).unwrap();
        
        // Check that output sums to 1.0
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that values are in correct order (monotonic for this input)
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }
    
    #[test]
    fn test_batch_normalize_optimized() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 5];
        let mean = 3.0;
        let variance = 2.0;
        let epsilon = 1e-8;
        
        batch_normalize_optimized(&input, &mut output, mean, variance, epsilon).unwrap();
        
        // Check that the normalization is correct
        let expected_std = (variance + epsilon).sqrt();
        for (i, &val) in input.iter().enumerate() {
            let expected = (val - mean) / expected_std;
            assert!((output[i] - expected).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_dot_product_optimized() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        
        let result = dot_product_optimized(&a, &b).unwrap();
        let expected = 1.0*2.0 + 2.0*3.0 + 3.0*4.0 + 4.0*5.0; // = 40.0
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_batch_process_audio_streams() {
        let stream1 = vec![0x00, 0x10, 0x00, 0x20]; // 16-bit samples
        let stream2 = vec![0x00, 0x30, 0x00, 0x40];
        let streams = vec![stream1.as_slice(), stream2.as_slice()];
        let mut outputs = vec![Vec::new(), Vec::new()];
        
        batch_process_audio_streams(&streams, &mut outputs).unwrap();
        
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].len(), 2); // 4 bytes = 2 samples
        assert_eq!(outputs[1].len(), 2);
    }
}