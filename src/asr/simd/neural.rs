//! SIMD-optimized neural network operations
//!
//! This module provides high-performance SIMD implementations for:
//! - Softmax activation
//! - Batch normalization
//! - Dot product operations

use crate::error::{AppError, Result};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::intrinsics::{exp_approx_avx2, exp_approx_avx512};

// ============================================================================
// SOFTMAX OPERATIONS
// ============================================================================

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
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vals = _mm512_loadu_ps(chunk.as_ptr());
            max_vec = _mm512_max_ps(max_vec, vals);
        }

        // Horizontal maximum
        let max_array: [f32; 16] = std::mem::transmute(max_vec);
        max_val = max_array.iter().fold(max_val, |acc, &x| acc.max(x));

        // Handle remainder
        for &val in remainder {
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
        let remainder_in = chunks_in.remainder();
        let chunks_out = output.chunks_exact_mut(16);

        for (chunk_in, chunk_out) in chunks_in.zip(chunks_out) {
            let vals = _mm512_loadu_ps(chunk_in.as_ptr());
            let shifted = _mm512_sub_ps(vals, max_broadcast);
            let exp_vals = exp_approx_avx512(shifted);

            _mm512_storeu_ps(chunk_out.as_mut_ptr(), exp_vals);
            sum_vec = _mm512_add_ps(sum_vec, exp_vals);
        }

        // Horizontal sum
        let sum_array: [f32; 16] = std::mem::transmute(sum_vec);
        sum = sum_array.iter().sum();

        // Handle remainder
        let remainder_out = &mut output[len - remainder_in.len()..];
        for (&val, out) in remainder_in.iter().zip(remainder_out.iter_mut()) {
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
        let remainder_len = len % 16;
        let remainder_start = len - remainder_len;
        let chunks = output.chunks_exact_mut(16);

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
        let remainder = chunks.remainder();

        for chunk in chunks {
            let vals = _mm256_loadu_ps(chunk.as_ptr());
            max_vec = _mm256_max_ps(max_vec, vals);
        }

        // Horizontal maximum
        let max_array: [f32; 8] = std::mem::transmute(max_vec);
        max_val = max_array.iter().fold(max_val, |acc, &x| acc.max(x));

        for &val in remainder {
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
        let remainder_in = chunks_in.remainder();
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
        let remainder_out = &mut output[len - remainder_in.len()..];
        for (&val, out) in remainder_in.iter().zip(remainder_out.iter_mut()) {
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

// ============================================================================
// BATCH NORMALIZATION
// ============================================================================

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
        let remainder_in = chunks_in.remainder();
        let chunks_out = output.chunks_exact_mut(8);

        for (chunk_in, chunk_out) in chunks_in.zip(chunks_out) {
            let vals = _mm256_loadu_ps(chunk_in.as_ptr());
            let centered = _mm256_sub_ps(vals, mean_broadcast);
            let normalized = _mm256_mul_ps(centered, inv_std_broadcast);
            _mm256_storeu_ps(chunk_out.as_mut_ptr(), normalized);
        }

        // Handle remainder
        let remainder_out = &mut output[len - remainder_in.len()..];
        for (&val, out) in remainder_in.iter().zip(remainder_out.iter_mut()) {
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

// ============================================================================
// DOT PRODUCT
// ============================================================================

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
        let remainder_a = chunks_a.remainder();
        let chunks_b = b.chunks_exact(8);
        let remainder_b = chunks_b.remainder();

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
        for (&a_val, &b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            result += a_val * b_val;
        }
    } else {
        // Scalar fallback
        for (&a_val, &b_val) in a.iter().zip(b.iter()) {
            result += a_val * b_val;
        }
    }

    Ok(result)
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
