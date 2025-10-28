//! SIMD-optimized kernels for audio processing and neural network operations.
//!
//! This module provides high-performance SIMD implementations for:
//! - Audio processing: conversion, amplitude calculation, smoothing
//! - Neural network operations: softmax, batch normalization, dot products
//! - Matrix operations: transpose, GEMM, argmax
//! - Batch processing utilities
//!
//! The implementation automatically selects the best SIMD instruction set
//! (AVX-512, AVX2) based on runtime CPU feature detection, with scalar fallbacks.

// Module declarations
pub mod audio;
pub mod intrinsics;
pub mod matrix;
pub mod neural;
pub mod utils;

// Re-export utility types
pub use utils::AlignedBuffer;

// Re-export audio processing functions
pub use audio::{
    bytes_to_f32_optimized, bytes_to_f32_safe_optimized, mean_amplitude_optimized,
    mean_amplitude_safe_optimized,
};

// Re-export neural network operations
pub use neural::{batch_normalize_optimized, dot_product_optimized, softmax_optimized};

// Re-export matrix operations
pub use matrix::{argmax_optimized, gemm_f32_optimized, transpose_encoder_output};

// Re-export utilities
pub use utils::{batch_process_audio_streams, smooth_audio_optimized};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_f32_consistency() {
        let test_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let mut scalar_result = Vec::new();
        audio::bytes_to_f32_scalar(&test_data, &mut scalar_result);

        let mut simd_result = Vec::new();
        bytes_to_f32_optimized(&test_data, &mut simd_result);

        assert_eq!(scalar_result.len(), simd_result.len());

        for (i, (&scalar, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-6,
                "Mismatch at index {}: scalar={}, simd={}",
                i,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_mean_amplitude_consistency() {
        let test_data: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) / 100.0).collect();

        let scalar_result = audio::mean_amplitude_scalar(&test_data);
        let simd_result = mean_amplitude_optimized(&test_data);

        assert!(
            (scalar_result - simd_result).abs() < 1e-6,
            "Mean amplitude mismatch: scalar={}, simd={}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_smoothing_consistency() {
        let test_data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut scalar_result = vec![0.0; test_data.len()];
        let mut simd_result = vec![0.0; test_data.len()];

        utils::smooth_audio_scalar(&test_data, &mut scalar_result, 5);
        smooth_audio_optimized(&test_data, &mut simd_result, 5).unwrap();

        for (i, (&scalar, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-6,
                "Smoothing mismatch at index {}: scalar={}, simd={}",
                i,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_tensor_transpose_consistency() {
        let features = 32;
        let time_steps = 64;
        let input: Vec<f32> = (0..(features * time_steps))
            .map(|i| i as f32 * 0.1)
            .collect();

        let mut scalar_result = vec![0.0; time_steps * features];
        let mut simd_result = vec![0.0; time_steps * features];

        matrix::transpose_encoder_output_scalar(&input, &mut scalar_result, features, time_steps);
        transpose_encoder_output(&input, &mut simd_result, features, time_steps);

        for (i, (&scalar, &simd)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-6,
                "Transpose mismatch at index {}: scalar={}, simd={}",
                i,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_tensor_transpose_small_matrix() {
        let features = 4;
        let time_steps = 8;
        let input: Vec<f32> = (0..(features * time_steps)).map(|i| i as f32).collect();

        let mut result = vec![0.0; time_steps * features];
        transpose_encoder_output(&input, &mut result, features, time_steps);

        // Verify specific values
        assert_eq!(result[0], 0.0); // [0, 0] -> input[0]
        assert_eq!(result[1], 8.0); // [0, 1] -> input[8]
        assert_eq!(result[4], 1.0); // [1, 0] -> input[1]
        assert_eq!(result[5], 9.0); // [1, 1] -> input[9]
    }

    #[test]
    fn test_gemm_consistency() {
        let m = 8;
        let n = 8;
        let k = 8;

        let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i + 1) as f32 * 0.1).collect();
        let mut c_scalar = vec![0.0; m * n];
        let mut c_simd = vec![0.0; m * n];

        matrix::gemm_f32_scalar(&a, &b, &mut c_scalar, m, n, k);
        gemm_f32_optimized(&a, &b, &mut c_simd, m, n, k);

        for (i, (&scalar, &simd)) in c_scalar.iter().zip(c_simd.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-5,
                "GEMM mismatch at index {}: scalar={}, simd={}",
                i,
                scalar,
                simd
            );
        }
    }

    #[test]
    fn test_argmax_consistency() {
        let test_data: Vec<f32> = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 8.0, 6.0, 0.0];

        let (scalar_idx, scalar_val) = matrix::argmax_scalar(&test_data);
        let (simd_idx, simd_val) = argmax_optimized(&test_data);

        assert_eq!(scalar_idx, simd_idx);
        assert!((scalar_val - simd_val).abs() < 1e-6);
        assert_eq!(scalar_idx, 3); // Index of 9.0
        assert!((scalar_val - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax_large_array() {
        let size = 1000;
        let mut test_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        test_data[500] = 2000.0; // Max value at index 500

        let (idx, val) = argmax_optimized(&test_data);
        assert_eq!(idx, 500);
        assert!((val - 2000.0).abs() < 1e-6);
    }
}
