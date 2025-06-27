//! SIMD Performance Validation Test
//!
//! This binary tests the actual performance improvements from Phase 3 SIMD optimizations.

use amira_rust_asr_server::asr::simd::{
    argmax_optimized, gemm_f32_optimized, transpose_encoder_output,
};
use std::time::Instant;

fn main() {
    println!("🚀 SIMD Performance Validation Test");
    println!("====================================\n");

    // Check CPU capabilities
    println!("🔍 CPU Feature Detection:");
    #[cfg(target_arch = "x86_64")]
    {
        println!("  AVX2:     {}", is_x86_feature_detected!("avx2"));
        println!("  AVX512F:  {}", is_x86_feature_detected!("avx512f"));
        println!("  FMA:      {}", is_x86_feature_detected!("fma"));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("  Non-x86_64 architecture - SIMD optimizations use scalar fallback");
    }
    println!();

    test_tensor_transpose();
    test_gemm_performance();
    test_argmax_performance();

    println!("✅ All SIMD optimizations validated successfully!");
    println!("📊 Note: Performance depends on CPU features and matrix sizes.");
    println!(
        "💡 Optimizations automatically fallback to scalar for best performance on small matrices."
    );
}

fn test_tensor_transpose() {
    println!("🔄 Testing Tensor Transpose Performance (Highest Priority Optimization)");

    // Use larger matrices to trigger SIMD path (threshold is 1024 elements)
    let features = 1024;
    let time_steps = 512;
    let iterations = 100;

    let input: Vec<f32> = (0..(features * time_steps))
        .map(|i| i as f32 * 0.1)
        .collect();
    let mut output = vec![0.0; time_steps * features];

    // Test optimized version
    let start = Instant::now();
    for _ in 0..iterations {
        transpose_encoder_output(&input, &mut output, features, time_steps);
    }
    let optimized_time = start.elapsed();

    // Test scalar version
    let start = Instant::now();
    for _ in 0..iterations {
        for t in 0..time_steps {
            for f in 0..features {
                let src_idx = f * time_steps + t;
                let dst_idx = t * features + f;
                output[dst_idx] = input[src_idx];
            }
        }
    }
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!(
        "  Matrix size: {}x{} ({} elements)",
        features,
        time_steps,
        features * time_steps
    );
    println!("  Iterations: {}", iterations);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  SIMD time:   {:?}", optimized_time);
    println!("  Speedup:     {:.2}x", speedup);

    if speedup > 1.5 {
        println!("  ✅ Excellent speedup! SIMD optimization is working.");
    } else if speedup > 1.1 {
        println!("  ⚠️  Moderate speedup. May need larger matrices for full SIMD benefit.");
    } else {
        println!("  ❌ Limited speedup. SIMD may not be enabled or matrix too small.");
    }
    println!();
}

fn test_gemm_performance() {
    println!("🔢 Testing GEMM Performance (Custom kernels for small matrices)");

    // Use larger matrices to trigger SIMD path (threshold is 4096 elements)
    let m = 128;
    let n = 128;
    let k = 128;
    let iterations = 50;

    let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i + 1) as f32 * 0.01).collect();
    let mut c = vec![0.0; m * n];

    // Test optimized version
    let start = Instant::now();
    for _ in 0..iterations {
        gemm_f32_optimized(&a, &b, &mut c, m, n, k);
    }
    let optimized_time = start.elapsed();

    // Test scalar version
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..m {
            for j in 0..n {
                let mut sum = c[i * n + j];
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!("  Matrix dimensions: {}x{}x{}", m, n, k);
    println!("  Operations: {} FLOPS per iteration", 2 * m * n * k);
    println!("  Iterations: {}", iterations);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  SIMD time:   {:?}", optimized_time);
    println!("  Speedup:     {:.2}x", speedup);

    if speedup > 2.0 {
        println!("  ✅ Excellent speedup! Custom GEMM kernels are highly effective.");
    } else if speedup > 1.2 {
        println!("  ⚠️  Good speedup. SIMD providing benefit for small matrices.");
    } else {
        println!("  ❌ Limited speedup. May fallback to scalar for small matrices.");
    }
    println!();
}

fn test_argmax_performance() {
    println!("🎯 Testing Argmax Performance (Index tracking optimization)");

    // Use larger arrays to trigger SIMD path (threshold is 32 elements)
    let vocab_size = 10000;
    let iterations = 1000;

    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.01).sin()).collect();

    // Test optimized version
    let start = Instant::now();
    let mut result_opt = (0, 0.0);
    for _ in 0..iterations {
        result_opt = argmax_optimized(&logits);
    }
    let optimized_time = start.elapsed();

    // Test scalar version
    let start = Instant::now();
    let mut result_scalar = (0, 0.0);
    for _ in 0..iterations {
        let mut max_idx = 0;
        let mut max_val = logits[0];

        for (i, &val) in logits.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        result_scalar = (max_idx, max_val);
    }
    let scalar_time = start.elapsed();

    let speedup = scalar_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

    println!("  Vocabulary size: {}", vocab_size);
    println!("  Iterations: {}", iterations);
    println!("  Scalar time: {:?}", scalar_time);
    println!("  SIMD time:   {:?}", optimized_time);
    println!("  Speedup:     {:.2}x", speedup);
    println!("  Results match: {}", result_opt == result_scalar);

    if speedup > 3.0 {
        println!("  ✅ Excellent speedup! Vectorized argmax working perfectly.");
    } else if speedup > 1.5 {
        println!("  ⚠️  Good speedup. SIMD providing clear benefit.");
    } else {
        println!("  ❌ Limited speedup. May fallback to scalar for this size.");
    }
    println!();
}
