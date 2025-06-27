//! SIMD Performance Benchmarks
//!
//! This benchmark validates the Phase 3 SIMD optimizations from PERFORMANCE.md
//! and measures the actual performance improvements achieved.

use amira_rust_asr_server::asr::simd::{
    argmax_optimized, bytes_to_f32_optimized, gemm_f32_optimized, transpose_encoder_output,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Benchmark tensor transpose operations (highest priority optimization)
fn benchmark_tensor_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_transpose");

    // Test different matrix sizes typical for RNN-T
    let sizes = vec![
        (128, 64),   // Small streaming chunk
        (256, 128),  // Medium streaming chunk
        (512, 256),  // Large streaming chunk
        (1024, 512), // Batch processing
    ];

    for (features, time_steps) in sizes {
        let input: Vec<f32> = (0..(features * time_steps))
            .map(|i| i as f32 * 0.1)
            .collect();
        let mut output = vec![0.0; time_steps * features];

        group.throughput(Throughput::Elements((features * time_steps) as u64));

        group.bench_with_input(
            BenchmarkId::new("optimized", format!("{}x{}", features, time_steps)),
            &(features, time_steps),
            |bench, &(f, t)| {
                bench.iter(|| {
                    transpose_encoder_output(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(f),
                        black_box(t),
                    )
                })
            },
        );

        // Benchmark scalar version for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", features, time_steps)),
            &(features, time_steps),
            |bench, &(f, t)| {
                bench.iter(|| {
                    // Scalar transpose implementation
                    for time in 0..t {
                        for feat in 0..f {
                            let src_idx = feat * t + time;
                            let dst_idx = time * f + feat;
                            output[dst_idx] = input[src_idx];
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark GEMM operations for RNN-T decoder
fn benchmark_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm");

    // Matrix sizes typical for RNN-T decoder operations
    let sizes = vec![
        (16, 16, 16),    // Very small matrices
        (32, 32, 32),    // Small matrices
        (64, 64, 64),    // Medium matrices
        (128, 128, 128), // Large matrices (approaching BLAS territory)
    ];

    for (m, n, k) in sizes {
        let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..(k * n)).map(|i| (i + 1) as f32 * 0.01).collect();
        let mut c = vec![0.0; m * n];

        group.throughput(Throughput::Elements((m * n * k) as u64));

        group.bench_with_input(
            BenchmarkId::new("optimized", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    gemm_f32_optimized(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                        black_box(m),
                        black_box(n),
                        black_box(k),
                    )
                })
            },
        );

        // Benchmark scalar version for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |bench, &(m, n, k)| {
                bench.iter(|| {
                    // Scalar GEMM implementation
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = c[i * n + j];
                            for l in 0..k {
                                sum += a[i * k + l] * b[l * n + j];
                            }
                            c[i * n + j] = sum;
                        }
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark argmax operations for logits processing
fn benchmark_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax");

    // Vocabulary sizes typical for ASR models
    let sizes = vec![
        100,   // Small vocabulary
        1000,  // Medium vocabulary
        5000,  // Large vocabulary
        10000, // Very large vocabulary
    ];

    for size in sizes {
        let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |bench, _| {
            bench.iter(|| {
                let (idx, val) = argmax_optimized(black_box(&logits));
                black_box((idx, val))
            })
        });

        // Benchmark scalar version for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                let mut max_idx = 0;
                let mut max_val = logits[0];

                for (i, &val) in logits.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }

                black_box((max_idx, max_val))
            })
        });
    }

    group.finish();
}

/// Benchmark audio conversion (should show compiler auto-vectorization is optimal)
fn benchmark_audio_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_conversion");

    // Audio chunk sizes typical for streaming ASR
    let sizes = vec![
        1024,  // Small chunk (64ms at 16kHz)
        4096,  // Medium chunk (256ms at 16kHz)
        16384, // Large chunk (1s at 16kHz)
        65536, // Very large chunk (4s at 16kHz)
    ];

    for size in sizes {
        let audio_bytes: Vec<u8> = (0..(size * 2)).map(|i| (i % 256) as u8).collect();
        let mut output = Vec::new();

        group.throughput(Throughput::Bytes((size * 2) as u64));

        group.bench_with_input(BenchmarkId::new("optimized", size), &size, |bench, _| {
            bench.iter(|| {
                output.clear();
                bytes_to_f32_optimized(black_box(&audio_bytes), black_box(&mut output))
            })
        });

        // Benchmark simple scalar version for comparison
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                output.clear();
                for chunk in audio_bytes.chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    output.push(sample as f32 / 32768.0);
                }
            })
        });
    }

    group.finish();
}

/// Comprehensive RNN-T pipeline benchmark
fn benchmark_rnnt_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnnt_pipeline");

    // Simulate a complete RNN-T decoder step
    let features = 512;
    let time_steps = 128;
    let vocab_size = 1000;

    // Encoder output (features x time_steps)
    let encoder_output: Vec<f32> = (0..(features * time_steps))
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    // Decoder weight matrix (features x vocab_size)
    let decoder_weights: Vec<f32> = (0..(features * vocab_size))
        .map(|i| (i as f32 * 0.0001).cos())
        .collect();

    // Bias vector
    let bias: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.01).collect();

    let mut transposed_output = vec![0.0; time_steps * features];
    let mut logits = vec![0.0; vocab_size];

    group.throughput(Throughput::Elements((features * time_steps) as u64));

    group.bench_function("optimized_pipeline", |b| {
        b.iter(|| {
            // Step 1: Transpose encoder output (features x time_steps -> time_steps x features)
            transpose_encoder_output(
                black_box(&encoder_output),
                black_box(&mut transposed_output),
                black_box(features),
                black_box(time_steps),
            );

            // Step 2: Matrix multiplication for first time step (1 x features) * (features x vocab_size)
            let frame = &transposed_output[0..features];
            gemm_f32_optimized(
                black_box(frame),
                black_box(&decoder_weights),
                black_box(&mut logits),
                black_box(1),
                black_box(vocab_size),
                black_box(features),
            );

            // Step 3: Add bias and find argmax
            for (logit, &bias_val) in logits.iter_mut().zip(bias.iter()) {
                *logit += bias_val;
            }

            let (predicted_token, confidence) = argmax_optimized(black_box(&logits));
            black_box((predicted_token, confidence))
        })
    });

    group.bench_function("scalar_pipeline", |b| {
        b.iter(|| {
            // Step 1: Scalar transpose
            for t in 0..time_steps {
                for f in 0..features {
                    let src_idx = f * time_steps + t;
                    let dst_idx = t * features + f;
                    transposed_output[dst_idx] = encoder_output[src_idx];
                }
            }

            // Step 2: Scalar matrix multiplication
            let frame = &transposed_output[0..features];
            for i in 0..vocab_size {
                let mut sum = 0.0;
                for j in 0..features {
                    sum += frame[j] * decoder_weights[j * vocab_size + i];
                }
                logits[i] = sum;
            }

            // Step 3: Add bias and scalar argmax
            for (logit, &bias_val) in logits.iter_mut().zip(bias.iter()) {
                *logit += bias_val;
            }

            let mut max_idx = 0;
            let mut max_val = logits[0];
            for (i, &val) in logits.iter().enumerate().skip(1) {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            black_box((max_idx, max_val))
        })
    });

    group.finish();
}

criterion_group!(
    simd_benches,
    benchmark_tensor_transpose,
    benchmark_gemm,
    benchmark_argmax,
    benchmark_audio_conversion,
    benchmark_rnnt_pipeline
);

criterion_main!(simd_benches);
