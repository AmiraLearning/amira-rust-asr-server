//! Phase 1 performance benchmarks for connection pooling, memory pools, and SIMD.

use amira_rust_asr_server::asr::{
    bytes_to_f32_samples, bytes_to_f32_samples_into, calculate_mean_amplitude, global_pools, simd,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint;

fn bench_audio_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Audio Conversion");

    // Test with different audio sizes (1 second to 30 seconds at 16kHz)
    let sizes = vec![16000, 32000, 64000, 160000, 480000]; // 1s, 2s, 4s, 10s, 30s

    for size in sizes {
        let audio_bytes: Vec<u8> = (0..size * 2).map(|i| (i % 256) as u8).collect();

        // Benchmark original implementation
        group.bench_with_input(
            BenchmarkId::new("Original", size),
            &audio_bytes,
            |b, data| {
                b.iter(|| {
                    let result = bytes_to_f32_samples(black_box(data));
                    hint::black_box(result);
                });
            },
        );

        // Benchmark SIMD optimized implementation
        group.bench_with_input(
            BenchmarkId::new("SIMD Optimized", size),
            &audio_bytes,
            |b, data| {
                b.iter(|| {
                    let mut result = Vec::new();
                    simd::bytes_to_f32_optimized(black_box(data), &mut result);
                    hint::black_box(result);
                });
            },
        );

        // Benchmark pooled memory implementation
        group.bench_with_input(
            BenchmarkId::new("Memory Pooled", size),
            &audio_bytes,
            |b, data| {
                b.iter(|| {
                    let mut buffer = global_pools().audio_buffers.get();
                    bytes_to_f32_samples_into(black_box(data), &mut buffer);
                    hint::black_box(buffer.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_mean_amplitude(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mean Amplitude");

    let sizes = vec![16000, 64000, 160000, 480000]; // 1s, 4s, 10s, 30s

    for size in sizes {
        let audio_samples: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();

        // Benchmark scalar implementation
        group.bench_with_input(
            BenchmarkId::new("Scalar", size),
            &audio_samples,
            |b, data| {
                b.iter(|| {
                    // Manual scalar implementation for comparison
                    let result = if data.is_empty() {
                        0.0
                    } else {
                        data.iter().map(|x| x.abs()).sum::<f32>() / data.len() as f32
                    };
                    hint::black_box(result);
                });
            },
        );

        // Benchmark SIMD optimized implementation
        group.bench_with_input(
            BenchmarkId::new("SIMD Optimized", size),
            &audio_samples,
            |b, data| {
                b.iter(|| {
                    let result = calculate_mean_amplitude(black_box(data));
                    hint::black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation");

    // Benchmark raw Vec allocation vs memory pool
    group.bench_function("Raw Vec::new", |b| {
        b.iter(|| {
            let v: Vec<f32> = Vec::with_capacity(16000);
            hint::black_box(v);
        });
    });

    group.bench_function("Memory Pool", |b| {
        b.iter(|| {
            let buffer = global_pools().audio_buffers.get();
            hint::black_box(buffer.len());
        });
    });

    // Benchmark allocation + deallocation cycles
    group.bench_function("Raw Vec Cycle", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut v: Vec<f32> = Vec::with_capacity(1000);
                v.resize(1000, 1.0);
                hint::black_box(&v);
            }
        });
    });

    group.bench_function("Memory Pool Cycle", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut buffer = global_pools().audio_buffers.get();
                buffer.clear();
                buffer.resize(1000, 1.0);
                hint::black_box(&*buffer);
            }
        });
    });

    group.finish();
}

fn bench_comprehensive_pipeline_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pipeline Simulation");

    // Simulate a typical ASR pipeline step with audio conversion + amplitude calculation
    let audio_bytes: Vec<u8> = (0..32000) // 2 seconds of audio
        .map(|i| ((i as f32 * 0.1).sin() * 32767.0) as i16)
        .flat_map(|sample| sample.to_le_bytes())
        .collect();

    group.bench_function("Original Pipeline", |b| {
        b.iter(|| {
            let samples = bytes_to_f32_samples(black_box(&audio_bytes));
            let amplitude = if samples.is_empty() {
                0.0
            } else {
                samples.iter().map(|x| x.abs()).sum::<f32>() / samples.len() as f32
            };
            hint::black_box((samples.len(), amplitude));
        });
    });

    group.bench_function("Optimized Pipeline", |b| {
        b.iter(|| {
            let mut buffer = global_pools().audio_buffers.get();
            bytes_to_f32_samples_into(black_box(&audio_bytes), &mut buffer);
            let amplitude = calculate_mean_amplitude(&buffer);
            hint::black_box((buffer.len(), amplitude));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_audio_conversion,
    bench_mean_amplitude,
    bench_memory_allocation,
    bench_comprehensive_pipeline_simulation
);
criterion_main!(benches);
