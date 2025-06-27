//! Memory pool performance benchmark.
//!
//! This benchmark measures the performance improvements from using memory pools
//! vs raw allocations, which is a key optimization in our ASR pipeline.

use amira_rust_asr_server::asr::global_pools;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor Operations");

    for size in [1024, 4096, 16384] {
        // Benchmark raw allocation pattern (what we had before)
        group.bench_with_input(
            BenchmarkId::new("Raw Allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    // Simulate the old decoder pattern
                    let mut encoder_frame = Vec::with_capacity(size);
                    for i in 0..size {
                        encoder_frame.push(i as f32 * 0.001);
                    }

                    let mut targets = Vec::with_capacity(200);
                    targets.push(1024); // BLANK_TOKEN_ID
                    for i in 0..10 {
                        targets.push(i);
                    }

                    black_box((encoder_frame.len(), targets.len()));
                });
            },
        );

        // Benchmark memory pool pattern (optimized)
        group.bench_with_input(BenchmarkId::new("Memory Pool", size), &size, |b, &size| {
            b.iter(|| {
                // Simulate the new pooled decoder pattern
                let mut encoder_frame_buffer = global_pools().encoder_inputs.get();
                encoder_frame_buffer.clear();
                encoder_frame_buffer.reserve(size);
                for i in 0..size {
                    encoder_frame_buffer.push(i as f32 * 0.001);
                }

                let mut targets_buffer = global_pools().decoder_targets.get();
                targets_buffer.clear();
                targets_buffer.push(1024); // BLANK_TOKEN_ID
                for i in 0..10 {
                    targets_buffer.push(i);
                }

                black_box((encoder_frame_buffer.len(), targets_buffer.len()));
            });
        });
    }

    group.finish();
}

fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Allocation Patterns");

    // Heavy allocation scenario (decoder hot loop simulation)
    group.bench_function("Raw Vec Creation Loop", |b| {
        b.iter(|| {
            let mut total = 0;
            for _ in 0..100 {
                let mut vec: Vec<f32> = Vec::with_capacity(1024);
                for i in 0..1024 {
                    vec.push(i as f32);
                }
                total += vec.len();
            }
            black_box(total);
        });
    });

    group.bench_function("Memory Pool Reuse Loop", |b| {
        b.iter(|| {
            let mut total = 0;
            for _ in 0..100 {
                let mut buffer = global_pools().encoder_inputs.get();
                buffer.clear();
                for i in 0..1024 {
                    buffer.push(i as f32);
                }
                total += buffer.len();
            }
            black_box(total);
        });
    });

    group.finish();
}

fn bench_decoder_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decoder Simulation");

    // Simulate the RNN-T decoder inner loop
    group.bench_function("Original Decoder Pattern", |b| {
        b.iter(|| {
            let mut total_operations = 0;

            // Simulate 10 time steps
            for t in 0..10 {
                // Encoder frame extraction (1024 features)
                let mut encoder_frame = Vec::with_capacity(1024);
                for i in 0..1024 {
                    encoder_frame.push((t * 1024 + i) as f32 * 0.001);
                }

                // Simulate inner decoder loop (up to 5 symbols per step)
                for s in 0..5 {
                    let mut current_targets = vec![1024]; // BLANK_TOKEN_ID
                    for tok in 0..s {
                        current_targets.push(tok as i32);
                    }

                    total_operations += encoder_frame.len() + current_targets.len();
                }
            }

            black_box(total_operations);
        });
    });

    group.bench_function("Optimized Decoder Pattern", |b| {
        b.iter(|| {
            let mut total_operations = 0;

            // Simulate 10 time steps
            for t in 0..10 {
                // Pooled encoder frame extraction
                let mut encoder_frame_buffer = global_pools().encoder_inputs.get();
                encoder_frame_buffer.clear();
                encoder_frame_buffer.reserve(1024);
                for i in 0..1024 {
                    encoder_frame_buffer.push((t * 1024 + i) as f32 * 0.001);
                }

                // Simulate inner decoder loop with pooled targets
                for s in 0..5 {
                    let mut targets_buffer = global_pools().decoder_targets.get();
                    targets_buffer.clear();
                    targets_buffer.push(1024); // BLANK_TOKEN_ID
                    for tok in 0..s {
                        targets_buffer.push(tok as i32);
                    }

                    total_operations += encoder_frame_buffer.len() + targets_buffer.len();
                }
            }

            black_box(total_operations);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_operations,
    bench_allocation_patterns,
    bench_decoder_simulation
);
criterion_main!(benches);
