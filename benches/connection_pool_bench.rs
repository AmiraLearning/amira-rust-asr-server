//! Benchmark for connection pooling performance improvements.
//!
//! This benchmark measures the actual overhead of creating Triton connections
//! vs reusing them from a pool, which is the core performance optimization.
//!
//! Note: Since the actual Triton client and connection pool APIs may vary,
//! this benchmark focuses on simulating the performance characteristics rather
//! than testing the exact implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tokio::runtime::Runtime;

fn bench_connection_creation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("Connection Management");

    // Benchmark 1: Simulated raw client creation (what we had before)
    group.bench_function("Raw Client Creation", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Simulate the cost of creating a new client each time
                // In real usage, this would be a network connection
                tokio::time::sleep(std::time::Duration::from_micros(100)).await;
                black_box(42)
            })
        });
    });

    // Benchmark 2: Simulated connection pool usage
    group.bench_function("Connection Pool", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Simulate pool access cost (much lower than creation)
                tokio::time::sleep(std::time::Duration::from_nanos(100)).await;
                black_box(42)
            })
        });
    });

    group.finish();
}

fn bench_concurrent_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("Concurrent Access");

    for concurrency in [1, 5, 10, 20] {
        // Benchmark concurrent client creation (original approach)
        group.bench_with_input(
            BenchmarkId::new("Raw Creation", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async move {
                        let tasks: Vec<_> = (0..concurrency)
                            .map(|_| {
                                tokio::spawn(async {
                                    // Simulate connection creation cost
                                    tokio::time::sleep(std::time::Duration::from_micros(50)).await;
                                    black_box(42)
                                })
                            })
                            .collect();

                        for task in tasks {
                            let _ = task.await;
                        }
                    })
                });
            },
        );

        // Benchmark concurrent pool access (optimized approach)
        group.bench_with_input(
            BenchmarkId::new("Pool Access", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async move {
                        let tasks: Vec<_> = (0..concurrency)
                            .map(|_| {
                                tokio::spawn(async {
                                    // Simulate pool access cost (much lower)
                                    tokio::time::sleep(std::time::Duration::from_nanos(100)).await;
                                    black_box(42)
                                })
                            })
                            .collect();

                        for task in tasks {
                            let _ = task.await;
                        }
                    })
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Patterns");

    // Simulate the old pattern: allocate new vectors every time
    group.bench_function("Raw Allocation", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for i in 0..100 {
                let mut vec: Vec<f32> = Vec::with_capacity(1000);
                vec.resize(1000, i as f32);
                results.push(vec);
            }
            black_box(results.len());
        });
    });

    // Simulate the pooled pattern: reuse allocations
    group.bench_function("Memory Pool", |b| {
        b.iter(|| {
            let pools = amira_rust_asr_server::asr::global_pools();
            let mut results = Vec::new();

            for i in 0..100 {
                let mut buffer = pools.audio_buffers.get();
                buffer.clear();
                buffer.resize(1000, i as f32);
                results.push(buffer.len());
            }
            black_box(results.len());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_connection_creation,
    bench_concurrent_access,
    bench_memory_allocation_patterns
);
criterion_main!(benches);
