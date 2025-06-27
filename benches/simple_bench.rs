//! Simple benchmark to isolate performance issues.

use amira_rust_asr_server::asr::{bytes_to_f32_samples, simd};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_simple_conversion(c: &mut Criterion) {
    // Small test case - 1000 samples (2000 bytes)
    let audio_bytes: Vec<u8> = (0..2000).map(|i| (i % 256) as u8).collect();

    c.bench_function("original_small", |b| {
        b.iter(|| {
            let result = bytes_to_f32_samples(black_box(&audio_bytes));
            black_box(result.len());
        });
    });

    c.bench_function("simd_small", |b| {
        b.iter(|| {
            let mut result = Vec::new();
            simd::bytes_to_f32_optimized(black_box(&audio_bytes), &mut result);
            black_box(result.len());
        });
    });

    c.bench_function("scalar_baseline", |b| {
        b.iter(|| {
            let mut result = Vec::new();
            for chunk in audio_bytes.chunks_exact(2) {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                result.push(sample as f32 / 32768.0);
            }
            black_box(result.len());
        });
    });
}

criterion_group!(benches, bench_simple_conversion);
criterion_main!(benches);
