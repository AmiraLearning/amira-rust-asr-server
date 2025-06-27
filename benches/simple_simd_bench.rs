//! Simple SIMD Validation Benchmark

use amira_rust_asr_server::asr::simd::{
    argmax_optimized, gemm_f32_optimized, transpose_encoder_output,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_transpose_simple(c: &mut Criterion) {
    let features = 512;
    let time_steps = 128;
    let input: Vec<f32> = (0..(features * time_steps))
        .map(|i| i as f32 * 0.1)
        .collect();
    let mut output = vec![0.0; time_steps * features];

    c.bench_function("transpose_optimized", |b| {
        b.iter(|| {
            transpose_encoder_output(
                black_box(&input),
                black_box(&mut output),
                black_box(features),
                black_box(time_steps),
            )
        })
    });

    c.bench_function("transpose_scalar", |b| {
        b.iter(|| {
            for t in 0..time_steps {
                for f in 0..features {
                    let src_idx = f * time_steps + t;
                    let dst_idx = t * features + f;
                    output[dst_idx] = input[src_idx];
                }
            }
        })
    });
}

fn benchmark_gemm_simple(c: &mut Criterion) {
    let m = 64;
    let n = 64;
    let k = 64;
    let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i + 1) as f32 * 0.01).collect();
    let mut c_matrix = vec![0.0; m * n];

    c.bench_function("gemm_optimized", |bench| {
        bench.iter(|| {
            gemm_f32_optimized(
                black_box(&a),
                black_box(&b),
                black_box(&mut c_matrix),
                black_box(m),
                black_box(n),
                black_box(k),
            )
        })
    });
}

fn benchmark_argmax_simple(c: &mut Criterion) {
    let size = 1000;
    let logits: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();

    c.bench_function("argmax_optimized", |b| {
        b.iter(|| {
            let (idx, val) = argmax_optimized(black_box(&logits));
            black_box((idx, val))
        })
    });

    c.bench_function("argmax_scalar", |b| {
        b.iter(|| {
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

criterion_group!(
    simple_simd_benches,
    benchmark_transpose_simple,
    benchmark_gemm_simple,
    benchmark_argmax_simple
);

criterion_main!(simple_simd_benches);
