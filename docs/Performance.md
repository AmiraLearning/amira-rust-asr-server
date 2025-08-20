# Performance Guide

## Build Flags

- Release: `cargo build --release`
- CPU: `RUSTFLAGS="-C target-cpu=native"`

## Runtime Tuning

- Specialized executors handle IO/Inference/Network
- Prefer spawn over block_on inside async contexts
- Configure thread counts via platform init recommendations

## Memory

- Use lock-free pools for hot buffers
- Avoid unnecessary allocations in hotpaths

## SIMD

- SIMD kernels auto-select via runtime detection
- Fallbacks exist; verify with benches

## Benchmarks

- `cargo bench` (SIMD, pools, connection pool)
- Track regressions over time; use criterion reports

## Cloud Environments

- Affinity may be disabled; NUMA reductions applied
- Epoll recommendations guarded by availability
