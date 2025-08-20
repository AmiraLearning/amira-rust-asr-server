# Contributor Guide (AI Assistants)

This file provides quick guidance to AI code assistants working in this repository. For canonical docs, see `docs/`.

## Development Commands

### Building and Running
- `cargo build` - Build the project
- `cargo build --release` - Build optimized release version
- `cargo run` - Run the main ASR server
- `cargo run --bin test-connection-pool` - Run connection pool test binary

### Testing and Benchmarking
- `cargo test` - Run all tests
- `cargo bench` - Run all benchmarks
- `cargo bench --bench phase1_benchmarks` - Run phase 1 benchmarks
- `cargo bench --bench simple_bench` - Run simple benchmarks
- `cargo bench --bench connection_pool_bench` - Run connection pool benchmarks
- `cargo bench --bench memory_pool_bench` - Run memory pool benchmarks

### Code Quality
- `cargo check` - Check code for errors without building
- `cargo clippy` - Run Rust linter for code quality
- `cargo fmt` - Format code using rustfmt

### Development with Docker
- `docker-compose up` - Start all services (ASR server, Triton, Redis, Prometheus)
- `docker-compose up triton` - Start only Triton Inference Server
- `docker-compose up -d` - Start services in detached mode

## Architecture Overview (Summary)

This is a high-performance ASR (Automatic Speech Recognition) server using RNN-T models via Triton Inference Server.

### Core Components

1. **ASR Pipeline** (`src/asr/`): Core speech recognition functionality
   - `pipeline.rs`: Main ASR pipeline orchestrating the entire process
   - `decoder.rs`/`decoder_optimized.rs`: Beam search and greedy decoding algorithms
   - `audio.rs`: Audio processing utilities and ring buffers
   - `simd*.rs`: SIMD-optimized audio processing functions
   - `memory.rs`/`lockfree_memory.rs`: Memory pool management for performance
   - `zero_copy.rs`: Zero-copy tensor operations

2. **Triton Integration** (`src/triton/`): gRPC client for Triton Inference Server
   - `client.rs`: Basic Triton gRPC client
   - `reliable_client.rs`: Fault-tolerant client with retry logic
   - `pool.rs`: Connection pooling for high throughput
   - `model.rs`: Model-specific input/output handling (encoder, decoder, preprocessor)

3. **Web Server** (`src/server/`): HTTP/WebSocket API
   - `handlers.rs`: HTTP endpoints and WebSocket handlers
   - `stream.rs`: Real-time audio stream processing
   - `state.rs`: Shared application state management

4. **Performance Optimization** (`src/performance/`):
   - `affinity.rs`: CPU affinity management
   - `numa_aware.rs`: NUMA-aware memory allocation
   - `specialized_pools.rs`: High-performance object pools

5. **Reliability** (`src/reliability/`):
   - `circuit_breaker.rs`: Circuit breaker pattern for fault tolerance
   - `graceful_shutdown.rs`: Clean shutdown handling
   - `metrics.rs`: Application metrics collection

### Key Performance Features
- SIMD-optimized audio processing using `wide` crate
- Lock-free memory pools for zero-allocation hot paths
- Connection pooling to Triton Inference Server
- CPU affinity and NUMA-aware allocation
- io_uring support on Linux for high-performance I/O

### Model Pipeline
The ASR pipeline processes audio through three Triton models:
1. **Preprocessor**: Converts raw audio to features
2. **Encoder**: RNN-T encoder for acoustic modeling  
3. **Decoder/Joint**: RNN-T decoder and joint network for text generation

### Configuration
- Server configuration via environment variables (see `src/config.rs`)
- Triton endpoint: `TRITON_ENDPOINT` (default: http://localhost:8001)
- Vocabulary file: `VOCABULARY_PATH` (default: ./vocab.txt)
- Server binding: `SERVER_HOST` and `SERVER_PORT`

### Development Notes
- Proto files in `proto/` are compiled via `build.rs` using `tonic-build`
- Model repository structure expected in `model-repo/` for Triton
- Benchmarks use `criterion` framework with HTML reports
- Observability stack included: Prometheus metrics, Grafana dashboards