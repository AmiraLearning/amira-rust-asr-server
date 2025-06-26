# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is AMIRA Rust ASR Server - a high-performance real-time Automatic Speech Recognition server built in Rust with Triton Inference Server integration. The server uses RNN-T (Recurrent Neural Network Transducer) models for speech-to-text conversion with WebSocket streaming and batch processing capabilities.

## Common Development Commands

### Building and Running
```bash
# Development build
cargo build

# Release build (optimized for performance)
cargo build --release

# Run the server
cargo run --release

# Run with custom environment variables
TRITON_ENDPOINT=http://localhost:8001 VOCABULARY_PATH=./vocab.txt cargo run --release
```

### Testing
```bash
# Run all tests
cargo test

# Run integration tests specifically
cargo test --test integration_tests

# Run benchmarks
cargo bench

# Run performance benchmarks with criterion
cargo bench --bench asr_benchmarks
```

### Code Quality
```bash
# Format code
cargo fmt

# Lint code and check for issues
cargo clippy

# Update dependencies
cargo update
```

### Docker Operations
```bash
# Build Docker image
docker build -t amira-asr-server .

# Run with Docker Compose
docker-compose up
```

## Architecture Overview

### Core Components

- **ASR Module** (`src/asr/`): Core speech recognition functionality including:
  - `pipeline.rs`: Main ASR pipeline using Triton models
  - `decoder.rs`: Greedy decoding for RNN-T output
  - `incremental.rs`: Streaming/incremental ASR processing
  - `audio.rs`: Audio buffer management and processing utilities
  - `weaving.rs`: Token weaving for incremental transcription

- **Server Module** (`src/server/`): HTTP/WebSocket server implementation:
  - `handlers.rs`: REST API endpoints for batch processing
  - `stream.rs`: WebSocket handlers for real-time streaming
  - `state.rs`: Shared application state management
  - `metrics.rs`: Performance metrics collection

- **Triton Module** (`src/triton/`): Integration with Triton Inference Server:
  - `client.rs`: gRPC client for Triton communication
  - `model.rs`: Model metadata and configuration
  - `types.rs`: Triton-specific data structures

### Key Architecture Patterns

1. **Zero-Copy Optimizations**: Audio data is processed with minimal copying throughout the pipeline
2. **Async/Await**: Built on Tokio for high-concurrency handling
3. **State Management**: Uses Arc<T> for shared state across concurrent streams
4. **Resource Pooling**: Decoder states and buffers are reused for efficiency

### Model Pipeline Flow
```
Audio Input → Preprocessor Model → Encoder Model → Decoder/Joint Model → Text Output
```

## Configuration

The application uses environment variables for configuration (see `src/config.rs`):

- `TRITON_ENDPOINT`: Triton server URL (default: `http://localhost:8001`)
- `VOCABULARY_PATH`: Path to vocabulary file (default: `../model-repo/vocab.txt`)
- `SERVER_HOST`: Server bind address (default: `0.0.0.0`)
- `SERVER_PORT`: Server port (default: `8057`)
- `INFERENCE_TIMEOUT_SECS`: Inference timeout (default: `5`)

### Performance Constants (in `src/config.rs`)

- **Concurrency**: `MAX_CONCURRENT_STREAMS: 10`, `MAX_CONCURRENT_BATCHES: 50`
- **Audio**: `SAMPLE_RATE: 16000`, `BUFFER_CAPACITY: 1MB`
- **Models**: Three-stage pipeline (preprocessor, encoder, decoder_joint)

## API Endpoints

- `GET /health`: Health check
- `POST /v2/decode/batch/{model}`: Batch transcription
- `WS /v2/decode/stream/{model}`: Real-time streaming transcription

## Development Notes

- The crate name in code is `wav2vec2_server` (see `src/lib.rs:1` and imports in `src/main.rs:10`)
- Proto files are compiled at build time via `build.rs` using `tonic-build`
- Performance is critical - look for zero-copy patterns and async optimizations
- Triton models expect specific input/output tensor formats defined in `src/triton/types.rs`
- Streaming uses control bytes for end-of-stream and keepalive signals