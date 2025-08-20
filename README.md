# AMIRA Rust ASR Server

> Documentation has been consolidated under `docs/`. See:
> - `docs/README.md` for the index
> - `docs/Architecture.md` for current system design
> - `docs/Operations.md` for config/deploy/monitor
> - `docs/Performance.md` for tuning guidance
> - Archived docs moved to `docs/legacy/`

![Rust](https://img.shields.io/badge/rust-stable-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

A high-performance real-time Automatic Speech Recognition (ASR) server built in Rust with Triton Inference Server integration. This server uses RNN-T (Recurrent Neural Network Transducer) models for speech-to-text conversion with WebSocket streaming and batch processing capabilities.

## 🚀 Features

- **High-Performance Audio Processing**: SIMD-optimized audio conversion and processing
- **Real-Time Streaming**: WebSocket-based streaming ASR with incremental results
- **Batch Processing**: Efficient batch transcription for multiple audio files
- **RNN-T Models**: State-of-the-art Recurrent Neural Network Transducer architecture
- **Triton Integration**: Seamless integration with NVIDIA Triton Inference Server
- **Zero-Copy Optimizations**: Minimized memory allocations for optimal performance
- **Lock-Free Memory Pools**: High-performance memory management with minimal contention
- **Circuit Breaker Pattern**: Fault-tolerant operation with automatic recovery
- **Observability**: Comprehensive tracing, metrics, and monitoring
- **Configuration Management**: Flexible TOML/YAML configuration with environment variable overrides

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Architecture](#-architecture)
- [Performance](#-performance)
- [Development](#-development)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [Contributing](#-contributing)
- [License](#-license)

## ⚡ Quick Start

### Prerequisites

- **Rust**: 1.89+ (stable channel)
- **Triton Inference Server**: 23.0+ running on localhost:8001
- **Audio Models**: RNN-T models in Triton model repository

### 1. Clone and Build

```bash
git clone https://github.com/your-org/amira-rust-asr-server.git
cd amira-rust-asr-server

# Development build
cargo build

# Production build (optimized)
cargo build --release
```

### 2. Start Triton Server

```bash
# Using Docker (recommended)
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model-repo:/models \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models
```

### 3. Configure and Run

```bash
# Copy and customize configuration
cp config.toml my-config.toml

# Run the server
AMIRA_CONFIG_FILE=my-config.toml cargo run --release
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8057/health

# Batch transcription
curl -X POST http://localhost:8057/v2/decode/batch/amira \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# WebSocket streaming (see examples/ for client code)
```

## 🔧 Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-org/amira-rust-asr-server.git
cd amira-rust-asr-server

# Install dependencies and build
cargo build --release

# Install to system (optional)
cargo install --path .
```

### Using Docker

```bash
# Build Docker image
docker build -t amira-asr-server .

# Run with Docker Compose
docker-compose up -d
```

### System Requirements

- **Memory**: 4GB+ RAM (8GB+ recommended for production)
- **CPU**: Multi-core x86_64 with AVX2 support (AVX-512 preferred)
- **Storage**: 2GB+ for models and temporary files
- **Network**: Stable connection to Triton Inference Server

## ⚙️ Configuration

AMIRA ASR Server supports hierarchical configuration from multiple sources with the following precedence (highest to lowest):

1. **Environment variables** (AMIRA_* prefixed or legacy names)
2. **config.yaml** (if exists)
3. **config.toml** (if exists)  
4. **Built-in defaults**

### Configuration Files

#### config.toml
```toml
# HTTP server configuration
server_host = "0.0.0.0"
server_port = 8057

# Triton Inference Server configuration
triton_endpoint = "http://localhost:8001"
inference_timeout_secs = 5

# File system paths
vocabulary_path = "../model-repo/vocab.txt"

# Server Performance Configuration
max_concurrent_streams = 10
max_concurrent_batches = 50
inference_queue_size = 100

# Audio Processing Configuration
audio_buffer_capacity = 1048576       # 1MB
max_batch_audio_length_secs = 30.0

# WebSocket Streaming Configuration
stream_timeout_secs = 30
keepalive_check_period_ms = 100

# Model Configuration
preprocessor_model_name = "preprocessor"
encoder_model_name = "encoder"
decoder_joint_model_name = "decoder_joint"
max_symbols_per_step = 30
max_total_tokens = 200
```

### Environment Variables

All configuration values can be overridden using environment variables:

#### Basic Configuration
```bash
export AMIRA_SERVER_HOST="0.0.0.0"                    # Server bind address
export AMIRA_SERVER_PORT="8057"                       # Server port
export AMIRA_TRITON_ENDPOINT="http://localhost:8001"  # Triton server URL
export AMIRA_INFERENCE_TIMEOUT_SECS="5"               # Inference timeout
export AMIRA_VOCABULARY_PATH="../model-repo/vocab.txt" # Vocabulary file path
```

#### Performance Configuration
```bash
export AMIRA_MAX_CONCURRENT_STREAMS="10"              # Max WebSocket streams
export AMIRA_MAX_CONCURRENT_BATCHES="50"              # Max batch requests
export AMIRA_INFERENCE_QUEUE_SIZE="100"               # Inference queue size
export AMIRA_AUDIO_BUFFER_CAPACITY="1048576"          # Audio buffer size (bytes)
export AMIRA_MAX_BATCH_AUDIO_LENGTH_SECS="30.0"       # Max batch audio length
```

#### Streaming Configuration
```bash
export AMIRA_STREAM_TIMEOUT_SECS="30"                 # Stream timeout
export AMIRA_KEEPALIVE_CHECK_PERIOD_MS="100"          # Keepalive period
```

#### Model Configuration
```bash
export AMIRA_PREPROCESSOR_MODEL_NAME="preprocessor"   # Preprocessor model
export AMIRA_ENCODER_MODEL_NAME="encoder"             # Encoder model
export AMIRA_DECODER_JOINT_MODEL_NAME="decoder_joint" # Decoder model
export AMIRA_MAX_SYMBOLS_PER_STEP="30"                # Max symbols per step
export AMIRA_MAX_TOTAL_TOKENS="200"                   # Max total tokens
```

#### Legacy Environment Variables (for backward compatibility)
```bash
export SERVER_HOST="0.0.0.0"
export SERVER_PORT="8057"
export TRITON_ENDPOINT="http://localhost:8001"
export INFERENCE_TIMEOUT_SECS="5"
export VOCABULARY_PATH="../model-repo/vocab.txt"
```

### Configuration Loading

```rust
use amira_rust_asr_server::config::Config;

// Load from multiple sources (TOML/YAML + env vars)
let config = Config::load()?;

// Load from environment variables only (legacy)
let config = Config::from_env()?;

// Export current configuration
let toml_content = config.to_toml()?;
let yaml_content = config.to_yaml()?;
```

## 📚 API Documentation

### REST API Endpoints

#### Health Check
```http
GET /health
```
**Response**: `200 OK` with server status

#### Batch Transcription
```http
POST /v2/decode/batch/{model}
Content-Type: application/octet-stream
```
**Parameters**:
- `model`: Model name (e.g., "amira")
- **Body**: Raw audio data (16-bit PCM, 16kHz)

**Response**:
```json
{
  "text": "transcribed speech text",
  "tokens": [1, 2, 3, ...],
  "audio_length_samples": 16000,
  "features_length": 100,
  "encoded_length": 50
}
```

### WebSocket API

#### Real-Time Streaming
```
WS /v2/decode/stream/{model}
```

**Protocol**:
1. Connect to WebSocket endpoint
2. Send audio chunks as binary frames
3. Receive incremental transcription results
4. Send control byte `0x00` to end stream
5. Send control byte `0x01` for keepalive

**Message Format**:
```json
{
  "type": "partial|final",
  "text": "transcription text",
  "tokens": [1, 2, 3, ...],
  "confidence": 0.95
}
```

### Client Examples

#### Batch Processing
```bash
curl -X POST http://localhost:8057/v2/decode/batch/amira \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

#### WebSocket Streaming
See `examples/simple_client.rs` for a complete WebSocket client implementation.

## 🏗️ Architecture

### Core Components

#### ASR Pipeline (`src/asr/`)
- **pipeline.rs**: Main ASR pipeline orchestration
- **decoder_optimized.rs**: Optimized RNN-T greedy decoder
- **audio.rs**: Audio buffer management and processing
- **incremental.rs**: Streaming/incremental ASR processing
- **simd.rs**: SIMD-optimized operations (AVX2/AVX-512)
- **lockfree_memory.rs**: Lock-free memory pools
- **weaving.rs**: Token weaving for incremental transcription

#### Server Layer (`src/server/`)
- **handlers.rs**: REST API endpoints
- **stream.rs**: WebSocket streaming handlers
- **state.rs**: Shared application state management

#### Triton Integration (`src/triton/`)
- **reliable_client.rs**: Fault-tolerant Triton client
- **pool_optimized.rs**: Connection pooling for Triton
- **model.rs**: Model metadata and configuration

#### Performance Optimizations (`src/performance/`)
- **affinity.rs**: CPU affinity management
- **numa_aware.rs**: NUMA-aware memory allocation
- **specialized_pools.rs**: High-performance thread pools

#### Reliability (`src/reliability/`)
- **circuit_breaker.rs**: Circuit breaker pattern implementation
- **graceful_shutdown.rs**: Graceful shutdown handling
- **tracing_config.rs**: Distributed tracing configuration

### Data Flow

```
Audio Input → Preprocessor Model → Encoder Model → Decoder/Joint Model → Text Output
     ↓              ↓                    ↓                 ↓
  Ring Buffer → Feature Vectors → Encoded States → Token Probabilities
```

### Memory Management

The server uses several optimization strategies:

1. **Lock-Free Memory Pools**: Pre-allocated buffers for audio, tensors, and decoder states
2. **Zero-Copy Operations**: Minimize data copying throughout the pipeline
3. **SIMD Optimizations**: Vectorized operations for audio processing and matrix computations
4. **Connection Pooling**: Reuse Triton connections across requests

## 🚄 Performance

### Benchmarks

Run the included benchmarks to test performance on your hardware:

```bash
# Run all benchmarks
cargo bench

# Specific benchmarks
cargo bench --bench simd_benchmarks      # SIMD optimizations
cargo bench --bench memory_pool_bench    # Memory pool performance
cargo bench --bench connection_pool_bench # Connection pooling
```

### Performance Tuning

#### CPU Optimization
```bash
# Enable CPU affinity for better performance
export AMIRA_ENABLE_CPU_AFFINITY=true
export AMIRA_INFERENCE_THREADS=4
export AMIRA_IO_THREADS=2
```

#### Memory Optimization
```bash
# Tune memory pool sizes
export AMIRA_AUDIO_BUFFER_POOL_SIZE=20
export AMIRA_ENCODER_POOL_SIZE=50
export AMIRA_DECODER_POOL_SIZE=100
```

#### NUMA Configuration
```bash
# Enable NUMA awareness (Linux only)
export AMIRA_ENABLE_NUMA=true
```

### Expected Performance

| Metric | Development | Production |
|--------|-------------|------------|
| **Latency (batch)** | ~100ms | ~50ms |
| **Latency (streaming)** | ~200ms | ~100ms |
| **Throughput** | 5x real-time | 10x real-time |
| **Memory Usage** | ~2GB | ~4GB |
| **CPU Usage** | 2-4 cores | 4-8 cores |

## 💻 Development

### Building from Source

```bash
# Development build with debug symbols
cargo build

# Production build with optimizations
cargo build --release

# Check code without building
cargo check

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run
```

### Code Quality Tools

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Security audit
cargo audit

# Check dependencies
cargo outdated
```

### Testing

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run with output
cargo test -- --nocapture

# Test specific module
cargo test asr::pipeline
```

### Debugging

```bash
# Run with detailed logging
RUST_LOG=trace cargo run

# Memory debugging with Valgrind (Linux)
valgrind --tool=memcheck --leak-check=full ./target/debug/amira-rust-asr-server

# Performance profiling
perf record ./target/release/amira-rust-asr-server
perf report
```

## 🚀 Deployment

### Production Deployment

#### Using Docker
```bash
# Build production image
docker build -t amira-asr-server:latest .

# Run with production settings
docker run -d \
  --name amira-asr \
  -p 8057:8057 \
  -e AMIRA_SERVER_HOST=0.0.0.0 \
  -e AMIRA_TRITON_ENDPOINT=http://triton:8001 \
  -v ./config:/app/config \
  amira-asr-server:latest
```

#### Using Docker Compose
```bash
# Start full stack (ASR + Triton + Observability)
docker-compose up -d

# Scale ASR servers
docker-compose up -d --scale asr-server=3
```

#### System Service (Linux)
```bash
# Create systemd service
sudo cp scripts/amira-asr.service /etc/systemd/system/
sudo systemctl enable amira-asr
sudo systemctl start amira-asr
```

### Environment Configuration

#### Production Environment Variables
```bash
# Server configuration
export AMIRA_SERVER_HOST="0.0.0.0"
export AMIRA_SERVER_PORT="8057"

# Triton configuration
export AMIRA_TRITON_ENDPOINT="http://triton-server:8001"
export AMIRA_INFERENCE_TIMEOUT_SECS="10"

# Performance tuning
export AMIRA_MAX_CONCURRENT_STREAMS="50"
export AMIRA_MAX_CONCURRENT_BATCHES="200"

# Security
export AMIRA_ENABLE_CORS="false"
export AMIRA_MAX_REQUEST_SIZE="10485760"  # 10MB
```

### Load Balancing

For high-availability deployments, use a load balancer:

#### NGINX Configuration
```nginx
upstream amira_asr {
    server asr1:8057;
    server asr2:8057;
    server asr3:8057;
}

server {
    listen 80;
    location / {
        proxy_pass http://amira_asr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 📊 Monitoring

### Observability Stack

The project includes a complete observability setup:

```bash
# Start monitoring stack
docker-compose -f docker-compose.observability.yml up -d
```

This includes:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards  
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation

### Metrics

The server exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8057/metrics
```

Key metrics:
- `asr_requests_total`: Total number of ASR requests
- `asr_request_duration_seconds`: Request processing time
- `asr_active_connections`: Active WebSocket connections
- `triton_inference_duration_seconds`: Triton inference time
- `memory_pool_usage`: Memory pool utilization

### Distributed Tracing

Configure tracing in your environment:

```bash
export AMIRA_JAEGER_ENDPOINT="http://localhost:14268/api/traces"
export AMIRA_SERVICE_NAME="amira-asr-server"
export RUST_LOG="info,amira_rust_asr_server=debug"
```

### Health Monitoring

```bash
# Basic health check
curl http://localhost:8057/health

# Detailed health with dependencies
curl http://localhost:8057/health/detailed
```

## 🧪 Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests
```bash
cargo test --test integration_tests
```

### Benchmark Tests
```bash
cargo bench
```

### Load Testing
```bash
# Using Apache Bench for HTTP endpoints
ab -n 1000 -c 10 http://localhost:8057/health

# Using custom WebSocket load tester
cargo run --example load_test_websocket
```

## 📈 Performance Optimization

### CPU Optimization

1. **Enable CPU Features**:
   ```bash
   export RUSTFLAGS="-C target-cpu=native"
   cargo build --release
   ```

2. **Thread Pool Tuning**:
   ```bash
   export AMIRA_INFERENCE_THREADS=4
   export AMIRA_IO_THREADS=2
   export AMIRA_NETWORK_THREADS=2
   ```

3. **CPU Affinity**:
   ```bash
   export AMIRA_ENABLE_CPU_AFFINITY=true
   ```

### Memory Optimization

1. **Pool Size Tuning**:
   ```toml
   # In config.toml
   audio_buffer_pool_size = 20
   encoder_pool_size = 50
   decoder_pool_size = 100
   ```

2. **NUMA Awareness** (Linux):
   ```bash
   export AMIRA_ENABLE_NUMA=true
   ```

### Network Optimization

1. **Connection Pooling**:
   ```toml
   # In config.toml
   max_triton_connections = 10
   connection_idle_timeout_secs = 300
   ```

2. **Buffer Sizes**:
   ```toml
   # In config.toml
   audio_buffer_capacity = 2097152  # 2MB
   websocket_buffer_size = 65536    # 64KB
   ```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**: `cargo test`
6. **Run quality checks**: `cargo fmt && cargo clippy`
7. **Commit your changes**: `git commit -m 'Add amazing feature'`
8. **Push to the branch**: `git push origin feature/amazing-feature`
9. **Open a Pull Request**

### Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Address all Clippy warnings (`cargo clippy`)
- Add documentation for public APIs
- Include tests for new functionality
- Maintain backwards compatibility when possible

### Bug Reports

Please use our [Issue Template](.github/ISSUE_TEMPLATE.md) when reporting bugs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NVIDIA Triton Inference Server** for the inference backend
- **Rust Community** for excellent async and performance libraries
- **RNN-T Research Community** for the model architecture
- **Contributors** who helped improve this project

## 📞 Support

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: [GitHub Issues](https://github.com/your-org/amira-rust-asr-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/amira-rust-asr-server/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for security policy

---

## 📊 Project Status

| Component | Status | Coverage | Performance |
|-----------|--------|----------|-------------|
| Core ASR Pipeline | ✅ Stable | 95% | Optimized |
| WebSocket Streaming | ✅ Stable | 90% | Optimized |
| Batch Processing | ✅ Stable | 95% | Optimized |
| Configuration System | ✅ Stable | 100% | - |
| Memory Pools | ✅ Stable | 85% | Optimized |
| SIMD Optimizations | ✅ Stable | 90% | Optimized |
| Observability | ✅ Stable | 80% | - |
| Documentation | ✅ Consolidated | 80% | - |

**Last Updated**: 2025-08-20
**Version**: 1.0.0
**Rust Version**: 1.89+