//! Domain constants for the ASR server.
//!
//! This module contains compile-time constants used throughout the application.
//! These are separated from runtime configuration to provide clear distinction
//! between values that never change and those that can be configured.

/// Audio processing constants.
pub mod audio {
    use crate::types::SampleRate;

    /// Standard sample rate for ASR processing.
    pub const SAMPLE_RATE: SampleRate = SampleRate::STANDARD_16KHZ;

    /// Buffer capacity for audio processing.
    pub const BUFFER_CAPACITY: usize = 1024 * 1024; // 1MB

    /// Maximum audio chunk size in samples.
    pub const MAX_CHUNK_SIZE: usize = 16000 * 10; // 10 seconds at 16kHz

    /// Minimum audio chunk size in samples.
    pub const MIN_CHUNK_SIZE: usize = 16000 / 10; // 0.1 seconds at 16kHz

    /// Maximum audio length for batch processing in seconds.
    pub const MAX_BATCH_AUDIO_LENGTH_SECS: u64 = 30;

    /// Minimum number of samples required for partial transcription.
    pub const MIN_PARTIAL_TRANSCRIPTION_SAMPLES: usize = 1600; // 0.1 seconds at 16kHz

    /// Default window size for STFT.
    pub const WINDOW_SIZE: usize = 512;

    /// Default hop size for STFT.
    pub const HOP_SIZE: usize = 256;

    /// Number of FFT bins.
    pub const FFT_SIZE: usize = 512;

    /// Number of mel filter banks.
    pub const MEL_FILTERS: usize = 80;

    /// Pre-emphasis coefficient for high-pass filtering.
    pub const PRE_EMPHASIS: f32 = 0.97;

    /// Maximum audio amplitude before clipping.
    pub const MAX_AMPLITUDE: f32 = 1.0;

    /// Minimum audio amplitude before clipping.
    pub const MIN_AMPLITUDE: f32 = -1.0;

    /// Silence threshold for VAD.
    pub const SILENCE_THRESHOLD: f32 = 0.01;
}

/// Model-related constants.
pub mod model {
    use crate::types::TokenId;

    /// Blank token ID for CTC/RNN-T models.
    pub const BLANK_TOKEN_ID: TokenId = TokenId::BLANK;

    /// Unknown token ID.
    pub const UNKNOWN_TOKEN_ID: TokenId = TokenId::UNKNOWN;

    /// Maximum sequence length for decoder.
    pub const MAX_SEQUENCE_LENGTH: usize = 512;

    /// Decoder state size for RNN-T.
    pub const DECODER_STATE_SIZE: usize = 640;

    /// Encoder state size for RNN-T.
    pub const ENCODER_STATE_SIZE: usize = 1024;

    /// Maximum beam width for beam search.
    pub const MAX_BEAM_WIDTH: usize = 100;

    /// Default beam width for beam search.
    pub const DEFAULT_BEAM_WIDTH: usize = 10;

    /// Length penalty for beam search.
    pub const LENGTH_PENALTY: f32 = 0.6;

    /// Coverage penalty for beam search.
    pub const COVERAGE_PENALTY: f32 = 0.0;

    /// Minimum log probability threshold.
    pub const MIN_LOG_PROB: f32 = -100.0;

    /// Maximum vocabulary size.
    pub const MAX_VOCAB_SIZE: usize = 10000;
}

/// Triton server constants.
pub mod triton {
    use crate::types::{PoolSize, TimeoutDuration};

    /// Default Triton server endpoint.
    pub const DEFAULT_ENDPOINT: &str = "http://localhost:8001";

    /// Default gRPC timeout.
    pub const DEFAULT_TIMEOUT: TimeoutDuration = TimeoutDuration::DEFAULT;

    /// Default connection pool size.
    pub const DEFAULT_POOL_SIZE: PoolSize = PoolSize::DEFAULT;

    /// Maximum number of retry attempts.
    pub const MAX_RETRY_ATTEMPTS: usize = 3;

    /// Retry backoff base delay in milliseconds.
    pub const RETRY_BASE_DELAY_MS: u64 = 100;

    /// Maximum retry delay in milliseconds.
    pub const MAX_RETRY_DELAY_MS: u64 = 5000;

    /// Model server ready timeout in seconds.
    pub const MODEL_READY_TIMEOUT_SECS: u64 = 30;

    /// Health check interval in seconds.
    pub const HEALTH_CHECK_INTERVAL_SECS: u64 = 10;

    /// Maximum batch size for inference.
    pub const MAX_BATCH_SIZE: usize = 32;

    /// Model names for the ASR pipeline.
    pub const PREPROCESSOR_MODEL: &str = "preprocessing";
    pub const ENCODER_MODEL: &str = "encoder";
    pub const DECODER_MODEL: &str = "decoder";
    pub const JOINT_MODEL: &str = "joint";

    /// Triton model constants for backward compatibility
    pub const PREPROCESSOR_MODEL_NAME: &str = "preprocessor";
    pub const ENCODER_MODEL_NAME: &str = "encoder";
    pub const DECODER_JOINT_MODEL_NAME: &str = "decoder_joint";
    pub const VOCABULARY_SIZE: usize = 1030;
    pub const BLANK_TOKEN_ID: i32 = 1024;
    pub const MAX_SYMBOLS_PER_STEP: usize = 30;
    pub const MAX_TOTAL_TOKENS: usize = 200;
    pub const DECODER_STATE_SIZE: usize = 640;
}

/// Performance tuning constants.
pub mod performance {
    /// Default NUMA node (use system default).
    pub const DEFAULT_NUMA_NODE: Option<usize> = None;

    /// Default CPU affinity mask (use all CPUs).
    pub const DEFAULT_CPU_AFFINITY: Option<Vec<usize>> = None;

    /// Memory pool initial size.
    pub const MEMORY_POOL_INITIAL_SIZE: usize = 1024 * 1024; // 1MB

    /// Memory pool maximum size.
    pub const MEMORY_POOL_MAX_SIZE: usize = 100 * 1024 * 1024; // 100MB

    /// Memory pool block size.
    pub const MEMORY_POOL_BLOCK_SIZE: usize = 4096; // 4KB

    /// Maximum number of concurrent requests.
    pub const MAX_CONCURRENT_REQUESTS: usize = 100;

    /// Circuit breaker failure threshold.
    pub const CIRCUIT_BREAKER_FAILURE_THRESHOLD: usize = 5;

    /// Circuit breaker recovery timeout in seconds.
    pub const CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECS: u64 = 60;

    /// Default number of worker threads.
    pub const DEFAULT_WORKER_THREADS: usize = 4;

    /// Maximum queue size for async operations.
    pub const MAX_QUEUE_SIZE: usize = 1000;

    /// Encoder output tensor size (1024 features * 100 frames)
    pub const ENCODER_OUTPUT_SIZE: usize = 1024 * 100;

    /// Raw tensor buffer size (1MB)
    pub const TENSOR_BUFFER_SIZE: usize = 1024 * 1024;

    /// Audio buffer capacity in seconds
    pub const AUDIO_BUFFER_SECONDS: usize = 2;

    /// Maximum tokens per decoding sequence
    pub const MAX_TOKENS_PER_SEQUENCE: usize = 200;

    /// Memory pool sizes
    pub const AUDIO_BUFFER_POOL_SIZE: usize = 20;
    pub const ENCODER_POOL_SIZE: usize = 50;
    pub const DECODER_POOL_SIZE: usize = 100;
    pub const WORKSPACE_POOL_SIZE: usize = 20;
    pub const RAW_TENSOR_POOL_SIZE: usize = 30;

    /// Pre-allocation sizes
    pub const AUDIO_BUFFER_PRE_ALLOC: usize = 5;
    pub const ENCODER_PRE_ALLOC: usize = 10;
    pub const DECODER_PRE_ALLOC: usize = 20;
    pub const WORKSPACE_PRE_ALLOC: usize = 5;
    pub const RAW_TENSOR_PRE_ALLOC: usize = 5;
}

/// Server configuration constants.
pub mod server {
    use crate::types::TimeoutDuration;

    /// Default server host.
    pub const DEFAULT_HOST: &str = "0.0.0.0";

    /// Default server port.
    pub const DEFAULT_PORT: u16 = 8080;

    /// Default request timeout.
    pub const DEFAULT_REQUEST_TIMEOUT: TimeoutDuration = TimeoutDuration::DEFAULT;

    /// Maximum request body size in bytes.
    pub const MAX_REQUEST_BODY_SIZE: usize = 10 * 1024 * 1024; // 10MB

    /// WebSocket ping interval in seconds.
    pub const WEBSOCKET_PING_INTERVAL_SECS: u64 = 30;

    /// WebSocket close timeout in seconds.
    pub const WEBSOCKET_CLOSE_TIMEOUT_SECS: u64 = 10;

    /// Maximum number of WebSocket connections.
    pub const MAX_WEBSOCKET_CONNECTIONS: usize = 1000;

    /// Rate limiting: requests per second.
    pub const RATE_LIMIT_RPS: u32 = 100;

    /// Rate limiting: burst size.
    pub const RATE_LIMIT_BURST: u32 = 10;

    /// CORS max age in seconds.
    pub const CORS_MAX_AGE_SECS: u64 = 3600;

    /// Default log level.
    pub const DEFAULT_LOG_LEVEL: &str = "info";

    /// Graceful shutdown timeout in seconds.
    pub const GRACEFUL_SHUTDOWN_TIMEOUT_SECS: u64 = 30;
}

/// Streaming constants.
pub mod streaming {
    /// End of stream control byte.
    pub const CONTROL_BYTE_END: u8 = 0xFF;

    /// Keepalive control byte.
    pub const CONTROL_BYTE_KEEPALIVE: u8 = 0x00;

    /// Keepalive check period in milliseconds.
    pub const KEEPALIVE_CHECK_PERIOD_MS: u64 = 5000;

    /// Stream timeout in seconds.
    pub const STREAM_TIMEOUT_SECS: u64 = 300;
}

/// Metrics and monitoring constants.
pub mod metrics {
    /// Metrics export interval in seconds.
    pub const EXPORT_INTERVAL_SECS: u64 = 10;

    /// Default metrics endpoint path.
    pub const METRICS_PATH: &str = "/metrics";

    /// Default health check path.
    pub const HEALTH_PATH: &str = "/health";

    /// Default readiness check path.
    pub const READY_PATH: &str = "/ready";

    /// Histogram buckets for latency metrics (in milliseconds).
    pub const LATENCY_BUCKETS_MS: &[f64] = &[
        0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
    ];

    /// Histogram buckets for audio duration metrics (in seconds).
    pub const AUDIO_DURATION_BUCKETS_SECS: &[f64] = &[
        0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ];

    /// Maximum number of metrics to retain in memory.
    pub const MAX_METRICS_RETENTION: usize = 10000;
}

/// File system constants.
pub mod fs {
    /// Default configuration file name.
    pub const DEFAULT_CONFIG_FILE: &str = "config.toml";

    /// Default vocabulary file name.
    pub const DEFAULT_VOCAB_FILE: &str = "vocab.txt";

    /// Default model repository path.
    pub const DEFAULT_MODEL_REPO: &str = "./model-repo";

    /// Default logs directory.
    pub const DEFAULT_LOGS_DIR: &str = "./logs";

    /// Default temporary directory.
    pub const DEFAULT_TEMP_DIR: &str = "./tmp";

    /// Maximum log file size in bytes.
    pub const MAX_LOG_FILE_SIZE: usize = 100 * 1024 * 1024; // 100MB

    /// Maximum number of log files to retain.
    pub const MAX_LOG_FILES: usize = 10;
}

/// CUDA constants (if enabled).
#[cfg(feature = "cuda")]
pub mod cuda {
    /// Default CUDA device ID.
    pub const DEFAULT_DEVICE_ID: i32 = 0;

    /// CUDA memory pool size in bytes.
    pub const MEMORY_POOL_SIZE: usize = 1024 * 1024 * 1024; // 1GB

    /// CUDA stream count for concurrent operations.
    pub const STREAM_COUNT: usize = 4;

    /// Maximum CUDA kernel execution time in seconds.
    pub const MAX_KERNEL_EXECUTION_TIME_SECS: u64 = 30;

    /// CUDA memory alignment in bytes.
    pub const MEMORY_ALIGNMENT: usize = 256;
}

/// Development and testing constants.
#[cfg(test)]
pub mod test {
    /// Test audio sample rate.
    pub const TEST_SAMPLE_RATE: u32 = 16000;

    /// Test audio duration in seconds.
    pub const TEST_AUDIO_DURATION_SECS: u64 = 1;

    /// Test batch size.
    pub const TEST_BATCH_SIZE: usize = 4;

    /// Test vocabulary size.
    pub const TEST_VOCAB_SIZE: usize = 1000;

    /// Test timeout in milliseconds.
    pub const TEST_TIMEOUT_MS: u64 = 5000;
}
