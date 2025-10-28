//! Configuration constants for the ASR server.
//!
//! This module contains all constant values organized by domain:
//! - Audio processing parameters
//! - Model configuration
//! - Streaming settings
//! - Concurrency limits
//! - Memory pool sizes
//! - Connection pool settings
//! - Stream processing parameters
//! - Timeout values

/// Audio processing constants
pub mod audio {
    /// Standard audio sample rate for all processing
    pub const SAMPLE_RATE: u32 = 16000;

    /// Size of the audio buffer in bytes
    pub const BUFFER_CAPACITY: usize = 1024 * 1024; // 1MB

    /// Minimum number of samples required for partial transcription
    pub const MIN_PARTIAL_TRANSCRIPTION_MS: u64 = 100;

    /// Minimum number of audio samples for partial transcription
    pub const MIN_PARTIAL_TRANSCRIPTION_SAMPLES: usize =
        (SAMPLE_RATE as u64 * MIN_PARTIAL_TRANSCRIPTION_MS / 1000) as usize;

    /// Maximum audio length for batch processing in seconds
    pub const MAX_BATCH_AUDIO_LENGTH_SECS: f32 = 30.0;
}

/// Triton model constants
pub mod model {
    /// Preprocessor model name
    pub const PREPROCESSOR_MODEL_NAME: &str = "preprocessor";

    /// Encoder model name
    pub const ENCODER_MODEL_NAME: &str = "encoder";

    /// Decoder and joint network model name
    pub const DECODER_JOINT_MODEL_NAME: &str = "decoder_joint";

    /// Expected vocabulary size (including special tokens)
    pub const VOCABULARY_SIZE: usize = 1030;

    /// Blank token ID
    pub const BLANK_TOKEN_ID: i32 = 1024;

    /// Size of the decoder state vectors
    pub const DECODER_STATE_SIZE: usize = 640;

    /// Maximum symbols to predict per encoder frame
    pub const MAX_SYMBOLS_PER_STEP: usize = 30;

    /// Maximum total tokens to generate in a single decoding session
    pub const MAX_TOTAL_TOKENS: usize = 200;
}

/// WebSocket streaming constants
pub mod streaming {
    /// Duration in milliseconds between keepalive checks
    pub const KEEPALIVE_CHECK_PERIOD_MS: u64 = 100;

    /// Maximum time in seconds a stream can be inactive before timeout
    pub const STREAM_TIMEOUT_SECS: u64 = 30;

    /// Control byte indicating end of stream
    pub const CONTROL_BYTE_END: u8 = 0x00;

    /// Control byte indicating keepalive
    pub const CONTROL_BYTE_KEEPALIVE: u8 = 0x01;
}

/// Server concurrency limits
pub mod concurrency {
    /// Maximum number of concurrent WebSocket streams
    pub const MAX_CONCURRENT_STREAMS: usize = 10;

    /// Maximum number of concurrent batch requests
    pub const MAX_CONCURRENT_BATCHES: usize = 50;

    /// Size of the inference queue
    pub const INFERENCE_QUEUE_SIZE: usize = 100;
}

/// Memory pool configuration constants
pub mod memory {
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

/// Connection pool configuration constants
pub mod connection_pool {
    /// Default maximum connections
    pub const DEFAULT_MAX_CONNECTIONS: usize = 50;

    /// Default minimum connections
    pub const DEFAULT_MIN_CONNECTIONS: usize = 5;

    /// Default connection idle timeout
    pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 300; // 5 minutes

    /// Default connection acquisition timeout
    pub const DEFAULT_ACQUIRE_TIMEOUT_MS: u64 = 500;

    /// Default cleanup interval
    pub const DEFAULT_CLEANUP_INTERVAL_SECS: u64 = 60;

    /// Maximum connection age
    pub const DEFAULT_MAX_CONNECTION_AGE_SECS: u64 = 3600; // 1 hour
}

/// Stream processing configuration constants
pub mod stream_processing {
    /// Processing chunk size in seconds
    pub const CHUNK_SIZE_SECONDS: f32 = 2.0;

    /// Leading context in seconds
    pub const LEADING_CONTEXT_SECONDS: f32 = 1.0;

    /// Trailing context in seconds
    pub const TRAILING_CONTEXT_SECONDS: f32 = 0.5;

    /// Buffer capacity in seconds
    pub const BUFFER_CAPACITY_SECONDS: f32 = 10.0;

    /// Maximum chunk size in bytes
    pub const MAX_CHUNK_SIZE_BYTES: usize = 1024 * 1024; // 1MB

    /// Rate limiting: max messages per window
    pub const MAX_MESSAGES_PER_WINDOW: u32 = 100;

    /// Rate limiting: window duration in seconds
    pub const RATE_LIMIT_WINDOW_SECS: u64 = 1;
}

/// Centralized timeout configuration constants
pub mod timeouts {
    use std::time::Duration;

    /// Standard inference timeout for ASR operations
    pub const INFERENCE_TIMEOUT: Duration = Duration::from_secs(5);

    /// Connection acquisition timeout from pool
    pub const CONNECTION_ACQUIRE_TIMEOUT: Duration = Duration::from_millis(500);

    /// Stream inactivity timeout before disconnection
    pub const STREAM_INACTIVITY_TIMEOUT: Duration = Duration::from_secs(30);

    /// Circuit breaker request timeout
    pub const CIRCUIT_BREAKER_TIMEOUT: Duration = Duration::from_secs(10);

    /// Keepalive check interval for WebSocket streams
    pub const KEEPALIVE_CHECK_INTERVAL: Duration = Duration::from_millis(100);

    /// Triton model inference timeout
    pub const TRITON_INFERENCE_TIMEOUT: Duration = Duration::from_secs(5);

    /// Connection pool cleanup interval
    pub const CONNECTION_CLEANUP_INTERVAL: Duration = Duration::from_secs(60);
}
