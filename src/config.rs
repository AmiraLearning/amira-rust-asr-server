//! Application-wide configuration and constants.
//!
//! This module centralizes all configuration values, whether loaded from environment
//! variables or defined as constants. This promotes the DRY principle and makes
//! configuration changes easier to manage.

use std::env;
use std::path::PathBuf;
use std::time::Duration;

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

/// Application configuration loaded from environment variables
#[derive(Debug, Clone)]
pub struct Config {
    /// URL for the Triton Inference Server
    pub triton_endpoint: String,

    /// Path to the vocabulary file
    pub vocabulary_path: PathBuf,

    /// HTTP server host
    pub server_host: String,

    /// HTTP server port
    pub server_port: u16,

    /// Timeout for inference requests
    pub inference_timeout: Duration,
}

impl Config {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Self {
        Self {
            triton_endpoint: env::var("TRITON_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8001".to_string()),

            vocabulary_path: PathBuf::from(
                env::var("VOCABULARY_PATH")
                    .unwrap_or_else(|_| "../model-repo/vocab.txt".to_string()),
            ),

            server_host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),

            server_port: env::var("SERVER_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8057),

            inference_timeout: Duration::from_secs(
                env::var("INFERENCE_TIMEOUT_SECS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5),
            ),
        }
    }
}
