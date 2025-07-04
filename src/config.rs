//! Application-wide configuration and constants.
//!
//! This module centralizes all configuration values, whether loaded from environment
//! variables or defined as constants. This promotes the DRY principle and makes
//! configuration changes easier to manage.

use std::env;
use std::path::PathBuf;
use std::time::Duration;
// use tracing::debug;  // Temporarily disabled
macro_rules! debug { ($($tt:tt)*) => {}; }

use crate::error::{AppError, Result};

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
    pub fn from_env() -> Result<Self> {
        let config = Self {
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
        };

        config.validate()?;
        Ok(config)
    }

    /// Validate configuration values
    fn validate(&self) -> Result<()> {
        // Validate Triton endpoint URL
        if !self.triton_endpoint.starts_with("http://")
            && !self.triton_endpoint.starts_with("https://")
        {
            return Err(AppError::Configuration(
                "TRITON_ENDPOINT must start with http:// or https://".to_string(),
            ));
        }

        // Validate vocabulary path (comprehensive path traversal protection)
        self.validate_path(&self.vocabulary_path, "VOCABULARY_PATH")?;

        // Validate server host (basic validation)
        if self.server_host.is_empty() {
            return Err(AppError::Configuration(
                "SERVER_HOST cannot be empty".to_string(),
            ));
        }

        // Validate server port range
        if self.server_port == 0 || self.server_port < 1024 {
            return Err(AppError::Configuration(
                "SERVER_PORT must be between 1024 and 65535".to_string(),
            ));
        }

        // Validate inference timeout
        if self.inference_timeout.as_secs() == 0 || self.inference_timeout.as_secs() > 300 {
            return Err(AppError::Configuration(
                "INFERENCE_TIMEOUT_SECS must be between 1 and 300 seconds".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate a file path for security issues.
    ///
    /// This method provides comprehensive protection against path traversal attacks
    /// by checking for various malicious patterns and ensuring the path is safe.
    fn validate_path(&self, path: &std::path::Path, field_name: &str) -> Result<()> {
        // Convert to string for analysis
        let path_str = path.to_string_lossy();

        // Check for obvious path traversal patterns
        if path_str.contains("..") || path_str.contains("//") {
            return Err(AppError::Configuration(format!(
                "{} contains invalid path components (.. or //)",
                field_name
            )));
        }

        // Check for null bytes (can be used to bypass filters)
        if path_str.contains('\0') {
            return Err(AppError::Configuration(format!(
                "{} contains null bytes",
                field_name
            )));
        }

        // Check for control characters that shouldn't be in file paths
        if path_str.chars().any(|c| c.is_control() && c != '\t') {
            return Err(AppError::Configuration(format!(
                "{} contains invalid control characters",
                field_name
            )));
        }

        // Attempt to canonicalize the path to resolve any .. components
        // This is more robust than string matching
        match path.canonicalize() {
            Ok(canonical_path) => {
                // Check if the canonical path is still within reasonable bounds
                // For security, we might want to ensure it's within a specific directory
                let canonical_str = canonical_path.to_string_lossy();

                // Additional check: ensure the canonicalized path doesn't contain suspicious patterns
                if canonical_str.contains("..") {
                    return Err(AppError::Configuration(format!(
                        "{} resolves to a path with traversal components",
                        field_name
                    )));
                }

                // Optionally, you could add a check to ensure the path is within an allowed directory:
                // if !canonical_path.starts_with("/allowed/directory") {
                //     return Err(AppError::Configuration(
                //         format!("{} is outside allowed directory", field_name)
                //     ));
                // }
            }
            Err(_) => {
                // If canonicalization fails, the path might not exist yet, which could be okay
                // depending on your use case. For vocabulary files, we might want to be more strict.
                debug!(
                    "Path canonicalization failed for {}: {:?} (file may not exist yet)",
                    field_name, path
                );

                // Still perform basic validation even if canonicalization fails
                if path_str.len() > 4096 {
                    return Err(AppError::Configuration(format!(
                        "{} is too long (max 4096 characters)",
                        field_name
                    )));
                }
            }
        }

        Ok(())
    }
}
