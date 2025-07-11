//! Application-wide configuration and constants.
//!
//! This module centralizes all configuration values, whether loaded from environment
//! variables or defined as constants. This promotes the DRY principle and makes
//! configuration changes easier to manage.

use std::env;
use std::path::PathBuf;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use figment::{Figment, providers::{Env, Format, Toml, Yaml}};
use tracing::debug;

use crate::error::{AppError, Result};

/// Serde helper for Duration serialization/deserialization as seconds
mod duration_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

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

// Default value functions for serde defaults
fn default_max_concurrent_streams() -> usize { 10 }
fn default_max_concurrent_batches() -> usize { 50 }
fn default_inference_queue_size() -> usize { 100 }
fn default_audio_buffer_capacity() -> usize { 1024 * 1024 } // 1MB
fn default_max_batch_audio_length() -> f32 { 30.0 }
fn default_stream_timeout_secs() -> u64 { 30 }
fn default_keepalive_check_period_ms() -> u64 { 100 }
fn default_preprocessor_model_name() -> String { "preprocessor".to_string() }
fn default_encoder_model_name() -> String { "encoder".to_string() }
fn default_decoder_joint_model_name() -> String { "decoder_joint".to_string() }
fn default_max_symbols_per_step() -> usize { 30 }
fn default_max_total_tokens() -> usize { 200 }
fn default_enable_platform_optimizations() -> bool { true }
fn default_disable_numa_in_cloud() -> bool { true }
fn default_disable_cpu_affinity() -> bool { false }
fn default_force_io_uring() -> bool { false }
fn default_inference_backend() -> String { "grpc".to_string() }
fn default_cuda_device_id() -> i32 { 0 }

/// Application configuration loaded from multiple sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// URL for the Triton Inference Server
    pub triton_endpoint: String,

    /// Path to the vocabulary file
    pub vocabulary_path: PathBuf,

    /// HTTP server host
    pub server_host: String,

    /// HTTP server port
    pub server_port: u16,

    /// Inference backend to use: "grpc" or "cuda"
    #[serde(default = "default_inference_backend")]
    pub inference_backend: String,

    /// CUDA device ID for CUDA backend
    #[serde(default = "default_cuda_device_id")]
    pub cuda_device_id: i32,

    /// Timeout for inference requests
    #[serde(with = "duration_secs")]
    pub inference_timeout: Duration,

    // Server Performance Configuration
    /// Maximum number of concurrent WebSocket streams
    #[serde(default = "default_max_concurrent_streams")]
    pub max_concurrent_streams: usize,

    /// Maximum number of concurrent batch requests  
    #[serde(default = "default_max_concurrent_batches")]
    pub max_concurrent_batches: usize,

    /// Size of the inference queue
    #[serde(default = "default_inference_queue_size")]
    pub inference_queue_size: usize,

    // Audio Processing Configuration
    /// Audio buffer capacity in bytes
    #[serde(default = "default_audio_buffer_capacity")]
    pub audio_buffer_capacity: usize,

    /// Maximum audio length for batch processing in seconds
    #[serde(default = "default_max_batch_audio_length")]
    pub max_batch_audio_length_secs: f32,

    // Streaming Configuration
    /// WebSocket stream timeout in seconds
    #[serde(default = "default_stream_timeout_secs")]
    pub stream_timeout_secs: u64,

    /// Keepalive check period in milliseconds
    #[serde(default = "default_keepalive_check_period_ms")]
    pub keepalive_check_period_ms: u64,

    // Model Configuration
    /// Preprocessor model name
    #[serde(default = "default_preprocessor_model_name")]
    pub preprocessor_model_name: String,

    /// Encoder model name
    #[serde(default = "default_encoder_model_name")]
    pub encoder_model_name: String,

    /// Decoder and joint network model name
    #[serde(default = "default_decoder_joint_model_name")]
    pub decoder_joint_model_name: String,

    /// Maximum symbols to predict per encoder frame
    #[serde(default = "default_max_symbols_per_step")]
    pub max_symbols_per_step: usize,

    /// Maximum total tokens to generate in a single decoding session
    #[serde(default = "default_max_total_tokens")]
    pub max_total_tokens: usize,

    // Platform Optimization Configuration
    /// Enable platform-specific optimizations
    #[serde(default = "default_enable_platform_optimizations")]
    pub enable_platform_optimizations: bool,

    /// Force specific I/O backend (if None, auto-detect optimal)
    #[serde(default)]
    pub force_io_backend: Option<String>,

    /// Disable NUMA optimizations in cloud environments
    #[serde(default = "default_disable_numa_in_cloud")]
    pub disable_numa_in_cloud: bool,

    /// Disable CPU affinity optimizations  
    #[serde(default = "default_disable_cpu_affinity")]
    pub disable_cpu_affinity: bool,

    /// Enable io_uring even in cloud environments (expert mode)
    #[serde(default = "default_force_io_uring")]
    pub force_io_uring: bool,
}

impl Config {
    /// Load configuration from multiple sources with precedence:
    /// 1. Environment variables (highest priority)  
    /// 2. config.yaml (if exists)
    /// 3. config.toml (if exists)
    /// 4. Built-in defaults (lowest priority)
    pub fn load() -> Result<Self> {
        let config: Config = Figment::new()
            .merge(Self::default_figment())
            .merge(Toml::file("config.toml"))
            .merge(Yaml::file("config.yaml"))
            .merge(Env::prefixed("AMIRA_"))
            .merge(Env::raw().only(&[
                "SERVER_HOST", "SERVER_PORT", "TRITON_ENDPOINT", 
                "INFERENCE_TIMEOUT_SECS", "VOCABULARY_PATH"
            ]))
            .extract()
            .map_err(|e| AppError::Configuration(format!("Failed to load configuration: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Generate default configuration values
    fn default_figment() -> Figment {
        use figment::providers::Serialized;
        
        Figment::from(Serialized::defaults(Config {
            triton_endpoint: "http://localhost:8001".to_string(),
            vocabulary_path: PathBuf::from("../model-repo/vocab.txt"),
            server_host: "0.0.0.0".to_string(),
            server_port: 8057,
            inference_timeout: Duration::from_secs(5),
            max_concurrent_streams: default_max_concurrent_streams(),
            max_concurrent_batches: default_max_concurrent_batches(),
            inference_queue_size: default_inference_queue_size(),
            audio_buffer_capacity: default_audio_buffer_capacity(),
            max_batch_audio_length_secs: default_max_batch_audio_length(),
            stream_timeout_secs: default_stream_timeout_secs(),
            keepalive_check_period_ms: default_keepalive_check_period_ms(),
            preprocessor_model_name: default_preprocessor_model_name(),
            encoder_model_name: default_encoder_model_name(),
            decoder_joint_model_name: default_decoder_joint_model_name(),
            max_symbols_per_step: default_max_symbols_per_step(),
            max_total_tokens: default_max_total_tokens(),
            enable_platform_optimizations: default_enable_platform_optimizations(),
            force_io_backend: None,
            disable_numa_in_cloud: default_disable_numa_in_cloud(),
            disable_cpu_affinity: default_disable_cpu_affinity(),
            force_io_uring: default_force_io_uring(),
        }))
    }

    /// Load configuration from environment variables with sensible defaults (legacy support)
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

            // Use defaults for new fields (can be overridden by env vars)
            max_concurrent_streams: env::var("MAX_CONCURRENT_STREAMS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_max_concurrent_streams),

            max_concurrent_batches: env::var("MAX_CONCURRENT_BATCHES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_max_concurrent_batches),

            inference_queue_size: env::var("INFERENCE_QUEUE_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_inference_queue_size),

            audio_buffer_capacity: env::var("AUDIO_BUFFER_CAPACITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_audio_buffer_capacity),

            max_batch_audio_length_secs: env::var("MAX_BATCH_AUDIO_LENGTH_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_max_batch_audio_length),

            stream_timeout_secs: env::var("STREAM_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_stream_timeout_secs),

            keepalive_check_period_ms: env::var("KEEPALIVE_CHECK_PERIOD_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_keepalive_check_period_ms),

            preprocessor_model_name: env::var("PREPROCESSOR_MODEL_NAME")
                .unwrap_or_else(|_| default_preprocessor_model_name()),

            encoder_model_name: env::var("ENCODER_MODEL_NAME")
                .unwrap_or_else(|_| default_encoder_model_name()),

            decoder_joint_model_name: env::var("DECODER_JOINT_MODEL_NAME")
                .unwrap_or_else(|_| default_decoder_joint_model_name()),

            max_symbols_per_step: env::var("MAX_SYMBOLS_PER_STEP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_max_symbols_per_step),

            max_total_tokens: env::var("MAX_TOTAL_TOKENS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_max_total_tokens),

            enable_platform_optimizations: env::var("ENABLE_PLATFORM_OPTIMIZATIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_enable_platform_optimizations),

            force_io_backend: env::var("FORCE_IO_BACKEND").ok(),

            disable_numa_in_cloud: env::var("DISABLE_NUMA_IN_CLOUD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_disable_numa_in_cloud),

            disable_cpu_affinity: env::var("DISABLE_CPU_AFFINITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_disable_cpu_affinity),

            force_io_uring: env::var("FORCE_IO_URING")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(default_force_io_uring),
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

        // Check for obvious path traversal patterns (but allow relative paths like ../model-repo)
        // Only block patterns that could be malicious like ../../etc/passwd
        if path_str.contains("../..") || path_str.contains("//") {
            return Err(AppError::Configuration(format!(
                "{} contains potentially unsafe path components",
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
    
    /// Export configuration to TOML format
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self)
            .map_err(|e| AppError::Configuration(format!("Failed to serialize to TOML: {}", e)))
    }
    
    /// Export configuration to YAML format
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .map_err(|e| AppError::Configuration(format!("Failed to serialize to YAML: {}", e)))
    }
}
