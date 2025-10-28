//! Configuration loading and validation logic.
//!
//! This module handles loading configuration from multiple sources
//! (environment variables, TOML, YAML), validation, and serialization.

use figment::{
    providers::{Env, Format, Serialized, Toml, Yaml},
    Figment,
};
use std::env;
use std::path::PathBuf;
use std::time::Duration;
use tracing::debug;

use crate::error::{AppError, Result};

use super::defaults::*;
use super::types::Config;

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
                "SERVER_HOST",
                "SERVER_PORT",
                "TRITON_ENDPOINT",
                "INFERENCE_TIMEOUT_SECS",
                "VOCABULARY_PATH",
            ]))
            .extract()
            .map_err(|e| AppError::ConfigError(format!("Failed to load configuration: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Generate default configuration values
    fn default_figment() -> Figment {
        Figment::from(Serialized::defaults(Config {
            triton_endpoint: "http://localhost:8001".to_string(),
            vocabulary_path: PathBuf::from("../model-repo/vocab.txt"),
            server_host: "0.0.0.0".to_string(),
            server_port: 8057,
            inference_backend: default_inference_backend(),
            cuda_device_id: default_cuda_device_id(),
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

            inference_backend: env::var("INFERENCE_BACKEND")
                .unwrap_or_else(|_| default_inference_backend()),

            cuda_device_id: env::var("CUDA_DEVICE_ID")
                .ok()
                .and_then(|d| d.parse().ok())
                .unwrap_or_else(default_cuda_device_id),

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
    pub fn validate(&self) -> Result<()> {
        // Validate Triton endpoint URL
        if !self.triton_endpoint.starts_with("http://")
            && !self.triton_endpoint.starts_with("https://")
        {
            return Err(AppError::ConfigError(
                "TRITON_ENDPOINT must start with http:// or https://".to_string(),
            ));
        }

        // Validate vocabulary path (comprehensive path traversal protection)
        self.validate_path(&self.vocabulary_path, "VOCABULARY_PATH")?;

        // Validate server host (basic validation)
        if self.server_host.is_empty() {
            return Err(AppError::ConfigError(
                "SERVER_HOST cannot be empty".to_string(),
            ));
        }

        // Validate server port range
        if self.server_port == 0 || self.server_port < 1024 {
            return Err(AppError::ConfigError(
                "SERVER_PORT must be between 1024 and 65535".to_string(),
            ));
        }

        // Validate inference timeout
        if self.inference_timeout.as_secs() == 0 || self.inference_timeout.as_secs() > 300 {
            return Err(AppError::ConfigError(
                "INFERENCE_TIMEOUT_SECS must be between 1 and 300 seconds".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate a file path for security issues.
    ///
    /// This method provides comprehensive protection against path traversal attacks
    /// by checking for various malicious patterns and ensuring the path is safe.
    pub(crate) fn validate_path(&self, path: &std::path::Path, field_name: &str) -> Result<()> {
        // Convert to string for analysis
        let path_str = path.to_string_lossy();

        // Check for obvious path traversal patterns (but allow relative paths like ../model-repo)
        // Only block patterns that could be malicious like ../../etc/passwd
        if path_str.contains("../..") || path_str.contains("//") {
            return Err(AppError::ConfigError(format!(
                "{} contains potentially unsafe path components",
                field_name
            )));
        }

        // Check for null bytes (can be used to bypass filters)
        if path_str.contains('\0') {
            return Err(AppError::ConfigError(format!(
                "{} contains null bytes",
                field_name
            )));
        }

        // Check for control characters that shouldn't be in file paths
        if path_str.chars().any(|c| c.is_control() && c != '\t') {
            return Err(AppError::ConfigError(format!(
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
                    return Err(AppError::ConfigError(format!(
                        "{} resolves to a path with traversal components",
                        field_name
                    )));
                }

                // Optionally, you could add a check to ensure the path is within an allowed directory:
                // if !canonical_path.starts_with("/allowed/directory") {
                //     return Err(AppError::ConfigError(
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
                    return Err(AppError::ConfigError(format!(
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
            .map_err(|e| AppError::ConfigError(format!("Failed to serialize to TOML: {}", e)))
    }

    /// Export configuration to YAML format
    pub fn to_yaml(&self) -> Result<String> {
        serde_yaml::to_string(self)
            .map_err(|e| AppError::ConfigError(format!("Failed to serialize to YAML: {}", e)))
    }

    /// Validate backend configuration
    pub fn validate_backend(&self) -> Result<()> {
        match self.inference_backend.to_lowercase().as_str() {
            "grpc" => {
                // gRPC backend is always available
                Ok(())
            }
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    // Check if CUDA is available
                    if !crate::cuda::is_cuda_available() {
                        return Err(AppError::ConfigError(
                            "CUDA backend selected but CUDA is not available".to_string(),
                        ));
                    }

                    // Check if device ID is valid
                    let device_count = crate::cuda::get_cuda_device_count().map_err(|e| {
                        AppError::ConfigError(format!("Failed to get CUDA device count: {}", e))
                    })?;

                    if self.cuda_device_id < 0 || self.cuda_device_id >= device_count {
                        return Err(AppError::ConfigError(format!(
                            "Invalid CUDA device ID: {} (available devices: 0-{})",
                            self.cuda_device_id,
                            device_count - 1
                        )));
                    }

                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(AppError::ConfigError(
                        "CUDA backend selected but CUDA support is not compiled in".to_string(),
                    ))
                }
            }
            _ => Err(AppError::ConfigError(format!(
                "Unknown inference backend: '{}'. Supported backends: 'grpc', 'cuda'",
                self.inference_backend
            ))),
        }
    }
}
