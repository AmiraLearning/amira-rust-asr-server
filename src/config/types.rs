//! Configuration types and structures.
//!
//! This module defines the main `Config` struct that holds all
//! application configuration values.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

use super::defaults::*;
use super::serde_helpers;

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
    #[serde(with = "serde_helpers::duration_secs")]
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
    /// Check if CUDA backend is enabled
    pub fn is_cuda_backend(&self) -> bool {
        self.inference_backend.to_lowercase() == "cuda"
    }

    /// Check if gRPC backend is enabled
    pub fn is_grpc_backend(&self) -> bool {
        self.inference_backend.to_lowercase() == "grpc"
    }

    /// Get the configured CUDA device ID
    pub fn get_cuda_device_id(&self) -> i32 {
        self.cuda_device_id
    }
}
