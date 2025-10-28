//! Application-wide configuration and constants.
//!
//! This module centralizes all configuration values, whether loaded from environment
//! variables or defined as constants. This promotes the DRY principle and makes
//! configuration changes easier to manage.

// Module declarations
pub mod constants;
mod defaults;
mod loader;
mod serde_helpers;
pub mod types;

// Re-export the Config struct and public constants
pub use types::Config;

// Re-export constant modules for convenient access
pub use constants::{
    audio, concurrency, connection_pool, memory, model, stream_processing, streaming, timeouts,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    // Helper to create a valid config for testing
    fn create_valid_config() -> Config {
        Config {
            triton_endpoint: "http://localhost:8001".to_string(),
            vocabulary_path: PathBuf::from("vocab.txt"),
            server_host: "0.0.0.0".to_string(),
            server_port: 8057,
            inference_backend: "grpc".to_string(),
            cuda_device_id: 0,
            inference_timeout: Duration::from_secs(5),
            max_concurrent_streams: 10,
            max_concurrent_batches: 50,
            inference_queue_size: 100,
            audio_buffer_capacity: 1024 * 1024,
            max_batch_audio_length_secs: 30.0,
            stream_timeout_secs: 30,
            keepalive_check_period_ms: 100,
            preprocessor_model_name: "preprocessor".to_string(),
            encoder_model_name: "encoder".to_string(),
            decoder_joint_model_name: "decoder_joint".to_string(),
            max_symbols_per_step: 30,
            max_total_tokens: 200,
            enable_platform_optimizations: true,
            force_io_backend: None,
            disable_numa_in_cloud: true,
            disable_cpu_affinity: false,
            force_io_uring: false,
        }
    }

    // Test default value functions
    #[test]
    fn test_default_values() {
        use defaults::*;
        assert_eq!(default_max_concurrent_streams(), 10);
        assert_eq!(default_max_concurrent_batches(), 50);
        assert_eq!(default_inference_queue_size(), 100);
        assert_eq!(default_audio_buffer_capacity(), 1024 * 1024);
        assert_eq!(default_max_batch_audio_length(), 30.0);
        assert_eq!(default_stream_timeout_secs(), 30);
        assert_eq!(default_keepalive_check_period_ms(), 100);
        assert_eq!(default_preprocessor_model_name(), "preprocessor");
        assert_eq!(default_encoder_model_name(), "encoder");
        assert_eq!(default_decoder_joint_model_name(), "decoder_joint");
        assert_eq!(default_max_symbols_per_step(), 30);
        assert_eq!(default_max_total_tokens(), 200);
        assert_eq!(default_enable_platform_optimizations(), true);
        assert_eq!(default_disable_numa_in_cloud(), true);
        assert_eq!(default_disable_cpu_affinity(), false);
        assert_eq!(default_force_io_uring(), false);
        assert_eq!(default_inference_backend(), "grpc");
        assert_eq!(default_cuda_device_id(), 0);
    }

    // Test validation: triton endpoint must start with http:// or https://
    #[test]
    fn test_validate_triton_endpoint_invalid() {
        let mut config = create_valid_config();
        config.triton_endpoint = "localhost:8001".to_string();

        let result = config.validate();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("must start with http"));
    }

    #[test]
    fn test_validate_triton_endpoint_valid() {
        let mut config = create_valid_config();

        // Test http://
        config.triton_endpoint = "http://localhost:8001".to_string();
        assert!(config.validate().is_ok());

        // Test https://
        config.triton_endpoint = "https://triton.example.com:8001".to_string();
        assert!(config.validate().is_ok());
    }

    // Test validation: server host cannot be empty
    #[test]
    fn test_validate_server_host_empty() {
        let mut config = create_valid_config();
        config.server_host = "".to_string();

        let result = config.validate();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("cannot be empty"));
    }

    // Test validation: server port must be >= 1024
    #[test]
    fn test_validate_server_port_too_low() {
        let mut config = create_valid_config();
        config.server_port = 80;

        let result = config.validate();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("must be between 1024"));
    }

    #[test]
    fn test_validate_server_port_zero() {
        let mut config = create_valid_config();
        config.server_port = 0;

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_server_port_valid() {
        let mut config = create_valid_config();
        config.server_port = 1024;
        assert!(config.validate().is_ok());

        config.server_port = 8080;
        assert!(config.validate().is_ok());

        config.server_port = 65535;
        assert!(config.validate().is_ok());
    }

    // Test validation: inference timeout bounds (1-300 seconds)
    #[test]
    fn test_validate_inference_timeout_zero() {
        let mut config = create_valid_config();
        config.inference_timeout = Duration::from_secs(0);

        let result = config.validate();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("must be between 1 and 300"));
    }

    #[test]
    fn test_validate_inference_timeout_too_large() {
        let mut config = create_valid_config();
        config.inference_timeout = Duration::from_secs(301);

        let result = config.validate();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("must be between 1 and 300"));
    }

    #[test]
    fn test_validate_inference_timeout_valid() {
        let mut config = create_valid_config();

        config.inference_timeout = Duration::from_secs(1);
        assert!(config.validate().is_ok());

        config.inference_timeout = Duration::from_secs(5);
        assert!(config.validate().is_ok());

        config.inference_timeout = Duration::from_secs(300);
        assert!(config.validate().is_ok());
    }

    // Test path validation: path traversal protection
    #[test]
    fn test_validate_path_traversal_double_dot() {
        let config = create_valid_config();
        let path = PathBuf::from("../../etc/passwd");

        let result = config.validate_path(&path, "TEST_PATH");
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("potentially unsafe"));
    }

    #[test]
    fn test_validate_path_double_slash() {
        let config = create_valid_config();
        let path = PathBuf::from("path//to//file");

        let result = config.validate_path(&path, "TEST_PATH");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_path_single_relative_allowed() {
        let config = create_valid_config();
        // Single ../ should be allowed for relative paths
        let path = PathBuf::from("../model-repo/vocab.txt");

        // This test may pass or fail depending on whether the path exists
        // Just ensure it doesn't panic and error is related to path issues, not traversal
        let result = config.validate_path(&path, "TEST_PATH");
        // Single ../ is allowed, only ../../ is blocked
        if result.is_err() {
            let err_msg = format!("{:?}", result.unwrap_err());
            // Should not complain about "unsafe" but may complain about length or existence
            assert!(!err_msg.contains("unsafe") || err_msg.contains("too long"));
        }
    }

    // Test backend helper methods
    #[test]
    fn test_is_cuda_backend() {
        let mut config = create_valid_config();

        config.inference_backend = "cuda".to_string();
        assert!(config.is_cuda_backend());
        assert!(!config.is_grpc_backend());

        config.inference_backend = "CUDA".to_string();
        assert!(config.is_cuda_backend());

        config.inference_backend = "grpc".to_string();
        assert!(!config.is_cuda_backend());
    }

    #[test]
    fn test_is_grpc_backend() {
        let mut config = create_valid_config();

        config.inference_backend = "grpc".to_string();
        assert!(config.is_grpc_backend());
        assert!(!config.is_cuda_backend());

        config.inference_backend = "GRPC".to_string();
        assert!(config.is_grpc_backend());

        config.inference_backend = "cuda".to_string();
        assert!(!config.is_grpc_backend());
    }

    #[test]
    fn test_get_cuda_device_id() {
        let mut config = create_valid_config();
        config.cuda_device_id = 2;
        assert_eq!(config.get_cuda_device_id(), 2);
    }

    // Test serialization
    #[test]
    fn test_to_toml() {
        let config = create_valid_config();
        let result = config.to_toml();

        assert!(result.is_ok());
        let toml_str = result.unwrap();
        assert!(toml_str.contains("triton_endpoint"));
        assert!(toml_str.contains("server_port"));
    }

    #[test]
    fn test_to_yaml() {
        let config = create_valid_config();
        let result = config.to_yaml();

        assert!(result.is_ok());
        let yaml_str = result.unwrap();
        assert!(yaml_str.contains("triton_endpoint"));
        assert!(yaml_str.contains("server_port"));
    }

    // Test validate_backend with gRPC (always works)
    #[test]
    fn test_validate_backend_grpc() {
        let mut config = create_valid_config();
        config.inference_backend = "grpc".to_string();

        let result = config.validate_backend();
        assert!(result.is_ok());
    }

    // Test validate_backend with unknown backend
    #[test]
    fn test_validate_backend_unknown() {
        let mut config = create_valid_config();
        config.inference_backend = "unknown".to_string();

        let result = config.validate_backend();
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Unknown inference backend"));
    }

    // Test module constants
    #[test]
    fn test_audio_constants() {
        assert_eq!(audio::SAMPLE_RATE, 16000);
        assert_eq!(audio::BUFFER_CAPACITY, 1024 * 1024);
        assert_eq!(audio::MIN_PARTIAL_TRANSCRIPTION_MS, 100);
        assert_eq!(audio::MAX_BATCH_AUDIO_LENGTH_SECS, 30.0);
    }

    #[test]
    fn test_model_constants() {
        assert_eq!(model::PREPROCESSOR_MODEL_NAME, "preprocessor");
        assert_eq!(model::ENCODER_MODEL_NAME, "encoder");
        assert_eq!(model::DECODER_JOINT_MODEL_NAME, "decoder_joint");
        assert_eq!(model::VOCABULARY_SIZE, 1030);
        assert_eq!(model::BLANK_TOKEN_ID, 1024);
        assert_eq!(model::DECODER_STATE_SIZE, 640);
        assert_eq!(model::MAX_SYMBOLS_PER_STEP, 30);
        assert_eq!(model::MAX_TOTAL_TOKENS, 200);
    }

    #[test]
    fn test_streaming_constants() {
        assert_eq!(streaming::KEEPALIVE_CHECK_PERIOD_MS, 100);
        assert_eq!(streaming::STREAM_TIMEOUT_SECS, 30);
        assert_eq!(streaming::CONTROL_BYTE_END, 0x00);
        assert_eq!(streaming::CONTROL_BYTE_KEEPALIVE, 0x01);
    }

    #[test]
    fn test_concurrency_constants() {
        assert_eq!(concurrency::MAX_CONCURRENT_STREAMS, 10);
        assert_eq!(concurrency::MAX_CONCURRENT_BATCHES, 50);
        assert_eq!(concurrency::INFERENCE_QUEUE_SIZE, 100);
    }
}
