//! Domain-specific error types for the amira-rust-asr-server.
//!
//! This module provides a hierarchical error handling system using the `thiserror` crate
//! to define structured, typed errors with clear messages and proper error conversion.
//! Each domain has its own error type for better error handling and debugging.

pub mod app;
pub mod types;

// Re-export commonly used types
pub use app::{AppError, ErrorContext, IntoAppError, Result, with_timeout, with_timeout_and_convert};
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use std::io;
    use std::time::Duration;
    use tonic::Status as TonicStatus;

    // Test HTTP status code mapping for various error types
    #[test]
    fn test_http_status_invalid_sample_rate() {
        let error = AppError::Asr(AsrError::AudioProcessing(AudioError::InvalidSampleRate {
            expected: 16000,
            actual: 8000,
        }));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_http_status_invalid_format() {
        let error = AppError::Asr(AsrError::AudioProcessing(AudioError::InvalidFormat(
            "PCM required".to_string(),
        )));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_http_status_asr_configuration() {
        let error = AppError::Asr(AsrError::Configuration("Invalid config".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_status_triton_connection() {
        let grpc_error = TonicStatus::unavailable("Connection refused");
        let error = AppError::Triton(TritonError::Grpc(grpc_error));

        let response = error.into_response();
        // Grpc errors fall through to default INTERNAL_SERVER_ERROR
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_http_status_triton_timeout() {
        // Create a timeout error by actually timing out an operation
        let operation = tokio::time::timeout(
            Duration::from_millis(1),
            tokio::time::sleep(Duration::from_millis(100)),
        )
        .await;

        let elapsed_error = operation.unwrap_err();
        let error = AppError::Triton(TritonError::Timeout(elapsed_error));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn test_http_status_pool_exhausted() {
        let error = AppError::Triton(TritonError::PoolExhausted("No connections".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_config_missing_field() {
        let error = AppError::Config(ConfigError::MissingField {
            field: "triton_endpoint".to_string(),
        });

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_status_config_invalid_value() {
        let error = AppError::Config(ConfigError::InvalidValue {
            field: "port".to_string(),
            value: "999999".to_string(),
        });

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_http_status_request_validation() {
        let error = AppError::Server(ServerError::RequestValidation(
            "Invalid audio buffer".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_http_status_service_unavailable() {
        let error = AppError::Server(ServerError::ServiceUnavailable("Overloaded".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_rate_limit_exceeded() {
        let error = AppError::Server(ServerError::RateLimitExceeded(
            "Too many requests".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn test_http_status_circuit_breaker() {
        let error = AppError::Performance(PerformanceError::CircuitBreakerOpen(
            "Circuit open".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_resource_exhausted() {
        let error = AppError::Performance(PerformanceError::ResourceExhausted(
            "Out of memory".to_string(),
        ));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_network_error() {
        let error = AppError::Network("Connection failed".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn test_http_status_timeout() {
        let error = AppError::Timeout("Request timeout".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    }

    #[test]
    fn test_http_status_capacity_exceeded() {
        let error = AppError::CapacityExceeded("Queue full".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_http_status_triton_inference() {
        let error = AppError::TritonInference("Model error".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn test_http_status_default_internal_server_error() {
        let error = AppError::Internal("Something went wrong".to_string());

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Test error_type() method
    #[test]
    fn test_error_type_audio_processing() {
        let error = AppError::Asr(AsrError::AudioProcessing(AudioError::BufferUnderrun));
        assert_eq!(error.error_type(), "audio_processing");
    }

    #[test]
    fn test_error_type_model_inference() {
        let error = AppError::Asr(AsrError::ModelInference(ModelError::Inference(
            "Test".to_string(),
        )));
        assert_eq!(error.error_type(), "model_inference");
    }

    #[test]
    fn test_error_type_decoder_state() {
        let error = AppError::Asr(AsrError::DecoderState("Invalid state".to_string()));
        assert_eq!(error.error_type(), "decoder_state");
    }

    #[test]
    fn test_error_type_asr_configuration() {
        let error = AppError::Asr(AsrError::Configuration("Config error".to_string()));
        assert_eq!(error.error_type(), "asr_configuration");
    }

    #[test]
    fn test_error_type_vocabulary() {
        let error = AppError::Asr(AsrError::Vocabulary("Vocab error".to_string()));
        assert_eq!(error.error_type(), "vocabulary");
    }

    #[test]
    fn test_error_type_pipeline() {
        let error = AppError::Asr(AsrError::Pipeline("Pipeline error".to_string()));
        assert_eq!(error.error_type(), "pipeline");
    }

    #[test]
    fn test_error_type_triton() {
        let error = AppError::Triton(TritonError::ModelServer("Server error".to_string()));
        assert_eq!(error.error_type(), "triton");
    }

    #[test]
    fn test_error_type_configuration() {
        let error = AppError::Config(ConfigError::Validation("Validation error".to_string()));
        assert_eq!(error.error_type(), "configuration");
    }

    #[test]
    fn test_error_type_server() {
        let error = AppError::Server(ServerError::WebSocket("WS error".to_string()));
        assert_eq!(error.error_type(), "server");
    }

    #[test]
    fn test_error_type_performance() {
        let error = AppError::Performance(PerformanceError::Numa("NUMA error".to_string()));
        assert_eq!(error.error_type(), "performance");
    }

    #[test]
    fn test_error_type_io() {
        let error = AppError::Io(io::Error::new(io::ErrorKind::NotFound, "File not found"));
        assert_eq!(error.error_type(), "io");
    }

    #[test]
    fn test_error_type_network() {
        let error = AppError::Network("Network error".to_string());
        assert_eq!(error.error_type(), "network");
    }

    #[test]
    fn test_error_type_service_unavailable() {
        let error = AppError::ServiceUnavailable("Service down".to_string());
        assert_eq!(error.error_type(), "service_unavailable");
    }

    #[test]
    fn test_error_type_timeout() {
        let error = AppError::Timeout("Timeout".to_string());
        assert_eq!(error.error_type(), "timeout");
    }

    #[test]
    fn test_error_type_capacity_exceeded() {
        let error = AppError::CapacityExceeded("Queue full".to_string());
        assert_eq!(error.error_type(), "capacity_exceeded");
    }

    #[test]
    fn test_error_type_internal() {
        let error = AppError::Internal("Internal error".to_string());
        assert_eq!(error.error_type(), "internal");
    }

    #[test]
    fn test_error_type_invalid_input() {
        let error = AppError::InvalidInput("Bad input".to_string());
        assert_eq!(error.error_type(), "invalid_input");
    }

    // Test ErrorContext trait
    #[test]
    fn test_error_context_with_context() {
        let result: std::result::Result<(), io::Error> =
            Err(io::Error::new(io::ErrorKind::NotFound, "file.txt"));

        let app_result = result.with_context(|| "Failed to read file".to_string());

        assert!(app_result.is_err());
        let err_msg = format!("{:?}", app_result.unwrap_err());
        assert!(err_msg.contains("Failed to read file"));
        assert!(err_msg.contains("file.txt"));
    }

    #[test]
    fn test_error_context_with_static_context() {
        let result: std::result::Result<(), io::Error> = Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "access denied",
        ));

        let app_result = result.with_static_context("Unable to access resource");

        assert!(app_result.is_err());
        let err_msg = format!("{:?}", app_result.unwrap_err());
        assert!(err_msg.contains("Unable to access resource"));
        assert!(err_msg.contains("access denied"));
    }

    #[test]
    fn test_error_context_success() {
        let result: std::result::Result<i32, io::Error> = Ok(42);

        let app_result = result.with_context(|| "This should not be called".to_string());

        assert!(app_result.is_ok());
        assert_eq!(app_result.unwrap(), 42);
    }

    // Test IntoAppError trait
    #[test]
    fn test_into_app_error_asr() {
        let result: std::result::Result<(), AsrError> =
            Err(AsrError::Vocabulary("Missing token".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Asr(_)));
    }

    #[test]
    fn test_into_app_error_triton() {
        let result: std::result::Result<(), TritonError> =
            Err(TritonError::InvalidResponse("Bad response".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Triton(_)));
    }

    #[test]
    fn test_into_app_error_config() {
        let result: std::result::Result<(), ConfigError> =
            Err(ConfigError::Parse("Invalid TOML".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Config(_)));
    }

    #[test]
    fn test_into_app_error_server() {
        let result: std::result::Result<(), ServerError> =
            Err(ServerError::WebSocket("Connection closed".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Server(_)));
    }

    #[test]
    fn test_into_app_error_performance() {
        let result: std::result::Result<(), PerformanceError> = Err(PerformanceError::CpuAffinity(
            "Failed to set affinity".to_string(),
        ));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Performance(_)));
    }

    #[test]
    fn test_into_app_error_success() {
        let result: std::result::Result<String, AsrError> = Ok("success".to_string());

        let app_result = result.into_app_error();

        assert!(app_result.is_ok());
        assert_eq!(app_result.unwrap(), "success");
    }

    // Test timeout helpers
    #[tokio::test]
    async fn test_with_timeout_success() {
        let operation = async { Ok::<i32, io::Error>(42) };

        let result = with_timeout(operation, Duration::from_secs(1), "Test operation").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_timeout_error() {
        let operation =
            async { Err::<i32, io::Error>(io::Error::new(io::ErrorKind::NotFound, "Not found")) };

        let result = with_timeout(operation, Duration::from_secs(1), "Failed operation").await;

        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Failed operation"));
        assert!(err_msg.contains("Not found"));
    }

    #[tokio::test]
    async fn test_with_timeout_timeout() {
        let operation = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<i32, io::Error>(42)
        };

        let result = with_timeout(operation, Duration::from_millis(50), "Slow operation").await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, AppError::Timeout(_)));
        assert_eq!(format!("{}", err), "Timeout: Operation timed out");
    }

    #[tokio::test]
    async fn test_with_timeout_and_convert_success() {
        let operation = async { Ok::<String, io::Error>("hello".to_string()) };

        let result =
            with_timeout_and_convert(operation, Duration::from_secs(1), "Test conversion", |e| {
                AppError::Network(e.to_string())
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "hello");
    }

    #[tokio::test]
    async fn test_with_timeout_and_convert_error() {
        let operation = async {
            Err::<String, io::Error>(io::Error::new(io::ErrorKind::ConnectionRefused, "Refused"))
        };

        let result = with_timeout_and_convert(
            operation,
            Duration::from_secs(1),
            "Failed conversion",
            |e| AppError::Network(e.to_string()),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AppError::Network(_)));
    }

    #[tokio::test]
    async fn test_with_timeout_and_convert_timeout() {
        let operation = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok::<String, io::Error>("delayed".to_string())
        };

        let result = with_timeout_and_convert(
            operation,
            Duration::from_millis(50),
            "Timeout conversion",
            |e| AppError::Network(e.to_string()),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AppError::Timeout(_)));
    }

    // Test error message formatting
    #[test]
    fn test_audio_error_invalid_sample_rate_message() {
        let error = AudioError::InvalidSampleRate {
            expected: 16000,
            actual: 8000,
        };
        assert_eq!(
            format!("{}", error),
            "Invalid sample rate: expected 16000, got 8000"
        );
    }

    #[test]
    fn test_model_error_not_found_message() {
        let error = ModelError::NotFound {
            model_name: "encoder".to_string(),
        };
        assert_eq!(format!("{}", error), "Model not found: encoder");
    }

    #[test]
    fn test_config_error_missing_field_message() {
        let error = ConfigError::MissingField {
            field: "vocabulary_path".to_string(),
        };
        assert_eq!(
            format!("{}", error),
            "Missing required field: vocabulary_path"
        );
    }

    #[test]
    fn test_config_error_invalid_value_message() {
        let error = ConfigError::InvalidValue {
            field: "port".to_string(),
            value: "999999".to_string(),
        };
        assert_eq!(format!("{}", error), "Invalid value for port: 999999");
    }

    // Test CUDA errors when feature is enabled
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_error_http_status() {
        let error = AppError::Cuda(CudaError::Initialization("GPU not found".to_string()));

        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_error_type() {
        let error = AppError::Cuda(CudaError::MemoryAllocation("Out of VRAM".to_string()));
        assert_eq!(error.error_type(), "cuda");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_into_app_error_cuda() {
        let result: std::result::Result<(), CudaError> =
            Err(CudaError::KernelExecution("Kernel failed".to_string()));

        let app_result = result.into_app_error();

        assert!(app_result.is_err());
        assert!(matches!(app_result.unwrap_err(), AppError::Cuda(_)));
    }
}
