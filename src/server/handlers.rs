//! HTTP and WebSocket request handlers.
//!
//! This module provides the HTTP and WebSocket handlers for the ASR service.

use std::sync::Arc;

use axum::{
    extract::{ws::WebSocket, Path, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

use crate::asr::types::{AsrResponse, StreamStatus};
use crate::constants::audio::{MAX_BATCH_AUDIO_LENGTH_SECS, SAMPLE_RATE};
use crate::error::{AppError, Result, ServerError};
use crate::server::stream::create_stream;
use crate::server::AppState;

/// RAII guard to ensure stream cleanup happens even if processing fails
struct StreamCleanupGuard<'a> {
    stream_id: String,
    state: Arc<AppState>,
    #[allow(dead_code)] // Field is used for RAII - permit is held until drop
    permit: Option<tokio::sync::SemaphorePermit<'a>>,
}

impl Drop for StreamCleanupGuard<'_> {
    fn drop(&mut self) {
        // Remove from active streams
        self.state.active_streams.remove(&self.stream_id);
        // Decrement metrics
        self.state.metrics.decrement_stream();
        // Permit will be automatically dropped
    }
}

/// Request body for batch ASR.
#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    /// Raw audio bytes (16-bit PCM)
    audio_buffer: Vec<u8>,

    /// Optional description
    #[serde(default)]
    _description: Option<String>,

    /// Optional opaque data to be returned in the response
    #[serde(default)]
    opaque: Option<serde_json::Value>,

    /// Whether to return incremental results
    #[serde(default = "default_true")]
    _incremental: bool,

    /// Optional model name
    #[serde(default)]
    _model: Option<String>,
}

impl BatchRequest {
    /// Validate the batch request.
    pub fn validate(&self) -> Result<()> {
        // Check audio buffer size
        if self.audio_buffer.is_empty() {
            return Err(AppError::Server(ServerError::RequestValidation(
                "Audio buffer cannot be empty".to_string(),
            )));
        }

        // Check if audio buffer length is even (16-bit samples)
        crate::asr::types::validate_pcm_audio_format(&self.audio_buffer)?;

        // Check maximum size (prevent DoS)
        const MAX_AUDIO_BYTES: usize = 100 * 1024 * 1024; // 100MB
        if self.audio_buffer.len() > MAX_AUDIO_BYTES {
            return Err(AppError::Server(ServerError::RequestValidation(format!(
                "Audio buffer too large: {} bytes (max: {} bytes)",
                self.audio_buffer.len(),
                MAX_AUDIO_BYTES
            ))));
        }

        // Validate audio length in seconds
        let audio_length_secs = self.audio_buffer.len() as f32 / (SAMPLE_RATE.as_f32() * 2.0);
        if audio_length_secs > MAX_BATCH_AUDIO_LENGTH_SECS as f32 {
            return Err(AppError::Server(ServerError::RequestValidation(format!(
                "Audio too long: {:.1}s (max: {}s)",
                audio_length_secs, MAX_BATCH_AUDIO_LENGTH_SECS
            ))));
        }

        // Validate opaque data size if present
        if let Some(ref opaque) = self.opaque {
            let opaque_str = serde_json::to_string(opaque).map_err(|_| {
                AppError::Server(ServerError::RequestValidation(
                    "Invalid opaque data format".to_string(),
                ))
            })?;
            if opaque_str.len() > 10_000 {
                return Err(AppError::Server(ServerError::RequestValidation(
                    "Opaque data too large (max: 10KB)".to_string(),
                )));
            }
        }

        Ok(())
    }
}

fn default_true() -> bool {
    true
}

/// Handle WebSocket upgrade for streaming ASR.
pub async fn handle_stream(
    ws: WebSocketUpgrade,
    Path(model): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_stream_connection(socket, state, model))
}

/// Handle a WebSocket connection.
async fn handle_stream_connection(ws: WebSocket, state: Arc<AppState>, model: String) {
    // Check concurrency limit
    let permit = match state.stream_semaphore.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            state.metrics.record_rejection();
            error!("Rejected stream request: too many concurrent streams");
            return;
        }
    };

    state.metrics.increment_stream();

    // Create stream and handle
    let (stream_id, handle, processor, shutdown_rx) = create_stream(ws, state.clone());

    info!("Stream {} started for model {}", stream_id, model);

    // Register stream
    state.active_streams.insert(stream_id.clone(), handle);

    // Ensure cleanup happens regardless of how processing ends
    let _cleanup_guard = StreamCleanupGuard {
        stream_id: stream_id.clone(),
        state: state.clone(),
        permit: Some(permit),
    };

    // Process stream
    processor.process(shutdown_rx).await;

    info!("Stream {} ended", stream_id);
    // cleanup_guard will automatically clean up when dropped
}

/// Handle batch ASR request.
pub async fn handle_batch(
    Path(_model): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchRequest>,
) -> Result<Json<AsrResponse>> {
    // Check concurrency limit
    let _permit = state.batch_semaphore.try_acquire().map_err(|_| {
        state.metrics.record_rejection();
        AppError::CapacityExceeded("Too many concurrent batch requests".to_string())
    })?;

    state.metrics.increment_batch();

    // Validate request
    request.validate()?;

    // Process audio
    let transcription = state
        .asr_pipeline
        .process_batch(&request.audio_buffer)
        .await?;

    // Convert to response
    let metadata_value = serde_json::json!({
        "audio_length_samples": transcription.audio_length_samples,
        "features_length": transcription.features_length,
        "encoded_length": transcription.encoded_length,
        "tokens": transcription.tokens,
    });

    // Convert to HashMap for AsrResponse
    let metadata = crate::asr::types::json_value_to_metadata(metadata_value);

    let response = AsrResponse {
        transcription: transcription.text,
        status: StreamStatus::Complete,
        message: None,
        metadata: Some(metadata),
        opaque: request.opaque,
    };

    state.metrics.decrement_batch();

    Ok(Json(response))
}

/// Health check endpoint.
pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "amira-rust-asr-server",
        "version": "1.0.0"
    }))
}

/// Metrics endpoint.
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(state.metrics.to_json())
}

/// Reset batch count endpoint.
/// This is useful for clearing zombie requests after server errors.
pub async fn reset_batch_count(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.metrics.reset_batch_count();
    Json(serde_json::json!({
        "status": "success",
        "message": "Batch count reset successfully"
    }))
}

/// Create the application router.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v2/decode/stream/:model", get(handle_stream))
        .route("/v2/decode/batch/:model", post(handle_batch))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_handler))
        .route("/admin/reset-batch-count", post(reset_batch_count))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a valid BatchRequest
    fn create_valid_request() -> BatchRequest {
        BatchRequest {
            audio_buffer: vec![0u8; 32000], // 1 second of 16-bit PCM at 16kHz
            _description: None,
            opaque: None,
            _incremental: true,
            _model: None,
        }
    }

    // Test BatchRequest validation - empty buffer
    #[test]
    fn test_batch_request_validate_empty_buffer() {
        let request = BatchRequest {
            audio_buffer: vec![],
            _description: None,
            opaque: None,
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Audio buffer cannot be empty"));
    }

    // Test BatchRequest validation - odd length buffer (invalid 16-bit PCM)
    #[test]
    fn test_batch_request_validate_odd_length() {
        let request = BatchRequest {
            audio_buffer: vec![0u8; 101], // Odd number
            _description: None,
            opaque: None,
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("must be even"));
    }

    // Test BatchRequest validation - buffer too large
    #[test]
    fn test_batch_request_validate_buffer_too_large() {
        let request = BatchRequest {
            audio_buffer: vec![0u8; 101 * 1024 * 1024], // 101MB
            _description: None,
            opaque: None,
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("too large"));
    }

    // Test BatchRequest validation - audio too long in duration
    #[test]
    fn test_batch_request_validate_audio_too_long() {
        // MAX_BATCH_AUDIO_LENGTH_SECS is typically 60 seconds
        // At 16kHz 16-bit: 60s * 16000 samples/s * 2 bytes/sample = 1,920,000 bytes
        // Let's create something longer
        let max_bytes = (MAX_BATCH_AUDIO_LENGTH_SECS as f32 * SAMPLE_RATE.as_f32() * 2.0) as usize;

        let request = BatchRequest {
            audio_buffer: vec![0u8; max_bytes + 10000], // Slightly over limit
            _description: None,
            opaque: None,
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("too long"));
    }

    // Test BatchRequest validation - valid request
    #[test]
    fn test_batch_request_validate_success() {
        let request = create_valid_request();
        let result = request.validate();
        assert!(result.is_ok());
    }

    // Test BatchRequest validation - opaque data too large
    #[test]
    fn test_batch_request_validate_opaque_too_large() {
        // Create a large JSON object
        let large_string = "a".repeat(11000);
        let large_opaque = serde_json::json!({ "data": large_string });

        let request = BatchRequest {
            audio_buffer: vec![0u8; 32000],
            _description: None,
            opaque: Some(large_opaque),
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("Opaque data too large"));
    }

    // Test BatchRequest validation - valid opaque data
    #[test]
    fn test_batch_request_validate_valid_opaque() {
        let request = BatchRequest {
            audio_buffer: vec![0u8; 32000],
            _description: None,
            opaque: Some(serde_json::json!({ "session_id": "test123", "user_id": 456 })),
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_ok());
    }

    // Test BatchRequest deserialization
    #[test]
    fn test_batch_request_deserialization() {
        let json_data = r#"{
            "audio_buffer": [1, 2, 3, 4],
            "opaque": {"test": "value"}
        }"#;

        let request: BatchRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(request.audio_buffer.len(), 4);
        assert!(request.opaque.is_some());
        assert_eq!(request._incremental, true); // default_true
    }

    // Test BatchRequest deserialization with defaults
    #[test]
    fn test_batch_request_deserialization_defaults() {
        let json_data = r#"{
            "audio_buffer": [1, 2, 3, 4]
        }"#;

        let request: BatchRequest = serde_json::from_str(json_data).unwrap();
        assert_eq!(request.audio_buffer.len(), 4);
        assert!(request.opaque.is_none());
        assert!(request._description.is_none());
        assert!(request._model.is_none());
        assert_eq!(request._incremental, true);
    }

    // Test health check endpoint
    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;

        // Convert response to JSON
        let body = axum::body::to_bytes(response.into_response().into_body(), usize::MAX)
            .await
            .unwrap();

        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["status"], "healthy");
        assert_eq!(json["service"], "wav2vec2-rust-server");
        assert_eq!(json["version"], "1.0.0");
    }

    // Test default_true helper
    #[test]
    fn test_default_true() {
        assert_eq!(default_true(), true);
    }

    // Test boundary: exactly at max audio length
    #[test]
    fn test_batch_request_validate_exactly_max_length() {
        let max_bytes = (MAX_BATCH_AUDIO_LENGTH_SECS as f32 * SAMPLE_RATE.as_f32() * 2.0) as usize;

        // Make sure it's even
        let max_bytes = if max_bytes % 2 == 0 {
            max_bytes
        } else {
            max_bytes - 1
        };

        let request = BatchRequest {
            audio_buffer: vec![0u8; max_bytes],
            _description: None,
            opaque: None,
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        // Should be valid at exactly the max
        assert!(result.is_ok());
    }

    // Test boundary: exactly at 10KB opaque data
    #[test]
    fn test_batch_request_validate_opaque_boundary() {
        // Create exactly 10KB of data (accounting for JSON overhead)
        let data_str = "a".repeat(9950); // Leave room for JSON structure
        let opaque = serde_json::json!({ "d": data_str });

        let request = BatchRequest {
            audio_buffer: vec![0u8; 1000],
            _description: None,
            opaque: Some(opaque),
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        // This should be valid (just under 10KB)
        assert!(result.is_ok());
    }

    // Test multiple validation errors
    #[test]
    fn test_batch_request_multiple_errors() {
        // Empty buffer should fail first
        let request = BatchRequest {
            audio_buffer: vec![],
            _description: None,
            opaque: Some(serde_json::json!({ "data": "x".repeat(11000) })),
            _incremental: true,
            _model: None,
        };

        let result = request.validate();
        assert!(result.is_err());
        // Should fail on empty buffer check first
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("cannot be empty"));
    }
}
