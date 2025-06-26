//! HTTP and WebSocket request handlers.
//!
//! This module provides the HTTP and WebSocket handlers for the ASR service.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{ws::WebSocket, Path, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tokio::sync::oneshot;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{error, info};

use crate::asr::types::{AsrResponse, StreamStatus};
use crate::config::audio::{MAX_BATCH_AUDIO_LENGTH_SECS, SAMPLE_RATE};
use crate::error::{AppError, Result};
use crate::server::stream::create_stream;
use crate::server::AppState;

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
    let (stream_id, handle, processor) = create_stream(ws, state.clone());

    info!("Stream {} started for model {}", stream_id, model);

    // Register stream
    state.active_streams.insert(stream_id.clone(), handle);

    // Process stream
    let (_shutdown_tx, shutdown_rx) = oneshot::channel();
    processor.process(shutdown_rx).await;

    // Cleanup
    state.active_streams.remove(&stream_id);
    state.metrics.decrement_stream();

    info!("Stream {} ended", stream_id);
    drop(permit);
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

    // Validate audio length
    let audio_length_secs = request.audio_buffer.len() as f32 / (SAMPLE_RATE as f32 * 2.0);
    if audio_length_secs > MAX_BATCH_AUDIO_LENGTH_SECS {
        return Err(AppError::Validation(format!(
            "Audio too long: {:.1}s (max: {}s)",
            audio_length_secs, MAX_BATCH_AUDIO_LENGTH_SECS
        )));
    }

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
    let mut metadata = HashMap::new();
    if let serde_json::Value::Object(map) = metadata_value {
        for (k, v) in map {
            metadata.insert(k, v);
        }
    }

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
        "service": "wav2vec2-rust-server",
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
