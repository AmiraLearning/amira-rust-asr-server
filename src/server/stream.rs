//! WebSocket stream processing.
//!
//! This module provides the stream processing functionality for
//! real-time ASR via WebSockets.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use futures::stream::StreamExt;
use serde::Deserialize;
use tokio::select;
use tokio::sync::oneshot;
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::asr::types::{AsrResponse, StreamStatus};
use crate::asr::{AudioRingBuffer, IncrementalAsr};
use crate::config::audio::{BUFFER_CAPACITY, MIN_PARTIAL_TRANSCRIPTION_SAMPLES};
use crate::config::streaming::{
    CONTROL_BYTE_END, CONTROL_BYTE_KEEPALIVE, KEEPALIVE_CHECK_PERIOD_MS, STREAM_TIMEOUT_SECS,
};
use crate::error::{AppError, Result};
use crate::server::AppState;

/// Handle for a streaming ASR session.
pub struct StreamHandle {
    /// Stream ID
    pub id: String,

    /// Start time
    pub start_time: Instant,

    /// Last activity time
    pub last_activity: Arc<tokio::sync::RwLock<Instant>>,

    /// Shutdown channel
    pub shutdown_tx: oneshot::Sender<()>,
}

/// Request for a streaming ASR session.
#[derive(Debug, Deserialize)]
pub struct StreamRequest {
    /// Opaque data provided by the client
    #[serde(default)]
    pub opaque: Option<serde_json::Value>,
}

/// High-performance stream processor for real-time ASR.
pub struct StreamProcessor {
    /// The WebSocket connection
    ws: WebSocket,

    /// The application state
    _state: Arc<AppState>,

    /// The stream ID
    stream_id: String,

    /// The audio buffer for raw audio data
    audio_buffer: AudioRingBuffer,

    /// Incremental ASR processor
    incremental_asr: IncrementalAsr<dyn crate::asr::AsrPipeline>,

    /// The last transcription
    last_transcription: String,

    /// Whether the stream is paused
    is_paused: bool,
}

impl StreamProcessor {
    /// Create a new stream processor.
    ///
    /// # Arguments
    /// * `ws` - The WebSocket connection
    /// * `state` - The application state
    /// * `stream_id` - The stream ID
    ///
    /// # Returns
    /// A new stream processor
    pub fn new(ws: WebSocket, state: Arc<AppState>, stream_id: String) -> Self {
        // Configuration for incremental processing
        const CHUNK_SIZE: f32 = 2.0; // 2 seconds
        const LEADING_CONTEXT: f32 = 1.0; // 1 second
        const TRAILING_CONTEXT: f32 = 0.5; // 0.5 seconds
        const BUFFER_CAPACITY_SECONDS: f32 = 10.0; // 10 seconds

        // Create incremental ASR processor
        let incremental_asr = IncrementalAsr::new(
            state.asr_pipeline.clone(),
            state.vocabulary.clone(),
            CHUNK_SIZE,
            LEADING_CONTEXT,
            TRAILING_CONTEXT,
            BUFFER_CAPACITY_SECONDS,
        );

        Self {
            ws,
            _state: state,
            stream_id,
            audio_buffer: AudioRingBuffer::new(BUFFER_CAPACITY),
            incremental_asr,
            last_transcription: String::new(),
            is_paused: false,
        }
    }

    /// Process a WebSocket stream.
    ///
    /// # Arguments
    /// * `shutdown_rx` - Channel to receive shutdown signal
    pub async fn process(mut self, mut shutdown_rx: oneshot::Receiver<()>) {
        let mut keepalive_interval = interval(Duration::from_millis(KEEPALIVE_CHECK_PERIOD_MS));
        let mut last_activity = Instant::now();

        loop {
            select! {
                // Handle incoming WebSocket messages
                msg = self.ws.next() => {
                    match msg {
                        Some(Ok(Message::Binary(data))) => {
                            last_activity = Instant::now();
                            if let Err(e) = self.handle_audio_chunk(data).await {
                                error!("Error handling audio chunk: {}", e);
                                let _ = self.send_error(&e.to_string()).await;
                                break;
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            info!("WebSocket closed by client for stream {}", self.stream_id);
                            break;
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error for stream {}: {}", self.stream_id, e);
                            break;
                        }
                        None => break,
                        _ => {}
                    }
                }

                // Keepalive and timeout check
                _ = keepalive_interval.tick() => {
                    if last_activity.elapsed() > Duration::from_secs(STREAM_TIMEOUT_SECS) {
                        warn!("Stream {} timed out", self.stream_id);
                        let _ = self.send_error("Stream timeout").await;
                        break;
                    }

                    if self.is_paused {
                        let response = AsrResponse {
                            transcription: self.last_transcription.clone(),
                            status: StreamStatus::Paused,
                            message: None,
                            metadata: None,
                            opaque: None,
                        };
                        let _ = self.ws.send(Message::Text(serde_json::to_string(&response).unwrap())).await;
                    }
                }

                // Shutdown signal
                _ = &mut shutdown_rx => {
                    info!("Stream {} received shutdown signal", self.stream_id);
                    break;
                }
            }
        }

        // Process any remaining audio
        if !self.audio_buffer.is_empty() {
            let _ = self.process_final_audio().await;
        }
    }

    /// Handle an audio chunk from the WebSocket.
    ///
    /// # Arguments
    /// * `data` - The audio chunk
    ///
    /// # Returns
    /// Ok(()) if successful, or an error
    async fn handle_audio_chunk(&mut self, data: Vec<u8>) -> Result<()> {
        self.is_paused = false;

        // Handle control bytes
        if data.len() == 1 {
            match data[0] {
                CONTROL_BYTE_END => {
                    debug!(
                        "End of stream signal received for stream {}",
                        self.stream_id
                    );
                    return Err(AppError::Validation("End of stream".to_string()));
                }
                CONTROL_BYTE_KEEPALIVE => {
                    self.is_paused = true;
                    return Ok(());
                }
                _ => return Err(AppError::Validation("Unknown control byte".to_string())),
            }
        }

        // Buffer audio data
        self.audio_buffer.write(&data)?;

        // Check if we should process
        if self.audio_buffer.available_read() >= MIN_PARTIAL_TRANSCRIPTION_SAMPLES * 2 {
            self.process_buffered_audio(false).await?;
        }

        Ok(())
    }

    /// Process buffered audio.
    ///
    /// # Arguments
    /// * `is_final` - Whether this is the final audio chunk
    ///
    /// # Returns
    /// Ok(()) if successful, or an error
    async fn process_buffered_audio(&mut self, is_final: bool) -> Result<()> {
        let available = self.audio_buffer.available_read();
        if available == 0 {
            return Ok(());
        }

        let audio_data = self
            .audio_buffer
            .read(available)
            .ok_or_else(|| AppError::Audio("Failed to read from buffer".to_string()))?;

        // Process audio with incremental ASR processor
        match timeout(
            Duration::from_secs(5),
            self.incremental_asr.process_chunk(&audio_data),
        )
        .await
        {
            Ok(Ok(transcription)) => {
                // Update transcription
                self.last_transcription = transcription;

                // Build metadata
                let audio_length = self.incremental_asr.audio_length();
                let metadata_value = serde_json::json!({
                    "audio_length_seconds": audio_length,
                });

                // Convert to HashMap for AsrResponse
                let mut metadata = HashMap::new();
                if let serde_json::Value::Object(map) = metadata_value {
                    for (k, v) in map {
                        metadata.insert(k, v);
                    }
                }

                // Build response for client
                let response = AsrResponse {
                    transcription: self.last_transcription.clone(),
                    status: if is_final {
                        StreamStatus::Complete
                    } else {
                        StreamStatus::Active
                    },
                    message: None,
                    metadata: Some(metadata),
                    opaque: None,
                };

                let json = serde_json::to_string(&response)?;
                self.ws
                    .send(Message::Text(json))
                    .await
                    .map_err(|e| AppError::Internal(format!("WebSocket send error: {}", e)))?;
            }
            Ok(Err(e)) => {
                error!("Inference error for stream {}: {}", self.stream_id, e);
                self.send_error(&e.to_string()).await?;
            }
            Err(_) => {
                error!("Inference timeout for stream {}", self.stream_id);
                self.send_error("Inference timeout").await?;
            }
        }

        Ok(())
    }

    /// Process final audio.
    ///
    /// # Returns
    /// Ok(()) if successful, or an error
    async fn process_final_audio(&mut self) -> Result<()> {
        self.process_buffered_audio(true).await
    }

    /// Send an error response.
    ///
    /// # Arguments
    /// * `message` - The error message
    ///
    /// # Returns
    /// Ok(()) if successful, or an error
    async fn send_error(&mut self, message: &str) -> Result<()> {
        let response = AsrResponse {
            transcription: String::new(),
            status: StreamStatus::Error,
            message: Some(message.to_string()),
            metadata: None,
            opaque: None,
        };

        let json = serde_json::to_string(&response)?;
        self.ws
            .send(Message::Text(json))
            .await
            .map_err(|e| AppError::Internal(format!("WebSocket send error: {}", e)))?;
        Ok(())
    }
}

/// Create a new stream processor and stream handle.
///
/// # Arguments
/// * `ws` - The WebSocket connection
/// * `state` - The application state
///
/// # Returns
/// A tuple of (stream_id, stream_handle, stream_processor, shutdown_receiver)
pub fn create_stream(
    ws: WebSocket,
    state: Arc<AppState>,
) -> (String, StreamHandle, StreamProcessor, oneshot::Receiver<()>) {
    let stream_id = Uuid::new_v4().to_string();
    let now = Instant::now();
    let last_activity = Arc::new(tokio::sync::RwLock::new(now));

    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    // Create stream handle
    let handle = StreamHandle {
        id: stream_id.clone(),
        start_time: now,
        last_activity: last_activity.clone(),
        shutdown_tx,
    };

    // Create stream processor
    let processor = StreamProcessor::new(ws, state, stream_id.clone());

    (stream_id, handle, processor, shutdown_rx)
}
