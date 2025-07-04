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
// use tracing::{debug, error, info, warn};  // Temporarily disabled
macro_rules! debug { ($($tt:tt)*) => {}; }
macro_rules! error { ($($tt:tt)*) => {}; }
macro_rules! info { ($($tt:tt)*) => {}; }
macro_rules! warn { ($($tt:tt)*) => {}; }
use uuid::Uuid;

use crate::asr::types::{AsrResponse, StreamStatus};
use crate::asr::{AudioRingBuffer, IncrementalAsr};
// use crate::performance::specialized_pools::spawn_io;  // Disabled for now
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
    incremental_asr: IncrementalAsr,

    /// The last transcription
    last_transcription: String,

    /// Whether the stream is paused
    is_paused: bool,

    /// Rate limiting: track message count in current window
    message_count: u32,

    /// Rate limiting: start time of current window  
    window_start: Instant,
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
            message_count: 0,
            window_start: Instant::now(),
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
                        // Use optimized async serialization to avoid blocking keepalive loop
                        let _ = self.send_response_async(response).await;
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

        // Validate message size to prevent DoS attacks
        const MAX_CHUNK_SIZE: usize = 1024 * 1024; // 1MB per chunk
        if data.len() > MAX_CHUNK_SIZE {
            return Err(AppError::Validation(format!(
                "Audio chunk too large: {} bytes (max: {} bytes)",
                data.len(),
                MAX_CHUNK_SIZE
            )));
        }

        // Rate limiting: check message frequency
        self.check_rate_limit()?;

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

        // Validate audio data format (must be even for 16-bit PCM)
        if data.len() % 2 != 0 {
            return Err(AppError::Validation(
                "Audio data length must be even for 16-bit PCM".to_string(),
            ));
        }

        // Check if data is empty (but not a control byte)
        if data.is_empty() {
            return Err(AppError::Validation(
                "Empty audio chunk received".to_string(),
            ));
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

                // Move serialization off the main event loop to prevent blocking
                self.send_response_async(response).await?;
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

        // Use async serialization for consistency
        self.send_response_async(response).await
    }

    /// Send response to client with off-main-thread serialization to prevent blocking.
    /// 
    /// This method moves JSON serialization to a dedicated I/O thread to avoid blocking
    /// the main WebSocket event loop during serialization of potentially large responses.
    ///
    /// # Arguments
    /// * `response` - The ASR response to serialize and send
    ///
    /// # Returns
    /// Ok(()) if successful, or an error
    async fn send_response_async(&mut self, response: AsrResponse) -> Result<()> {
        // Create a channel for communicating serialization result
        let (tx, rx) = oneshot::channel::<Result<String>>();
        
        // Spawn serialization on dedicated I/O thread pool to avoid blocking main event loop
        tokio::spawn(async move {
            let json_result = serde_json::to_string(&response)
                .map_err(|e| AppError::Internal(format!("JSON serialization error: {}", e)));
            
            // Send result back to main thread
            let _ = tx.send(json_result);
        });

        // Wait for serialization to complete
        let json = match rx.await {
            Ok(Ok(json)) => json,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(AppError::Internal("Serialization task failed".to_string())),
        };

        // Send the pre-serialized JSON on the main thread
        self.ws
            .send(Message::Text(json))
            .await
            .map_err(|e| AppError::Internal(format!("WebSocket send error: {}", e)))?;
            
        Ok(())
    }

    /// Check rate limiting for incoming messages.
    ///
    /// Implements a simple sliding window rate limiter to prevent abuse.
    /// Allows up to 100 messages per second for real-time audio streaming.
    fn check_rate_limit(&mut self) -> Result<()> {
        const MAX_MESSAGES_PER_WINDOW: u32 = 100;
        const WINDOW_DURATION_SECS: u64 = 1;

        let now = Instant::now();
        let window_duration = Duration::from_secs(WINDOW_DURATION_SECS);

        // Check if we need to reset the window
        if now.duration_since(self.window_start) >= window_duration {
            self.message_count = 0;
            self.window_start = now;
        }

        // Increment message count
        self.message_count += 1;

        // Check if rate limit exceeded
        if self.message_count > MAX_MESSAGES_PER_WINDOW {
            warn!(
                "Rate limit exceeded for stream {}: {} messages in current window",
                self.stream_id, self.message_count
            );
            return Err(AppError::Validation(format!(
                "Rate limit exceeded: max {} messages per second allowed",
                MAX_MESSAGES_PER_WINDOW
            )));
        }

        // Log warning when approaching limit
        if self.message_count > MAX_MESSAGES_PER_WINDOW * 8 / 10 {
            debug!(
                "Stream {} approaching rate limit: {}/{} messages",
                self.stream_id, self.message_count, MAX_MESSAGES_PER_WINDOW
            );
        }

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
