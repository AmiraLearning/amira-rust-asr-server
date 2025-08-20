//! Trait abstractions for ASR processing components.
//!
//! This module provides clean abstractions for the different components of the ASR pipeline,
//! allowing for better testability, modularity, and potential backend swapping.

use crate::error::{AsrError, ModelError};
use crate::types::{AudioBuffer, ConfidenceScore, TensorShape, TokenId};
use async_trait::async_trait;
use std::sync::Arc;

/// Transcription result from ASR processing.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The transcribed text.
    pub text: String,
    /// Confidence score for the transcription.
    pub confidence: ConfidenceScore,
    /// Individual token information.
    pub tokens: Vec<TokenInfo>,
    /// Processing metadata.
    pub metadata: TranscriptionMetadata,
}

/// Individual token information.
#[derive(Debug, Clone)]
pub struct TokenInfo {
    /// The token ID.
    pub id: TokenId,
    /// The token text.
    pub text: String,
    /// Start time in the audio.
    pub start_time: f32,
    /// End time in the audio.
    pub end_time: f32,
    /// Confidence score for this token.
    pub confidence: ConfidenceScore,
}

/// Partial transcription for streaming processing.
#[derive(Debug, Clone)]
pub struct PartialTranscription {
    /// Current transcription text.
    pub text: String,
    /// Whether this is a final result.
    pub is_final: bool,
    /// Confidence score.
    pub confidence: ConfidenceScore,
    /// Tokens in this partial result.
    pub tokens: Vec<TokenInfo>,
}

/// Complete transcription for batch processing.
#[derive(Debug, Clone)]
pub struct CompleteTranscription {
    /// Final transcription text.
    pub text: String,
    /// Overall confidence score.
    pub confidence: ConfidenceScore,
    /// All tokens with timing information.
    pub tokens: Vec<TokenInfo>,
    /// Processing metadata.
    pub metadata: TranscriptionMetadata,
}

/// Metadata about the transcription process.
#[derive(Debug, Clone)]
pub struct TranscriptionMetadata {
    /// Processing time in milliseconds.
    pub processing_time_ms: f32,
    /// Audio duration in seconds.
    pub audio_duration_secs: f32,
    /// Real-time factor (processing_time / audio_duration).
    pub real_time_factor: f32,
    /// Model information used.
    pub model_info: ModelInfo,
}

/// Model information.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Model type (e.g., "RNN-T", "Transformer").
    pub model_type: String,
}

/// Decoder state for streaming processing.
#[derive(Debug, Clone)]
pub struct DecoderState {
    /// Hidden states for the decoder.
    pub hidden_states: Vec<f32>,
    /// Current prediction history.
    pub prediction_history: Vec<TokenId>,
    /// Current time step.
    pub time_step: usize,
    /// Internal state version for consistency checking.
    pub state_version: u64,
}

impl DecoderState {
    /// Create a new decoder state.
    pub fn new() -> Self {
        Self {
            hidden_states: Vec::new(),
            prediction_history: Vec::new(),
            time_step: 0,
            state_version: 0,
        }
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.hidden_states.clear();
        self.prediction_history.clear();
        self.time_step = 0;
        self.state_version += 1;
    }

    /// Check if the state is empty.
    pub fn is_empty(&self) -> bool {
        self.hidden_states.is_empty()
    }
}

impl Default for DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming ASR processor for real-time audio processing.
#[async_trait]
pub trait StreamingAsrProcessor: Send + Sync {
    /// Process an audio chunk and update the decoder state.
    async fn process_chunk(
        &self,
        audio: &AudioBuffer,
        state: &mut DecoderState,
    ) -> Result<PartialTranscription, AsrError>;

    /// Finalize the current transcription.
    async fn finalize(&self, state: &mut DecoderState) -> Result<CompleteTranscription, AsrError>;

    /// Reset the processor state.
    async fn reset(&self, state: &mut DecoderState) -> Result<(), AsrError>;

    /// Get the expected audio chunk size for optimal processing.
    fn optimal_chunk_size(&self) -> usize;

    /// Get the minimum audio chunk size.
    fn min_chunk_size(&self) -> usize;
}

/// Batch ASR processor for complete audio processing.
#[async_trait]
pub trait BatchAsrProcessor: Send + Sync {
    /// Process complete audio and return transcription.
    async fn process_complete(
        &self,
        audio: &AudioBuffer,
    ) -> Result<CompleteTranscription, AsrError>;

    /// Process multiple audio buffers in batch.
    async fn process_batch(
        &self,
        audio_batch: &[AudioBuffer],
    ) -> Result<Vec<CompleteTranscription>, AsrError>;

    /// Get the maximum audio length that can be processed.
    fn max_audio_length(&self) -> usize;

    /// Get the optimal batch size for processing.
    fn optimal_batch_size(&self) -> usize;
}

/// Model input trait for type-safe model operations.
pub trait ModelInput: Send + Sync {
    /// Get the input tensor shape.
    fn shape(&self) -> TensorShape;

    /// Get the input data as a slice.
    fn data(&self) -> &[f32];

    /// Validate the input data.
    fn validate(&self) -> Result<(), ModelError>;
}

/// Model output trait for type-safe model operations.
pub trait ModelOutput: Send + Sync {
    /// Create from raw tensor data.
    fn from_tensor_data(data: Vec<f32>, shape: TensorShape) -> Result<Self, ModelError>
    where
        Self: Sized;

    /// Get the output tensor shape.
    fn shape(&self) -> TensorShape;

    /// Get the output data as a slice.
    fn data(&self) -> &[f32];

    /// Validate the output data.
    fn validate(&self) -> Result<(), ModelError>;
}

/// Abstract model backend for different inference engines.
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Error type for this backend.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Perform inference with the model.
    async fn infer<I, O>(&self, input: I) -> Result<O, Self::Error>
    where
        I: ModelInput + Send,
        O: ModelOutput + Send;

    /// Get model information.
    fn model_info(&self) -> ModelInfo;

    /// Check if the model is ready for inference.
    async fn is_ready(&self) -> bool;

    /// Get the model's input shape requirements.
    fn input_shape(&self) -> TensorShape;

    /// Get the model's output shape.
    fn output_shape(&self) -> TensorShape;
}

/// Audio preprocessor trait for feature extraction.
#[async_trait]
pub trait AudioPreprocessor: Send + Sync {
    /// Preprocess audio into features.
    async fn preprocess(&self, audio: &AudioBuffer) -> Result<AudioFeatures, AsrError>;

    /// Get the expected input sample rate.
    fn expected_sample_rate(&self) -> crate::types::SampleRate;

    /// Get the output feature dimensions.
    fn output_dimensions(&self) -> (usize, usize); // (time_steps, feature_dim)
}

/// Audio features output from preprocessing.
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// Feature vectors (time_steps x feature_dim).
    pub features: Vec<Vec<f32>>,
    /// Feature dimensions.
    pub dimensions: (usize, usize),
    /// Original audio duration.
    pub audio_duration: std::time::Duration,
}

impl ModelInput for AudioFeatures {
    fn shape(&self) -> TensorShape {
        TensorShape::new(vec![self.dimensions.0, self.dimensions.1])
    }

    fn data(&self) -> &[f32] {
        // This is a simplified implementation - in practice, you'd flatten the 2D features
        &[]
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.features.is_empty() {
            return Err(ModelError::Preprocessing("Empty features".to_string()));
        }
        Ok(())
    }
}

/// Encoder output containing encoded audio features.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Encoded features.
    pub encoded_features: Vec<f32>,
    /// Output shape.
    pub shape: TensorShape,
    /// Encoder hidden states.
    pub hidden_states: Vec<f32>,
}

impl ModelOutput for EncoderOutput {
    fn from_tensor_data(data: Vec<f32>, shape: TensorShape) -> Result<Self, ModelError> {
        Ok(Self {
            encoded_features: data,
            shape,
            hidden_states: Vec::new(),
        })
    }

    fn shape(&self) -> TensorShape {
        self.shape.clone()
    }

    fn data(&self) -> &[f32] {
        &self.encoded_features
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.encoded_features.is_empty() {
            return Err(ModelError::Postprocessing(
                "Empty encoded features".to_string(),
            ));
        }
        Ok(())
    }
}

/// Decoder output containing predicted tokens.
#[derive(Debug, Clone)]
pub struct DecoderOutput {
    /// Predicted token logits.
    pub logits: Vec<f32>,
    /// Output shape.
    pub shape: TensorShape,
    /// Updated decoder state.
    pub decoder_state: Vec<f32>,
}

impl ModelOutput for DecoderOutput {
    fn from_tensor_data(data: Vec<f32>, shape: TensorShape) -> Result<Self, ModelError> {
        Ok(Self {
            logits: data,
            shape,
            decoder_state: Vec::new(),
        })
    }

    fn shape(&self) -> TensorShape {
        self.shape.clone()
    }

    fn data(&self) -> &[f32] {
        &self.logits
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.logits.is_empty() {
            return Err(ModelError::Postprocessing("Empty logits".to_string()));
        }
        Ok(())
    }
}

/// Vocabulary trait for token-text conversion.
pub trait Vocabulary: Send + Sync {
    /// Get the vocabulary size.
    fn size(&self) -> usize;

    /// Convert token ID to text.
    fn id_to_token(&self, id: TokenId) -> Option<&str>;

    /// Convert text to token ID.
    fn token_to_id(&self, token: &str) -> Option<TokenId>;

    /// Get the blank token ID.
    fn blank_token_id(&self) -> TokenId;

    /// Get the unknown token ID.
    fn unknown_token_id(&self) -> TokenId;

    /// Check if a token ID is valid.
    fn is_valid_token(&self, id: TokenId) -> bool;

    /// Convert a sequence of token IDs to text.
    fn decode(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.id_to_token(id))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Memory pool trait for efficient resource management.
pub trait MemoryPool<T>: Send + Sync {
    /// Get an object from the pool.
    fn get(&self) -> Option<T>;

    /// Return an object to the pool.
    fn put(&self, item: T);

    /// Get the current pool size.
    fn size(&self) -> usize;

    /// Get the number of available objects.
    fn available(&self) -> usize;

    /// Get pool statistics.
    fn stats(&self) -> PoolStats;
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total objects created.
    pub total_created: usize,
    /// Total objects borrowed.
    pub total_borrowed: usize,
    /// Total objects returned.
    pub total_returned: usize,
    /// Current pool size.
    pub current_size: usize,
    /// Current available objects.
    pub available: usize,
}

impl PoolStats {
    /// Calculate the hit rate (objects returned / objects borrowed).
    pub fn hit_rate(&self) -> f64 {
        if self.total_borrowed == 0 {
            0.0
        } else {
            self.total_returned as f64 / self.total_borrowed as f64
        }
    }

    /// Calculate the utilization rate (current_size / total_created).
    pub fn utilization_rate(&self) -> f64 {
        if self.total_created == 0 {
            0.0
        } else {
            self.current_size as f64 / self.total_created as f64
        }
    }
}

/// Configuration trait for dependency injection.
pub trait AsrConfig: Send + Sync {
    /// Get the Triton endpoint URL.
    fn triton_endpoint(&self) -> &str;

    /// Get the vocabulary file path.
    fn vocabulary_path(&self) -> &str;

    /// Get the connection pool size.
    fn pool_size(&self) -> usize;

    /// Get the inference timeout.
    fn inference_timeout(&self) -> std::time::Duration;

    /// Get the maximum audio length.
    fn max_audio_length(&self) -> usize;

    /// Get the optimal batch size.
    fn optimal_batch_size(&self) -> usize;
}

/// Time provider trait for testability.
pub trait TimeProvider: Send + Sync {
    /// Get the current time.
    fn now(&self) -> std::time::Instant;
}

/// System time provider implementation.
#[derive(Debug, Clone)]
pub struct SystemTimeProvider;

impl TimeProvider for SystemTimeProvider {
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// Mock time provider for testing.
#[derive(Debug, Clone)]
pub struct MockTimeProvider {
    current_time: Arc<parking_lot::Mutex<std::time::Instant>>,
}

impl MockTimeProvider {
    /// Create a new mock time provider.
    pub fn new() -> Self {
        Self {
            current_time: Arc::new(parking_lot::Mutex::new(std::time::Instant::now())),
        }
    }

    /// Advance the mock time by a duration.
    pub fn advance(&self, duration: std::time::Duration) {
        let mut time = self.current_time.lock();
        *time += duration;
    }

    /// Set the mock time to a specific instant.
    pub fn set_time(&self, time: std::time::Instant) {
        *self.current_time.lock() = time;
    }
}

impl TimeProvider for MockTimeProvider {
    fn now(&self) -> std::time::Instant {
        *self.current_time.lock()
    }
}

impl Default for MockTimeProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_decoder_state_creation() {
        let state = DecoderState::new();
        assert!(state.is_empty());
        assert_eq!(state.time_step, 0);
        assert_eq!(state.state_version, 0);
    }

    #[test]
    fn test_decoder_state_reset() {
        let mut state = DecoderState::new();
        state.hidden_states.push(1.0);
        state.prediction_history.push(TokenId::new(1));
        state.time_step = 5;

        let initial_version = state.state_version;
        state.reset();

        assert!(state.is_empty());
        assert_eq!(state.time_step, 0);
        assert_eq!(state.state_version, initial_version + 1);
    }

    #[test]
    fn test_pool_stats_calculations() {
        let stats = PoolStats {
            total_created: 10,
            total_borrowed: 8,
            total_returned: 6,
            current_size: 7,
            available: 3,
        };

        assert_eq!(stats.hit_rate(), 0.75);
        assert_eq!(stats.utilization_rate(), 0.7);
    }

    #[test]
    fn test_mock_time_provider() {
        let provider = MockTimeProvider::new();
        let initial_time = provider.now();

        provider.advance(Duration::from_secs(1));
        let advanced_time = provider.now();

        assert!(advanced_time > initial_time);
        assert_eq!(advanced_time - initial_time, Duration::from_secs(1));
    }
}
