//! Core ASR domain types.
//!
//! This module defines the fundamental data structures used throughout the ASR pipeline,
//! including vocabulary management, decoder state tracking, and transcription results.

use crate::constants::model::DECODER_STATE_SIZE;
use crate::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::debug;

// Constants for audio processing and transcript weaving
/// Expected ratio for silence detection
pub const EXPECTED_SILENCE_RATIO: f32 = 2.0;
/// Maximum alignment distance for transcript weaving
pub const MAX_ALIGN_DIST: f32 = 0.6;
/// Alpha parameter for distance scoring
pub const ALPHA: f32 = 0.1;
/// Sample rate for WAV2VEC2 models
pub const W2V_SAMPLE_RATE: usize = 16000;

/// Represents a slice of a sequence with start and end indices.
#[derive(Debug, Clone, Copy)]
pub struct SeqSlice {
    /// Start index (inclusive)
    pub start: usize,

    /// End index (exclusive)
    pub end: usize,
}

impl SeqSlice {
    /// Create a new sequence slice.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Get the length of the slice.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Apply a function to both start and end.
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(usize) -> usize,
    {
        Self {
            start: f(self.start),
            end: f(self.end),
        }
    }

    /// Create a new slice with an offset subtracted.
    pub fn minus(&self, offset: usize) -> Self {
        Self {
            start: self.start.saturating_sub(offset),
            end: self.end.saturating_sub(offset),
        }
    }

    /// Convert to a standard Rust slice range.
    pub fn as_range(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

/// Represents the vocabulary for token decoding.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Mapping from token IDs to string tokens
    id_to_token: HashMap<i32, String>,
}

impl Vocabulary {
    /// Load vocabulary from a file.
    ///
    /// The file format should be: `<token> <id>` on each line.
    /// For example: `▁the 5` or `<blk> 1024`
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).map_err(AppError::Io)?;

        let mut id_to_token = HashMap::new();

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                // Token is everything except the last part (which is the ID)
                let token = parts[0..parts.len() - 1].join(" ");
                // ID is the last part
                if let Ok(id) = parts[parts.len() - 1].parse::<i32>() {
                    id_to_token.insert(id, token);
                }
            }
        }

        debug!("Loaded vocabulary with {} tokens", id_to_token.len());
        Ok(Self { id_to_token })
    }

    /// Decode a sequence of token IDs into text.
    ///
    /// Handles special BPE tokens (▁) by converting them to spaces.
    pub fn decode_tokens(&self, token_ids: &[i32]) -> String {
        let mut result = String::new();

        for &token_id in token_ids {
            if let Some(token) = self.id_to_token.get(&token_id) {
                // Handle BPE tokens with ▁ prefix
                if token.starts_with("▁") {
                    // Add space before the token (except at the beginning)
                    // Use proper UTF-8 character boundary detection
                    if let Some(stripped) = token.strip_prefix("▁") {
                        result.push_str(&format!(" {}", stripped));
                    } else {
                        // Fallback if prefix removal fails
                        result.push_str(token);
                    }
                } else {
                    result.push_str(token);
                }
            }
        }

        // Trim any leading space that might have been added
        result.trim().to_string()
    }

    /// Get the token for a given ID, or None if not found.
    pub fn get_token(&self, id: i32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Get the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// Check if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// Create a vocabulary from a HashMap (for testing).
    pub fn from_map(id_to_token: HashMap<i32, String>) -> Self {
        Self { id_to_token }
    }
}

/// Tracks the RNN-T decoder state between inference calls.
#[derive(Debug, Clone)]
pub struct DecoderState {
    /// First state tensor with shape [2, 1, 640] flattened to Vec<f32>
    pub states_1: Vec<f32>,

    /// Second state tensor with shape [2, 1, 640] flattened to Vec<f32>
    pub states_2: Vec<f32>,
}

impl DecoderState {
    /// Create a new decoder state initialized to zeros.
    pub fn new() -> Self {
        Self {
            states_1: vec![0.0; 2 * DECODER_STATE_SIZE],
            states_2: vec![0.0; 2 * DECODER_STATE_SIZE],
        }
    }
}

impl Default for DecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks accumulated predictions for incremental ASR processing.
#[derive(Debug, Clone, Default)]
pub struct AccumulatedPredictions {
    /// Accumulated token predictions
    pub token_ids: Vec<i32>,

    /// Accumulated transcript text
    pub transcript: String,

    /// Mean amplitude of the audio for silence detection
    pub mean_amplitude: f32,
}

impl AccumulatedPredictions {
    /// Create a new empty accumulated predictions state.
    pub fn new() -> Self {
        Self {
            token_ids: Vec::new(),
            transcript: String::new(),
            mean_amplitude: 0.0,
        }
    }

    /// Clear all accumulated predictions.
    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.transcript.clear();
        self.mean_amplitude = 0.0;
    }
}

/// Represents a complete transcription result with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transcription {
    /// The transcribed text
    pub text: String,

    /// The token IDs that produced the transcription
    pub tokens: Vec<i32>,

    /// Audio length in samples
    pub audio_length_samples: usize,

    /// Features length (time dimension after preprocessing)
    pub features_length: i64,

    /// Encoded length (time dimension after encoder)
    pub encoded_length: i64,
}

/// Status of a streaming ASR session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum StreamStatus {
    /// Stream is active and receiving audio
    Active,

    /// Stream has completed successfully
    Complete,

    /// Stream is temporarily paused
    Paused,

    /// Stream has encountered an error
    Error,
}

/// Response from the ASR service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrResponse {
    /// The transcribed text
    pub transcription: String,

    /// Current status of the stream
    pub status: StreamStatus,

    /// Optional error or information message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,

    /// Optional metadata about the transcription
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Optional client-provided opaque data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opaque: Option<serde_json::Value>,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Validate that audio data is in the correct format for 16-bit PCM.
///
/// 16-bit PCM audio requires that the data length be even (each sample is 2 bytes).
///
/// # Arguments
/// * `data` - The audio data to validate
///
/// # Returns
/// * `Ok(())` if the data is valid
/// * `Err(AppError)` if the data length is not even
///
/// # Errors
/// Returns `AppError::Server(ServerError::RequestValidation)` if the data length is odd.
pub fn validate_pcm_audio_format(data: &[u8]) -> Result<()> {
    use crate::error::ServerError;

    if !data.len().is_multiple_of(2) {
        return Err(AppError::Server(ServerError::RequestValidation(
            "Audio data length must be even for 16-bit PCM".to_string(),
        )));
    }
    Ok(())
}

/// Convert a serde_json::Value to a HashMap for metadata.
///
/// This is a common pattern when building metadata for ASR responses.
/// If the value is not an object, an empty HashMap is returned.
///
/// # Arguments
/// * `value` - The JSON value to convert
///
/// # Returns
/// A HashMap containing the key-value pairs from the JSON object
pub fn json_value_to_metadata(value: serde_json::Value) -> HashMap<String, serde_json::Value> {
    let mut metadata = HashMap::new();
    if let serde_json::Value::Object(map) = value {
        for (k, v) in map {
            metadata.insert(k, v);
        }
    }
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    // SeqSlice tests
    #[test]
    fn test_seq_slice_new() {
        let slice = SeqSlice::new(10, 20);
        assert_eq!(slice.start, 10);
        assert_eq!(slice.end, 20);
    }

    #[test]
    fn test_seq_slice_len() {
        let slice = SeqSlice::new(10, 25);
        assert_eq!(slice.len(), 15);

        let empty_slice = SeqSlice::new(10, 10);
        assert_eq!(empty_slice.len(), 0);
    }

    #[test]
    fn test_seq_slice_is_empty() {
        let slice = SeqSlice::new(10, 20);
        assert!(!slice.is_empty());

        let empty_slice = SeqSlice::new(5, 5);
        assert!(empty_slice.is_empty());
    }

    #[test]
    fn test_seq_slice_map() {
        let slice = SeqSlice::new(10, 20);
        let doubled = slice.map(|x| x * 2);
        assert_eq!(doubled.start, 20);
        assert_eq!(doubled.end, 40);
    }

    #[test]
    fn test_seq_slice_minus() {
        let slice = SeqSlice::new(10, 20);
        let shifted = slice.minus(5);
        assert_eq!(shifted.start, 5);
        assert_eq!(shifted.end, 15);

        // Test saturating subtraction
        let saturated = slice.minus(15);
        assert_eq!(saturated.start, 0); // saturating_sub(10, 15) = 0
        assert_eq!(saturated.end, 5);
    }

    #[test]
    fn test_seq_slice_as_range() {
        let slice = SeqSlice::new(10, 20);
        let range = slice.as_range();
        assert_eq!(range, 10..20);
    }

    // Vocabulary tests
    #[test]
    fn test_vocabulary_from_map() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(1, "world".to_string());
        map.insert(2, "▁the".to_string());

        let vocab = Vocabulary::from_map(map);
        assert_eq!(vocab.len(), 3);
        assert!(!vocab.is_empty());
    }

    #[test]
    fn test_vocabulary_get_token() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(1, "world".to_string());

        let vocab = Vocabulary::from_map(map);
        assert_eq!(vocab.get_token(0), Some("hello"));
        assert_eq!(vocab.get_token(1), Some("world"));
        assert_eq!(vocab.get_token(2), None);
    }

    #[test]
    fn test_vocabulary_decode_tokens_basic() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());
        map.insert(1, "world".to_string());

        let vocab = Vocabulary::from_map(map);
        let tokens = vec![0, 1];
        let decoded = vocab.decode_tokens(&tokens);
        assert_eq!(decoded, "helloworld");
    }

    #[test]
    fn test_vocabulary_decode_tokens_with_bpe() {
        let mut map = HashMap::new();
        map.insert(0, "▁hello".to_string());
        map.insert(1, "▁world".to_string());
        map.insert(2, "!".to_string());

        let vocab = Vocabulary::from_map(map);
        let tokens = vec![0, 1, 2];
        let decoded = vocab.decode_tokens(&tokens);
        assert_eq!(decoded, "hello world!");
    }

    #[test]
    fn test_vocabulary_decode_tokens_mixed() {
        let mut map = HashMap::new();
        map.insert(0, "▁the".to_string());
        map.insert(1, "cat".to_string()); // No ▁ prefix, so attached to previous
        map.insert(2, "▁sat".to_string());

        let vocab = Vocabulary::from_map(map);
        let tokens = vec![0, 1, 2];
        let decoded = vocab.decode_tokens(&tokens);
        assert_eq!(decoded, "thecat sat"); // "cat" has no space prefix
    }

    #[test]
    fn test_vocabulary_decode_tokens_empty() {
        let vocab = Vocabulary::from_map(HashMap::new());
        let tokens = vec![];
        let decoded = vocab.decode_tokens(&tokens);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_vocabulary_decode_tokens_unknown_ids() {
        let mut map = HashMap::new();
        map.insert(0, "hello".to_string());

        let vocab = Vocabulary::from_map(map);
        let tokens = vec![0, 999, 0]; // 999 is unknown
        let decoded = vocab.decode_tokens(&tokens);
        assert_eq!(decoded, "hellohello"); // Unknown tokens are skipped
    }

    // DecoderState tests
    #[test]
    fn test_decoder_state_new() {
        let state = DecoderState::new();
        assert_eq!(state.states_1.len(), 2 * DECODER_STATE_SIZE);
        assert_eq!(state.states_2.len(), 2 * DECODER_STATE_SIZE);

        // Verify all zeros
        assert!(state.states_1.iter().all(|&x| x == 0.0));
        assert!(state.states_2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_decoder_state_default() {
        let state = DecoderState::default();
        assert_eq!(state.states_1.len(), 2 * DECODER_STATE_SIZE);
        assert_eq!(state.states_2.len(), 2 * DECODER_STATE_SIZE);
    }

    // AccumulatedPredictions tests
    #[test]
    fn test_accumulated_predictions_new() {
        let preds = AccumulatedPredictions::new();
        assert!(preds.token_ids.is_empty());
        assert_eq!(preds.transcript, "");
        assert_eq!(preds.mean_amplitude, 0.0);
    }

    #[test]
    fn test_accumulated_predictions_clear() {
        let mut preds = AccumulatedPredictions {
            token_ids: vec![1, 2, 3],
            transcript: "test".to_string(),
            mean_amplitude: 0.5,
        };

        preds.clear();

        assert!(preds.token_ids.is_empty());
        assert_eq!(preds.transcript, "");
        assert_eq!(preds.mean_amplitude, 0.0);
    }

    #[test]
    fn test_accumulated_predictions_default() {
        let preds = AccumulatedPredictions::default();
        assert!(preds.token_ids.is_empty());
        assert_eq!(preds.transcript, "");
        assert_eq!(preds.mean_amplitude, 0.0);
    }

    // Transcription serialization tests
    #[test]
    fn test_transcription_serialization() {
        let transcription = Transcription {
            text: "hello world".to_string(),
            tokens: vec![1, 2, 3],
            audio_length_samples: 16000,
            features_length: 100,
            encoded_length: 50,
        };

        let json = serde_json::to_string(&transcription).unwrap();
        assert!(json.contains("audioLengthSamples")); // camelCase
        assert!(json.contains("featuresLength"));
        assert!(json.contains("encodedLength"));
    }

    #[test]
    fn test_transcription_deserialization() {
        let json = r#"{
            "text": "test",
            "tokens": [1, 2],
            "audioLengthSamples": 32000,
            "featuresLength": 200,
            "encodedLength": 100
        }"#;

        let transcription: Transcription = serde_json::from_str(json).unwrap();
        assert_eq!(transcription.text, "test");
        assert_eq!(transcription.tokens, vec![1, 2]);
        assert_eq!(transcription.audio_length_samples, 32000);
    }

    // StreamStatus serialization tests
    #[test]
    fn test_stream_status_serialization() {
        let active = StreamStatus::Active;
        let json = serde_json::to_string(&active).unwrap();
        assert_eq!(json, "\"ACTIVE\""); // UPPERCASE

        let complete = StreamStatus::Complete;
        let json = serde_json::to_string(&complete).unwrap();
        assert_eq!(json, "\"COMPLETE\"");
    }

    #[test]
    fn test_stream_status_deserialization() {
        let active: StreamStatus = serde_json::from_str("\"ACTIVE\"").unwrap();
        assert!(matches!(active, StreamStatus::Active));

        let error: StreamStatus = serde_json::from_str("\"ERROR\"").unwrap();
        assert!(matches!(error, StreamStatus::Error));
    }

    // AsrResponse serialization tests
    #[test]
    fn test_asr_response_serialization_minimal() {
        let response = AsrResponse {
            transcription: "hello".to_string(),
            status: StreamStatus::Complete,
            message: None,
            metadata: None,
            opaque: None,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains("message")); // skip_serializing_if None
        assert!(!json.contains("metadata"));
        assert!(!json.contains("opaque"));
    }

    #[test]
    fn test_asr_response_serialization_full() {
        let mut metadata = HashMap::new();
        metadata.insert("duration".to_string(), serde_json::json!(1.5));

        let response = AsrResponse {
            transcription: "hello world".to_string(),
            status: StreamStatus::Complete,
            message: Some("Success".to_string()),
            metadata: Some(metadata),
            opaque: Some(serde_json::json!({"session_id": "123"})),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("message"));
        assert!(json.contains("metadata"));
        assert!(json.contains("opaque"));
        assert!(json.contains("session_id"));
    }

    // Utility function tests
    #[test]
    fn test_validate_pcm_audio_format_valid() {
        let even_data = vec![0u8; 100]; // 100 bytes (even)
        assert!(validate_pcm_audio_format(&even_data).is_ok());
    }

    #[test]
    fn test_validate_pcm_audio_format_invalid() {
        let odd_data = vec![0u8; 101]; // 101 bytes (odd)
        let result = validate_pcm_audio_format(&odd_data);
        assert!(result.is_err());

        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("must be even"));
    }

    #[test]
    fn test_json_value_to_metadata_object() {
        let value = serde_json::json!({
            "duration": 1.5,
            "samples": 16000
        });

        let metadata = json_value_to_metadata(value);
        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata.get("duration").unwrap(), &serde_json::json!(1.5));
        assert_eq!(metadata.get("samples").unwrap(), &serde_json::json!(16000));
    }

    #[test]
    fn test_json_value_to_metadata_non_object() {
        let value = serde_json::json!("not an object");
        let metadata = json_value_to_metadata(value);
        assert!(metadata.is_empty());

        let array_value = serde_json::json!([1, 2, 3]);
        let metadata = json_value_to_metadata(array_value);
        assert!(metadata.is_empty());
    }
}
