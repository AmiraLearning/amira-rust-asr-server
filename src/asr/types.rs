//! Core ASR domain types.
//!
//! This module defines the fundamental data structures used throughout the ASR pipeline,
//! including vocabulary management, decoder state tracking, and transcription results.

use crate::config::model::DECODER_STATE_SIZE;
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
        let content = fs::read_to_string(path).map_err(|e| AppError::Io(e))?;

        let mut id_to_token = HashMap::new();

        for line in content.lines() {
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
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
            states_1: vec![0.0; 2 * 1 * DECODER_STATE_SIZE],
            states_2: vec![0.0; 2 * 1 * DECODER_STATE_SIZE],
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
