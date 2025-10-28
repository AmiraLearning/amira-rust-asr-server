//! ASR-related error types.
//!
//! This module contains error types for audio processing and speech recognition,
//! including audio processing errors, model inference errors, and ASR pipeline errors.

use thiserror::Error;

use super::super::types::model::ModelError;

/// ASR-specific errors for audio processing and speech recognition.
#[derive(Debug, Error)]
pub enum AsrError {
    #[error("Audio processing failed: {0}")]
    AudioProcessing(#[from] AudioError),

    #[error("Model inference failed: {0}")]
    ModelInference(#[from] ModelError),

    #[error("Decoder state invalid: {0}")]
    DecoderState(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Vocabulary error: {0}")]
    Vocabulary(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

/// Audio processing errors.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("Invalid sample rate: expected {expected}, got {actual}")]
    InvalidSampleRate { expected: u32, actual: u32 },

    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Buffer underrun: insufficient audio data")]
    BufferUnderrun,

    #[error("Buffer overflow: audio data too large")]
    BufferOverflow,

    #[error("SIMD processing error: {0}")]
    SimdProcessing(String),

    #[error("Windowing error: {0}")]
    Windowing(String),
}
