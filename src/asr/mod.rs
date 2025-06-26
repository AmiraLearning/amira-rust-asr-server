//! Core ASR (Automatic Speech Recognition) functionality.
//!
//! This module contains the components for speech-to-text conversion,
//! including the ASR pipeline, decoders, and audio utilities.

mod audio;
mod decoder;
mod incremental;
mod pipeline;
pub mod types;
mod weaving;

pub use audio::{
    audio_len, bytes_to_f32_samples, calculate_mean_amplitude, AudioRingBuffer,
    OverlappingAudioBuffer,
};
pub use decoder::greedy_decode;
pub use incremental::IncrementalAsr;
pub use pipeline::{AsrPipeline, TritonAsrPipeline};
pub use types::{AccumulatedPredictions, DecoderState, SeqSlice, Transcription, Vocabulary};
