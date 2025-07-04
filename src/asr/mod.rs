//! Core ASR (Automatic Speech Recognition) functionality.
//!
//! This module contains the components for speech-to-text conversion,
//! including the ASR pipeline, decoders, and audio utilities.

mod audio;
mod decoder;
mod decoder_optimized;
mod incremental;
mod memory;
mod lockfree_memory;
mod pipeline;
pub mod simd;
pub mod simd_advanced;
pub mod simd_optimized;
pub mod types;
mod weaving;
mod zero_copy;

pub use audio::{
    audio_len, bytes_to_f32_samples, bytes_to_f32_samples_into, calculate_mean_amplitude,
    AudioRingBuffer, OverlappingAudioBuffer,
};
pub use decoder::greedy_decode;
pub use decoder_optimized::{greedy_decode_zero_copy, greedy_decode_zero_copy_async};
pub use incremental::IncrementalAsr;
pub use memory::{global_pools, AsrMemoryPools, AsrMemoryStats, ObjectPool, PooledObject};
pub use lockfree_memory::{
    global_lockfree_pools, get_lockfree_decoder_workspace, LockFreeAsrMemoryPools, 
    LockFreeAsrMemoryStats, LockFreeObjectPool, LockFreePooledObject
};
pub use pipeline::{AsrPipeline, TritonAsrPipeline};
pub use types::{AccumulatedPredictions, DecoderState, SeqSlice, Transcription, Vocabulary};
pub use zero_copy::{argmax_zero_copy, with_decoder_workspace, DecoderWorkspace, TensorView};
pub use simd_advanced::{
    softmax_optimized, batch_normalize_optimized, dot_product_optimized, 
    batch_process_audio_streams
};
pub use simd_optimized::{
    bytes_to_f32_safe_optimized, mean_amplitude_safe_optimized, smooth_audio_optimized,
    AlignedBuffer
};
