//! CUDA-based ASR pipeline implementation
//!
//! This module provides an ASR pipeline that uses CUDA shared memory and direct
//! Triton C API calls instead of gRPC, eliminating network overhead and enabling
//! zero-copy inference.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

use crate::asr::pipeline::AsrPipeline;
use crate::asr::types::{DecoderState, Transcription, Vocabulary};
use crate::cuda::{AsyncCudaStreamPool, CudaSharedMemoryPool, CudaSharedMemoryRegion, ModelConfig};
use crate::error::{AppError, Result};

// Type alias for region maps
type RegionMap = HashMap<String, CudaSharedMemoryRegion>;

// Region name constants for ensemble model
const REGION_AUDIO_FRAMES: &str = "AUDIO_FRAMES";
const REGION_ENCODER_STATE: &str = "ENCODER_STATE";
const REGION_DECODER_STATE: &str = "DECODER_STATE";
const REGION_LOGITS: &str = "LOGITS";
const REGION_UPDATED_ENCODER_STATE: &str = "UPDATED_ENCODER_STATE";
const REGION_UPDATED_DECODER_STATE: &str = "UPDATED_DECODER_STATE";

// Stream pool index for ensemble operations
const ENSEMBLE_STREAM_ID: usize = 0;

// Audio normalization constant (16-bit PCM max value)
const AUDIO_NORMALIZATION_FACTOR: f32 = 32768.0;

// Default state dimensions
const DEFAULT_ENCODER_STATE_DIM1: usize = 512;
const DEFAULT_ENCODER_STATE_DIM2: usize = 2048;
const DEFAULT_DECODER_STATE_DIM1: usize = 512;
const DEFAULT_DECODER_STATE_DIM2: usize = 1024;

// Default vocabulary size fallback
const DEFAULT_VOCAB_SIZE: usize = 4096;

// Size of f32 in bytes
const F32_SIZE: usize = std::mem::size_of::<f32>();

// Audio processing constants
const BYTES_PER_I16_SAMPLE: usize = 2;

// Stream and device configuration
const ENSEMBLE_STREAM_COUNT: usize = 1;
const DEFAULT_DEVICE_ID: i32 = 0;

// Fallback and placeholder values
const DEFAULT_TOKEN_FALLBACK: usize = 0;
const NOT_COMPUTED: i64 = 0;

// String labels for region registration
const LABEL_ENSEMBLE_INPUT: &str = "ensemble input";
const LABEL_ENSEMBLE_OUTPUT: &str = "ensemble output";

/// Helper function to create CUDA errors with consistent formatting
fn cuda_error(msg: &str, e: impl std::fmt::Display) -> AppError {
    AppError::Cuda(crate::error::CudaError::Device(format!("{}: {}", msg, e)))
}

/// CUDA-based ASR pipeline using direct Triton C API with zero-copy ensemble inference
pub struct CudaAsrPipeline {
    /// Vocabulary for token decoding
    vocabulary: Arc<Vocabulary>,

    /// Memory pool for ensemble model (true 2-copy zero-copy inference)
    ensemble_pool: CudaSharedMemoryPool,

    /// Async CUDA stream pool for overlapping operations
    stream_pool: AsyncCudaStreamPool,
}

impl CudaAsrPipeline {
    /// Create a new CUDA-based ASR pipeline
    pub fn new(device_id: i32, vocabulary: Arc<Vocabulary>) -> Result<Self> {
        info!(
            "Initializing CUDA ASR pipeline with zero-copy ensemble on device {}",
            device_id
        );

        // Create ensemble model configuration
        let ensemble_config = ModelConfig::rnnt_ensemble();

        // Create ensemble memory pool for zero-copy inference
        let ensemble_pool = CudaSharedMemoryPool::new_for_model(ensemble_config, device_id)
            .map_err(|e| cuda_error("Failed to create ensemble pool", e))?;

        // Register ensemble regions with Triton server
        Self::register_ensemble_with_triton(&ensemble_pool)?;

        // Create async CUDA stream pool (only need 1 stream for ensemble)
        let stream_pool = AsyncCudaStreamPool::new(device_id, ENSEMBLE_STREAM_COUNT)
            .map_err(|e| cuda_error("Failed to create stream pool", e))?;

        info!("CUDA ASR pipeline with zero-copy ensemble initialized successfully");

        Ok(Self {
            vocabulary,
            ensemble_pool,
            stream_pool,
        })
    }

    /// Helper to register a set of regions with Triton server
    fn register_regions(regions: &RegionMap, kind: &str) -> Result<()> {
        for (name, region) in regions {
            region
                .register_with_triton_server()
                .map_err(|e| cuda_error(&format!("Failed to register {} {}", kind, name), e))?;
        }
        Ok(())
    }

    /// Register ensemble memory pool with Triton server
    fn register_ensemble_with_triton(ensemble_pool: &CudaSharedMemoryPool) -> Result<()> {
        debug!("Registering ensemble CUDA memory regions with Triton server");

        // Register ensemble input and output regions
        Self::register_regions(&ensemble_pool.input_regions, LABEL_ENSEMBLE_INPUT)?;
        Self::register_regions(&ensemble_pool.output_regions, LABEL_ENSEMBLE_OUTPUT)?;

        info!("Successfully registered ensemble CUDA memory regions with Triton server");
        Ok(())
    }

    /// Get a region (input or output) with consistent error handling
    fn get_region(&self, name: &str, is_input: bool) -> Result<&CudaSharedMemoryRegion> {
        let region = if is_input {
            self.ensemble_pool.get_input_region(name)
        } else {
            self.ensemble_pool.get_output_region(name)
        };

        region.ok_or_else(|| {
            let kind = if is_input { "input" } else { "output" };
            cuda_error(&format!("Missing ensemble {} region", kind), name)
        })
    }

    /// Calculate buffer size for output in number of f32 elements
    fn calculate_output_elements(&self, name: &str) -> Result<usize> {
        let byte_size = self
            .ensemble_pool
            .config
            .calculate_output_buffer_size(name)
            .ok_or_else(|| cuda_error("Failed to calculate buffer size", name))?;
        Ok(byte_size / F32_SIZE)
    }

    /// Run inference using the Triton ensemble model
    ///
    /// The ensemble model chains preprocessor → encoder → decoder_joint entirely on GPU.
    /// All complexity is handled by the Triton graph configuration.
    ///
    /// **Memory copies: Only 2!**
    /// 1. CPU → GPU: Audio + states upload
    /// 2. GPU → CPU: Logits + updated states download
    ///
    /// All intermediate tensors (MEL_FEATURES, ENCODER_OUTPUT) stay on GPU!
    async fn infer(
        &self,
        audio_samples: &[f32],
        encoder_state: &[f32],
        decoder_state: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        debug!(
            "Running ensemble inference with {} audio samples",
            audio_samples.len()
        );

        // Get dedicated stream for ensemble operations
        let stream = self
            .stream_pool
            .get_stream(ENSEMBLE_STREAM_ID)
            .ok_or_else(|| cuda_error("Failed to get ensemble stream", "stream not available"))?;

        // Get input regions
        let audio_input_region = self.get_region(REGION_AUDIO_FRAMES, true)?;
        let encoder_state_input_region = self.get_region(REGION_ENCODER_STATE, true)?;
        let decoder_state_input_region = self.get_region(REGION_DECODER_STATE, true)?;

        // Get output regions
        let logits_output_region = self.get_region(REGION_LOGITS, false)?;
        let updated_encoder_state_output_region =
            self.get_region(REGION_UPDATED_ENCODER_STATE, false)?;
        let updated_decoder_state_output_region =
            self.get_region(REGION_UPDATED_DECODER_STATE, false)?;

        // COPY #1: Upload all inputs to GPU (single H2D transfer)
        audio_input_region
            .enqueue_write_f32_data(audio_samples, &stream)
            .map_err(|e| cuda_error("Failed to enqueue audio upload", e))?;

        encoder_state_input_region
            .enqueue_write_f32_data(encoder_state, &stream)
            .map_err(|e| cuda_error("Failed to enqueue encoder state upload", e))?;

        decoder_state_input_region
            .enqueue_write_f32_data(decoder_state, &stream)
            .map_err(|e| cuda_error("Failed to enqueue decoder state upload", e))?;

        // Run ensemble inference (all on GPU - zero copy between stages!)
        // This single call executes: preprocessor → encoder → decoder_joint
        audio_input_region
            .enqueue_inference_with_output_regions(
                logits_output_region,
                &self.ensemble_pool.config,
                REGION_AUDIO_FRAMES,
                REGION_LOGITS,
                &stream,
            )
            .map_err(|e| cuda_error("Failed to enqueue ensemble inference", e))?;

        // Calculate output buffer sizes (in f32 elements, not bytes)
        let logits_elements = self.calculate_output_elements(REGION_LOGITS)?;
        let encoder_state_elements =
            self.calculate_output_elements(REGION_UPDATED_ENCODER_STATE)?;
        let decoder_state_elements =
            self.calculate_output_elements(REGION_UPDATED_DECODER_STATE)?;

        // COPY #2: Download all outputs from GPU (single D2H transfer)
        let mut logits = vec![0.0; logits_elements];
        let mut updated_encoder_state = vec![0.0; encoder_state_elements];
        let mut updated_decoder_state = vec![0.0; decoder_state_elements];

        logits_output_region
            .enqueue_read_f32_data(&mut logits, &stream)
            .map_err(|e| cuda_error("Failed to enqueue logits download", e))?;

        updated_encoder_state_output_region
            .enqueue_read_f32_data(&mut updated_encoder_state, &stream)
            .map_err(|e| cuda_error("Failed to enqueue encoder state download", e))?;

        updated_decoder_state_output_region
            .enqueue_read_f32_data(&mut updated_decoder_state, &stream)
            .map_err(|e| cuda_error("Failed to enqueue decoder state download", e))?;

        // Wait for all GPU operations to complete
        stream
            .wait()
            .await
            .map_err(|e| cuda_error("Failed to wait for ensemble completion", e))?;

        debug!("Ensemble inference completed: {} logits", logits.len());

        Ok((logits, updated_encoder_state, updated_decoder_state))
    }

    /// Initialize state vector with zeros
    fn initialize_state(
        &self,
        region_name: &str,
        default_dim1: usize,
        default_dim2: usize,
    ) -> Vec<f32> {
        let state_size = self
            .ensemble_pool
            .config
            .calculate_buffer_size(region_name)
            .unwrap_or(default_dim1 * default_dim2 * F32_SIZE);
        vec![0.0; state_size / F32_SIZE]
    }

    /// Initialize encoder state with zeros
    fn initialize_encoder_state(&self) -> Vec<f32> {
        self.initialize_state(
            REGION_ENCODER_STATE,
            DEFAULT_ENCODER_STATE_DIM1,
            DEFAULT_ENCODER_STATE_DIM2,
        )
    }

    /// Initialize decoder state with zeros
    fn initialize_decoder_state(&self) -> Vec<f32> {
        self.initialize_state(
            REGION_DECODER_STATE,
            DEFAULT_DECODER_STATE_DIM1,
            DEFAULT_DECODER_STATE_DIM2,
        )
    }

    /// Convert audio bytes to normalized f32 samples
    fn audio_bytes_to_samples(&self, audio_bytes: &[u8]) -> Result<Vec<f32>> {
        if !audio_bytes.len().is_multiple_of(BYTES_PER_I16_SAMPLE) {
            return Err(AppError::InvalidInput(
                "Audio bytes must be even length (16-bit samples)".to_string(),
            ));
        }

        let mut samples = Vec::with_capacity(audio_bytes.len() / BYTES_PER_I16_SAMPLE);
        for chunk in audio_bytes.chunks_exact(BYTES_PER_I16_SAMPLE) {
            // chunks_exact guarantees chunk is exactly BYTES_PER_I16_SAMPLE bytes
            // Use defensive error handling instead of expect() to prevent panic-based DoS
            let bytes: [u8; 2] = chunk.try_into().map_err(|_| {
                AppError::Internal(format!(
                    "Invalid audio chunk size: expected {} bytes, got {}",
                    BYTES_PER_I16_SAMPLE,
                    chunk.len()
                ))
            })?;
            let sample = i16::from_le_bytes(bytes);
            // Normalize to [-1.0, 1.0]
            samples.push(sample as f32 / AUDIO_NORMALIZATION_FACTOR);
        }

        Ok(samples)
    }

    /// Get vocabulary size from model config
    fn get_vocab_size(&self) -> usize {
        // Get vocab size from LOGITS output dimensions
        // LOGITS shape is [time, vocab_size], so vocab_size is the last dimension
        self.ensemble_pool
            .config
            .outputs
            .iter()
            .find(|(name, _spec)| name.as_str() == REGION_LOGITS)
            .and_then(|(_name, spec)| spec.dims.last())
            .copied()
            .map(|dim| dim as usize)
            .unwrap_or(DEFAULT_VOCAB_SIZE)
    }

    /// Convert logits to tokens using greedy decoding
    fn logits_to_tokens(&self, logits: &[f32]) -> Result<Vec<i32>> {
        let vocab_size = self.get_vocab_size();

        // Guard against division by zero
        if vocab_size == 0 {
            return Err(AppError::Internal(
                "Vocabulary size is 0, cannot decode tokens".to_string(),
            ));
        }

        // STRICT: Reject logits that aren't evenly divisible by vocab_size
        // This prevents silent data loss from truncation
        let remainder = logits.len() % vocab_size;
        if remainder != 0 {
            return Err(AppError::Internal(format!(
                "Logits length {} not evenly divisible by vocab_size {} (remainder: {}). This indicates a model output mismatch.",
                logits.len(),
                vocab_size,
                remainder
            )));
        }

        let time_steps = logits.len() / vocab_size;

        let mut tokens = Vec::with_capacity(time_steps);
        for t in 0..time_steps {
            let start_idx = t * vocab_size;
            let end_idx = start_idx + vocab_size;
            let time_logits = &logits[start_idx..end_idx];

            // Find the token with highest probability
            let max_idx = time_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(DEFAULT_TOKEN_FALLBACK);

            tokens.push(max_idx as i32);
        }

        Ok(tokens)
    }

    /// Convert tokens to text using vocabulary
    fn tokens_to_text(&self, tokens: &[i32]) -> String {
        tokens
            .iter()
            .filter_map(|&token| self.vocabulary.get_token(token))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Helper to build a Transcription from logits and audio samples
    fn build_transcription(&self, logits: Vec<f32>, audio_samples: &[f32]) -> Result<Transcription> {
        let tokens = self.logits_to_tokens(&logits)?;
        let text = self.tokens_to_text(&tokens);

        Ok(Transcription {
            text,
            tokens,
            audio_length_samples: audio_samples.len(),
            features_length: NOT_COMPUTED, // Not computed in ensemble mode
            encoded_length: NOT_COMPUTED,  // Not computed in ensemble mode
        })
    }
}

#[async_trait]
impl AsrPipeline for CudaAsrPipeline {
    async fn process_stream_chunk(
        &self,
        audio_bytes: &[u8],
        state: &mut DecoderState,
    ) -> Result<Transcription> {
        debug!("Processing stream chunk of {} bytes", audio_bytes.len());

        // Convert audio bytes to samples
        let audio_samples = self.audio_bytes_to_samples(audio_bytes)?;

        self.process_stream_samples(&audio_samples, state).await
    }

    async fn process_batch(&self, audio_bytes: &[u8]) -> Result<Transcription> {
        debug!("Processing batch of {} bytes", audio_bytes.len());

        // Convert audio bytes to samples
        let audio_samples = self.audio_bytes_to_samples(audio_bytes)?;

        self.process_batch_samples(&audio_samples).await
    }

    async fn process_stream_samples(
        &self,
        audio_samples: &[f32],
        state: &mut DecoderState,
    ) -> Result<Transcription> {
        debug!("Processing stream samples: {} samples", audio_samples.len());

        // Run zero-copy ensemble inference (2 memory copies total!)
        // No cloning! Pass existing state as slices or initialize new ones
        let (logits, updated_encoder_state, updated_decoder_state) = if state.states_1.is_empty() {
            // First call - initialize states
            let encoder_state = self.initialize_encoder_state();
            let decoder_state = self.initialize_decoder_state();
            self.infer(audio_samples, &encoder_state, &decoder_state)
                .await?
        } else {
            // Subsequent calls - use existing states (no clone, just slices!)
            self.infer(audio_samples, &state.states_1, &state.states_2)
                .await?
        };

        // Update state for next chunk
        state.states_1 = updated_encoder_state;
        state.states_2 = updated_decoder_state;

        // Build transcription from logits
        let transcription = self.build_transcription(logits, audio_samples)?;
        debug!("Stream processing completed: {}", transcription.text);

        Ok(transcription)
    }

    async fn process_batch_samples(&self, audio_samples: &[f32]) -> Result<Transcription> {
        debug!("Processing batch samples: {} samples", audio_samples.len());

        // Initialize fresh states for batch processing
        let encoder_state = self.initialize_encoder_state();
        let decoder_state = self.initialize_decoder_state();

        // Run zero-copy ensemble inference (2 memory copies total!)
        let (logits, ..) = self
            .infer(audio_samples, &encoder_state, &decoder_state)
            .await?;

        // Build transcription from logits
        let transcription = self.build_transcription(logits, audio_samples)?;
        debug!("Batch processing completed: {}", transcription.text);

        Ok(transcription)
    }
}

/// Builder for creating CUDA ASR pipelines
pub struct CudaAsrPipelineBuilder {
    device_id: Option<i32>,
    vocabulary: Option<Arc<Vocabulary>>,
}

impl CudaAsrPipelineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            device_id: None,
            vocabulary: None,
        }
    }

    /// Set CUDA device ID
    pub fn device_id(mut self, device_id: i32) -> Self {
        self.device_id = Some(device_id);
        self
    }

    /// Set vocabulary
    pub fn vocabulary(mut self, vocabulary: Arc<Vocabulary>) -> Self {
        self.vocabulary = Some(vocabulary);
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Result<CudaAsrPipeline> {
        let device_id = self.device_id.unwrap_or(DEFAULT_DEVICE_ID);
        let vocabulary = self
            .vocabulary
            .ok_or_else(|| AppError::ConfigError("Vocabulary is required".to_string()))?;

        CudaAsrPipeline::new(device_id, vocabulary)
    }
}

impl Default for CudaAsrPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
