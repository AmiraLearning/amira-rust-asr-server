//! CUDA-based ASR pipeline implementation
//!
//! This module provides an ASR pipeline that uses CUDA shared memory and direct
//! Triton C API calls instead of gRPC, eliminating network overhead and enabling
//! zero-copy inference.

use async_trait::async_trait;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use crate::asr::decoder_optimized::greedy_decode;
use crate::asr::pipeline::AsrPipeline;
use crate::asr::types::{DecoderState, Transcription, Vocabulary};
use crate::cuda::{
    CudaSharedMemoryError, CudaSharedMemoryPool, CudaSharedMemoryRegion, ModelConfig,
};
use crate::error::{AppError, Result};

/// CUDA-based ASR pipeline using direct Triton C API
pub struct CudaAsrPipeline {
    /// Device ID for CUDA operations
    device_id: i32,
    
    /// Vocabulary for token decoding
    vocabulary: Arc<Vocabulary>,
    
    /// Memory pools for each model
    preprocessor_pool: CudaSharedMemoryPool,
    encoder_pool: CudaSharedMemoryPool,
    decoder_joint_pool: CudaSharedMemoryPool,
    
    /// Sample rate for audio processing
    sample_rate: f32,
    
    /// Window size for audio chunks
    window_size: usize,
}

impl CudaAsrPipeline {
    /// Create a new CUDA-based ASR pipeline
    pub fn new(
        device_id: i32,
        vocabulary: Arc<Vocabulary>,
        sample_rate: f32,
        window_size: usize,
    ) -> Result<Self> {
        info!("Initializing CUDA ASR pipeline on device {}", device_id);
        
        // Create model configurations
        let preprocessor_config = ModelConfig::preprocessor();
        let encoder_config = ModelConfig::encoder();
        let decoder_joint_config = ModelConfig::decoder_joint();
        
        // Create memory pools for each model
        let preprocessor_pool = CudaSharedMemoryPool::new_for_model(preprocessor_config, device_id)
            .map_err(|e| AppError::CudaError(format!("Failed to create preprocessor pool: {}", e)))?;
        
        let encoder_pool = CudaSharedMemoryPool::new_for_model(encoder_config, device_id)
            .map_err(|e| AppError::CudaError(format!("Failed to create encoder pool: {}", e)))?;
        
        let decoder_joint_pool = CudaSharedMemoryPool::new_for_model(decoder_joint_config, device_id)
            .map_err(|e| AppError::CudaError(format!("Failed to create decoder joint pool: {}", e)))?;
        
        // Register all regions with Triton server
        Self::register_pools_with_triton(&preprocessor_pool, &encoder_pool, &decoder_joint_pool)?;
        
        info!("CUDA ASR pipeline initialized successfully");
        
        Ok(Self {
            device_id,
            vocabulary,
            preprocessor_pool,
            encoder_pool,
            decoder_joint_pool,
            sample_rate,
            window_size,
        })
    }
    
    /// Register all memory pools with Triton server
    fn register_pools_with_triton(
        preprocessor_pool: &CudaSharedMemoryPool,
        encoder_pool: &CudaSharedMemoryPool,
        decoder_joint_pool: &CudaSharedMemoryPool,
    ) -> Result<()> {
        debug!("Registering CUDA memory regions with Triton server");
        
        // Register preprocessor regions
        for (name, region) in &preprocessor_pool.input_regions {
            region.register_with_triton_server()
                .map_err(|e| AppError::CudaError(format!("Failed to register preprocessor input {}: {}", name, e)))?;
        }
        for (name, region) in &preprocessor_pool.output_regions {
            region.register_with_triton_server()
                .map_err(|e| AppError::CudaError(format!("Failed to register preprocessor output {}: {}", name, e)))?;
        }
        
        // Register encoder regions
        for (name, region) in &encoder_pool.input_regions {
            region.register_with_triton_server()
                .map_err(|e| AppError::CudaError(format!("Failed to register encoder input {}: {}", name, e)))?;
        }
        for (name, region) in &encoder_pool.output_regions {
            region.register_with_triton_server()
                .map_err(|e| AppError::CudaError(format!("Failed to register encoder output {}: {}", name, e)))?;
        }
        
        // Register decoder joint regions
        for (name, region) in &decoder_joint_pool.input_regions {
            region.register_with_triton_server()
                .map_err(|e| AppError::CudaError(format!("Failed to register decoder joint input {}: {}", name, e)))?;
        }
        for (name, region) in &decoder_joint_pool.output_regions {
            region.register_with_triton_server()
                .map_err(|e| AppError::CudaError(format!("Failed to register decoder joint output {}: {}", name, e)))?;
        }
        
        info!("Successfully registered all CUDA memory regions with Triton server");
        Ok(())
    }
    
    /// Convert audio bytes to normalized f32 samples
    fn audio_bytes_to_samples(&self, audio_bytes: &[u8]) -> Result<Vec<f32>> {
        if audio_bytes.len() % 2 != 0 {
            return Err(AppError::InvalidInput("Audio bytes must be even length (16-bit samples)".to_string()));
        }
        
        let mut samples = Vec::with_capacity(audio_bytes.len() / 2);
        for chunk in audio_bytes.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            // Normalize to [-1.0, 1.0]
            samples.push(sample as f32 / 32768.0);
        }
        
        Ok(samples)
    }
    
    /// Run preprocessing step
    async fn run_preprocessor(&self, audio_samples: &[f32]) -> Result<Vec<f32>> {
        debug!("Running preprocessor with {} samples", audio_samples.len());
        
        // Get input and output regions
        let input_region = self.preprocessor_pool.get_input_region("AUDIO_FRAMES")
            .ok_or_else(|| AppError::CudaError("Missing preprocessor input region".to_string()))?;
        
        let output_region = self.preprocessor_pool.get_output_region("MEL_FEATURES")
            .ok_or_else(|| AppError::CudaError("Missing preprocessor output region".to_string()))?;
        
        // Write input data to CUDA memory
        input_region.write_f32_data(audio_samples)
            .map_err(|e| AppError::CudaError(format!("Failed to write preprocessor input: {}", e)))?;
        
        // Run inference
        input_region.run_inference_with_output_regions(
            output_region,
            &self.preprocessor_pool.config,
            "AUDIO_FRAMES",
            "MEL_FEATURES",
        ).map_err(|e| AppError::CudaError(format!("Preprocessor inference failed: {}", e)))?;
        
        // Read output data
        let output_size = self.preprocessor_pool.config.calculate_output_buffer_size("MEL_FEATURES")
            .ok_or_else(|| AppError::CudaError("Failed to calculate output buffer size".to_string()))?;
        
        let output_elements = output_size / 4; // f32 is 4 bytes
        let mel_features = output_region.read_f32_data(output_elements)
            .map_err(|e| AppError::CudaError(format!("Failed to read preprocessor output: {}", e)))?;
        
        debug!("Preprocessor completed, output size: {}", mel_features.len());
        Ok(mel_features)
    }
    
    /// Run encoder step
    async fn run_encoder(&self, mel_features: &[f32], encoder_state: &mut Vec<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
        debug!("Running encoder with {} mel features", mel_features.len());
        
        // Get input and output regions
        let mel_input_region = self.encoder_pool.get_input_region("MEL_FEATURES")
            .ok_or_else(|| AppError::CudaError("Missing encoder MEL_FEATURES input region".to_string()))?;
        
        let state_input_region = self.encoder_pool.get_input_region("ENCODER_STATE")
            .ok_or_else(|| AppError::CudaError("Missing encoder ENCODER_STATE input region".to_string()))?;
        
        let output_region = self.encoder_pool.get_output_region("ENCODER_OUTPUT")
            .ok_or_else(|| AppError::CudaError("Missing encoder ENCODER_OUTPUT output region".to_string()))?;
        
        let state_output_region = self.encoder_pool.get_output_region("UPDATED_ENCODER_STATE")
            .ok_or_else(|| AppError::CudaError("Missing encoder UPDATED_ENCODER_STATE output region".to_string()))?;
        
        // Write input data to CUDA memory
        mel_input_region.write_f32_data(mel_features)
            .map_err(|e| AppError::CudaError(format!("Failed to write encoder mel features: {}", e)))?;
        
        state_input_region.write_f32_data(encoder_state)
            .map_err(|e| AppError::CudaError(format!("Failed to write encoder state: {}", e)))?;
        
        // Run inference with separate input/output regions
        mel_input_region.run_inference_with_output_regions(
            output_region,
            &self.encoder_pool.config,
            "MEL_FEATURES",
            "ENCODER_OUTPUT",
        ).map_err(|e| AppError::CudaError(format!("Encoder inference failed: {}", e)))?;
        
        // Read output data
        let output_size = self.encoder_pool.config.calculate_output_buffer_size("ENCODER_OUTPUT")
            .ok_or_else(|| AppError::CudaError("Failed to calculate encoder output buffer size".to_string()))?;
        
        let state_size = self.encoder_pool.config.calculate_output_buffer_size("UPDATED_ENCODER_STATE")
            .ok_or_else(|| AppError::CudaError("Failed to calculate encoder state buffer size".to_string()))?;
        
        let encoder_output = output_region.read_f32_data(output_size / 4)
            .map_err(|e| AppError::CudaError(format!("Failed to read encoder output: {}", e)))?;
        
        let updated_encoder_state = state_output_region.read_f32_data(state_size / 4)
            .map_err(|e| AppError::CudaError(format!("Failed to read encoder state: {}", e)))?;
        
        // Update the encoder state
        *encoder_state = updated_encoder_state.clone();
        
        debug!("Encoder completed, output size: {}", encoder_output.len());
        Ok((encoder_output, updated_encoder_state))
    }
    
    /// Run decoder/joint step
    async fn run_decoder_joint(&self, encoder_output: &[f32], decoder_state: &mut Vec<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
        debug!("Running decoder/joint with {} encoder outputs", encoder_output.len());
        
        // Get input and output regions
        let encoder_input_region = self.decoder_joint_pool.get_input_region("ENCODER_OUTPUT")
            .ok_or_else(|| AppError::CudaError("Missing decoder joint ENCODER_OUTPUT input region".to_string()))?;
        
        let state_input_region = self.decoder_joint_pool.get_input_region("DECODER_STATE")
            .ok_or_else(|| AppError::CudaError("Missing decoder joint DECODER_STATE input region".to_string()))?;
        
        let logits_output_region = self.decoder_joint_pool.get_output_region("LOGITS")
            .ok_or_else(|| AppError::CudaError("Missing decoder joint LOGITS output region".to_string()))?;
        
        let state_output_region = self.decoder_joint_pool.get_output_region("UPDATED_DECODER_STATE")
            .ok_or_else(|| AppError::CudaError("Missing decoder joint UPDATED_DECODER_STATE output region".to_string()))?;
        
        // Write input data to CUDA memory
        encoder_input_region.write_f32_data(encoder_output)
            .map_err(|e| AppError::CudaError(format!("Failed to write decoder joint encoder output: {}", e)))?;
        
        state_input_region.write_f32_data(decoder_state)
            .map_err(|e| AppError::CudaError(format!("Failed to write decoder joint state: {}", e)))?;
        
        // Run inference
        encoder_input_region.run_inference_with_output_regions(
            logits_output_region,
            &self.decoder_joint_pool.config,
            "ENCODER_OUTPUT",
            "LOGITS",
        ).map_err(|e| AppError::CudaError(format!("Decoder joint inference failed: {}", e)))?;
        
        // Read output data
        let logits_size = self.decoder_joint_pool.config.calculate_output_buffer_size("LOGITS")
            .ok_or_else(|| AppError::CudaError("Failed to calculate logits buffer size".to_string()))?;
        
        let state_size = self.decoder_joint_pool.config.calculate_output_buffer_size("UPDATED_DECODER_STATE")
            .ok_or_else(|| AppError::CudaError("Failed to calculate decoder state buffer size".to_string()))?;
        
        let logits = logits_output_region.read_f32_data(logits_size / 4)
            .map_err(|e| AppError::CudaError(format!("Failed to read logits: {}", e)))?;
        
        let updated_decoder_state = state_output_region.read_f32_data(state_size / 4)
            .map_err(|e| AppError::CudaError(format!("Failed to read decoder state: {}", e)))?;
        
        // Update the decoder state
        *decoder_state = updated_decoder_state.clone();
        
        debug!("Decoder joint completed, logits size: {}", logits.len());
        Ok((logits, updated_decoder_state))
    }
    
    /// Initialize encoder state with zeros
    fn initialize_encoder_state(&self) -> Vec<f32> {
        let state_size = self.encoder_pool.config.calculate_buffer_size("ENCODER_STATE")
            .unwrap_or(512 * 2048 * 4); // Default size
        vec![0.0; state_size / 4] // f32 is 4 bytes
    }
    
    /// Initialize decoder state with zeros
    fn initialize_decoder_state(&self) -> Vec<f32> {
        let state_size = self.decoder_joint_pool.config.calculate_buffer_size("DECODER_STATE")
            .unwrap_or(512 * 1024 * 4); // Default size
        vec![0.0; state_size / 4] // f32 is 4 bytes
    }
    
    /// Convert logits to tokens using greedy decoding
    fn logits_to_tokens(&self, logits: &[f32]) -> Vec<u32> {
        // Assuming logits are shaped as [batch=1, time, vocab_size]
        // For now, simple greedy decoding - take argmax of each time step
        let vocab_size = 4096; // From model config
        let time_steps = logits.len() / vocab_size;
        
        let mut tokens = Vec::new();
        for t in 0..time_steps {
            let start_idx = t * vocab_size;
            let end_idx = start_idx + vocab_size;
            let time_logits = &logits[start_idx..end_idx];
            
            // Find the token with highest probability
            let mut max_idx = 0;
            let mut max_val = time_logits[0];
            for (i, &val) in time_logits.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            tokens.push(max_idx as u32);
        }
        
        tokens
    }
    
    /// Convert tokens to text using vocabulary
    fn tokens_to_text(&self, tokens: &[u32]) -> String {
        tokens.iter()
            .filter_map(|&token| self.vocabulary.get_token(token))
            .collect::<Vec<_>>()
            .join(" ")
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
        
        // Get or initialize encoder/decoder states
        let mut encoder_state = state.encoder_state.clone()
            .unwrap_or_else(|| self.initialize_encoder_state());
        
        let mut decoder_state = state.decoder_state.clone()
            .unwrap_or_else(|| self.initialize_decoder_state());
        
        // Run the three-stage pipeline
        let mel_features = self.run_preprocessor(audio_samples).await?;
        let (encoder_output, updated_encoder_state) = self.run_encoder(&mel_features, &mut encoder_state).await?;
        let (logits, updated_decoder_state) = self.run_decoder_joint(&encoder_output, &mut decoder_state).await?;
        
        // Update state
        state.encoder_state = Some(updated_encoder_state);
        state.decoder_state = Some(updated_decoder_state);
        
        // Convert logits to tokens and then to text
        let tokens = self.logits_to_tokens(&logits);
        let text = self.tokens_to_text(&tokens);
        
        debug!("Stream processing completed: {}", text);
        
        Ok(Transcription {
            text,
            tokens,
            confidence: 1.0, // TODO: Calculate actual confidence
            is_final: false,
        })
    }
    
    async fn process_batch_samples(&self, audio_samples: &[f32]) -> Result<Transcription> {
        debug!("Processing batch samples: {} samples", audio_samples.len());
        
        // Initialize fresh states for batch processing
        let mut encoder_state = self.initialize_encoder_state();
        let mut decoder_state = self.initialize_decoder_state();
        
        // Run the three-stage pipeline
        let mel_features = self.run_preprocessor(audio_samples).await?;
        let (encoder_output, _) = self.run_encoder(&mel_features, &mut encoder_state).await?;
        let (logits, _) = self.run_decoder_joint(&encoder_output, &mut decoder_state).await?;
        
        // Convert logits to tokens and then to text
        let tokens = self.logits_to_tokens(&logits);
        let text = self.tokens_to_text(&tokens);
        
        debug!("Batch processing completed: {}", text);
        
        Ok(Transcription {
            text,
            tokens,
            confidence: 1.0, // TODO: Calculate actual confidence
            is_final: true,
        })
    }
}

/// Builder for creating CUDA ASR pipelines
pub struct CudaAsrPipelineBuilder {
    device_id: Option<i32>,
    vocabulary: Option<Arc<Vocabulary>>,
    sample_rate: Option<f32>,
    window_size: Option<usize>,
}

impl CudaAsrPipelineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            device_id: None,
            vocabulary: None,
            sample_rate: None,
            window_size: None,
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
    
    /// Set sample rate
    pub fn sample_rate(mut self, sample_rate: f32) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }
    
    /// Set window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }
    
    /// Build the pipeline
    pub fn build(self) -> Result<CudaAsrPipeline> {
        let device_id = self.device_id.unwrap_or(0);
        let vocabulary = self.vocabulary.ok_or_else(|| AppError::ConfigError("Vocabulary is required".to_string()))?;
        let sample_rate = self.sample_rate.unwrap_or(16000.0);
        let window_size = self.window_size.unwrap_or(1024);
        
        CudaAsrPipeline::new(device_id, vocabulary, sample_rate, window_size)
    }
}

impl Default for CudaAsrPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}