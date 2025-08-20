# REFACTORING.md (Developer Notes)

This file contains internal refactoring ideas and notes. For the current system design and performance guidance, see:

- `docs/Architecture.md`
- `docs/Performance.md`

## 🏗️ **Major Architectural & Stylistic Improvements**

### **1. Error Handling & Result Types**

**Current Issues:**
- Inconsistent error handling patterns across modules
- Some modules use `Result<T>` while others use `std::result::Result<T, E>`
- Error context is sometimes lost in conversions

**Improvements:**
```rust
// Create domain-specific error types instead of generic AppError
#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("Audio processing failed: {0}")]
    AudioProcessing(#[from] AudioError),
    
    #[error("Model inference failed: {0}")]
    ModelInference(#[from] ModelError),
    
    #[error("Decoder state invalid: {0}")]
    DecoderState(String),
}

#[derive(Debug, thiserror::Error)]
pub enum TritonError {
    #[error("Connection failed: {0}")]
    Connection(#[from] tonic::transport::Error),
    
    #[error("Inference timeout: {0}")]
    Timeout(#[from] tokio::time::error::Elapsed),
    
    #[error("Pool exhausted: {0}")]
    PoolExhausted(String),
}
```

### **2. Type System & Domain Modeling**

**Current Issues:**
- Primitive obsession (using `Vec<f32>`, `String`, `i32` directly)
- Missing newtype wrappers for domain concepts
- Lack of compile-time guarantees for audio sample rates, tensor shapes

**Improvements:**
```rust
// Strong typing for domain concepts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRate(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSamples(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenId(pub i32);

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    samples: Vec<f32>,
    sample_rate: SampleRate,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: SampleRate) -> Self {
        Self { samples, sample_rate }
    }
    
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f32(self.samples.len() as f32 / self.sample_rate.0 as f32)
    }
}
```

### **3. Module Organization & Separation of Concerns**

**Current Issues:**
- Mixed concerns in some modules (e.g., `config.rs` has both config and constants)
- Inconsistent naming patterns
- Some modules are too large and handle multiple responsibilities

**Improvements:**
```rust
// Separate domain constants from configuration
pub mod constants {
    pub mod audio {
        pub const SAMPLE_RATE: u32 = 16000;
        pub const BUFFER_CAPACITY: usize = 1024 * 1024;
    }
    
    pub mod model {
        pub const BLANK_TOKEN_ID: i32 = 1024;
        pub const DECODER_STATE_SIZE: usize = 640;
    }
}

// Separate config loading from config types
pub mod config {
    mod loader;
    mod types;
    mod validation;
    
    pub use loader::ConfigLoader;
    pub use types::{Config, ServerConfig, TritonConfig};
    pub use validation::ConfigValidator;
}
```

### **4. Trait Design & Abstractions**

**Current Issues:**
- Some traits are too broad (e.g., `AsrPipeline` handles both streaming and batch)
- Missing abstractions for common patterns
- Inconsistent async trait usage

**Improvements:**
```rust
// Separate streaming and batch processing
#[async_trait]
pub trait StreamingAsrProcessor {
    async fn process_chunk(
        &self,
        audio: &AudioBuffer,
        state: &mut DecoderState,
    ) -> Result<PartialTranscription, AsrError>;
}

#[async_trait]
pub trait BatchAsrProcessor {
    async fn process_complete(
        &self,
        audio: &AudioBuffer,
    ) -> Result<CompleteTranscription, AsrError>;
}

// Abstract over different model backends
#[async_trait]
pub trait ModelBackend {
    type Error: std::error::Error + Send + Sync + 'static;
    
    async fn infer<I, O>(&self, input: I) -> Result<O, Self::Error>
    where
        I: ModelInput + Send,
        O: ModelOutput + Send;
}
```

### **5. Builder Pattern & Configuration**

**Current Issues:**
- Direct struct construction in many places
- Missing validation during construction
- Configuration scattered across multiple locations

**Improvements:**
```rust
// Builder pattern for complex objects
pub struct AsrPipelineBuilder {
    triton_endpoint: Option<String>,
    vocabulary: Option<Arc<Vocabulary>>,
    connection_pool_config: Option<PoolConfig>,
    performance_config: Option<PerformanceConfig>,
}

impl AsrPipelineBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_triton_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.triton_endpoint = Some(endpoint.into());
        self
    }
    
    pub fn with_vocabulary(mut self, vocab: Arc<Vocabulary>) -> Self {
        self.vocabulary = Some(vocab);
        self
    }
    
    pub fn build(self) -> Result<TritonAsrPipeline, AsrError> {
        let triton_endpoint = self.triton_endpoint
            .ok_or_else(|| AsrError::Configuration("Triton endpoint required".into()))?;
        
        // Validation and construction logic
        // ...
    }
}
```

### **6. Memory Management & Resource Handling**

**Current Issues:**
- Manual memory pool management
- Inconsistent use of `Arc` vs `Rc`
- Missing RAII patterns for resources

**Improvements:**
```rust
// RAII for pooled resources
pub struct PooledConnection<T> {
    inner: Option<T>,
    pool: Arc<ConnectionPool<T>>,
}

impl<T> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        if let Some(conn) = self.inner.take() {
            self.pool.return_connection(conn);
        }
    }
}

impl<T> std::ops::Deref for PooledConnection<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().expect("Connection already returned")
    }
}
```

### **7. Async Patterns & Concurrency**

**Current Issues:**
- Inconsistent use of `tokio::spawn` vs direct `.await`
- Missing structured concurrency patterns
- Some blocking operations in async contexts

**Improvements:**
```rust
// Structured concurrency with proper error handling
pub async fn process_pipeline_concurrent(
    audio: &AudioBuffer,
    models: &ModelTriad,
) -> Result<Transcription, AsrError> {
    let (preprocessor_result, encoder_ready, decoder_ready) = tokio::try_join!(
        models.preprocessor.prepare(audio),
        models.encoder.prepare(),
        models.decoder.prepare()
    )?;
    
    let (features, encoded) = tokio::try_join!(
        models.preprocessor.process(preprocessor_result),
        models.encoder.process(encoder_ready, preprocessor_result)
    )?;
    
    models.decoder.process(decoder_ready, encoded).await
}
```

### **8. Testing & Testability**

**Current Issues:**
- Hard-coded dependencies make testing difficult
- Missing mock implementations
- Integration tests mixed with unit tests

**Improvements:**
```rust
// Dependency injection for testability
pub trait TimeProvider {
    fn now(&self) -> Instant;
}

pub struct SystemTimeProvider;

impl TimeProvider for SystemTimeProvider {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

pub struct MockTimeProvider {
    current_time: Arc<Mutex<Instant>>,
}

// Make components generic over their dependencies
pub struct CircuitBreaker<T: TimeProvider> {
    time_provider: T,
    // ... other fields
}
```

### **9. Performance & Optimization**

**Current Issues:**
- Unnecessary allocations in hot paths
- Missing `#[inline]` attributes on small functions
- Suboptimal data structures for specific use cases

**Improvements:**
```rust
// Zero-allocation hot paths
pub struct AudioProcessor {
    buffer: Vec<f32>,  // Reused buffer
    workspace: Vec<f32>,  // Reused workspace
}

impl AudioProcessor {
    #[inline]
    pub fn process_samples(&mut self, input: &[f32]) -> &[f32] {
        self.buffer.clear();
        self.buffer.extend_from_slice(input);
        
        // Process in-place to avoid allocations
        self.apply_windowing_inplace();
        self.apply_normalization_inplace();
        
        &self.buffer
    }
    
    #[inline(always)]
    fn apply_windowing_inplace(&mut self) {
        // SIMD-optimized windowing
    }
}
```

### **10. Documentation & API Design**

**Current Issues:**
- Inconsistent documentation style
- Missing examples in docs
- Some public APIs are too complex

**Improvements:**
```rust
/// High-performance ASR pipeline for real-time speech recognition.
///
/// This pipeline processes audio through three stages:
/// 1. Preprocessing: Converts raw audio to feature vectors
/// 2. Encoding: Processes features through RNN-T encoder
/// 3. Decoding: Generates text tokens from encoded features
///
/// # Examples
///
/// ```rust
/// use amira_rust_asr_server::asr::{AsrPipeline, AsrPipelineBuilder};
/// 
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let pipeline = AsrPipelineBuilder::new()
///     .with_triton_endpoint("http://localhost:8001")
///     .with_vocabulary_file("vocab.txt")
///     .build()
///     .await?;
/// 
/// let audio = AudioBuffer::from_file("audio.wav")?;
/// let transcription = pipeline.process_batch(&audio).await?;
/// println!("Transcription: {}", transcription.text);
/// # Ok(())
/// # }
/// ```
///
/// # Performance Notes
///
/// - Uses connection pooling for optimal throughput
/// - Implements zero-copy operations where possible
/// - Supports both streaming and batch processing modes
pub struct TritonAsrPipeline {
    // ...
}
```

## 🎯 **Priority Recommendations**

1. **Start with error handling** - Create domain-specific error types
2. **Implement builder patterns** - For complex object construction
3. **Add strong typing** - Replace primitive types with newtypes
4. **Separate concerns** - Split large modules into focused ones
5. **Add comprehensive tests** - With dependency injection for testability

These improvements will make your codebase more maintainable, testable, and idiomatic Rust while preserving the high-performance characteristics you've built.