//! Builder patterns for ASR components.
//!
//! This module provides builder patterns for complex ASR objects, ensuring proper
//! validation and configuration during construction.

use crate::error::ConfigError;
use crate::types::{PoolSize, TimeoutDuration, ModelName, SampleRate};
use crate::asr::traits::{AsrConfig, TimeProvider, SystemTimeProvider};
use std::path::PathBuf;
use std::time::Duration;

/// Performance configuration for ASR pipeline.
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations.
    pub enable_simd: bool,
    /// CPU affinity settings.
    pub cpu_affinity: Option<Vec<usize>>,
    /// NUMA node preference.
    pub numa_node: Option<usize>,
    /// Memory pool configuration.
    pub memory_pool: MemoryPoolConfig,
    /// Circuit breaker configuration.
    pub circuit_breaker: CircuitBreakerConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            cpu_affinity: None,
            numa_node: None,
            memory_pool: MemoryPoolConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

/// Memory pool configuration.
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size.
    pub initial_size: usize,
    /// Maximum pool size.
    pub max_size: usize,
    /// Pool growth factor.
    pub growth_factor: f32,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024,
            max_size: 10240,
            growth_factor: 1.5,
        }
    }
}

/// Circuit breaker configuration.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening.
    pub failure_threshold: usize,
    /// Recovery timeout.
    pub recovery_timeout: Duration,
    /// Half-open max calls.
    pub half_open_max_calls: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

/// Configuration for ASR pipeline.
#[derive(Debug, Clone)]
pub struct AsrPipelineConfig {
    /// Triton server endpoint.
    pub triton_endpoint: String,
    /// Connection pool size.
    pub pool_size: PoolSize,
    /// Inference timeout.
    pub inference_timeout: TimeoutDuration,
    /// Vocabulary file path.
    pub vocabulary_path: PathBuf,
    /// Expected sample rate.
    pub sample_rate: SampleRate,
    /// Maximum audio length in samples.
    pub max_audio_length: usize,
    /// Optimal batch size for processing.
    pub optimal_batch_size: usize,
    /// Performance configuration.
    pub performance: PerformanceConfig,
}

impl AsrConfig for AsrPipelineConfig {
    fn triton_endpoint(&self) -> &str {
        &self.triton_endpoint
    }
    
    fn vocabulary_path(&self) -> &str {
        self.vocabulary_path.to_str().unwrap_or("")
    }
    
    fn pool_size(&self) -> usize {
        self.pool_size.value()
    }
    
    fn inference_timeout(&self) -> Duration {
        self.inference_timeout.value()
    }
    
    fn max_audio_length(&self) -> usize {
        self.max_audio_length
    }
    
    fn optimal_batch_size(&self) -> usize {
        self.optimal_batch_size
    }
}

/// Builder for ASR pipeline configuration.
pub struct AsrPipelineBuilder<T = SystemTimeProvider> {
    triton_endpoint: Option<String>,
    vocabulary_path: Option<PathBuf>,
    pool_size: Option<PoolSize>,
    inference_timeout: Option<TimeoutDuration>,
    sample_rate: Option<SampleRate>,
    max_audio_length: Option<usize>,
    optimal_batch_size: Option<usize>,
    performance_config: Option<PerformanceConfig>,
    #[allow(dead_code)]
    time_provider: Option<T>,
}

impl AsrPipelineBuilder<SystemTimeProvider> {
    /// Create a new ASR pipeline builder.
    pub fn new() -> Self {
        Self {
            triton_endpoint: None,
            vocabulary_path: None,
            pool_size: None,
            inference_timeout: None,
            sample_rate: None,
            max_audio_length: None,
            optimal_batch_size: None,
            performance_config: None,
            time_provider: None,
        }
    }
}

impl<T> AsrPipelineBuilder<T> 
where
    T: TimeProvider + Clone,
{
    /// Set the Triton endpoint.
    pub fn with_triton_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.triton_endpoint = Some(endpoint.into());
        self
    }
    
    /// Set the vocabulary file path.
    pub fn with_vocabulary_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.vocabulary_path = Some(path.into());
        self
    }
    
    /// Set the connection pool size.
    pub fn with_pool_size(mut self, size: PoolSize) -> Self {
        self.pool_size = Some(size);
        self
    }
    
    /// Set the inference timeout.
    pub fn with_inference_timeout(mut self, timeout: TimeoutDuration) -> Self {
        self.inference_timeout = Some(timeout);
        self
    }
    
    /// Set the expected sample rate.
    pub fn with_sample_rate(mut self, rate: SampleRate) -> Self {
        self.sample_rate = Some(rate);
        self
    }
    
    /// Set the maximum audio length.
    pub fn with_max_audio_length(mut self, length: usize) -> Self {
        self.max_audio_length = Some(length);
        self
    }
    
    /// Set the optimal batch size.
    pub fn with_optimal_batch_size(mut self, size: usize) -> Self {
        self.optimal_batch_size = Some(size);
        self
    }
    
    /// Set the performance configuration.
    pub fn with_performance_config(mut self, config: PerformanceConfig) -> Self {
        self.performance_config = Some(config);
        self
    }
    
    /// Set a custom time provider (for testing).
    pub fn with_time_provider<U>(self, provider: U) -> AsrPipelineBuilder<U>
    where
        U: TimeProvider + Clone,
    {
        AsrPipelineBuilder {
            triton_endpoint: self.triton_endpoint,
            vocabulary_path: self.vocabulary_path,
            pool_size: self.pool_size,
            inference_timeout: self.inference_timeout,
            sample_rate: self.sample_rate,
            max_audio_length: self.max_audio_length,
            optimal_batch_size: self.optimal_batch_size,
            performance_config: self.performance_config,
            time_provider: Some(provider),
        }
    }
    
    /// Enable SIMD optimizations.
    pub fn enable_simd(mut self) -> Self {
        let mut config = self.performance_config.unwrap_or_default();
        config.enable_simd = true;
        self.performance_config = Some(config);
        self
    }
    
    /// Disable SIMD optimizations.
    pub fn disable_simd(mut self) -> Self {
        let mut config = self.performance_config.unwrap_or_default();
        config.enable_simd = false;
        self.performance_config = Some(config);
        self
    }
    
    /// Set CPU affinity.
    pub fn with_cpu_affinity(mut self, affinity: Vec<usize>) -> Self {
        let mut config = self.performance_config.unwrap_or_default();
        config.cpu_affinity = Some(affinity);
        self.performance_config = Some(config);
        self
    }
    
    /// Set NUMA node preference.
    pub fn with_numa_node(mut self, node: usize) -> Self {
        let mut config = self.performance_config.unwrap_or_default();
        config.numa_node = Some(node);
        self.performance_config = Some(config);
        self
    }
    
    /// Build the ASR pipeline configuration.
    pub fn build(self) -> Result<AsrPipelineConfig, ConfigError> {
        let triton_endpoint = self.triton_endpoint.ok_or_else(|| ConfigError::MissingField {
            field: "triton_endpoint".to_string(),
        })?;
        
        let vocabulary_path = self.vocabulary_path.ok_or_else(|| ConfigError::MissingField {
            field: "vocabulary_path".to_string(),
        })?;
        
        // Validate vocabulary file exists
        if !vocabulary_path.exists() {
            return Err(ConfigError::FileNotFound {
                path: vocabulary_path.display().to_string(),
            });
        }
        
        // Validate Triton endpoint format
        if !triton_endpoint.starts_with("http://") && !triton_endpoint.starts_with("https://") {
            return Err(ConfigError::InvalidValue {
                field: "triton_endpoint".to_string(),
                value: triton_endpoint,
            });
        }
        
        let pool_size = self.pool_size.unwrap_or(PoolSize::DEFAULT);
        let inference_timeout = self.inference_timeout.unwrap_or(TimeoutDuration::DEFAULT);
        let sample_rate = self.sample_rate.unwrap_or(SampleRate::STANDARD_16KHZ);
        let max_audio_length = self.max_audio_length.unwrap_or(16000 * 300); // 5 minutes at 16kHz
        let optimal_batch_size = self.optimal_batch_size.unwrap_or(8);
        let performance = self.performance_config.unwrap_or_default();
        
        // Additional validation
        if max_audio_length == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_audio_length".to_string(),
                value: "0".to_string(),
            });
        }
        
        if optimal_batch_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "optimal_batch_size".to_string(),
                value: "0".to_string(),
            });
        }
        
        Ok(AsrPipelineConfig {
            triton_endpoint,
            pool_size,
            inference_timeout,
            vocabulary_path,
            sample_rate,
            max_audio_length,
            optimal_batch_size,
            performance,
        })
    }
}

impl Default for AsrPipelineBuilder<SystemTimeProvider> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for model configurations.
pub struct ModelConfigBuilder {
    name: Option<ModelName>,
    input_shape: Option<Vec<usize>>,
    output_shape: Option<Vec<usize>>,
    batch_size: Option<usize>,
    timeout: Option<TimeoutDuration>,
}

impl ModelConfigBuilder {
    /// Create a new model configuration builder.
    pub fn new() -> Self {
        Self {
            name: None,
            input_shape: None,
            output_shape: None,
            batch_size: None,
            timeout: None,
        }
    }
    
    /// Set the model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Result<Self, ConfigError> {
        self.name = Some(ModelName::new(name)?);
        Ok(self)
    }
    
    /// Set the input shape.
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = Some(shape);
        self
    }
    
    /// Set the output shape.
    pub fn with_output_shape(mut self, shape: Vec<usize>) -> Self {
        self.output_shape = Some(shape);
        self
    }
    
    /// Set the batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }
    
    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: TimeoutDuration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    
    /// Build the model configuration.
    pub fn build(self) -> Result<ModelConfig, ConfigError> {
        let name = self.name.ok_or_else(|| ConfigError::MissingField {
            field: "name".to_string(),
        })?;
        
        let input_shape = self.input_shape.ok_or_else(|| ConfigError::MissingField {
            field: "input_shape".to_string(),
        })?;
        
        let output_shape = self.output_shape.ok_or_else(|| ConfigError::MissingField {
            field: "output_shape".to_string(),
        })?;
        
        if input_shape.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "input_shape".to_string(),
                value: "empty".to_string(),
            });
        }
        
        if output_shape.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "output_shape".to_string(),
                value: "empty".to_string(),
            });
        }
        
        let batch_size = self.batch_size.unwrap_or(1);
        let timeout = self.timeout.unwrap_or(TimeoutDuration::DEFAULT);
        
        Ok(ModelConfig {
            name,
            input_shape,
            output_shape,
            batch_size,
            timeout,
        })
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Model configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model name.
    pub name: ModelName,
    /// Input tensor shape.
    pub input_shape: Vec<usize>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Batch size for inference.
    pub batch_size: usize,
    /// Inference timeout.
    pub timeout: TimeoutDuration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::traits::MockTimeProvider;
    use std::fs;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_asr_pipeline_builder_success() {
        let temp_vocab = NamedTempFile::new().unwrap();
        fs::write(&temp_vocab, "word1\nword2\nword3").unwrap();
        
        let config = AsrPipelineBuilder::new()
            .with_triton_endpoint("http://localhost:8001")
            .with_vocabulary_path(temp_vocab.path())
            .with_pool_size(PoolSize::new(5).unwrap())
            .with_inference_timeout(TimeoutDuration::from_secs(10).unwrap())
            .build()
            .unwrap();
        
        assert_eq!(config.triton_endpoint, "http://localhost:8001");
        assert_eq!(config.pool_size.value(), 5);
        assert_eq!(config.inference_timeout.value(), Duration::from_secs(10));
    }
    
    #[test]
    fn test_asr_pipeline_builder_missing_endpoint() {
        let result = AsrPipelineBuilder::new()
            .with_vocabulary_path("/nonexistent/path")
            .build();
        
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::MissingField { field } => assert_eq!(field, "triton_endpoint"),
            _ => panic!("Expected MissingField error"),
        }
    }
    
    #[test]
    fn test_asr_pipeline_builder_invalid_endpoint() {
        let temp_vocab = NamedTempFile::new().unwrap();
        fs::write(&temp_vocab, "word1\nword2\nword3").unwrap();
        
        let result = AsrPipelineBuilder::new()
            .with_triton_endpoint("invalid-url")
            .with_vocabulary_path(temp_vocab.path())
            .build();
        
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::InvalidValue { field, value } => {
                assert_eq!(field, "triton_endpoint");
                assert_eq!(value, "invalid-url");
            }
            _ => panic!("Expected InvalidValue error"),
        }
    }
    
    #[test]
    fn test_asr_pipeline_builder_with_time_provider() {
        let temp_vocab = NamedTempFile::new().unwrap();
        fs::write(&temp_vocab, "word1\nword2\nword3").unwrap();
        
        let mock_time = MockTimeProvider::new();
        let config = AsrPipelineBuilder::new()
            .with_triton_endpoint("http://localhost:8001")
            .with_vocabulary_path(temp_vocab.path())
            .with_time_provider(mock_time)
            .build()
            .unwrap();
        
        assert_eq!(config.triton_endpoint, "http://localhost:8001");
    }
    
    #[test]
    fn test_model_config_builder_success() {
        let config = ModelConfigBuilder::new()
            .with_name("test_model").unwrap()
            .with_input_shape(vec![1, 128, 80])
            .with_output_shape(vec![1, 128, 1024])
            .with_batch_size(4)
            .build()
            .unwrap();
        
        assert_eq!(config.name.as_str(), "test_model");
        assert_eq!(config.input_shape, vec![1, 128, 80]);
        assert_eq!(config.output_shape, vec![1, 128, 1024]);
        assert_eq!(config.batch_size, 4);
    }
    
    #[test]
    fn test_model_config_builder_missing_name() {
        let result = ModelConfigBuilder::new()
            .with_input_shape(vec![1, 128, 80])
            .with_output_shape(vec![1, 128, 1024])
            .build();
        
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfigError::MissingField { field } => assert_eq!(field, "name"),
            _ => panic!("Expected MissingField error"),
        }
    }
    
    #[test]
    fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();
        assert!(config.enable_simd);
        assert!(config.cpu_affinity.is_none());
        assert!(config.numa_node.is_none());
        assert_eq!(config.memory_pool.initial_size, 1024);
        assert_eq!(config.circuit_breaker.failure_threshold, 5);
    }
}