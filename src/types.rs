//! Strong typing with newtypes for domain concepts.
//!
//! This module provides type-safe wrappers around primitive types to prevent
//! common errors and provide better API design through the type system.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use crate::error::AudioError;

/// Sample rate in Hz.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SampleRate(pub u32);

impl SampleRate {
    /// Standard 16kHz sample rate for ASR.
    pub const STANDARD_16KHZ: Self = Self(16000);
    
    /// Standard 8kHz sample rate for telephony.
    pub const STANDARD_8KHZ: Self = Self(8000);
    
    /// High quality 48kHz sample rate.
    pub const HIGH_QUALITY_48KHZ: Self = Self(48000);
    
    /// Create a new sample rate with validation.
    pub fn new(rate: u32) -> Result<Self, AudioError> {
        if rate == 0 {
            return Err(AudioError::InvalidFormat(
                "Sample rate cannot be zero".to_string()
            ));
        }
        if rate > 192000 {
            return Err(AudioError::InvalidFormat(
                format!("Sample rate {} too high (max 192kHz)", rate)
            ));
        }
        Ok(Self(rate))
    }
    
    /// Get the sample rate value.
    pub fn value(self) -> u32 {
        self.0
    }
    
    /// Convert to f32 for calculations.
    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }
}

impl std::fmt::Display for SampleRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Hz", self.0)
    }
}

/// Number of audio samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AudioSamples(pub usize);

impl AudioSamples {
    /// Create a new sample count.
    pub fn new(count: usize) -> Self {
        Self(count)
    }
    
    /// Get the sample count.
    pub fn count(self) -> usize {
        self.0
    }
    
    /// Calculate duration given a sample rate.
    pub fn duration(self, sample_rate: SampleRate) -> Duration {
        Duration::from_secs_f64(self.0 as f64 / sample_rate.as_f32() as f64)
    }
    
    /// Check if this represents silence (zero samples).
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for AudioSamples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} samples", self.0)
    }
}

/// Token ID for vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TokenId(pub i32);

impl TokenId {
    /// Blank token ID (commonly used in CTC/RNN-T).
    pub const BLANK: Self = Self(0);
    
    /// Unknown token ID.
    pub const UNKNOWN: Self = Self(-1);
    
    /// Create a new token ID.
    pub fn new(id: i32) -> Self {
        Self(id)
    }
    
    /// Get the token ID value.
    pub fn value(self) -> i32 {
        self.0
    }
    
    /// Check if this is a blank token.
    pub fn is_blank(self) -> bool {
        self.0 == 0
    }
    
    /// Check if this is an unknown token.
    pub fn is_unknown(self) -> bool {
        self.0 == -1
    }
}

impl std::fmt::Display for TokenId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "token_{}", self.0)
    }
}

/// Model name identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelName(pub String);

impl ModelName {
    /// Create a new model name with validation.
    pub fn new(name: impl Into<String>) -> Result<Self, crate::error::ModelError> {
        let name = name.into();
        if name.is_empty() {
            return Err(crate::error::ModelError::NotFound { 
                model_name: "empty model name".to_string() 
            });
        }
        Ok(Self(name))
    }
    
    /// Get the model name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Get the owned model name.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Tensor shape dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    /// Create a new tensor shape.
    pub fn new(dimensions: Vec<usize>) -> Self {
        Self(dimensions)
    }
    
    /// Get the dimensions.
    pub fn dimensions(&self) -> &[usize] {
        &self.0
    }
    
    /// Get the number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    
    /// Calculate total number of elements.
    pub fn total_elements(&self) -> usize {
        self.0.iter().product()
    }
    
    /// Check if this is a scalar (0 dimensions).
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }
    
    /// Check if this is a vector (1 dimension).
    pub fn is_vector(&self) -> bool {
        self.0.len() == 1
    }
    
    /// Check if this is a matrix (2 dimensions).
    pub fn is_matrix(&self) -> bool {
        self.0.len() == 2
    }
}

impl std::fmt::Display for TensorShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
}

/// Connection pool size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PoolSize(pub usize);

impl PoolSize {
    /// Default pool size.
    pub const DEFAULT: Self = Self(10);
    
    /// Minimum pool size.
    pub const MIN: Self = Self(1);
    
    /// Maximum pool size.
    pub const MAX: Self = Self(1000);
    
    /// Create a new pool size with validation.
    pub fn new(size: usize) -> Result<Self, crate::error::ConfigError> {
        if size == 0 {
            return Err(crate::error::ConfigError::InvalidValue {
                field: "pool_size".to_string(),
                value: size.to_string(),
            });
        }
        if size > Self::MAX.0 {
            return Err(crate::error::ConfigError::InvalidValue {
                field: "pool_size".to_string(),
                value: format!("{} (max {})", size, Self::MAX.0),
            });
        }
        Ok(Self(size))
    }
    
    /// Get the pool size.
    pub fn value(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for PoolSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} connections", self.0)
    }
}

/// Timeout duration with validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimeoutDuration(pub Duration);

impl TimeoutDuration {
    /// Default timeout (5 seconds).
    pub const DEFAULT: Self = Self(Duration::from_secs(5));
    
    /// Short timeout (1 second).
    pub const SHORT: Self = Self(Duration::from_secs(1));
    
    /// Long timeout (30 seconds).
    pub const LONG: Self = Self(Duration::from_secs(30));
    
    /// Create a new timeout duration with validation.
    pub fn new(duration: Duration) -> Result<Self, crate::error::ConfigError> {
        if duration.is_zero() {
            return Err(crate::error::ConfigError::InvalidValue {
                field: "timeout".to_string(),
                value: "0".to_string(),
            });
        }
        if duration > Duration::from_secs(300) {
            return Err(crate::error::ConfigError::InvalidValue {
                field: "timeout".to_string(),
                value: format!("{:?} (max 300s)", duration),
            });
        }
        Ok(Self(duration))
    }
    
    /// Get the duration.
    pub fn value(self) -> Duration {
        self.0
    }
    
    /// Create from milliseconds.
    pub fn from_millis(ms: u64) -> Result<Self, crate::error::ConfigError> {
        Self::new(Duration::from_millis(ms))
    }
    
    /// Create from seconds.
    pub fn from_secs(secs: u64) -> Result<Self, crate::error::ConfigError> {
        Self::new(Duration::from_secs(secs))
    }
}

impl std::fmt::Display for TimeoutDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Audio buffer with validated sample rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBuffer {
    samples: Vec<f32>,
    sample_rate: SampleRate,
}

impl AudioBuffer {
    /// Create a new audio buffer with validation.
    pub fn new(samples: Vec<f32>, sample_rate: SampleRate) -> Result<Self, AudioError> {
        if samples.is_empty() {
            return Err(AudioError::BufferUnderrun);
        }
        
        // Check for invalid samples (NaN, infinity)
        for (i, &sample) in samples.iter().enumerate() {
            if !sample.is_finite() {
                return Err(AudioError::InvalidFormat(
                    format!("Invalid sample at index {}: {}", i, sample)
                ));
            }
        }
        
        Ok(Self { samples, sample_rate })
    }
    
    /// Get the samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    
    /// Get mutable samples.
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.samples
    }
    
    /// Get the sample rate.
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }
    
    /// Get the number of samples.
    pub fn len(&self) -> AudioSamples {
        AudioSamples::new(self.samples.len())
    }
    
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
    
    /// Calculate the duration of this audio buffer.
    pub fn duration(&self) -> Duration {
        self.len().duration(self.sample_rate)
    }
    
    /// Split into owned samples and sample rate.
    pub fn into_parts(self) -> (Vec<f32>, SampleRate) {
        (self.samples, self.sample_rate)
    }
    
    /// Create from raw parts without validation (unsafe).
    /// 
    /// # Safety
    /// 
    /// The caller must ensure that:
    /// - samples contains valid finite f32 values
    /// - samples is not empty
    /// - sample_rate is valid
    pub unsafe fn from_raw_parts(samples: Vec<f32>, sample_rate: SampleRate) -> Self {
        Self { samples, sample_rate }
    }
    
    /// Normalize audio samples to [-1.0, 1.0] range.
    pub fn normalize(&mut self) {
        if self.samples.is_empty() {
            return;
        }
        
        let max_abs = self.samples.iter()
            .map(|&s| s.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        
        if max_abs > 0.0 {
            for sample in &mut self.samples {
                *sample /= max_abs;
            }
        }
    }
    
    /// Apply a window function to the samples.
    pub fn apply_window(&mut self, window: &[f32]) -> Result<(), AudioError> {
        if window.len() != self.samples.len() {
            return Err(AudioError::Windowing(
                format!("Window length {} doesn't match sample length {}", 
                    window.len(), self.samples.len())
            ));
        }
        
        for (sample, &window_val) in self.samples.iter_mut().zip(window) {
            *sample *= window_val;
        }
        
        Ok(())
    }
}

impl std::fmt::Display for AudioBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AudioBuffer({} @ {})", self.len(), self.sample_rate)
    }
}

/// Confidence score for ASR results.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ConfidenceScore(pub f32);

impl ConfidenceScore {
    /// Create a new confidence score with validation.
    pub fn new(score: f32) -> Result<Self, crate::error::AsrError> {
        if !score.is_finite() {
            return Err(crate::error::AsrError::Pipeline(
                format!("Invalid confidence score: {}", score)
            ));
        }
        if !(0.0..=1.0).contains(&score) {
            return Err(crate::error::AsrError::Pipeline(
                format!("Confidence score {} must be between 0.0 and 1.0", score)
            ));
        }
        Ok(Self(score))
    }
    
    /// Get the confidence score.
    pub fn value(self) -> f32 {
        self.0
    }
    
    /// Check if this is a high confidence score (> 0.8).
    pub fn is_high_confidence(self) -> bool {
        self.0 > 0.8
    }
    
    /// Check if this is a low confidence score (< 0.3).
    pub fn is_low_confidence(self) -> bool {
        self.0 < 0.3
    }
}

impl std::fmt::Display for ConfidenceScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}%", self.0 * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sample_rate_validation() {
        assert!(SampleRate::new(0).is_err());
        assert!(SampleRate::new(200000).is_err());
        assert!(SampleRate::new(16000).is_ok());
    }
    
    #[test]
    fn test_audio_buffer_creation() {
        let samples = vec![0.1, 0.2, 0.3];
        let sample_rate = SampleRate::STANDARD_16KHZ;
        let buffer = AudioBuffer::new(samples, sample_rate).unwrap();
        
        assert_eq!(buffer.len().count(), 3);
        assert_eq!(buffer.sample_rate(), sample_rate);
    }
    
    #[test]
    fn test_confidence_score_validation() {
        assert!(ConfidenceScore::new(-0.1).is_err());
        assert!(ConfidenceScore::new(1.1).is_err());
        assert!(ConfidenceScore::new(f32::NAN).is_err());
        assert!(ConfidenceScore::new(0.5).is_ok());
    }
    
    #[test]
    fn test_audio_samples_duration() {
        let samples = AudioSamples::new(16000);
        let sample_rate = SampleRate::STANDARD_16KHZ;
        let duration = samples.duration(sample_rate);
        
        assert_eq!(duration, Duration::from_secs(1));
    }
}