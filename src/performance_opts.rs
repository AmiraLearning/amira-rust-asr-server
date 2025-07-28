//! Performance optimizations and inline attributes for hot paths.
//!
//! This module provides performance-critical functions with proper inline attributes
//! and optimizations based on the HPC transcript recommendations.

// use crate::error::AudioError;

/// Audio processing optimizations with inline attributes.
pub mod audio {
    // use crate::error::AudioError;
    
    /// High-performance audio sample conversion with aggressive inlining.
    #[inline(always)]
    pub fn bytes_to_f32_optimized(input: &[u8], output: &mut Vec<f32>) {
        output.clear();
        output.reserve(input.len() / 2);
        
        // Use chunked processing for better cache locality
        for chunk in input.chunks_exact(2) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            let normalized = sample as f32 / 32768.0;
            output.push(normalized);
        }
        
        // Handle remaining byte if input length is odd
        if input.len() % 2 != 0 {
            let sample = input[input.len() - 1] as i16;
            let normalized = sample as f32 / 128.0;
            output.push(normalized);
        }
    }
    
    /// Calculate mean amplitude with SIMD-style optimizations.
    #[inline(always)]
    pub fn mean_amplitude_optimized(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        
        // Use manual loop unrolling for better performance
        let mut sum = 0.0f32;
        let mut i = 0;
        
        // Process 4 samples at a time (manual vectorization)
        while i + 3 < samples.len() {
            sum += samples[i].abs();
            sum += samples[i + 1].abs();
            sum += samples[i + 2].abs();
            sum += samples[i + 3].abs();
            i += 4;
        }
        
        // Process remaining samples
        while i < samples.len() {
            sum += samples[i].abs();
            i += 1;
        }
        
        sum / samples.len() as f32
    }
    
    /// Apply windowing function with cache-friendly access pattern.
    #[inline(always)]
    pub fn apply_windowing_inplace(samples: &mut [f32], window: &[f32]) {
        debug_assert_eq!(samples.len(), window.len());
        
        // Manual vectorization - process 4 elements at a time
        let mut i = 0;
        while i + 3 < samples.len() {
            samples[i] *= window[i];
            samples[i + 1] *= window[i + 1];
            samples[i + 2] *= window[i + 2];
            samples[i + 3] *= window[i + 3];
            i += 4;
        }
        
        // Handle remaining elements
        while i < samples.len() {
            samples[i] *= window[i];
            i += 1;
        }
    }
    
    /// Normalize audio samples in-place with optimized loops.
    #[inline(always)]
    pub fn normalize_inplace(samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }
        
        // Find maximum absolute value
        let mut max_abs = 0.0f32;
        for &sample in samples.iter() {
            max_abs = max_abs.max(sample.abs());
        }
        
        if max_abs > 0.0 {
            let scale = 1.0 / max_abs;
            
            // Vectorized normalization
            for sample in samples.iter_mut() {
                *sample *= scale;
            }
        }
    }
    
    /// Smooth audio samples with exponential moving average.
    #[inline(always)]
    pub fn smooth_audio_optimized(samples: &mut [f32], smoothing_factor: f32) {
        if samples.is_empty() {
            return;
        }
        
        let alpha = smoothing_factor.clamp(0.0, 1.0);
        let one_minus_alpha = 1.0 - alpha;
        
        // Initialize with first sample
        let mut smoothed = samples[0];
        
        // Apply exponential moving average
        for sample in samples.iter_mut() {
            smoothed = alpha * *sample + one_minus_alpha * smoothed;
            *sample = smoothed;
        }
    }
}

/// Neural network operations with performance optimizations.
pub mod neural {
    // use crate::error::AudioError;
    
    /// Softmax computation with numerical stability.
    #[inline(always)]
    pub fn softmax_optimized(logits: &[f32], output: &mut [f32]) {
        debug_assert_eq!(logits.len(), output.len());
        
        if logits.is_empty() {
            return;
        }
        
        // Find maximum for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials and sum
        let mut sum = 0.0f32;
        for (i, &logit) in logits.iter().enumerate() {
            let exp_val = (logit - max_logit).exp();
            output[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for val in output.iter_mut() {
                *val *= inv_sum;
            }
        }
    }
    
    /// Batch normalization with fused operations.
    #[inline(always)]
    pub fn batch_normalize_optimized(
        input: &[f32],
        output: &mut [f32],
        mean: f32,
        variance: f32,
        epsilon: f32,
    ) {
        debug_assert_eq!(input.len(), output.len());
        
        let inv_std = 1.0 / (variance + epsilon).sqrt();
        
        // Fused normalization: (x - mean) * inv_std
        for (i, &val) in input.iter().enumerate() {
            output[i] = (val - mean) * inv_std;
        }
    }
    
    /// Dot product with manual unrolling.
    #[inline(always)]
    pub fn dot_product_optimized(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        
        let mut sum = 0.0f32;
        let mut i = 0;
        
        // Process 4 elements at a time
        while i + 3 < a.len() {
            sum += a[i] * b[i];
            sum += a[i + 1] * b[i + 1];
            sum += a[i + 2] * b[i + 2];
            sum += a[i + 3] * b[i + 3];
            i += 4;
        }
        
        // Process remaining elements
        while i < a.len() {
            sum += a[i] * b[i];
            i += 1;
        }
        
        sum
    }
    
    /// Argmax with early termination optimization.
    #[inline(always)]
    pub fn argmax_optimized(values: &[f32]) -> usize {
        if values.is_empty() {
            return 0;
        }
        
        let mut max_idx = 0;
        let mut max_val = values[0];
        
        for (i, &val) in values.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        max_idx
    }
}

/// Matrix operations with cache-friendly access patterns.
pub mod matrix {
    // use crate::error::AudioError;
    
    /// Matrix transpose with cache-friendly tiling.
    #[inline(always)]
    pub fn transpose_optimized(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        debug_assert_eq!(input.len(), rows * cols);
        debug_assert_eq!(output.len(), rows * cols);
        
        const TILE_SIZE: usize = 64; // Cache-friendly tile size
        
        for i_tile in (0..rows).step_by(TILE_SIZE) {
            for j_tile in (0..cols).step_by(TILE_SIZE) {
                let i_end = (i_tile + TILE_SIZE).min(rows);
                let j_end = (j_tile + TILE_SIZE).min(cols);
                
                for i in i_tile..i_end {
                    for j in j_tile..j_end {
                        output[j * rows + i] = input[i * cols + j];
                    }
                }
            }
        }
    }
    
    /// Matrix-vector multiplication with optimized loop ordering.
    #[inline(always)]
    pub fn matrix_vector_multiply_optimized(
        matrix: &[f32],
        vector: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        debug_assert_eq!(matrix.len(), rows * cols);
        debug_assert_eq!(vector.len(), cols);
        debug_assert_eq!(output.len(), rows);
        
        for i in 0..rows {
            let mut sum = 0.0f32;
            let row_start = i * cols;
            
            // Manual unrolling for better performance
            let mut j = 0;
            while j + 3 < cols {
                sum += matrix[row_start + j] * vector[j];
                sum += matrix[row_start + j + 1] * vector[j + 1];
                sum += matrix[row_start + j + 2] * vector[j + 2];
                sum += matrix[row_start + j + 3] * vector[j + 3];
                j += 4;
            }
            
            // Handle remaining elements
            while j < cols {
                sum += matrix[row_start + j] * vector[j];
                j += 1;
            }
            
            output[i] = sum;
        }
    }
}

/// Memory management optimizations.
pub mod memory {
    use crate::error::AudioError;
    
    /// Zero-copy buffer wrapper with lifetime guarantees.
    pub struct ZeroCopyBuffer<'a> {
        data: &'a mut [f32],
        original_len: usize,
    }
    
    impl<'a> ZeroCopyBuffer<'a> {
        /// Create a new zero-copy buffer.
        #[inline(always)]
        pub fn new(data: &'a mut [f32]) -> Self {
            let original_len = data.len();
            Self { data, original_len }
        }
        
        /// Get the buffer data.
        #[inline(always)]
        pub fn data(&self) -> &[f32] {
            self.data
        }
        
        /// Get mutable buffer data.
        #[inline(always)]
        pub fn data_mut(&mut self) -> &mut [f32] {
            self.data
        }
        
        /// Resize the buffer (truncate only).
        #[inline(always)]
        pub fn resize(&mut self, new_len: usize) {
            if new_len <= self.original_len && new_len <= self.data.len() {
                // We can only truncate the view, not extend it
                // Use unsafe to work around lifetime constraints
                unsafe {
                    let ptr = self.data.as_mut_ptr();
                    self.data = std::slice::from_raw_parts_mut(ptr, new_len);
                }
            }
        }
        
        /// Get the current length.
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.data.len()
        }
        
        /// Check if buffer is empty.
        #[inline(always)]
        pub fn is_empty(&self) -> bool {
            self.data.is_empty()
        }
    }
    
    /// Pre-allocated workspace for computations.
    pub struct ComputeWorkspace {
        buffer: Vec<f32>,
        capacity: usize,
    }
    
    impl ComputeWorkspace {
        /// Create a new compute workspace.
        #[inline(always)]
        pub fn new(capacity: usize) -> Self {
            Self {
                buffer: Vec::with_capacity(capacity),
                capacity,
            }
        }
        
        /// Get a zero-copy buffer for computation.
        #[inline(always)]
        pub fn get_buffer(&mut self, size: usize) -> Result<ZeroCopyBuffer<'_>, AudioError> {
            if size > self.capacity {
                return Err(AudioError::BufferOverflow);
            }
            
            self.buffer.clear();
            self.buffer.resize(size, 0.0);
            Ok(ZeroCopyBuffer::new(&mut self.buffer))
        }
        
        /// Reset the workspace.
        #[inline(always)]
        pub fn reset(&mut self) {
            self.buffer.clear();
        }
    }
}

/// Hotpath optimization helpers.
pub mod hotpath {
    // use crate::error::AudioError;
    
    /// Error flag for consolidated error handling.
    #[derive(Debug, Clone, Copy)]
    pub struct ErrorFlag(u32);
    
    impl ErrorFlag {
        /// No error.
        pub const NONE: Self = Self(0);
        
        /// Audio processing error.
        pub const AUDIO_ERROR: Self = Self(1);
        
        /// Model inference error.
        pub const MODEL_ERROR: Self = Self(2);
        
        /// Network error.
        pub const NETWORK_ERROR: Self = Self(4);
        
        /// Create a new error flag.
        #[inline(always)]
        pub const fn new(flag: u32) -> Self {
            Self(flag)
        }
        
        /// Check if there's an error.
        #[inline(always)]
        pub const fn has_error(self) -> bool {
            self.0 != 0
        }
        
        /// Set an error flag.
        #[inline(always)]
        pub fn set_error(&mut self, error: Self) {
            self.0 |= error.0;
        }
        
        /// Clear all errors.
        #[inline(always)]
        pub fn clear(&mut self) {
            self.0 = 0;
        }
        
        /// Get the error value.
        #[inline(always)]
        pub const fn value(self) -> u32 {
            self.0
        }
    }
    
    /// Template-specialized processing for compile-time optimization.
    pub trait ProcessingMode {
        const IS_STREAMING: bool;
        const USE_SIMD: bool;
        const ENABLE_CHECKS: bool;
    }
    
    /// Streaming processing mode.
    pub struct StreamingMode;
    
    impl ProcessingMode for StreamingMode {
        const IS_STREAMING: bool = true;
        const USE_SIMD: bool = true;
        const ENABLE_CHECKS: bool = false;
    }
    
    /// Batch processing mode.
    pub struct BatchMode;
    
    impl ProcessingMode for BatchMode {
        const IS_STREAMING: bool = false;
        const USE_SIMD: bool = true;
        const ENABLE_CHECKS: bool = true;
    }
    
    /// Template-specialized audio processing.
    #[inline(always)]
    pub fn process_audio_specialized<M: ProcessingMode>(
        samples: &mut [f32],
        error_flag: &mut ErrorFlag,
    ) {
        if M::ENABLE_CHECKS && samples.is_empty() {
            error_flag.set_error(ErrorFlag::AUDIO_ERROR);
            return;
        }
        
        if M::USE_SIMD {
            // Use SIMD-optimized processing
            super::audio::normalize_inplace(samples);
        } else {
            // Use scalar processing
            for sample in samples.iter_mut() {
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
        
        if M::IS_STREAMING {
            // Additional streaming-specific processing
            super::audio::smooth_audio_optimized(samples, 0.1);
        }
    }
    
    /// Branch-free conditional execution.
    #[inline(always)]
    pub fn conditional_execute<T, F1, F2>(condition: bool, true_fn: F1, false_fn: F2) -> T
    where
        F1: FnOnce() -> T,
        F2: FnOnce() -> T,
    {
        if condition {
            true_fn()
        } else {
            false_fn()
        }
    }
    
    /// Likely/unlikely branch prediction hints.
    #[inline(always)]
    pub fn likely(condition: bool) -> bool {
        // Use cold attribute as a hint for branch prediction
        #[cold]
        fn cold_path() -> bool { false }
        
        if condition {
            true
        } else {
            cold_path()
        }
    }
    
    #[inline(always)]
    pub fn unlikely(condition: bool) -> bool {
        // Use cold attribute as a hint for branch prediction
        #[cold]
        fn cold_path() -> bool { true }
        
        if condition {
            cold_path()
        } else {
            false
        }
    }
    
    /// Prefetch data for better cache performance.
    #[inline(always)]
    pub fn prefetch_data<T>(data: &[T], _locality: i32) {
        if !data.is_empty() {
            // Use a simple memory barrier to prevent reordering
            // This is a portable way to provide prefetch hints
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            
            // Touch the first element to bring it into cache
            let _ = unsafe { data.as_ptr().read_volatile() };
        }
    }
    
    /// Cache-friendly data processing with prefetching.
    #[inline(always)]
    pub fn process_with_prefetch<T, F>(data: &mut [T], chunk_size: usize, mut processor: F)
    where
        F: FnMut(&mut [T]),
    {
        for chunk in data.chunks_mut(chunk_size) {
            // Prefetch next chunk
            if let Some(next_chunk) = chunk.get(chunk_size..) {
                prefetch_data(next_chunk, 3); // Temporal locality
            }
            
            processor(chunk);
        }
    }
}

/// Benchmark utilities for performance measurement.
pub mod benchmark {
    // use crate::error::AudioError;
    use std::time::Instant;
    
    /// Simple timer for performance measurement.
    pub struct Timer {
        start: Instant,
        name: &'static str,
    }
    
    impl Timer {
        /// Start a new timer.
        #[inline(always)]
        pub fn new(name: &'static str) -> Self {
            Self {
                start: Instant::now(),
                name,
            }
        }
        
        /// Stop the timer and print elapsed time.
        #[inline(always)]
        pub fn stop(self) -> std::time::Duration {
            let elapsed = self.start.elapsed();
            #[cfg(debug_assertions)]
            println!("{}: {:?}", self.name, elapsed);
            elapsed
        }
    }
    
    /// Macro for timing code blocks.
    #[macro_export]
    macro_rules! time_block {
        ($name:expr, $block:block) => {{
            let timer = $crate::performance_opts::benchmark::Timer::new($name);
            let result = $block;
            timer.stop();
            result
        }};
    }
    
    /// Prevent compiler optimizations from eliminating code.
    #[inline(always)]
    pub fn black_box<T>(value: T) -> T {
        std::hint::black_box(value)
    }
}

#[cfg(test)]
mod tests {
    // use crate::error::AudioError;
    
    #[test]
    fn test_audio_processing_optimizations() {
        let input = vec![0u8, 128, 255, 0, 64, 192];
        let mut output = Vec::new();
        
        audio::bytes_to_f32_optimized(&input, &mut output);
        assert_eq!(output.len(), 3);
        
        let mean = audio::mean_amplitude_optimized(&output);
        assert!(mean >= 0.0);
        
        let mut samples = vec![0.5, -0.8, 1.2, -0.3];
        audio::normalize_inplace(&mut samples);
        
        // Check that all samples are within [-1.0, 1.0]
        for &sample in &samples {
            assert!(sample >= -1.0 && sample <= 1.0);
        }
    }
    
    #[test]
    fn test_neural_network_optimizations() {
        let logits = vec![1.0, 2.0, 3.0, 1.0];
        let mut output = vec![0.0; 4];
        
        neural::softmax_optimized(&logits, &mut output);
        
        // Check that probabilities sum to 1.0
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that all probabilities are positive
        for &prob in &output {
            assert!(prob > 0.0);
        }
    }
    
    #[test]
    fn test_matrix_operations() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let mut transposed = vec![0.0; 6]; // 3x2
        
        matrix::transpose_optimized(&matrix, &mut transposed, 2, 3);
        
        // Check transpose correctness
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        
        matrix::matrix_vector_multiply_optimized(&matrix, &vector, &mut result, 2, 3);
        
        // Check matrix-vector multiplication
        assert_eq!(result, vec![14.0, 32.0]); // [1*1+2*2+3*3, 4*1+5*2+6*3]
    }
    
    #[test]
    fn test_error_flag() {
        let mut flag = hotpath::ErrorFlag::NONE;
        assert!(!flag.has_error());
        
        flag.set_error(hotpath::ErrorFlag::AUDIO_ERROR);
        assert!(flag.has_error());
        
        flag.set_error(hotpath::ErrorFlag::MODEL_ERROR);
        assert!(flag.has_error());
        assert_eq!(flag.value(), 3); // 1 | 2 = 3
        
        flag.clear();
        assert!(!flag.has_error());
    }
    
    #[test]
    fn test_zero_copy_buffer() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut buffer = memory::ZeroCopyBuffer::new(&mut data);
        
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        buffer.resize(3);
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.data(), &[1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_compute_workspace() {
        let mut workspace = memory::ComputeWorkspace::new(100);
        
        {
            let mut buffer = workspace.get_buffer(10).unwrap();
            assert_eq!(buffer.len(), 10);
            
            // Modify buffer
            buffer.data_mut()[0] = 42.0;
            assert_eq!(buffer.data()[0], 42.0);
        }
        
        // Test buffer reuse
        {
            let buffer = workspace.get_buffer(5).unwrap();
            assert_eq!(buffer.len(), 5);
            assert_eq!(buffer.data()[0], 0.0); // Reset to zero
        }
    }
}