//! Zero-copy tensor operations for high-performance ASR processing.
//!
//! This module provides zero-allocation alternatives to common tensor operations
//! used in the RNN-T decoder, eliminating memory allocation overhead in hot paths.

/// Zero-copy tensor view for efficient matrix operations.
#[derive(Debug)]
pub struct TensorView<'a> {
    data: &'a [f32],
    shape: &'a [usize],
    strides: Vec<usize>,
}

impl<'a> TensorView<'a> {
    /// Create a new tensor view from raw data and shape.
    ///
    /// # Arguments
    /// * `data` - Raw tensor data (borrowed, zero-copy)
    /// * `shape` - Tensor dimensions [features, time_steps, ...]
    ///
    /// # Returns
    /// A zero-copy view into the tensor data
    pub fn new(data: &'a [f32], shape: &'a [usize]) -> Self {
        // Calculate strides for row-major layout
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Extract a frame at a specific time step without allocation.
    ///
    /// For encoder output with shape [1, features, time_steps], this extracts
    /// the feature vector for a single time step.
    ///
    /// # Arguments
    /// * `time_step` - The time step to extract
    /// * `output` - Pre-allocated buffer to write the frame into
    ///
    /// # Returns
    /// Number of features copied
    pub fn extract_frame_into(&self, time_step: usize, output: &mut [f32]) -> usize {
        if self.shape.len() != 3 {
            return 0; // Only works for 3D tensors
        }

        let [_batch, features, time_steps] = [self.shape[0], self.shape[1], self.shape[2]];

        if time_step >= time_steps || output.len() < features {
            return 0;
        }

        // Zero-copy extraction: copy feature vector for this time step
        for feature_idx in 0..features {
            let tensor_idx = feature_idx * time_steps + time_step;
            if tensor_idx < self.data.len() {
                output[feature_idx] = self.data[tensor_idx];
            }
        }

        features
    }

    /// Get the raw data slice (zero-copy).
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }
}

/// Zero-copy mutable tensor view for in-place operations.
pub struct TensorViewMut<'a> {
    data: &'a mut [f32],
    shape: &'a [usize],
    strides: Vec<usize>,
}

impl<'a> TensorViewMut<'a> {
    /// Create a new mutable tensor view.
    pub fn new(data: &'a mut [f32], shape: &'a [usize]) -> Self {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        Self {
            data,
            shape,
            strides,
        }
    }

    /// Fill a tensor slice with values in-place.
    pub fn fill_slice(&mut self, start_idx: usize, values: &[f32]) {
        let end_idx = (start_idx + values.len()).min(self.data.len());
        let copy_len = end_idx - start_idx;

        if copy_len > 0 {
            self.data[start_idx..end_idx].copy_from_slice(&values[..copy_len]);
        }
    }
}

/// Pre-allocated workspace for zero-copy decoder operations.
pub struct DecoderWorkspace {
    /// Pre-allocated encoder frame buffer (1024 features)
    pub encoder_frame: Vec<f32>,

    /// Pre-allocated targets buffer (up to 200 tokens)
    pub targets_buffer: Vec<i32>,

    /// Pre-allocated logits buffer (vocabulary size)
    pub logits_buffer: Vec<f32>,

    /// Pre-allocated temporary computation buffer
    pub temp_buffer: Vec<f32>,
}

impl DecoderWorkspace {
    /// Create a new decoder workspace with pre-allocated buffers.
    pub fn new() -> Self {
        Self {
            encoder_frame: Vec::with_capacity(1024), // Standard feature dimension
            targets_buffer: Vec::with_capacity(200), // Max tokens per sequence
            logits_buffer: Vec::with_capacity(1030), // Vocabulary size
            temp_buffer: Vec::with_capacity(4096),   // General purpose buffer
        }
    }

    /// Prepare encoder frame buffer for a specific time step.
    /// Returns a mutable slice ready for zero-copy frame extraction.
    pub fn prepare_encoder_frame(&mut self, feature_dim: usize) -> &mut [f32] {
        self.encoder_frame.clear();
        self.encoder_frame.resize(feature_dim, 0.0);
        &mut self.encoder_frame
    }

    /// Prepare targets buffer with blank token and existing tokens.
    /// Returns a slice ready for decoder input.
    pub fn prepare_targets(&mut self, blank_token: i32, tokens: &[i32]) -> &[i32] {
        self.targets_buffer.clear();
        self.targets_buffer.push(blank_token);
        self.targets_buffer.extend_from_slice(tokens);
        &self.targets_buffer
    }

    /// Get mutable logits buffer for decoder output.
    pub fn logits_buffer_mut(&mut self, vocab_size: usize) -> &mut [f32] {
        self.logits_buffer.clear();
        self.logits_buffer.resize(vocab_size, 0.0);
        &mut self.logits_buffer
    }

    /// Reset all buffers for reuse (keeps allocated capacity).
    pub fn reset(&mut self) {
        self.encoder_frame.clear();
        self.targets_buffer.clear();
        self.logits_buffer.clear();
        self.temp_buffer.clear();
    }
}

impl Default for DecoderWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy argmax implementation for logits processing.
///
/// # Arguments
/// * `logits` - The logits array (borrowed, zero-copy)
///
/// # Returns
/// (index, value) of the maximum element
pub fn argmax_zero_copy(logits: &[f32]) -> (usize, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    let mut max_idx = 0;
    let mut max_val = logits[0];

    // Unrolled loop for better performance
    let chunks = logits.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut base_idx = 0;
    for chunk in chunks {
        if chunk[0] > max_val {
            max_val = chunk[0];
            max_idx = base_idx;
        }
        if chunk[1] > max_val {
            max_val = chunk[1];
            max_idx = base_idx + 1;
        }
        if chunk[2] > max_val {
            max_val = chunk[2];
            max_idx = base_idx + 2;
        }
        if chunk[3] > max_val {
            max_val = chunk[3];
            max_idx = base_idx + 3;
        }
        base_idx += 4;
    }

    // Handle remainder
    for (i, &val) in remainder.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = base_idx + i;
        }
    }

    (max_idx, max_val)
}

thread_local! {
    static DECODER_WORKSPACE: std::cell::RefCell<DecoderWorkspace> =
        std::cell::RefCell::new(DecoderWorkspace::new());
}

/// Get access to thread-local decoder workspace.
pub fn with_decoder_workspace<F, R>(f: F) -> R
where
    F: FnOnce(&mut DecoderWorkspace) -> R,
{
    DECODER_WORKSPACE.with(|workspace| {
        let mut workspace = workspace.borrow_mut();
        workspace.reset(); // Ensure clean state
        f(&mut workspace)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_view_frame_extraction() {
        // Create test tensor: 2 features, 3 time steps
        let data = vec![
            1.0, 2.0, 3.0, // Feature 0: [1, 2, 3]
            4.0, 5.0, 6.0, // Feature 1: [4, 5, 6]
        ];
        let shape = vec![1, 2, 3]; // [batch, features, time_steps]

        let tensor = TensorView::new(&data, &shape);
        let mut output = vec![0.0; 2];

        // Extract frame at time step 1
        let features_copied = tensor.extract_frame_into(1, &mut output);

        assert_eq!(features_copied, 2);
        assert_eq!(output, vec![2.0, 5.0]); // [feature0[1], feature1[1]]
    }

    #[test]
    fn test_argmax_zero_copy() {
        let logits = vec![0.1, 0.7, 0.2, 0.9, 0.3];
        let (idx, val) = argmax_zero_copy(&logits);

        assert_eq!(idx, 3);
        assert_eq!(val, 0.9);
    }

    #[test]
    fn test_decoder_workspace() {
        let mut workspace = DecoderWorkspace::new();

        // Test encoder frame preparation
        let frame = workspace.prepare_encoder_frame(1024);
        assert_eq!(frame.len(), 1024);

        // Test targets preparation
        let tokens = vec![1, 2, 3];
        let targets = workspace.prepare_targets(1024, &tokens);
        assert_eq!(targets, &[1024, 1, 2, 3]);

        // Test logits buffer
        let logits = workspace.logits_buffer_mut(1030);
        assert_eq!(logits.len(), 1030);
    }

    #[test]
    fn test_thread_local_workspace() {
        let result = with_decoder_workspace(|workspace| {
            let frame = workspace.prepare_encoder_frame(512);
            frame.len()
        });

        assert_eq!(result, 512);
    }
}
