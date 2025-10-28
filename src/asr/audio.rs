//! Audio processing utilities.
//!
//! This module contains utilities for working with audio data, including
//! conversion between formats and a high-performance ring buffer for
//! streaming audio.

use crate::asr::types::{SeqSlice, W2V_SAMPLE_RATE};
use crate::error::{AppError, AsrError, AudioError, Result};
use tracing::debug;

/// Exponential moving average weight for existing value (70%)
const EMA_WEIGHT_EXISTING: f32 = 0.7;

/// Exponential moving average weight for new value (30%)
const EMA_WEIGHT_NEW: f32 = 0.3;

/// Convert raw audio bytes (16-bit PCM) to floating point samples.
///
/// # Arguments
/// * `audio_bytes` - Raw audio bytes in 16-bit PCM format
///
/// # Returns
/// Vector of f32 samples normalized to [-1.0, 1.0]
pub fn bytes_to_f32_samples(audio_bytes: &[u8]) -> Vec<f32> {
    audio_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Convert raw audio bytes (16-bit PCM) to floating point samples into an existing buffer.
/// This version avoids allocations by reusing the provided buffer and uses SIMD optimizations.
///
/// # Arguments
/// * `audio_bytes` - Raw audio bytes in 16-bit PCM format
/// * `output` - Output buffer to write samples into (will be cleared first)
pub fn bytes_to_f32_samples_into(audio_bytes: &[u8], output: &mut Vec<f32>) {
    // Use optimized version from performance_opts
    crate::performance_opts::audio::bytes_to_f32_optimized(audio_bytes, output);
}

/// Get the length of audio in seconds.
///
/// # Arguments
/// * `audio` - Audio samples
///
/// # Returns
/// Length in seconds
pub fn audio_len(audio: &[f32]) -> f32 {
    audio.len() as f32 / W2V_SAMPLE_RATE as f32
}

/// Calculate the mean amplitude of audio using SIMD optimizations.
///
/// # Arguments
/// * `audio` - Audio samples
///
/// # Returns
/// Mean amplitude value
pub fn calculate_mean_amplitude(audio: &[f32]) -> f32 {
    // Use optimized version from performance_opts
    crate::performance_opts::audio::mean_amplitude_optimized(audio)
}

/// Generate sequence windows with overlap.
///
/// # Arguments
/// * `total_len` - Total length of the sequence
/// * `window_size` - Size of each window
/// * `leading_context` - Size of leading context
/// * `trailing_context` - Size of trailing context
///
/// # Returns
/// Iterator of tuples containing (source_slice, target_slice, overlap_ratio)
pub fn window_sequence(
    total_len: usize,
    window_size: usize,
    leading_context: usize,
    trailing_context: usize,
) -> impl Iterator<Item = (SeqSlice, SeqSlice, f32)> {
    WindowSequenceIterator {
        total_len,
        window_size,
        leading_context,
        trailing_context,
        consumed: 0,
    }
}

/// Iterator for generating sequence windows with overlap.
struct WindowSequenceIterator {
    total_len: usize,
    window_size: usize,
    leading_context: usize,
    trailing_context: usize,
    consumed: usize,
}

impl Iterator for WindowSequenceIterator {
    type Item = (SeqSlice, SeqSlice, f32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed >= self.total_len {
            return None;
        }

        let start = self.consumed;
        let end = std::cmp::min(self.total_len, self.consumed + self.window_size);
        let offset = std::cmp::min(self.leading_context, self.consumed);
        let mut total_overlap = self.trailing_context + self.leading_context;

        if end < self.total_len {
            self.consumed = end - self.leading_context - self.trailing_context;
        } else {
            self.consumed = end;
            if end - start < self.window_size {
                let new_start = std::cmp::max(0, end - self.window_size);
                total_overlap += start - new_start;
            }
        }

        let source_slice = SeqSlice::new(start, end);
        let target_slice = SeqSlice::new(start + offset, end);
        let overlap_ratio = total_overlap as f32 / self.window_size as f32;

        debug!(
            "Audio window [{:.2}, {:.2}] seconds, overlap ratio: {:.2}",
            start as f32 / W2V_SAMPLE_RATE as f32,
            end as f32 / W2V_SAMPLE_RATE as f32,
            overlap_ratio
        );

        Some((source_slice, target_slice, overlap_ratio))
    }
}

/// Enhanced audio buffer for streaming with overlap support.
///
/// This structure allows efficient storage and management of audio data
/// with support for overlapping windows and audio energy analysis.
#[derive(Debug)]
pub struct OverlappingAudioBuffer {
    /// The underlying audio samples
    buffer: Vec<f32>,

    /// Current buffer length
    length: usize,

    /// Maximum capacity
    capacity: usize,

    /// Leading context size in samples
    leading_context: usize,

    /// Trailing context size in samples
    trailing_context: usize,

    /// Main chunk size in samples
    chunk_size: usize,

    /// Mean amplitude of processed audio
    mean_amplitude: f32,
}

impl OverlappingAudioBuffer {
    /// Create a new overlapping audio buffer.
    ///
    /// # Arguments
    /// * `capacity` - Maximum buffer capacity in samples
    /// * `chunk_size` - Size of processing chunks in seconds
    /// * `leading_context` - Leading context size in seconds
    /// * `trailing_context` - Trailing context size in seconds
    pub fn new(
        capacity: usize,
        chunk_size: f32,
        leading_context: f32,
        trailing_context: f32,
    ) -> Self {
        let chunk_samples = (chunk_size * W2V_SAMPLE_RATE as f32) as usize;
        let leading_samples = (leading_context * W2V_SAMPLE_RATE as f32) as usize;
        let trailing_samples = (trailing_context * W2V_SAMPLE_RATE as f32) as usize;

        Self {
            buffer: vec![0.0; capacity],
            length: 0,
            capacity,
            leading_context: leading_samples,
            trailing_context: trailing_samples,
            chunk_size: chunk_samples,
            mean_amplitude: 0.0,
        }
    }

    /// Add audio samples to the buffer.
    ///
    /// If the buffer would overflow, the oldest samples are discarded
    /// while preserving context.
    ///
    /// # Arguments
    /// * `samples` - Audio samples to add
    pub fn add_samples(&mut self, samples: &[f32]) {
        let samples_len = samples.len();

        // If samples exceed available space, we need to shift the buffer
        if self.length + samples_len > self.capacity {
            // Keep enough context from the old buffer
            let context_to_keep = std::cmp::min(self.leading_context, self.length);
            let start_idx = self.length - context_to_keep;

            // Shift buffer to make room
            if context_to_keep > 0 {
                self.buffer.copy_within(start_idx..self.length, 0);
            }

            self.length = context_to_keep;
        }

        // Add new samples
        let start_idx = self.length;
        let end_idx = start_idx + samples_len;

        if end_idx <= self.capacity {
            self.buffer[start_idx..end_idx].copy_from_slice(samples);
            self.length = end_idx;

            // Update mean amplitude
            if self.mean_amplitude == 0.0 {
                self.mean_amplitude = calculate_mean_amplitude(samples);
            } else {
                // Exponential moving average
                let new_amplitude = calculate_mean_amplitude(samples);
                self.mean_amplitude =
                    EMA_WEIGHT_EXISTING * self.mean_amplitude + EMA_WEIGHT_NEW * new_amplitude;
            }
        } else {
            debug!("Buffer capacity exceeded, truncating samples");
            let available = self.capacity - start_idx;
            self.buffer[start_idx..self.capacity].copy_from_slice(&samples[..available]);
            self.length = self.capacity;
        }
    }

    /// Get the current audio window for processing.
    ///
    /// Returns a slice of the buffer containing the current window.
    pub fn get_window(&self) -> &[f32] {
        &self.buffer[..self.length]
    }

    /// Get the mean amplitude of the audio.
    pub fn mean_amplitude(&self) -> f32 {
        self.mean_amplitude
    }

    /// Generate overlapping windows from the buffer.
    ///
    /// Each call returns a window of audio with appropriate context,
    /// and the overlap ratio for weaving transcripts.
    pub fn overlapping_windows(&self) -> impl Iterator<Item = (SeqSlice, SeqSlice, f32)> + '_ {
        window_sequence(
            self.length,
            self.chunk_size + self.leading_context + self.trailing_context,
            self.leading_context,
            self.trailing_context,
        )
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.length = 0;
        self.mean_amplitude = 0.0;
    }

    /// Get the audio from a specific slice.
    ///
    /// # Arguments
    /// * `slice` - The sequence slice to extract
    ///
    /// # Returns
    /// The audio samples for the slice
    pub fn get_slice(&self, slice: &SeqSlice) -> &[f32] {
        if slice.end <= self.length {
            &self.buffer[slice.start..slice.end]
        } else {
            debug!(
                "Slice end {} exceeds buffer length {}",
                slice.end, self.length
            );
            &self.buffer[slice.start..self.length]
        }
    }
}

/// A truly lock-free ring buffer for audio streaming.
///
/// This structure allows efficient storage and retrieval of audio data
/// in a streaming context, with support for circular read/write operations.
/// Uses atomic operations for thread-safe access without locks.
pub struct AudioRingBuffer {
    /// The underlying byte buffer (boxed for stable memory address)
    buffer: Box<[u8]>,

    /// Current write position (atomic for lock-free access)
    write_pos: std::sync::atomic::AtomicUsize,

    /// Current read position (atomic for lock-free access)
    read_pos: std::sync::atomic::AtomicUsize,

    /// Total capacity of the buffer
    capacity: usize,
}

impl std::fmt::Debug for AudioRingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioRingBuffer")
            .field("capacity", &self.capacity)
            .field(
                "write_pos",
                &self.write_pos.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field(
                "read_pos",
                &self.read_pos.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("available_read", &self.available_read())
            .field("available_write", &self.available_write())
            .finish()
    }
}

impl AudioRingBuffer {
    /// Create a new audio ring buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity].into_boxed_slice(),
            write_pos: std::sync::atomic::AtomicUsize::new(0),
            read_pos: std::sync::atomic::AtomicUsize::new(0),
            capacity,
        }
    }

    /// Write data to the buffer (lock-free).
    ///
    /// # Arguments
    /// * `data` - Data to write to the buffer
    ///
    /// # Returns
    /// Ok(()) if successful, or an error if the buffer would overflow
    pub fn write(&self, data: &[u8]) -> Result<()> {
        use std::sync::atomic::Ordering;

        let len = data.len();
        if len > self.available_write() {
            return Err(AppError::Asr(AsrError::AudioProcessing(
                AudioError::InvalidFormat("Buffer overflow".to_string()),
            )));
        }

        let current_write_pos = self.write_pos.load(Ordering::Acquire);
        let end_pos = (current_write_pos + len) % self.capacity;

        // Perform the write operation
        // SAFETY:
        // - We cast self.buffer.as_ptr() to *mut u8, which is safe because:
        //   1. The buffer is Box<[u8]> with a stable memory address
        //   2. We're the exclusive writer (enforced by &self and atomic positions)
        //   3. The lifetime of buffer_ptr doesn't exceed the buffer's lifetime
        // - ptr::copy_nonoverlapping is safe because:
        //   1. data.as_ptr() and buffer_ptr.add(pos) are valid, non-null, properly aligned
        //   2. Ranges don't overlap (source is external data slice, dest is internal buffer)
        //   3. current_write_pos < capacity (enforced by modulo)
        //   4. We validated len <= available_write() so no overflow
        //   5. For wraparound: first_part and (len - first_part) sum to len, both in bounds
        unsafe {
            let buffer_ptr = self.buffer.as_ptr() as *mut u8;
            if end_pos > current_write_pos {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer_ptr.add(current_write_pos),
                    len,
                );
            } else {
                let first_part = self.capacity - current_write_pos;
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer_ptr.add(current_write_pos),
                    first_part,
                );
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(first_part),
                    buffer_ptr,
                    len - first_part,
                );
            }
        }

        // Atomically update write position
        self.write_pos.store(end_pos, Ordering::Release);
        Ok(())
    }

    /// Read data into an existing buffer (zero-copy, lock-free).
    ///
    /// # Arguments
    /// * `len` - Number of bytes to read
    /// * `output` - Buffer to read data into
    ///
    /// # Returns
    /// Number of bytes actually read, or None if not enough data is available
    pub fn read_into(&self, len: usize, output: &mut [u8]) -> Option<usize> {
        use std::sync::atomic::Ordering;

        if len > output.len() || len > self.available_read() {
            return None;
        }

        let current_read_pos = self.read_pos.load(Ordering::Acquire);
        let end_pos = (current_read_pos + len) % self.capacity;

        // Perform the read operation
        // SAFETY:
        // - buffer_ptr is valid because self.buffer is a Box<[u8]> with stable memory
        // - ptr::copy_nonoverlapping is safe because:
        //   1. buffer_ptr and output.as_mut_ptr() are valid, non-null, properly aligned
        //   2. Ranges don't overlap (source is internal buffer, dest is external output)
        //   3. current_read_pos < capacity (enforced by modulo)
        //   4. We validated len <= available_read() so data exists and no underflow
        //   5. For wraparound: first_part and (len - first_part) sum to len, both in bounds
        unsafe {
            let buffer_ptr = self.buffer.as_ptr();
            if end_pos > current_read_pos {
                std::ptr::copy_nonoverlapping(
                    buffer_ptr.add(current_read_pos),
                    output.as_mut_ptr(),
                    len,
                );
            } else {
                let first_part = self.capacity - current_read_pos;
                std::ptr::copy_nonoverlapping(
                    buffer_ptr.add(current_read_pos),
                    output.as_mut_ptr(),
                    first_part,
                );
                std::ptr::copy_nonoverlapping(
                    buffer_ptr,
                    output.as_mut_ptr().add(first_part),
                    len - first_part,
                );
            }
        }

        // Atomically update read position
        self.read_pos.store(end_pos, Ordering::Release);
        Some(len)
    }

    /// Get the number of bytes available to read (lock-free).
    pub fn available_read(&self) -> usize {
        use std::sync::atomic::Ordering;

        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        }
    }

    /// Get the number of bytes available to write.
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read() - 1
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    /// Clear the buffer (lock-free).
    pub fn clear(&self) {
        use std::sync::atomic::Ordering;

        self.read_pos.store(0, Ordering::Release);
        self.write_pos.store(0, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test bytes_to_f32_samples conversion
    #[test]
    fn test_bytes_to_f32_samples_basic() {
        // Test silence (zero samples)
        let silence = vec![0u8, 0u8, 0u8, 0u8];
        let samples = bytes_to_f32_samples(&silence);
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 0.0);
    }

    #[test]
    fn test_bytes_to_f32_samples_max_min_values() {
        // Test maximum positive value (32767)
        let max_bytes = vec![0xFF, 0x7F]; // 32767 in little-endian
        let max_samples = bytes_to_f32_samples(&max_bytes);
        assert_eq!(max_samples.len(), 1);
        assert!((max_samples[0] - 32767.0 / 32768.0).abs() < 0.0001);

        // Test minimum negative value (-32768)
        let min_bytes = vec![0x00, 0x80]; // -32768 in little-endian
        let min_samples = bytes_to_f32_samples(&min_bytes);
        assert_eq!(min_samples.len(), 1);
        assert_eq!(min_samples[0], -1.0);
    }

    #[test]
    fn test_bytes_to_f32_samples_empty() {
        let empty: Vec<u8> = vec![];
        let samples = bytes_to_f32_samples(&empty);
        assert_eq!(samples.len(), 0);
    }

    #[test]
    fn test_bytes_to_f32_samples_known_values() {
        // Test known conversion: 16384 = 0x4000 (little-endian: 0x00, 0x40)
        let bytes = vec![0x00, 0x40];
        let samples = bytes_to_f32_samples(&bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 0.0001); // 16384 / 32768 = 0.5
    }

    // Test AudioRingBuffer basic operations
    #[test]
    fn test_ring_buffer_basic_write_read() {
        let buffer = AudioRingBuffer::new(100);

        // Initially empty
        assert_eq!(buffer.available_read(), 0);
        assert!(buffer.is_empty());

        // Write some data
        let data = vec![1u8, 2, 3, 4, 5];
        buffer.write(&data).unwrap();

        assert_eq!(buffer.available_read(), 5);
        assert!(!buffer.is_empty());

        // Read data back
        let mut output = vec![0u8; 5];
        let read_count = buffer.read_into(5, &mut output).unwrap();
        assert_eq!(read_count, 5);
        assert_eq!(output, data);

        // Now empty again
        assert_eq!(buffer.available_read(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let buffer = AudioRingBuffer::new(10);

        // Write 8 bytes
        let data1 = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        buffer.write(&data1).unwrap();

        // Read 5 bytes
        let mut output1 = vec![0u8; 5];
        buffer.read_into(5, &mut output1).unwrap();
        assert_eq!(output1, vec![1, 2, 3, 4, 5]);

        // Write 6 more bytes (this will wrap around)
        let data2 = vec![9u8, 10, 11, 12, 13, 14];
        buffer.write(&data2).unwrap();

        // Read all remaining data
        let mut output2 = vec![0u8; 9];
        let read_count = buffer.read_into(9, &mut output2).unwrap();
        assert_eq!(read_count, 9);
        assert_eq!(output2, vec![6, 7, 8, 9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let buffer = AudioRingBuffer::new(10);

        // Try to write more than capacity - 1
        let data = vec![1u8; 11];
        let result = buffer.write(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_ring_buffer_read_insufficient_data() {
        let buffer = AudioRingBuffer::new(100);

        // Write 5 bytes
        buffer.write(&vec![1u8, 2, 3, 4, 5]).unwrap();

        // Try to read 10 bytes
        let mut output = vec![0u8; 10];
        let result = buffer.read_into(10, &mut output);
        assert!(result.is_none());
    }

    #[test]
    fn test_ring_buffer_clear() {
        let buffer = AudioRingBuffer::new(100);

        // Write data
        buffer.write(&vec![1u8, 2, 3, 4, 5]).unwrap();
        assert_eq!(buffer.available_read(), 5);

        // Clear buffer
        buffer.clear();
        assert_eq!(buffer.available_read(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_ring_buffer_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let buffer = Arc::new(AudioRingBuffer::new(1000));
        let buffer_clone = Arc::clone(&buffer);

        // Writer thread
        let writer = thread::spawn(move || {
            for i in 0..10 {
                let data = vec![i as u8; 10];
                while buffer_clone.write(&data).is_err() {
                    thread::yield_now(); // Wait for space
                }
                thread::sleep(std::time::Duration::from_micros(100));
            }
        });

        // Reader thread
        let buffer_clone2 = Arc::clone(&buffer);
        let reader = thread::spawn(move || {
            let mut total_read = 0;
            let mut output = vec![0u8; 10];

            while total_read < 100 {
                if let Some(count) = buffer_clone2.read_into(10, &mut output) {
                    total_read += count;
                }
                thread::yield_now();
            }

            assert_eq!(total_read, 100);
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }

    // Test OverlappingAudioBuffer
    #[test]
    fn test_overlapping_buffer_basic() {
        let mut buffer = OverlappingAudioBuffer::new(1000, 1.0, 0.1, 0.1);

        assert!(buffer.is_empty());
        assert_eq!(buffer.mean_amplitude(), 0.0);

        // Add samples
        let samples = vec![0.5f32; 100];
        buffer.add_samples(&samples);

        assert!(!buffer.is_empty());
        assert!(buffer.mean_amplitude() > 0.0);
        assert_eq!(buffer.get_window().len(), 100);
    }

    #[test]
    fn test_overlapping_buffer_overflow_handling() {
        let mut buffer = OverlappingAudioBuffer::new(100, 1.0, 0.1, 0.1);

        // Add samples that exceed capacity
        let samples1 = vec![1.0f32; 80];
        buffer.add_samples(&samples1);
        assert_eq!(buffer.get_window().len(), 80);

        // Add more samples - should trigger shift
        let samples2 = vec![2.0f32; 50];
        buffer.add_samples(&samples2);

        // Buffer should keep context and new samples
        assert!(buffer.get_window().len() <= 100);
        assert!(buffer.get_window().len() > 0);
    }

    #[test]
    fn test_overlapping_buffer_mean_amplitude_tracking() {
        let mut buffer = OverlappingAudioBuffer::new(1000, 1.0, 0.1, 0.1);

        // Add samples with amplitude 0.5
        let samples1 = vec![0.5f32; 100];
        buffer.add_samples(&samples1);
        let amp1 = buffer.mean_amplitude();
        assert!(amp1 > 0.4 && amp1 < 0.6);

        // Add samples with amplitude 1.0
        let samples2 = vec![1.0f32; 100];
        buffer.add_samples(&samples2);
        let amp2 = buffer.mean_amplitude();

        // Mean should have increased (exponential moving average)
        assert!(amp2 > amp1);
    }

    #[test]
    fn test_overlapping_buffer_clear() {
        let mut buffer = OverlappingAudioBuffer::new(1000, 1.0, 0.1, 0.1);

        buffer.add_samples(&vec![0.5f32; 100]);
        assert!(!buffer.is_empty());
        assert!(buffer.mean_amplitude() > 0.0);

        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.mean_amplitude(), 0.0);
    }

    #[test]
    fn test_overlapping_buffer_slice_extraction() {
        let mut buffer = OverlappingAudioBuffer::new(1000, 1.0, 0.1, 0.1);

        // Create test pattern
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        buffer.add_samples(&samples);

        // Extract slice
        let slice = SeqSlice::new(10, 20);
        let extracted = buffer.get_slice(&slice);

        assert_eq!(extracted.len(), 10);
        assert!((extracted[0] - 0.10).abs() < 0.01);
        assert!((extracted[9] - 0.19).abs() < 0.01);
    }

    // Test window_sequence
    #[test]
    fn test_window_sequence_basic() {
        let windows: Vec<_> = window_sequence(100, 50, 5, 5).collect();

        // Should generate at least 2 windows
        assert!(windows.len() >= 2);

        // First window should start at 0
        assert_eq!(windows[0].0.start, 0);

        // Check overlap ratios are reasonable
        for (_, _, overlap_ratio) in &windows {
            assert!(*overlap_ratio >= 0.0 && *overlap_ratio <= 1.0);
        }
    }

    #[test]
    fn test_window_sequence_coverage() {
        let total_len = 100;
        let windows: Vec<_> = window_sequence(total_len, 50, 5, 5).collect();

        // Last window should reach the end
        let last_window = windows.last().unwrap();
        assert_eq!(last_window.0.end, total_len);
    }

    // Test audio_len
    #[test]
    fn test_audio_len() {
        // W2V_SAMPLE_RATE is 16000
        let samples = vec![0.0f32; 16000];
        let length = audio_len(&samples);
        assert!((length - 1.0).abs() < 0.001); // Should be 1 second

        let samples2 = vec![0.0f32; 32000];
        let length2 = audio_len(&samples2);
        assert!((length2 - 2.0).abs() < 0.001); // Should be 2 seconds
    }
}
