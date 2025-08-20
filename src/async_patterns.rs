//! Improved async patterns and structured concurrency.
//!
//! This module provides structured concurrency patterns and improved async utilities
//! for better error handling, resource management, and performance.

use crate::error::AsrError;
use crate::types::AudioBuffer;
// use crate::asr::traits::{ModelBackend, ModelInput, AudioFeatures, EncoderOutput, DecoderOutput};
use futures::future::BoxFuture;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::{timeout, Instant};

/// Result type for async operations.
pub type AsyncResult<T> = Result<T, AsrError>;

// /// Structured concurrency for ASR pipeline processing.
// /// Note: This is currently commented out due to trait object safety issues.
// /// The ModelBackend trait has generic methods which makes it not object-safe.
// // TODO: Reimplement this with concrete types when needed
/*
pub struct StructuredAsrPipeline {
    preprocessor: Arc<dyn ModelBackend<Error = TritonError>>,
    encoder: Arc<dyn ModelBackend<Error = TritonError>>,
    decoder: Arc<dyn ModelBackend<Error = TritonError>>,
    concurrency_limiter: Arc<Semaphore>,
    timeout_duration: Duration,
}

impl StructuredAsrPipeline {
    // Implementation would go here...
}
*/

/// Async task manager for structured concurrency.
pub struct AsyncTaskManager {
    semaphore: Arc<Semaphore>,
    timeout_duration: Duration,
}

impl AsyncTaskManager {
    /// Create a new async task manager.
    pub fn new(max_concurrent_tasks: usize, timeout_duration: Duration) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent_tasks)),
            timeout_duration,
        }
    }

    /// Execute a task with concurrency control and timeout.
    pub async fn execute_task<F, T>(&self, task: F) -> AsyncResult<T>
    where
        F: std::future::Future<Output = Result<T, AsrError>>,
    {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| AsrError::Pipeline("Failed to acquire task permit".to_string()))?;

        timeout(self.timeout_duration, task)
            .await
            .map_err(|_| AsrError::Pipeline("Task timeout".to_string()))?
    }
}

/// Batch processing utilities for improved throughput.
pub struct BatchProcessor<T> {
    batch_size: usize,
    timeout_duration: Duration,
    items: Vec<T>,
}

impl<T> BatchProcessor<T> {
    /// Create a new batch processor.
    pub fn new(batch_size: usize, timeout_duration: Duration) -> Self {
        Self {
            batch_size,
            timeout_duration,
            items: Vec::with_capacity(batch_size),
        }
    }

    /// Add an item to the batch.
    pub fn add_item(&mut self, item: T) -> bool {
        self.items.push(item);
        self.items.len() >= self.batch_size
    }

    /// Process the current batch.
    pub async fn process_batch<F, R>(&mut self, processor: F) -> AsyncResult<Vec<R>>
    where
        F: FnOnce(Vec<T>) -> BoxFuture<'static, Result<Vec<R>, AsrError>>,
    {
        if self.items.is_empty() {
            return Ok(Vec::new());
        }

        let items = std::mem::take(&mut self.items);
        timeout(self.timeout_duration, processor(items))
            .await
            .map_err(|_| AsrError::Pipeline("Batch processing timeout".to_string()))?
    }
}

/// Stream processing utilities for real-time audio.
pub struct StreamProcessor {
    window_size: usize,
    overlap: usize,
    buffer: Vec<f32>,
}

impl StreamProcessor {
    /// Create a new stream processor.
    pub fn new(window_size: usize, overlap: usize) -> Self {
        Self {
            window_size,
            overlap,
            buffer: Vec::new(),
        }
    }

    /// Add audio data to the stream.
    pub fn add_audio(&mut self, audio: &[f32]) -> Vec<Vec<f32>> {
        self.buffer.extend_from_slice(audio);

        let mut windows = Vec::new();
        let step = self.window_size - self.overlap;

        while self.buffer.len() >= self.window_size {
            windows.push(self.buffer[..self.window_size].to_vec());
            self.buffer.drain(..step);
        }

        windows
    }

    /// Get the current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Enhanced error recovery utilities.
pub struct ErrorRecoveryManager {
    max_retries: usize,
    base_delay: Duration,
    max_delay: Duration,
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager.
    pub fn new(max_retries: usize, base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
        }
    }

    /// Execute an operation with exponential backoff retry.
    pub async fn retry_with_backoff<F, T, E>(&self, operation: F) -> Result<T, E>
    where
        F: Fn() -> BoxFuture<'static, Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut attempts = 0;
        let mut delay = self.base_delay;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    if attempts > self.max_retries {
                        return Err(e);
                    }

                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(delay * 2, self.max_delay);
                }
            }
        }
    }
}

/// Performance monitoring utilities.
pub struct PerformanceMonitor {
    request_count: AtomicUsize,
    error_count: AtomicUsize,
    total_processing_time: std::sync::Mutex<Duration>,
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    pub fn new() -> Self {
        Self {
            request_count: AtomicUsize::new(0),
            error_count: AtomicUsize::new(0),
            total_processing_time: std::sync::Mutex::new(Duration::ZERO),
        }
    }

    /// Record a request.
    pub fn record_request(&self, processing_time: Duration) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut total_time) = self.total_processing_time.lock() {
            *total_time += processing_time;
        }
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.error_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> (usize, usize, Duration) {
        let requests = self
            .request_count
            .load(std::sync::atomic::Ordering::Relaxed);
        let errors = self.error_count.load(std::sync::atomic::Ordering::Relaxed);
        let total_time = self
            .total_processing_time
            .lock()
            .map(|time| *time)
            .unwrap_or(Duration::ZERO);

        (requests, errors, total_time)
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the stream processor.
#[derive(Debug)]
pub struct ProcessorStats {
    items_received: std::sync::atomic::AtomicUsize,
    items_processed: std::sync::atomic::AtomicUsize,
    batches_processed: std::sync::atomic::AtomicUsize,
    start_time: Instant,
}

impl Clone for ProcessorStats {
    fn clone(&self) -> Self {
        Self {
            items_received: AtomicUsize::new(
                self.items_received
                    .load(std::sync::atomic::Ordering::SeqCst),
            ),
            items_processed: AtomicUsize::new(
                self.items_processed
                    .load(std::sync::atomic::Ordering::SeqCst),
            ),
            batches_processed: AtomicUsize::new(
                self.batches_processed
                    .load(std::sync::atomic::Ordering::SeqCst),
            ),
            start_time: self.start_time,
        }
    }
}

impl ProcessorStats {
    /// Create new processor statistics.
    pub fn new() -> Self {
        Self {
            items_received: AtomicUsize::new(0),
            items_processed: AtomicUsize::new(0),
            batches_processed: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record an item received.
    pub fn record_item_received(&self) {
        self.items_received
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record an item processed.
    pub fn record_item_processed(&self) {
        self.items_processed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a batch processed.
    pub fn record_batch_processed(&self) {
        self.batches_processed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> (usize, usize, usize, Duration) {
        (
            self.items_received
                .load(std::sync::atomic::Ordering::Relaxed),
            self.items_processed
                .load(std::sync::atomic::Ordering::Relaxed),
            self.batches_processed
                .load(std::sync::atomic::Ordering::Relaxed),
            self.start_time.elapsed(),
        )
    }
}

impl Default for ProcessorStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced concurrency patterns for high-performance ASR.
pub struct ConcurrencyManager {
    #[allow(dead_code)]
    max_concurrent_requests: usize,
    semaphore: Arc<Semaphore>,
    stats: Arc<ProcessorStats>,
}

impl ConcurrencyManager {
    /// Create a new concurrency manager.
    pub fn new(max_concurrent_requests: usize) -> Self {
        Self {
            max_concurrent_requests,
            semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            stats: Arc::new(ProcessorStats::new()),
        }
    }

    /// Execute a task with concurrency control.
    pub async fn execute_with_concurrency<F, T>(&self, task: F) -> AsyncResult<T>
    where
        F: std::future::Future<Output = AsyncResult<T>>,
    {
        let _permit =
            self.semaphore.acquire().await.map_err(|_| {
                AsrError::Pipeline("Failed to acquire concurrency permit".to_string())
            })?;

        self.stats.record_item_received();

        let result = task.await;

        if let Ok(_) = &result {
            self.stats.record_item_processed()
        }

        result
    }

    /// Get current statistics.
    pub fn get_stats(&self) -> ProcessorStats {
        self.stats.as_ref().clone()
    }
}

/// Resource pool for managing expensive objects.
pub struct ResourcePool<T> {
    pool: std::sync::Mutex<Vec<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T> ResourcePool<T> {
    /// Create a new resource pool.
    pub fn new<F>(factory: F, max_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        Self {
            pool: std::sync::Mutex::new(Vec::new()),
            factory: Box::new(factory),
            max_size,
        }
    }

    /// Get a resource from the pool.
    pub fn get(&self) -> T {
        if let Ok(mut pool) = self.pool.lock() {
            if let Some(resource) = pool.pop() {
                return resource;
            }
        }

        (self.factory)()
    }

    /// Return a resource to the pool.
    pub fn return_resource(&self, resource: T) {
        if let Ok(mut pool) = self.pool.lock() {
            if pool.len() < self.max_size {
                pool.push(resource);
            }
        }
    }
}

/// High-performance async channel for audio streaming.
pub struct AudioStreamChannel {
    sender: tokio::sync::mpsc::Sender<AudioBuffer>,
    receiver: tokio::sync::mpsc::Receiver<AudioBuffer>,
}

impl AudioStreamChannel {
    /// Create a new audio stream channel.
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(buffer_size);
        Self { sender, receiver }
    }

    /// Send audio data.
    pub async fn send(
        &self,
        audio: AudioBuffer,
    ) -> Result<(), tokio::sync::mpsc::error::SendError<AudioBuffer>> {
        self.sender.send(audio).await
    }

    /// Receive audio data.
    pub async fn recv(&mut self) -> Option<AudioBuffer> {
        self.receiver.recv().await
    }
}
