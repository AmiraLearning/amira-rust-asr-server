//! Service metrics tracking.
//!
//! This module provides metrics tracking for the ASR service.

use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

/// Tracks service metrics like request counts and active streams.
#[derive(Debug, Clone)]
pub struct ServiceMetrics {
    /// Total number of requests processed
    total_requests: Arc<Mutex<u64>>,

    /// Number of currently active WebSocket streams
    active_streams: Arc<Mutex<u32>>,

    /// Number of currently active batch requests
    active_batches: Arc<Mutex<u32>>,

    /// Maximum number of concurrent streams observed
    max_concurrent_streams: Arc<Mutex<u32>>,

    /// Maximum number of concurrent batches observed
    max_concurrent_batches: Arc<Mutex<u32>>,

    /// Number of rejected requests due to capacity limits
    rejected_requests: Arc<Mutex<u64>>,

    /// Number of errors encountered
    errors: Arc<Mutex<u64>>,

    /// Server start time
    start_time: Instant,
}

impl ServiceMetrics {
    /// Create a new metrics tracker.
    pub fn new() -> Self {
        Self {
            total_requests: Arc::new(Mutex::new(0)),
            active_streams: Arc::new(Mutex::new(0)),
            active_batches: Arc::new(Mutex::new(0)),
            max_concurrent_streams: Arc::new(Mutex::new(0)),
            max_concurrent_batches: Arc::new(Mutex::new(0)),
            rejected_requests: Arc::new(Mutex::new(0)),
            errors: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Increment the stream count.
    pub fn increment_stream(&self) {
        let mut active = self.active_streams.lock();
        let mut total = self.total_requests.lock();
        *active += 1;
        *total += 1;

        let mut max = self.max_concurrent_streams.lock();
        if *active > *max {
            *max = *active;
        }
    }

    /// Decrement the stream count.
    pub fn decrement_stream(&self) {
        let mut active = self.active_streams.lock();
        *active = active.saturating_sub(1);
    }

    /// Increment the batch count.
    pub fn increment_batch(&self) {
        let mut active = self.active_batches.lock();
        let mut total = self.total_requests.lock();
        *active += 1;
        *total += 1;

        let mut max = self.max_concurrent_batches.lock();
        if *active > *max {
            *max = *active;
        }
    }

    /// Decrement the batch count.
    pub fn decrement_batch(&self) {
        let mut active = self.active_batches.lock();
        *active = active.saturating_sub(1);
    }

    /// Record a rejected request.
    pub fn record_rejection(&self) {
        *self.rejected_requests.lock() += 1;
    }

    /// Record an error.
    pub fn record_error(&self) {
        *self.errors.lock() += 1;
    }

    /// Reset the active batch count.
    /// This is useful for clearing zombie requests after server errors.
    pub fn reset_batch_count(&self) {
        let mut active = self.active_batches.lock();
        *active = 0;
    }

    /// Get all metrics as a serde_json::Value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "uptime_seconds": self.start_time.elapsed().as_secs(),
            "total_requests": *self.total_requests.lock(),
            "active_streams": *self.active_streams.lock(),
            "active_batches": *self.active_batches.lock(),
            "max_concurrent_streams": *self.max_concurrent_streams.lock(),
            "max_concurrent_batches": *self.max_concurrent_batches.lock(),
            "rejected_requests": *self.rejected_requests.lock(),
            "errors": *self.errors.lock(),
        })
    }
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self::new()
    }
}
