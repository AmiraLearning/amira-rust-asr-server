//! Service metrics tracking.
//!
//! This module provides metrics tracking for the ASR service.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Tracks service metrics like request counts and active streams.
#[derive(Debug, Clone)]
pub struct ServiceMetrics {
    /// Total number of requests processed
    total_requests: Arc<AtomicU64>,

    /// Number of currently active WebSocket streams
    active_streams: Arc<AtomicU32>,

    /// Number of currently active batch requests
    active_batches: Arc<AtomicU32>,

    /// Maximum number of concurrent streams observed
    max_concurrent_streams: Arc<AtomicU32>,

    /// Maximum number of concurrent batches observed
    max_concurrent_batches: Arc<AtomicU32>,

    /// Number of rejected requests due to capacity limits
    rejected_requests: Arc<AtomicU64>,

    /// Number of errors encountered
    errors: Arc<AtomicU64>,

    /// Server start time
    start_time: Instant,
}

impl ServiceMetrics {
    /// Create a new metrics tracker.
    pub fn new() -> Self {
        Self {
            total_requests: Arc::new(AtomicU64::new(0)),
            active_streams: Arc::new(AtomicU32::new(0)),
            active_batches: Arc::new(AtomicU32::new(0)),
            max_concurrent_streams: Arc::new(AtomicU32::new(0)),
            max_concurrent_batches: Arc::new(AtomicU32::new(0)),
            rejected_requests: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Increment the stream count.
    pub fn increment_stream(&self) {
        let active = self.active_streams.fetch_add(1, Ordering::SeqCst) + 1;
        self.total_requests.fetch_add(1, Ordering::SeqCst);

        // Update max streams if necessary (atomic compare-and-swap loop)
        self.max_concurrent_streams
            .fetch_max(active, Ordering::SeqCst);
    }

    /// Decrement the stream count.
    pub fn decrement_stream(&self) {
        self.active_streams.fetch_sub(1, Ordering::SeqCst);
    }

    /// Increment the batch count.
    pub fn increment_batch(&self) {
        let active = self.active_batches.fetch_add(1, Ordering::SeqCst) + 1;
        self.total_requests.fetch_add(1, Ordering::SeqCst);

        // Update max batches if necessary (atomic compare-and-swap loop)
        self.max_concurrent_batches
            .fetch_max(active, Ordering::SeqCst);
    }

    /// Decrement the batch count.
    pub fn decrement_batch(&self) {
        self.active_batches.fetch_sub(1, Ordering::SeqCst);
    }

    /// Record a rejected request.
    pub fn record_rejection(&self) {
        self.rejected_requests.fetch_add(1, Ordering::SeqCst);
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::SeqCst);
    }

    /// Reset the active batch count.
    /// This is useful for clearing zombie requests after server errors.
    pub fn reset_batch_count(&self) {
        self.active_batches.store(0, Ordering::SeqCst);
    }

    /// Get all metrics as a serde_json::Value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "uptime_seconds": self.start_time.elapsed().as_secs(),
            "total_requests": self.total_requests.load(Ordering::SeqCst),
            "active_streams": self.active_streams.load(Ordering::SeqCst),
            "active_batches": self.active_batches.load(Ordering::SeqCst),
            "max_concurrent_streams": self.max_concurrent_streams.load(Ordering::SeqCst),
            "max_concurrent_batches": self.max_concurrent_batches.load(Ordering::SeqCst),
            "rejected_requests": self.rejected_requests.load(Ordering::SeqCst),
            "errors": self.errors.load(Ordering::SeqCst),
        })
    }
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self::new()
    }
}
