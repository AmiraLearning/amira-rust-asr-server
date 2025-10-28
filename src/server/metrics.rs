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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_metrics_initialization() {
        let metrics = ServiceMetrics::new();
        let json = metrics.to_json();

        assert_eq!(json["total_requests"], 0);
        assert_eq!(json["active_streams"], 0);
        assert_eq!(json["active_batches"], 0);
        assert_eq!(json["max_concurrent_streams"], 0);
        assert_eq!(json["max_concurrent_batches"], 0);
        assert_eq!(json["rejected_requests"], 0);
        assert_eq!(json["errors"], 0);
    }

    #[test]
    fn test_increment_decrement_stream() {
        let metrics = ServiceMetrics::new();

        metrics.increment_stream();
        let json = metrics.to_json();
        assert_eq!(json["active_streams"], 1);
        assert_eq!(json["total_requests"], 1);
        assert_eq!(json["max_concurrent_streams"], 1);

        metrics.decrement_stream();
        let json = metrics.to_json();
        assert_eq!(json["active_streams"], 0);
        assert_eq!(json["total_requests"], 1); // Total requests doesn't decrease
        assert_eq!(json["max_concurrent_streams"], 1); // Max stays at peak
    }

    #[test]
    fn test_increment_decrement_batch() {
        let metrics = ServiceMetrics::new();

        metrics.increment_batch();
        let json = metrics.to_json();
        assert_eq!(json["active_batches"], 1);
        assert_eq!(json["total_requests"], 1);
        assert_eq!(json["max_concurrent_batches"], 1);

        metrics.decrement_batch();
        let json = metrics.to_json();
        assert_eq!(json["active_batches"], 0);
        assert_eq!(json["total_requests"], 1);
        assert_eq!(json["max_concurrent_batches"], 1);
    }

    #[test]
    fn test_max_concurrent_streams_tracking() {
        let metrics = ServiceMetrics::new();

        // Increment to 3 streams
        metrics.increment_stream();
        metrics.increment_stream();
        metrics.increment_stream();

        let json = metrics.to_json();
        assert_eq!(json["active_streams"], 3);
        assert_eq!(json["max_concurrent_streams"], 3);

        // Decrement to 1
        metrics.decrement_stream();
        metrics.decrement_stream();

        let json = metrics.to_json();
        assert_eq!(json["active_streams"], 1);
        assert_eq!(json["max_concurrent_streams"], 3); // Still at peak

        // Increment back to 4 (new peak)
        metrics.increment_stream();
        metrics.increment_stream();
        metrics.increment_stream();

        let json = metrics.to_json();
        assert_eq!(json["active_streams"], 4);
        assert_eq!(json["max_concurrent_streams"], 4); // New peak
    }

    #[test]
    fn test_max_concurrent_batches_tracking() {
        let metrics = ServiceMetrics::new();

        metrics.increment_batch();
        metrics.increment_batch();

        let json = metrics.to_json();
        assert_eq!(json["active_batches"], 2);
        assert_eq!(json["max_concurrent_batches"], 2);

        metrics.decrement_batch();

        let json = metrics.to_json();
        assert_eq!(json["active_batches"], 1);
        assert_eq!(json["max_concurrent_batches"], 2);
    }

    #[test]
    fn test_record_rejection() {
        let metrics = ServiceMetrics::new();

        metrics.record_rejection();
        metrics.record_rejection();
        metrics.record_rejection();

        let json = metrics.to_json();
        assert_eq!(json["rejected_requests"], 3);
    }

    #[test]
    fn test_record_error() {
        let metrics = ServiceMetrics::new();

        metrics.record_error();
        metrics.record_error();

        let json = metrics.to_json();
        assert_eq!(json["errors"], 2);
    }

    #[test]
    fn test_reset_batch_count() {
        let metrics = ServiceMetrics::new();

        metrics.increment_batch();
        metrics.increment_batch();
        metrics.increment_batch();

        let json = metrics.to_json();
        assert_eq!(json["active_batches"], 3);

        metrics.reset_batch_count();

        let json = metrics.to_json();
        assert_eq!(json["active_batches"], 0);
        assert_eq!(json["total_requests"], 3); // Total requests unchanged
    }

    #[test]
    fn test_total_requests_accumulation() {
        let metrics = ServiceMetrics::new();

        metrics.increment_stream();
        metrics.increment_batch();
        metrics.increment_stream();
        metrics.increment_batch();

        let json = metrics.to_json();
        assert_eq!(json["total_requests"], 4);
    }

    #[test]
    fn test_uptime_seconds() {
        let metrics = ServiceMetrics::new();

        // Sleep a bit to ensure uptime is present
        std::thread::sleep(std::time::Duration::from_millis(10));

        let json = metrics.to_json();
        let uptime = json["uptime_seconds"].as_u64();
        assert!(uptime.is_some()); // Should have an uptime value
    }

    #[test]
    fn test_concurrent_stream_increments() {
        let metrics = Arc::new(ServiceMetrics::new());
        let mut handles = vec![];

        // Spawn 10 threads, each incrementing 100 times
        for _ in 0..10 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    metrics_clone.increment_stream();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let json = metrics.to_json();
        // All 1000 increments should be counted
        assert_eq!(json["total_requests"], 1000);
    }

    #[test]
    fn test_concurrent_batch_increments() {
        let metrics = Arc::new(ServiceMetrics::new());
        let mut handles = vec![];

        // Spawn 5 threads, each incrementing 50 times
        for _ in 0..5 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..50 {
                    metrics_clone.increment_batch();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let json = metrics.to_json();
        assert_eq!(json["total_requests"], 250);
    }

    #[test]
    fn test_concurrent_increment_decrement() {
        let metrics = Arc::new(ServiceMetrics::new());
        let mut handles = vec![];

        // Spawn threads that increment
        for _ in 0..5 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    metrics_clone.increment_stream();
                    std::thread::sleep(std::time::Duration::from_micros(1));
                }
            });
            handles.push(handle);
        }

        // Spawn threads that decrement
        for _ in 0..5 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    std::thread::sleep(std::time::Duration::from_micros(1));
                    metrics_clone.decrement_stream();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let json = metrics.to_json();
        // Active streams should be 0 (equal increments and decrements)
        assert_eq!(json["active_streams"], 0);
        // Total requests should be 500 (from increments only)
        assert_eq!(json["total_requests"], 500);
    }

    #[test]
    fn test_default_implementation() {
        let metrics = ServiceMetrics::default();
        let json = metrics.to_json();

        assert_eq!(json["total_requests"], 0);
        assert_eq!(json["active_streams"], 0);
    }

    #[test]
    fn test_clone_implementation() {
        let metrics1 = ServiceMetrics::new();
        metrics1.increment_stream();
        metrics1.record_error();

        let metrics2 = metrics1.clone();
        let json2 = metrics2.to_json();

        // Cloned metrics should share the same atomic values (Arc)
        assert_eq!(json2["active_streams"], 1);
        assert_eq!(json2["errors"], 1);

        // Changes to clone should affect original
        metrics2.increment_stream();
        let json1 = metrics1.to_json();
        assert_eq!(json1["active_streams"], 2);
    }
}
