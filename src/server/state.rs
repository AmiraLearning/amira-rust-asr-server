//! Application state for dependency injection.
//!
//! This module provides the application state that is shared
//! between all request handlers.

use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;

use crate::asr::{AsrPipeline, Vocabulary};
use crate::server::metrics::ServiceMetrics;
use crate::server::stream::StreamHandle;

/// Shared application state containing dependencies.
#[derive(Clone)]
pub struct AppState {
    /// The ASR pipeline implementation
    pub asr_pipeline: Arc<dyn AsrPipeline>,

    /// Vocabulary for token decoding
    pub vocabulary: Arc<Vocabulary>,

    /// Service metrics
    pub metrics: Arc<ServiceMetrics>,

    /// Semaphore to limit concurrent streams
    pub stream_semaphore: Arc<Semaphore>,

    /// Semaphore to limit concurrent batch requests
    pub batch_semaphore: Arc<Semaphore>,

    /// Map of active streams
    pub active_streams: Arc<DashMap<String, StreamHandle>>,
}

impl AppState {
    /// Create a new application state.
    ///
    /// # Arguments
    /// * `asr_pipeline` - The ASR pipeline implementation
    /// * `vocabulary` - Vocabulary for token decoding
    /// * `max_concurrent_streams` - Maximum number of concurrent streams
    /// * `max_concurrent_batches` - Maximum number of concurrent batch requests
    ///
    /// # Returns
    /// A new application state
    pub fn new(
        asr_pipeline: Arc<dyn AsrPipeline>,
        vocabulary: Arc<Vocabulary>,
        max_concurrent_streams: usize,
        max_concurrent_batches: usize,
    ) -> Self {
        Self {
            asr_pipeline,
            vocabulary,
            metrics: Arc::new(ServiceMetrics::new()),
            stream_semaphore: Arc::new(Semaphore::new(max_concurrent_streams)),
            batch_semaphore: Arc::new(Semaphore::new(max_concurrent_batches)),
            active_streams: Arc::new(DashMap::new()),
        }
    }
}
