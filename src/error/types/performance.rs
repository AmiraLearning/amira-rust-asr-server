//! Performance and reliability error types.
//!
//! This module contains error types for performance-related issues,
//! including memory allocation, CPU affinity, NUMA, and circuit breaker errors.

use thiserror::Error;

/// Performance and reliability errors.
#[derive(Debug, Error)]
pub enum PerformanceError {
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("CPU affinity error: {0}")]
    CpuAffinity(String),

    #[error("NUMA error: {0}")]
    Numa(String),

    #[error("Circuit breaker open: {0}")]
    CircuitBreakerOpen(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}
