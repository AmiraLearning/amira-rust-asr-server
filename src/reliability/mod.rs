//! Production reliability and observability features.
//!
//! This module provides enterprise-grade reliability features including:
//! - Circuit breakers for fault tolerance
//! - Graceful shutdown handling
//! - Metrics collection and export
//! - Distributed tracing

pub mod circuit_breaker;
pub mod graceful_shutdown;
// pub mod metrics;  // Temporarily disabled - requires external metrics crates
pub mod tracing_config;

pub use circuit_breaker::*;
pub use graceful_shutdown::*;
// pub use metrics::*;
pub use tracing_config::*;
