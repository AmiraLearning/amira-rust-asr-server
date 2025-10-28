//! Domain-specific error type definitions.
//!
//! This module organizes error types by domain for better structure and maintainability.

pub mod asr;
pub mod config;
pub mod model;
pub mod performance;
pub mod server;
pub mod triton;

#[cfg(feature = "cuda")]
pub mod cuda;

// Re-export all error types for convenient access
pub use asr::{AsrError, AudioError};
pub use config::ConfigError;
pub use model::ModelError;
pub use performance::PerformanceError;
pub use server::ServerError;
pub use triton::TritonError;

#[cfg(feature = "cuda")]
pub use cuda::CudaError;
