//! The `amira_rust_asr_server` core library.
//!
//! This crate provides a high-performance ASR (Automatic Speech Recognition)
//! server using the RNN-T (Recurrent Neural Network Transducer) architecture
//! via Triton Inference Server.

pub mod error;
pub mod types;
pub mod constants;
pub mod config;
pub mod raii;
pub mod async_patterns;
pub mod performance_opts;
pub mod triton;
pub mod asr;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod performance;
pub mod reliability;
pub mod server;
pub mod platform;
