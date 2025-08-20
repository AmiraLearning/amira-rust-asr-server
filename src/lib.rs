//! The `amira_rust_asr_server` core library.
//!
//! This crate provides a high-performance ASR (Automatic Speech Recognition)
//! server using the RNN-T (Recurrent Neural Network Transducer) architecture
//! via Triton Inference Server.

pub mod asr;
pub mod async_patterns;
pub mod config;
pub mod constants;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod error;
pub mod performance;
pub mod performance_opts;
pub mod platform;
pub mod raii;
pub mod reliability;
pub mod server;
pub mod triton;
pub mod types;
