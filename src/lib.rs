//! The `amira_rust_asr_server` core library.
//!
//! This crate provides a high-performance ASR (Automatic Speech Recognition)
//! server using the RNN-T (Recurrent Neural Network Transducer) architecture
//! via Triton Inference Server.

pub mod error;
pub mod config;
pub mod triton;
pub mod asr;
pub mod performance;
pub mod reliability;
pub mod server;
