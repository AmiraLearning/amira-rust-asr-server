//! Triton Inference Server integration.
//!
//! This module provides types and utilities for communicating with the
//! Triton Inference Server via gRPC.

// Re-export proto definitions
pub mod proto {
    tonic::include_proto!("inference");
}

mod client;
mod model;
mod pool;
mod types;

pub use client::{TritonClient, TritonClientError};
pub use pool::{ConnectionPool, PoolConfig, PoolStats, PooledConnection};
pub use model::{
    DecoderJointInput, DecoderJointModel, DecoderJointOutput, EncoderInput, EncoderModel,
    EncoderOutput, PreprocessorInput, PreprocessorModel, PreprocessorOutput, TritonModel,
};
pub use types::{RawTensor, TensorData, TensorDef, TensorShape};
