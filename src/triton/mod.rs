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
mod reliable_client;
mod types;

pub use client::{TritonClient, TritonClientError};
pub use model::{
    DecoderJointInput, DecoderJointModel, DecoderJointOutput, EncoderInput, EncoderModel,
    EncoderOutput, PreprocessorInput, PreprocessorModel, PreprocessorOutput, TritonModel,
};
pub use pool::{ConnectionPool, PoolConfig, PoolStats, PooledConnection};
pub use reliable_client::{ReliableTritonClient, ReliableTritonClientBuilder};
pub use types::{RawTensor, TensorData, TensorDef, TensorShape};
