//! Triton client for communicating with the Triton Inference Server.

use crate::error::{AppError, Result};
use crate::triton::proto::{
    grpc_inference_service_client::GrpcInferenceServiceClient, ModelInferRequest,
    ModelInferResponse,
};
use std::fmt;
use tonic::transport::Channel;
use uuid::Uuid;

/// Error type for Triton client operations.
#[derive(Debug)]
pub enum TritonClientError {
    /// Error creating the Triton client.
    ConnectionError(tonic::transport::Error),

    /// Error executing an inference request.
    InferenceError(tonic::Status),
}

impl fmt::Display for TritonClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionError(e) => write!(f, "Triton connection error: {}", e),
            Self::InferenceError(e) => write!(f, "Triton inference error: {}", e),
        }
    }
}

impl std::error::Error for TritonClientError {}

impl From<TritonClientError> for AppError {
    fn from(err: TritonClientError) -> Self {
        match err {
            TritonClientError::ConnectionError(e) => {
                AppError::Internal(format!("Triton connection error: {}", e))
            }
            TritonClientError::InferenceError(e) => AppError::TritonInference(e.to_string()),
        }
    }
}

/// A client for the Triton Inference Server.
#[derive(Clone)]
pub struct TritonClient {
    /// The gRPC client for communicating with Triton.
    client: GrpcInferenceServiceClient<Channel>,
}

impl TritonClient {
    /// Create a new Triton client.
    ///
    /// # Arguments
    /// * `endpoint` - The URL of the Triton Inference Server.
    ///
    /// # Returns
    /// A new Triton client.
    pub async fn connect(endpoint: &str) -> std::result::Result<Self, TritonClientError> {
        let client = GrpcInferenceServiceClient::connect(endpoint.to_string())
            .await
            .map_err(TritonClientError::ConnectionError)?;

        Ok(Self { client })
    }

    /// Execute an inference request.
    ///
    /// # Arguments
    /// * `request` - The inference request to execute.
    ///
    /// # Returns
    /// The inference response.
    pub async fn infer(&mut self, request: ModelInferRequest) -> Result<ModelInferResponse> {
        let response = self
            .client
            .model_infer(request)
            .await
            .map_err(|e| AppError::TritonInference(e.to_string()))?;

        Ok(response.into_inner())
    }

    /// Create a new inference request builder.
    ///
    /// # Arguments
    /// * `model_name` - The name of the model to infer.
    ///
    /// # Returns
    /// A new inference request builder.
    pub fn request_builder(&self, model_name: &str) -> InferRequestBuilder {
        InferRequestBuilder::new(model_name)
    }
}

/// Builder for Triton inference requests.
pub struct InferRequestBuilder {
    /// The request being built.
    request: ModelInferRequest,
}

impl InferRequestBuilder {
    /// Create a new inference request builder.
    ///
    /// # Arguments
    /// * `model_name` - The name of the model to infer.
    ///
    /// # Returns
    /// A new inference request builder.
    pub fn new(model_name: &str) -> Self {
        let mut request = ModelInferRequest::default();
        request.model_name = model_name.to_string();
        request.id = Uuid::new_v4().to_string();

        Self { request }
    }

    /// Set the model version.
    ///
    /// # Arguments
    /// * `version` - The model version.
    ///
    /// # Returns
    /// This builder for method chaining.
    pub fn with_model_version(mut self, version: &str) -> Self {
        self.request.model_version = version.to_string();
        self
    }

    /// Add an input tensor to the request.
    ///
    /// # Arguments
    /// * `tensor` - The input tensor to add.
    ///
    /// # Returns
    /// This builder for method chaining.
    pub fn with_input(
        mut self,
        tensor: crate::triton::proto::model_infer_request::InferInputTensor,
    ) -> Self {
        self.request.inputs.push(tensor);
        self
    }

    /// Add an output tensor to the request.
    ///
    /// # Arguments
    /// * `tensor` - The output tensor to add.
    ///
    /// # Returns
    /// This builder for method chaining.
    pub fn with_output(
        mut self,
        tensor: crate::triton::proto::model_infer_request::InferRequestedOutputTensor,
    ) -> Self {
        self.request.outputs.push(tensor);
        self
    }

    /// Build the inference request.
    ///
    /// # Returns
    /// The built inference request.
    pub fn build(self) -> ModelInferRequest {
        self.request
    }
}
