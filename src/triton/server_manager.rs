//! Embedded Triton Inference Server Manager
//!
//! This module manages an embedded Triton Inference Server instance using the C API.
//! Unlike the gRPC-based approach, this enables true zero-copy CUDA memory sharing
//! between the ASR pipeline and Triton models, eliminating cross-process IPC issues.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │           Rust Process (ASR Server)              │
//! │                                                  │
//! │  ┌──────────────┐    ┌─────────────────────┐   │
//! │  │   WebSocket  │    │  Embedded Triton    │   │
//! │  │   Handler    │    │  Server Instance    │   │
//! │  └──────┬───────┘    └─────────┬───────────┘   │
//! │         │                       │               │
//! │         │    ┌──────────────────┼─────┐         │
//! │         └───→│  CUDA Shared Memory     │←───────┤
//! │              │  (Zero-Copy Access)     │        │
//! │              └─────────────────────────┘        │
//! │                        │                        │
//! │              ┌─────────┴─────────┐              │
//! │              │   GPU Memory      │              │
//! │              └───────────────────┘              │
//! └──────────────────────────────────────────────────┘
//! ```
//!
//! ## Benefits
//!
//! - **Zero-Copy Inference**: Triton accesses CUDA memory directly without IPC
//! - **No Memory Lifecycle Conflicts**: Single-process ownership prevents double-free
//! - **Lower Latency**: Eliminates gRPC serialization and network overhead
//! - **Simplified Deployment**: Single binary with embedded model server
//!
//! ## Safety
//!
//! The C API integration uses FFI to communicate with Triton's C++ implementation.
//! All unsafe blocks are carefully documented and bounded by safe Rust wrappers.

use crate::error::{AppError, Result};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, error, info, warn};

// FFI declarations for Triton C API
// These match the functions implemented in src/cuda/cuda_helper.cu

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // These variants may be used by C API or in future implementations
pub enum CudaError {
    Success = 0,
    InvalidValue = 1,
    NotFound = 2,
    InitializationError = 3,
    AlreadyRegistered = 4,
    Unknown = 999,
}

extern "C" {
    /// Initialize the embedded Triton server
    fn InitializeTritonServer() -> CudaError;

    /// Shutdown the embedded Triton server
    fn ShutdownTritonServer() -> CudaError;

    /// Check if Triton server is ready
    fn IsTritonServerReady() -> bool;

    /// Perform inference on a model
    #[allow(dead_code)] // Reserved for future direct FFI inference
    fn TritonInference(
        model_name: *const c_char,
        input_data: *const c_void,
        input_size: usize,
        output_data: *mut c_void,
        output_size: *mut usize,
    ) -> CudaError;
}

/// Configuration for the embedded Triton server
#[derive(Debug, Clone)]
pub struct TritonServerConfig {
    /// Path to the model repository
    pub model_repository: String,

    /// Triton server options
    pub log_verbose: bool,
    pub exit_on_error: bool,

    /// Resource limits
    pub min_supported_compute_capability: f32,

    /// Model control mode
    pub model_control_mode: ModelControlMode,
}

impl Default for TritonServerConfig {
    fn default() -> Self {
        Self {
            model_repository: "./model-repo".to_string(),
            log_verbose: false,
            exit_on_error: false,
            min_supported_compute_capability: 7.5,
            model_control_mode: ModelControlMode::Explicit,
        }
    }
}

/// Model control modes for Triton
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelControlMode {
    /// Load all models at startup
    All,
    /// Load models explicitly via API
    Explicit,
    /// Load no models (manual control)
    None,
}

impl ModelControlMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelControlMode::All => "all",
            ModelControlMode::Explicit => "explicit",
            ModelControlMode::None => "none",
        }
    }
}

/// Manager for the embedded Triton Inference Server
///
/// This struct provides a safe Rust interface to the embedded Triton server.
/// It ensures proper initialization and cleanup of the server instance.
#[derive(Debug)]
pub struct TritonServerManager {
    config: TritonServerConfig,
    initialized: bool,
}

impl TritonServerManager {
    /// Create a new Triton server manager
    ///
    /// # Arguments
    /// * `config` - Server configuration
    ///
    /// # Returns
    /// A new uninitialized server manager
    pub fn new(config: TritonServerConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }

    /// Initialize the embedded Triton server
    ///
    /// This must be called before any inference operations can be performed.
    /// It loads the model repository and prepares all models for inference.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The server is already initialized
    /// - The model repository path is invalid
    /// - Triton initialization fails
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Err(AppError::Internal(
                "Triton server already initialized".to_string(),
            ));
        }

        // Validate model repository exists
        let model_repo_path = Path::new(&self.config.model_repository);
        if !model_repo_path.exists() {
            return Err(AppError::Internal(format!(
                "Model repository not found: {}",
                self.config.model_repository
            )));
        }

        info!(
            "Initializing embedded Triton server with model repository: {}",
            self.config.model_repository
        );

        // Initialize Triton server via C API
        let result = unsafe { InitializeTritonServer() };

        match result {
            CudaError::Success => {
                self.initialized = true;
                info!("✓ Embedded Triton server initialized successfully");
                Ok(())
            }
            CudaError::AlreadyRegistered => {
                warn!("Triton server was already initialized");
                self.initialized = true;
                Ok(())
            }
            error => {
                error!("Failed to initialize Triton server: {:?}", error);
                Err(AppError::Internal(format!(
                    "Triton initialization failed: {:?}",
                    error
                )))
            }
        }
    }

    /// Check if the server is ready to accept inference requests
    ///
    /// # Returns
    /// `true` if the server is initialized and ready, `false` otherwise
    pub fn is_ready(&self) -> bool {
        if !self.initialized {
            return false;
        }

        unsafe { IsTritonServerReady() }
    }

    /// Get the server configuration
    pub fn config(&self) -> &TritonServerConfig {
        &self.config
    }

    /// Check if the server is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Drop for TritonServerManager {
    fn drop(&mut self) {
        if self.initialized {
            info!("Shutting down embedded Triton server");

            let result = unsafe { ShutdownTritonServer() };

            match result {
                CudaError::Success => {
                    debug!("✓ Triton server shutdown successfully");
                }
                error => {
                    error!("Error shutting down Triton server: {:?}", error);
                }
            }

            self.initialized = false;
        }
    }
}

/// Thread-safe handle to the Triton server manager
///
/// This allows the server to be shared across multiple threads while
/// ensuring proper synchronization.
pub type SharedTritonServerManager = Arc<parking_lot::RwLock<TritonServerManager>>;

/// Create a new shared Triton server manager
///
/// # Arguments
/// * `config` - Server configuration
///
/// # Returns
/// A thread-safe handle to the server manager
pub fn create_shared_server_manager(config: TritonServerConfig) -> SharedTritonServerManager {
    Arc::new(parking_lot::RwLock::new(TritonServerManager::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TritonServerConfig::default();
        assert_eq!(config.model_repository, "./model-repo");
        assert!(!config.log_verbose);
        assert!(!config.exit_on_error);
    }

    #[test]
    fn test_model_control_mode_str() {
        assert_eq!(ModelControlMode::All.as_str(), "all");
        assert_eq!(ModelControlMode::Explicit.as_str(), "explicit");
        assert_eq!(ModelControlMode::None.as_str(), "none");
    }

    #[test]
    fn test_manager_creation() {
        let config = TritonServerConfig::default();
        let manager = TritonServerManager::new(config);
        assert!(!manager.is_initialized());
        assert!(!manager.is_ready());
    }

    #[test]
    fn test_shared_manager_creation() {
        let config = TritonServerConfig::default();
        let shared_manager = create_shared_server_manager(config);

        let manager = shared_manager.read();
        assert!(!manager.is_initialized());
    }
}
