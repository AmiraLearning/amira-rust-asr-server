//! ASR server using RNN-T models via Triton Inference Server.
//!
//! This is the entry point for the ASR server. It initializes the configuration,
//! sets up the ASR pipeline and HTTP server, and starts listening for requests.

use std::sync::Arc;
use tracing::info;
use tracing_subscriber::fmt;

use amira_rust_asr_server::{
    asr::{TritonAsrPipeline, Vocabulary},
    config::{concurrency::*, Config},
    error::{AppError, Result},
    platform::{initialize_platform},
    server::{create_router, AppState},
    triton::{ConnectionPool, PoolConfig},
};

#[cfg(feature = "cuda")]
use amira_rust_asr_server::asr::CudaAsrPipeline;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .json()
        .init();

    // Load configuration
    let config = Config::load()?;
    
    // Initialize platform detection and configuration optimization
    let platform_init = initialize_platform(config).await?;
    let config = platform_init.effective_config;
    
    info!("Platform initialization complete");

    // Load vocabulary
    info!("Loading vocabulary from {:?}", config.vocabulary_path);
    let vocabulary = Vocabulary::load_from_file(&config.vocabulary_path)?;
    info!("Loaded vocabulary with {} tokens", vocabulary.len());

    // Create shared vocabulary
    let shared_vocabulary = Arc::new(vocabulary);

    // Create ASR pipeline based on backend configuration
    info!("DEBUG: inference_backend = '{}', is_cuda = {}", config.inference_backend, config.is_cuda_backend());
    let asr_pipeline = if config.is_cuda_backend() {
        #[cfg(feature = "cuda")]
        {
            info!("Using CUDA backend for in-process inference");
            Arc::new(CudaAsrPipeline::new(0, shared_vocabulary.clone(), 16000.0, 1024)?) as Arc<dyn amira_rust_asr_server::asr::AsrPipeline + Send + Sync>
        }
        #[cfg(not(feature = "cuda"))]
        {
            use amira_rust_asr_server::error::ConfigError;
            return Err(AppError::Config(ConfigError::Validation("CUDA backend requested but cuda feature not enabled. Build with --features cuda".to_string())));
        }
    } else {
        info!("Using gRPC backend with Triton connection pool for {}", config.triton_endpoint);
        let pool_config = PoolConfig {
            max_connections: MAX_CONCURRENT_STREAMS + MAX_CONCURRENT_BATCHES,
            min_connections: 5,
            ..Default::default()
        };
        let triton_pool = ConnectionPool::new(&config.triton_endpoint, pool_config)
            .await
            .map_err(AppError::from)?;
        
        Arc::new(TritonAsrPipeline::new(triton_pool, shared_vocabulary.clone())) as Arc<dyn amira_rust_asr_server::asr::AsrPipeline + Send + Sync>
    };

    // Create application state
    let state = Arc::new(AppState::new(
        asr_pipeline,
        shared_vocabulary,
        MAX_CONCURRENT_STREAMS,
        MAX_CONCURRENT_BATCHES,
    ));

    // Create router
    let app = create_router(state);

    // Start server
    let addr = format!("{}:{}", config.server_host, config.server_port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
