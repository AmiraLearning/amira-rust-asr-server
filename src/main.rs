//! ASR server using RNN-T models via Triton Inference Server.
//!
//! This is the entry point for the ASR server. It initializes the configuration,
//! sets up the ASR pipeline and HTTP server, and starts listening for requests.

use std::sync::Arc;
use tracing::info;
use tracing_subscriber::fmt;

use amira_rust_asr_server::{
    asr::{TritonAsrPipeline, Vocabulary},
    config::Config,
    error::Result,
    platform::initialize_platform,
    server::{create_router, AppState},
    triton::{ConnectionPool, PoolConfig},
};

#[cfg(not(feature = "cuda"))]
use amira_rust_asr_server::error::{AppError, ConfigError};

#[cfg(feature = "cuda")]
use amira_rust_asr_server::{
    asr::CudaAsrPipeline,
    triton::{TritonServerConfig, TritonServerManager},
};

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
    info!(
        "Using inference backend '{}' (is_cuda: {})",
        config.inference_backend,
        config.is_cuda_backend()
    );
    let max_streams = config.max_concurrent_streams;
    let max_batches = config.max_concurrent_batches;

    let asr_pipeline = if config.is_cuda_backend() {
        #[cfg(feature = "cuda")]
        {
            info!("Using CUDA backend for in-process inference with embedded Triton server");

            // Initialize embedded Triton server
            let triton_config = TritonServerConfig {
                model_repository: "./model-repo".to_string(),
                log_verbose: false,
                exit_on_error: false,
                ..Default::default()
            };

            let mut triton_server = TritonServerManager::new(triton_config);
            match triton_server.initialize() {
                Ok(_) => info!("Embedded Triton server initialized successfully"),
                Err(e) => {
                    info!(
                        "Warning: Failed to initialize embedded Triton server: {}",
                        e
                    );
                    info!("Continuing with CUDA memory operations only (no model inference)");
                }
            }

            // Keep server alive for the lifetime of the application
            // We intentionally leak this Arc to ensure the Triton server stays alive
            // This is safer than risking early shutdown
            let triton_server = Arc::new(triton_server);
            std::mem::forget(triton_server);

            Arc::new(CudaAsrPipeline::new(0, shared_vocabulary.clone())?)
                as Arc<dyn amira_rust_asr_server::asr::AsrPipeline + Send + Sync>
        }
        #[cfg(not(feature = "cuda"))]
        {
            use amira_rust_asr_server::error::ConfigError;
            return Err(AppError::Config(ConfigError::Validation(
                "CUDA backend requested but cuda feature not enabled. Build with --features cuda"
                    .to_string(),
            )));
        }
    } else {
        info!(
            "Using gRPC backend with Triton connection pool for {}",
            config.triton_endpoint
        );
        let default_pool_config = PoolConfig::default();
        let pool_max_connections = std::cmp::max(1, max_streams + max_batches);
        let pool_min_connections = std::cmp::min(
            pool_max_connections,
            std::cmp::max(1, default_pool_config.min_connections),
        );

        let pool_config = PoolConfig {
            max_connections: pool_max_connections,
            min_connections: pool_min_connections,
            ..default_pool_config
        };
        let triton_pool = ConnectionPool::new(&config.triton_endpoint, pool_config).await?;

        Arc::new(TritonAsrPipeline::new(
            triton_pool,
            shared_vocabulary.clone(),
        )) as Arc<dyn amira_rust_asr_server::asr::AsrPipeline + Send + Sync>
    };

    // Create application state
    let state = Arc::new(AppState::new(
        asr_pipeline,
        shared_vocabulary,
        max_streams,
        max_batches,
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
