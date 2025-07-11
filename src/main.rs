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

    // Create Triton connection pool
    info!(
        "Creating Triton connection pool for {}",
        config.triton_endpoint
    );
    let pool_config = PoolConfig {
        max_connections: MAX_CONCURRENT_STREAMS + MAX_CONCURRENT_BATCHES,
        min_connections: 5,
        ..Default::default()
    };
    let triton_pool = ConnectionPool::new(&config.triton_endpoint, pool_config)
        .await
        .map_err(AppError::from)?;

    // Load vocabulary
    info!("Loading vocabulary from {:?}", config.vocabulary_path);
    let vocabulary = Vocabulary::load_from_file(&config.vocabulary_path)?;
    info!("Loaded vocabulary with {} tokens", vocabulary.len());

    // Create shared vocabulary
    let shared_vocabulary = Arc::new(vocabulary);

    // Create ASR pipeline with connection pool
    let asr_pipeline = Arc::new(TritonAsrPipeline::new(
        triton_pool,
        shared_vocabulary.clone(),
    ));

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
