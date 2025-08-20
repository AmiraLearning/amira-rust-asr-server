//! Test binary for CUDA integration
//!
//! This binary tests the CUDA pipeline integration without actually running
//! CUDA operations (since we might not have CUDA available in all environments).

use amira_rust_asr_server::config::Config;
use amira_rust_asr_server::error::Result;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Testing CUDA integration...");

    // Test configuration loading
    let config = Config::from_env()?;
    info!(
        "Loaded configuration: backend = {}, device_id = {}",
        config.inference_backend, config.cuda_device_id
    );

    // Test backend validation
    match config.validate_backend() {
        Ok(()) => info!("Backend configuration is valid"),
        Err(e) => warn!("Backend validation failed: {}", e),
    }

    // Test CUDA availability (if compiled with CUDA support)
    #[cfg(feature = "cuda")]
    {
        info!("Testing CUDA module...");

        match amira_rust_asr_server::cuda::is_cuda_available() {
            true => {
                info!("CUDA is available!");
                match amira_rust_asr_server::cuda::get_cuda_device_count() {
                    Ok(count) => info!("Found {} CUDA device(s)", count),
                    Err(e) => warn!("Failed to get CUDA device count: {}", e),
                }
            }
            false => warn!("CUDA is not available on this system"),
        }

        // Test model configuration
        let preprocessor_config = amira_rust_asr_server::cuda::ModelConfig::preprocessor();
        info!(
            "Preprocessor config: total input size = {} bytes",
            preprocessor_config.total_input_size()
        );

        let encoder_config = amira_rust_asr_server::cuda::ModelConfig::encoder();
        info!(
            "Encoder config: total input size = {} bytes",
            encoder_config.total_input_size()
        );

        let decoder_config = amira_rust_asr_server::cuda::ModelConfig::decoder_joint();
        info!(
            "Decoder config: total input size = {} bytes",
            decoder_config.total_input_size()
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        info!("CUDA support not compiled in - this is expected for non-CUDA builds");
    }

    info!("CUDA integration test completed successfully!");
    Ok(())
}
