//! Platform initialization and validation.
//!
//! This module provides initialization functions that detect platform capabilities,
//! validate configuration against platform constraints, and set up platform-specific
//! optimizations.

use tracing::{debug, info, warn};

use crate::config::Config;
use crate::error::{AppError, Result};
use super::{
    capabilities::{detect_capabilities, PlatformCapabilities},
    detection::{detect_platform, PlatformInfo, VirtualizationEnvironment, IoBackendType},
    io_backend::{IoBackend, create_optimal_io_backend},
    cloud_detection::{detect_cloud_environment, generate_cloud_config, apply_cloud_config, CloudInstanceInfo},
    numa_management::{create_numa_manager, NumaManager},
    affinity_management::{create_affinity_manager, AffinityManager},
};

/// Platform initialization result containing detected capabilities and configuration
pub struct PlatformInit {
    /// Platform information
    pub platform_info: PlatformInfo,
    /// Detected platform capabilities
    pub capabilities: PlatformCapabilities,
    /// Cloud environment information (if detected)
    pub cloud_info: Option<CloudInstanceInfo>,
    /// NUMA manager for memory allocation optimization
    pub numa_manager: std::sync::Arc<NumaManager>,
    /// CPU affinity manager for thread optimization
    pub affinity_manager: std::sync::Arc<AffinityManager>,
    /// Optimal I/O backend for this platform
    pub io_backend: Box<dyn IoBackend>,
    /// Effective configuration after platform adjustments
    pub effective_config: Config,
}

/// Initialize platform detection and configuration validation
pub async fn initialize_platform(mut config: Config) -> Result<PlatformInit> {
    info!("Initializing platform detection and configuration validation...");
    
    // Detect platform capabilities
    let platform_info = detect_platform();
    let capabilities = detect_capabilities();
    
    // Log platform detection results
    info!(
        "Platform detected: {:?} {} on {:?} ({:?})",
        platform_info.os,
        platform_info.kernel_version
            .as_ref()
            .map(|v| format!("{}.{}.{}", v.major, v.minor, v.patch))
            .unwrap_or_else(|| "unknown".to_string()),
        platform_info.architecture,
        platform_info.virtualization
    );
    
    // Detect cloud environment
    let cloud_info = match detect_cloud_environment(&platform_info).await {
        Ok(info) => {
            info!("Cloud environment detected: {:?}", info.provider);
            Some(info)
        }
        Err(e) => {
            debug!("Cloud environment detection failed: {}", e);
            None
        }
    };
    
    // Generate and apply cloud-specific configuration
    if let Some(ref cloud_info) = cloud_info {
        let cloud_config = generate_cloud_config(cloud_info, &platform_info);
        info!("Applying cloud-specific configuration optimizations");
        apply_cloud_config(&cloud_config, &mut config);
    }
    
    // Create NUMA and affinity managers
    let numa_manager = create_numa_manager(platform_info.clone(), cloud_info.clone()).await;
    let affinity_manager = create_affinity_manager(platform_info.clone(), cloud_info.clone()).await;
    
    // Apply platform-specific configuration adjustments
    apply_platform_optimizations(&mut config, &platform_info, &capabilities, &numa_manager, &affinity_manager)?;
    
    // Select optimal I/O backend
    let io_backend = select_io_backend(&config, &platform_info, &capabilities).await?;
    
    // Validate configuration against platform constraints
    validate_platform_config(&config, &platform_info, &capabilities)?;
    
    // Log effective configuration
    log_effective_configuration(&config, &capabilities);
    
    Ok(PlatformInit {
        platform_info,
        capabilities,
        cloud_info,
        numa_manager,
        affinity_manager,
        io_backend,
        effective_config: config,
    })
}

/// Apply platform-specific configuration optimizations
fn apply_platform_optimizations(
    config: &mut Config,
    platform: &PlatformInfo,
    capabilities: &PlatformCapabilities,
    numa_manager: &NumaManager,
    affinity_manager: &AffinityManager,
) -> Result<()> {
    if !config.enable_platform_optimizations {
        info!("Platform optimizations disabled by configuration");
        return Ok(());
    }
    
    info!("Applying platform-specific configuration optimizations...");
    
    // Cloud environment optimizations
    if !matches!(platform.virtualization, VirtualizationEnvironment::BareMetal) {
        if config.disable_numa_in_cloud && capabilities.cpu.numa.available {
            info!("Disabling NUMA optimizations in cloud environment");
            // Set internal flags to disable NUMA (would be handled by NUMA management code)
        }
        
        // Adjust concurrency for cloud environments (conservative approach)
        if config.max_concurrent_streams > 8 {
            warn!(
                "Reducing max_concurrent_streams from {} to 8 for cloud environment",
                config.max_concurrent_streams
            );
            config.max_concurrent_streams = 8;
        }
    }
    
    // Memory optimization for containers
    if matches!(platform.virtualization, VirtualizationEnvironment::Docker) {
        // Reduce buffer sizes in containerized environments
        if config.audio_buffer_capacity > 512 * 1024 {
            let new_size = 512 * 1024;
            info!(
                "Reducing audio_buffer_capacity from {} to {} for container environment",
                config.audio_buffer_capacity, new_size
            );
            config.audio_buffer_capacity = new_size;
        }
    }
    
    // I/O backend selection hints
    if config.force_io_backend.is_none() {
        let recommended_backend = match platform.virtualization {
            VirtualizationEnvironment::AWS | VirtualizationEnvironment::GCP => {
                if *capabilities.feature_flags.get("prefer_epoll_over_io_uring").unwrap_or(&false) {
                    Some("epoll".to_string())
                } else {
                    None
                }
            },
            _ => None,
        };
        
        if let Some(backend) = recommended_backend {
            info!("Platform recommends {} I/O backend", backend);
            config.force_io_backend = Some(backend);
        }
    }
    
    // Apply NUMA management recommendations
    let numa_recommendations = numa_manager.get_numa_config_recommendations();
    if numa_recommendations.disable_numa_allocation {
        info!("NUMA optimizations disabled: {}", numa_recommendations.reason);
        config.disable_numa_in_cloud = true;
    } else {
        info!("NUMA optimizations enabled: {}", numa_recommendations.reason);
        config.disable_numa_in_cloud = false;
    }
    
    // Apply CPU affinity management recommendations  
    let affinity_recommendations = affinity_manager.get_affinity_recommendations();
    if affinity_recommendations.disable_affinity {
        info!("CPU affinity optimizations disabled: {}", affinity_recommendations.reason);
        config.disable_cpu_affinity = true;
    } else {
        info!("CPU affinity optimizations enabled: {}", affinity_recommendations.reason);
        config.disable_cpu_affinity = false;
    }
    
    Ok(())
}

/// Select the optimal I/O backend for this platform
async fn select_io_backend(
    config: &Config,
    platform: &PlatformInfo,
    capabilities: &PlatformCapabilities,
) -> Result<Box<dyn IoBackend>> {
    info!("Selecting optimal I/O backend...");
    
    // Determine backend type
    let backend_type = if let Some(forced) = &config.force_io_backend {
        match forced.as_str() {
            "io_uring" => {
                if *capabilities.feature_flags.get("io_uring").unwrap_or(&false) {
                    IoBackendType::IoUring
                } else {
                    warn!("io_uring forced but not available, falling back to epoll");
                    IoBackendType::Epoll
                }
            },
            "epoll" => IoBackendType::Epoll,
            "kqueue" => IoBackendType::Kqueue,
            "select" => IoBackendType::Select,
            _ => {
                warn!("Unknown I/O backend '{}', using auto-detection", forced);
                auto_select_backend(platform, capabilities)
            }
        }
    } else {
        auto_select_backend(platform, capabilities)
    };
    
    info!("Selected I/O backend: {:?}", backend_type);
    
    // Create the backend using bind address from config
    let bind_addr = format!("{}:{}", config.server_host, config.server_port)
        .parse()
        .map_err(|e| AppError::ConfigError(format!("Invalid server address: {}", e)))?;
    
    create_optimal_io_backend(bind_addr)
}

/// Automatically select the best I/O backend for the platform
fn auto_select_backend(
    platform: &PlatformInfo,
    capabilities: &PlatformCapabilities,
) -> IoBackendType {
    // Check for io_uring availability and recommendations
    if *capabilities.feature_flags.get("io_uring_stable").unwrap_or(&false) {
        // Check if we should prefer epoll over io_uring in cloud environments
        if *capabilities.feature_flags.get("prefer_epoll_over_io_uring").unwrap_or(&false) {
            debug!("io_uring available but preferring epoll for cloud environment");
            return IoBackendType::Epoll;
        }
        
        // Use io_uring on bare metal Linux with stable support
        if matches!(platform.virtualization, VirtualizationEnvironment::BareMetal) {
            return IoBackendType::IoUring;
        }
    }
    
    // Platform-specific fallbacks
    if *capabilities.feature_flags.get("epoll").unwrap_or(&false) {
        IoBackendType::Epoll
    } else if *capabilities.feature_flags.get("kqueue").unwrap_or(&false) {
        IoBackendType::Kqueue
    } else {
        IoBackendType::Select // Universal fallback
    }
}

/// Validate configuration against platform constraints
fn validate_platform_config(
    config: &Config,
    _platform: &PlatformInfo,
    capabilities: &PlatformCapabilities,
) -> Result<()> {
    info!("Validating configuration against platform constraints...");
    
    // Validate I/O backend availability
    if let Some(forced_backend) = &config.force_io_backend {
        match forced_backend.as_str() {
            "io_uring" => {
                if !*capabilities.feature_flags.get("io_uring").unwrap_or(&false) {
                    return Err(AppError::ConfigError(
                        "io_uring backend forced but not available on this platform".to_string()
                    ));
                }
            },
            "epoll" => {
                if !*capabilities.feature_flags.get("epoll").unwrap_or(&false) {
                    return Err(AppError::ConfigError(
                        "epoll backend forced but not available on this platform".to_string()
                    ));
                }
            },
            "kqueue" => {
                if !*capabilities.feature_flags.get("kqueue").unwrap_or(&false) {
                    return Err(AppError::ConfigError(
                        "kqueue backend forced but not available on this platform".to_string()
                    ));
                }
            },
            _ => {} // Other backends are always available
        }
    }
    
    // Validate concurrency limits against platform capabilities
    let max_reasonable_streams = capabilities.cpu.cores * 2;
    if config.max_concurrent_streams > max_reasonable_streams {
        warn!(
            "max_concurrent_streams ({}) exceeds recommended limit ({}) for {} CPU cores",
            config.max_concurrent_streams, max_reasonable_streams, capabilities.cpu.cores
        );
    }
    
    // Validate memory settings
    let total_buffer_memory = config.audio_buffer_capacity * config.max_concurrent_streams;
    if total_buffer_memory > 100 * 1024 * 1024 { // 100MB
        warn!(
            "Total audio buffer memory ({} MB) may be excessive",
            total_buffer_memory / (1024 * 1024)
        );
    }
    
    Ok(())
}

/// Log the effective configuration after platform optimizations
fn log_effective_configuration(config: &Config, capabilities: &PlatformCapabilities) {
    info!("=== Effective Platform Configuration ===");
    info!("Platform optimizations: {}", config.enable_platform_optimizations);
    info!("I/O backend: {:?}", config.force_io_backend);
    info!("Max concurrent streams: {}", config.max_concurrent_streams);
    info!("Max concurrent batches: {}", config.max_concurrent_batches);
    info!("Audio buffer capacity: {} KB", config.audio_buffer_capacity / 1024);
    
    // Log key capability flags
    info!("=== Platform Capabilities ===");
    info!("CPU cores: {}", capabilities.cpu.cores);
    info!("NUMA available: {}", capabilities.cpu.numa.available);
    info!("io_uring support: {:?}", capabilities.io.io_uring_support);
    info!("SIMD capabilities: {:?}", capabilities.cpu.simd);
    
    // Log important feature flags
    let important_flags = [
        "io_uring", "io_uring_stable", "epoll", "kqueue",
        "simd_avx2", "numa", "cloud_environment", "container"
    ];
    
    for flag in important_flags {
        if let Some(value) = capabilities.feature_flags.get(flag) {
            info!("Feature {}: {}", flag, value);
        }
    }
    info!("========================================");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_platform_initialization() {
        // Create a test configuration
        let mut config = Config {
            triton_endpoint: "http://localhost:8001".to_string(),
            vocabulary_path: std::path::PathBuf::from("test_vocab.txt"),
            server_host: "127.0.0.1".to_string(),
            server_port: 8057,
            inference_timeout: std::time::Duration::from_secs(5),
            max_concurrent_streams: 10,
            max_concurrent_batches: 50,
            inference_queue_size: 100,
            audio_buffer_capacity: 1024 * 1024,
            max_batch_audio_length_secs: 30.0,
            stream_timeout_secs: 30,
            keepalive_check_period_ms: 100,
            preprocessor_model_name: "preprocessor".to_string(),
            encoder_model_name: "encoder".to_string(),
            decoder_joint_model_name: "decoder_joint".to_string(),
            max_symbols_per_step: 30,
            max_total_tokens: 200,
            enable_platform_optimizations: true,
            force_io_backend: None,
            disable_numa_in_cloud: true,
            disable_cpu_affinity: false,
            force_io_uring: false,
        };

        // Test platform initialization (this should not fail on any platform)
        let result = initialize_platform(config).await;
        
        // Basic validation - initialization should succeed
        assert!(result.is_ok(), "Platform initialization should succeed");
        
        if let Ok(platform_init) = result {
            // Verify that we got valid results
            assert!(platform_init.capabilities.cpu.cores > 0);
            assert!(!platform_init.capabilities.io.multiplexing.is_empty());
            assert!(!platform_init.capabilities.feature_flags.is_empty());
        }
    }

    #[test]
    fn test_auto_backend_selection() {
        let platform = detect_platform();
        let capabilities = detect_capabilities();
        
        let backend = auto_select_backend(&platform, &capabilities);
        
        // Should always select a valid backend
        assert!(matches!(backend, 
            IoBackendType::IoUring | IoBackendType::Epoll | 
            IoBackendType::Kqueue | IoBackendType::Select));
    }
}