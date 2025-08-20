//! Cloud environment detection and configuration optimization.
//!
//! This module detects various cloud environments (AWS, GCP, Azure, etc.) and
//! provides intelligent configuration adjustments for NUMA topology and CPU
//! affinity management in virtualized environments.

use std::fs;
use tracing::{debug, info, warn};

use super::detection::{PlatformInfo, VirtualizationEnvironment};
use crate::error::{AppError, Result};

/// Cloud provider detection results
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudProvider {
    AWS,
    GCP,
    Azure,
    DigitalOcean,
    Linode,
    Oracle,
    Vultr,
    Unknown,
}

/// Cloud instance metadata
#[derive(Debug, Clone)]
pub struct CloudInstanceInfo {
    /// Detected cloud provider
    pub provider: CloudProvider,
    /// Instance type/size (e.g., "c5.xlarge", "n1-standard-4")
    pub instance_type: Option<String>,
    /// Instance ID
    pub instance_id: Option<String>,
    /// Availability zone
    pub availability_zone: Option<String>,
    /// Region
    pub region: Option<String>,
    /// Detected via metadata service
    pub metadata_available: bool,
    /// Detected via DMI/system information
    pub dmi_detection: bool,
}

/// Cloud-specific configuration recommendations
#[derive(Debug, Clone)]
pub struct CloudConfig {
    /// Whether NUMA optimizations should be disabled
    pub disable_numa: bool,
    /// Whether CPU affinity should be disabled
    pub disable_cpu_affinity: bool,
    /// Recommended I/O backend
    pub preferred_io_backend: Option<String>,
    /// Maximum recommended concurrent streams
    pub max_concurrent_streams: Option<usize>,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Network optimization strategy
    pub network_strategy: NetworkStrategy,
}

/// Memory allocation strategies for cloud environments
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStrategy {
    /// Standard allocation, suitable for dedicated instances
    Standard,
    /// Conservative allocation for shared/burstable instances
    Conservative,
    /// Minimize memory footprint for cost optimization
    Minimal,
}

/// Network optimization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum NetworkStrategy {
    /// Optimized for high-throughput, dedicated networking
    HighThroughput,
    /// Balanced approach for general cloud instances
    Balanced,
    /// Conservative for shared networking environments
    Conservative,
}

/// Comprehensive cloud environment detection
pub async fn detect_cloud_environment(platform: &PlatformInfo) -> Result<CloudInstanceInfo> {
    info!("Detecting cloud environment...");

    let mut cloud_info = CloudInstanceInfo {
        provider: CloudProvider::Unknown,
        instance_type: None,
        instance_id: None,
        availability_zone: None,
        region: None,
        metadata_available: false,
        dmi_detection: false,
    };

    // Try DMI-based detection first (works even without network access)
    if let Ok(dmi_info) = detect_via_dmi() {
        cloud_info.provider = dmi_info.provider;
        cloud_info.dmi_detection = true;
        debug!("DMI detection: {:?}", dmi_info.provider);
    }

    // Try metadata service detection (more detailed information)
    // TODO: Re-enable when reqwest dependency is properly configured
    /*
    if let Ok(metadata_info) = detect_via_metadata(&cloud_info.provider).await {
        cloud_info.instance_type = metadata_info.instance_type;
        cloud_info.instance_id = metadata_info.instance_id;
        cloud_info.availability_zone = metadata_info.availability_zone;
        cloud_info.region = metadata_info.region;
        cloud_info.metadata_available = true;

        // Update provider if metadata detection found something more specific
        if cloud_info.provider == CloudProvider::Unknown {
            cloud_info.provider = metadata_info.provider;
        }
    }
    */

    // Fallback to virtualization environment detection
    if cloud_info.provider == CloudProvider::Unknown {
        cloud_info.provider = match platform.virtualization {
            VirtualizationEnvironment::AWS => CloudProvider::AWS,
            VirtualizationEnvironment::GCP => CloudProvider::GCP,
            VirtualizationEnvironment::Azure => CloudProvider::Azure,
            _ => CloudProvider::Unknown,
        };
    }

    info!(
        "Cloud detection complete: provider={:?}, instance_type={:?}, dmi={}, metadata={}",
        cloud_info.provider,
        cloud_info.instance_type,
        cloud_info.dmi_detection,
        cloud_info.metadata_available
    );

    Ok(cloud_info)
}

/// Detect cloud provider via DMI (System Management BIOS) information
fn detect_via_dmi() -> Result<CloudInstanceInfo> {
    let mut provider = CloudProvider::Unknown;

    // Check DMI system information
    if let Ok(vendor) = read_dmi_field("sys_vendor") {
        provider = match vendor.to_lowercase().as_str() {
            s if s.contains("amazon") || s.contains("ec2") => CloudProvider::AWS,
            s if s.contains("google") => CloudProvider::GCP,
            s if s.contains("microsoft") => CloudProvider::Azure,
            s if s.contains("digitalocean") => CloudProvider::DigitalOcean,
            s if s.contains("linode") => CloudProvider::Linode,
            s if s.contains("oracle") => CloudProvider::Oracle,
            _ => CloudProvider::Unknown,
        };
    }

    // Check product name if vendor didn't match
    if provider == CloudProvider::Unknown {
        if let Ok(product) = read_dmi_field("product_name") {
            provider = match product.to_lowercase().as_str() {
                s if s.contains("amazon") || s.contains("ec2") => CloudProvider::AWS,
                s if s.contains("google") => CloudProvider::GCP,
                s if s.contains("virtual machine") => CloudProvider::Azure, // Hyper-V
                s if s.contains("droplet") => CloudProvider::DigitalOcean,
                _ => CloudProvider::Unknown,
            };
        }
    }

    // Check BIOS vendor
    if provider == CloudProvider::Unknown {
        if let Ok(bios_vendor) = read_dmi_field("bios_vendor") {
            provider = match bios_vendor.to_lowercase().as_str() {
                s if s.contains("amazon") => CloudProvider::AWS,
                s if s.contains("google") => CloudProvider::GCP,
                s if s.contains("microsoft") => CloudProvider::Azure,
                _ => CloudProvider::Unknown,
            };
        }
    }

    Ok(CloudInstanceInfo {
        provider,
        instance_type: None,
        instance_id: None,
        availability_zone: None,
        region: None,
        metadata_available: false,
        dmi_detection: true,
    })
}

/// Read DMI field from sysfs
fn read_dmi_field(field: &str) -> Result<String> {
    let path = format!("/sys/class/dmi/id/{}", field);
    fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| AppError::Internal(format!("Failed to read DMI field {}: {}", field, e)))
}

/// Detect cloud environment via metadata services (disabled for now)
#[allow(dead_code)]
async fn detect_via_metadata(_hint: &CloudProvider) -> Result<CloudInstanceInfo> {
    // TODO: Implement when reqwest dependency is properly configured
    Err(crate::error::AppError::Internal(
        "Metadata detection not implemented".to_string(),
    ))
}

/// Detect AWS EC2 metadata (disabled)
#[allow(dead_code)]
async fn detect_aws_metadata() -> Result<CloudInstanceInfo> {
    Err(AppError::Internal(
        "AWS metadata detection not implemented".to_string(),
    ))
}

/// Detect GCP metadata (disabled)
#[allow(dead_code)]
async fn detect_gcp_metadata() -> Result<CloudInstanceInfo> {
    Err(AppError::Internal(
        "GCP metadata detection not implemented".to_string(),
    ))
}

/// Detect Azure metadata (disabled)
#[allow(dead_code)]
async fn detect_azure_metadata() -> Result<CloudInstanceInfo> {
    Err(AppError::Internal(
        "Azure metadata detection not implemented".to_string(),
    ))
}

/// Generate cloud-specific configuration recommendations
pub fn generate_cloud_config(
    cloud_info: &CloudInstanceInfo,
    platform: &PlatformInfo,
) -> CloudConfig {
    info!(
        "Generating cloud-specific configuration for {:?}",
        cloud_info.provider
    );

    match cloud_info.provider {
        CloudProvider::AWS => generate_aws_config(cloud_info, platform),
        CloudProvider::GCP => generate_gcp_config(cloud_info, platform),
        CloudProvider::Azure => generate_azure_config(cloud_info, platform),
        CloudProvider::DigitalOcean => generate_digitalocean_config(cloud_info, platform),
        _ => generate_generic_cloud_config(cloud_info, platform),
    }
}

/// AWS-specific configuration
fn generate_aws_config(cloud_info: &CloudInstanceInfo, _platform: &PlatformInfo) -> CloudConfig {
    let instance_type = cloud_info.instance_type.as_deref().unwrap_or("");

    // Parse instance family and size
    let (disable_numa, disable_cpu_affinity, memory_strategy, max_streams) =
        if instance_type.starts_with("t") {
            // Burstable instances (t2, t3, t4g) - very conservative
            (true, true, MemoryStrategy::Conservative, Some(4))
        } else if instance_type.starts_with("c") {
            // Compute optimized (c5, c6i) - good for CPU-intensive workloads
            (false, false, MemoryStrategy::Standard, Some(8))
        } else if instance_type.starts_with("m") {
            // General purpose (m5, m6i) - balanced
            (true, false, MemoryStrategy::Standard, Some(6))
        } else if instance_type.starts_with("r") || instance_type.starts_with("x") {
            // Memory optimized - good for memory-intensive workloads
            (false, false, MemoryStrategy::Standard, Some(10))
        } else {
            // Unknown instance type - conservative approach
            (true, true, MemoryStrategy::Conservative, Some(4))
        };

    CloudConfig {
        disable_numa,
        disable_cpu_affinity,
        preferred_io_backend: Some("epoll".to_string()), // AWS recommends epoll over io_uring
        max_concurrent_streams: max_streams,
        memory_strategy,
        network_strategy: NetworkStrategy::HighThroughput,
    }
}

/// GCP-specific configuration
fn generate_gcp_config(cloud_info: &CloudInstanceInfo, _platform: &PlatformInfo) -> CloudConfig {
    let machine_type = cloud_info.instance_type.as_deref().unwrap_or("");

    let (disable_numa, memory_strategy, max_streams) = if machine_type.contains("e2-") {
        // E2 instances - cost-optimized, shared CPU
        (true, MemoryStrategy::Conservative, Some(4))
    } else if machine_type.contains("n1-") || machine_type.contains("n2-") {
        // N1/N2 instances - general purpose
        (false, MemoryStrategy::Standard, Some(6))
    } else if machine_type.contains("c2-") {
        // C2 instances - compute optimized
        (false, MemoryStrategy::Standard, Some(8))
    } else if machine_type.contains("m1-") || machine_type.contains("m2-") {
        // Memory optimized instances
        (false, MemoryStrategy::Standard, Some(10))
    } else {
        // Unknown machine type
        (true, MemoryStrategy::Conservative, Some(4))
    };

    CloudConfig {
        disable_numa,
        disable_cpu_affinity: false, // GCP generally has good CPU isolation
        preferred_io_backend: Some("epoll".to_string()),
        max_concurrent_streams: max_streams,
        memory_strategy,
        network_strategy: NetworkStrategy::Balanced,
    }
}

/// Azure-specific configuration
fn generate_azure_config(cloud_info: &CloudInstanceInfo, _platform: &PlatformInfo) -> CloudConfig {
    let vm_size = cloud_info.instance_type.as_deref().unwrap_or("");

    let (memory_strategy, max_streams) = if vm_size.starts_with("Standard_B") {
        // Burstable instances
        (MemoryStrategy::Conservative, Some(4))
    } else if vm_size.starts_with("Standard_F") {
        // Compute optimized
        (MemoryStrategy::Standard, Some(8))
    } else if vm_size.starts_with("Standard_D") || vm_size.starts_with("Standard_E") {
        // General purpose and memory optimized
        (MemoryStrategy::Standard, Some(6))
    } else {
        // Unknown VM size
        (MemoryStrategy::Conservative, Some(4))
    };

    CloudConfig {
        disable_numa: true,         // Azure virtualization often interferes with NUMA
        disable_cpu_affinity: true, // Hyper-V can cause issues with CPU affinity
        preferred_io_backend: Some("epoll".to_string()),
        max_concurrent_streams: max_streams,
        memory_strategy,
        network_strategy: NetworkStrategy::Balanced,
    }
}

/// DigitalOcean-specific configuration
fn generate_digitalocean_config(
    _cloud_info: &CloudInstanceInfo,
    _platform: &PlatformInfo,
) -> CloudConfig {
    CloudConfig {
        disable_numa: true, // DigitalOcean droplets are virtualized
        disable_cpu_affinity: true,
        preferred_io_backend: Some("epoll".to_string()),
        max_concurrent_streams: Some(4),
        memory_strategy: MemoryStrategy::Conservative,
        network_strategy: NetworkStrategy::Conservative,
    }
}

/// Generic cloud configuration for unknown providers
fn generate_generic_cloud_config(
    _cloud_info: &CloudInstanceInfo,
    _platform: &PlatformInfo,
) -> CloudConfig {
    CloudConfig {
        disable_numa: true, // Conservative approach for unknown cloud
        disable_cpu_affinity: true,
        preferred_io_backend: None,
        max_concurrent_streams: Some(4),
        memory_strategy: MemoryStrategy::Conservative,
        network_strategy: NetworkStrategy::Conservative,
    }
}

/// Apply cloud configuration to application settings
pub fn apply_cloud_config(config: &CloudConfig, app_config: &mut crate::config::Config) {
    info!("Applying cloud-specific configuration optimizations...");

    // Apply NUMA and CPU affinity settings
    if config.disable_numa {
        info!("Disabling NUMA optimizations for cloud environment");
        app_config.disable_numa_in_cloud = true;
    }

    if config.disable_cpu_affinity {
        info!("Disabling CPU affinity optimizations for cloud environment");
        app_config.disable_cpu_affinity = true;
    }

    // Apply I/O backend preference
    if let Some(backend) = &config.preferred_io_backend {
        if app_config.force_io_backend.is_none() {
            info!("Setting preferred I/O backend to: {}", backend);
            app_config.force_io_backend = Some(backend.clone());
        }
    }

    // Apply concurrency limits
    if let Some(max_streams) = config.max_concurrent_streams {
        if app_config.max_concurrent_streams > max_streams {
            warn!(
                "Reducing max_concurrent_streams from {} to {} for cloud environment",
                app_config.max_concurrent_streams, max_streams
            );
            app_config.max_concurrent_streams = max_streams;
        }
    }

    // Apply memory strategy
    match config.memory_strategy {
        MemoryStrategy::Conservative => {
            let new_buffer_size = (app_config.audio_buffer_capacity / 2).max(256 * 1024);
            if new_buffer_size < app_config.audio_buffer_capacity {
                info!(
                    "Reducing audio buffer size from {} to {} bytes for conservative memory usage",
                    app_config.audio_buffer_capacity, new_buffer_size
                );
                app_config.audio_buffer_capacity = new_buffer_size;
            }
        }
        MemoryStrategy::Minimal => {
            let new_buffer_size = 256 * 1024; // 256KB minimum
            if new_buffer_size < app_config.audio_buffer_capacity {
                info!(
                    "Setting minimal audio buffer size: {} bytes",
                    new_buffer_size
                );
                app_config.audio_buffer_capacity = new_buffer_size;
            }
        }
        MemoryStrategy::Standard => {
            // Keep default settings
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::detection::detect_platform;
    use std::path::Path;

    #[test]
    fn test_dmi_detection() {
        // This test may fail on non-Linux systems or systems without DMI
        if cfg!(target_os = "linux") && Path::new("/sys/class/dmi/id").exists() {
            let result = detect_via_dmi();
            // Should not panic, but may return Unknown provider
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_cloud_config_generation() {
        let platform = detect_platform();

        let aws_info = CloudInstanceInfo {
            provider: CloudProvider::AWS,
            instance_type: Some("c5.xlarge".to_string()),
            instance_id: Some("i-1234567890abcdef0".to_string()),
            availability_zone: Some("us-west-2a".to_string()),
            region: Some("us-west-2".to_string()),
            metadata_available: true,
            dmi_detection: false,
        };

        let config = generate_cloud_config(&aws_info, &platform);

        // C5 instances should have specific optimizations
        assert!(!config.disable_numa);
        assert!(!config.disable_cpu_affinity);
        assert_eq!(config.preferred_io_backend, Some("epoll".to_string()));
        assert_eq!(config.memory_strategy, MemoryStrategy::Standard);
    }

    #[test]
    fn test_azure_config() {
        let platform = detect_platform();

        let azure_info = CloudInstanceInfo {
            provider: CloudProvider::Azure,
            instance_type: Some("Standard_D4s_v3".to_string()),
            instance_id: Some("12345678-1234-1234-1234-123456789012".to_string()),
            availability_zone: Some("1".to_string()),
            region: Some("eastus".to_string()),
            metadata_available: true,
            dmi_detection: false,
        };

        let config = generate_cloud_config(&azure_info, &platform);

        // Azure should disable NUMA and CPU affinity due to Hyper-V
        assert!(config.disable_numa);
        assert!(config.disable_cpu_affinity);
    }

    #[test]
    fn test_burstable_instance_config() {
        let platform = detect_platform();

        let aws_burstable = CloudInstanceInfo {
            provider: CloudProvider::AWS,
            instance_type: Some("t3.medium".to_string()),
            instance_id: Some("i-1234567890abcdef0".to_string()),
            availability_zone: Some("us-west-2a".to_string()),
            region: Some("us-west-2".to_string()),
            metadata_available: true,
            dmi_detection: false,
        };

        let config = generate_cloud_config(&aws_burstable, &platform);

        // Burstable instances should be very conservative
        assert!(config.disable_numa);
        assert!(config.disable_cpu_affinity);
        assert_eq!(config.memory_strategy, MemoryStrategy::Conservative);
        assert_eq!(config.max_concurrent_streams, Some(4));
    }
}
