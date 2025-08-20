//! NUMA topology management with cloud environment awareness.
//!
//! This module provides intelligent NUMA (Non-Uniform Memory Access) management
//! that automatically disables optimizations in cloud environments where they
//! may be counterproductive or unavailable.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::cloud_detection::{CloudInstanceInfo, CloudProvider};
use super::detection::PlatformInfo;
use crate::error::Result;
use crate::performance::numa_aware::{NumaNode, NumaTopology};

/// NUMA management strategy based on environment detection
#[derive(Debug, Clone, PartialEq)]
pub enum NumaStrategy {
    /// NUMA optimizations are enabled and recommended
    Enabled,
    /// NUMA optimizations are disabled due to cloud environment
    DisabledCloud,
    /// NUMA optimizations are disabled due to lack of hardware support
    DisabledUnsupported,
    /// NUMA optimizations are disabled by user configuration
    DisabledUser,
    /// NUMA detection failed, using fallback
    Fallback,
}

/// Intelligent NUMA manager that adapts to the environment
#[derive(Debug)]
pub struct NumaManager {
    /// Current NUMA strategy
    strategy: NumaStrategy,
    /// Detected NUMA topology (if available)
    topology: Option<NumaTopology>,
    /// Whether NUMA operations are globally disabled
    disabled: AtomicBool,
    /// Cloud environment information
    cloud_info: Option<CloudInstanceInfo>,
    /// Platform information
    #[allow(dead_code)]
    platform_info: PlatformInfo,
}

impl NumaManager {
    /// Create a new NUMA manager with environment detection
    pub fn new(platform_info: PlatformInfo, cloud_info: Option<CloudInstanceInfo>) -> Self {
        let mut manager = Self {
            strategy: NumaStrategy::Fallback,
            topology: None,
            disabled: AtomicBool::new(false),
            cloud_info,
            platform_info,
        };

        manager.initialize();
        manager
    }

    /// Initialize NUMA management based on environment detection
    fn initialize(&mut self) {
        info!("Initializing NUMA management...");

        // First check if we're in a cloud environment that should disable NUMA
        if let Some(cloud_info) = &self.cloud_info {
            if self.should_disable_numa_for_cloud(cloud_info) {
                self.strategy = NumaStrategy::DisabledCloud;
                self.disabled.store(true, Ordering::Relaxed);
                info!(
                    "NUMA disabled for cloud environment: {:?}",
                    cloud_info.provider
                );
                return;
            }
        }

        // Try to detect NUMA topology
        match self.detect_numa_topology() {
            Ok(topology) => {
                if topology.numa_available && topology.nodes.len() > 1 {
                    self.topology = Some(topology);
                    self.strategy = NumaStrategy::Enabled;
                    info!(
                        "NUMA enabled with {} nodes",
                        self.topology.as_ref().unwrap().nodes.len()
                    );
                } else {
                    self.strategy = NumaStrategy::DisabledUnsupported;
                    self.disabled.store(true, Ordering::Relaxed);
                    info!("NUMA disabled: single node or not supported");
                }
            }
            Err(e) => {
                warn!("NUMA topology detection failed: {}", e);
                self.strategy = NumaStrategy::Fallback;
                self.disabled.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Check if NUMA should be disabled for the detected cloud environment
    fn should_disable_numa_for_cloud(&self, cloud_info: &CloudInstanceInfo) -> bool {
        match cloud_info.provider {
            CloudProvider::Azure => {
                // Azure Hyper-V virtualization often interferes with NUMA
                info!("Disabling NUMA for Azure Hyper-V environment");
                true
            }
            CloudProvider::AWS => {
                // AWS: disable for burstable instances, enable for compute optimized
                if let Some(instance_type) = &cloud_info.instance_type {
                    let should_disable = instance_type.starts_with('t') ||  // t2, t3, t4g burstable
                                        instance_type.starts_with("a1") || // ARM instances
                                        instance_type.starts_with("m6a"); // AMD instances with potential NUMA issues

                    if should_disable {
                        info!("Disabling NUMA for AWS instance type: {}", instance_type);
                    } else {
                        info!("Enabling NUMA for AWS instance type: {}", instance_type);
                    }
                    should_disable
                } else {
                    // Unknown instance type, be conservative
                    true
                }
            }
            CloudProvider::GCP => {
                // GCP: disable for shared-core instances, enable for dedicated
                if let Some(machine_type) = &cloud_info.instance_type {
                    let should_disable = machine_type.contains("e2-") ||    // Cost-optimized shared CPU
                                        machine_type.contains("f1-") ||    // Micro instances
                                        machine_type.contains("g1-"); // Small shared instances

                    if should_disable {
                        info!("Disabling NUMA for GCP machine type: {}", machine_type);
                    } else {
                        info!("Enabling NUMA for GCP machine type: {}", machine_type);
                    }
                    should_disable
                } else {
                    // Unknown machine type, be conservative
                    true
                }
            }
            CloudProvider::DigitalOcean | CloudProvider::Linode | CloudProvider::Vultr => {
                // Most smaller cloud providers use full virtualization
                info!(
                    "Disabling NUMA for cloud provider: {:?}",
                    cloud_info.provider
                );
                true
            }
            CloudProvider::Oracle => {
                // Oracle Cloud has bare metal instances that support NUMA
                if let Some(instance_type) = &cloud_info.instance_type {
                    let is_bare_metal =
                        instance_type.contains("BM") || instance_type.contains("BareMetal");
                    if is_bare_metal {
                        info!(
                            "Enabling NUMA for Oracle bare metal instance: {}",
                            instance_type
                        );
                        false
                    } else {
                        info!("Disabling NUMA for Oracle VM instance: {}", instance_type);
                        true
                    }
                } else {
                    true
                }
            }
            CloudProvider::Unknown => {
                // Unknown cloud provider, be conservative
                info!("Disabling NUMA for unknown cloud provider");
                true
            }
        }
    }

    /// Detect NUMA topology with enhanced error handling
    fn detect_numa_topology(&self) -> Result<NumaTopology> {
        debug!("Detecting NUMA topology...");

        // Use the existing NumaTopology::detect() but add validation
        let topology = NumaTopology::detect();

        // Validate the detected topology
        if !topology.numa_available {
            return Ok(topology); // Not an error, just not available
        }

        if topology.nodes.is_empty() {
            warn!("NUMA reported as available but no nodes detected");
            return Ok(NumaTopology {
                nodes: vec![],
                cores_per_node: HashMap::new(),
                numa_available: false,
            });
        }

        // Check if NUMA topology makes sense
        let total_cores: usize = topology
            .cores_per_node
            .values()
            .map(|cores| cores.len())
            .sum();

        let system_cores = num_cpus::get();

        if total_cores == 0 {
            warn!("NUMA topology detected but no cores assigned to nodes");
            return Ok(NumaTopology {
                nodes: vec![],
                cores_per_node: HashMap::new(),
                numa_available: false,
            });
        }

        if total_cores > system_cores * 2 {
            warn!(
                "NUMA topology reports {} cores but system has {} cores, topology may be unreliable",
                total_cores, system_cores
            );
        }

        info!(
            "NUMA topology validated: {} nodes, {} total cores assigned",
            topology.nodes.len(),
            total_cores
        );

        Ok(topology)
    }

    /// Check if NUMA operations are enabled
    pub fn is_enabled(&self) -> bool {
        !self.disabled.load(Ordering::Relaxed)
    }

    /// Get the current NUMA strategy
    pub fn strategy(&self) -> &NumaStrategy {
        &self.strategy
    }

    /// Get the detected NUMA topology (if available)
    pub fn topology(&self) -> Option<&NumaTopology> {
        self.topology.as_ref()
    }

    /// Disable NUMA operations (can be called by user configuration)
    pub fn disable(&self, reason: &str) {
        info!("Disabling NUMA operations: {}", reason);
        self.disabled.store(true, Ordering::Relaxed);
    }

    /// Get the optimal NUMA node for memory allocation
    pub fn get_optimal_node(&self, _thread_type: ThreadType) -> Option<NumaNode> {
        if !self.is_enabled() {
            return None;
        }

        let topology = self.topology.as_ref()?;

        // Simple round-robin allocation for now
        // In a more sophisticated implementation, you might consider:
        // - Current thread's CPU
        // - Memory pressure on different nodes
        // - Thread type (I/O vs compute)

        if !topology.nodes.is_empty() {
            Some(topology.nodes[0])
        } else {
            None
        }
    }

    /// Get recommended memory allocation size for NUMA-aware allocation
    pub fn get_recommended_allocation_size(&self, requested_size: usize) -> usize {
        if !self.is_enabled() {
            return requested_size;
        }

        // Align allocation to page boundaries for better NUMA performance
        // Use 2MB pages if available, otherwise 4KB pages
        let page_size = if self.supports_large_pages() {
            2 * 1024 * 1024 // 2MB
        } else {
            4 * 1024 // 4KB
        };

        // Round up to the next page boundary
        requested_size.div_ceil(page_size) * page_size
    }

    /// Check if large pages (2MB) are supported
    fn supports_large_pages(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/sys/kernel/mm/hugepages/hugepages-2048kB").exists()
        }

        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Get NUMA-aware configuration recommendations
    pub fn get_numa_config_recommendations(&self) -> NumaConfigRecommendations {
        let mut recommendations = NumaConfigRecommendations::default();

        if !self.is_enabled() {
            recommendations.disable_numa_allocation = true;
            recommendations.use_interleaved_allocation = false;
            recommendations.bind_to_node = None;

            match &self.strategy {
                NumaStrategy::DisabledCloud => {
                    recommendations.reason = "NUMA disabled for cloud environment".to_string();
                }
                NumaStrategy::DisabledUnsupported => {
                    recommendations.reason = "NUMA not supported on this hardware".to_string();
                }
                NumaStrategy::DisabledUser => {
                    recommendations.reason = "NUMA disabled by user configuration".to_string();
                }
                NumaStrategy::Fallback => {
                    recommendations.reason = "NUMA detection failed, using fallback".to_string();
                }
                _ => {
                    recommendations.reason = "NUMA disabled for unknown reason".to_string();
                }
            }

            return recommendations;
        }

        // NUMA is enabled, provide optimization recommendations
        if let Some(topology) = &self.topology {
            recommendations.disable_numa_allocation = false;
            recommendations.use_interleaved_allocation = topology.nodes.len() > 2;
            recommendations.bind_to_node = Some(topology.nodes[0]);
            recommendations.reason = format!(
                "NUMA enabled with {} nodes, {} total cores",
                topology.nodes.len(),
                topology
                    .cores_per_node
                    .values()
                    .map(|v| v.len())
                    .sum::<usize>()
            );
        }

        recommendations
    }
}

/// Thread types for NUMA node selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadType {
    /// I/O-bound threads
    Io,
    /// Compute-intensive threads
    Compute,
    /// Background/maintenance threads
    Background,
    /// Network communication threads
    Network,
}

/// NUMA configuration recommendations
#[derive(Debug, Clone)]
pub struct NumaConfigRecommendations {
    /// Whether to disable NUMA-specific memory allocation
    pub disable_numa_allocation: bool,
    /// Whether to use interleaved allocation across nodes
    pub use_interleaved_allocation: bool,
    /// Specific node to bind allocations to (if any)
    pub bind_to_node: Option<NumaNode>,
    /// Reason for the recommendations
    pub reason: String,
}

impl Default for NumaConfigRecommendations {
    fn default() -> Self {
        Self {
            disable_numa_allocation: true,
            use_interleaved_allocation: false,
            bind_to_node: None,
            reason: "Default conservative configuration".to_string(),
        }
    }
}

/// Create a NUMA manager with full environment detection
pub async fn create_numa_manager(
    platform_info: PlatformInfo,
    cloud_info: Option<CloudInstanceInfo>,
) -> Arc<NumaManager> {
    let manager = NumaManager::new(platform_info, cloud_info);
    Arc::new(manager)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::cloud_detection::{CloudInstanceInfo, CloudProvider};
    use crate::platform::detection::detect_platform;

    #[test]
    fn test_numa_manager_creation() {
        let platform_info = detect_platform();
        let manager = NumaManager::new(platform_info, None);

        // Should not panic and should have a valid strategy
        assert!(matches!(
            manager.strategy(),
            NumaStrategy::Enabled
                | NumaStrategy::DisabledUnsupported
                | NumaStrategy::DisabledCloud
                | NumaStrategy::Fallback
        ));
    }

    #[test]
    fn test_cloud_numa_disabling() {
        let platform_info = detect_platform();

        // Test Azure (should disable NUMA)
        let azure_info = CloudInstanceInfo {
            provider: CloudProvider::Azure,
            instance_type: Some("Standard_D4s_v3".to_string()),
            instance_id: None,
            availability_zone: None,
            region: None,
            metadata_available: false,
            dmi_detection: true,
        };

        let manager = NumaManager::new(platform_info.clone(), Some(azure_info));
        assert_eq!(manager.strategy(), &NumaStrategy::DisabledCloud);
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_aws_instance_type_numa() {
        let platform_info = detect_platform();

        // Test AWS burstable instance (should disable NUMA)
        let aws_burstable = CloudInstanceInfo {
            provider: CloudProvider::AWS,
            instance_type: Some("t3.medium".to_string()),
            instance_id: None,
            availability_zone: None,
            region: None,
            metadata_available: false,
            dmi_detection: true,
        };

        let manager = NumaManager::new(platform_info.clone(), Some(aws_burstable));
        assert_eq!(manager.strategy(), &NumaStrategy::DisabledCloud);

        // Test AWS compute optimized instance (might enable NUMA)
        let aws_compute = CloudInstanceInfo {
            provider: CloudProvider::AWS,
            instance_type: Some("c5.xlarge".to_string()),
            instance_id: None,
            availability_zone: None,
            region: None,
            metadata_available: false,
            dmi_detection: true,
        };

        let manager = NumaManager::new(platform_info, Some(aws_compute));
        // Strategy depends on actual NUMA hardware availability
        assert!(matches!(
            manager.strategy(),
            NumaStrategy::Enabled | NumaStrategy::DisabledUnsupported | NumaStrategy::Fallback
        ));
    }

    #[test]
    fn test_numa_config_recommendations() {
        let platform_info = detect_platform();
        let manager = NumaManager::new(platform_info, None);

        let recommendations = manager.get_numa_config_recommendations();

        // Should have valid recommendations
        assert!(!recommendations.reason.is_empty());

        if manager.is_enabled() {
            assert!(!recommendations.disable_numa_allocation);
        } else {
            assert!(recommendations.disable_numa_allocation);
        }
    }

    #[test]
    fn test_allocation_size_alignment() {
        let platform_info = detect_platform();
        let manager = NumaManager::new(platform_info, None);

        // Test various allocation sizes
        let test_sizes = [1024, 4096, 1048576, 2097152];

        for size in test_sizes {
            let recommended = manager.get_recommended_allocation_size(size);
            assert!(recommended >= size);

            if manager.is_enabled() {
                // Should be aligned to page boundaries
                assert_eq!(recommended % 4096, 0);
            }
        }
    }
}
