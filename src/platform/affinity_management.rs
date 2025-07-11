//! CPU affinity management with cloud environment awareness.
//!
//! This module provides intelligent CPU affinity management that automatically
//! disables optimizations in cloud environments where they may be
//! counterproductive or cause performance issues.

use core_affinity::{self, CoreId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use tracing::{debug, info, warn};

use crate::error::{AppError, Result};
use super::cloud_detection::{CloudInstanceInfo, CloudProvider};
use super::detection::PlatformInfo;
use crate::performance::affinity::{ThreadType, CpuSet};

/// CPU affinity management strategy
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityStrategy {
    /// CPU affinity is enabled and cores are assigned
    Enabled,
    /// CPU affinity is disabled due to cloud environment
    DisabledCloud,
    /// CPU affinity is disabled due to platform limitations
    DisabledPlatform,
    /// CPU affinity is disabled by user configuration
    DisabledUser,
    /// CPU affinity detection failed
    Fallback,
}

/// Intelligent CPU affinity manager
#[derive(Debug)]
pub struct AffinityManager {
    /// Current affinity strategy
    strategy: AffinityStrategy,
    /// Available CPU cores
    available_cores: Vec<CoreId>,
    /// Core assignments by thread type
    core_assignments: HashMap<ThreadType, CpuSet>,
    /// Whether affinity operations are globally disabled
    disabled: AtomicBool,
    /// Current assignment index for round-robin
    assignment_index: AtomicUsize,
    /// Cloud environment information
    cloud_info: Option<CloudInstanceInfo>,
    /// Platform information
    platform_info: PlatformInfo,
}

impl AffinityManager {
    /// Create a new affinity manager with environment detection
    pub fn new(platform_info: PlatformInfo, cloud_info: Option<CloudInstanceInfo>) -> Self {
        let mut manager = Self {
            strategy: AffinityStrategy::Fallback,
            available_cores: Vec::new(),
            core_assignments: HashMap::new(),
            disabled: AtomicBool::new(false),
            assignment_index: AtomicUsize::new(0),
            cloud_info,
            platform_info,
        };
        
        manager.initialize();
        manager
    }
    
    /// Initialize CPU affinity management based on environment detection
    fn initialize(&mut self) {
        info!("Initializing CPU affinity management...");
        
        // Check platform support first
        if !self.platform_supports_affinity() {
            self.strategy = AffinityStrategy::DisabledPlatform;
            self.disabled.store(true, Ordering::Relaxed);
            info!("CPU affinity disabled: platform does not support fine-grained affinity");
            return;
        }
        
        // Check if we're in a cloud environment that should disable affinity
        if let Some(cloud_info) = &self.cloud_info {
            if self.should_disable_affinity_for_cloud(cloud_info) {
                self.strategy = AffinityStrategy::DisabledCloud;
                self.disabled.store(true, Ordering::Relaxed);
                info!("CPU affinity disabled for cloud environment: {:?}", cloud_info.provider);
                return;
            }
        }
        
        // Try to get available cores
        match self.detect_available_cores() {
            Ok(cores) => {
                if cores.is_empty() {
                    warn!("No CPU cores detected for affinity assignment");
                    self.strategy = AffinityStrategy::Fallback;
                    self.disabled.store(true, Ordering::Relaxed);
                } else {
                    self.available_cores = cores;
                    self.setup_core_assignments();
                    self.strategy = AffinityStrategy::Enabled;
                    info!("CPU affinity enabled with {} cores", self.available_cores.len());
                }
            }
            Err(e) => {
                warn!("Failed to detect CPU cores: {}", e);
                self.strategy = AffinityStrategy::Fallback;
                self.disabled.store(true, Ordering::Relaxed);
            }
        }
    }
    
    /// Check if the platform supports CPU affinity
    fn platform_supports_affinity(&self) -> bool {
        match &self.platform_info.os {
            crate::platform::detection::OperatingSystem::Linux => true,
            crate::platform::detection::OperatingSystem::FreeBSD => true,
            crate::platform::detection::OperatingSystem::Windows => true,
            crate::platform::detection::OperatingSystem::MacOS => {
                // macOS doesn't support fine-grained CPU affinity
                false
            }
            crate::platform::detection::OperatingSystem::Other(_) => false,
        }
    }
    
    /// Check if CPU affinity should be disabled for the detected cloud environment
    fn should_disable_affinity_for_cloud(&self, cloud_info: &CloudInstanceInfo) -> bool {
        match cloud_info.provider {
            CloudProvider::Azure => {
                // Azure Hyper-V can cause issues with CPU affinity
                info!("Disabling CPU affinity for Azure Hyper-V environment");
                true
            }
            CloudProvider::AWS => {
                // AWS: disable for burstable and some shared instances
                if let Some(instance_type) = &cloud_info.instance_type {
                    let should_disable = instance_type.starts_with('t') ||  // Burstable instances
                                        instance_type.starts_with("a1") || // ARM Graviton
                                        instance_type.contains("nano") ||  // Very small instances
                                        instance_type.contains("micro");   // Micro instances
                    
                    if should_disable {
                        info!("Disabling CPU affinity for AWS instance type: {}", instance_type);
                    } else {
                        info!("Enabling CPU affinity for AWS instance type: {}", instance_type);
                    }
                    should_disable
                } else {
                    // Unknown instance type, be conservative
                    true
                }
            }
            CloudProvider::GCP => {
                // GCP: disable for shared-core instances
                if let Some(machine_type) = &cloud_info.instance_type {
                    let should_disable = machine_type.contains("e2-") ||    // Shared CPU instances
                                        machine_type.contains("f1-") ||    // Micro instances
                                        machine_type.contains("g1-") ||    // Small shared instances
                                        machine_type.ends_with("-micro") || 
                                        machine_type.ends_with("-small");
                    
                    if should_disable {
                        info!("Disabling CPU affinity for GCP machine type: {}", machine_type);
                    } else {
                        info!("Enabling CPU affinity for GCP machine type: {}", machine_type);
                    }
                    should_disable
                } else {
                    // Unknown machine type, be conservative
                    true
                }
            }
            CloudProvider::DigitalOcean => {
                // DigitalOcean droplets share physical cores
                info!("Disabling CPU affinity for DigitalOcean shared environment");
                true
            }
            CloudProvider::Linode => {
                // Linode instances may share cores depending on plan
                if let Some(instance_type) = &cloud_info.instance_type {
                    let is_dedicated = instance_type.contains("dedicated") || 
                                      instance_type.contains("high-cpu") ||
                                      instance_type.contains("high-memory");
                    if is_dedicated {
                        info!("Enabling CPU affinity for Linode dedicated instance");
                        false
                    } else {
                        info!("Disabling CPU affinity for Linode shared instance");
                        true
                    }
                } else {
                    true
                }
            }
            CloudProvider::Oracle => {
                // Oracle Cloud: enable for bare metal, disable for VM
                if let Some(instance_type) = &cloud_info.instance_type {
                    let is_bare_metal = instance_type.contains("BM") || 
                                       instance_type.contains("BareMetal");
                    if is_bare_metal {
                        info!("Enabling CPU affinity for Oracle bare metal instance");
                        false
                    } else {
                        info!("Disabling CPU affinity for Oracle VM instance");
                        true
                    }
                } else {
                    true
                }
            }
            CloudProvider::Vultr | CloudProvider::Unknown => {
                // Conservative approach for unknown or smaller providers
                true
            }
        }
    }
    
    /// Detect available CPU cores
    fn detect_available_cores(&self) -> Result<Vec<CoreId>> {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        
        if core_ids.is_empty() {
            return Err(AppError::Internal("No CPU cores detected".to_string()));
        }
        
        info!("Detected {} CPU cores for affinity assignment", core_ids.len());
        debug!("Available core IDs: {:?}", core_ids);
        
        Ok(core_ids)
    }
    
    /// Set up core assignments for different thread types
    fn setup_core_assignments(&mut self) {
        let total_cores = self.available_cores.len();
        
        if total_cores == 0 {
            return;
        }
        
        // Assign cores based on thread types and priorities
        // This is a simplified strategy - in production you might want more sophisticated allocation
        
        let (io_cores, inference_cores, background_cores, network_cores) = 
            self.calculate_core_distribution(total_cores);
        
        // Assign I/O cores (high priority)
        if !io_cores.is_empty() {
            let cpu_set = CpuSet::new(io_cores.clone());
            self.core_assignments.insert(ThreadType::Io, cpu_set);
            info!("Assigned {} cores for I/O threads", io_cores.len());
        }
        
        // Assign inference cores (highest priority)
        if !inference_cores.is_empty() {
            let cpu_set = CpuSet::new(inference_cores.clone());
            self.core_assignments.insert(ThreadType::Inference, cpu_set);
            info!("Assigned {} cores for inference threads", inference_cores.len());
        }
        
        // Assign network cores
        if !network_cores.is_empty() {
            let cpu_set = CpuSet::new(network_cores.clone());
            self.core_assignments.insert(ThreadType::Network, cpu_set);
            info!("Assigned {} cores for network threads", network_cores.len());
        }
        
        // Assign background cores (lowest priority)
        if !background_cores.is_empty() {
            let cpu_set = CpuSet::new(background_cores.clone());
            self.core_assignments.insert(ThreadType::Background, cpu_set);
            info!("Assigned {} cores for background threads", background_cores.len());
        }
    }
    
    /// Calculate optimal core distribution based on system resources
    fn calculate_core_distribution(&self, total_cores: usize) -> (Vec<CoreId>, Vec<CoreId>, Vec<CoreId>, Vec<CoreId>) {
        let cores = &self.available_cores;
        
        match total_cores {
            1 => {
                // Single core: everything shares
                let single_core = vec![cores[0]];
                (single_core.clone(), single_core.clone(), single_core.clone(), single_core)
            }
            2 => {
                // Two cores: inference gets dedicated core, everything else shares
                let inference_cores = vec![cores[0]];
                let shared_cores = vec![cores[1]];
                (shared_cores.clone(), inference_cores, shared_cores.clone(), shared_cores)
            }
            3..=4 => {
                // 3-4 cores: inference and I/O get dedicated cores
                let inference_cores = vec![cores[0]];
                let io_cores = vec![cores[1]];
                let remaining: Vec<CoreId> = cores[2..].to_vec();
                (io_cores, inference_cores, remaining.clone(), remaining)
            }
            5..=8 => {
                // 5-8 cores: more balanced distribution
                let inference_cores = cores[0..2].to_vec();
                let io_cores = vec![cores[2]];
                let network_cores = vec![cores[3]];
                let background_cores: Vec<CoreId> = cores[4..].to_vec();
                (io_cores, inference_cores, background_cores, network_cores)
            }
            _ => {
                // 8+ cores: sophisticated distribution
                let num_inference = (total_cores / 2).max(2);
                let num_io = (total_cores / 8).max(1);
                let num_network = (total_cores / 8).max(1);
                
                let inference_cores = cores[0..num_inference].to_vec();
                let io_cores = cores[num_inference..num_inference + num_io].to_vec();
                let network_cores = cores[num_inference + num_io..num_inference + num_io + num_network].to_vec();
                let background_cores = cores[num_inference + num_io + num_network..].to_vec();
                
                (io_cores, inference_cores, background_cores, network_cores)
            }
        }
    }
    
    /// Check if CPU affinity operations are enabled
    pub fn is_enabled(&self) -> bool {
        !self.disabled.load(Ordering::Relaxed)
    }
    
    /// Get the current affinity strategy
    pub fn strategy(&self) -> &AffinityStrategy {
        &self.strategy
    }
    
    /// Disable CPU affinity operations (can be called by user configuration)
    pub fn disable(&self, reason: &str) {
        info!("Disabling CPU affinity operations: {}", reason);
        self.disabled.store(true, Ordering::Relaxed);
    }
    
    /// Set CPU affinity for the current thread
    pub fn set_thread_affinity(&self, thread_type: ThreadType) -> Result<()> {
        if !self.is_enabled() {
            debug!("CPU affinity disabled, skipping affinity setting for {:?}", thread_type);
            return Ok(());
        }
        
        if let Some(cpu_set) = self.core_assignments.get(&thread_type) {
            if let Some(core_id) = cpu_set.next_core() {
                debug!("Setting thread affinity to core {:?} for thread type {:?}", core_id, thread_type);
                
                if !core_affinity::set_for_current(core_id) {
                    warn!("Failed to set CPU affinity to core {:?}", core_id);
                    return Err(AppError::Internal(format!(
                        "Failed to set CPU affinity to core {:?}",
                        core_id
                    )));
                }
                
                debug!("Successfully set CPU affinity to core {:?}", core_id);
                return Ok(());
            }
        }
        
        // Fallback: try to set affinity to any available core
        if !self.available_cores.is_empty() {
            let index = self.assignment_index.fetch_add(1, Ordering::Relaxed);
            let core_id = self.available_cores[index % self.available_cores.len()];
            
            debug!("Fallback: setting thread affinity to core {:?}", core_id);
            
            if !core_affinity::set_for_current(core_id) {
                warn!("Failed to set fallback CPU affinity to core {:?}", core_id);
                return Err(AppError::Internal(format!(
                    "Failed to set fallback CPU affinity to core {:?}",
                    core_id
                )));
            }
        }
        
        Ok(())
    }
    
    /// Get the assigned cores for a specific thread type
    pub fn get_assigned_cores(&self, thread_type: ThreadType) -> Option<&CpuSet> {
        self.core_assignments.get(&thread_type)
    }
    
    /// Get all available cores
    pub fn get_available_cores(&self) -> &[CoreId] {
        &self.available_cores
    }
    
    /// Get affinity configuration recommendations
    pub fn get_affinity_recommendations(&self) -> AffinityRecommendations {
        let mut recommendations = AffinityRecommendations::default();
        
        if !self.is_enabled() {
            recommendations.disable_affinity = true;
            recommendations.use_thread_pinning = false;
            
            match &self.strategy {
                AffinityStrategy::DisabledCloud => {
                    recommendations.reason = "CPU affinity disabled for cloud environment".to_string();
                }
                AffinityStrategy::DisabledPlatform => {
                    recommendations.reason = "CPU affinity not supported on this platform".to_string();
                }
                AffinityStrategy::DisabledUser => {
                    recommendations.reason = "CPU affinity disabled by user configuration".to_string();
                }
                AffinityStrategy::Fallback => {
                    recommendations.reason = "CPU affinity detection failed".to_string();
                }
                _ => {
                    recommendations.reason = "CPU affinity disabled for unknown reason".to_string();
                }
            }
            
            return recommendations;
        }
        
        // Affinity is enabled, provide optimization recommendations
        recommendations.disable_affinity = false;
        recommendations.use_thread_pinning = true;
        recommendations.numa_aware = self.available_cores.len() > 8; // Enable NUMA awareness for larger systems
        recommendations.isolation_cores = self.core_assignments.get(&ThreadType::Inference)
            .map(|cs| cs.len())
            .unwrap_or(0);
        recommendations.reason = format!(
            "CPU affinity enabled with {} cores, {} thread types configured",
            self.available_cores.len(),
            self.core_assignments.len()
        );
        
        recommendations
    }
}

/// CPU affinity configuration recommendations
#[derive(Debug, Clone)]
pub struct AffinityRecommendations {
    /// Whether to disable all CPU affinity operations
    pub disable_affinity: bool,
    /// Whether to use thread pinning to specific cores
    pub use_thread_pinning: bool,
    /// Whether to use NUMA-aware core assignment
    pub numa_aware: bool,
    /// Number of cores to isolate for inference threads
    pub isolation_cores: usize,
    /// Reason for the recommendations
    pub reason: String,
}

impl Default for AffinityRecommendations {
    fn default() -> Self {
        Self {
            disable_affinity: true,
            use_thread_pinning: false,
            numa_aware: false,
            isolation_cores: 0,
            reason: "Default conservative configuration".to_string(),
        }
    }
}

/// Create an affinity manager with full environment detection
pub async fn create_affinity_manager(
    platform_info: PlatformInfo,
    cloud_info: Option<CloudInstanceInfo>,
) -> Arc<AffinityManager> {
    let manager = AffinityManager::new(platform_info, cloud_info);
    Arc::new(manager)
}

/// Helper function to set affinity for the current thread
pub fn set_current_thread_affinity(
    manager: &AffinityManager,
    thread_type: ThreadType,
    thread_name: &str,
) -> Result<()> {
    // Set thread name for debugging
    if let Err(e) = thread::Builder::new().name(thread_name.to_string()).spawn(|| {}) {
        debug!("Failed to set thread name '{}': {}", thread_name, e);
    }
    
    // Set CPU affinity
    manager.set_thread_affinity(thread_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::detection::detect_platform;

    #[test]
    fn test_affinity_manager_creation() {
        let platform_info = detect_platform();
        let manager = AffinityManager::new(platform_info, None);
        
        // Should not panic and should have a valid strategy
        assert!(matches!(
            manager.strategy(),
            AffinityStrategy::Enabled | AffinityStrategy::DisabledPlatform | 
            AffinityStrategy::DisabledCloud | AffinityStrategy::Fallback
        ));
    }

    #[test]
    fn test_cloud_affinity_disabling() {
        let platform_info = detect_platform();
        
        // Test Azure (should disable affinity)
        let azure_info = CloudInstanceInfo {
            provider: CloudProvider::Azure,
            instance_type: Some("Standard_D4s_v3".to_string()),
            instance_id: None,
            availability_zone: None,
            region: None,
            metadata_available: false,
            dmi_detection: true,
        };
        
        let manager = AffinityManager::new(platform_info, Some(azure_info));
        assert_eq!(manager.strategy(), &AffinityStrategy::DisabledCloud);
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_core_distribution() {
        let platform_info = detect_platform();
        let manager = AffinityManager::new(platform_info, None);
        
        if manager.is_enabled() && !manager.available_cores.is_empty() {
            // Should have some core assignments
            assert!(!manager.core_assignments.is_empty());
            
            // Inference threads should always have cores assigned if enabled
            if let Some(inference_cores) = manager.get_assigned_cores(ThreadType::Inference) {
                assert!(!inference_cores.cores().is_empty());
            }
        }
    }

    #[test]
    fn test_affinity_recommendations() {
        let platform_info = detect_platform();
        let manager = AffinityManager::new(platform_info, None);
        
        let recommendations = manager.get_affinity_recommendations();
        
        // Should have valid recommendations
        assert!(!recommendations.reason.is_empty());
        
        if manager.is_enabled() {
            assert!(!recommendations.disable_affinity);
            assert!(recommendations.use_thread_pinning);
        } else {
            assert!(recommendations.disable_affinity);
            assert!(!recommendations.use_thread_pinning);
        }
    }

    #[test]
    fn test_aws_instance_type_affinity() {
        let platform_info = detect_platform();
        
        // Test AWS burstable instance (should disable affinity)
        let aws_burstable = CloudInstanceInfo {
            provider: CloudProvider::AWS,
            instance_type: Some("t3.medium".to_string()),
            instance_id: None,
            availability_zone: None,
            region: None,
            metadata_available: false,
            dmi_detection: true,
        };
        
        let manager = AffinityManager::new(platform_info.clone(), Some(aws_burstable));
        assert_eq!(manager.strategy(), &AffinityStrategy::DisabledCloud);
        
        // Test AWS compute optimized instance (might enable affinity)
        let aws_compute = CloudInstanceInfo {
            provider: CloudProvider::AWS,
            instance_type: Some("c5.xlarge".to_string()),
            instance_id: None,
            availability_zone: None,
            region: None,
            metadata_available: false,
            dmi_detection: true,
        };
        
        let manager = AffinityManager::new(platform_info, Some(aws_compute));
        // Strategy depends on platform support
        assert!(matches!(
            manager.strategy(),
            AffinityStrategy::Enabled | AffinityStrategy::DisabledPlatform | AffinityStrategy::Fallback
        ));
    }
}