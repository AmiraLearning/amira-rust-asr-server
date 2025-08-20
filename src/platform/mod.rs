//! Platform detection and capability management.
//!
//! This module provides runtime detection of platform capabilities to enable
//! optimal performance features while maintaining compatibility across different
//! operating systems and hardware configurations.

pub mod affinity_management;
pub mod capabilities;
pub mod cloud_detection;
pub mod detection;
pub mod init;
pub mod io_backend;
pub mod numa_management;

pub use affinity_management::{create_affinity_manager, AffinityManager, AffinityStrategy};
pub use capabilities::{detect_capabilities, PlatformCapabilities};
pub use cloud_detection::{
    apply_cloud_config, detect_cloud_environment, generate_cloud_config, CloudConfig,
    CloudInstanceInfo, CloudProvider,
};
pub use detection::{detect_platform, is_io_uring_available, PlatformInfo};
pub use init::{initialize_platform, PlatformInit};
pub use io_backend::{create_optimal_io_backend, IoBackend};
pub use numa_management::{create_numa_manager, NumaManager, NumaStrategy};
