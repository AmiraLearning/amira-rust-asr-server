//! Platform detection and capability management.
//!
//! This module provides runtime detection of platform capabilities to enable
//! optimal performance features while maintaining compatibility across different
//! operating systems and hardware configurations.

pub mod detection;
pub mod io_backend;
pub mod capabilities;
pub mod init;
pub mod cloud_detection;
pub mod numa_management;
pub mod affinity_management;

pub use detection::{PlatformInfo, detect_platform, is_io_uring_available};
pub use io_backend::{IoBackend, create_optimal_io_backend};
pub use capabilities::{PlatformCapabilities, detect_capabilities};
pub use init::{PlatformInit, initialize_platform};
pub use cloud_detection::{CloudProvider, CloudInstanceInfo, CloudConfig, detect_cloud_environment, generate_cloud_config, apply_cloud_config};
pub use numa_management::{NumaManager, NumaStrategy, create_numa_manager};
pub use affinity_management::{AffinityManager, AffinityStrategy, create_affinity_manager};