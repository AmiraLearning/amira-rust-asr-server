//! Platform detection utilities for runtime capability assessment.

use std::fs;
use std::process::Command;
use tracing::{debug, info, warn};

/// Information about the current platform and its capabilities
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Operating system type
    pub os: OperatingSystem,
    /// Kernel version (Linux only)
    pub kernel_version: Option<KernelVersion>,
    /// CPU architecture
    pub architecture: Architecture,
    /// Detected virtualization environment
    pub virtualization: VirtualizationEnvironment,
    /// Available I/O backends
    pub io_backends: Vec<IoBackendType>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperatingSystem {
    Linux,
    MacOS,
    Windows,
    FreeBSD,
    Other(String),
}

#[derive(Debug, Clone)]
pub struct KernelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Architecture {
    X86_64,
    AArch64,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum VirtualizationEnvironment {
    BareMetal,
    AWS,
    GCP,
    Azure,
    VMware,
    Docker,
    Other(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum IoBackendType {
    IoUring,
    Epoll,
    Kqueue,
    Select,
}

impl KernelVersion {
    /// Check if this kernel version supports io_uring (requires >= 5.1)
    pub fn supports_io_uring(&self) -> bool {
        self.major > 5 || (self.major == 5 && self.minor >= 1)
    }

    /// Check if this kernel version has stable io_uring (requires >= 5.4)
    pub fn has_stable_io_uring(&self) -> bool {
        self.major > 5 || (self.major == 5 && self.minor >= 4)
    }
}

/// Detect the current platform and its capabilities
pub fn detect_platform() -> PlatformInfo {
    info!("Detecting platform capabilities...");

    let os = detect_operating_system();
    let kernel_version = if matches!(os, OperatingSystem::Linux) {
        detect_kernel_version()
    } else {
        None
    };
    let architecture = detect_architecture();
    let virtualization = detect_virtualization_environment();
    let io_backends = detect_available_io_backends(&os, &kernel_version);

    let platform_info = PlatformInfo {
        os,
        kernel_version,
        architecture,
        virtualization,
        io_backends,
    };

    info!("Platform detection complete: {:?}", platform_info);
    platform_info
}

/// Check if io_uring is available and recommended on this platform
pub fn is_io_uring_available() -> bool {
    let platform = detect_platform();

    // Only available on Linux
    if !matches!(platform.os, OperatingSystem::Linux) {
        debug!("io_uring not available: not running on Linux");
        return false;
    }

    // Check kernel version
    if let Some(kernel) = &platform.kernel_version {
        if !kernel.supports_io_uring() {
            debug!(
                "io_uring not available: kernel version {} too old (requires >= 5.1)",
                format_kernel_version(kernel)
            );
            return false;
        }

        if !kernel.has_stable_io_uring() {
            warn!(
                "io_uring available but unstable: kernel version {} (stable >= 5.4)",
                format_kernel_version(kernel)
            );
        }
    }

    // Check if it's disabled in virtualized environments
    match platform.virtualization {
        VirtualizationEnvironment::AWS
        | VirtualizationEnvironment::GCP
        | VirtualizationEnvironment::Azure => {
            debug!(
                "io_uring available but may have limited performance in cloud environment: {:?}",
                platform.virtualization
            );
            // Still return true but with warning - let configuration decide
        }
        VirtualizationEnvironment::Docker => {
            debug!("io_uring available in Docker, performance depends on host kernel");
        }
        _ => {}
    }

    // Final check: try to detect if io_uring is actually available
    check_io_uring_runtime_availability()
}

fn detect_operating_system() -> OperatingSystem {
    match std::env::consts::OS {
        "linux" => OperatingSystem::Linux,
        "macos" => OperatingSystem::MacOS,
        "windows" => OperatingSystem::Windows,
        "freebsd" => OperatingSystem::FreeBSD,
        other => OperatingSystem::Other(other.to_string()),
    }
}

fn detect_kernel_version() -> Option<KernelVersion> {
    // Try reading from /proc/version first
    if let Ok(version_info) = fs::read_to_string("/proc/version") {
        if let Some(version) = parse_kernel_version_from_proc(&version_info) {
            return Some(version);
        }
    }

    // Fallback to uname command
    if let Ok(output) = Command::new("uname").arg("-r").output() {
        if let Ok(version_str) = String::from_utf8(output.stdout) {
            return parse_kernel_version_string(version_str.trim());
        }
    }

    warn!("Could not detect kernel version");
    None
}

fn parse_kernel_version_from_proc(proc_version: &str) -> Option<KernelVersion> {
    // Example: "Linux version 5.15.0-72-generic (buildd@lcy02-amd64-044) ..."
    let parts: Vec<&str> = proc_version.split_whitespace().collect();
    if parts.len() >= 3 && parts[0] == "Linux" && parts[1] == "version" {
        return parse_kernel_version_string(parts[2]);
    }
    None
}

fn parse_kernel_version_string(version_str: &str) -> Option<KernelVersion> {
    // Parse version string like "5.15.0-72-generic" or "5.15.0"
    let version_part = version_str.split('-').next()?;
    let parts: Vec<&str> = version_part.split('.').collect();

    if parts.len() >= 2 {
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = if parts.len() >= 3 {
            parts[2].parse().unwrap_or(0)
        } else {
            0
        };

        Some(KernelVersion {
            major,
            minor,
            patch,
        })
    } else {
        None
    }
}

fn detect_architecture() -> Architecture {
    match std::env::consts::ARCH {
        "x86_64" => Architecture::X86_64,
        "aarch64" => Architecture::AArch64,
        other => Architecture::Other(other.to_string()),
    }
}

fn detect_virtualization_environment() -> VirtualizationEnvironment {
    // Check DMI information for hypervisor signatures
    if let Ok(dmi_info) = fs::read_to_string("/sys/class/dmi/id/product_name") {
        let dmi_lower = dmi_info.to_lowercase();
        if dmi_lower.contains("ec2") || dmi_lower.contains("amazon") {
            return VirtualizationEnvironment::AWS;
        }
        if dmi_lower.contains("google") || dmi_lower.contains("gce") {
            return VirtualizationEnvironment::GCP;
        }
        if dmi_lower.contains("microsoft") || dmi_lower.contains("azure") {
            return VirtualizationEnvironment::Azure;
        }
        if dmi_lower.contains("vmware") {
            return VirtualizationEnvironment::VMware;
        }
    }

    // Check for container environments
    if fs::metadata("/.dockerenv").is_ok() {
        return VirtualizationEnvironment::Docker;
    }

    // Check cgroup information for container detection
    if let Ok(cgroup_info) = fs::read_to_string("/proc/1/cgroup") {
        if cgroup_info.contains("docker") {
            return VirtualizationEnvironment::Docker;
        }
    }

    // Check CPUID for hypervisor bit (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    {
        if is_hypervisor_present() {
            return VirtualizationEnvironment::Other("unknown_hypervisor".to_string());
        }
    }

    VirtualizationEnvironment::BareMetal
}

#[cfg(target_arch = "x86_64")]
fn is_hypervisor_present() -> bool {
    use std::arch::x86_64::__cpuid;

    // CPUID leaf 1, ECX bit 31 indicates hypervisor present
    unsafe {
        let cpuid_result = __cpuid(1);
        (cpuid_result.ecx & (1 << 31)) != 0
    }
}

fn detect_available_io_backends(
    os: &OperatingSystem,
    kernel_version: &Option<KernelVersion>,
) -> Vec<IoBackendType> {
    let mut backends = Vec::new();

    match os {
        OperatingSystem::Linux => {
            // Check for io_uring support
            if let Some(kernel) = kernel_version {
                if kernel.supports_io_uring() && check_io_uring_runtime_availability() {
                    backends.push(IoBackendType::IoUring);
                }
            }
            // Epoll is always available on Linux
            backends.push(IoBackendType::Epoll);
        }
        OperatingSystem::MacOS | OperatingSystem::FreeBSD => {
            // Kqueue is available on BSD-based systems
            backends.push(IoBackendType::Kqueue);
        }
        OperatingSystem::Windows => {
            // Windows has its own mechanisms, but we'll use select as fallback
            backends.push(IoBackendType::Select);
        }
        OperatingSystem::Other(_) => {
            // Fallback to select for unknown systems
            backends.push(IoBackendType::Select);
        }
    }

    // Select is always available as ultimate fallback
    if !backends.contains(&IoBackendType::Select) {
        backends.push(IoBackendType::Select);
    }

    backends
}

fn check_io_uring_runtime_availability() -> bool {
    // Try to create a simple io_uring instance to verify availability
    #[cfg(target_os = "linux")]
    {
        // This is a simple check - in a real implementation we'd use io_uring crate
        // For now, we'll check for the existence of io_uring related files
        std::path::Path::new("/proc/sys/kernel/io_uring_disabled").exists()
            || std::path::Path::new("/sys/kernel/debug/io_uring").exists()
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

fn format_kernel_version(version: &KernelVersion) -> String {
    format!("{}.{}.{}", version.major, version.minor, version.patch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_version_parsing() {
        let version = parse_kernel_version_string("5.15.0-72-generic").unwrap();
        assert_eq!(version.major, 5);
        assert_eq!(version.minor, 15);
        assert_eq!(version.patch, 0);
        assert!(version.supports_io_uring());
        assert!(version.has_stable_io_uring());
    }

    #[test]
    fn test_old_kernel_version() {
        let version = parse_kernel_version_string("4.19.0").unwrap();
        assert_eq!(version.major, 4);
        assert_eq!(version.minor, 19);
        assert!(!version.supports_io_uring());
    }

    #[test]
    fn test_platform_detection() {
        let platform = detect_platform();
        // This should not panic and should detect something reasonable
        assert!(!platform.io_backends.is_empty());
    }
}
