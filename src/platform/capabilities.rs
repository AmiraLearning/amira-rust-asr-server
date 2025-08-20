//! Platform capability detection and feature flags.

use std::collections::HashMap;
use tracing::{debug, info};

use super::detection::{detect_platform, OperatingSystem, PlatformInfo, VirtualizationEnvironment};

/// Comprehensive platform capabilities assessment
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// I/O capabilities
    pub io: IoCapabilities,
    /// CPU and architecture capabilities
    pub cpu: CpuCapabilities,
    /// Memory management capabilities
    pub memory: MemoryCapabilities,
    /// Networking capabilities
    pub network: NetworkCapabilities,
    /// Security capabilities
    pub security: SecurityCapabilities,
    /// Feature flags based on detected capabilities
    pub feature_flags: HashMap<String, bool>,
}

#[derive(Debug, Clone)]
pub struct IoCapabilities {
    /// io_uring support level
    pub io_uring_support: IoUringSupport,
    /// Available I/O multiplexing mechanisms
    pub multiplexing: Vec<IoMultiplexing>,
    /// Direct I/O support
    pub direct_io: bool,
    /// Memory-mapped I/O support
    pub mmap_io: bool,
    /// Asynchronous I/O support
    pub async_io: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IoUringSupport {
    None,
    Basic,    // Kernel >= 5.1, basic functionality
    Stable,   // Kernel >= 5.4, production ready
    Advanced, // Kernel >= 5.6, advanced features
    Full,     // Kernel >= 5.10, all features
}

#[derive(Debug, Clone, PartialEq)]
pub enum IoMultiplexing {
    Select,
    Poll,
    Epoll,
    Kqueue,
    IoUring,
    WindowsIOCP,
}

#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    /// SIMD instruction sets available
    pub simd: Vec<SimdInstructionSet>,
    /// Number of CPU cores
    pub cores: usize,
    /// NUMA topology
    pub numa: NumaCapabilities,
    /// CPU affinity support
    pub affinity_support: bool,
    /// Hardware threading (SMT/Hyperthreading)
    pub hardware_threading: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimdInstructionSet {
    SSE,
    SSE2,
    SSE3,
    SSSE3,
    SSE41,
    SSE42,
    AVX,
    AVX2,
    AVX512F,
    AVX512BW,
    AVX512DQ,
    NEON, // ARM
}

#[derive(Debug, Clone)]
pub struct NumaCapabilities {
    /// NUMA is available and functional
    pub available: bool,
    /// Number of NUMA nodes
    pub nodes: usize,
    /// Can bind memory to specific nodes
    pub memory_binding: bool,
    /// Can migrate memory between nodes
    pub memory_migration: bool,
    /// Topology detection is reliable
    pub topology_reliable: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryCapabilities {
    /// Large page support
    pub large_pages: LargePageSupport,
    /// Memory locking capabilities
    pub memory_locking: bool,
    /// NUMA memory allocation
    pub numa_allocation: bool,
    /// Memory-mapped files
    pub mmap_support: bool,
    /// Zero-copy networking
    pub zero_copy_networking: bool,
}

#[derive(Debug, Clone)]
pub struct LargePageSupport {
    /// 2MB pages supported
    pub pages_2mb: bool,
    /// 1GB pages supported  
    pub pages_1gb: bool,
    /// Transparent huge pages enabled
    pub transparent_huge_pages: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkCapabilities {
    /// TCP_NODELAY support
    pub tcp_nodelay: bool,
    /// SO_REUSEPORT support
    pub so_reuseport: bool,
    /// TCP_FASTOPEN support
    pub tcp_fastopen: bool,
    /// Zero-copy networking (sendfile, splice)
    pub zero_copy: bool,
    /// Kernel bypass networking
    pub kernel_bypass: bool,
    /// IPv6 support
    pub ipv6: bool,
}

#[derive(Debug, Clone)]
pub struct SecurityCapabilities {
    /// Address Space Layout Randomization
    pub aslr: bool,
    /// Control Flow Integrity
    pub cfi: bool,
    /// Stack protection
    pub stack_protection: bool,
    /// Fortify source
    pub fortify_source: bool,
    /// Capability-based security
    pub capabilities: bool,
    /// Seccomp filtering
    pub seccomp: bool,
}

/// Detect comprehensive platform capabilities
pub fn detect_capabilities() -> PlatformCapabilities {
    info!("Detecting comprehensive platform capabilities...");

    let platform_info = detect_platform();

    let io = detect_io_capabilities(&platform_info);
    let cpu = detect_cpu_capabilities(&platform_info);
    let memory = detect_memory_capabilities(&platform_info);
    let network = detect_network_capabilities(&platform_info);
    let security = detect_security_capabilities(&platform_info);
    let feature_flags = generate_feature_flags(&platform_info, &io, &cpu, &memory, &network);

    let capabilities = PlatformCapabilities {
        io,
        cpu,
        memory,
        network,
        security,
        feature_flags,
    };

    info!("Platform capability detection complete");
    debug!("Detected capabilities: {:#?}", capabilities);

    capabilities
}

fn detect_io_capabilities(platform: &PlatformInfo) -> IoCapabilities {
    let io_uring_support = if matches!(platform.os, OperatingSystem::Linux) {
        if let Some(kernel) = &platform.kernel_version {
            if kernel.major > 5 || (kernel.major == 5 && kernel.minor >= 10) {
                IoUringSupport::Full
            } else if kernel.major == 5 && kernel.minor >= 6 {
                IoUringSupport::Advanced
            } else if kernel.major == 5 && kernel.minor >= 4 {
                IoUringSupport::Stable
            } else if kernel.major == 5 && kernel.minor >= 1 {
                IoUringSupport::Basic
            } else {
                IoUringSupport::None
            }
        } else {
            IoUringSupport::None
        }
    } else {
        IoUringSupport::None
    };

    let mut multiplexing = Vec::new();
    match platform.os {
        OperatingSystem::Linux => {
            multiplexing.extend([
                IoMultiplexing::Select,
                IoMultiplexing::Poll,
                IoMultiplexing::Epoll,
            ]);
            if !matches!(io_uring_support, IoUringSupport::None) {
                multiplexing.push(IoMultiplexing::IoUring);
            }
        }
        OperatingSystem::MacOS | OperatingSystem::FreeBSD => {
            multiplexing.extend([
                IoMultiplexing::Select,
                IoMultiplexing::Poll,
                IoMultiplexing::Kqueue,
            ]);
        }
        OperatingSystem::Windows => {
            multiplexing.extend([IoMultiplexing::Select, IoMultiplexing::WindowsIOCP]);
        }
        OperatingSystem::Other(_) => {
            multiplexing.push(IoMultiplexing::Select);
        }
    }

    IoCapabilities {
        io_uring_support,
        multiplexing,
        direct_io: matches!(
            platform.os,
            OperatingSystem::Linux | OperatingSystem::FreeBSD
        ),
        mmap_io: true, // Available on all modern platforms
        async_io: true,
    }
}

fn detect_cpu_capabilities(platform: &PlatformInfo) -> CpuCapabilities {
    let simd = detect_simd_capabilities(platform);
    let cores = num_cpus::get();
    let numa = detect_numa_capabilities(platform);

    // CPU affinity support varies by platform
    let affinity_support = match platform.os {
        OperatingSystem::Linux | OperatingSystem::FreeBSD => true,
        OperatingSystem::MacOS => false, // macOS doesn't support fine-grained CPU affinity
        OperatingSystem::Windows => true, // Windows supports processor affinity
        OperatingSystem::Other(_) => false,
    };

    CpuCapabilities {
        simd,
        cores,
        numa,
        affinity_support,
        hardware_threading: cores > num_cpus::get_physical(),
    }
}

fn detect_simd_capabilities(platform: &PlatformInfo) -> Vec<SimdInstructionSet> {
    let mut simd = Vec::new();

    match platform.architecture {
        super::detection::Architecture::X86_64 => {
            // Use runtime CPU feature detection
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("sse") {
                    simd.push(SimdInstructionSet::SSE);
                }
                if std::is_x86_feature_detected!("sse2") {
                    simd.push(SimdInstructionSet::SSE2);
                }
                if std::is_x86_feature_detected!("sse3") {
                    simd.push(SimdInstructionSet::SSE3);
                }
                if std::is_x86_feature_detected!("ssse3") {
                    simd.push(SimdInstructionSet::SSSE3);
                }
                if std::is_x86_feature_detected!("sse4.1") {
                    simd.push(SimdInstructionSet::SSE41);
                }
                if std::is_x86_feature_detected!("sse4.2") {
                    simd.push(SimdInstructionSet::SSE42);
                }
                if std::is_x86_feature_detected!("avx") {
                    simd.push(SimdInstructionSet::AVX);
                }
                if std::is_x86_feature_detected!("avx2") {
                    simd.push(SimdInstructionSet::AVX2);
                }
                if std::is_x86_feature_detected!("avx512f") {
                    simd.push(SimdInstructionSet::AVX512F);
                }
                if std::is_x86_feature_detected!("avx512bw") {
                    simd.push(SimdInstructionSet::AVX512BW);
                }
                if std::is_x86_feature_detected!("avx512dq") {
                    simd.push(SimdInstructionSet::AVX512DQ);
                }
            }
        }
        super::detection::Architecture::AArch64 => {
            // ARM NEON is standard on AArch64
            simd.push(SimdInstructionSet::NEON);
        }
        super::detection::Architecture::Other(_) => {
            // No SIMD capabilities assumed for unknown architectures
        }
    }

    simd
}

fn detect_numa_capabilities(platform: &PlatformInfo) -> NumaCapabilities {
    // NUMA is primarily useful on bare metal servers
    let numa_likely_useful = matches!(
        platform.virtualization,
        VirtualizationEnvironment::BareMetal
    ) && num_cpus::get() > 8; // Likely multi-socket if > 8 cores

    let available = numa_likely_useful && matches!(platform.os, OperatingSystem::Linux);

    // Detect actual NUMA topology if available
    let (nodes, topology_reliable) = if available {
        detect_numa_topology()
    } else {
        (1, false)
    };

    NumaCapabilities {
        available,
        nodes,
        memory_binding: available,
        memory_migration: available,
        topology_reliable,
    }
}

fn detect_numa_topology() -> (usize, bool) {
    #[cfg(target_os = "linux")]
    {
        // Try to read NUMA topology from /sys/devices/system/node/
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node/") {
            let node_count = entries
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.file_name().to_string_lossy().starts_with("node"))
                .count();

            if node_count > 1 {
                return (node_count, true);
            }
        }
    }

    (1, false)
}

fn detect_memory_capabilities(platform: &PlatformInfo) -> MemoryCapabilities {
    let large_pages = detect_large_page_support(platform);

    MemoryCapabilities {
        large_pages,
        memory_locking: matches!(
            platform.os,
            OperatingSystem::Linux | OperatingSystem::FreeBSD
        ),
        numa_allocation: matches!(platform.os, OperatingSystem::Linux),
        mmap_support: true, // Available on all modern platforms
        zero_copy_networking: matches!(platform.os, OperatingSystem::Linux),
    }
}

fn detect_large_page_support(platform: &PlatformInfo) -> LargePageSupport {
    match platform.os {
        OperatingSystem::Linux => {
            // Check for huge page support
            let pages_2mb =
                std::path::Path::new("/sys/kernel/mm/hugepages/hugepages-2048kB").exists();
            let pages_1gb =
                std::path::Path::new("/sys/kernel/mm/hugepages/hugepages-1048576kB").exists();
            let transparent_huge_pages =
                std::path::Path::new("/sys/kernel/mm/transparent_hugepage").exists();

            LargePageSupport {
                pages_2mb,
                pages_1gb,
                transparent_huge_pages,
            }
        }
        _ => LargePageSupport {
            pages_2mb: false,
            pages_1gb: false,
            transparent_huge_pages: false,
        },
    }
}

fn detect_network_capabilities(platform: &PlatformInfo) -> NetworkCapabilities {
    NetworkCapabilities {
        tcp_nodelay: true, // Available on all platforms
        so_reuseport: matches!(
            platform.os,
            OperatingSystem::Linux | OperatingSystem::FreeBSD
        ),
        tcp_fastopen: matches!(platform.os, OperatingSystem::Linux),
        zero_copy: matches!(platform.os, OperatingSystem::Linux),
        kernel_bypass: false, // Requires special hardware/drivers
        ipv6: true,           // Standard on modern platforms
    }
}

fn detect_security_capabilities(platform: &PlatformInfo) -> SecurityCapabilities {
    match platform.os {
        OperatingSystem::Linux => SecurityCapabilities {
            aslr: true,
            cfi: true, // Modern distros enable this
            stack_protection: true,
            fortify_source: true,
            capabilities: true,
            seccomp: true,
        },
        OperatingSystem::MacOS => SecurityCapabilities {
            aslr: true,
            cfi: true,
            stack_protection: true,
            fortify_source: true,
            capabilities: false, // macOS uses different security model
            seccomp: false,
        },
        OperatingSystem::Windows => SecurityCapabilities {
            aslr: true,
            cfi: true,
            stack_protection: true,
            fortify_source: false,
            capabilities: false,
            seccomp: false,
        },
        _ => SecurityCapabilities {
            aslr: false,
            cfi: false,
            stack_protection: false,
            fortify_source: false,
            capabilities: false,
            seccomp: false,
        },
    }
}

fn generate_feature_flags(
    platform: &PlatformInfo,
    io: &IoCapabilities,
    cpu: &CpuCapabilities,
    memory: &MemoryCapabilities,
    network: &NetworkCapabilities,
) -> HashMap<String, bool> {
    let mut flags = HashMap::new();

    // I/O feature flags
    flags.insert(
        "io_uring".to_string(),
        !matches!(io.io_uring_support, IoUringSupport::None),
    );
    flags.insert(
        "io_uring_stable".to_string(),
        matches!(
            io.io_uring_support,
            IoUringSupport::Stable | IoUringSupport::Advanced | IoUringSupport::Full
        ),
    );
    flags.insert(
        "epoll".to_string(),
        io.multiplexing.contains(&IoMultiplexing::Epoll),
    );
    flags.insert(
        "kqueue".to_string(),
        io.multiplexing.contains(&IoMultiplexing::Kqueue),
    );

    // CPU feature flags
    flags.insert(
        "simd_avx2".to_string(),
        cpu.simd.contains(&SimdInstructionSet::AVX2),
    );
    flags.insert(
        "simd_avx512".to_string(),
        cpu.simd.contains(&SimdInstructionSet::AVX512F),
    );
    flags.insert("cpu_affinity".to_string(), cpu.affinity_support);
    flags.insert("numa".to_string(), cpu.numa.available);
    flags.insert("numa_reliable".to_string(), cpu.numa.topology_reliable);

    // Memory feature flags
    flags.insert("large_pages_2mb".to_string(), memory.large_pages.pages_2mb);
    flags.insert("large_pages_1gb".to_string(), memory.large_pages.pages_1gb);
    flags.insert(
        "transparent_huge_pages".to_string(),
        memory.large_pages.transparent_huge_pages,
    );
    flags.insert("numa_allocation".to_string(), memory.numa_allocation);

    // Network feature flags
    flags.insert("so_reuseport".to_string(), network.so_reuseport);
    flags.insert("tcp_fastopen".to_string(), network.tcp_fastopen);
    flags.insert("zero_copy_networking".to_string(), network.zero_copy);

    // Platform-specific flags
    flags.insert(
        "cloud_environment".to_string(),
        !matches!(
            platform.virtualization,
            VirtualizationEnvironment::BareMetal
        ),
    );
    flags.insert(
        "container".to_string(),
        matches!(platform.virtualization, VirtualizationEnvironment::Docker),
    );

    // Optimization recommendation flags
    flags.insert(
        "disable_numa_in_cloud".to_string(),
        !matches!(
            platform.virtualization,
            VirtualizationEnvironment::BareMetal
        ) && cpu.numa.available,
    );
    flags.insert(
        "prefer_epoll_over_io_uring".to_string(),
        matches!(
            platform.virtualization,
            VirtualizationEnvironment::AWS | VirtualizationEnvironment::GCP
        ),
    );

    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_detection() {
        let capabilities = detect_capabilities();

        // Basic sanity checks
        assert!(!capabilities.io.multiplexing.is_empty());
        assert!(capabilities.cpu.cores > 0);
        assert!(!capabilities.feature_flags.is_empty());
    }

    #[test]
    fn test_simd_detection() {
        let platform = detect_platform();
        let simd = detect_simd_capabilities(&platform);

        // On x86_64, we should at least have SSE2
        #[cfg(target_arch = "x86_64")]
        {
            assert!(simd.contains(&SimdInstructionSet::SSE2));
        }

        // On AArch64, we should have NEON
        #[cfg(target_arch = "aarch64")]
        {
            assert!(simd.contains(&SimdInstructionSet::NEON));
        }
    }

    #[test]
    fn test_feature_flags() {
        let capabilities = detect_capabilities();

        // Check that some basic flags are present
        assert!(capabilities.feature_flags.contains_key("simd_avx2"));
        assert!(capabilities.feature_flags.contains_key("numa"));
        assert!(capabilities.feature_flags.contains_key("cloud_environment"));
    }
}
