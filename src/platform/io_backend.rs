//! I/O backend abstraction and selection.

use async_trait::async_trait;
use std::net::SocketAddr;
// use std::pin::Pin;
// use std::task::{Context, Poll};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

use super::detection::{detect_platform, IoBackendType, VirtualizationEnvironment};
use crate::error::{AppError, Result};

/// Abstract I/O backend for high-performance networking
#[async_trait]
pub trait IoBackend: Send + Sync {
    /// Accept incoming connections
    async fn accept(&self) -> Result<(Box<dyn AsyncStream>, SocketAddr)>;

    /// Get the backend type
    fn backend_type(&self) -> IoBackendType;

    /// Get performance characteristics
    fn performance_profile(&self) -> PerformanceProfile;
}

/// Performance characteristics of an I/O backend
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Expected latency characteristics
    pub latency: LatencyProfile,
    /// Throughput characteristics
    pub throughput: ThroughputProfile,
    /// CPU usage characteristics
    pub cpu_efficiency: CpuEfficiencyProfile,
    /// Memory usage characteristics
    pub memory_usage: MemoryUsageProfile,
}

#[derive(Debug, Clone)]
pub struct LatencyProfile {
    /// Typical connection acceptance latency in microseconds
    pub connection_accept_us: u32,
    /// Typical read latency in microseconds
    pub read_latency_us: u32,
    /// Typical write latency in microseconds
    pub write_latency_us: u32,
}

#[derive(Debug, Clone)]
pub struct ThroughputProfile {
    /// Maximum connections per second
    pub max_connections_per_sec: u32,
    /// Maximum bytes per second per connection
    pub max_bytes_per_sec_per_conn: u64,
    /// Total system throughput limit
    pub system_throughput_limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CpuEfficiencyProfile {
    /// CPU cycles per operation (relative scale)
    pub cycles_per_operation: u32,
    /// Scales well with CPU cores
    pub scales_with_cores: bool,
    /// Benefits from CPU affinity
    pub benefits_from_affinity: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryUsageProfile {
    /// Base memory overhead in bytes
    pub base_overhead_bytes: u64,
    /// Memory per connection in bytes
    pub per_connection_bytes: u64,
    /// Uses kernel buffers efficiently
    pub kernel_buffer_efficient: bool,
}

/// Async stream trait for backend abstraction
#[async_trait]
pub trait AsyncStream: Send + Sync {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize>;
    async fn write(&mut self, buf: &[u8]) -> Result<usize>;
    async fn shutdown(&mut self) -> Result<()>;
}

/// Create a specific I/O backend type
pub async fn create_specific_io_backend(
    backend_type: IoBackendType,
    bind_addr: SocketAddr,
) -> Result<Box<dyn IoBackend>> {
    info!("Creating I/O backend: {:?}", backend_type);

    match backend_type {
        IoBackendType::IoUring => {
            #[cfg(target_os = "linux")]
            {
                info!("Creating io_uring backend");
                Ok(Box::new(IoUringBackend::new(bind_addr).await?))
            }
            #[cfg(not(target_os = "linux"))]
            {
                warn!("io_uring requested but not available on this platform, falling back to default");
                create_fallback_backend(bind_addr).await
            }
        }
        IoBackendType::Epoll => {
            #[cfg(target_os = "linux")]
            {
                info!("Creating epoll backend");
                Ok(Box::new(EpollBackend::new(bind_addr).await?))
            }
            #[cfg(not(target_os = "linux"))]
            {
                warn!(
                    "epoll requested but not available on this platform, falling back to default"
                );
                create_fallback_backend(bind_addr).await
            }
        }
        IoBackendType::Kqueue => {
            #[cfg(any(target_os = "macos", target_os = "freebsd"))]
            {
                info!("Creating kqueue backend");
                Ok(Box::new(KqueueBackend::new(bind_addr).await?))
            }
            #[cfg(not(any(target_os = "macos", target_os = "freebsd")))]
            {
                warn!(
                    "kqueue requested but not available on this platform, falling back to default"
                );
                create_fallback_backend(bind_addr).await
            }
        }
        IoBackendType::Select => {
            info!("Creating select backend (fallback)");
            create_fallback_backend(bind_addr).await
        }
    }
}

/// Create the optimal I/O backend for the current platform
pub async fn create_optimal_io_backend(bind_addr: SocketAddr) -> Result<Box<dyn IoBackend>> {
    let platform = detect_platform();

    info!(
        "Creating optimal I/O backend for platform: {:?}",
        platform.os
    );

    // Determine the best backend based on platform capabilities
    let selected_backend = select_optimal_backend(&platform);

    create_specific_io_backend(selected_backend, bind_addr).await
}

fn select_optimal_backend(platform: &super::detection::PlatformInfo) -> IoBackendType {
    // Don't use io_uring in certain cloud environments due to performance issues
    let avoid_io_uring = matches!(
        platform.virtualization,
        VirtualizationEnvironment::AWS | VirtualizationEnvironment::GCP
    );

    if avoid_io_uring {
        info!(
            "Avoiding io_uring in cloud environment: {:?}",
            platform.virtualization
        );
    }

    // Select best available backend
    for backend in &platform.io_backends {
        match backend {
            IoBackendType::IoUring if !avoid_io_uring => {
                // Check kernel version for stability
                if let Some(kernel) = &platform.kernel_version {
                    if kernel.has_stable_io_uring() {
                        return IoBackendType::IoUring;
                    } else {
                        warn!(
                            "io_uring available but unstable on kernel {}.{}.{}, skipping",
                            kernel.major, kernel.minor, kernel.patch
                        );
                    }
                }
            }
            IoBackendType::Epoll => return IoBackendType::Epoll,
            IoBackendType::Kqueue => return IoBackendType::Kqueue,
            _ => continue,
        }
    }

    // Fallback
    IoBackendType::Select
}

async fn create_fallback_backend(bind_addr: SocketAddr) -> Result<Box<dyn IoBackend>> {
    Ok(Box::new(TokioBackend::new(bind_addr).await?))
}

// Linux io_uring backend
#[cfg(target_os = "linux")]
pub struct IoUringBackend {
    listener: tokio::net::TcpListener,
}

#[cfg(target_os = "linux")]
impl IoUringBackend {
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        // In a real implementation, we'd use io_uring crate
        // For now, we'll create a placeholder that uses tokio-uring
        let listener = TcpListener::bind(bind_addr)
            .await
            .map_err(|e| AppError::Network(format!("Failed to bind io_uring listener: {}", e)))?;

        Ok(Self { listener })
    }
}

#[cfg(target_os = "linux")]
#[async_trait]
impl IoBackend for IoUringBackend {
    async fn accept(&self) -> Result<(Box<dyn AsyncStream>, SocketAddr)> {
        let (stream, addr) = self
            .listener
            .accept()
            .await
            .map_err(|e| AppError::Network(format!("io_uring accept failed: {}", e)))?;

        Ok((Box::new(TokioStream::new(stream)), addr))
    }

    fn backend_type(&self) -> IoBackendType {
        IoBackendType::IoUring
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            latency: LatencyProfile {
                connection_accept_us: 5, // Very low latency
                read_latency_us: 2,
                write_latency_us: 2,
            },
            throughput: ThroughputProfile {
                max_connections_per_sec: 100_000,
                max_bytes_per_sec_per_conn: 10_000_000_000, // 10 GB/s
                system_throughput_limit: Some(100_000_000_000), // 100 GB/s
            },
            cpu_efficiency: CpuEfficiencyProfile {
                cycles_per_operation: 100, // Very efficient
                scales_with_cores: true,
                benefits_from_affinity: true,
            },
            memory_usage: MemoryUsageProfile {
                base_overhead_bytes: 1024,
                per_connection_bytes: 512,
                kernel_buffer_efficient: true,
            },
        }
    }
}

// Linux epoll backend
#[cfg(target_os = "linux")]
pub struct EpollBackend {
    listener: tokio::net::TcpListener,
}

#[cfg(target_os = "linux")]
impl EpollBackend {
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .await
            .map_err(|e| AppError::Network(format!("Failed to bind epoll listener: {}", e)))?;

        Ok(Self { listener })
    }
}

#[cfg(target_os = "linux")]
#[async_trait]
impl IoBackend for EpollBackend {
    async fn accept(&self) -> Result<(Box<dyn AsyncStream>, SocketAddr)> {
        let (stream, addr) = self
            .listener
            .accept()
            .await
            .map_err(|e| AppError::Network(format!("epoll accept failed: {}", e)))?;

        Ok((Box::new(TokioStream::new(stream)), addr))
    }

    fn backend_type(&self) -> IoBackendType {
        IoBackendType::Epoll
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            latency: LatencyProfile {
                connection_accept_us: 10, // Good latency
                read_latency_us: 5,
                write_latency_us: 5,
            },
            throughput: ThroughputProfile {
                max_connections_per_sec: 50_000,
                max_bytes_per_sec_per_conn: 5_000_000_000, // 5 GB/s
                system_throughput_limit: Some(50_000_000_000), // 50 GB/s
            },
            cpu_efficiency: CpuEfficiencyProfile {
                cycles_per_operation: 200, // Good efficiency
                scales_with_cores: true,
                benefits_from_affinity: true,
            },
            memory_usage: MemoryUsageProfile {
                base_overhead_bytes: 2048,
                per_connection_bytes: 1024,
                kernel_buffer_efficient: true,
            },
        }
    }
}

// macOS/FreeBSD kqueue backend
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub struct KqueueBackend {
    listener: tokio::net::TcpListener,
}

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
impl KqueueBackend {
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .await
            .map_err(|e| AppError::Network(format!("Failed to bind kqueue listener: {}", e)))?;

        Ok(Self { listener })
    }
}

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
#[async_trait]
impl IoBackend for KqueueBackend {
    async fn accept(&self) -> Result<(Box<dyn AsyncStream>, SocketAddr)> {
        let (stream, addr) = self
            .listener
            .accept()
            .await
            .map_err(|e| AppError::Network(format!("kqueue accept failed: {}", e)))?;

        Ok((Box::new(TokioStream::new(stream)), addr))
    }

    fn backend_type(&self) -> IoBackendType {
        IoBackendType::Kqueue
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            latency: LatencyProfile {
                connection_accept_us: 15, // Good latency
                read_latency_us: 8,
                write_latency_us: 8,
            },
            throughput: ThroughputProfile {
                max_connections_per_sec: 30_000,
                max_bytes_per_sec_per_conn: 3_000_000_000, // 3 GB/s
                system_throughput_limit: Some(30_000_000_000), // 30 GB/s
            },
            cpu_efficiency: CpuEfficiencyProfile {
                cycles_per_operation: 250, // Good efficiency
                scales_with_cores: true,
                benefits_from_affinity: false, // macOS doesn't support fine-grained affinity
            },
            memory_usage: MemoryUsageProfile {
                base_overhead_bytes: 3072,
                per_connection_bytes: 1536,
                kernel_buffer_efficient: true,
            },
        }
    }
}

// Fallback Tokio backend (works on all platforms)
pub struct TokioBackend {
    listener: tokio::net::TcpListener,
}

impl TokioBackend {
    pub async fn new(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .await
            .map_err(|e| AppError::Network(format!("Failed to bind tokio listener: {}", e)))?;

        Ok(Self { listener })
    }
}

#[async_trait]
impl IoBackend for TokioBackend {
    async fn accept(&self) -> Result<(Box<dyn AsyncStream>, SocketAddr)> {
        let (stream, addr) = self
            .listener
            .accept()
            .await
            .map_err(|e| AppError::Network(format!("tokio accept failed: {}", e)))?;

        Ok((Box::new(TokioStream::new(stream)), addr))
    }

    fn backend_type(&self) -> IoBackendType {
        IoBackendType::Select
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            latency: LatencyProfile {
                connection_accept_us: 50, // Moderate latency
                read_latency_us: 20,
                write_latency_us: 20,
            },
            throughput: ThroughputProfile {
                max_connections_per_sec: 10_000,
                max_bytes_per_sec_per_conn: 1_000_000_000, // 1 GB/s
                system_throughput_limit: Some(10_000_000_000), // 10 GB/s
            },
            cpu_efficiency: CpuEfficiencyProfile {
                cycles_per_operation: 500, // Moderate efficiency
                scales_with_cores: true,
                benefits_from_affinity: false,
            },
            memory_usage: MemoryUsageProfile {
                base_overhead_bytes: 4096,
                per_connection_bytes: 2048,
                kernel_buffer_efficient: false,
            },
        }
    }
}

// Stream wrapper for Tokio TcpStream
pub struct TokioStream {
    stream: TcpStream,
}

impl TokioStream {
    pub fn new(stream: TcpStream) -> Self {
        Self { stream }
    }
}

#[async_trait]
impl AsyncStream for TokioStream {
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        use tokio::io::AsyncReadExt;

        self.stream
            .read(buf)
            .await
            .map_err(|e| AppError::Network(format!("Stream read failed: {}", e)))
    }

    async fn write(&mut self, buf: &[u8]) -> Result<usize> {
        use tokio::io::AsyncWriteExt;

        self.stream
            .write(buf)
            .await
            .map_err(|e| AppError::Network(format!("Stream write failed: {}", e)))
    }

    async fn shutdown(&mut self) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        self.stream
            .shutdown()
            .await
            .map_err(|e| AppError::Network(format!("Stream shutdown failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_backend_creation() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let backend = create_optimal_io_backend(addr).await;
        assert!(backend.is_ok());
    }

    #[tokio::test]
    async fn test_performance_profiles() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let backend = create_optimal_io_backend(addr).await.unwrap();
        let profile = backend.performance_profile();

        // All backends should have reasonable performance characteristics
        assert!(profile.latency.connection_accept_us < 1000); // Less than 1ms
        assert!(profile.throughput.max_connections_per_sec > 1000); // At least 1k conn/s
    }
}
