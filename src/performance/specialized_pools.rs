//! Specialized thread pools for different workload types.
//!
//! This module provides specialized thread pools optimized for different
//! types of work: I/O-bound, compute-bound inference, and network operations.

use super::affinity::{AffinityManager, ThreadType};
use std::future::Future;
use std::sync::Arc;
use tokio::runtime::{Builder, Runtime};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// Specialized executor for different types of workloads
pub struct SpecializedExecutor {
    /// Runtime for I/O operations (WebSocket, file I/O)
    io_runtime: Runtime,
    
    /// Runtime for compute-intensive inference operations
    inference_runtime: Runtime,
    
    /// Runtime for network operations (Triton gRPC)
    network_runtime: Runtime,
    
    /// Affinity manager for CPU binding
    affinity_manager: Arc<AffinityManager>,
}

impl SpecializedExecutor {
    /// Create a new specialized executor with optimal thread pool configuration
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let affinity_manager = Arc::new(AffinityManager::new());
        
        let io_threads = affinity_manager.recommended_thread_count(ThreadType::Io);
        let inference_threads = affinity_manager.recommended_thread_count(ThreadType::Inference);
        let network_threads = affinity_manager.recommended_thread_count(ThreadType::Network);
        
        info!(
            "Creating specialized thread pools - I/O: {}, Inference: {}, Network: {}",
            io_threads, inference_threads, network_threads
        );
        
        // Create I/O runtime - optimized for high concurrency
        let io_runtime = Builder::new_multi_thread()
            .worker_threads(io_threads)
            .thread_name("asr-io")
            .enable_all()
            .on_thread_start({
                let affinity_manager = Arc::clone(&affinity_manager);
                move || {
                    if let Err(e) = affinity_manager.set_thread_affinity(ThreadType::Io) {
                        warn!("Failed to set I/O thread affinity: {}", e);
                    }
                }
            })
            .build()?;
        
        // Create inference runtime - optimized for CPU-intensive work
        let inference_runtime = Builder::new_multi_thread()
            .worker_threads(inference_threads)
            .thread_name("asr-inference")
            .enable_time()
            .enable_io()
            .on_thread_start({
                let affinity_manager = Arc::clone(&affinity_manager);
                move || {
                    if let Err(e) = affinity_manager.set_thread_affinity(ThreadType::Inference) {
                        warn!("Failed to set inference thread affinity: {}", e);
                    }
                }
            })
            .build()?;
        
        // Create network runtime - optimized for network I/O
        let network_runtime = Builder::new_multi_thread()
            .worker_threads(network_threads)
            .thread_name("asr-network")
            .enable_all()
            .on_thread_start({
                let affinity_manager = Arc::clone(&affinity_manager);
                move || {
                    if let Err(e) = affinity_manager.set_thread_affinity(ThreadType::Network) {
                        warn!("Failed to set network thread affinity: {}", e);
                    }
                }
            })
            .build()?;
        
        Ok(Self {
            io_runtime,
            inference_runtime,
            network_runtime,
            affinity_manager,
        })
    }
    
    /// Spawn an I/O task (WebSocket handling, file operations)
    pub fn spawn_io<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.io_runtime.spawn(future)
    }
    
    /// Spawn an inference task (ASR processing, model inference)
    pub fn spawn_inference<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.inference_runtime.spawn(future)
    }
    
    /// Spawn a network task (Triton gRPC calls)
    pub fn spawn_network<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.network_runtime.spawn(future)
    }
    
    /// Block on an I/O future
    pub fn block_on_io<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.io_runtime.block_on(future)
    }
    
    /// Block on an inference future
    pub fn block_on_inference<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.inference_runtime.block_on(future)
    }
    
    /// Block on a network future
    pub fn block_on_network<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.network_runtime.block_on(future)
    }
    
    /// Get executor statistics
    pub fn stats(&self) -> ExecutorStats {
        ExecutorStats {
            io_threads: self.affinity_manager.recommended_thread_count(ThreadType::Io),
            inference_threads: self.affinity_manager.recommended_thread_count(ThreadType::Inference),
            network_threads: self.affinity_manager.recommended_thread_count(ThreadType::Network),
            affinity_supported: self.affinity_manager.is_affinity_supported(),
        }
    }
    
    /// Shutdown all runtimes gracefully
    pub fn shutdown(self) {
        debug!("Shutting down specialized executor");
        
        self.io_runtime.shutdown_background();
        self.inference_runtime.shutdown_background();
        self.network_runtime.shutdown_background();
    }
}

impl Default for SpecializedExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create specialized executor")
    }
}

/// Statistics about the specialized executor
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub io_threads: usize,
    pub inference_threads: usize,
    pub network_threads: usize,
    pub affinity_supported: bool,
}

impl std::fmt::Display for ExecutorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Executor Stats - I/O: {} threads, Inference: {} threads, Network: {} threads, Affinity: {}",
            self.io_threads,
            self.inference_threads,
            self.network_threads,
            self.affinity_supported
        )
    }
}

/// Specialized thread pool for inference operations
pub struct InferenceThreadPool {
    runtime: Runtime,
    affinity_manager: Arc<AffinityManager>,
}

impl InferenceThreadPool {
    /// Create a new inference thread pool
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let affinity_manager = Arc::new(AffinityManager::new());
        let thread_count = affinity_manager.recommended_thread_count(ThreadType::Inference);
        
        info!("Creating inference thread pool with {} threads", thread_count);
        
        let runtime = Builder::new_multi_thread()
            .worker_threads(thread_count)
            .thread_name("inference-worker")
            .enable_time()
            .enable_io()
            .on_thread_start({
                let affinity_manager = Arc::clone(&affinity_manager);
                move || {
                    if let Err(e) = affinity_manager.set_thread_affinity(ThreadType::Inference) {
                        warn!("Failed to set inference thread affinity: {}", e);
                    }
                }
            })
            .build()?;
        
        Ok(Self {
            runtime,
            affinity_manager,
        })
    }
    
    /// Spawn an inference task
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future)
    }
    
    /// Block on an inference future
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.runtime.block_on(future)
    }
    
    /// Get the number of worker threads
    pub fn thread_count(&self) -> usize {
        self.affinity_manager.recommended_thread_count(ThreadType::Inference)
    }
}

/// Specialized thread pool for I/O operations
pub struct IoThreadPool {
    runtime: Runtime,
    affinity_manager: Arc<AffinityManager>,
}

impl IoThreadPool {
    /// Create a new I/O thread pool
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let affinity_manager = Arc::new(AffinityManager::new());
        let thread_count = affinity_manager.recommended_thread_count(ThreadType::Io);
        
        info!("Creating I/O thread pool with {} threads", thread_count);
        
        let runtime = Builder::new_multi_thread()
            .worker_threads(thread_count)
            .thread_name("io-worker")
            .enable_all()
            .on_thread_start({
                let affinity_manager = Arc::clone(&affinity_manager);
                move || {
                    if let Err(e) = affinity_manager.set_thread_affinity(ThreadType::Io) {
                        warn!("Failed to set I/O thread affinity: {}", e);
                    }
                }
            })
            .build()?;
        
        Ok(Self {
            runtime,
            affinity_manager,
        })
    }
    
    /// Spawn an I/O task
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.runtime.spawn(future)
    }
    
    /// Block on an I/O future
    pub fn block_on<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        self.runtime.block_on(future)
    }
    
    /// Get the number of worker threads
    pub fn thread_count(&self) -> usize {
        self.affinity_manager.recommended_thread_count(ThreadType::Io)
    }
}

/// Global specialized executor instance
static GLOBAL_EXECUTOR: once_cell::sync::Lazy<SpecializedExecutor> =
    once_cell::sync::Lazy::new(|| {
        SpecializedExecutor::new().expect("Failed to create global specialized executor")
    });

/// Get access to the global specialized executor
pub fn global_executor() -> &'static SpecializedExecutor {
    &GLOBAL_EXECUTOR
}

/// Spawn an I/O task on the global executor
pub fn spawn_io<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    global_executor().spawn_io(future)
}

/// Spawn an inference task on the global executor
pub fn spawn_inference<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    global_executor().spawn_inference(future)
}

/// Spawn a network task on the global executor
pub fn spawn_network<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    global_executor().spawn_network(future)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_specialized_executor() {
        let executor = SpecializedExecutor::new().unwrap();
        
        // Test I/O task
        let io_handle = executor.spawn_io(async {
            sleep(Duration::from_millis(10)).await;
            "io_result"
        });
        
        // Test inference task
        let inference_handle = executor.spawn_inference(async {
            // Simulate compute-intensive work
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });
        
        // Test network task
        let network_handle = executor.spawn_network(async {
            sleep(Duration::from_millis(5)).await;
            42
        });
        
        // Wait for all tasks
        let io_result = io_handle.await.unwrap();
        let inference_result = inference_handle.await.unwrap();
        let network_result = network_handle.await.unwrap();
        
        assert_eq!(io_result, "io_result");
        assert_eq!(inference_result, 499500); // Sum of 0..1000
        assert_eq!(network_result, 42);
        
        let stats = executor.stats();
        println!("Executor stats: {}", stats);
    }
    
    #[tokio::test]
    async fn test_global_executor() {
        // Test global executor functions
        let io_handle = spawn_io(async { "global_io" });
        let inference_handle = spawn_inference(async { 123 });
        let network_handle = spawn_network(async { "network" });
        
        let results = tokio::join!(io_handle, inference_handle, network_handle);
        
        assert_eq!(results.0.unwrap(), "global_io");
        assert_eq!(results.1.unwrap(), 123);
        assert_eq!(results.2.unwrap(), "network");
    }
    
    #[test]
    fn test_thread_pool_creation() {
        let inference_pool = InferenceThreadPool::new().unwrap();
        let io_pool = IoThreadPool::new().unwrap();
        
        println!("Inference threads: {}", inference_pool.thread_count());
        println!("I/O threads: {}", io_pool.thread_count());
        
        assert!(inference_pool.thread_count() > 0);
        assert!(io_pool.thread_count() > 0);
    }
}