//! CUDA FFI module for direct Triton Server integration
//!
//! This module provides a direct C API interface to Triton Server using CUDA shared memory,
//! eliminating the network overhead of gRPC calls and enabling zero-copy inference.
//!
//! ## Architecture
//!
//! Instead of using gRPC clients to communicate with Triton Server, this module:
//! - Uses Triton's C API directly (embedded in the same process)
//! - Allocates CUDA shared memory for zero-copy tensor operations
//! - Leverages CUDA IPC handles for efficient memory sharing
//! - Supports both simple and complex model configurations
//!
//! ## Performance Benefits
//!
//! - **Zero Network Overhead**: Direct C API calls instead of gRPC
//! - **Zero Copy**: Tensors stay in GPU memory throughout the pipeline
//! - **Reduced Latency**: Eliminates serialization/deserialization overhead
//! - **Better Memory Management**: Direct control over CUDA memory allocation
//!
//! ## Memory Management Architecture
//!
//! This module provides two complementary memory management abstractions:
//!
//! 1. **`DeviceBuffer<T>`**: Low-level, generic CUDA memory buffer with RAII semantics
//!    - Type-safe memory operations
//!    - Automatic cleanup on drop
//!    - Zero-copy casting between compatible types
//!    - Host/device memory transfers
//!
//! 2. **`CudaSharedMemoryRegion`**: High-level Triton C-API integration
//!    - CUDA IPC handles for inter-process sharing
//!    - Direct integration with Triton inference server
//!    - Model-specific memory pool management
//!
//! The `DeviceBuffer` provides the foundation for general CUDA memory operations,
//! while `CudaSharedMemoryRegion` handles the specifics of Triton server integration.

// Module declarations
pub mod async_stream;
pub mod device_buffer;
pub mod error;
pub mod ffi;
pub mod memory_pool;
pub mod shared_memory;
pub mod types;
pub mod utils;

// Re-exports for async stream functionality
pub use async_stream::{AsyncCudaEvent, AsyncCudaStream, AsyncCudaStreamPool};

// Re-exports for device buffer functionality
pub use device_buffer::{DeviceBuffer, DevicePod, DeviceSlice};

// Re-export utility functions from device_buffer
pub use device_buffer::utils::{default_device, device_count, is_available};

// Re-exports for error types
pub use error::{CudaError, CudaSharedMemoryError};

// Re-exports for shared memory functionality
pub use shared_memory::{CudaSharedMemoryRegion, MemoryLease};

// Re-exports for memory pool functionality
pub use memory_pool::CudaSharedMemoryPool;

// Re-exports for type definitions
pub use types::{DataType, ModelConfig, TensorSpec};

// Re-exports for utility functions
pub use utils::{get_cuda_device_count, is_cuda_available};

#[cfg(test)]
mod tests {
    //! # Test Suite for CUDA Memory Lease Management
    //!
    //! This test suite validates the reference counting and memory lifecycle
    //! management for CUDA shared memory regions with IPC support.
    //!
    //! ## Running Tests
    //!
    //! **Requirements:**
    //! - Linux system with NVIDIA GPU and CUDA drivers
    //! - Build with `--features cuda` flag
    //!
    //! **Basic tests (fast, no Triton required):**
    //! ```bash
    //! cargo test --features cuda --lib test_memory_lease
    //! ```
    //!
    //! **Integration tests (requires Triton server):**
    //! ```bash
    //! cargo test --features cuda --lib test_inference_auto_lease
    //! cargo test --features cuda --lib test_register_sets_ipc_flag
    //! ```
    //!
    //! **Longer tests and benchmarks:**
    //! ```bash
    //! cargo test --features cuda --lib -- --ignored --nocapture
    //! ```
    //!
    //! ## Test Categories
    //!
    //! 1. **Unit Tests** - Test MemoryLease RAII behavior without CUDA:
    //!    - `test_memory_lease_increment_decrement`
    //!    - `test_memory_lease_concurrent`
    //!
    //! 2. **CUDA Integration Tests** - Require GPU and CUDA drivers:
    //!    - `test_active_lease_count`
    //!    - `test_inference_auto_lease`
    //!    - `test_drop_waits_for_leases`
    //!    - `test_ipc_shared_flag`
    //!    - `test_register_sets_ipc_flag`
    //!
    //! 3. **Benchmarks** (marked with `#[ignore]`):
    //!    - `bench_lease_acquisition`
    //!    - `test_full_lifecycle_concurrent`
    //!
    //! ## macOS / Non-CUDA Systems
    //!
    //! These tests cannot run on macOS or systems without CUDA because:
    //! - The entire `cuda` module is feature-gated (`#[cfg(feature = "cuda")]`)
    //! - CUDA drivers and runtime are Linux-only
    //! - FFI functions require CUDA C API
    //!
    //! To validate the core logic on macOS, you would need to extract
    //! `MemoryLease` into a standalone module without CUDA dependencies.

    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    /// Test that MemoryLease properly increments and decrements the counter
    #[test]
    fn test_memory_lease_increment_decrement() {
        let counter = Arc::new(AtomicUsize::new(0));

        assert_eq!(counter.load(Ordering::SeqCst), 0);

        {
            let _lease1 = MemoryLease::new(Arc::clone(&counter));
            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "Counter should be 1 after first lease"
            );

            {
                let _lease2 = MemoryLease::new(Arc::clone(&counter));
                assert_eq!(
                    counter.load(Ordering::SeqCst),
                    2,
                    "Counter should be 2 with two leases"
                );

                let _lease3 = MemoryLease::new(Arc::clone(&counter));
                assert_eq!(
                    counter.load(Ordering::SeqCst),
                    3,
                    "Counter should be 3 with three leases"
                );
            } // lease2 and lease3 dropped

            assert_eq!(
                counter.load(Ordering::SeqCst),
                1,
                "Counter should be back to 1"
            );
        } // lease1 dropped

        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "Counter should be back to 0"
        );
    }

    /// Test that MemoryLease works correctly across threads
    #[test]
    fn test_memory_lease_concurrent() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        // Spawn 10 threads that each acquire a lease
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                let _lease = MemoryLease::new(counter_clone);
                thread::sleep(Duration::from_millis(10));
                // Lease will be dropped when thread exits
            });
            handles.push(handle);
        }

        // Wait a bit and check that some leases are active
        thread::sleep(Duration::from_millis(5));
        let active = counter.load(Ordering::SeqCst);
        assert!(
            active > 0 && active <= 10,
            "Should have some active leases: {}",
            active
        );

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // All leases should be released
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "All leases should be released"
        );
    }

    /// Test that active_lease_count() returns correct values
    #[test]
    #[cfg(feature = "cuda")]
    fn test_active_lease_count() {
        // This test requires CUDA to be available
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region = CudaSharedMemoryRegion::new("test_lease_count", 1024, 0)
            .expect("Failed to create region");

        assert_eq!(region.active_lease_count(), 0, "Should start with 0 leases");

        {
            let _lease1 = region.acquire_lease();
            assert_eq!(region.active_lease_count(), 1);

            {
                let _lease2 = region.acquire_lease();
                assert_eq!(region.active_lease_count(), 2);
            }

            assert_eq!(region.active_lease_count(), 1);
        }

        assert_eq!(region.active_lease_count(), 0);
    }

    /// Test that leases are automatically managed in inference
    #[test]
    #[cfg(feature = "cuda")]
    fn test_inference_auto_lease() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region = CudaSharedMemoryRegion::new("test_inference", 4096, 0)
            .expect("Failed to create region");

        assert_eq!(region.active_lease_count(), 0);

        // During inference, lease should be active
        // We can't easily test this without mocking, but we can verify
        // the lease count returns to 0 after inference

        let config = ModelConfig::preprocessor();

        // This will fail if Triton isn't running, but that's OK for this test
        // We're testing the lease mechanism, not the inference itself
        let _ = region.run_inference_with_config(&config, "AUDIO_FRAMES", "MEL_FEATURES");

        // After inference completes (even with error), leases should be released
        assert_eq!(
            region.active_lease_count(),
            0,
            "Leases should be released after inference"
        );
    }

    /// Test that Drop waits for leases (with quick timeout for testing)
    #[test]
    #[cfg(feature = "cuda")]
    fn test_drop_waits_for_leases() {
        use std::sync::mpsc;

        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let (tx, rx) = mpsc::channel();

        let region = Arc::new(
            CudaSharedMemoryRegion::new("test_drop_wait", 1024, 0)
                .expect("Failed to create region"),
        );

        // Acquire a lease in another thread
        let region_clone = Arc::clone(&region);
        let handle = thread::spawn(move || {
            let _lease = region_clone.acquire_lease();
            tx.send(()).unwrap(); // Signal that lease is acquired
            thread::sleep(Duration::from_millis(500)); // Hold lease for 500ms
                                                       // Lease dropped here
        });

        // Wait for lease to be acquired
        rx.recv().unwrap();
        assert_eq!(region.active_lease_count(), 1);

        // Drop the Arc - the last strong reference will trigger Drop
        // but it should wait for the lease in the other thread
        let start = std::time::Instant::now();
        drop(region);
        let elapsed = start.elapsed();

        // Should have waited at least 400ms (allowing some slack)
        assert!(
            elapsed >= Duration::from_millis(400),
            "Drop should wait for leases: waited {:?}",
            elapsed
        );

        handle.join().unwrap();
    }

    /// Test IPC flag behavior
    #[test]
    #[cfg(feature = "cuda")]
    fn test_ipc_shared_flag() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region =
            CudaSharedMemoryRegion::new("test_ipc_flag", 1024, 0).expect("Failed to create region");

        // Initially not IPC-shared
        assert!(!region.is_ipc_shared.load(Ordering::Acquire));

        // Getting raw handle should mark it as IPC-shared
        let _ = region.get_raw_handle();
        assert!(
            region.is_ipc_shared.load(Ordering::Acquire),
            "Should be marked as IPC-shared after get_raw_handle()"
        );
    }

    /// Test register_with_triton_server sets IPC flag
    #[test]
    #[cfg(feature = "cuda")]
    fn test_register_sets_ipc_flag() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region =
            CudaSharedMemoryRegion::new("test_register", 1024, 0).expect("Failed to create region");

        assert!(!region.is_ipc_shared.load(Ordering::Acquire));

        // This will likely fail without Triton running, but should still set the flag
        let _ = region.register_with_triton_server();
        assert!(
            region.is_ipc_shared.load(Ordering::Acquire),
            "Should be marked as IPC-shared after registration"
        );
    }

    /// Benchmark: Lease acquisition overhead
    #[test]
    #[ignore = "Benchmark test - run with --ignored"]
    fn bench_lease_acquisition() {
        let counter = Arc::new(AtomicUsize::new(0));
        let iterations = 1_000_000;

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _lease = MemoryLease::new(Arc::clone(&counter));
            // Lease immediately dropped
        }
        let elapsed = start.elapsed();

        println!(
            "Lease acquisition: {} iterations in {:?}",
            iterations, elapsed
        );
        println!("Average: {:?} per lease", elapsed / iterations);
        println!(
            "Rate: {:.2} million leases/sec",
            iterations as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
    }

    /// Integration test: Full lifecycle with concurrent access
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "Long-running test - run with --ignored"]
    fn test_full_lifecycle_concurrent() {
        if !is_cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let region = Arc::new(
            CudaSharedMemoryRegion::new("test_concurrent_lifecycle", 4096, 0)
                .expect("Failed to create region"),
        );

        let mut handles = vec![];

        // Spawn multiple threads that acquire leases and do work
        for i in 0..5 {
            let region_clone = Arc::clone(&region);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let _lease = region_clone.acquire_lease();
                    // Simulate work
                    thread::sleep(Duration::from_millis(10));
                    println!(
                        "Thread {} iteration {} - active leases: {}",
                        i,
                        j,
                        region_clone.active_lease_count()
                    );
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // All leases should be released
        assert_eq!(
            region.active_lease_count(),
            0,
            "All leases should be released"
        );

        println!("Concurrent lifecycle test passed!");
    }
}
