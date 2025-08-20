//! CPU affinity management for optimal performance.
//!
//! This module provides thread affinity management to minimize context switching
//! and optimize CPU cache utilization for different types of workloads.

use core_affinity::{self, CoreId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tracing::{debug, info, warn};

/// Types of threads for affinity assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThreadType {
    /// I/O-bound threads (WebSocket, file operations)
    Io,
    /// Compute-intensive inference threads
    Inference,
    /// Background maintenance threads
    Background,
    /// Network communication threads (Triton gRPC)
    Network,
}

/// CPU core set for thread affinity
#[derive(Debug)]
pub struct CpuSet {
    cores: Vec<CoreId>,
    current_index: AtomicUsize,
}

impl Clone for CpuSet {
    fn clone(&self) -> Self {
        Self {
            cores: self.cores.clone(),
            current_index: AtomicUsize::new(self.current_index.load(Ordering::Relaxed)),
        }
    }
}

impl CpuSet {
    /// Create a new CPU set from core IDs
    pub fn new(cores: Vec<CoreId>) -> Self {
        Self {
            cores,
            current_index: AtomicUsize::new(0),
        }
    }

    /// Get the next core in round-robin fashion
    pub fn next_core(&self) -> Option<CoreId> {
        if self.cores.is_empty() {
            return None;
        }

        let index = self.current_index.fetch_add(1, Ordering::Relaxed);
        Some(self.cores[index % self.cores.len()])
    }

    /// Get all cores in this set
    pub fn cores(&self) -> &[CoreId] {
        &self.cores
    }

    /// Get the number of cores in this set
    pub fn len(&self) -> usize {
        self.cores.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.cores.is_empty()
    }
}

/// CPU affinity manager for optimal thread placement
pub struct AffinityManager {
    /// Available CPU cores
    available_cores: Vec<CoreId>,

    /// CPU sets assigned to different thread types
    thread_type_cores: HashMap<ThreadType, CpuSet>,

    /// Whether the system supports CPU affinity
    affinity_supported: bool,
}

impl AffinityManager {
    /// Create a new affinity manager with automatic core detection
    pub fn new() -> Self {
        let available_cores = core_affinity::get_core_ids().unwrap_or_default();
        let affinity_supported = !available_cores.is_empty();

        if affinity_supported {
            info!(
                "CPU affinity manager initialized with {} cores: {:?}",
                available_cores.len(),
                available_cores
            );
        } else {
            warn!("CPU affinity not supported on this platform");
        }

        let mut manager = Self {
            available_cores: available_cores.clone(),
            thread_type_cores: HashMap::new(),
            affinity_supported,
        };

        // Automatically assign cores based on system topology
        manager.auto_assign_cores();

        manager
    }

    /// Automatically assign cores to thread types based on system characteristics
    fn auto_assign_cores(&mut self) {
        if !self.affinity_supported || self.available_cores.is_empty() {
            return;
        }

        let total_cores = self.available_cores.len();

        match total_cores {
            1..=2 => {
                // Very small systems - share cores
                let all_cores = CpuSet::new(self.available_cores.clone());
                self.thread_type_cores
                    .insert(ThreadType::Io, all_cores.clone());
                self.thread_type_cores
                    .insert(ThreadType::Inference, all_cores.clone());
                self.thread_type_cores
                    .insert(ThreadType::Network, all_cores.clone());
                self.thread_type_cores
                    .insert(ThreadType::Background, all_cores);
            }
            3..=4 => {
                // Small systems - dedicate some cores to inference
                let io_cores = CpuSet::new(vec![self.available_cores[0]]);
                let inference_cores = CpuSet::new(self.available_cores[1..].to_vec());
                let network_cores = CpuSet::new(vec![self.available_cores[0]]);
                let background_cores = CpuSet::new(vec![self.available_cores[0]]);

                self.thread_type_cores.insert(ThreadType::Io, io_cores);
                self.thread_type_cores
                    .insert(ThreadType::Inference, inference_cores);
                self.thread_type_cores
                    .insert(ThreadType::Network, network_cores);
                self.thread_type_cores
                    .insert(ThreadType::Background, background_cores);
            }
            5..=8 => {
                // Medium systems - better separation
                let cores_per_type = total_cores / 3;
                let io_cores = CpuSet::new(self.available_cores[0..1].to_vec());
                let inference_cores =
                    CpuSet::new(self.available_cores[1..1 + cores_per_type * 2].to_vec());
                let network_cores =
                    CpuSet::new(self.available_cores[1 + cores_per_type * 2..].to_vec());
                let background_cores = CpuSet::new(vec![self.available_cores[0]]);

                self.thread_type_cores.insert(ThreadType::Io, io_cores);
                self.thread_type_cores
                    .insert(ThreadType::Inference, inference_cores);
                self.thread_type_cores
                    .insert(ThreadType::Network, network_cores);
                self.thread_type_cores
                    .insert(ThreadType::Background, background_cores);
            }
            _ => {
                // Large systems - optimal separation
                // TODO: inference is happening on the GPU, so we should not assign it to the CPU
                let inference_cores = total_cores * 60 / 100; // 60% for inference
                let io_cores = total_cores * 20 / 100; // 20% for I/O
                let network_cores = total_cores * 15 / 100; // 15% for network
                let _background_cores = total_cores * 5 / 100; // 5% for background

                let io_start = 0;
                let inference_start = io_start + io_cores;
                let network_start = inference_start + inference_cores;
                let background_start = network_start + network_cores;

                let io_set = CpuSet::new(self.available_cores[io_start..inference_start].to_vec());
                let inference_set =
                    CpuSet::new(self.available_cores[inference_start..network_start].to_vec());
                let network_set =
                    CpuSet::new(self.available_cores[network_start..background_start].to_vec());
                let background_set = CpuSet::new(self.available_cores[background_start..].to_vec());

                self.thread_type_cores.insert(ThreadType::Io, io_set);
                self.thread_type_cores
                    .insert(ThreadType::Inference, inference_set);
                self.thread_type_cores
                    .insert(ThreadType::Network, network_set);
                self.thread_type_cores
                    .insert(ThreadType::Background, background_set);
            }
        }

        // Log the assignment
        for (thread_type, cpu_set) in &self.thread_type_cores {
            info!(
                "Assigned {:?} threads to {} cores: {:?}",
                thread_type,
                cpu_set.len(),
                cpu_set.cores()
            );
        }
    }

    /// Set the CPU affinity for the current thread
    pub fn set_thread_affinity(&self, thread_type: ThreadType) -> Result<(), String> {
        if !self.affinity_supported {
            debug!("CPU affinity not supported, skipping affinity setting");
            return Ok(());
        }

        if let Some(cpu_set) = self.thread_type_cores.get(&thread_type) {
            if let Some(core_id) = cpu_set.next_core() {
                if core_affinity::set_for_current(core_id) {
                    debug!(
                        "Set thread affinity for {:?} to core {:?}",
                        thread_type, core_id
                    );
                    Ok(())
                } else {
                    Err(format!("Failed to set affinity to core {:?}", core_id))
                }
            } else {
                Err(format!(
                    "No cores available for thread type {:?}",
                    thread_type
                ))
            }
        } else {
            Err(format!(
                "No CPU set configured for thread type {:?}",
                thread_type
            ))
        }
    }

    /// Spawn a thread with specific CPU affinity
    pub fn spawn_with_affinity<F, T>(
        &self,
        thread_type: ThreadType,
        name: Option<String>,
        f: F,
    ) -> thread::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let affinity_manager = self.clone();

        let mut builder = thread::Builder::new();
        if let Some(name) = name {
            builder = builder.name(name);
        }

        builder
            .spawn(move || {
                // Set affinity for this thread
                if let Err(e) = affinity_manager.set_thread_affinity(thread_type) {
                    warn!("Failed to set thread affinity: {}", e);
                }

                // Execute the user function
                f()
            })
            .expect("Failed to spawn thread")
    }

    /// Get the recommended number of threads for a specific thread type
    pub fn recommended_thread_count(&self, thread_type: ThreadType) -> usize {
        if let Some(cpu_set) = self.thread_type_cores.get(&thread_type) {
            cpu_set.len().max(1)
        } else {
            1
        }
    }

    /// Get information about the current CPU topology
    pub fn topology_info(&self) -> CpuTopologyInfo {
        CpuTopologyInfo {
            total_cores: self.available_cores.len(),
            affinity_supported: self.affinity_supported,
            thread_assignments: self
                .thread_type_cores
                .iter()
                .map(|(k, v)| (*k, v.len()))
                .collect(),
        }
    }

    /// Check if CPU affinity is supported on this system
    pub fn is_affinity_supported(&self) -> bool {
        self.affinity_supported
    }
}

impl Clone for AffinityManager {
    fn clone(&self) -> Self {
        Self {
            available_cores: self.available_cores.clone(),
            thread_type_cores: self
                .thread_type_cores
                .iter()
                .map(|(k, v)| (*k, CpuSet::new(v.cores().to_vec())))
                .collect(),
            affinity_supported: self.affinity_supported,
        }
    }
}

impl Default for AffinityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about CPU topology and thread assignments
#[derive(Debug, Clone)]
pub struct CpuTopologyInfo {
    pub total_cores: usize,
    pub affinity_supported: bool,
    pub thread_assignments: HashMap<ThreadType, usize>,
}

impl std::fmt::Display for CpuTopologyInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CPU Topology: {} cores, affinity_supported={}, assignments={:?}",
            self.total_cores, self.affinity_supported, self.thread_assignments
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_affinity_manager_creation() {
        let manager = AffinityManager::new();
        let topology = manager.topology_info();

        println!("CPU topology: {}", topology);

        // Should have some thread type assignments
        assert!(!topology.thread_assignments.is_empty());
    }

    #[test]
    fn test_cpu_set() {
        let cores = core_affinity::get_core_ids().unwrap_or_default();
        if cores.is_empty() {
            return; // Skip test if no cores available
        }

        let cpu_set = CpuSet::new(cores.clone());

        assert_eq!(cpu_set.len(), cores.len());
        assert!(!cpu_set.is_empty());

        // Test round-robin assignment
        let first_core = cpu_set.next_core();
        let second_core = cpu_set.next_core();

        assert!(first_core.is_some());
        assert!(second_core.is_some());
    }

    #[test]
    fn test_thread_spawning_with_affinity() {
        let manager = Arc::new(AffinityManager::new());
        let manager_clone = Arc::clone(&manager);

        let handle = manager.spawn_with_affinity(
            ThreadType::Background,
            Some("test-thread".to_string()),
            move || {
                // Simple test function
                thread::sleep(Duration::from_millis(10));
                42
            },
        );

        let result = handle.join().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_recommended_thread_counts() {
        let manager = AffinityManager::new();

        let inference_threads = manager.recommended_thread_count(ThreadType::Inference);
        let io_threads = manager.recommended_thread_count(ThreadType::Io);

        // Should recommend at least 1 thread for each type
        assert!(inference_threads >= 1);
        assert!(io_threads >= 1);

        println!(
            "Recommended threads - Inference: {}, I/O: {}",
            inference_threads, io_threads
        );
    }
}
