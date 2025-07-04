//! NUMA-aware memory allocation for optimal performance.
//!
//! This module provides NUMA (Non-Uniform Memory Access) awareness to ensure
//! memory allocations happen close to the CPU cores that will use them.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
// Temporary tracing macros while resolving external dependencies
macro_rules! debug { ($($tt:tt)*) => {}; }
macro_rules! info { ($($tt:tt)*) => {}; }

/// Represents a NUMA node in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumaNode {
    pub id: usize,
}

impl NumaNode {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Available NUMA nodes
    pub nodes: Vec<NumaNode>,
    
    /// CPU cores per NUMA node
    pub cores_per_node: HashMap<NumaNode, Vec<usize>>,
    
    /// Whether NUMA is available on this system
    pub numa_available: bool,
}

impl NumaTopology {
    /// Detect NUMA topology on the current system
    pub fn detect() -> Self {
        // Try to detect NUMA topology
        // This is a simplified implementation - in production you might want
        // to use the libnuma bindings or read from /sys/devices/system/node/
        
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems
            Self {
                nodes: vec![NumaNode::new(0)],
                cores_per_node: HashMap::new(),
                numa_available: false,
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;
        
        let numa_available = std::path::Path::new("/sys/devices/system/node").exists();
        
        if !numa_available {
            return Self {
                nodes: vec![NumaNode::new(0)],
                cores_per_node: HashMap::new(),
                numa_available: false,
            };
        }
        
        let mut nodes = Vec::new();
        let mut cores_per_node = HashMap::new();
        
        // Read NUMA nodes from /sys/devices/system/node/
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let name = file_name.to_string_lossy();
                
                if name.starts_with("node") {
                    if let Ok(node_id) = name[4..].parse::<usize>() {
                        let numa_node = NumaNode::new(node_id);
                        nodes.push(numa_node);
                        
                        // Try to read CPU list for this node
                        let cpulist_path = format!("/sys/devices/system/node/node{}/cpulist", node_id);
                        if let Ok(cpulist) = fs::read_to_string(cpulist_path) {
                            let cpu_cores = Self::parse_cpu_list(&cpulist.trim());
                            cores_per_node.insert(numa_node, cpu_cores);
                        }
                    }
                }
            }
        }
        
        if nodes.is_empty() {
            // Fallback if detection failed
            nodes.push(NumaNode::new(0));
        }
        
        Self {
            nodes,
            cores_per_node,
            numa_available,
        }
    }
    
    /// Parse CPU list format (e.g., "0-3,8-11" -> [0,1,2,3,8,9,10,11])
    fn parse_cpu_list(cpulist: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        
        for range in cpulist.split(',') {
            if let Some((start, end)) = range.split_once('-') {
                if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    cpus.extend(start..=end);
                }
            } else if let Ok(cpu) = range.parse::<usize>() {
                cpus.push(cpu);
            }
        }
        
        cpus
    }
    
    /// Get the NUMA node for a specific CPU core
    pub fn node_for_cpu(&self, cpu: usize) -> Option<NumaNode> {
        for (node, cpus) in &self.cores_per_node {
            if cpus.contains(&cpu) {
                return Some(*node);
            }
        }
        None
    }
    
    /// Get the number of NUMA nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if NUMA is available
    pub fn is_numa_available(&self) -> bool {
        self.numa_available && self.nodes.len() > 1
    }
}

/// NUMA-aware allocator for optimizing memory locality
pub struct NumaAwareAllocator {
    /// NUMA topology
    topology: NumaTopology,
    
    /// Current allocation round-robin counter per node
    allocation_counters: HashMap<NumaNode, AtomicUsize>,
    
    /// Whether to enable NUMA optimizations
    numa_enabled: bool,
}

impl NumaAwareAllocator {
    /// Create a new NUMA-aware allocator
    pub fn new() -> Self {
        let topology = NumaTopology::detect();
        let numa_enabled = topology.is_numa_available();
        
        if numa_enabled {
            info!(
                "NUMA-aware allocator initialized with {} nodes",
                topology.node_count()
            );
            
            for (node, cpus) in &topology.cores_per_node {
                info!("NUMA node {}: CPUs {:?}", node.id, cpus);
            }
        } else {
            info!("NUMA not available or single-node system, using standard allocation");
        }
        
        let allocation_counters = topology
            .nodes
            .iter()
            .map(|node| (*node, AtomicUsize::new(0)))
            .collect();
        
        Self {
            topology,
            allocation_counters,
            numa_enabled,
        }
    }
    
    /// Get the preferred NUMA node for the current thread
    pub fn preferred_node_for_current_thread(&self) -> NumaNode {
        if !self.numa_enabled {
            return NumaNode::new(0);
        }
        
        // Try to get the current CPU core
        #[cfg(target_os = "linux")]
        {
            if let Some(cpu_core) = self.get_current_cpu() {
                if let Some(node) = self.topology.node_for_cpu(cpu_core) {
                    return node;
                }
            }
        }
        
        // Fallback to round-robin allocation
        self.next_node_round_robin()
    }
    
    #[cfg(target_os = "linux")]
    fn get_current_cpu(&self) -> Option<usize> {
        // Use sched_getcpu() to get current CPU
        unsafe {
            let cpu = libc::sched_getcpu();
            if cpu >= 0 {
                Some(cpu as usize)
            } else {
                None
            }
        }
    }
    
    /// Get the next NUMA node in round-robin fashion
    fn next_node_round_robin(&self) -> NumaNode {
        if self.topology.nodes.is_empty() {
            return NumaNode::new(0);
        }
        
        // Use the first node's counter for global round-robin
        let first_node = self.topology.nodes[0];
        let default_counter = AtomicUsize::new(0);
        let counter = self.allocation_counters
            .get(&first_node)
            .unwrap_or(&default_counter);
        
        let index = counter.fetch_add(1, Ordering::Relaxed);
        self.topology.nodes[index % self.topology.nodes.len()]
    }
    
    /// Allocate a vector with NUMA awareness
    pub fn allocate_vec<T>(&self, capacity: usize) -> Vec<T> {
        if !self.numa_enabled {
            return Vec::with_capacity(capacity);
        }
        
        let preferred_node = self.preferred_node_for_current_thread();
        
        // For now, we just use standard allocation with a hint
        // In a full implementation, you would use libnuma to allocate
        // memory on the specific node
        debug!(
            "Allocating vector of capacity {} on NUMA node {}",
            capacity, preferred_node.id
        );
        
        Vec::with_capacity(capacity)
    }
    
    /// Allocate a boxed value with NUMA awareness
    pub fn allocate_box<T>(&self, value: T) -> Box<T> {
        if !self.numa_enabled {
            return Box::new(value);
        }
        
        let preferred_node = self.preferred_node_for_current_thread();
        debug!("Allocating boxed value on NUMA node {}", preferred_node.id);
        
        // For now, use standard allocation
        // In production, you would use numa_alloc_onnode()
        Box::new(value)
    }
    
    /// Get NUMA topology information
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
    
    /// Check if NUMA optimizations are enabled
    pub fn is_numa_enabled(&self) -> bool {
        self.numa_enabled
    }
    
    /// Get allocation statistics per NUMA node
    pub fn allocation_stats(&self) -> HashMap<NumaNode, usize> {
        self.allocation_counters
            .iter()
            .map(|(node, counter)| (*node, counter.load(Ordering::Relaxed)))
            .collect()
    }
}

impl Default for NumaAwareAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Global NUMA-aware allocator instance
static GLOBAL_NUMA_ALLOCATOR: once_cell::sync::Lazy<NumaAwareAllocator> =
    once_cell::sync::Lazy::new(NumaAwareAllocator::new);

/// Get access to the global NUMA-aware allocator
pub fn global_numa_allocator() -> &'static NumaAwareAllocator {
    &GLOBAL_NUMA_ALLOCATOR
}

/// Allocate a vector with NUMA awareness using the global allocator
pub fn numa_allocate_vec<T>(capacity: usize) -> Vec<T> {
    global_numa_allocator().allocate_vec(capacity)
}

/// Allocate a boxed value with NUMA awareness using the global allocator
pub fn numa_allocate_box<T>(value: T) -> Box<T> {
    global_numa_allocator().allocate_box(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect();
        
        println!("NUMA topology: {:?}", topology);
        
        // Should have at least one node
        assert!(!topology.nodes.is_empty());
        
        // Test node for CPU lookup
        if let Some(node) = topology.node_for_cpu(0) {
            println!("CPU 0 is on NUMA node {}", node.id);
        }
    }
    
    #[test]
    fn test_numa_allocator() {
        let allocator = NumaAwareAllocator::new();
        
        println!("NUMA enabled: {}", allocator.is_numa_enabled());
        println!("Topology: {:?}", allocator.topology());
        
        // Test vector allocation
        let vec: Vec<i32> = allocator.allocate_vec(1000);
        assert_eq!(vec.capacity(), 1000);
        
        // Test box allocation
        let boxed = allocator.allocate_box(42);
        assert_eq!(*boxed, 42);
        
        // Test stats
        let stats = allocator.allocation_stats();
        println!("Allocation stats: {:?}", stats);
    }
    
    #[test]
    fn test_global_allocator() {
        let allocator = global_numa_allocator();
        
        // Test global allocation functions
        let vec: Vec<f32> = numa_allocate_vec(500);
        assert_eq!(vec.capacity(), 500);
        
        let boxed = numa_allocate_box("test");
        assert_eq!(*boxed, "test");
    }
    
    #[test]
    fn test_cpu_list_parsing() {
        let cpus = NumaTopology::parse_cpu_list("0-3,8-11");
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 9, 10, 11]);
        
        let cpus = NumaTopology::parse_cpu_list("0,2,4,6");
        assert_eq!(cpus, vec![0, 2, 4, 6]);
        
        let cpus = NumaTopology::parse_cpu_list("0-1");
        assert_eq!(cpus, vec![0, 1]);
    }
}