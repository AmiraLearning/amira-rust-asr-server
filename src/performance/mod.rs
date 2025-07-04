//! Performance optimization modules.
//!
//! This module contains advanced performance optimizations including
//! CPU affinity management, NUMA awareness, and specialized thread pools.

pub mod affinity;
pub mod numa_aware;
pub mod specialized_pools;

pub use affinity::{AffinityManager, CpuSet, ThreadType};
pub use numa_aware::{NumaAwareAllocator, NumaNode, numa_allocate_vec};
pub use specialized_pools::{InferenceThreadPool, IoThreadPool, SpecializedExecutor};