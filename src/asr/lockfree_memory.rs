//! Lock-free memory pools for high-performance ASR processing.
//!
//! This module provides lock-free object pools using crossbeam data structures
//! to eliminate mutex contention in hot paths and improve scalability under
//! high concurrent load.

use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{AppError, Result};
use crate::constants::{performance as config, audio, model, triton};

use tracing::error;

/// A lock-free object pool for reusing allocations.
/// 
/// Uses crossbeam's SegQueue for lock-free MPMC operations and atomic
/// counters for statistics to eliminate all mutex overhead.
pub struct LockFreeObjectPool<T> {
    /// Lock-free queue for storing available objects
    pool: SegQueue<T>,
    
    /// Factory function for creating new objects when pool is empty
    factory: Box<dyn Fn() -> T + Send + Sync>,
    
    /// Maximum number of objects to keep in the pool
    max_size: usize,
    
    /// Current number of objects in the pool (atomic for lock-free access)
    current_size: AtomicUsize,
    
    /// Total number of objects created (for metrics)
    total_created: AtomicUsize,
    
    /// Total number of objects returned to pool (for metrics)
    total_returned: AtomicUsize,
}

impl<T> LockFreeObjectPool<T> {
    /// Create a new lock-free object pool.
    ///
    /// # Arguments
    /// * `factory` - Function to create new objects when pool is empty
    /// * `max_size` - Maximum number of objects to keep in the pool
    /// * `initial_size` - Number of objects to pre-allocate
    pub fn new<F>(factory: F, max_size: usize, initial_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let pool = SegQueue::new();
        
        // Pre-allocate objects
        for _ in 0..initial_size {
            pool.push(factory());
        }
        
        Self {
            pool,
            factory: Box::new(factory),
            max_size,
            current_size: AtomicUsize::new(initial_size),
            total_created: AtomicUsize::new(initial_size),
            total_returned: AtomicUsize::new(0),
        }
    }
    
    /// Get an object from the pool, creating a new one if empty.
    /// 
    /// This operation is completely lock-free and scales linearly with
    /// the number of CPU cores.
    pub fn get(&self) -> LockFreePooledObject<T> {
        let obj = match self.pool.pop() {
            Some(obj) => {
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                obj
            }
            None => {
                // Pool is empty, create new object
                self.total_created.fetch_add(1, Ordering::Relaxed);
                (self.factory)()
            }
        };
        
        LockFreePooledObject {
            obj: Some(obj),
            pool: self,
        }
    }
    
    /// Return an object to the pool (internal use only).
    /// 
    /// This method enforces the max_size limit and is called automatically
    /// when a PooledObject is dropped.
    fn return_object(&self, obj: T) {
        let current = self.current_size.load(Ordering::Relaxed);
        
        if current < self.max_size {
            // Try to increment the counter first
            let prev = self.current_size.fetch_add(1, Ordering::Relaxed);
            
            if prev < self.max_size {
                // Successfully reserved a slot, add to pool
                self.pool.push(obj);
                self.total_returned.fetch_add(1, Ordering::Relaxed);
            } else {
                // Pool became full while we were checking, decrement and drop
                self.current_size.fetch_sub(1, Ordering::Relaxed);
                // Object is automatically dropped here
            }
        }
        // If pool is full, just drop the object
    }
    
    /// Get current pool statistics.
    /// 
    /// All statistics are gathered atomically without any locking.
    pub fn stats(&self) -> LockFreePoolStats {
        LockFreePoolStats {
            available: self.current_size.load(Ordering::Relaxed),
            max_size: self.max_size,
            total_created: self.total_created.load(Ordering::Relaxed),
            total_returned: self.total_returned.load(Ordering::Relaxed),
        }
    }
}

/// A pooled object that automatically returns to the pool when dropped.
/// 
/// This is the lock-free equivalent of the original PooledObject, providing
/// the same API but with zero mutex overhead.
pub struct LockFreePooledObject<'a, T> {
    obj: Option<T>,
    pool: &'a LockFreeObjectPool<T>,
}

impl<'a, T> LockFreePooledObject<'a, T> {
    /// Get a mutable reference to the contained object.
    pub fn get_mut(&mut self) -> Result<&mut T> {
        self.obj
            .as_mut()
            .ok_or_else(|| AppError::Internal("Object already taken".to_string()))
    }
    
    /// Get an immutable reference to the contained object.
    pub fn get(&self) -> Result<&T> {
        self.obj
            .as_ref()
            .ok_or_else(|| AppError::Internal("Object already taken".to_string()))
    }
    
    /// Take ownership of the object (won't return to pool).
    pub fn take(mut self) -> Result<T> {
        self.obj
            .take()
            .ok_or_else(|| AppError::Internal("Object already taken".to_string()))
    }
}

impl<'a, T> Drop for LockFreePooledObject<'a, T> {
    fn drop(&mut self) {
        if let Some(obj) = self.obj.take() {
            self.pool.return_object(obj);
        }
    }
}

impl<'a, T> std::ops::Deref for LockFreePooledObject<'a, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        match self.obj.as_ref() {
            Some(obj) => obj,
            None => {
                error!("Attempted to dereference taken LockFreePooledObject - this indicates a bug");
                panic!("LockFreePooledObject already taken - use get() method for safe access")
            }
        }
    }
}

impl<'a, T> std::ops::DerefMut for LockFreePooledObject<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.obj.as_mut() {
            Some(obj) => obj,
            None => {
                error!("Attempted to mutably dereference taken LockFreePooledObject - this indicates a bug");
                panic!("LockFreePooledObject already taken - use get_mut() method for safe access")
            }
        }
    }
}

/// Lock-free pool statistics.
#[derive(Debug, Clone)]
pub struct LockFreePoolStats {
    pub available: usize,
    pub max_size: usize,
    pub total_created: usize,
    pub total_returned: usize,
}

impl LockFreePoolStats {
    /// Calculate the hit rate (percentage of requests served from pool).
    pub fn hit_rate(&self) -> f64 {
        if self.total_created == 0 {
            return 0.0;
        }
        
        let total_requests = self.total_created + self.total_returned;
        (self.total_returned as f64 / total_requests as f64) * 100.0
    }
    
    /// Calculate the current utilization (percentage of pool capacity used).
    pub fn utilization(&self) -> f64 {
        if self.max_size == 0 {
            return 0.0;
        }
        
        (self.available as f64 / self.max_size as f64) * 100.0
    }
}

/// Pre-configured lock-free pools for ASR processing.
pub struct LockFreeAsrMemoryPools {
    /// Pool for f32 audio buffers
    pub audio_buffers: LockFreeObjectPool<Vec<f32>>,
    
    /// Pool for encoder input tensors
    pub encoder_inputs: LockFreeObjectPool<Vec<f32>>,
    
    /// Pool for encoder output tensors
    pub encoder_outputs: LockFreeObjectPool<Vec<f32>>,
    
    /// Pool for decoder target sequences
    pub decoder_targets: LockFreeObjectPool<Vec<i32>>,
    
    /// Pool for decoder state vectors
    pub decoder_states: LockFreeObjectPool<Vec<f32>>,
    
    /// Pool for logits tensors
    pub logits: LockFreeObjectPool<Vec<f32>>,
    
    /// Pool for raw tensor data
    pub raw_tensors: LockFreeObjectPool<Vec<u8>>,
    
    /// Pool for zero-copy workspaces
    pub decoder_workspaces: LockFreeObjectPool<crate::asr::zero_copy::DecoderWorkspace>,
}

impl LockFreeAsrMemoryPools {
    /// Create lock-free memory pools optimized for ASR workloads.
    pub fn new() -> Self {
        Self {
            audio_buffers: LockFreeObjectPool::new(
                || Vec::with_capacity(audio::SAMPLE_RATE.value() as usize * config::AUDIO_BUFFER_SECONDS),
                config::AUDIO_BUFFER_POOL_SIZE,
                config::AUDIO_BUFFER_PRE_ALLOC,
            ),
            
            encoder_inputs: LockFreeObjectPool::new(
                || Vec::with_capacity(config::ENCODER_OUTPUT_SIZE),
                config::ENCODER_POOL_SIZE,
                config::ENCODER_PRE_ALLOC,
            ),
            
            encoder_outputs: LockFreeObjectPool::new(
                || Vec::with_capacity(config::ENCODER_OUTPUT_SIZE),
                config::ENCODER_POOL_SIZE,
                config::ENCODER_PRE_ALLOC,
            ),
            
            decoder_targets: LockFreeObjectPool::new(
                || Vec::with_capacity(config::MAX_TOKENS_PER_SEQUENCE),
                config::DECODER_POOL_SIZE,
                config::DECODER_PRE_ALLOC,
            ),
            
            decoder_states: LockFreeObjectPool::new(
                || Vec::with_capacity(model::DECODER_STATE_SIZE),
                config::DECODER_POOL_SIZE,
                config::DECODER_PRE_ALLOC,
            ),
            
            logits: LockFreeObjectPool::new(
                || Vec::with_capacity(triton::VOCABULARY_SIZE),
                config::DECODER_POOL_SIZE,
                config::DECODER_PRE_ALLOC,
            ),
            
            raw_tensors: LockFreeObjectPool::new(
                || Vec::with_capacity(config::TENSOR_BUFFER_SIZE),
                config::RAW_TENSOR_POOL_SIZE,
                config::RAW_TENSOR_PRE_ALLOC,
            ),
            
            decoder_workspaces: LockFreeObjectPool::new(
                crate::asr::zero_copy::DecoderWorkspace::new,
                config::WORKSPACE_POOL_SIZE,
                config::WORKSPACE_PRE_ALLOC,
            ),
        }
    }
    
    /// Get comprehensive memory pool statistics.
    pub fn stats(&self) -> LockFreeAsrMemoryStats {
        LockFreeAsrMemoryStats {
            audio_buffers: self.audio_buffers.stats(),
            encoder_inputs: self.encoder_inputs.stats(),
            encoder_outputs: self.encoder_outputs.stats(),
            decoder_targets: self.decoder_targets.stats(),
            decoder_states: self.decoder_states.stats(),
            logits: self.logits.stats(),
            raw_tensors: self.raw_tensors.stats(),
            decoder_workspaces: self.decoder_workspaces.stats(),
        }
    }
}

impl Default for LockFreeAsrMemoryPools {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive lock-free memory pool statistics.
#[derive(Debug, Clone)]
pub struct LockFreeAsrMemoryStats {
    pub audio_buffers: LockFreePoolStats,
    pub encoder_inputs: LockFreePoolStats,
    pub encoder_outputs: LockFreePoolStats,
    pub decoder_targets: LockFreePoolStats,
    pub decoder_states: LockFreePoolStats,
    pub logits: LockFreePoolStats,
    pub raw_tensors: LockFreePoolStats,
    pub decoder_workspaces: LockFreePoolStats,
}

impl std::fmt::Display for LockFreeAsrMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Lock-Free ASR Memory Pools - Audio: {}/{} ({:.1}% hit), Encoder: {}/{} ({:.1}% hit), Decoder: {}/{} ({:.1}% hit), Raw: {}/{} ({:.1}% hit), Workspaces: {}/{} ({:.1}% hit)",
            self.audio_buffers.available,
            self.audio_buffers.max_size,
            self.audio_buffers.hit_rate(),
            self.encoder_inputs.available,
            self.encoder_inputs.max_size,
            self.encoder_inputs.hit_rate(),
            self.decoder_targets.available,
            self.decoder_targets.max_size,
            self.decoder_targets.hit_rate(),
            self.raw_tensors.available,
            self.raw_tensors.max_size,
            self.raw_tensors.hit_rate(),
            self.decoder_workspaces.available,
            self.decoder_workspaces.max_size,
            self.decoder_workspaces.hit_rate()
        )
    }
}

/// Global lock-free memory pools instance.
static GLOBAL_LOCKFREE_POOLS: once_cell::sync::Lazy<LockFreeAsrMemoryPools> =
    once_cell::sync::Lazy::new(LockFreeAsrMemoryPools::new);

/// Get access to the global lock-free memory pools.
pub fn global_lockfree_pools() -> &'static LockFreeAsrMemoryPools {
    &GLOBAL_LOCKFREE_POOLS
}

/// Get a decoder workspace from the global lock-free pool.
pub fn get_lockfree_decoder_workspace() -> LockFreePooledObject<'static, crate::asr::zero_copy::DecoderWorkspace> {
    global_lockfree_pools().decoder_workspaces.get()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_lockfree_object_pool() {
        let pool = LockFreeObjectPool::new(|| Vec::<i32>::new(), 5, 0);
        
        // Get an object
        let mut obj1 = pool.get();
        assert_eq!(obj1.len(), 0);
        obj1.push(42);
        assert_eq!(obj1.len(), 1);
        
        // Return it (via drop)
        drop(obj1);
        
        // Get it again - should reuse the object
        let obj2 = pool.get();
        assert_eq!(obj2.len(), 1); // Should still have the 42
        assert_eq!(obj2[0], 42);
        
        let stats = pool.stats();
        assert_eq!(stats.max_size, 5);
    }
    
    #[test]
    fn test_concurrent_access() {
        let pool = Arc::new(LockFreeObjectPool::new(|| Vec::<i32>::new(), 10, 5));
        let mut handles = vec![];
        
        // Spawn multiple threads to test concurrent access
        for i in 0..8 {
            let pool_clone = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let mut obj = pool_clone.get();
                    obj.push(i);
                    // Object automatically returns to pool on drop
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = pool.stats();
        println!("Final stats: {:?}", stats);
        assert!(stats.hit_rate() > 0.0); // Should have some cache hits
    }
    
    #[test]
    fn test_lockfree_asr_memory_pools() {
        let pools = LockFreeAsrMemoryPools::new();
        
        let mut audio_buf = pools.audio_buffers.get();
        audio_buf.push(1.0);
        
        let mut encoder_buf = pools.encoder_inputs.get();
        encoder_buf.push(2.0);
        
        drop(audio_buf);
        drop(encoder_buf);
        
        let stats = pools.stats();
        println!("{}", stats);
        
        // Verify pools are working
        assert!(stats.audio_buffers.max_size > 0);
        assert!(stats.encoder_inputs.max_size > 0);
    }
    
    #[test]
    fn test_stats_calculations() {
        let pool = LockFreeObjectPool::new(|| Vec::<i32>::new(), 5, 2);
        
        // Get and return some objects to generate stats
        let obj1 = pool.get();
        let obj2 = pool.get();
        let obj3 = pool.get(); // This should create a new one
        
        drop(obj1);
        drop(obj2);
        drop(obj3);
        
        let stats = pool.stats();
        assert!(stats.hit_rate() > 0.0);
        assert!(stats.utilization() > 0.0);
    }
}