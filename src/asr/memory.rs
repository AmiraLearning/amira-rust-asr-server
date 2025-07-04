//! High-performance memory pools for ASR processing.
//!
//! This module provides object pools for frequently allocated data structures
//! to eliminate allocation overhead in hot paths.

use parking_lot::Mutex;
use std::collections::VecDeque;

/// A generic object pool for reusing allocations.
pub struct ObjectPool<T> {
    pool: Mutex<VecDeque<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
    max_size: usize,
}

impl<T> ObjectPool<T> {
    /// Create a new object pool.
    ///
    /// # Arguments
    /// * `factory` - Function to create new objects when pool is empty.
    /// * `max_size` - Maximum number of objects to keep in the pool.
    /// * `initial_size` - Number of objects to pre-allocate.
    pub fn new<F>(factory: F, max_size: usize, initial_size: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let mut pool = VecDeque::with_capacity(max_size);

        // Pre-allocate objects
        for _ in 0..initial_size {
            pool.push_back(factory());
        }

        Self {
            pool: Mutex::new(pool),
            factory: Box::new(factory),
            max_size,
        }
    }

    /// Get an object from the pool, creating a new one if empty.
    pub fn get(&self) -> PooledObject<T> {
        // Fast path: try to get from pool without blocking
        let obj = {
            let mut pool = self.pool.lock();
            pool.pop_front()
        };

        let obj = match obj {
            Some(o) => o,
            None => (self.factory)(), // Create new if pool empty
        };

        PooledObject {
            obj: Some(obj),
            pool: self,
        }
    }

    /// Return an object to the pool.
    fn return_object(&self, obj: T) {
        let mut pool = self.pool.lock();
        if pool.len() < self.max_size {
            pool.push_back(obj);
        }
        // If pool is full, just drop the object
    }
    /// Get current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let pool = self.pool.lock();
        PoolStats {
            available: pool.len(),
            max_size: self.max_size,
        }
    }
}

/// A pooled object that automatically returns to the pool when dropped.
/// 
/// # Safety Contract
/// 
/// This object can be accessed via:
/// - `Deref`/`DerefMut` traits: Convenient but may panic if object was taken
/// - `get()`/`get_mut()` methods: Safe, returns `Result` 
/// - `take()` method: Consumes the object, preventing further access
/// 
/// Once `take()` is called, any subsequent access via Deref traits will panic.
/// This is by design to prevent use-after-take bugs.
/// 
/// # Example
/// 
/// ```rust
/// let mut pooled = pool.get();
/// pooled.push(42);  // Works via Deref
/// 
/// // Safe access
/// if let Ok(obj) = pooled.get() {
///     println!("Length: {}", obj.len());
/// }
/// 
/// // Take ownership (object can't be accessed after this)
/// let owned = pooled.take().unwrap();
/// // pooled.len(); // Would panic!
/// ```
pub struct PooledObject<'a, T> {
    obj: Option<T>,
    pool: &'a ObjectPool<T>,
}

impl<'a, T> PooledObject<'a, T> {
    /// Get a mutable reference to the contained object.
    pub fn get_mut(&mut self) -> Result<&mut T, &'static str> {
        self.obj.as_mut().ok_or("Object already taken")
    }

    /// Get an immutable reference to the contained object.
    pub fn get(&self) -> Result<&T, &'static str> {
        self.obj.as_ref().ok_or("Object already taken")
    }

    /// Take ownership of the object (won't return to pool).
    pub fn take(mut self) -> Result<T, &'static str> {
        self.obj.take().ok_or("Object already taken")
    }
}

impl<'a, T> Drop for PooledObject<'a, T> {
    fn drop(&mut self) {
        if let Some(obj) = self.obj.take() {
            self.pool.return_object(obj);
        }
    }
}

impl<'a, T> std::ops::Deref for PooledObject<'a, T> {
    type Target = T;

    /// Dereference the pooled object.
    /// 
    /// # Panics
    /// 
    /// Panics if the object has already been taken via `take()`. This indicates
    /// a programming error - the object should not be accessed after taking.
    /// Use `get()` method for non-panicking access.
    fn deref(&self) -> &Self::Target {
        match self.obj.as_ref() {
            Some(obj) => obj,
            None => {
                // In debug builds, provide detailed error information
                debug_assert!(false, "Attempted to dereference a taken PooledObject. This is a programming error.");
                
                // Log the error for debugging
                tracing::error!("Attempted to dereference taken PooledObject - this indicates a bug in object lifecycle management");
                
                // Unfortunately, Deref trait requires returning a reference, not Result
                // This panic is the only safe option to prevent undefined behavior
                panic!("PooledObject already taken - use get() method for safe access or fix object lifecycle")
            }
        }
    }
}

impl<'a, T> std::ops::DerefMut for PooledObject<'a, T> {
    /// Mutably dereference the pooled object.
    /// 
    /// # Panics
    /// 
    /// Panics if the object has already been taken via `take()`. This indicates
    /// a programming error - the object should not be accessed after taking.
    /// Use `get_mut()` method for non-panicking access.
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.obj.as_mut() {
            Some(obj) => obj,
            None => {
                // In debug builds, provide detailed error information
                debug_assert!(false, "Attempted to mutably dereference a taken PooledObject. This is a programming error.");
                
                // Log the error for debugging
                tracing::error!("Attempted to mutably dereference taken PooledObject - this indicates a bug in object lifecycle management");
                
                // Unfortunately, DerefMut trait requires returning a reference, not Result
                // This panic is the only safe option to prevent undefined behavior
                panic!("PooledObject already taken - use get_mut() method for safe access or fix object lifecycle")
            }
        }
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub available: usize,
    pub max_size: usize,
}

/// Pre-configured pools for ASR processing.
pub struct AsrMemoryPools {
    /// Pool for f32 audio buffers (typical size: 16000 samples = 1 second).
    pub audio_buffers: ObjectPool<Vec<f32>>,

    /// Pool for encoder input tensors (typical size: depends on model).
    pub encoder_inputs: ObjectPool<Vec<f32>>,

    /// Pool for encoder output tensors.
    pub encoder_outputs: ObjectPool<Vec<f32>>,

    /// Pool for decoder target sequences.
    pub decoder_targets: ObjectPool<Vec<i32>>,

    /// Pool for decoder state vectors.
    pub decoder_states: ObjectPool<Vec<f32>>,

    /// Pool for logits tensors.
    pub logits: ObjectPool<Vec<f32>>,

    /// Pool for raw tensor data.
    pub raw_tensors: ObjectPool<Vec<u8>>,
}

impl AsrMemoryPools {
    /// Create memory pools optimized for ASR workloads.
    pub fn new() -> Self {
        const AUDIO_SAMPLE_RATE: usize = 16000;
        const ENCODER_OUTPUT_SIZE: usize = 1024 * 100; // 1024 features * 100 frames
        const DECODER_STATE_SIZE: usize = 640;
        const VOCABULARY_SIZE: usize = 1030;
        const TENSOR_BUFFER_SIZE: usize = 1024 * 1024; // 1MB

        Self {
            audio_buffers: ObjectPool::new(
                || Vec::with_capacity(AUDIO_SAMPLE_RATE * 2), // 2 seconds capacity
                20,                                           // max 20 buffers
                5,                                            // pre-allocate 5
            ),

            encoder_inputs: ObjectPool::new(
                || Vec::with_capacity(ENCODER_OUTPUT_SIZE),
                50, // max 50 buffers
                10, // pre-allocate 10
            ),

            encoder_outputs: ObjectPool::new(|| Vec::with_capacity(ENCODER_OUTPUT_SIZE), 50, 10),

            decoder_targets: ObjectPool::new(
                || Vec::with_capacity(200), // max 200 tokens
                100,                        // max 100 buffers
                20,                         // pre-allocate 20
            ),

            decoder_states: ObjectPool::new(|| Vec::with_capacity(DECODER_STATE_SIZE), 100, 20),

            logits: ObjectPool::new(|| Vec::with_capacity(VOCABULARY_SIZE), 100, 20),

            raw_tensors: ObjectPool::new(
                || Vec::with_capacity(TENSOR_BUFFER_SIZE),
                30, // max 30MB total
                5,  // pre-allocate 5MB
            ),
        }
    }

    /// Get comprehensive memory pool statistics.
    pub fn stats(&self) -> AsrMemoryStats {
        AsrMemoryStats {
            audio_buffers: self.audio_buffers.stats(),
            encoder_inputs: self.encoder_inputs.stats(),
            encoder_outputs: self.encoder_outputs.stats(),
            decoder_targets: self.decoder_targets.stats(),
            decoder_states: self.decoder_states.stats(),
            logits: self.logits.stats(),
            raw_tensors: self.raw_tensors.stats(),
        }
    }
}

impl Default for AsrMemoryPools {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive memory pool statistics.
#[derive(Debug, Clone)]
pub struct AsrMemoryStats {
    pub audio_buffers: PoolStats,
    pub encoder_inputs: PoolStats,
    pub encoder_outputs: PoolStats,
    pub decoder_targets: PoolStats,
    pub decoder_states: PoolStats,
    pub logits: PoolStats,
    pub raw_tensors: PoolStats,
}

impl std::fmt::Display for AsrMemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ASR Memory Pools - Audio: {}/{}, Encoder: {}/{}, Decoder: {}/{}, Raw: {}/{}",
            self.audio_buffers.available,
            self.audio_buffers.max_size,
            self.encoder_inputs.available,
            self.encoder_inputs.max_size,
            self.decoder_targets.available,
            self.decoder_targets.max_size,
            self.raw_tensors.available,
            self.raw_tensors.max_size
        )
    }
}

/// Global memory pools instance.
static GLOBAL_POOLS: once_cell::sync::Lazy<AsrMemoryPools> =
    once_cell::sync::Lazy::new(AsrMemoryPools::new);

/// Get access to the global memory pools.
pub fn global_pools() -> &'static AsrMemoryPools {
    &GLOBAL_POOLS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_pool() {
        let pool = ObjectPool::new(|| Vec::<i32>::new(), 5, 0); // Start with empty pool

        // Get an object 
        let mut obj1 = pool.get();
        assert_eq!(obj1.len(), 0);
        obj1.push(42);
        assert_eq!(obj1.len(), 1);

        // Return it (via drop)
        drop(obj1);

        // Should now have 1 object in pool
        let stats = pool.stats();
        assert_eq!(stats.available, 1);

        // Get it again - should be the same object with the value intact
        let obj2 = pool.get();
        assert_eq!(obj2.len(), 1); // Should still have the 42
        assert_eq!(obj2[0], 42);

        let stats = pool.stats();
        assert_eq!(stats.max_size, 5);
        assert_eq!(stats.available, 0); // Should be taken from pool
    }

    #[test]
    fn test_asr_memory_pools() {
        let pools = AsrMemoryPools::new();

        let mut audio_buf = pools.audio_buffers.get();
        audio_buf.push(1.0);

        let mut encoder_buf = pools.encoder_inputs.get();
        encoder_buf.push(2.0);

        drop(audio_buf);
        drop(encoder_buf);

        let stats = pools.stats();
        println!("{}", stats);
    }
}
