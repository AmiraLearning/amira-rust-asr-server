//! RAII patterns for resource management.
//!
//! This module provides RAII (Resource Acquisition Is Initialization) patterns
//! for managing resources like connections, memory pools, and locks.

// No specific error imports needed for this module
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// A pooled resource that is automatically returned to the pool when dropped.
pub struct PooledResource<T, P>
where
    P: ResourcePool<T>,
{
    resource: Option<T>,
    pool: Arc<P>,
}

impl<T, P> PooledResource<T, P>
where
    P: ResourcePool<T>,
{
    /// Create a new pooled resource.
    pub fn new(resource: T, pool: Arc<P>) -> Self {
        Self {
            resource: Some(resource),
            pool,
        }
    }

    /// Take the resource out of the pool (prevents automatic return).
    pub fn take(mut self) -> T {
        self.resource.take().expect("Resource already taken")
    }

    /// Check if the resource is still available.
    pub fn is_available(&self) -> bool {
        self.resource.is_some()
    }
}

impl<T, P> Drop for PooledResource<T, P>
where
    P: ResourcePool<T>,
{
    fn drop(&mut self) {
        if let Some(resource) = self.resource.take() {
            self.pool.return_resource(resource);
        }
    }
}

impl<T, P> Deref for PooledResource<T, P>
where
    P: ResourcePool<T>,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.resource.as_ref().expect("Resource not available")
    }
}

impl<T, P> DerefMut for PooledResource<T, P>
where
    P: ResourcePool<T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resource.as_mut().expect("Resource not available")
    }
}

/// Trait for resource pools.
pub trait ResourcePool<T>: Send + Sync {
    /// Return a resource to the pool.
    fn return_resource(&self, resource: T);

    /// Get pool statistics.
    fn pool_stats(&self) -> PoolStats;
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total resources in the pool.
    pub total_resources: usize,
    /// Available resources.
    pub available_resources: usize,
    /// Resources currently in use.
    pub resources_in_use: usize,
    /// Total resources created.
    pub total_created: usize,
    /// Total resources destroyed.
    pub total_destroyed: usize,
}

impl PoolStats {
    /// Calculate the utilization rate.
    pub fn utilization_rate(&self) -> f64 {
        if self.total_resources == 0 {
            0.0
        } else {
            self.resources_in_use as f64 / self.total_resources as f64
        }
    }
}

/// A connection that is automatically returned to the pool when dropped.
pub struct PooledConnection<T> {
    inner: Option<T>,
    pool: Arc<dyn ConnectionPool<T>>,
}

impl<T> PooledConnection<T> {
    /// Create a new pooled connection.
    pub fn new(connection: T, pool: Arc<dyn ConnectionPool<T>>) -> Self {
        Self {
            inner: Some(connection),
            pool,
        }
    }

    /// Take the connection out of the pool.
    pub fn take(mut self) -> T {
        self.inner.take().expect("Connection already taken")
    }

    /// Check if the connection is healthy.
    pub fn is_healthy(&self) -> bool {
        self.inner.is_some()
    }
}

impl<T> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        if let Some(conn) = self.inner.take() {
            self.pool.return_connection(conn);
        }
    }
}

impl<T> Deref for PooledConnection<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().expect("Connection not available")
    }
}

impl<T> DerefMut for PooledConnection<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().expect("Connection not available")
    }
}

/// Trait for connection pools.
pub trait ConnectionPool<T>: Send + Sync {
    /// Return a connection to the pool.
    fn return_connection(&self, connection: T);

    /// Get a connection from the pool.
    fn get_connection(&self) -> Option<PooledConnection<T>>;

    /// Get pool statistics.
    fn stats(&self) -> PoolStats;
}

/// A memory buffer that is automatically returned to the pool when dropped.
pub struct PooledBuffer<T> {
    buffer: Option<Vec<T>>,
    pool: Arc<dyn BufferPool<T>>,
}

impl<T> PooledBuffer<T> {
    /// Create a new pooled buffer.
    pub fn new(buffer: Vec<T>, pool: Arc<dyn BufferPool<T>>) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Take the buffer out of the pool.
    pub fn take(mut self) -> Vec<T> {
        self.buffer.take().expect("Buffer already taken")
    }

    /// Clear the buffer without returning it to the pool.
    pub fn clear(&mut self) {
        if let Some(ref mut buffer) = self.buffer {
            buffer.clear();
        }
    }

    /// Get the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.buffer.as_ref().map_or(0, |b| b.capacity())
    }

    /// Get the buffer length.
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map_or(0, |b| b.len())
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.as_ref().map_or(true, |b| b.is_empty())
    }
}

impl<T> Drop for PooledBuffer<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer);
        }
    }
}

impl<T> Deref for PooledBuffer<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("Buffer not available")
    }
}

impl<T> DerefMut for PooledBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().expect("Buffer not available")
    }
}

/// Trait for buffer pools.
pub trait BufferPool<T>: Send + Sync {
    /// Return a buffer to the pool.
    fn return_buffer(&self, buffer: Vec<T>);

    /// Get a buffer from the pool.
    fn get_buffer(&self) -> Option<PooledBuffer<T>>;

    /// Get a buffer with a specific capacity.
    fn get_buffer_with_capacity(&self, capacity: usize) -> Option<PooledBuffer<T>>;

    /// Get pool statistics.
    fn stats(&self) -> PoolStats;
}

/// A guard that ensures a resource is cleaned up when dropped.
pub struct ResourceGuard<T, F>
where
    F: FnOnce(T),
{
    resource: Option<T>,
    cleanup: Option<F>,
}

impl<T, F> ResourceGuard<T, F>
where
    F: FnOnce(T),
{
    /// Create a new resource guard.
    pub fn new(resource: T, cleanup: F) -> Self {
        Self {
            resource: Some(resource),
            cleanup: Some(cleanup),
        }
    }

    /// Take the resource and disable cleanup.
    pub fn take(mut self) -> T {
        self.cleanup.take(); // Disable cleanup
        self.resource.take().expect("Resource already taken")
    }

    /// Get a reference to the resource.
    pub fn get(&self) -> &T {
        self.resource.as_ref().expect("Resource not available")
    }

    /// Get a mutable reference to the resource.
    pub fn get_mut(&mut self) -> &mut T {
        self.resource.as_mut().expect("Resource not available")
    }
}

impl<T, F> Drop for ResourceGuard<T, F>
where
    F: FnOnce(T),
{
    fn drop(&mut self) {
        if let (Some(resource), Some(cleanup)) = (self.resource.take(), self.cleanup.take()) {
            cleanup(resource);
        }
    }
}

impl<T, F> Deref for ResourceGuard<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T, F> DerefMut for ResourceGuard<T, F>
where
    F: FnOnce(T),
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// A scoped resource that ensures cleanup on scope exit.
pub struct ScopedResource<T> {
    resource: Option<T>,
    cleanup: Option<Box<dyn FnOnce() + Send>>,
}

impl<T> ScopedResource<T> {
    /// Create a new scoped resource.
    pub fn new<F>(resource: T, cleanup: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            resource: Some(resource),
            cleanup: Some(Box::new(cleanup)),
        }
    }

    /// Take the resource and disable cleanup.
    pub fn take(mut self) -> T {
        self.cleanup.take();
        self.resource.take().expect("Resource already taken")
    }
}

impl<T> Drop for ScopedResource<T> {
    fn drop(&mut self) {
        if let Some(c) = self.cleanup.take() {
            c();
        }
    }
}

impl<T> Deref for ScopedResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.resource.as_ref().expect("Resource not available")
    }
}

impl<T> DerefMut for ScopedResource<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resource.as_mut().expect("Resource not available")
    }
}

/// A lock guard that provides automatic unlocking.
pub struct LockGuard<T, L> {
    data: *mut T,
    #[allow(dead_code)]
    lock: L,
    _phantom: PhantomData<T>,
}

impl<T, L> LockGuard<T, L> {
    /// Create a new lock guard.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `data` points to valid memory
    /// - The lock `L` properly protects access to `data`
    /// - The lock will be properly released when `L` is dropped
    pub unsafe fn new(data: *mut T, lock: L) -> Self {
        Self {
            data,
            lock,
            _phantom: PhantomData,
        }
    }
}

impl<T, L> Deref for LockGuard<T, L> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.data }
    }
}

impl<T, L> DerefMut for LockGuard<T, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.data }
    }
}

// Safety: LockGuard is safe to send between threads as long as T and L are Send
unsafe impl<T, L> Send for LockGuard<T, L>
where
    T: Send,
    L: Send,
{
}

// Safety: LockGuard is safe to share between threads as long as T and L are Sync
unsafe impl<T, L> Sync for LockGuard<T, L>
where
    T: Sync,
    L: Sync,
{
}

/// A timer that measures elapsed time and executes cleanup on drop.
pub struct Timer {
    start_time: std::time::Instant,
    #[allow(dead_code)]
    name: String,
    cleanup: Option<Box<dyn FnOnce(std::time::Duration) + Send>>,
}

impl Timer {
    /// Create a new timer.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            name: name.into(),
            cleanup: None,
        }
    }

    /// Create a timer with a cleanup function.
    pub fn with_cleanup<F>(name: impl Into<String>, cleanup: F) -> Self
    where
        F: FnOnce(std::time::Duration) + Send + 'static,
    {
        Self {
            start_time: std::time::Instant::now(),
            name: name.into(),
            cleanup: Some(Box::new(cleanup)),
        }
    }

    /// Get the elapsed time.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Stop the timer and return the elapsed time.
    pub fn stop(mut self) -> std::time::Duration {
        let elapsed = self.elapsed();
        self.cleanup.take(); // Disable cleanup
        elapsed
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup(self.elapsed());
        }
    }
}

/// Convenience macros for RAII patterns.
#[macro_export]
macro_rules! scoped_timer {
    ($name:expr) => {
        let _timer = $crate::raii::Timer::new($name);
    };
    ($name:expr, $cleanup:expr) => {
        let _timer = $crate::raii::Timer::with_cleanup($name, $cleanup);
    };
}

#[macro_export]
macro_rules! with_pooled_resource {
    ($pool:expr, $resource_var:ident => $body:block) => {
        if let Some($resource_var) = $pool.get() {
            $body
        } else {
            return Err("Pool exhausted".to_string());
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    struct MockPool {
        returned_count: Arc<AtomicUsize>,
    }

    impl MockPool {
        fn new() -> Self {
            Self {
                returned_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn returned_count(&self) -> usize {
            self.returned_count.load(Ordering::SeqCst)
        }
    }

    impl ResourcePool<String> for MockPool {
        fn return_resource(&self, _resource: String) {
            self.returned_count.fetch_add(1, Ordering::SeqCst);
        }

        fn pool_stats(&self) -> PoolStats {
            PoolStats {
                total_resources: 10,
                available_resources: 5,
                resources_in_use: 5,
                total_created: 10,
                total_destroyed: 0,
            }
        }
    }

    #[test]
    fn test_pooled_resource_automatic_return() {
        let pool = Arc::new(MockPool::new());
        let resource = "test_resource".to_string();

        {
            let _pooled = PooledResource::new(resource, pool.clone());
            // Resource should be automatically returned when dropped
        }

        assert_eq!(pool.returned_count(), 1);
    }

    #[test]
    fn test_pooled_resource_take() {
        let pool = Arc::new(MockPool::new());
        let resource = "test_resource".to_string();

        {
            let pooled = PooledResource::new(resource, pool.clone());
            let _taken = pooled.take();
            // Resource should not be returned because it was taken
        }

        assert_eq!(pool.returned_count(), 0);
    }

    #[test]
    fn test_resource_guard_cleanup() {
        let cleanup_called = Arc::new(AtomicUsize::new(0));
        let cleanup_called_clone = cleanup_called.clone();

        {
            let resource = "test_resource".to_string();
            let _guard = ResourceGuard::new(resource, move |_| {
                cleanup_called_clone.fetch_add(1, Ordering::SeqCst);
            });
            // Cleanup should be called when guard is dropped
        }

        assert_eq!(cleanup_called.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_resource_guard_take() {
        let cleanup_called = Arc::new(AtomicUsize::new(0));
        let cleanup_called_clone = cleanup_called.clone();

        {
            let resource = "test_resource".to_string();
            let guard = ResourceGuard::new(resource, move |_| {
                cleanup_called_clone.fetch_add(1, Ordering::SeqCst);
            });
            let _taken = guard.take();
            // Cleanup should not be called because resource was taken
        }

        assert_eq!(cleanup_called.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_timer_elapsed() {
        let timer = Timer::new("test_timer");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_timer_with_cleanup() {
        let cleanup_called = Arc::new(Mutex::new(false));
        let cleanup_called_clone = cleanup_called.clone();

        {
            let _timer = Timer::with_cleanup("test_timer", move |_duration| {
                *cleanup_called_clone.lock().unwrap() = true;
            });
            // Cleanup should be called when timer is dropped
        }

        assert!(*cleanup_called.lock().unwrap());
    }

    #[test]
    fn test_pool_stats_utilization() {
        let stats = PoolStats {
            total_resources: 10,
            available_resources: 3,
            resources_in_use: 7,
            total_created: 10,
            total_destroyed: 0,
        };

        assert_eq!(stats.utilization_rate(), 0.7);
    }
}
