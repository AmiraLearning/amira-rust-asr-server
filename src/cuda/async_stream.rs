use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::os::raw::{c_int, c_void};
use std::sync::Arc;
use tokio::sync::oneshot;
use futures::FutureExt;

use crate::cuda::{CudaError, CudaSharedMemoryError};

/// FFI declarations for async CUDA operations
extern "C" {
    fn cuda_stream_create(stream: *mut *mut c_void) -> CudaError;
    fn cuda_stream_destroy(stream: *mut c_void) -> CudaError;
    fn cuda_stream_synchronize(stream: *mut c_void) -> CudaError;
    fn cuda_stream_query(stream: *mut c_void) -> CudaError;
    fn cuda_event_create(event: *mut *mut c_void) -> CudaError;
    fn cuda_event_destroy(event: *mut c_void) -> CudaError;
    fn cuda_event_record(event: *mut c_void, stream: *mut c_void) -> CudaError;
    fn cuda_event_query(event: *mut c_void) -> CudaError;
    fn cuda_event_synchronize(event: *mut c_void) -> CudaError;
    fn cuda_memcpy_h2d_async(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void) -> CudaError;
    fn cuda_memcpy_d2h_async(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void) -> CudaError;
    fn cuda_memcpy_d2d_async(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void) -> CudaError;
    fn cuda_memset_async(ptr: *mut c_void, value: c_int, size: usize, stream: *mut c_void) -> CudaError;
}

/// Async CUDA stream wrapper integrated with Tokio
pub struct AsyncCudaStream {
    stream: *mut c_void,
    device_id: i32,
}

unsafe impl Send for AsyncCudaStream {}
unsafe impl Sync for AsyncCudaStream {}

impl AsyncCudaStream {
    /// Create a new async CUDA stream
    pub fn new(device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        let mut stream: *mut c_void = std::ptr::null_mut();
        let result = unsafe { cuda_stream_create(&mut stream) };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        if stream.is_null() {
            return Err(CudaSharedMemoryError::NullPointer);
        }
        
        Ok(AsyncCudaStream { stream, device_id })
    }
    
    /// Get the raw CUDA stream handle
    pub fn raw_handle(&self) -> *mut c_void {
        self.stream
    }
    
    /// Get the device ID this stream is associated with
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
    
    /// Check if stream operations are complete (non-blocking)
    pub fn is_ready(&self) -> bool {
        let status = unsafe { cuda_stream_query(self.stream) };
        matches!(status, CudaError::CudaSuccess)
    }
    
    /// Synchronize the stream (blocking)
    pub fn synchronize(&self) -> Result<(), CudaSharedMemoryError> {
        let result = unsafe { cuda_stream_synchronize(self.stream) };
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        Ok(())
    }
    
    /// Asynchronously wait for stream to complete
    pub async fn wait(&self) -> Result<(), CudaSharedMemoryError> {
        StreamWaiter::new(self.stream).await
    }
    
    /// Enqueue async copy from host to device (non-blocking)
    pub fn enqueue_memcpy_h2d<T>(&self, dst: *mut T, src: &[T]) -> Result<(), CudaSharedMemoryError> {
        let size = src.len() * std::mem::size_of::<T>();
        let result = unsafe {
            cuda_memcpy_h2d_async(
                dst as *mut c_void,
                src.as_ptr() as *const c_void,
                size,
                self.stream,
            )
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(())
    }
    
    /// Asynchronously copy data from host to device (blocks until complete)
    pub async fn memcpy_h2d_async<T>(&self, dst: *mut T, src: &[T]) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_memcpy_h2d(dst, src)?;
        self.wait().await
    }
    
    /// Enqueue async copy from device to host (non-blocking)
    pub fn enqueue_memcpy_d2h<T>(&self, dst: &mut [T], src: *const T) -> Result<(), CudaSharedMemoryError> {
        let size = dst.len() * std::mem::size_of::<T>();
        let result = unsafe {
            cuda_memcpy_d2h_async(
                dst.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                size,
                self.stream,
            )
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(())
    }
    
    /// Asynchronously copy data from device to host (blocks until complete)
    pub async fn memcpy_d2h_async<T>(&self, dst: &mut [T], src: *const T) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_memcpy_d2h(dst, src)?;
        self.wait().await
    }
    
    /// Enqueue async copy from device to device (non-blocking)
    pub fn enqueue_memcpy_d2d<T>(&self, dst: *mut T, src: *const T, count: usize) -> Result<(), CudaSharedMemoryError> {
        let size = count * std::mem::size_of::<T>();
        let result = unsafe {
            cuda_memcpy_d2d_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.stream,
            )
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(())
    }
    
    /// Asynchronously copy data from device to device (blocks until complete)
    pub async fn memcpy_d2d_async<T>(&self, dst: *mut T, src: *const T, count: usize) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_memcpy_d2d(dst, src, count)?;
        self.wait().await
    }
    
    /// Enqueue async memset (non-blocking)
    pub fn enqueue_memset<T>(&self, ptr: *mut T, value: i32, count: usize) -> Result<(), CudaSharedMemoryError> {
        let size = count * std::mem::size_of::<T>();
        let result = unsafe {
            cuda_memset_async(
                ptr as *mut c_void,
                value,
                size,
                self.stream,
            )
        };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        Ok(())
    }
    
    /// Asynchronously set device memory to a value (blocks until complete)
    pub async fn memset_async<T>(&self, ptr: *mut T, value: i32, count: usize) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_memset(ptr, value, count)?;
        self.wait().await
    }
    
    /// Create an event and record it on this stream
    pub fn record_event(&self) -> Result<AsyncCudaEvent, CudaSharedMemoryError> {
        let event = AsyncCudaEvent::new()?;
        event.record(self)?;
        Ok(event)
    }
}

impl Drop for AsyncCudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let result = unsafe { cuda_stream_destroy(self.stream) };
            if result != CudaError::CudaSuccess {
                eprintln!("Warning: Failed to destroy CUDA stream: {:?}", result);
            }
        }
    }
}

/// CUDA event for synchronization between streams
pub struct AsyncCudaEvent {
    event: *mut c_void,
}

unsafe impl Send for AsyncCudaEvent {}
unsafe impl Sync for AsyncCudaEvent {}

impl AsyncCudaEvent {
    /// Create a new CUDA event
    pub fn new() -> Result<Self, CudaSharedMemoryError> {
        let mut event: *mut c_void = std::ptr::null_mut();
        let result = unsafe { cuda_event_create(&mut event) };
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        
        if event.is_null() {
            return Err(CudaSharedMemoryError::NullPointer);
        }
        
        Ok(AsyncCudaEvent { event })
    }
    
    /// Record this event on the given stream
    pub fn record(&self, stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        let result = unsafe { cuda_event_record(self.event, stream.stream) };
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        Ok(())
    }
    
    /// Check if event has completed (non-blocking)
    pub fn is_ready(&self) -> bool {
        unsafe { cuda_event_query(self.event) == CudaError::CudaSuccess }
    }
    
    /// Synchronize with this event (blocking)
    pub fn synchronize(&self) -> Result<(), CudaSharedMemoryError> {
        let result = unsafe { cuda_event_synchronize(self.event) };
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }
        Ok(())
    }
    
    /// Asynchronously wait for this event to complete
    pub async fn wait(&self) -> Result<(), CudaSharedMemoryError> {
        EventWaiter::new(self.event).await
    }
}

impl Drop for AsyncCudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let result = unsafe { cuda_event_destroy(self.event) };
            if result != CudaError::CudaSuccess {
                eprintln!("Warning: Failed to destroy CUDA event: {:?}", result);
            }
        }
    }
}

/// Safe wrapper for CUDA handle that implements Send
#[derive(Clone, Copy)]
struct SendCudaHandle(*mut c_void);

unsafe impl Send for SendCudaHandle {}
unsafe impl Sync for SendCudaHandle {}

impl SendCudaHandle {
    fn query_stream(self) -> CudaError {
        unsafe { cuda_stream_query(self.0) }
    }
    
    fn query_event(self) -> CudaError {
        unsafe { cuda_event_query(self.0) }
    }
}

/// Future for waiting on CUDA stream completion
struct StreamWaiter {
    stream: SendCudaHandle,
    receiver: Option<oneshot::Receiver<Result<(), CudaSharedMemoryError>>>,
}

impl StreamWaiter {
    fn new(stream: *mut c_void) -> Self {
        StreamWaiter {
            stream: SendCudaHandle(stream),
            receiver: None,
        }
    }
}

impl Future for StreamWaiter {
    type Output = Result<(), CudaSharedMemoryError>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check if stream is already complete
        if matches!(self.stream.query_stream(), CudaError::CudaSuccess) {
            return Poll::Ready(Ok(()));
        }
        
        // If we don't have a receiver, spawn a task to wait for completion
        if self.receiver.is_none() {
            let (tx, rx) = oneshot::channel();
            let stream = self.stream;
            let waker = cx.waker().clone();
            
            tokio::spawn(async move {
                // Poll in a loop until complete
                loop {
                    if matches!(stream.query_stream(), CudaError::CudaSuccess) {
                        let _ = tx.send(Ok(()));
                        waker.wake();
                        break;
                    }
                    
                    // Yield to allow other tasks to run
                    tokio::task::yield_now().await;
                }
            });
            
            self.receiver = Some(rx);
        }
        
        // Poll the receiver
        match self.receiver.as_mut().unwrap().poll_unpin(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(CudaSharedMemoryError::Unknown)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Future for waiting on CUDA event completion
struct EventWaiter {
    event: SendCudaHandle,
    receiver: Option<oneshot::Receiver<Result<(), CudaSharedMemoryError>>>,
}

impl EventWaiter {
    fn new(event: *mut c_void) -> Self {
        EventWaiter {
            event: SendCudaHandle(event),
            receiver: None,
        }
    }
}

impl Future for EventWaiter {
    type Output = Result<(), CudaSharedMemoryError>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check if event is already complete
        if matches!(self.event.query_event(), CudaError::CudaSuccess) {
            return Poll::Ready(Ok(()));
        }
        
        // If we don't have a receiver, spawn a task to wait for completion
        if self.receiver.is_none() {
            let (tx, rx) = oneshot::channel();
            let event = self.event;
            let waker = cx.waker().clone();
            
            tokio::spawn(async move {
                // Poll in a loop until complete
                loop {
                    if matches!(event.query_event(), CudaError::CudaSuccess) {
                        let _ = tx.send(Ok(()));
                        waker.wake();
                        break;
                    }
                    
                    // Yield to allow other tasks to run
                    tokio::task::yield_now().await;
                }
            });
            
            self.receiver = Some(rx);
        }
        
        // Poll the receiver
        match self.receiver.as_mut().unwrap().poll_unpin(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(CudaSharedMemoryError::Unknown)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Stream pool for managing multiple CUDA streams
pub struct AsyncCudaStreamPool {
    streams: Vec<Arc<AsyncCudaStream>>,
    device_id: i32,
    current_index: std::sync::atomic::AtomicUsize,
}

impl AsyncCudaStreamPool {
    /// Create a new stream pool with the specified number of streams
    pub fn new(device_id: i32, num_streams: usize) -> Result<Self, CudaSharedMemoryError> {
        let mut streams = Vec::with_capacity(num_streams);
        
        for _ in 0..num_streams {
            let stream = AsyncCudaStream::new(device_id)?;
            streams.push(Arc::new(stream));
        }
        
        Ok(AsyncCudaStreamPool {
            streams,
            device_id,
            current_index: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    /// Get the next available stream in round-robin fashion
    pub fn next_stream(&self) -> Arc<AsyncCudaStream> {
        let index = self.current_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stream_index = index % self.streams.len();
        self.streams[stream_index].clone()
    }
    
    /// Get a specific stream by index
    pub fn get_stream(&self, index: usize) -> Option<Arc<AsyncCudaStream>> {
        self.streams.get(index).cloned()
    }
    
    /// Get the number of streams in the pool
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }
    
    /// Synchronize all streams in the pool
    pub async fn synchronize_all(&self) -> Result<(), CudaSharedMemoryError> {
        for stream in &self.streams {
            stream.wait().await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_async_stream_creation() {
        let stream = AsyncCudaStream::new(0);
        assert!(stream.is_ok());
        
        let stream = stream.unwrap();
        assert_eq!(stream.device_id(), 0);
        assert!(!stream.raw_handle().is_null());
    }
    
    #[tokio::test]
    async fn test_stream_pool() {
        let pool = AsyncCudaStreamPool::new(0, 3);
        if let Ok(pool) = pool {
            assert_eq!(pool.num_streams(), 3);
            
            let stream1 = pool.next_stream();
            let stream2 = pool.next_stream();
            let stream3 = pool.next_stream();
            let stream4 = pool.next_stream(); // Should wrap around
            
            assert_eq!(stream1.device_id(), 0);
            assert_eq!(stream2.device_id(), 0);
            assert_eq!(stream3.device_id(), 0);
            assert_eq!(stream4.device_id(), 0);
        }
    }
    
    #[tokio::test]
    async fn test_cuda_event() {
        let event = AsyncCudaEvent::new();
        assert!(event.is_ok());
        
        let event = event.unwrap();
        // Event should be ready immediately since nothing was recorded
        assert!(event.is_ready());
    }
}