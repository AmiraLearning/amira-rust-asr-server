use std::ffi::CString;
use std::mem::{self, align_of, size_of, transmute, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use crate::cuda::{CudaError, CudaSharedMemoryError};
use crate::cuda::async_stream::AsyncCudaStream;

/// Fixed-size device-side buffer for CUDA operations.
/// Provides safe access to device memory with RAII cleanup.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
    device_id: i32,
}

/// A view into a device buffer, analogous to `&[T]` for host memory.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceSlice<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}
unsafe impl<T: Send> Send for DeviceSlice<T> {}
unsafe impl<T: Sync> Sync for DeviceSlice<T> {}

// Foreign function interface for CUDA operations
extern "C" {
    fn cuda_malloc_device(size: usize, device_id: c_int) -> *mut c_void;
    fn cuda_free_device(ptr: *mut c_void, device_id: c_int) -> CudaError;
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize) -> CudaError;
    fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize) -> CudaError;
    fn cuda_memcpy_d2d(dst: *mut c_void, src: *const c_void, size: usize) -> CudaError;
    fn cuda_memset_device(ptr: *mut c_void, value: c_int, size: usize) -> CudaError;
    fn cuda_get_device_count() -> c_int;
}

impl<T: 'static> DeviceBuffer<T> {
    /// Allocate a new device buffer large enough to hold `capacity` `T`'s, but without
    /// initializing the contents.
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails or if `capacity` would cause overflow.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the contents of the buffer are initialized before reading.
    pub unsafe fn uninitialized(capacity: usize, device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        if capacity == 0 {
            return Ok(DeviceBuffer {
                ptr: ptr::null_mut(),
                len: 0,
                capacity: 0,
                device_id,
            });
        }

        let byte_size = capacity.checked_mul(size_of::<T>())
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let ptr = cuda_malloc_device(byte_size, device_id);
        if ptr.is_null() {
            return Err(CudaSharedMemoryError::OutOfMemory);
        }

        Ok(DeviceBuffer {
            ptr: ptr as *mut T,
            len: 0,
            capacity,
            device_id,
        })
    }

    /// Allocate device memory and fill it with zeroes.
    pub fn zeroed(capacity: usize, device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        if capacity == 0 {
            return Ok(DeviceBuffer {
                ptr: ptr::null_mut(),
                len: 0,
                capacity: 0,
                device_id,
            });
        }

        let byte_size = capacity.checked_mul(size_of::<T>())
            .ok_or(CudaSharedMemoryError::InvalidValue)?;

        let ptr = unsafe { cuda_malloc_device(byte_size, device_id) };
        if ptr.is_null() {
            return Err(CudaSharedMemoryError::OutOfMemory);
        }

        let result = unsafe { cuda_memset_device(ptr, 0, byte_size) };
        if result != CudaError::CudaSuccess {
            unsafe { cuda_free_device(ptr, device_id) };
            return Err(result.into());
        }

        Ok(DeviceBuffer {
            ptr: ptr as *mut T,
            len: capacity,
            capacity,
            device_id,
        })
    }

    /// Allocate a new device buffer initialized with a copy of the data in `slice`.
    pub fn from_slice(slice: &[T], device_id: i32) -> Result<Self, CudaSharedMemoryError> {
        if slice.is_empty() {
            return Ok(DeviceBuffer {
                ptr: ptr::null_mut(),
                len: 0,
                capacity: 0,
                device_id,
            });
        }

        let mut buffer = unsafe { Self::uninitialized(slice.len(), device_id)? };
        buffer.copy_from_host(slice)?;
        buffer.len = slice.len();
        Ok(buffer)
    }

    /// Creates a `DeviceBuffer<T>` directly from raw components.
    ///
    /// # Safety
    ///
    /// This is highly unsafe due to the invariants that aren't checked:
    /// - `ptr` must have been allocated via CUDA on the specified device
    /// - `capacity` must be the actual capacity allocated
    /// - The caller must ensure no other references to this memory exist
    pub unsafe fn from_raw_parts(ptr: *mut T, capacity: usize, device_id: i32) -> Self {
        DeviceBuffer {
            ptr,
            len: capacity,
            capacity,
            device_id,
        }
    }

    /// Consumes the buffer and returns the raw components.
    ///
    /// The caller is responsible for properly deallocating the memory.
    pub fn into_raw_parts(self) -> (*mut T, usize, usize, i32) {
        let me = ManuallyDrop::new(self);
        (me.ptr, me.len, me.capacity, me.device_id)
    }

    /// Returns the device ID this buffer was allocated on.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the buffer's data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable raw pointer to the buffer's data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copy data from host memory to this device buffer.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), CudaSharedMemoryError> {
        if src.len() > self.capacity {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if src.is_empty() {
            self.len = 0;
            return Ok(());
        }

        let byte_size = src.len() * size_of::<T>();
        let result = unsafe {
            cuda_memcpy_h2d(
                self.ptr as *mut c_void,
                src.as_ptr() as *const c_void,
                byte_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        self.len = src.len();
        Ok(())
    }

    /// Enqueue async copy from host memory to this device buffer (non-blocking)
    pub fn enqueue_copy_from_host(&mut self, src: &[T], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        if src.len() > self.capacity {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if src.is_empty() {
            self.len = 0;
            return Ok(());
        }

        if stream.device_id() != self.device_id {
            return Err(CudaSharedMemoryError::InvalidValue);
        }
        stream.enqueue_memcpy_h2d(self.ptr, src)?;
        self.len = src.len();
        Ok(())
    }
    
    /// Asynchronously copy data from host memory to this device buffer (blocks until complete)
    pub async fn copy_from_host_async(&mut self, src: &[T], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_copy_from_host(src, stream)?;
        stream.wait().await
    }

    /// Copy data from this device buffer to host memory.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), CudaSharedMemoryError> {
        if dst.len() != self.len {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if self.is_empty() {
            return Ok(());
        }

        let byte_size = self.len * size_of::<T>();
        let result = unsafe {
            cuda_memcpy_d2h(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                byte_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }

    /// Enqueue async copy from this device buffer to host memory (non-blocking)
    pub fn enqueue_copy_to_host(&self, dst: &mut [T], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        if dst.len() != self.len {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if self.is_empty() {
            return Ok(());
        }

        if stream.device_id() != self.device_id {
            return Err(CudaSharedMemoryError::InvalidValue);
        }
        stream.enqueue_memcpy_d2h(dst, self.ptr)?;
        Ok(())
    }
    
    /// Asynchronously copy data from this device buffer to host memory (blocks until complete)
    pub async fn copy_to_host_async(&self, dst: &mut [T], stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_copy_to_host(dst, stream)?;
        stream.wait().await
    }

    /// Copy data from another device buffer to this one.
    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>) -> Result<(), CudaSharedMemoryError> {
        if src.len() > self.capacity {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if src.is_empty() {
            self.len = 0;
            return Ok(());
        }

        let byte_size = src.len() * size_of::<T>();
        let result = unsafe {
            cuda_memcpy_d2d(
                self.ptr as *mut c_void,
                src.ptr as *const c_void,
                byte_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        self.len = src.len();
        Ok(())
    }

    /// Enqueue async copy from another device buffer to this one (non-blocking)
    pub fn enqueue_copy_from_device(&mut self, src: &DeviceBuffer<T>, stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        if src.len() > self.capacity {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if src.is_empty() {
            self.len = 0;
            return Ok(());
        }

        if stream.device_id() != self.device_id || stream.device_id() != src.device_id {
            return Err(CudaSharedMemoryError::InvalidValue);
        }
        stream.enqueue_memcpy_d2d(self.ptr, src.ptr, src.len())?;
        self.len = src.len();
        Ok(())
    }
    
    /// Asynchronously copy data from another device buffer to this one (blocks until complete)
    pub async fn copy_from_device_async(&mut self, src: &DeviceBuffer<T>, stream: &AsyncCudaStream) -> Result<(), CudaSharedMemoryError> {
        self.enqueue_copy_from_device(src, stream)?;
        stream.wait().await
    }

    /// Resize the buffer to contain `new_len` elements.
    /// 
    /// If `new_len` is greater than the current capacity, this will fail.
    pub fn resize(&mut self, new_len: usize) -> Result<(), CudaSharedMemoryError> {
        if new_len > self.capacity {
            return Err(CudaSharedMemoryError::InvalidValue);
        }
        self.len = new_len;
        Ok(())
    }

    /// Clear the buffer, setting its length to 0.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Truncate the buffer to `len` elements.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }

    /// Returns a slice view of the buffer.
    pub fn as_slice(&self) -> &DeviceSlice<T> {
        unsafe { DeviceSlice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut DeviceSlice<T> {
        unsafe { DeviceSlice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Explicitly destroy the buffer, returning any error from the operation.
    pub fn destroy(mut self) -> Result<(), CudaSharedMemoryError> {
        if self.ptr.is_null() {
            return Ok(());
        }

        let ptr = mem::replace(&mut self.ptr, ptr::null_mut());
        let result = unsafe { cuda_free_device(ptr as *mut c_void, self.device_id) };
        
        // Prevent the Drop impl from running
        mem::forget(self);
        
        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }
}

impl<T> DeviceSlice<T> {
    /// Create a device slice from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid device memory
    /// - `len` is the correct length
    /// - The memory is properly aligned for type `T`
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> &'static DeviceSlice<T> {
        &*(ptr::slice_from_raw_parts(ptr, len) as *const DeviceSlice<T>)
    }

    /// Create a mutable device slice from raw parts.
    ///
    /// # Safety
    ///
    /// Same safety requirements as `from_raw_parts`.
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> &'static mut DeviceSlice<T> {
        &mut *(ptr::slice_from_raw_parts_mut(ptr, len) as *mut DeviceSlice<T>)
    }

    /// Returns the number of elements in the slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the slice's data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable raw pointer to the slice's data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copy data from host memory to this device slice.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), CudaSharedMemoryError> {
        if src.len() != self.len {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if src.is_empty() {
            return Ok(());
        }

        let byte_size = src.len() * size_of::<T>();
        let result = unsafe {
            cuda_memcpy_h2d(
                self.ptr as *mut c_void,
                src.as_ptr() as *const c_void,
                byte_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }

    /// Copy data from this device slice to host memory.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), CudaSharedMemoryError> {
        if dst.len() != self.len {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        if self.is_empty() {
            return Ok(());
        }

        let byte_size = self.len * size_of::<T>();
        let result = unsafe {
            cuda_memcpy_d2h(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                byte_size,
            )
        };

        if result != CudaError::CudaSuccess {
            return Err(result.into());
        }

        Ok(())
    }
}

impl<T: 'static> Deref for DeviceBuffer<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: 'static> DerefMut for DeviceBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let result = unsafe { cuda_free_device(self.ptr as *mut c_void, self.device_id) };
            if result != CudaError::CudaSuccess {
                eprintln!("Warning: Failed to free device memory: {:?}", result);
            }
        }
    }
}

/// Type casting support for device buffers containing Pod types.
/// 
/// This allows safe casting between compatible types (e.g., u8 to i8).
pub trait DevicePod: Copy + 'static {}

// Implement DevicePod for common types
impl DevicePod for u8 {}
impl DevicePod for i8 {}
impl DevicePod for u16 {}
impl DevicePod for i16 {}
impl DevicePod for u32 {}
impl DevicePod for i32 {}
impl DevicePod for u64 {}
impl DevicePod for i64 {}
impl DevicePod for f32 {}
impl DevicePod for f64 {}

impl<A: DevicePod> DeviceBuffer<A> {
    /// Cast this buffer to a different type.
    ///
    /// # Panics
    ///
    /// Panics if the cast is invalid (misaligned or would result in fractional elements).
    pub fn cast<B: DevicePod>(self) -> DeviceBuffer<B> {
        self.try_cast().unwrap_or_else(|_| panic!("Invalid cast"))
    }

    /// Try to cast this buffer to a different type.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Target type has greater alignment and buffer is misaligned
    /// - Cast would result in fractional number of elements
    /// - Either type is zero-sized (but not both)
    pub fn try_cast<B: DevicePod>(self) -> Result<DeviceBuffer<B>, CudaSharedMemoryError> {
        // Check alignment
        if align_of::<B>() > align_of::<A>() 
            && (self.ptr as usize) % align_of::<B>() != 0 
        {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        // Handle same-size types
        if size_of::<B>() == size_of::<A>() {
            let me = ManuallyDrop::new(self);
            return Ok(DeviceBuffer {
                ptr: me.ptr as *mut B,
                len: me.len,
                capacity: me.capacity,
                device_id: me.device_id,
            });
        }

        // Handle zero-sized types
        if size_of::<A>() == 0 || size_of::<B>() == 0 {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        // Check for fractional elements
        let total_bytes = size_of::<A>() * self.len;
        if total_bytes % size_of::<B>() != 0 {
            return Err(CudaSharedMemoryError::InvalidValue);
        }

        let new_len = total_bytes / size_of::<B>();
        let new_capacity = (size_of::<A>() * self.capacity) / size_of::<B>();

        let me = ManuallyDrop::new(self);
        Ok(DeviceBuffer {
            ptr: me.ptr as *mut B,
            len: new_len,
            capacity: new_capacity,
            device_id: me.device_id,
        })
    }
}

/// Utility functions for device operations.
pub mod utils {
    use super::*;

    /// Get the number of available CUDA devices.
    pub fn device_count() -> i32 {
        unsafe { cuda_get_device_count() }
    }

    /// Check if CUDA is available.
    pub fn is_available() -> bool {
        device_count() > 0
    }

    /// Get the default device ID (0).
    pub fn default_device() -> i32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_buffer() {
        let buffer: DeviceBuffer<f32> = unsafe { 
            DeviceBuffer::uninitialized(0, 0).unwrap() 
        };
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 0);
    }

    #[test]
    fn test_buffer_resize() {
        let mut buffer = DeviceBuffer::<f32>::zeroed(10, 0).unwrap();
        assert_eq!(buffer.len(), 10);
        
        buffer.resize(5).unwrap();
        assert_eq!(buffer.len(), 5);
        
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_truncate() {
        let mut buffer = DeviceBuffer::<f32>::zeroed(10, 0).unwrap();
        buffer.truncate(5);
        assert_eq!(buffer.len(), 5);
        
        buffer.truncate(20); // Should be no-op
        assert_eq!(buffer.len(), 5);
    }
}