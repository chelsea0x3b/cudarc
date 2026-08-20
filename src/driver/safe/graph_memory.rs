use super::core::CudaStream;
use crate::driver::{
    result::{self, DriverError},
    sys,
};
use std::sync::Arc;
#[cfg(not(feature = "no-std"))]
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

const DEVICE_ALLOCATION_ALIGNMENT: usize = 256;

#[derive(Debug)]
pub(crate) struct CudaGraphMemoryPoolInner {
    pointer: sys::CUdeviceptr,
    capacity: usize,
    stream: Arc<CudaStream>,
}

impl Drop for CudaGraphMemoryPoolInner {
    fn drop(&mut self) {
        let context = self.stream.context();
        context.record_err(self.stream.synchronize());
        if context.has_async_alloc {
            context
                .record_err(unsafe { result::free_async(self.pointer, self.stream.cu_stream()) });
            context.record_err(self.stream.synchronize());
        } else {
            context.record_err(unsafe { result::free_sync(self.pointer) });
        }
    }
}

/// Rank- or device-local backing storage shared by CUDA graphs that are never
/// replayed concurrently.
///
/// Each capture session uses a fresh bump cursor over the same storage. This
/// deliberately keeps captured addresses stable without placing allocation or
/// free nodes in the CUDA graph.
#[derive(Clone, Debug)]
pub struct CudaGraphMemoryPool {
    inner: Arc<CudaGraphMemoryPoolInner>,
}

impl CudaGraphMemoryPool {
    /// Allocates graph-private backing storage on `stream`.
    pub fn new(stream: Arc<CudaStream>, capacity: usize) -> Result<Self, DriverError> {
        let capacity = align_up(capacity).ok_or(out_of_memory())?;
        if capacity == 0 {
            return Err(invalid_value());
        }
        stream.context().bind_to_thread()?;
        let pointer = if stream.context().has_async_alloc {
            unsafe { result::malloc_async(stream.cu_stream(), capacity) }?
        } else {
            unsafe { result::malloc_sync(capacity) }?
        };
        let pool = Self {
            inner: Arc::new(CudaGraphMemoryPoolInner {
                pointer,
                capacity,
                stream,
            }),
        };
        pool.inner.stream.synchronize()?;
        Ok(pool)
    }

    /// Total reusable device storage in bytes.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Routes allocations on this pool's stream into a fresh bump-allocation
    /// session.
    ///
    /// Only available with threading support (not under `no-std`).
    ///
    /// # Safety
    ///
    /// Captures sharing a pool may retain aliased `CudaSlice` values. The
    /// caller must ensure their executable graphs never overlap and that no
    /// retained tensor is accessed outside its graph's serialized replay.
    #[cfg(not(feature = "no-std"))]
    pub unsafe fn begin_session(&self) -> Result<CudaGraphMemoryPoolSession, DriverError> {
        GRAPH_MEMORY_MODE.with(|mode| {
            let mut mode = mode.borrow_mut();
            if mode.is_some() {
                return Err(invalid_value());
            }
            *mode = Some(GraphMemoryMode::Pool {
                stream: self.inner.stream.clone(),
                pool: self.inner.clone(),
                offset: 0,
            });
            Ok(CudaGraphMemoryPoolSession {
                active: true,
                not_send: PhantomData,
            })
        })
    }
}

/// Measures the aligned storage needed by one allocation sequence while
/// leaving its actual allocations unchanged.
#[cfg(not(feature = "no-std"))]
#[must_use]
pub struct CudaGraphMemoryPoolProbe {
    active: bool,
    not_send: PhantomData<Rc<()>>,
}

#[cfg(not(feature = "no-std"))]
impl CudaGraphMemoryPoolProbe {
    /// Completes the probe and returns the required arena capacity.
    pub fn finish(mut self) -> Result<usize, DriverError> {
        let required = take_probe()?;
        self.active = false;
        Ok(required)
    }
}

#[cfg(not(feature = "no-std"))]
impl Drop for CudaGraphMemoryPoolProbe {
    fn drop(&mut self) {
        if self.active {
            clear_mode();
        }
    }
}

/// Thread-local allocation routing active for a warmup or CUDA graph capture.
#[cfg(not(feature = "no-std"))]
#[must_use]
pub struct CudaGraphMemoryPoolSession {
    active: bool,
    not_send: PhantomData<Rc<()>>,
}

#[cfg(not(feature = "no-std"))]
impl CudaGraphMemoryPoolSession {
    /// Completes the session and returns the bytes consumed by its bump cursor.
    pub fn finish(mut self) -> Result<usize, DriverError> {
        let used = take_pool_session()?;
        self.active = false;
        Ok(used)
    }
}

#[cfg(not(feature = "no-std"))]
impl Drop for CudaGraphMemoryPoolSession {
    fn drop(&mut self) {
        if self.active {
            clear_mode();
        }
    }
}

impl CudaStream {
    /// Begins measuring the storage required by allocations on this stream.
    #[cfg(not(feature = "no-std"))]
    pub fn probe_graph_memory_pool(
        self: &Arc<Self>,
    ) -> Result<CudaGraphMemoryPoolProbe, DriverError> {
        GRAPH_MEMORY_MODE.with(|mode| {
            let mut mode = mode.borrow_mut();
            if mode.is_some() {
                return Err(invalid_value());
            }
            *mode = Some(GraphMemoryMode::Probe {
                stream: self.clone(),
                required: 0,
            });
            Ok(CudaGraphMemoryPoolProbe {
                active: true,
                not_send: PhantomData,
            })
        })
    }
}

#[cfg(not(feature = "no-std"))]
enum GraphMemoryMode {
    Probe {
        stream: Arc<CudaStream>,
        required: usize,
    },
    Pool {
        stream: Arc<CudaStream>,
        pool: Arc<CudaGraphMemoryPoolInner>,
        offset: usize,
    },
}

#[cfg(not(feature = "no-std"))]
thread_local! {
    static GRAPH_MEMORY_MODE: RefCell<Option<GraphMemoryMode>> = const { RefCell::new(None) };
}

#[cfg(not(feature = "no-std"))]
pub(crate) fn route_allocation(
    stream: &Arc<CudaStream>,
    bytes: usize,
) -> Result<Option<(sys::CUdeviceptr, Arc<CudaGraphMemoryPoolInner>)>, DriverError> {
    let aligned = align_up(bytes).ok_or(out_of_memory())?;
    GRAPH_MEMORY_MODE.with(|mode| {
        let mut mode = mode.borrow_mut();
        match mode.as_mut() {
            None => Ok(None),
            Some(GraphMemoryMode::Probe {
                stream: active_stream,
                required,
            }) => {
                ensure_stream(active_stream, stream)?;
                *required = required.checked_add(aligned).ok_or(out_of_memory())?;
                Ok(None)
            }
            Some(GraphMemoryMode::Pool {
                stream: active_stream,
                pool,
                offset,
            }) => {
                ensure_stream(active_stream, stream)?;
                let end = offset.checked_add(aligned).ok_or(out_of_memory())?;
                if end > pool.capacity {
                    return Err(out_of_memory());
                }
                let device_offset = u64::try_from(*offset).map_err(|_| out_of_memory())?;
                let pointer = pool
                    .pointer
                    .checked_add(device_offset)
                    .ok_or(out_of_memory())?;
                *offset = end;
                Ok(Some((pointer, pool.clone())))
            }
        }
    })
}

#[cfg(feature = "no-std")]
pub(crate) fn route_allocation(
    _stream: &Arc<CudaStream>,
    _bytes: usize,
) -> Result<Option<(sys::CUdeviceptr, Arc<CudaGraphMemoryPoolInner>)>, DriverError> {
    Ok(None)
}

fn align_up(bytes: usize) -> Option<usize> {
    bytes
        .checked_add(DEVICE_ALLOCATION_ALIGNMENT - 1)
        .map(|value| value & !(DEVICE_ALLOCATION_ALIGNMENT - 1))
}

#[cfg(not(feature = "no-std"))]
fn ensure_stream(expected: &Arc<CudaStream>, actual: &Arc<CudaStream>) -> Result<(), DriverError> {
    if Arc::ptr_eq(expected, actual) {
        Ok(())
    } else {
        Err(invalid_value())
    }
}

#[cfg(not(feature = "no-std"))]
fn take_probe() -> Result<usize, DriverError> {
    GRAPH_MEMORY_MODE.with(|mode| {
        let state = mode.borrow_mut().take();
        match state {
            Some(GraphMemoryMode::Probe { required, .. }) => Ok(required),
            other => {
                *mode.borrow_mut() = other;
                Err(invalid_value())
            }
        }
    })
}

#[cfg(not(feature = "no-std"))]
fn take_pool_session() -> Result<usize, DriverError> {
    GRAPH_MEMORY_MODE.with(|mode| {
        let state = mode.borrow_mut().take();
        match state {
            Some(GraphMemoryMode::Pool { offset, .. }) => Ok(offset),
            other => {
                *mode.borrow_mut() = other;
                Err(invalid_value())
            }
        }
    })
}

#[cfg(not(feature = "no-std"))]
fn clear_mode() {
    GRAPH_MEMORY_MODE.with(|mode| {
        mode.borrow_mut().take();
    });
}

fn invalid_value() -> DriverError {
    DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE)
}

fn out_of_memory() -> DriverError {
    DriverError(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)
}

#[cfg(test)]
mod tests {
    use super::align_up;

    #[test]
    fn graph_pool_allocations_use_cuda_alignment() {
        assert_eq!(align_up(0), Some(0));
        assert_eq!(align_up(1), Some(256));
        assert_eq!(align_up(256), Some(256));
        assert_eq!(align_up(257), Some(512));
        assert_eq!(align_up(usize::MAX), None);
    }
}
