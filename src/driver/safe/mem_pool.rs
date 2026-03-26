use std::sync::Arc;

use crate::driver::{result, sys};

use super::{CudaContext, DriverError};

/// A CUDA memory pool. Wraps a [sys::CUmemoryPool].
///
/// Pools let you control how GPU memory is allocated and reused.
/// Memory allocated from a pool via [CudaStream::alloc_from_pool()]
/// is returned to the pool (not the OS) when freed, reducing
/// allocation overhead.
///
/// Create a custom pool with [CudaContext::create_mem_pool()], or
/// get the device default pool with [CudaContext::default_mem_pool()].
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html)
#[derive(Debug)]
pub struct CudaMemPool {
    pub(crate) pool: sys::CUmemoryPool,
    pub(crate) ctx: Arc<CudaContext>,
    /// If true, this pool was created by us and will be destroyed on Drop.
    /// If false, this is a reference to the device default pool and must not
    /// be destroyed.
    owned: bool,
}

unsafe impl Send for CudaMemPool {}
unsafe impl Sync for CudaMemPool {}

impl Drop for CudaMemPool {
    fn drop(&mut self) {
        if self.owned {
            let pool = std::mem::replace(&mut self.pool, std::ptr::null_mut());
            if !pool.is_null() {
                self.ctx.record_err(self.ctx.bind_to_thread());
                self.ctx
                    .record_err(unsafe { result::mem_pool::destroy(pool) });
            }
        }
    }
}

impl CudaMemPool {
    /// Release unused memory held by the pool back to the OS.
    ///
    /// `min_bytes_to_keep` sets a floor — the pool will retain at least
    /// this much memory even if it is unused.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g6b3f1ea779bda578c8e26101caa3d958)
    pub fn trim_to(&self, min_bytes_to_keep: usize) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::mem_pool::trim_to(self.pool, min_bytes_to_keep) }
    }

    /// Get the underlying [sys::CUmemoryPool].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not destroy the pool**, as it is still
    /// owned by the [CudaMemPool].
    pub fn cu_mem_pool(&self) -> sys::CUmemoryPool {
        self.pool
    }
}

impl CudaContext {
    /// Get the default memory pool for this device.
    ///
    /// The returned pool is **not owned** — dropping it will not destroy
    /// the device default pool.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g2170a6e24f7e596854f0c48e1e98120e)
    pub fn default_mem_pool(self: &Arc<Self>) -> Result<CudaMemPool, DriverError> {
        self.check_err()?;
        let pool = unsafe { result::device::get_default_mem_pool(self.cu_device) }?;
        Ok(CudaMemPool {
            pool,
            ctx: self.clone(),
            owned: false,
        })
    }

    /// Create a new memory pool on this device.
    ///
    /// The returned pool is **owned** — it will be destroyed when dropped.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa7ef8b06b0df48350794e1e8bba704)
    pub fn create_mem_pool(self: &Arc<Self>) -> Result<CudaMemPool, DriverError> {
        self.bind_to_thread()?;
        let props = sys::CUmemPoolProps {
            allocType: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.ordinal as i32,
            },
            win32SecurityAttributes: std::ptr::null_mut(),
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000",
                feature = "cuda-13010",
            ))]
            maxSize: 0,
            #[cfg(any(
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000",
                feature = "cuda-13010",
            ))]
            usage: 0,
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
            ))]
            reserved: [0u8; 64],
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
            ))]
            reserved: [0u8; 56],
            #[cfg(any(
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000",
                feature = "cuda-13010",
            ))]
            reserved: [0u8; 54],
        };
        let pool = unsafe { result::mem_pool::create(&props) }?;
        Ok(CudaMemPool {
            pool,
            ctx: self.clone(),
            owned: true,
        })
    }
}
