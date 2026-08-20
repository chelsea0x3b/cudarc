//! Safe abstractions around [crate::cublas::result] for doing gemm and gemv.
#![allow(clippy::too_many_arguments)]

use super::{result, result::CublasError, sys};
use crate::driver::{CudaSlice, CudaStream, DevicePtrMut};
use std::sync::Arc;

mod asum;
mod gemm;
mod gemv;
mod grouped_gemm;

pub use asum::*;
pub use gemm::*;
pub use gemv::*;
pub use grouped_gemm::*;

/// Wrapper around [sys::cublasHandle_t]
///
/// 1. Create with [CudaBlas::new()]
/// 2. Execute gemm/gemv/gmm kernels with [Gemv], [Gemm] and [Gmm]. Both f32 and f64 are supported
///    for [Gemm] and [Gemv], f16 and bf16 are supported for [Gmm] if feature `half` is activated.
///
/// Note: This maintains a instance of [`Arc<CudaDevice>`], so will prevent the device
/// from being dropped.
#[derive(Debug)]
pub struct CudaBlas {
    pub(crate) handle: sys::cublasHandle_t,
    pub(crate) stream: Arc<CudaStream>,
    workspace: Option<CudaSlice<u8>>,
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl CudaBlas {
    /// Creates a new cublas handle and sets the stream to the `device`'s stream.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::create_handle()?;
        unsafe { result::set_stream(handle, stream.cu_stream() as _) }?;
        let blas = Self {
            handle,
            stream,
            workspace: None,
        };
        Ok(blas)
    }

    /// Returns a reference to the underlying cublas handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// Sets the handle's current to either the stream specified, or the device's default work
    /// stream.
    ///
    /// # Safety
    /// This is unsafe because you can end up scheduling multiple concurrent kernels that all
    /// write to the same memory address.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CublasError> {
        self.stream = stream;
        unsafe { result::set_stream(self.handle, self.stream.cu_stream() as _) }?;
        // `cublasSetStream` unconditionally resets the workspace to cuBLAS's
        // default pool, so a caller-owned workspace must be bound again.
        if let Some(workspace) = self.workspace.as_mut() {
            bind_workspace(self.handle, &self.stream, workspace)?;
        }
        Ok(())
    }

    /// Assigns a caller-owned workspace to this handle and retains it for the
    /// handle's lifetime, including across [CudaBlas::set_stream()].
    /// Assigning workspace before CUDA Graph capture keeps cuBLAS from
    /// recording internal allocation and free nodes into the graph.
    /// ref: <https://docs.nvidia.com/cuda/cublas/#cublassetworkspace>
    ///
    /// # Safety
    /// Any previously assigned workspace is dropped, so no kernel launched on
    /// this handle may still be using it.
    pub unsafe fn set_workspace(
        &mut self,
        mut workspace: CudaSlice<u8>,
    ) -> Result<(), CublasError> {
        bind_workspace(self.handle, &self.stream, &mut workspace)?;
        self.workspace = Some(workspace);
        Ok(())
    }

    /// Set the handle's pointer mode.
    /// ref: <https://docs.nvidia.com/cuda/cublas/#cublassetpointermode>
    ///
    /// Some cublas functions require the pointer mode to be set to `cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE`
    /// when passing a device memory result buffer into the function, such as `cublas<t>asum()`.
    /// Otherwise the operation will panic with `SIGSEGV: invalid memory reference`,
    /// or one has to use a host memory reference, which has performance implications.
    pub fn set_pointer_mode(
        &self,
        pointer_mode: sys::cublasPointerMode_t,
    ) -> Result<(), CublasError> {
        unsafe {
            sys::cublasSetPointerMode_v2(self.handle, pointer_mode).result()?;
        }
        Ok(())
    }

    /// Get the handle's current pointer mode.
    /// ref: <https://docs.nvidia.com/cuda/cublas/#cublasgetpointermode>
    pub fn get_pointer_mode(&self) -> Result<sys::cublasPointerMode_t, CublasError> {
        unsafe {
            let mut mode = ::core::mem::MaybeUninit::uninit();
            sys::cublasGetPointerMode_v2(self.handle, mode.as_mut_ptr()).result()?;
            Ok(mode.assume_init())
        }
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

fn bind_workspace(
    handle: sys::cublasHandle_t,
    stream: &Arc<CudaStream>,
    workspace: &mut CudaSlice<u8>,
) -> Result<(), CublasError> {
    // `cublasSetWorkspace` accepts a pointer from another context without
    // error, so the mismatch has to be caught here.
    if !Arc::ptr_eq(workspace.context(), stream.context()) {
        return Err(CublasError(
            sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
        ));
    }
    let workspace_size = workspace.num_bytes();
    let (workspace_ptr, _workspace_access) = workspace.device_ptr_mut(stream);
    unsafe {
        result::set_workspace(
            handle,
            workspace_ptr as *mut core::ffi::c_void,
            workspace_size,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use crate::driver::CudaContext;

    use super::*;

    #[test]
    fn cublas_pointer_mode() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).unwrap();
        assert_eq!(
            blas.get_pointer_mode().unwrap(),
            sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            "The default pointer mode uses host pointers"
        );

        blas.set_pointer_mode(sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE)
            .unwrap();
        assert_eq!(
            blas.get_pointer_mode().unwrap(),
            sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            "We have set the mode to use device pointers"
        );
    }
}
