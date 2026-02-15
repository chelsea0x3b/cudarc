//! Safe abstractions around [crate::cufft::result] with [CudaFft].

use super::{result, sys};
use crate::driver::{CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits};
use core::ffi::c_int;
use std::sync::Arc;

unsafe impl DeviceRepr for sys::float2 {}
unsafe impl ValidAsZeroBits for sys::float2 {}
unsafe impl DeviceRepr for sys::double2 {}
unsafe impl ValidAsZeroBits for sys::double2 {}

/// The direction of an FFT operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward (R2C / C2C forward) transform.
    Forward = -1,
    /// Inverse (C2R / C2C inverse) transform.
    Inverse = 1,
}

/// Safe wrapper around a cuFFT plan ([sys::cufftHandle]).
///
/// # Creating a plan
///
/// Use one of the plan constructors:
/// - [`CudaFft::plan_1d`] for 1-D transforms
/// - [`CudaFft::plan_2d`] for 2-D transforms
/// - [`CudaFft::plan_3d`] for 3-D transforms
/// - [`CudaFft::plan_many`] for advanced batched transforms
///
/// # Executing transforms
///
/// Each transform type has its own method:
/// - [`CudaFft::exec_r2c`] — `f32` real → complex
/// - [`CudaFft::exec_c2r`] — `f32` complex → real
/// - [`CudaFft::exec_c2c`] — `f32` complex ↔ complex
/// - [`CudaFft::exec_d2z`] — `f64` real → complex
/// - [`CudaFft::exec_z2d`] — `f64` complex → real
/// - [`CudaFft::exec_z2z`] — `f64` complex ↔ complex
///
/// # Example
///
/// ```rust
/// # use cudarc::{driver::*, cufft::*};
/// let ctx = CudaContext::new(0).unwrap();
/// let stream = ctx.default_stream();
/// let fft = CudaFft::plan_1d(256, cufft::sys::cufftType::CUFFT_C2C, 1, stream).unwrap();
/// ```
pub struct CudaFft {
    pub(crate) handle: sys::cufftHandle,
    pub(crate) stream: Arc<CudaStream>,
}

unsafe impl Send for CudaFft {}
unsafe impl Sync for CudaFft {}

impl CudaFft {
    /// Creates a simple 1-D FFT plan.
    ///
    /// - `nx`: The transform size (number of elements).
    /// - `type_`: The transform type (e.g. `CUFFT_C2C`, `CUFFT_R2C`).
    /// - `batch`: Number of transforms to perform in batch.
    pub fn plan_1d(
        nx: i32,
        type_: sys::cufftType,
        batch: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, result::CufftError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::plan_1d(nx, type_, batch)?;
        unsafe { result::set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    /// Creates a simple 2-D FFT plan.
    ///
    /// - `nx`, `ny`: The transform dimensions.
    /// - `type_`: The transform type.
    pub fn plan_2d(
        nx: i32,
        ny: i32,
        type_: sys::cufftType,
        stream: Arc<CudaStream>,
    ) -> Result<Self, result::CufftError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::plan_2d(nx, ny, type_)?;
        unsafe { result::set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    /// Creates a simple 3-D FFT plan.
    ///
    /// - `nx`, `ny`, `nz`: The transform dimensions.
    /// - `type_`: The transform type.
    pub fn plan_3d(
        nx: i32,
        ny: i32,
        nz: i32,
        type_: sys::cufftType,
        stream: Arc<CudaStream>,
    ) -> Result<Self, result::CufftError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::plan_3d(nx, ny, nz, type_)?;
        unsafe { result::set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    /// Creates an advanced batched FFT plan.
    ///
    /// This is the most flexible plan constructor and supports custom data layouts
    /// with strides, distances, and embedded dimensions.
    ///
    /// - `rank`: Dimensionality of the transform (1, 2, or 3).
    /// - `n`: Array of size `rank`, describing the size of each dimension.
    /// - `inembed`: Storage dimensions of the input data in memory (or `None` for default).
    /// - `istride`: Stride between consecutive input elements in the innermost dimension.
    /// - `idist`: Distance between the first elements of two consecutive input signals.
    /// - `onembed`: Storage dimensions of the output data in memory (or `None` for default).
    /// - `ostride`: Stride between consecutive output elements in the innermost dimension.
    /// - `odist`: Distance between the first elements of two consecutive output signals.
    /// - `type_`: The transform type (e.g. `CUFFT_R2C`, `CUFFT_C2R`).
    /// - `batch`: Number of transforms to perform.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_many(
        n: &mut [c_int],
        inembed: Option<&mut [c_int]>,
        istride: i32,
        idist: i32,
        onembed: Option<&mut [c_int]>,
        ostride: i32,
        odist: i32,
        type_: sys::cufftType,
        batch: i32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, result::CufftError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let rank = n.len() as c_int;
        let inembed_ptr = match inembed {
            Some(slice) => slice.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        let onembed_ptr = match onembed {
            Some(slice) => slice.as_mut_ptr(),
            None => std::ptr::null_mut(),
        };
        let handle = unsafe {
            result::plan_many(
                rank,
                n.as_mut_ptr(),
                inembed_ptr,
                istride,
                idist,
                onembed_ptr,
                ostride,
                odist,
                type_,
                batch,
            )
        }?;
        unsafe { result::set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    /// Returns the underlying cuFFT plan handle.
    pub fn handle(&self) -> sys::cufftHandle {
        self.handle
    }

    /// # Safety
    /// Users must ensure this stream is properly synchronized and belongs to
    /// the same CUDA context.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), result::CufftError> {
        self.stream = stream;
        result::set_stream(self.handle, self.stream.cu_stream() as _)
    }

    /// Execute a real-to-complex (R2C) transform (f32).
    ///
    /// `input` contains `f32` real values, `output` receives `sys::float2` complex values.
    pub fn exec_r2c<Src: DevicePtr<f32>, Dst: DevicePtrMut<sys::float2>>(
        &self,
        input: &Src,
        output: &mut Dst,
    ) -> Result<(), result::CufftError> {
        let (idata, _record_src) = input.device_ptr(&self.stream);
        let (odata, _record_dst) = output.device_ptr_mut(&self.stream);
        unsafe {
            result::exec_r2c(
                self.handle,
                idata as *mut sys::cufftReal,
                odata as *mut sys::cufftComplex,
            )
        }
    }

    /// Execute a complex-to-real (C2R) transform (f32).
    ///
    /// `input` contains `sys::float2` complex values, `output` receives `f32` real values.
    pub fn exec_c2r<Src: DevicePtrMut<sys::float2>, Dst: DevicePtrMut<f32>>(
        &self,
        input: &mut Src,
        output: &mut Dst,
    ) -> Result<(), result::CufftError> {
        let (idata, _record_src) = input.device_ptr_mut(&self.stream);
        let (odata, _record_dst) = output.device_ptr_mut(&self.stream);
        unsafe {
            result::exec_c2r(
                self.handle,
                idata as *mut sys::cufftComplex,
                odata as *mut sys::cufftReal,
            )
        }
    }

    /// Execute a complex-to-complex (C2C) transform (f32).
    ///
    /// `input` and `output` are `sys::float2` complex values.
    pub fn exec_c2c<Src: DevicePtrMut<sys::float2>, Dst: DevicePtrMut<sys::float2>>(
        &self,
        input: &mut Src,
        output: &mut Dst,
        direction: FftDirection,
    ) -> Result<(), result::CufftError> {
        let (idata, _record_src) = input.device_ptr_mut(&self.stream);
        let (odata, _record_dst) = output.device_ptr_mut(&self.stream);
        unsafe {
            result::exec_c2c(
                self.handle,
                idata as *mut sys::cufftComplex,
                odata as *mut sys::cufftComplex,
                direction as c_int,
            )
        }
    }

    /// Execute a real-to-complex (D2Z) transform (f64).
    ///
    /// `input` contains `f64` real values, `output` receives `sys::double2` complex values.
    pub fn exec_d2z<Src: DevicePtr<f64>, Dst: DevicePtrMut<sys::double2>>(
        &self,
        input: &Src,
        output: &mut Dst,
    ) -> Result<(), result::CufftError> {
        let (idata, _record_src) = input.device_ptr(&self.stream);
        let (odata, _record_dst) = output.device_ptr_mut(&self.stream);
        unsafe {
            result::exec_d2z(
                self.handle,
                idata as *mut sys::cufftDoubleReal,
                odata as *mut sys::cufftDoubleComplex,
            )
        }
    }

    /// Execute a complex-to-real (Z2D) transform (f64).
    ///
    /// `input` contains `sys::double2` complex values, `output` receives `f64` real values.
    pub fn exec_z2d<Src: DevicePtrMut<sys::double2>, Dst: DevicePtrMut<f64>>(
        &self,
        input: &mut Src,
        output: &mut Dst,
    ) -> Result<(), result::CufftError> {
        let (idata, _record_src) = input.device_ptr_mut(&self.stream);
        let (odata, _record_dst) = output.device_ptr_mut(&self.stream);
        unsafe {
            result::exec_z2d(
                self.handle,
                idata as *mut sys::cufftDoubleComplex,
                odata as *mut sys::cufftDoubleReal,
            )
        }
    }

    /// Execute a complex-to-complex (Z2Z) transform (f64).
    ///
    /// `input` and `output` are `sys::double2` complex values.
    pub fn exec_z2z<Src: DevicePtrMut<sys::double2>, Dst: DevicePtrMut<sys::double2>>(
        &self,
        input: &mut Src,
        output: &mut Dst,
        direction: FftDirection,
    ) -> Result<(), result::CufftError> {
        let (idata, _record_src) = input.device_ptr_mut(&self.stream);
        let (odata, _record_dst) = output.device_ptr_mut(&self.stream);
        unsafe {
            result::exec_z2z(
                self.handle,
                idata as *mut sys::cufftDoubleComplex,
                odata as *mut sys::cufftDoubleComplex,
                direction as c_int,
            )
        }
    }
}

impl Drop for CudaFft {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, 0);
        if handle != 0 {
            unsafe { result::destroy(handle) }.unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cufft::sys;
    use crate::driver::*;
    use std::vec::Vec;

    #[test]
    fn test_plan_1d_c2c() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let _fft = CudaFft::plan_1d(256, sys::cufftType::CUFFT_C2C, 1, stream.clone()).unwrap();
    }

    #[test]
    fn test_plan_2d_r2c() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let _fft = CudaFft::plan_2d(64, 64, sys::cufftType::CUFFT_R2C, stream.clone()).unwrap();
    }

    #[test]
    fn test_plan_many_r2c_batched() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let height = 4i32;
        let width = 8i32;
        let batch = 2i32;
        let w_half = width / 2 + 1;

        let _fft = CudaFft::plan_many(
            &mut [height, width],
            Some(&mut [height, width]),
            1,
            height * width,
            Some(&mut [height, w_half]),
            1,
            height * w_half,
            sys::cufftType::CUFFT_R2C,
            batch,
            stream.clone(),
        )
        .unwrap();
    }

    #[test]
    fn test_exec_c2c_roundtrip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let n = 4;
        // Create a C2C plan
        let fft = CudaFft::plan_1d(n as i32, sys::cufftType::CUFFT_C2C, 1, stream.clone()).unwrap();

        // Input: 4 complex numbers as float2 = [(1,0), (2,0), (3,0), (4,0)]
        let input_data: Vec<sys::float2> = (1..=n)
            .map(|i| sys::float2 {
                x: i as f32,
                y: 0.0,
            })
            .collect();

        let input_dev = stream.clone_htod(&input_data).unwrap();
        let mut freq_dev = stream.alloc_zeros::<sys::float2>(n).unwrap();
        let mut output_dev = stream.alloc_zeros::<sys::float2>(n).unwrap();

        // Forward transform
        let mut input_dev = input_dev;
        fft.exec_c2c(&mut input_dev, &mut freq_dev, FftDirection::Forward)
            .unwrap();

        // Inverse transform
        fft.exec_c2c(&mut freq_dev, &mut output_dev, FftDirection::Inverse)
            .unwrap();

        let output: Vec<sys::float2> = stream.clone_dtoh(&output_dev).unwrap();

        // After forward + inverse, values are scaled by n
        for i in 0..n {
            let expected = input_data[i].x * n as f32;
            assert!(
                (output[i].x - expected).abs() < 1e-3,
                "real mismatch at {i}: got {} expected {expected}",
                output[i].x
            );
            assert!(
                output[i].y.abs() < 1e-3,
                "imag mismatch at {i}: got {}",
                output[i].y
            );
        }
    }

    #[test]
    fn test_exec_r2c_c2r_roundtrip() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let n = 8;
        let n_complex = n / 2 + 1;

        // R2C plan
        let fft_r2c =
            CudaFft::plan_1d(n as i32, sys::cufftType::CUFFT_R2C, 1, stream.clone()).unwrap();
        // C2R plan
        let fft_c2r =
            CudaFft::plan_1d(n as i32, sys::cufftType::CUFFT_C2R, 1, stream.clone()).unwrap();

        let input_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
        let input_dev = stream.clone_htod(&input_data).unwrap();
        let mut freq_dev = stream.alloc_zeros::<sys::float2>(n_complex).unwrap();
        let mut output_dev = stream.alloc_zeros::<f32>(n).unwrap();

        // R2C
        fft_r2c.exec_r2c(&input_dev, &mut freq_dev).unwrap();

        // C2R
        fft_c2r.exec_c2r(&mut freq_dev, &mut output_dev).unwrap();

        let output: Vec<f32> = stream.clone_dtoh(&output_dev).unwrap();

        // After R2C + C2R, values are scaled by n
        for i in 0..n {
            let expected = input_data[i] * n as f32;
            assert!(
                (output[i] - expected).abs() < 1e-2,
                "mismatch at {i}: got {} expected {expected}",
                output[i]
            );
        }
    }
}
