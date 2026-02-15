//! [CudaFft] safe bindings around [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html).
//!
//! Create an FFT plan with [`CudaFft::plan_1d`], [`CudaFft::plan_2d`], [`CudaFft::plan_3d`],
//! or [`CudaFft::plan_many`], then execute with the appropriate method:
//! - [`CudaFft::exec_r2c`] / [`CudaFft::exec_c2r`] for f32 real ↔ complex
//! - [`CudaFft::exec_c2c`] for f32 complex ↔ complex
//! - [`CudaFft::exec_d2z`] / [`CudaFft::exec_z2d`] for f64 real ↔ complex
//! - [`CudaFft::exec_z2z`] for f64 complex ↔ complex

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
