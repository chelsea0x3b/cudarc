//! Wrappers around the [NVIDIA OptiX Ray Tracing Engine API](https://developer.nvidia.com/rtx/ray-tracing/optix),
//! organized into three levels: [`safe`], [`result`], and [`sys`].
//!
//! # Architecture & Runtime Dynamic Loading
//!
//! `libnvoptix.so.1` (Linux) and `nvoptix.dll` (Windows) ship directly inside the standard
//! NVIDIA display driver alongside `libcuda.so.1` / `nvcuda.dll`. Unlike `libcuda`, `libnvoptix`
//! exports a single entry point, [`sys::optixQueryFunctionTable`], which populates an
//! ABI-versioned [`sys::OptixFunctionTable`] at runtime with zero build-time C/C++ compiler,
//! CMake, or OptiX SDK header dependencies.
//!
//! # Safe API Usage
//!
//! 1. Check availability and create an [`OptixContext`] from a [`crate::driver::CudaContext`]:
//!
//! ```no_run
//! use cudarc::driver::CudaContext;
//! use cudarc::optix::OptixContext;
//!
//! if OptixContext::is_available() {
//!     let cuda_ctx = CudaContext::new(0).unwrap();
//!     let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();
//! }
//! ```
//!
//! 2. Build a hardware Bottom-Level Acceleration Structure (BLAS) and Top-Level Acceleration
//!    Structure (TLAS) with [`OptixAccel`]:
//!
//! ```no_run
//! use cudarc::driver::CudaContext;
//! use cudarc::optix::{sys, OptixAccel, OptixContext};
//!
//! let cuda_ctx = CudaContext::new(0).unwrap();
//! let stream = cuda_ctx.default_stream();
//! let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();
//!
//! let vertices = stream
//!     .clone_htod(&[[-1.0f32, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
//!     .unwrap();
//! let indices = stream.clone_htod(&[[0u32, 1, 2]]).unwrap();
//! let blas = OptixAccel::build_triangles(
//!     &optix_ctx,
//!     &stream,
//!     &vertices,
//!     Some(&indices),
//!     sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
//!     None,
//! ).unwrap();
//! assert_ne!(blas.handle(), 0);
//!
//! let instance = sys::OptixInstance::with_identity(blas.handle(), 0);
//! let d_instances = stream.clone_htod(&[instance]).unwrap();
//! let tlas = OptixAccel::build_instances(&optix_ctx, &stream, &d_instances, None).unwrap();
//! assert_ne!(tlas.handle(), 0);
//! ```
//!
//! 3. Compile an [`OptixModule`] from PTX, create [`OptixProgramGroup`]s (`raygen`, `miss`, `hitgroup`),
//!    pack an [`OptixSbt`], link an [`OptixPipeline`], and launch hardware RT Core ray tracing via
//!    [`OptixPipeline::launch`]. See `examples/optix-raytrace.rs` for a complete runnable example.

#![warn(missing_docs)]

pub mod result;
pub mod safe;
#[allow(warnings)]
#[rustfmt::skip]
pub mod sys;

pub use result::OptixError;
pub use safe::*;
