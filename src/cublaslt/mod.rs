//! [CudaBlasLT] wraps around [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api) via:
//!
//! # Simple path
//!
//! 1. Instantiate a [CudaBlasLT] handle with [CudaBlasLT::new()]
//! 2. Execute a gemm using [CudaBlasLT::matmul()]
//!
//! # Advanced path: algorithm selection and preference control
//!
//! For control over algorithm selection (e.g., constraining workspace or pinning algorithms):
//!
//! 1. Instantiate a [CudaBlasLT] handle with [CudaBlasLT::new()]
//! 2. Create a [MatmulOperation] with [CudaBlasLT::matmul_op()]
//! 3. Create a [MatmulPreference] and configure constraints (e.g., [MatmulPreference::set_max_workspace_bytes()])
//! 4. Pick an algorithm via [MatmulOperation::pick_algorithm()] or [MatmulOperation::pick_algorithms()]
//! 5. Execute with [MatmulOperation::launch()]
//!
//! Algorithms can also be selected by ID via [MatmulOperation::get_algo_ids()] and
//! [MatmulOperation::algo_from_id()] for pinning a specific algorithm across runs.
//!
//! Note that all above apis work with [crate::driver::DevicePtr]/[crate::driver::DevicePtrMut], so they
//! accept [crate::driver::CudaSlice], [crate::driver::CudaView], and [crate::driver::CudaViewMut].

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
