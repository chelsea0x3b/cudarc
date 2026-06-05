#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
use std::sync::OnceLock;
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
#[cfg(feature = "dynamic-loading")]
fn load<F: Copy>(name: &str) -> F {
    unsafe { *culib().get::<F>(name.as_bytes()).unwrap_or_else(|e| panic!("Missing symbol {name}: {e}")) }
}
pub use self::cudaDataType_t as cudaDataType;
pub type cudaLibMgGrid_t = *mut ::core::ffi::c_void;
pub type cudaLibMgMatrixDesc_t = *mut ::core::ffi::c_void;
pub type cusolverMgHandle_t = *mut cusolverMgContext;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_CONJG = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
    CUDA_R_8F_UE8M0 = 30,
    CUDA_R_6F_E2M3 = 31,
    CUDA_R_6F_E3M2 = 32,
    CUDA_R_4F_E2M1 = 33,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigMode_t {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverMgGridMapping_t {
    CUDALIBMG_GRID_MAPPING_ROW_MAJOR = 1,
    CUDALIBMG_GRID_MAPPING_COL_MAJOR = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverStatus_t {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_MAPPING_ERROR = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT = 10,
    CUSOLVER_STATUS_INVALID_LICENSE = 11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30,
    CUSOLVER_STATUS_INVALID_WORKSPACE = 31,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverMgContext {
    _unused: [u8; 0],
}
impl cublasOperation_t {
    pub const CUBLAS_OP_HERMITAN: cublasOperation_t = cublasOperation_t::CUBLAS_OP_C;
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
pub unsafe fn cusolverMgCreate(handle: *mut cusolverMgHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverMgHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgCreate(handle: *mut cusolverMgHandle_t) -> cusolverStatus_t;
        }
        cusolverMgCreate(handle)
    }
}
pub unsafe fn cusolverMgCreateDeviceGrid(grid: *mut cudaLibMgGrid_t, numRowDevices: i32, numColDevices: i32, deviceId: *const i32, mapping: cusolverMgGridMapping_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLibMgGrid_t, i32, i32, *const i32, cusolverMgGridMapping_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgCreateDeviceGrid") });
        _f(grid, numRowDevices, numColDevices, deviceId, mapping)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgCreateDeviceGrid(grid: *mut cudaLibMgGrid_t, numRowDevices: i32, numColDevices: i32, deviceId: *const i32, mapping: cusolverMgGridMapping_t) -> cusolverStatus_t;
        }
        cusolverMgCreateDeviceGrid(grid, numRowDevices, numColDevices, deviceId, mapping)
    }
}
pub unsafe fn cusolverMgCreateMatrixDesc(desc: *mut cudaLibMgMatrixDesc_t, numRows: i64, numCols: i64, rowBlockSize: i64, colBlockSize: i64, dataType: cudaDataType, grid: cudaLibMgGrid_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLibMgMatrixDesc_t, i64, i64, i64, i64, cudaDataType, cudaLibMgGrid_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgCreateMatrixDesc") });
        _f(desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgCreateMatrixDesc(desc: *mut cudaLibMgMatrixDesc_t, numRows: i64, numCols: i64, rowBlockSize: i64, colBlockSize: i64, dataType: cudaDataType, grid: cudaLibMgGrid_t) -> cusolverStatus_t;
        }
        cusolverMgCreateMatrixDesc(desc, numRows, numCols, rowBlockSize, colBlockSize, dataType, grid)
    }
}
pub unsafe fn cusolverMgDestroy(handle: cusolverMgHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgDestroy") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgDestroy(handle: cusolverMgHandle_t) -> cusolverStatus_t;
        }
        cusolverMgDestroy(handle)
    }
}
pub unsafe fn cusolverMgDestroyGrid(grid: cudaLibMgGrid_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLibMgGrid_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgDestroyGrid") });
        _f(grid)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgDestroyGrid(grid: cudaLibMgGrid_t) -> cusolverStatus_t;
        }
        cusolverMgDestroyGrid(grid)
    }
}
pub unsafe fn cusolverMgDestroyMatrixDesc(desc: cudaLibMgMatrixDesc_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLibMgMatrixDesc_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgDestroyMatrixDesc") });
        _f(desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgDestroyMatrixDesc(desc: cudaLibMgMatrixDesc_t) -> cusolverStatus_t;
        }
        cusolverMgDestroyMatrixDesc(desc)
    }
}
pub unsafe fn cusolverMgDeviceSelect(handle: cusolverMgHandle_t, nbDevices: ::core::ffi::c_int, deviceId: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgDeviceSelect") });
        _f(handle, nbDevices, deviceId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgDeviceSelect(handle: cusolverMgHandle_t, nbDevices: ::core::ffi::c_int, deviceId: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgDeviceSelect(handle, nbDevices, deviceId)
    }
}
pub unsafe fn cusolverMgGetrf(handle: cusolverMgHandle_t, M: ::core::ffi::c_int, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut *mut ::core::ffi::c_int, cudaDataType, *mut *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgGetrf") });
        _f(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, array_d_work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgGetrf(handle: cusolverMgHandle_t, M: ::core::ffi::c_int, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgGetrf(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, array_d_work, lwork, info)
    }
}
pub unsafe fn cusolverMgGetrf_bufferSize(handle: cusolverMgHandle_t, M: ::core::ffi::c_int, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut *mut ::core::ffi::c_int, cudaDataType, *mut i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgGetrf_bufferSize") });
        _f(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgGetrf_bufferSize(handle: cusolverMgHandle_t, M: ::core::ffi::c_int, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t;
        }
        cusolverMgGetrf_bufferSize(handle, M, N, array_d_A, IA, JA, descrA, array_d_IPIV, computeType, lwork)
    }
}
pub unsafe fn cusolverMgGetrs(handle: cusolverMgHandle_t, TRANS: cublasOperation_t, N: ::core::ffi::c_int, NRHS: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgGetrs") });
        _f(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgGetrs(handle: cusolverMgHandle_t, TRANS: cublasOperation_t, N: ::core::ffi::c_int, NRHS: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgGetrs(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, info)
    }
}
pub unsafe fn cusolverMgGetrs_bufferSize(handle: cusolverMgHandle_t, TRANS: cublasOperation_t, N: ::core::ffi::c_int, NRHS: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgGetrs_bufferSize") });
        _f(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgGetrs_bufferSize(handle: cusolverMgHandle_t, TRANS: cublasOperation_t, N: ::core::ffi::c_int, NRHS: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_IPIV: *mut *mut ::core::ffi::c_int, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t;
        }
        cusolverMgGetrs_bufferSize(handle, TRANS, N, NRHS, array_d_A, IA, JA, descrA, array_d_IPIV, array_d_B, IB, JB, descrB, computeType, lwork)
    }
}
pub unsafe fn cusolverMgPotrf(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, h_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgPotrf") });
        _f(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgPotrf(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, h_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgPotrf(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
    }
}
pub unsafe fn cusolverMgPotrf_bufferSize(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgPotrf_bufferSize") });
        _f(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgPotrf_bufferSize(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t;
        }
        cusolverMgPotrf_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
    }
}
pub unsafe fn cusolverMgPotri(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, h_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgPotri") });
        _f(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgPotri(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, h_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgPotri(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, array_d_work, lwork, h_info)
    }
}
pub unsafe fn cusolverMgPotri_bufferSize(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgPotri_bufferSize") });
        _f(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgPotri_bufferSize(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t;
        }
        cusolverMgPotri_bufferSize(handle, uplo, N, array_d_A, IA, JA, descrA, computeType, lwork)
    }
}
pub unsafe fn cusolverMgPotrs(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, h_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgPotrs") });
        _f(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, h_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgPotrs(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, h_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgPotrs(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, array_d_work, lwork, h_info)
    }
}
pub unsafe fn cusolverMgPotrs_bufferSize(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, cudaDataType, *mut i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgPotrs_bufferSize") });
        _f(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgPotrs_bufferSize(handle: cusolverMgHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, array_d_B: *mut *mut ::core::ffi::c_void, IB: ::core::ffi::c_int, JB: ::core::ffi::c_int, descrB: cudaLibMgMatrixDesc_t, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t;
        }
        cusolverMgPotrs_bufferSize(handle, uplo, n, nrhs, array_d_A, IA, JA, descrA, array_d_B, IB, JB, descrB, computeType, lwork)
    }
}
pub unsafe fn cusolverMgSyevd(handle: cusolverMgHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, W: *mut ::core::ffi::c_void, dataTypeW: cudaDataType, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut ::core::ffi::c_void, cudaDataType, cudaDataType, *mut *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgSyevd") });
        _f(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, array_d_work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgSyevd(handle: cusolverMgHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, W: *mut ::core::ffi::c_void, dataTypeW: cudaDataType, computeType: cudaDataType, array_d_work: *mut *mut ::core::ffi::c_void, lwork: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverMgSyevd(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, array_d_work, lwork, info)
    }
}
pub unsafe fn cusolverMgSyevd_bufferSize(handle: cusolverMgHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, W: *mut ::core::ffi::c_void, dataTypeW: cudaDataType, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverMgHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int, cudaLibMgMatrixDesc_t, *mut ::core::ffi::c_void, cudaDataType, cudaDataType, *mut i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverMgSyevd_bufferSize") });
        _f(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverMgSyevd_bufferSize(handle: cusolverMgHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, N: ::core::ffi::c_int, array_d_A: *mut *mut ::core::ffi::c_void, IA: ::core::ffi::c_int, JA: ::core::ffi::c_int, descrA: cudaLibMgMatrixDesc_t, W: *mut ::core::ffi::c_void, dataTypeW: cudaDataType, computeType: cudaDataType, lwork: *mut i64) -> cusolverStatus_t;
        }
        cusolverMgSyevd_bufferSize(handle, jobz, uplo, N, array_d_A, IA, JA, descrA, W, dataTypeW, computeType, lwork)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cusolverMg"];
    let choices = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten();
    for choice in choices {
        if ::libloading::Library::new(choice).is_ok() {
            return true;
        }
    }
    false
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn culib() -> &'static ::libloading::Library {
    static LIB: OnceLock<::libloading::Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let lib_names = std::vec!["cusolverMg"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
