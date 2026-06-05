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
pub use self::cufftCompatibility_t as cufftCompatibility;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cufftProperty_t as cufftProperty;
pub use self::cufftResult_t as cufftResult;
pub use self::cufftType_t as cufftType;
pub use self::libraryPropertyType_t as libraryPropertyType;
pub type cuComplex = cuFloatComplex;
pub type cuDoubleComplex = double2;
pub type cuFloatComplex = float2;
pub type cudaStream_t = *mut CUstream_st;
pub type cufftComplex = cuComplex;
pub type cufftDoubleComplex = cuDoubleComplex;
pub type cufftDoubleReal = f64;
pub type cufftHandle = ::core::ffi::c_int;
pub type cufftReal = f32;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftCompatibility_t {
    CUFFT_COMPATIBILITY_FFTW_PADDING = 1,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftProperty_t {
    NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 1,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftProperty_t {
    NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 1,
    NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = 2,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftResult_t {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftResult_t {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_NOT_SUPPORTED = 16,
    CUFFT_MISSING_DEPENDENCY = 17,
    CUFFT_NVRTC_FAILURE = 18,
    CUFFT_NVJITLINK_FAILURE = 19,
    CUFFT_NVSHMEM_FAILURE = 20,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftType_t {
    CUFFT_R2C = 42,
    CUFFT_C2R = 44,
    CUFFT_C2C = 41,
    CUFFT_D2Z = 106,
    CUFFT_Z2D = 108,
    CUFFT_Z2Z = 105,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
#[repr(C)]
#[repr(align(8))]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
pub unsafe fn cufftCreate(handle: *mut cufftHandle) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cufftHandle) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftCreate(handle: *mut cufftHandle) -> cufftResult;
        }
        cufftCreate(handle)
    }
}
pub unsafe fn cufftDestroy(plan: cufftHandle) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftDestroy") });
        _f(plan)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftDestroy(plan: cufftHandle) -> cufftResult;
        }
        cufftDestroy(plan)
    }
}
pub unsafe fn cufftEstimate1d(nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, cufftType, ::core::ffi::c_int, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftEstimate1d") });
        _f(nx, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftEstimate1d(nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult;
        }
        cufftEstimate1d(nx, type_, batch, workSize)
    }
}
pub unsafe fn cufftEstimate2d(nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, cufftType, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftEstimate2d") });
        _f(nx, ny, type_, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftEstimate2d(nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult;
        }
        cufftEstimate2d(nx, ny, type_, workSize)
    }
}
pub unsafe fn cufftEstimate3d(nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftEstimate3d") });
        _f(nx, ny, nz, type_, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftEstimate3d(nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult;
        }
        cufftEstimate3d(nx, ny, nz, type_, workSize)
    }
}
pub unsafe fn cufftEstimateMany(rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, ::core::ffi::c_int, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftEstimateMany") });
        _f(rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftEstimateMany(rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult;
        }
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
}
pub unsafe fn cufftExecC2C(plan: cufftHandle, idata: *mut cufftComplex, odata: *mut cufftComplex, direction: ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut cufftComplex, *mut cufftComplex, ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftExecC2C") });
        _f(plan, idata, odata, direction)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftExecC2C(plan: cufftHandle, idata: *mut cufftComplex, odata: *mut cufftComplex, direction: ::core::ffi::c_int) -> cufftResult;
        }
        cufftExecC2C(plan, idata, odata, direction)
    }
}
pub unsafe fn cufftExecC2R(plan: cufftHandle, idata: *mut cufftComplex, odata: *mut cufftReal) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut cufftComplex, *mut cufftReal) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftExecC2R") });
        _f(plan, idata, odata)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftExecC2R(plan: cufftHandle, idata: *mut cufftComplex, odata: *mut cufftReal) -> cufftResult;
        }
        cufftExecC2R(plan, idata, odata)
    }
}
pub unsafe fn cufftExecD2Z(plan: cufftHandle, idata: *mut cufftDoubleReal, odata: *mut cufftDoubleComplex) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut cufftDoubleReal, *mut cufftDoubleComplex) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftExecD2Z") });
        _f(plan, idata, odata)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftExecD2Z(plan: cufftHandle, idata: *mut cufftDoubleReal, odata: *mut cufftDoubleComplex) -> cufftResult;
        }
        cufftExecD2Z(plan, idata, odata)
    }
}
pub unsafe fn cufftExecR2C(plan: cufftHandle, idata: *mut cufftReal, odata: *mut cufftComplex) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut cufftReal, *mut cufftComplex) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftExecR2C") });
        _f(plan, idata, odata)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftExecR2C(plan: cufftHandle, idata: *mut cufftReal, odata: *mut cufftComplex) -> cufftResult;
        }
        cufftExecR2C(plan, idata, odata)
    }
}
pub unsafe fn cufftExecZ2D(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut cufftDoubleReal) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut cufftDoubleComplex, *mut cufftDoubleReal) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftExecZ2D") });
        _f(plan, idata, odata)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftExecZ2D(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut cufftDoubleReal) -> cufftResult;
        }
        cufftExecZ2D(plan, idata, odata)
    }
}
pub unsafe fn cufftExecZ2Z(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut cufftDoubleComplex, direction: ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut cufftDoubleComplex, *mut cufftDoubleComplex, ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftExecZ2Z") });
        _f(plan, idata, odata, direction)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftExecZ2Z(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut cufftDoubleComplex, direction: ::core::ffi::c_int) -> cufftResult;
        }
        cufftExecZ2Z(plan, idata, odata, direction)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cufftGetPlanPropertyInt64(plan: cufftHandle, property: cufftProperty, returnPtrValue: *mut ::core::ffi::c_longlong) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, cufftProperty, *mut ::core::ffi::c_longlong) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetPlanPropertyInt64") });
        _f(plan, property, returnPtrValue)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetPlanPropertyInt64(plan: cufftHandle, property: cufftProperty, returnPtrValue: *mut ::core::ffi::c_longlong) -> cufftResult;
        }
        cufftGetPlanPropertyInt64(plan, property, returnPtrValue)
    }
}
pub unsafe fn cufftGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(libraryPropertyType, *mut ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetProperty") });
        _f(type_, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cufftResult;
        }
        cufftGetProperty(type_, value)
    }
}
pub unsafe fn cufftGetSize(handle: cufftHandle, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetSize") });
        _f(handle, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetSize(handle: cufftHandle, workSize: *mut usize) -> cufftResult;
        }
        cufftGetSize(handle, workSize)
    }
}
pub unsafe fn cufftGetSize1d(handle: cufftHandle, nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, cufftType, ::core::ffi::c_int, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetSize1d") });
        _f(handle, nx, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetSize1d(handle: cufftHandle, nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult;
        }
        cufftGetSize1d(handle, nx, type_, batch, workSize)
    }
}
pub unsafe fn cufftGetSize2d(handle: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetSize2d") });
        _f(handle, nx, ny, type_, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetSize2d(handle: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult;
        }
        cufftGetSize2d(handle, nx, ny, type_, workSize)
    }
}
pub unsafe fn cufftGetSize3d(handle: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetSize3d") });
        _f(handle, nx, ny, nz, type_, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetSize3d(handle: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult;
        }
        cufftGetSize3d(handle, nx, ny, nz, type_, workSize)
    }
}
pub unsafe fn cufftGetSizeMany(handle: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workArea: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, ::core::ffi::c_int, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetSizeMany") });
        _f(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workArea)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetSizeMany(handle: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workArea: *mut usize) -> cufftResult;
        }
        cufftGetSizeMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workArea)
    }
}
pub unsafe fn cufftGetSizeMany64(plan: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_longlong, inembed: *mut ::core::ffi::c_longlong, istride: ::core::ffi::c_longlong, idist: ::core::ffi::c_longlong, onembed: *mut ::core::ffi::c_longlong, ostride: ::core::ffi::c_longlong, odist: ::core::ffi::c_longlong, type_: cufftType, batch: ::core::ffi::c_longlong, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, *mut ::core::ffi::c_longlong, *mut ::core::ffi::c_longlong, ::core::ffi::c_longlong, ::core::ffi::c_longlong, *mut ::core::ffi::c_longlong, ::core::ffi::c_longlong, ::core::ffi::c_longlong, cufftType, ::core::ffi::c_longlong, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetSizeMany64") });
        _f(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetSizeMany64(plan: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_longlong, inembed: *mut ::core::ffi::c_longlong, istride: ::core::ffi::c_longlong, idist: ::core::ffi::c_longlong, onembed: *mut ::core::ffi::c_longlong, ostride: ::core::ffi::c_longlong, odist: ::core::ffi::c_longlong, type_: cufftType, batch: ::core::ffi::c_longlong, workSize: *mut usize) -> cufftResult;
        }
        cufftGetSizeMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
}
pub unsafe fn cufftGetVersion(version: *mut ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftGetVersion") });
        _f(version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftGetVersion(version: *mut ::core::ffi::c_int) -> cufftResult;
        }
        cufftGetVersion(version)
    }
}
pub unsafe fn cufftMakePlan1d(plan: cufftHandle, nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, cufftType, ::core::ffi::c_int, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftMakePlan1d") });
        _f(plan, nx, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftMakePlan1d(plan: cufftHandle, nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult;
        }
        cufftMakePlan1d(plan, nx, type_, batch, workSize)
    }
}
pub unsafe fn cufftMakePlan2d(plan: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftMakePlan2d") });
        _f(plan, nx, ny, type_, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftMakePlan2d(plan: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult;
        }
        cufftMakePlan2d(plan, nx, ny, type_, workSize)
    }
}
pub unsafe fn cufftMakePlan3d(plan: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftMakePlan3d") });
        _f(plan, nx, ny, nz, type_, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftMakePlan3d(plan: cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType, workSize: *mut usize) -> cufftResult;
        }
        cufftMakePlan3d(plan, nx, ny, nz, type_, workSize)
    }
}
pub unsafe fn cufftMakePlanMany(plan: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, ::core::ffi::c_int, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftMakePlanMany") });
        _f(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftMakePlanMany(plan: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int, workSize: *mut usize) -> cufftResult;
        }
        cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
}
pub unsafe fn cufftMakePlanMany64(plan: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_longlong, inembed: *mut ::core::ffi::c_longlong, istride: ::core::ffi::c_longlong, idist: ::core::ffi::c_longlong, onembed: *mut ::core::ffi::c_longlong, ostride: ::core::ffi::c_longlong, odist: ::core::ffi::c_longlong, type_: cufftType, batch: ::core::ffi::c_longlong, workSize: *mut usize) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int, *mut ::core::ffi::c_longlong, *mut ::core::ffi::c_longlong, ::core::ffi::c_longlong, ::core::ffi::c_longlong, *mut ::core::ffi::c_longlong, ::core::ffi::c_longlong, ::core::ffi::c_longlong, cufftType, ::core::ffi::c_longlong, *mut usize) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftMakePlanMany64") });
        _f(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftMakePlanMany64(plan: cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_longlong, inembed: *mut ::core::ffi::c_longlong, istride: ::core::ffi::c_longlong, idist: ::core::ffi::c_longlong, onembed: *mut ::core::ffi::c_longlong, ostride: ::core::ffi::c_longlong, odist: ::core::ffi::c_longlong, type_: cufftType, batch: ::core::ffi::c_longlong, workSize: *mut usize) -> cufftResult;
        }
        cufftMakePlanMany64(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize)
    }
}
pub unsafe fn cufftPlan1d(plan: *mut cufftHandle, nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cufftHandle, ::core::ffi::c_int, cufftType, ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftPlan1d") });
        _f(plan, nx, type_, batch)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftPlan1d(plan: *mut cufftHandle, nx: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int) -> cufftResult;
        }
        cufftPlan1d(plan, nx, type_, batch)
    }
}
pub unsafe fn cufftPlan2d(plan: *mut cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cufftHandle, ::core::ffi::c_int, ::core::ffi::c_int, cufftType) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftPlan2d") });
        _f(plan, nx, ny, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftPlan2d(plan: *mut cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, type_: cufftType) -> cufftResult;
        }
        cufftPlan2d(plan, nx, ny, type_)
    }
}
pub unsafe fn cufftPlan3d(plan: *mut cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cufftHandle, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftPlan3d") });
        _f(plan, nx, ny, nz, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftPlan3d(plan: *mut cufftHandle, nx: ::core::ffi::c_int, ny: ::core::ffi::c_int, nz: ::core::ffi::c_int, type_: cufftType) -> cufftResult;
        }
        cufftPlan3d(plan, nx, ny, nz, type_)
    }
}
pub unsafe fn cufftPlanMany(plan: *mut cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cufftHandle, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cufftType, ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftPlanMany") });
        _f(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftPlanMany(plan: *mut cufftHandle, rank: ::core::ffi::c_int, n: *mut ::core::ffi::c_int, inembed: *mut ::core::ffi::c_int, istride: ::core::ffi::c_int, idist: ::core::ffi::c_int, onembed: *mut ::core::ffi::c_int, ostride: ::core::ffi::c_int, odist: ::core::ffi::c_int, type_: cufftType, batch: ::core::ffi::c_int) -> cufftResult;
        }
        cufftPlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cufftResetPlanProperty(plan: cufftHandle, property: cufftProperty) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, cufftProperty) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftResetPlanProperty") });
        _f(plan, property)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftResetPlanProperty(plan: cufftHandle, property: cufftProperty) -> cufftResult;
        }
        cufftResetPlanProperty(plan, property)
    }
}
pub unsafe fn cufftSetAutoAllocation(plan: cufftHandle, autoAllocate: ::core::ffi::c_int) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, ::core::ffi::c_int) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftSetAutoAllocation") });
        _f(plan, autoAllocate)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftSetAutoAllocation(plan: cufftHandle, autoAllocate: ::core::ffi::c_int) -> cufftResult;
        }
        cufftSetAutoAllocation(plan, autoAllocate)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cufftSetPlanPropertyInt64(plan: cufftHandle, property: cufftProperty, inputValueInt: ::core::ffi::c_longlong) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, cufftProperty, ::core::ffi::c_longlong) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftSetPlanPropertyInt64") });
        _f(plan, property, inputValueInt)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftSetPlanPropertyInt64(plan: cufftHandle, property: cufftProperty, inputValueInt: ::core::ffi::c_longlong) -> cufftResult;
        }
        cufftSetPlanPropertyInt64(plan, property, inputValueInt)
    }
}
pub unsafe fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, cudaStream_t) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftSetStream") });
        _f(plan, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult;
        }
        cufftSetStream(plan, stream)
    }
}
pub unsafe fn cufftSetWorkArea(plan: cufftHandle, workArea: *mut ::core::ffi::c_void) -> cufftResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cufftHandle, *mut ::core::ffi::c_void) -> cufftResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cufftSetWorkArea") });
        _f(plan, workArea)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cufftSetWorkArea(plan: cufftHandle, workArea: *mut ::core::ffi::c_void) -> cufftResult;
        }
        cufftSetWorkArea(plan, workArea)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cufft"];
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
        let lib_names = std::vec!["cufft"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
