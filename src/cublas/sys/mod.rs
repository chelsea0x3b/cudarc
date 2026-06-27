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
pub use self::cudaDataType as cublasDataType_t;
pub use self::cudaDataType_t as cudaDataType;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaEmulationMantissaControl_t as cudaEmulationMantissaControl;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaEmulationSpecialValuesSupport_t as cudaEmulationSpecialValuesSupport;
pub use self::libraryPropertyType_t as libraryPropertyType;
pub type cuComplex = cuFloatComplex;
pub type cuDoubleComplex = double2;
pub type cuFloatComplex = float2;
pub type cublasHandle_t = *mut cublasContext;
pub type cublasLogCallback = ::core::option::Option<unsafe extern "C" fn(msg: *const ::core::ffi::c_char)>;
pub type cudaStream_t = *mut CUstream_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasAtomicsMode_t {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_32F_EMULATED_16BFX9 = 78,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_16F = 74,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_COMPUTE_32F_EMULATED_16BFX9 = 78,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,
    CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT = 79,
    CUBLAS_COMPUTE_32I = 72,
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasDiagType_t {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasEmulationStrategy_t {
    CUBLAS_EMULATION_STRATEGY_DEFAULT = 0,
    CUBLAS_EMULATION_STRATEGY_PERFORMANT = 1,
    CUBLAS_EMULATION_STRATEGY_EAGER = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,
    CUBLAS_GEMM_ALGO19 = 19,
    CUBLAS_GEMM_ALGO20 = 20,
    CUBLAS_GEMM_ALGO21 = 21,
    CUBLAS_GEMM_ALGO22 = 22,
    CUBLAS_GEMM_ALGO23 = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7,
    CUBLAS_GEMM_ALGO8 = 8,
    CUBLAS_GEMM_ALGO9 = 9,
    CUBLAS_GEMM_ALGO10 = 10,
    CUBLAS_GEMM_ALGO11 = 11,
    CUBLAS_GEMM_ALGO12 = 12,
    CUBLAS_GEMM_ALGO13 = 13,
    CUBLAS_GEMM_ALGO14 = 14,
    CUBLAS_GEMM_ALGO15 = 15,
    CUBLAS_GEMM_ALGO16 = 16,
    CUBLAS_GEMM_ALGO17 = 17,
    CUBLAS_GEMM_ALGO18 = 18,
    CUBLAS_GEMM_ALGO19 = 19,
    CUBLAS_GEMM_ALGO20 = 20,
    CUBLAS_GEMM_ALGO21 = 21,
    CUBLAS_GEMM_ALGO22 = 22,
    CUBLAS_GEMM_ALGO23 = 23,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
    CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
    CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
    CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
    CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
    CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
    CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
    CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
    CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
    CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
    CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
    CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
    CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
    CUBLAS_GEMM_ALGO15_TENSOR_OP = 115,
    CUBLAS_GEMM_AUTOTUNE = 999,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_FP32_EMULATED_BF16X9_MATH = 4,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
    CUBLAS_FP32_EMULATED_BF16X9_MATH = 4,
    CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH = 8,
    CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_CONJG = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasPointerMode_t {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16,
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
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaEmulationMantissaControl_t {
    CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC = 0,
    CUDA_EMULATION_MANTISSA_CONTROL_FIXED = 1,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaEmulationSpecialValuesSupport_t {
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT = 65535,
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE = 0,
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY = 1,
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN = 2,
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
#[derive(Debug, Copy, Clone)]
pub struct cublasContext {
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
impl cublasGemmAlgo_t {
    pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DFALT;
}
impl cublasGemmAlgo_t {
    pub const CUBLAS_GEMM_DFALT_TENSOR_OP: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}
impl cublasOperation_t {
    pub const CUBLAS_OP_HERMITAN: cublasOperation_t = cublasOperation_t::CUBLAS_OP_C;
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
pub unsafe fn cublasAsumEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasAsumEx") });
        _f(handle, n, x, xType, incx, result, resultType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasAsumEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasAsumEx(handle, n, x, xType, incx, result, resultType, executiontype)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasAsumEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasAsumEx_64") });
        _f(handle, n, x, xType, incx, result, resultType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasAsumEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasAsumEx_64(handle, n, x, xType, incx, result, resultType, executiontype)
    }
}
pub unsafe fn cublasAxpyEx(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasAxpyEx") });
        _f(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasAxpyEx(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasAxpyEx_64(handle: cublasHandle_t, n: i64, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, i64, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasAxpyEx_64") });
        _f(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasAxpyEx_64(handle: cublasHandle_t, n: i64, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasAxpyEx_64(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype)
    }
}
pub unsafe fn cublasCaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCaxpy_v2") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCaxpy_v2_64") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasCaxpy_v2_64(handle, n, alpha, x, incx, y, incy)
    }
}
pub unsafe fn cublasCcopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCcopy_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCcopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCcopy_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCcopy_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCcopy_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCcopy_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasCcopy_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasCdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCdgmm") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, i64, i64, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCdgmm_64") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
pub unsafe fn cublasCdotc_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, result: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCdotc_v2") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCdotc_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, result: *mut cuComplex) -> cublasStatus_t;
        }
        cublasCdotc_v2(handle, n, x, incx, y, incy, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCdotc_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, result: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCdotc_v2_64") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCdotc_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, result: *mut cuComplex) -> cublasStatus_t;
        }
        cublasCdotc_v2_64(handle, n, x, incx, y, incy, result)
    }
}
pub unsafe fn cublasCdotu_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, result: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCdotu_v2") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCdotu_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, result: *mut cuComplex) -> cublasStatus_t;
        }
        cublasCdotu_v2(handle, n, x, incx, y, incy, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCdotu_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, result: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCdotu_v2_64") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCdotu_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, result: *mut cuComplex) -> cublasStatus_t;
        }
        cublasCdotu_v2_64(handle, n, x, incx, y, incy, result)
    }
}
pub unsafe fn cublasCgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgbmv_v2") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgbmv_v2_64") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasCgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasCgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, beta: *const cuComplex, B: *const cuComplex, ldb: ::core::ffi::c_int, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgeam") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, beta: *const cuComplex, B: *const cuComplex, ldb: ::core::ffi::c_int, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, beta: *const cuComplex, B: *const cuComplex, ldb: i64, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgeam_64") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, beta: *const cuComplex, B: *const cuComplex, ldb: i64, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasCgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut cuComplex, lda: ::core::ffi::c_int, Carray: *const *mut cuComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgelsBatched") });
        _f(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut cuComplex, lda: ::core::ffi::c_int, Carray: *const *mut cuComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
}
pub unsafe fn cublasCgemm3m(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3m") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3m(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasCgemm3mBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, Barray: *const *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const *const cuComplex, ::core::ffi::c_int, *const *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3mBatched") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3mBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, Barray: *const *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemm3mBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemm3mBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: i64, Barray: *const *const cuComplex, ldb: i64, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const *const cuComplex, i64, *const *const cuComplex, i64, *const cuComplex, *const *mut cuComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3mBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3mBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: i64, Barray: *const *const cuComplex, ldb: i64, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasCgemm3mBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
pub unsafe fn cublasCgemm3mEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3mEx") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3mEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemm3mEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemm3mEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, i64, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3mEx_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3mEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasCgemm3mEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
}
pub unsafe fn cublasCgemm3mStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3mStridedBatched") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3mStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemm3mStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const cuComplex, i64, ::core::ffi::c_longlong, *const cuComplex, i64, ::core::ffi::c_longlong, *const cuComplex, *mut cuComplex, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3mStridedBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3mStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasCgemm3mStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemm3m_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm3m_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm3m_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCgemm3m_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasCgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, Barray: *const *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const *const cuComplex, ::core::ffi::c_int, *const *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemmBatched") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, Barray: *const *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: i64, Barray: *const *const cuComplex, ldb: i64, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const *const cuComplex, i64, *const *const cuComplex, i64, *const cuComplex, *const *mut cuComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemmBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: i64, Barray: *const *const cuComplex, ldb: i64, beta: *const cuComplex, Carray: *const *mut cuComplex, ldc: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasCgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
pub unsafe fn cublasCgemmEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemmEx") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemmEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemmEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, i64, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemmEx_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemmEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasCgemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
}
pub unsafe fn cublasCgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemmStridedBatched") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const cuComplex, i64, ::core::ffi::c_longlong, *const cuComplex, i64, ::core::ffi::c_longlong, *const cuComplex, *mut cuComplex, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemmStridedBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, strideA: ::core::ffi::c_longlong, B: *const cuComplex, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const cuComplex, C: *mut cuComplex, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasCgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
pub unsafe fn cublasCgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm_v2") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemm_v2_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, xarray: *const *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, yarray: *const *mut cuComplex, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const *const cuComplex, ::core::ffi::c_int, *const *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemvBatched") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, xarray: *const *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, yarray: *const *mut cuComplex, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: i64, xarray: *const *const cuComplex, incx: i64, beta: *const cuComplex, yarray: *const *mut cuComplex, incy: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const cuComplex, *const *const cuComplex, i64, *const *const cuComplex, i64, *const cuComplex, *const *mut cuComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemvBatched_64") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, Aarray: *const *const cuComplex, lda: i64, xarray: *const *const cuComplex, incx: i64, beta: *const cuComplex, yarray: *const *mut cuComplex, incy: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasCgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const cuComplex, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemvStridedBatched") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const cuComplex, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, strideA: ::core::ffi::c_longlong, x: *const cuComplex, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const cuComplex, y: *mut cuComplex, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, ::core::ffi::c_longlong, *const cuComplex, i64, ::core::ffi::c_longlong, *const cuComplex, *mut cuComplex, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemvStridedBatched_64") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, strideA: ::core::ffi::c_longlong, x: *const cuComplex, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const cuComplex, y: *mut cuComplex, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasCgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
pub unsafe fn cublasCgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemv_v2") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgemv_v2_64") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasCgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasCgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut cuComplex, lda: ::core::ffi::c_int, TauArray: *const *mut cuComplex, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *const *mut cuComplex, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgeqrfBatched") });
        _f(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut cuComplex, lda: ::core::ffi::c_int, TauArray: *const *mut cuComplex, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
}
pub unsafe fn cublasCgerc_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgerc_v2") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgerc_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgerc_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgerc_v2_64") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgerc_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t;
        }
        cublasCgerc_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasCgeru_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgeru_v2") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgeru_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCgeru_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgeru_v2_64") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgeru_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t;
        }
        cublasCgeru_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasCgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut cuComplex, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgetrfBatched") });
        _f(handle, n, A, lda, P, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut cuComplex, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize)
    }
}
pub unsafe fn cublasCgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuComplex, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut cuComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgetriBatched") });
        _f(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuComplex, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut cuComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
}
pub unsafe fn cublasCgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut cuComplex, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *const cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCgetrsBatched") });
        _f(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const cuComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut cuComplex, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
}
pub unsafe fn cublasChbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChbmv_v2") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasChbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasChbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChbmv_v2_64") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasChbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasChemm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChemm_v2") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChemm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasChemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasChemm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChemm_v2_64") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChemm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasChemm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasChemv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChemv_v2") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChemv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasChemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasChemv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChemv_v2_64") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChemv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasChemv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasCher2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCher2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCher2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCher2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCher2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCher2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t;
        }
        cublasCher2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasCher2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCher2k_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCher2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCher2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const f32, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const f32, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCher2k_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCher2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const f32, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCher2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasCher_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const cuComplex, incx: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCher_v2") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCher_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const cuComplex, incx: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCher_v2(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCher_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const cuComplex, incx: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCher_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCher_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const cuComplex, incx: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t;
        }
        cublasCher_v2_64(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
pub unsafe fn cublasCherk3mEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherk3mEx") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherk3mEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCherk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCherk3mEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f32, *const ::core::ffi::c_void, cudaDataType, i64, *const f32, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherk3mEx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherk3mEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasCherk3mEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
pub unsafe fn cublasCherkEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherkEx") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherkEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCherkEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f32, *const ::core::ffi::c_void, cudaDataType, i64, *const f32, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherkEx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherkEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasCherkEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
pub unsafe fn cublasCherk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const cuComplex, lda: ::core::ffi::c_int, beta: *const f32, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const cuComplex, ::core::ffi::c_int, *const f32, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherk_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const cuComplex, lda: ::core::ffi::c_int, beta: *const f32, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCherk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const cuComplex, lda: i64, beta: *const f32, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f32, *const cuComplex, i64, *const f32, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherk_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const cuComplex, lda: i64, beta: *const f32, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCherk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
pub unsafe fn cublasCherkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherkx") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCherkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const f32, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const f32, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCherkx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCherkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const f32, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCherkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasChpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, AP: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChpmv_v2") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, AP: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasChpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasChpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, AP: *const cuComplex, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChpmv_v2_64") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, AP: *const cuComplex, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasChpmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasChpr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, AP: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChpr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChpr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, AP: *mut cuComplex) -> cublasStatus_t;
        }
        cublasChpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasChpr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, AP: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChpr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChpr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, AP: *mut cuComplex) -> cublasStatus_t;
        }
        cublasChpr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
pub unsafe fn cublasChpr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const cuComplex, incx: ::core::ffi::c_int, AP: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const cuComplex, ::core::ffi::c_int, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChpr_v2") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChpr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const cuComplex, incx: ::core::ffi::c_int, AP: *mut cuComplex) -> cublasStatus_t;
        }
        cublasChpr_v2(handle, uplo, n, alpha, x, incx, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasChpr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const cuComplex, incx: i64, AP: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const cuComplex, i64, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasChpr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasChpr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const cuComplex, incx: i64, AP: *mut cuComplex) -> cublasStatus_t;
        }
        cublasChpr_v2_64(handle, uplo, n, alpha, x, incx, AP)
    }
}
pub unsafe fn cublasCmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuComplex, lda: ::core::ffi::c_int, Ainv: *const *mut cuComplex, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const cuComplex, ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCmatinvBatched") });
        _f(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuComplex, lda: ::core::ffi::c_int, Ainv: *const *mut cuComplex, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
}
pub unsafe fn cublasCopyEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCopyEx") });
        _f(handle, n, x, xType, incx, y, yType, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCopyEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCopyEx(handle, n, x, xType, incx, y, yType, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCopyEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCopyEx_64") });
        _f(handle, n, x, xType, incx, y, yType, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCopyEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64) -> cublasStatus_t;
        }
        cublasCopyEx_64(handle, n, x, xType, incx, y, yType, incy)
    }
}
pub unsafe fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCreate_v2") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
        }
        cublasCreate_v2(handle)
    }
}
pub unsafe fn cublasCrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int, c: *const f32, s: *const cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const f32, *const cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCrot_v2") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int, c: *const f32, s: *const cuComplex) -> cublasStatus_t;
        }
        cublasCrot_v2(handle, n, x, incx, y, incy, c, s)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuComplex, incx: i64, y: *mut cuComplex, incy: i64, c: *const f32, s: *const cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut cuComplex, i64, *mut cuComplex, i64, *const f32, *const cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCrot_v2_64") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuComplex, incx: i64, y: *mut cuComplex, incy: i64, c: *const f32, s: *const cuComplex) -> cublasStatus_t;
        }
        cublasCrot_v2_64(handle, n, x, incx, y, incy, c, s)
    }
}
pub unsafe fn cublasCrotg_v2(handle: cublasHandle_t, a: *mut cuComplex, b: *mut cuComplex, c: *mut f32, s: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cuComplex, *mut cuComplex, *mut f32, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCrotg_v2") });
        _f(handle, a, b, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCrotg_v2(handle: cublasHandle_t, a: *mut cuComplex, b: *mut cuComplex, c: *mut f32, s: *mut cuComplex) -> cublasStatus_t;
        }
        cublasCrotg_v2(handle, a, b, c, s)
    }
}
pub unsafe fn cublasCscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCscal_v2") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCscal_v2(handle, n, alpha, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuComplex, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCscal_v2_64") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuComplex, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCscal_v2_64(handle, n, alpha, x, incx)
    }
}
pub unsafe fn cublasCsrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int, c: *const f32, s: *const f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const f32, *const f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsrot_v2") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int, c: *const f32, s: *const f32) -> cublasStatus_t;
        }
        cublasCsrot_v2(handle, n, x, incx, y, incy, c, s)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuComplex, incx: i64, y: *mut cuComplex, incy: i64, c: *const f32, s: *const f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut cuComplex, i64, *mut cuComplex, i64, *const f32, *const f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsrot_v2_64") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuComplex, incx: i64, y: *mut cuComplex, incy: i64, c: *const f32, s: *const f32) -> cublasStatus_t;
        }
        cublasCsrot_v2_64(handle, n, x, incx, y, incy, c, s)
    }
}
pub unsafe fn cublasCsscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f32, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsscal_v2") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f32, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsscal_v2(handle, n, alpha, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f32, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsscal_v2_64") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f32, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCsscal_v2_64(handle, n, alpha, x, incx)
    }
}
pub unsafe fn cublasCswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCswap_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCswap_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuComplex, incx: i64, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCswap_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuComplex, incx: i64, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasCswap_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasCsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsymm_v2") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsymm_v2_64") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasCsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsymv_v2") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsymv_v2_64") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, x: *const cuComplex, incx: i64, beta: *const cuComplex, y: *mut cuComplex, incy: i64) -> cublasStatus_t;
        }
        cublasCsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasCsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, y: *const cuComplex, incy: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, y: *const cuComplex, incy: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t;
        }
        cublasCsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasCsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyr2k_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyr2k_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasCsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyr_v2") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuComplex, x: *const cuComplex, incx: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuComplex, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuComplex, x: *const cuComplex, incx: i64, A: *mut cuComplex, lda: i64) -> cublasStatus_t;
        }
        cublasCsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
pub unsafe fn cublasCsyrk3mEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrk3mEx") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrk3mEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyrk3mEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyrk3mEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, i64, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrk3mEx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrk3mEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasCsyrk3mEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
pub unsafe fn cublasCsyrkEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrkEx") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrkEx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyrkEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const ::core::ffi::c_void, cudaDataType, i64, *const cuComplex, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrkEx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrkEx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, beta: *const cuComplex, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasCsyrkEx_64(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
    }
}
pub unsafe fn cublasCsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrk_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrk_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
pub unsafe fn cublasCsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrkx") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCsyrkx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, beta: *const cuComplex, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasCtbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtbmv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtbmv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCtbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasCtbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtbsv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtbsv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCtbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasCtpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuComplex, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtpmv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuComplex, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuComplex, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtpmv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuComplex, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCtpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasCtpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuComplex, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtpsv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuComplex, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuComplex, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuComplex, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtpsv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuComplex, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCtpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasCtpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const cuComplex, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtpttr") });
        _f(handle, uplo, n, AP, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const cuComplex, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtpttr(handle, uplo, n, AP, A, lda)
    }
}
pub unsafe fn cublasCtrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrmm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, C: *mut cuComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuComplex, *const cuComplex, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrmm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *const cuComplex, ldb: i64, C: *mut cuComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasCtrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasCtrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrmv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrmv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCtrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasCtrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const *const cuComplex, lda: ::core::ffi::c_int, B: *const *mut cuComplex, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const *const cuComplex, ::core::ffi::c_int, *const *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrsmBatched") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const *const cuComplex, lda: ::core::ffi::c_int, B: *const *mut cuComplex, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuComplex, A: *const *const cuComplex, lda: i64, B: *const *mut cuComplex, ldb: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuComplex, *const *const cuComplex, i64, *const *mut cuComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrsmBatched_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuComplex, A: *const *const cuComplex, lda: i64, B: *const *mut cuComplex, ldb: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasCtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
pub unsafe fn cublasCtrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrsm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *mut cuComplex, ldb: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuComplex, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrsm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuComplex, A: *const cuComplex, lda: i64, B: *mut cuComplex, ldb: i64) -> cublasStatus_t;
        }
        cublasCtrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
pub unsafe fn cublasCtrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrsv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, x: *mut cuComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasCtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasCtrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuComplex, i64, *mut cuComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrsv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuComplex, lda: i64, x: *mut cuComplex, incx: i64) -> cublasStatus_t;
        }
        cublasCtrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasCtrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, AP: *mut cuComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasCtrttp") });
        _f(handle, uplo, n, A, lda, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasCtrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, AP: *mut cuComplex) -> cublasStatus_t;
        }
        cublasCtrttp(handle, uplo, n, A, lda, AP)
    }
}
pub unsafe fn cublasDasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDasum_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t;
        }
        cublasDasum_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDasum_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDasum_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDasum_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut f64) -> cublasStatus_t;
        }
        cublasDasum_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasDaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDaxpy_v2") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDaxpy_v2_64") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDaxpy_v2_64(handle, n, alpha, x, incx, y, incy)
    }
}
pub unsafe fn cublasDcopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDcopy_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDcopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDcopy_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDcopy_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDcopy_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDcopy_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDcopy_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasDdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDdgmm") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const f64, lda: i64, x: *const f64, incx: i64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, i64, i64, *const f64, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDdgmm_64") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const f64, lda: i64, x: *const f64, incx: i64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
pub unsafe fn cublasDdot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDdot_v2") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDdot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t;
        }
        cublasDdot_v2(handle, n, x, incx, y, incy, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDdot_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, i64, *const f64, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDdot_v2_64") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDdot_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64, result: *mut f64) -> cublasStatus_t;
        }
        cublasDdot_v2_64(handle, n, x, incx, y, incy, result)
    }
}
pub unsafe fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDestroy_v2") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
        }
        cublasDestroy_v2(handle)
    }
}
pub unsafe fn cublasDgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgbmv_v2") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgbmv_v2_64") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasDgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, beta: *const f64, B: *const f64, ldb: ::core::ffi::c_int, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgeam") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, beta: *const f64, B: *const f64, ldb: ::core::ffi::c_int, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, beta: *const f64, B: *const f64, ldb: i64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, *const f64, *const f64, i64, *const f64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgeam_64") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, beta: *const f64, B: *const f64, ldb: i64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasDgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut f64, lda: ::core::ffi::c_int, Carray: *const *mut f64, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgelsBatched") });
        _f(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut f64, lda: ::core::ffi::c_int, Carray: *const *mut f64, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
}
pub unsafe fn cublasDgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, Aarray: *const *const f64, lda: ::core::ffi::c_int, Barray: *const *const f64, ldb: ::core::ffi::c_int, beta: *const f64, Carray: *const *mut f64, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const *const f64, ::core::ffi::c_int, *const *const f64, ::core::ffi::c_int, *const f64, *const *mut f64, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemmBatched") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, Aarray: *const *const f64, lda: ::core::ffi::c_int, Barray: *const *const f64, ldb: ::core::ffi::c_int, beta: *const f64, Carray: *const *mut f64, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f64, Aarray: *const *const f64, lda: i64, Barray: *const *const f64, ldb: i64, beta: *const f64, Carray: *const *mut f64, ldc: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f64, *const *const f64, i64, *const *const f64, i64, *const f64, *const *mut f64, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemmBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f64, Aarray: *const *const f64, lda: i64, Barray: *const *const f64, ldb: i64, beta: *const f64, Carray: *const *mut f64, ldc: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasDgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemmGroupedBatched(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const ::core::ffi::c_int, n_array: *const ::core::ffi::c_int, k_array: *const ::core::ffi::c_int, alpha_array: *const f64, Aarray: *const *const f64, lda_array: *const ::core::ffi::c_int, Barray: *const *const f64, ldb_array: *const ::core::ffi::c_int, beta_array: *const f64, Carray: *const *mut f64, ldc_array: *const ::core::ffi::c_int, group_count: ::core::ffi::c_int, group_size: *const ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *const cublasOperation_t, *const cublasOperation_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, *const *const f64, *const ::core::ffi::c_int, *const *const f64, *const ::core::ffi::c_int, *const f64, *const *mut f64, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemmGroupedBatched") });
        _f(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemmGroupedBatched(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const ::core::ffi::c_int, n_array: *const ::core::ffi::c_int, k_array: *const ::core::ffi::c_int, alpha_array: *const f64, Aarray: *const *const f64, lda_array: *const ::core::ffi::c_int, Barray: *const *const f64, ldb_array: *const ::core::ffi::c_int, beta_array: *const f64, Carray: *const *mut f64, ldc_array: *const ::core::ffi::c_int, group_count: ::core::ffi::c_int, group_size: *const ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemmGroupedBatched(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemmGroupedBatched_64(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const i64, n_array: *const i64, k_array: *const i64, alpha_array: *const f64, Aarray: *const *const f64, lda_array: *const i64, Barray: *const *const f64, ldb_array: *const i64, beta_array: *const f64, Carray: *const *mut f64, ldc_array: *const i64, group_count: i64, group_size: *const i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *const cublasOperation_t, *const cublasOperation_t, *const i64, *const i64, *const i64, *const f64, *const *const f64, *const i64, *const *const f64, *const i64, *const f64, *const *mut f64, *const i64, i64, *const i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemmGroupedBatched_64") });
        _f(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemmGroupedBatched_64(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const i64, n_array: *const i64, k_array: *const i64, alpha_array: *const f64, Aarray: *const *const f64, lda_array: *const i64, Barray: *const *const f64, ldb_array: *const i64, beta_array: *const f64, Carray: *const *mut f64, ldc_array: *const i64, group_count: i64, group_size: *const i64) -> cublasStatus_t;
        }
        cublasDgemmGroupedBatched_64(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
}
pub unsafe fn cublasDgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const f64, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, *mut f64, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemmStridedBatched") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const f64, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, strideA: ::core::ffi::c_longlong, B: *const f64, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const f64, C: *mut f64, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f64, *const f64, i64, ::core::ffi::c_longlong, *const f64, i64, ::core::ffi::c_longlong, *const f64, *mut f64, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemmStridedBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, strideA: ::core::ffi::c_longlong, B: *const f64, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const f64, C: *mut f64, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasDgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
pub unsafe fn cublasDgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemm_v2") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemm_v2_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, Aarray: *const *const f64, lda: ::core::ffi::c_int, xarray: *const *const f64, incx: ::core::ffi::c_int, beta: *const f64, yarray: *const *mut f64, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const *const f64, ::core::ffi::c_int, *const *const f64, ::core::ffi::c_int, *const f64, *const *mut f64, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemvBatched") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, Aarray: *const *const f64, lda: ::core::ffi::c_int, xarray: *const *const f64, incx: ::core::ffi::c_int, beta: *const f64, yarray: *const *mut f64, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f64, Aarray: *const *const f64, lda: i64, xarray: *const *const f64, incx: i64, beta: *const f64, yarray: *const *mut f64, incy: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const f64, *const *const f64, i64, *const *const f64, i64, *const f64, *const *mut f64, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemvBatched_64") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f64, Aarray: *const *const f64, lda: i64, xarray: *const *const f64, incx: i64, beta: *const f64, yarray: *const *mut f64, incy: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasDgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const f64, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, *mut f64, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemvStridedBatched") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const f64, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, strideA: ::core::ffi::c_longlong, x: *const f64, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const f64, y: *mut f64, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const f64, *const f64, i64, ::core::ffi::c_longlong, *const f64, i64, ::core::ffi::c_longlong, *const f64, *mut f64, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemvStridedBatched_64") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, strideA: ::core::ffi::c_longlong, x: *const f64, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const f64, y: *mut f64, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasDgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
pub unsafe fn cublasDgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemv_v2") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgemv_v2_64") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasDgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut f64, lda: ::core::ffi::c_int, TauArray: *const *mut f64, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *const *mut f64, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgeqrfBatched") });
        _f(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut f64, lda: ::core::ffi::c_int, TauArray: *const *mut f64, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
}
pub unsafe fn cublasDger_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDger_v2") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDger_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDger_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, A: *mut f64, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDger_v2_64") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDger_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, A: *mut f64, lda: i64) -> cublasStatus_t;
        }
        cublasDger_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasDgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut f64, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgetrfBatched") });
        _f(handle, n, A, lda, P, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut f64, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize)
    }
}
pub unsafe fn cublasDgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f64, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut f64, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgetriBatched") });
        _f(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f64, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut f64, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
}
pub unsafe fn cublasDgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const f64, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut f64, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *const f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDgetrsBatched") });
        _f(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const f64, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut f64, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
}
pub unsafe fn cublasDmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f64, lda: ::core::ffi::c_int, Ainv: *const *mut f64, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const f64, ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDmatinvBatched") });
        _f(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f64, lda: ::core::ffi::c_int, Ainv: *const *mut f64, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
}
pub unsafe fn cublasDnrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDnrm2_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDnrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t;
        }
        cublasDnrm2_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDnrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDnrm2_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDnrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut f64) -> cublasStatus_t;
        }
        cublasDnrm2_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasDotEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDotEx") });
        _f(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDotEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDotEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDotEx_64") });
        _f(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDotEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasDotEx_64(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
}
pub unsafe fn cublasDotcEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDotcEx") });
        _f(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDotcEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDotcEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDotcEx_64") });
        _f(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDotcEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *const ::core::ffi::c_void, yType: cudaDataType, incy: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasDotcEx_64(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    }
}
pub unsafe fn cublasDrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int, c: *const f64, s: *const f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const f64, *const f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDrot_v2") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int, c: *const f64, s: *const f64) -> cublasStatus_t;
        }
        cublasDrot_v2(handle, n, x, incx, y, incy, c, s)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64, c: *const f64, s: *const f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut f64, i64, *mut f64, i64, *const f64, *const f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDrot_v2_64") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64, c: *const f64, s: *const f64) -> cublasStatus_t;
        }
        cublasDrot_v2_64(handle, n, x, incx, y, incy, c, s)
    }
}
pub unsafe fn cublasDrotg_v2(handle: cublasHandle_t, a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut f64, *mut f64, *mut f64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDrotg_v2") });
        _f(handle, a, b, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDrotg_v2(handle: cublasHandle_t, a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64) -> cublasStatus_t;
        }
        cublasDrotg_v2(handle, a, b, c, s)
    }
}
pub unsafe fn cublasDrotm_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int, param: *const f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDrotm_v2") });
        _f(handle, n, x, incx, y, incy, param)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDrotm_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int, param: *const f64) -> cublasStatus_t;
        }
        cublasDrotm_v2(handle, n, x, incx, y, incy, param)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDrotm_v2_64(handle: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64, param: *const f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut f64, i64, *mut f64, i64, *const f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDrotm_v2_64") });
        _f(handle, n, x, incx, y, incy, param)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDrotm_v2_64(handle: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64, param: *const f64) -> cublasStatus_t;
        }
        cublasDrotm_v2_64(handle, n, x, incx, y, incy, param)
    }
}
pub unsafe fn cublasDrotmg_v2(handle: cublasHandle_t, d1: *mut f64, d2: *mut f64, x1: *mut f64, y1: *const f64, param: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut f64, *mut f64, *mut f64, *const f64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDrotmg_v2") });
        _f(handle, d1, d2, x1, y1, param)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDrotmg_v2(handle: cublasHandle_t, d1: *mut f64, d2: *mut f64, x1: *mut f64, y1: *const f64, param: *mut f64) -> cublasStatus_t;
        }
        cublasDrotmg_v2(handle, d1, d2, x1, y1, param)
    }
}
pub unsafe fn cublasDsbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsbmv_v2") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsbmv_v2_64") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDsbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasDscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f64, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDscal_v2") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f64, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDscal_v2(handle, n, alpha, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDscal_v2_64") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDscal_v2_64(handle, n, alpha, x, incx)
    }
}
pub unsafe fn cublasDspmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, AP: *const f64, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const f64, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDspmv_v2") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDspmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, AP: *const f64, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDspmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, AP: *const f64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const f64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDspmv_v2_64") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDspmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, AP: *const f64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDspmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasDspr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, AP: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDspr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDspr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, AP: *mut f64) -> cublasStatus_t;
        }
        cublasDspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDspr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, AP: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const f64, i64, *const f64, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDspr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDspr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, AP: *mut f64) -> cublasStatus_t;
        }
        cublasDspr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
pub unsafe fn cublasDspr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, AP: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDspr_v2") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDspr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, AP: *mut f64) -> cublasStatus_t;
        }
        cublasDspr_v2(handle, uplo, n, alpha, x, incx, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDspr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, AP: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const f64, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDspr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDspr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, AP: *mut f64) -> cublasStatus_t;
        }
        cublasDspr_v2_64(handle, uplo, n, alpha, x, incx, AP)
    }
}
pub unsafe fn cublasDswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDswap_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDswap_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDswap_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDswap_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasDsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsymm_v2") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsymm_v2_64") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasDsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsymv_v2") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, beta: *const f64, y: *mut f64, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsymv_v2_64") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, A: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) -> cublasStatus_t;
        }
        cublasDsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasDsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, y: *const f64, incy: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, A: *mut f64, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const f64, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, A: *mut f64, lda: i64) -> cublasStatus_t;
        }
        cublasDsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasDsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyr2k_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyr2k_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasDsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyr_v2") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const f64, incx: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, A: *mut f64, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, A: *mut f64, lda: i64) -> cublasStatus_t;
        }
        cublasDsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
pub unsafe fn cublasDsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyrk_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyrk_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
pub unsafe fn cublasDsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyrkx") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDsyrkx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, beta: *const f64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasDtbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtbmv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtbmv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDtbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasDtbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtbsv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtbsv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDtbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasDtpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f64, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtpmv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f64, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtpmv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDtpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasDtpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f64, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtpsv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f64, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtpsv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDtpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasDtpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const f64, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtpttr") });
        _f(handle, uplo, n, AP, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const f64, A: *mut f64, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtpttr(handle, uplo, n, AP, A, lda)
    }
}
pub unsafe fn cublasDtrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrmm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, C: *mut f64, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, C: *mut f64, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f64, *const f64, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrmm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, B: *const f64, ldb: i64, C: *mut f64, ldc: i64) -> cublasStatus_t;
        }
        cublasDtrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasDtrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrmv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrmv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDtrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasDtrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const *const f64, lda: ::core::ffi::c_int, B: *const *mut f64, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const *const f64, ::core::ffi::c_int, *const *mut f64, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrsmBatched") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const *const f64, lda: ::core::ffi::c_int, B: *const *mut f64, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f64, A: *const *const f64, lda: i64, B: *const *mut f64, ldb: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f64, *const *const f64, i64, *const *mut f64, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrsmBatched_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f64, A: *const *const f64, lda: i64, B: *const *mut f64, ldb: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasDtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
pub unsafe fn cublasDtrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrsm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, B: *mut f64, ldb: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrsm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f64, A: *const f64, lda: i64, B: *mut f64, ldb: i64) -> cublasStatus_t;
        }
        cublasDtrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
pub unsafe fn cublasDtrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrsv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, x: *mut f64, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasDtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDtrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f64, i64, *mut f64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrsv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f64, lda: i64, x: *mut f64, incx: i64) -> cublasStatus_t;
        }
        cublasDtrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasDtrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, AP: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDtrttp") });
        _f(handle, uplo, n, A, lda, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDtrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, AP: *mut f64) -> cublasStatus_t;
        }
        cublasDtrttp(handle, uplo, n, A, lda, AP)
    }
}
pub unsafe fn cublasDzasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDzasum_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDzasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t;
        }
        cublasDzasum_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDzasum_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDzasum_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDzasum_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut f64) -> cublasStatus_t;
        }
        cublasDzasum_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasDznrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDznrm2_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDznrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut f64) -> cublasStatus_t;
        }
        cublasDznrm2_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasDznrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *mut f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasDznrm2_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasDznrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut f64) -> cublasStatus_t;
        }
        cublasDznrm2_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasGemmBatchedEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, *const *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, ::core::ffi::c_int, cublasComputeType_t, cublasGemmAlgo_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmBatchedEx") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmBatchedEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t;
        }
        cublasGemmBatchedEx(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGemmBatchedEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64, batchCount: i64, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, cudaDataType, i64, *const *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, *const *mut ::core::ffi::c_void, cudaDataType, i64, i64, cublasComputeType_t, cublasGemmAlgo_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmBatchedEx_64") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmBatchedEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64, batchCount: i64, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t;
        }
        cublasGemmBatchedEx_64(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray, Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo)
    }
}
pub unsafe fn cublasGemmEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, cublasComputeType_t, cublasGemmAlgo_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmEx") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t;
        }
        cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGemmEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, i64, cublasComputeType_t, cublasGemmAlgo_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmEx_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t;
        }
        cublasGemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGemmGroupedBatchedEx(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const ::core::ffi::c_int, n_array: *const ::core::ffi::c_int, k_array: *const ::core::ffi::c_int, alpha_array: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType_t, lda_array: *const ::core::ffi::c_int, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType_t, ldb_array: *const ::core::ffi::c_int, beta_array: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType_t, ldc_array: *const ::core::ffi::c_int, group_count: ::core::ffi::c_int, group_size: *const ::core::ffi::c_int, computeType: cublasComputeType_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *const cublasOperation_t, *const cublasOperation_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, cudaDataType_t, *const ::core::ffi::c_int, *const *const ::core::ffi::c_void, cudaDataType_t, *const ::core::ffi::c_int, *const ::core::ffi::c_void, *const *mut ::core::ffi::c_void, cudaDataType_t, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, cublasComputeType_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmGroupedBatchedEx") });
        _f(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmGroupedBatchedEx(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const ::core::ffi::c_int, n_array: *const ::core::ffi::c_int, k_array: *const ::core::ffi::c_int, alpha_array: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType_t, lda_array: *const ::core::ffi::c_int, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType_t, ldb_array: *const ::core::ffi::c_int, beta_array: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType_t, ldc_array: *const ::core::ffi::c_int, group_count: ::core::ffi::c_int, group_size: *const ::core::ffi::c_int, computeType: cublasComputeType_t) -> cublasStatus_t;
        }
        cublasGemmGroupedBatchedEx(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGemmGroupedBatchedEx_64(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const i64, n_array: *const i64, k_array: *const i64, alpha_array: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType_t, lda_array: *const i64, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType_t, ldb_array: *const i64, beta_array: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType_t, ldc_array: *const i64, group_count: i64, group_size: *const i64, computeType: cublasComputeType_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *const cublasOperation_t, *const cublasOperation_t, *const i64, *const i64, *const i64, *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, cudaDataType_t, *const i64, *const *const ::core::ffi::c_void, cudaDataType_t, *const i64, *const ::core::ffi::c_void, *const *mut ::core::ffi::c_void, cudaDataType_t, *const i64, i64, *const i64, cublasComputeType_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmGroupedBatchedEx_64") });
        _f(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmGroupedBatchedEx_64(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const i64, n_array: *const i64, k_array: *const i64, alpha_array: *const ::core::ffi::c_void, Aarray: *const *const ::core::ffi::c_void, Atype: cudaDataType_t, lda_array: *const i64, Barray: *const *const ::core::ffi::c_void, Btype: cudaDataType_t, ldb_array: *const i64, beta_array: *const ::core::ffi::c_void, Carray: *const *mut ::core::ffi::c_void, Ctype: cudaDataType_t, ldc_array: *const i64, group_count: i64, group_size: *const i64, computeType: cublasComputeType_t) -> cublasStatus_t;
        }
        cublasGemmGroupedBatchedEx_64(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype, lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count, group_size, computeType)
    }
}
pub unsafe fn cublasGemmStridedBatchedEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, ::core::ffi::c_longlong, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, ::core::ffi::c_longlong, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int, cublasComputeType_t, cublasGemmAlgo_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmStridedBatchedEx") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmStridedBatchedEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t;
        }
        cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGemmStridedBatchedEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, strideA: ::core::ffi::c_longlong, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, i64, ::core::ffi::c_longlong, *const ::core::ffi::c_void, cudaDataType, i64, ::core::ffi::c_longlong, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, i64, ::core::ffi::c_longlong, i64, cublasComputeType_t, cublasGemmAlgo_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGemmStridedBatchedEx_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGemmStridedBatchedEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, strideA: ::core::ffi::c_longlong, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const ::core::ffi::c_void, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64, computeType: cublasComputeType_t, algo: cublasGemmAlgo_t) -> cublasStatus_t;
        }
        cublasGemmStridedBatchedEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo)
    }
}
pub unsafe fn cublasGetAtomicsMode(handle: cublasHandle_t, mode: *mut cublasAtomicsMode_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cublasAtomicsMode_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetAtomicsMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetAtomicsMode(handle: cublasHandle_t, mode: *mut cublasAtomicsMode_t) -> cublasStatus_t;
        }
        cublasGetAtomicsMode(handle, mode)
    }
}
pub unsafe fn cublasGetCudartVersion() -> usize {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> usize;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetCudartVersion") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetCudartVersion() -> usize;
        }
        cublasGetCudartVersion()
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGetEmulationStrategy(handle: cublasHandle_t, emulationStrategy: *mut cublasEmulationStrategy_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cublasEmulationStrategy_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetEmulationStrategy") });
        _f(handle, emulationStrategy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetEmulationStrategy(handle: cublasHandle_t, emulationStrategy: *mut cublasEmulationStrategy_t) -> cublasStatus_t;
        }
        cublasGetEmulationStrategy(handle, emulationStrategy)
    }
}
pub unsafe fn cublasGetLoggerCallback(userCallback: *mut cublasLogCallback) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cublasLogCallback) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetLoggerCallback") });
        _f(userCallback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetLoggerCallback(userCallback: *mut cublasLogCallback) -> cublasStatus_t;
        }
        cublasGetLoggerCallback(userCallback)
    }
}
pub unsafe fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cublasMath_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetMathMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t;
        }
        cublasGetMathMode(handle, mode)
    }
}
pub unsafe fn cublasGetMatrix(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetMatrix") });
        _f(rows, cols, elemSize, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetMatrix(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    }
}
pub unsafe fn cublasGetMatrixAsync(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetMatrixAsync") });
        _f(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetMatrixAsync(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGetMatrixAsync_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetMatrixAsync_64") });
        _f(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetMatrixAsync_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasGetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGetMatrix_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetMatrix_64") });
        _f(rows, cols, elemSize, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetMatrix_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64) -> cublasStatus_t;
        }
        cublasGetMatrix_64(rows, cols, elemSize, A, lda, B, ldb)
    }
}
pub unsafe fn cublasGetPointerMode_v2(handle: cublasHandle_t, mode: *mut cublasPointerMode_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cublasPointerMode_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetPointerMode_v2") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetPointerMode_v2(handle: cublasHandle_t, mode: *mut cublasPointerMode_t) -> cublasStatus_t;
        }
        cublasGetPointerMode_v2(handle, mode)
    }
}
pub unsafe fn cublasGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(libraryPropertyType, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetProperty") });
        _f(type_, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasGetProperty(type_, value)
    }
}
pub unsafe fn cublasGetSmCountTarget(handle: cublasHandle_t, smCountTarget: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetSmCountTarget") });
        _f(handle, smCountTarget)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetSmCountTarget(handle: cublasHandle_t, smCountTarget: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasGetSmCountTarget(handle, smCountTarget)
    }
}
pub unsafe fn cublasGetStatusName(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasStatus_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetStatusName") });
        _f(status)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetStatusName(status: cublasStatus_t) -> *const ::core::ffi::c_char;
        }
        cublasGetStatusName(status)
    }
}
pub unsafe fn cublasGetStatusString(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasStatus_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetStatusString") });
        _f(status)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetStatusString(status: cublasStatus_t) -> *const ::core::ffi::c_char;
        }
        cublasGetStatusString(status)
    }
}
pub unsafe fn cublasGetStream_v2(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetStream_v2") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetStream_v2(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> cublasStatus_t;
        }
        cublasGetStream_v2(handle, streamId)
    }
}
pub unsafe fn cublasGetVector(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, x: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetVector") });
        _f(n, elemSize, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetVector(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, x: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasGetVector(n, elemSize, x, incx, y, incy)
    }
}
pub unsafe fn cublasGetVectorAsync(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, devicePtr: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, hostPtr: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetVectorAsync") });
        _f(n, elemSize, devicePtr, incx, hostPtr, incy, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetVectorAsync(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, devicePtr: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, hostPtr: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasGetVectorAsync(n, elemSize, devicePtr, incx, hostPtr, incy, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGetVectorAsync_64(n: i64, elemSize: i64, devicePtr: *const ::core::ffi::c_void, incx: i64, hostPtr: *mut ::core::ffi::c_void, incy: i64, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetVectorAsync_64") });
        _f(n, elemSize, devicePtr, incx, hostPtr, incy, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetVectorAsync_64(n: i64, elemSize: i64, devicePtr: *const ::core::ffi::c_void, incx: i64, hostPtr: *mut ::core::ffi::c_void, incy: i64, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasGetVectorAsync_64(n, elemSize, devicePtr, incx, hostPtr, incy, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasGetVector_64(n: i64, elemSize: i64, x: *const ::core::ffi::c_void, incx: i64, y: *mut ::core::ffi::c_void, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetVector_64") });
        _f(n, elemSize, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetVector_64(n: i64, elemSize: i64, x: *const ::core::ffi::c_void, incx: i64, y: *mut ::core::ffi::c_void, incy: i64) -> cublasStatus_t;
        }
        cublasGetVector_64(n, elemSize, x, incx, y, incy)
    }
}
pub unsafe fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasGetVersion_v2") });
        _f(handle, version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasGetVersion_v2(handle, version)
    }
}
pub unsafe fn cublasIamaxEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIamaxEx") });
        _f(handle, n, x, xType, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIamaxEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIamaxEx(handle, n, x, xType, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIamaxEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIamaxEx_64") });
        _f(handle, n, x, xType, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIamaxEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIamaxEx_64(handle, n, x, xType, incx, result)
    }
}
pub unsafe fn cublasIaminEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIaminEx") });
        _f(handle, n, x, xType, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIaminEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIaminEx(handle, n, x, xType, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIaminEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIaminEx_64") });
        _f(handle, n, x, xType, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIaminEx_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIaminEx_64(handle, n, x, xType, incx, result)
    }
}
pub unsafe fn cublasIcamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIcamax_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIcamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIcamax_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIcamax_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIcamax_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIcamax_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIcamax_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIcamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIcamin_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIcamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIcamin_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIcamin_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIcamin_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIcamin_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIcamin_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIdamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIdamax_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIdamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIdamax_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIdamax_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIdamax_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIdamax_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIdamax_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIdamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIdamin_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIdamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f64, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIdamin_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIdamin_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIdamin_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIdamin_v2_64(handle: cublasHandle_t, n: i64, x: *const f64, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIdamin_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIsamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIsamax_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIsamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIsamax_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIsamax_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIsamax_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIsamax_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIsamax_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIsamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIsamin_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIsamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIsamin_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIsamin_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIsamin_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIsamin_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIsamin_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIzamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIzamax_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIzamax_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIzamax_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIzamax_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIzamax_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIzamax_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIzamax_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasIzamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIzamin_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIzamin_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasIzamin_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasIzamin_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *mut i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasIzamin_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasIzamin_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, result: *mut i64) -> cublasStatus_t;
        }
        cublasIzamin_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasLoggerConfigure(logIsOn: ::core::ffi::c_int, logToStdOut: ::core::ffi::c_int, logToStdErr: ::core::ffi::c_int, logFileName: *const ::core::ffi::c_char) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_char) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasLoggerConfigure") });
        _f(logIsOn, logToStdOut, logToStdErr, logFileName)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasLoggerConfigure(logIsOn: ::core::ffi::c_int, logToStdOut: ::core::ffi::c_int, logToStdErr: ::core::ffi::c_int, logFileName: *const ::core::ffi::c_char) -> cublasStatus_t;
        }
        cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName)
    }
}
pub unsafe fn cublasNrm2Ex(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasNrm2Ex") });
        _f(handle, n, x, xType, incx, result, resultType, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasNrm2Ex(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasNrm2Ex_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasNrm2Ex_64") });
        _f(handle, n, x, xType, incx, result, resultType, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasNrm2Ex_64(handle: cublasHandle_t, n: i64, x: *const ::core::ffi::c_void, xType: cudaDataType, incx: i64, result: *mut ::core::ffi::c_void, resultType: cudaDataType, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasNrm2Ex_64(handle, n, x, xType, incx, result, resultType, executionType)
    }
}
pub unsafe fn cublasRotEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, c: *const ::core::ffi::c_void, s: *const ::core::ffi::c_void, csType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasRotEx") });
        _f(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasRotEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, c: *const ::core::ffi::c_void, s: *const ::core::ffi::c_void, csType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasRotEx_64(handle: cublasHandle_t, n: i64, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64, c: *const ::core::ffi::c_void, s: *const ::core::ffi::c_void, csType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasRotEx_64") });
        _f(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasRotEx_64(handle: cublasHandle_t, n: i64, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64, c: *const ::core::ffi::c_void, s: *const ::core::ffi::c_void, csType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasRotEx_64(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype)
    }
}
pub unsafe fn cublasRotgEx(handle: cublasHandle_t, a: *mut ::core::ffi::c_void, b: *mut ::core::ffi::c_void, abType: cudaDataType, c: *mut ::core::ffi::c_void, s: *mut ::core::ffi::c_void, csType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasRotgEx") });
        _f(handle, a, b, abType, c, s, csType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasRotgEx(handle: cublasHandle_t, a: *mut ::core::ffi::c_void, b: *mut ::core::ffi::c_void, abType: cudaDataType, c: *mut ::core::ffi::c_void, s: *mut ::core::ffi::c_void, csType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasRotgEx(handle, a, b, abType, c, s, csType, executiontype)
    }
}
pub unsafe fn cublasRotmEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, param: *const ::core::ffi::c_void, paramType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasRotmEx") });
        _f(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasRotmEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int, param: *const ::core::ffi::c_void, paramType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasRotmEx(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasRotmEx_64(handle: cublasHandle_t, n: i64, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64, param: *const ::core::ffi::c_void, paramType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasRotmEx_64") });
        _f(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasRotmEx_64(handle: cublasHandle_t, n: i64, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64, param: *const ::core::ffi::c_void, paramType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasRotmEx_64(handle, n, x, xType, incx, y, yType, incy, param, paramType, executiontype)
    }
}
pub unsafe fn cublasRotmgEx(handle: cublasHandle_t, d1: *mut ::core::ffi::c_void, d1Type: cudaDataType, d2: *mut ::core::ffi::c_void, d2Type: cudaDataType, x1: *mut ::core::ffi::c_void, x1Type: cudaDataType, y1: *const ::core::ffi::c_void, y1Type: cudaDataType, param: *mut ::core::ffi::c_void, paramType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasRotmgEx") });
        _f(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasRotmgEx(handle: cublasHandle_t, d1: *mut ::core::ffi::c_void, d1Type: cudaDataType, d2: *mut ::core::ffi::c_void, d2Type: cudaDataType, x1: *mut ::core::ffi::c_void, x1Type: cudaDataType, y1: *const ::core::ffi::c_void, y1Type: cudaDataType, param: *mut ::core::ffi::c_void, paramType: cudaDataType, executiontype: cudaDataType) -> cublasStatus_t;
        }
        cublasRotmgEx(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype)
    }
}
pub unsafe fn cublasSasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSasum_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t;
        }
        cublasSasum_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSasum_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSasum_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSasum_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut f32) -> cublasStatus_t;
        }
        cublasSasum_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasSaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSaxpy_v2") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSaxpy_v2_64") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSaxpy_v2_64(handle, n, alpha, x, incx, y, incy)
    }
}
pub unsafe fn cublasScalEx(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScalEx") });
        _f(handle, n, alpha, alphaType, x, xType, incx, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScalEx(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasScalEx_64(handle: cublasHandle_t, n: i64, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, executionType: cudaDataType) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, i64, cudaDataType) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScalEx_64") });
        _f(handle, n, alpha, alphaType, x, xType, incx, executionType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScalEx_64(handle: cublasHandle_t, n: i64, alpha: *const ::core::ffi::c_void, alphaType: cudaDataType, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, executionType: cudaDataType) -> cublasStatus_t;
        }
        cublasScalEx_64(handle, n, alpha, alphaType, x, xType, incx, executionType)
    }
}
pub unsafe fn cublasScasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScasum_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScasum_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t;
        }
        cublasScasum_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasScasum_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScasum_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScasum_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut f32) -> cublasStatus_t;
        }
        cublasScasum_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasScnrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScnrm2_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScnrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuComplex, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t;
        }
        cublasScnrm2_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasScnrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuComplex, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScnrm2_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScnrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const cuComplex, incx: i64, result: *mut f32) -> cublasStatus_t;
        }
        cublasScnrm2_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasScopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScopy_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasScopy_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasScopy_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasScopy_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasScopy_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasScopy_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasSdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSdgmm") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const f32, lda: i64, x: *const f32, incx: i64, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, i64, i64, *const f32, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSdgmm_64") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const f32, lda: i64, x: *const f32, incx: i64, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
pub unsafe fn cublasSdot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSdot_v2") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSdot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t;
        }
        cublasSdot_v2(handle, n, x, incx, y, incy, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSdot_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, y: *const f32, incy: i64, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, i64, *const f32, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSdot_v2_64") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSdot_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, y: *const f32, incy: i64, result: *mut f32) -> cublasStatus_t;
        }
        cublasSdot_v2_64(handle, n, x, incx, y, incy, result)
    }
}
pub unsafe fn cublasSetAtomicsMode(handle: cublasHandle_t, mode: cublasAtomicsMode_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasAtomicsMode_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetAtomicsMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetAtomicsMode(handle: cublasHandle_t, mode: cublasAtomicsMode_t) -> cublasStatus_t;
        }
        cublasSetAtomicsMode(handle, mode)
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSetEmulationStrategy(handle: cublasHandle_t, emulationStrategy: cublasEmulationStrategy_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasEmulationStrategy_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetEmulationStrategy") });
        _f(handle, emulationStrategy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetEmulationStrategy(handle: cublasHandle_t, emulationStrategy: cublasEmulationStrategy_t) -> cublasStatus_t;
        }
        cublasSetEmulationStrategy(handle, emulationStrategy)
    }
}
pub unsafe fn cublasSetLoggerCallback(userCallback: cublasLogCallback) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasLogCallback) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetLoggerCallback") });
        _f(userCallback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetLoggerCallback(userCallback: cublasLogCallback) -> cublasStatus_t;
        }
        cublasSetLoggerCallback(userCallback)
    }
}
pub unsafe fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasMath_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetMathMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;
        }
        cublasSetMathMode(handle, mode)
    }
}
pub unsafe fn cublasSetMatrix(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetMatrix") });
        _f(rows, cols, elemSize, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetMatrix(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)
    }
}
pub unsafe fn cublasSetMatrixAsync(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetMatrixAsync") });
        _f(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetMatrixAsync(rows: ::core::ffi::c_int, cols: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, A: *const ::core::ffi::c_void, lda: ::core::ffi::c_int, B: *mut ::core::ffi::c_void, ldb: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSetMatrixAsync_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetMatrixAsync_64") });
        _f(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetMatrixAsync_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasSetMatrixAsync_64(rows, cols, elemSize, A, lda, B, ldb, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSetMatrix_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetMatrix_64") });
        _f(rows, cols, elemSize, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetMatrix_64(rows: i64, cols: i64, elemSize: i64, A: *const ::core::ffi::c_void, lda: i64, B: *mut ::core::ffi::c_void, ldb: i64) -> cublasStatus_t;
        }
        cublasSetMatrix_64(rows, cols, elemSize, A, lda, B, ldb)
    }
}
pub unsafe fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasPointerMode_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetPointerMode_v2") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t;
        }
        cublasSetPointerMode_v2(handle, mode)
    }
}
pub unsafe fn cublasSetSmCountTarget(handle: cublasHandle_t, smCountTarget: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetSmCountTarget") });
        _f(handle, smCountTarget)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetSmCountTarget(handle: cublasHandle_t, smCountTarget: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSetSmCountTarget(handle, smCountTarget)
    }
}
pub unsafe fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetStream_v2") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
        }
        cublasSetStream_v2(handle, streamId)
    }
}
pub unsafe fn cublasSetVector(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, x: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, devicePtr: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetVector") });
        _f(n, elemSize, x, incx, devicePtr, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetVector(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, x: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, devicePtr: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSetVector(n, elemSize, x, incx, devicePtr, incy)
    }
}
pub unsafe fn cublasSetVectorAsync(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, hostPtr: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, devicePtr: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_int, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetVectorAsync") });
        _f(n, elemSize, hostPtr, incx, devicePtr, incy, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetVectorAsync(n: ::core::ffi::c_int, elemSize: ::core::ffi::c_int, hostPtr: *const ::core::ffi::c_void, incx: ::core::ffi::c_int, devicePtr: *mut ::core::ffi::c_void, incy: ::core::ffi::c_int, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasSetVectorAsync(n, elemSize, hostPtr, incx, devicePtr, incy, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSetVectorAsync_64(n: i64, elemSize: i64, hostPtr: *const ::core::ffi::c_void, incx: i64, devicePtr: *mut ::core::ffi::c_void, incy: i64, stream: cudaStream_t) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64, cudaStream_t) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetVectorAsync_64") });
        _f(n, elemSize, hostPtr, incx, devicePtr, incy, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetVectorAsync_64(n: i64, elemSize: i64, hostPtr: *const ::core::ffi::c_void, incx: i64, devicePtr: *mut ::core::ffi::c_void, incy: i64, stream: cudaStream_t) -> cublasStatus_t;
        }
        cublasSetVectorAsync_64(n, elemSize, hostPtr, incx, devicePtr, incy, stream)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSetVector_64(n: i64, elemSize: i64, x: *const ::core::ffi::c_void, incx: i64, devicePtr: *mut ::core::ffi::c_void, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i64, i64, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetVector_64") });
        _f(n, elemSize, x, incx, devicePtr, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetVector_64(n: i64, elemSize: i64, x: *const ::core::ffi::c_void, incx: i64, devicePtr: *mut ::core::ffi::c_void, incy: i64) -> cublasStatus_t;
        }
        cublasSetVector_64(n, elemSize, x, incx, devicePtr, incy)
    }
}
pub unsafe fn cublasSetWorkspace_v2(handle: cublasHandle_t, workspace: *mut ::core::ffi::c_void, workspaceSizeInBytes: usize) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut ::core::ffi::c_void, usize) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSetWorkspace_v2") });
        _f(handle, workspace, workspaceSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSetWorkspace_v2(handle: cublasHandle_t, workspace: *mut ::core::ffi::c_void, workspaceSizeInBytes: usize) -> cublasStatus_t;
        }
        cublasSetWorkspace_v2(handle, workspace, workspaceSizeInBytes)
    }
}
pub unsafe fn cublasSgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgbmv_v2") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgbmv_v2_64") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasSgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, beta: *const f32, B: *const f32, ldb: ::core::ffi::c_int, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgeam") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, beta: *const f32, B: *const f32, ldb: ::core::ffi::c_int, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, beta: *const f32, B: *const f32, ldb: i64, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, *const f32, *const f32, i64, *const f32, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgeam_64") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, beta: *const f32, B: *const f32, ldb: i64, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasSgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut f32, lda: ::core::ffi::c_int, Carray: *const *mut f32, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgelsBatched") });
        _f(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut f32, lda: ::core::ffi::c_int, Carray: *const *mut f32, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
}
pub unsafe fn cublasSgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, Aarray: *const *const f32, lda: ::core::ffi::c_int, Barray: *const *const f32, ldb: ::core::ffi::c_int, beta: *const f32, Carray: *const *mut f32, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const *const f32, ::core::ffi::c_int, *const *const f32, ::core::ffi::c_int, *const f32, *const *mut f32, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmBatched") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, Aarray: *const *const f32, lda: ::core::ffi::c_int, Barray: *const *const f32, ldb: ::core::ffi::c_int, beta: *const f32, Carray: *const *mut f32, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, Aarray: *const *const f32, lda: i64, Barray: *const *const f32, ldb: i64, beta: *const f32, Carray: *const *mut f32, ldc: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f32, *const *const f32, i64, *const *const f32, i64, *const f32, *const *mut f32, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, Aarray: *const *const f32, lda: i64, Barray: *const *const f32, ldb: i64, beta: *const f32, Carray: *const *mut f32, ldc: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasSgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
pub unsafe fn cublasSgemmEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmEx") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmEx(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemmEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f32, *const ::core::ffi::c_void, cudaDataType, i64, *const ::core::ffi::c_void, cudaDataType, i64, *const f32, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmEx_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmEx_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, A: *const ::core::ffi::c_void, Atype: cudaDataType, lda: i64, B: *const ::core::ffi::c_void, Btype: cudaDataType, ldb: i64, beta: *const f32, C: *mut ::core::ffi::c_void, Ctype: cudaDataType, ldc: i64) -> cublasStatus_t;
        }
        cublasSgemmEx_64(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemmGroupedBatched(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const ::core::ffi::c_int, n_array: *const ::core::ffi::c_int, k_array: *const ::core::ffi::c_int, alpha_array: *const f32, Aarray: *const *const f32, lda_array: *const ::core::ffi::c_int, Barray: *const *const f32, ldb_array: *const ::core::ffi::c_int, beta_array: *const f32, Carray: *const *mut f32, ldc_array: *const ::core::ffi::c_int, group_count: ::core::ffi::c_int, group_size: *const ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *const cublasOperation_t, *const cublasOperation_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, *const *const f32, *const ::core::ffi::c_int, *const *const f32, *const ::core::ffi::c_int, *const f32, *const *mut f32, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmGroupedBatched") });
        _f(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmGroupedBatched(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const ::core::ffi::c_int, n_array: *const ::core::ffi::c_int, k_array: *const ::core::ffi::c_int, alpha_array: *const f32, Aarray: *const *const f32, lda_array: *const ::core::ffi::c_int, Barray: *const *const f32, ldb_array: *const ::core::ffi::c_int, beta_array: *const f32, Carray: *const *mut f32, ldc_array: *const ::core::ffi::c_int, group_count: ::core::ffi::c_int, group_size: *const ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemmGroupedBatched(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemmGroupedBatched_64(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const i64, n_array: *const i64, k_array: *const i64, alpha_array: *const f32, Aarray: *const *const f32, lda_array: *const i64, Barray: *const *const f32, ldb_array: *const i64, beta_array: *const f32, Carray: *const *mut f32, ldc_array: *const i64, group_count: i64, group_size: *const i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *const cublasOperation_t, *const cublasOperation_t, *const i64, *const i64, *const i64, *const f32, *const *const f32, *const i64, *const *const f32, *const i64, *const f32, *const *mut f32, *const i64, i64, *const i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmGroupedBatched_64") });
        _f(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmGroupedBatched_64(handle: cublasHandle_t, transa_array: *const cublasOperation_t, transb_array: *const cublasOperation_t, m_array: *const i64, n_array: *const i64, k_array: *const i64, alpha_array: *const f32, Aarray: *const *const f32, lda_array: *const i64, Barray: *const *const f32, ldb_array: *const i64, beta_array: *const f32, Carray: *const *mut f32, ldc_array: *const i64, group_count: i64, group_size: *const i64) -> cublasStatus_t;
        }
        cublasSgemmGroupedBatched_64(handle, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, lda_array, Barray, ldb_array, beta_array, Carray, ldc_array, group_count, group_size)
    }
}
pub unsafe fn cublasSgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const f32, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, *mut f32, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmStridedBatched") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const f32, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, strideA: ::core::ffi::c_longlong, B: *const f32, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const f32, C: *mut f32, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f32, *const f32, i64, ::core::ffi::c_longlong, *const f32, i64, ::core::ffi::c_longlong, *const f32, *mut f32, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemmStridedBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, strideA: ::core::ffi::c_longlong, B: *const f32, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const f32, C: *mut f32, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasSgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
pub unsafe fn cublasSgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemm_v2") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemm_v2_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, Aarray: *const *const f32, lda: ::core::ffi::c_int, xarray: *const *const f32, incx: ::core::ffi::c_int, beta: *const f32, yarray: *const *mut f32, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const *const f32, ::core::ffi::c_int, *const *const f32, ::core::ffi::c_int, *const f32, *const *mut f32, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemvBatched") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, Aarray: *const *const f32, lda: ::core::ffi::c_int, xarray: *const *const f32, incx: ::core::ffi::c_int, beta: *const f32, yarray: *const *mut f32, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f32, Aarray: *const *const f32, lda: i64, xarray: *const *const f32, incx: i64, beta: *const f32, yarray: *const *mut f32, incy: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const f32, *const *const f32, i64, *const *const f32, i64, *const f32, *const *mut f32, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemvBatched_64") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f32, Aarray: *const *const f32, lda: i64, xarray: *const *const f32, incx: i64, beta: *const f32, yarray: *const *mut f32, incy: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasSgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const f32, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, *mut f32, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemvStridedBatched") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const f32, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, strideA: ::core::ffi::c_longlong, x: *const f32, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const f32, y: *mut f32, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const f32, *const f32, i64, ::core::ffi::c_longlong, *const f32, i64, ::core::ffi::c_longlong, *const f32, *mut f32, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemvStridedBatched_64") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, strideA: ::core::ffi::c_longlong, x: *const f32, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const f32, y: *mut f32, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasSgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
pub unsafe fn cublasSgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemv_v2") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgemv_v2_64") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasSgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut f32, lda: ::core::ffi::c_int, TauArray: *const *mut f32, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *const *mut f32, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgeqrfBatched") });
        _f(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut f32, lda: ::core::ffi::c_int, TauArray: *const *mut f32, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
}
pub unsafe fn cublasSger_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSger_v2") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSger_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSger_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSger_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, A: *mut f32, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSger_v2_64") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSger_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, A: *mut f32, lda: i64) -> cublasStatus_t;
        }
        cublasSger_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasSgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut f32, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgetrfBatched") });
        _f(handle, n, A, lda, P, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut f32, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize)
    }
}
pub unsafe fn cublasSgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f32, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut f32, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgetriBatched") });
        _f(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f32, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut f32, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
}
pub unsafe fn cublasSgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const f32, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut f32, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *const f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSgetrsBatched") });
        _f(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const f32, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut f32, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
}
pub unsafe fn cublasSmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f32, lda: ::core::ffi::c_int, Ainv: *const *mut f32, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const f32, ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSmatinvBatched") });
        _f(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const f32, lda: ::core::ffi::c_int, Ainv: *const *mut f32, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
}
pub unsafe fn cublasSnrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSnrm2_v2") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSnrm2_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, result: *mut f32) -> cublasStatus_t;
        }
        cublasSnrm2_v2(handle, n, x, incx, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSnrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSnrm2_v2_64") });
        _f(handle, n, x, incx, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSnrm2_v2_64(handle: cublasHandle_t, n: i64, x: *const f32, incx: i64, result: *mut f32) -> cublasStatus_t;
        }
        cublasSnrm2_v2_64(handle, n, x, incx, result)
    }
}
pub unsafe fn cublasSrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int, c: *const f32, s: *const f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const f32, *const f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSrot_v2") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int, c: *const f32, s: *const f32) -> cublasStatus_t;
        }
        cublasSrot_v2(handle, n, x, incx, y, incy, c, s)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64, c: *const f32, s: *const f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut f32, i64, *mut f32, i64, *const f32, *const f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSrot_v2_64") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64, c: *const f32, s: *const f32) -> cublasStatus_t;
        }
        cublasSrot_v2_64(handle, n, x, incx, y, incy, c, s)
    }
}
pub unsafe fn cublasSrotg_v2(handle: cublasHandle_t, a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut f32, *mut f32, *mut f32, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSrotg_v2") });
        _f(handle, a, b, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSrotg_v2(handle: cublasHandle_t, a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) -> cublasStatus_t;
        }
        cublasSrotg_v2(handle, a, b, c, s)
    }
}
pub unsafe fn cublasSrotm_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int, param: *const f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSrotm_v2") });
        _f(handle, n, x, incx, y, incy, param)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSrotm_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int, param: *const f32) -> cublasStatus_t;
        }
        cublasSrotm_v2(handle, n, x, incx, y, incy, param)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSrotm_v2_64(handle: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64, param: *const f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut f32, i64, *mut f32, i64, *const f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSrotm_v2_64") });
        _f(handle, n, x, incx, y, incy, param)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSrotm_v2_64(handle: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64, param: *const f32) -> cublasStatus_t;
        }
        cublasSrotm_v2_64(handle, n, x, incx, y, incy, param)
    }
}
pub unsafe fn cublasSrotmg_v2(handle: cublasHandle_t, d1: *mut f32, d2: *mut f32, x1: *mut f32, y1: *const f32, param: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut f32, *mut f32, *mut f32, *const f32, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSrotmg_v2") });
        _f(handle, d1, d2, x1, y1, param)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSrotmg_v2(handle: cublasHandle_t, d1: *mut f32, d2: *mut f32, x1: *mut f32, y1: *const f32, param: *mut f32) -> cublasStatus_t;
        }
        cublasSrotmg_v2(handle, d1, d2, x1, y1, param)
    }
}
pub unsafe fn cublasSsbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsbmv_v2") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsbmv_v2_64") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSsbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasSscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f32, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSscal_v2") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f32, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSscal_v2(handle, n, alpha, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f32, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSscal_v2_64") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f32, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasSscal_v2_64(handle, n, alpha, x, incx)
    }
}
pub unsafe fn cublasSspmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, AP: *const f32, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const f32, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSspmv_v2") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSspmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, AP: *const f32, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSspmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSspmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, AP: *const f32, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const f32, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSspmv_v2_64") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSspmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, AP: *const f32, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSspmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasSspr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, AP: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSspr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSspr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, AP: *mut f32) -> cublasStatus_t;
        }
        cublasSspr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSspr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, AP: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const f32, i64, *const f32, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSspr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSspr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, AP: *mut f32) -> cublasStatus_t;
        }
        cublasSspr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
pub unsafe fn cublasSspr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, AP: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSspr_v2") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSspr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, AP: *mut f32) -> cublasStatus_t;
        }
        cublasSspr_v2(handle, uplo, n, alpha, x, incx, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSspr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, AP: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const f32, i64, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSspr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSspr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, AP: *mut f32) -> cublasStatus_t;
        }
        cublasSspr_v2_64(handle, uplo, n, alpha, x, incx, AP)
    }
}
pub unsafe fn cublasSswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSswap_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSswap_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSswap_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSswap_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasSsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsymm_v2") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsymm_v2_64") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasSsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsymv_v2") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, x: *const f32, incx: ::core::ffi::c_int, beta: *const f32, y: *mut f32, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsymv_v2_64") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, A: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) -> cublasStatus_t;
        }
        cublasSsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasSsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, y: *const f32, incy: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, A: *mut f32, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const f32, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, A: *mut f32, lda: i64) -> cublasStatus_t;
        }
        cublasSsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasSsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyr2k_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyr2k_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasSsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyr_v2") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f32, x: *const f32, incx: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, A: *mut f32, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f32, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, A: *mut f32, lda: i64) -> cublasStatus_t;
        }
        cublasSsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
pub unsafe fn cublasSsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyrk_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f32, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyrk_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
pub unsafe fn cublasSsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyrkx") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSsyrkx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, beta: *const f32, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasSsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasStbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStbmv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStbmv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasStbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasStbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStbsv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStbsv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasStbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasStpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f32, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStpmv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f32, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f32, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStpmv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f32, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasStpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasStpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f32, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStpsv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const f32, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f32, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f32, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStpsv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const f32, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasStpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasStpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const f32, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStpttr") });
        _f(handle, uplo, n, AP, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const f32, A: *mut f32, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStpttr(handle, uplo, n, AP, A, lda)
    }
}
pub unsafe fn cublasStrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrmm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, C: *mut f32, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, C: *mut f32, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f32, *const f32, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrmm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, B: *const f32, ldb: i64, C: *mut f32, ldc: i64) -> cublasStatus_t;
        }
        cublasStrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasStrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrmv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrmv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasStrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasStrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const *const f32, lda: ::core::ffi::c_int, B: *const *mut f32, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const *const f32, ::core::ffi::c_int, *const *mut f32, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrsmBatched") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const *const f32, lda: ::core::ffi::c_int, B: *const *mut f32, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f32, A: *const *const f32, lda: i64, B: *const *mut f32, ldb: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f32, *const *const f32, i64, *const *mut f32, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrsmBatched_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f32, A: *const *const f32, lda: i64, B: *const *mut f32, ldb: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasStrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
pub unsafe fn cublasStrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrsm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, B: *mut f32, ldb: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const f32, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrsm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const f32, A: *const f32, lda: i64, B: *mut f32, ldb: i64) -> cublasStatus_t;
        }
        cublasStrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
pub unsafe fn cublasStrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrsv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, x: *mut f32, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasStrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasStrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const f32, i64, *mut f32, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrsv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const f32, lda: i64, x: *mut f32, incx: i64) -> cublasStatus_t;
        }
        cublasStrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasStrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, AP: *mut f32) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasStrttp") });
        _f(handle, uplo, n, A, lda, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasStrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, AP: *mut f32) -> cublasStatus_t;
        }
        cublasStrttp(handle, uplo, n, A, lda, AP)
    }
}
pub unsafe fn cublasSwapEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int, *mut ::core::ffi::c_void, cudaDataType, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSwapEx") });
        _f(handle, n, x, xType, incx, y, yType, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSwapEx(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: ::core::ffi::c_int, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasSwapEx(handle, n, x, xType, incx, y, yType, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasSwapEx_64(handle: cublasHandle_t, n: i64, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut ::core::ffi::c_void, cudaDataType, i64, *mut ::core::ffi::c_void, cudaDataType, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasSwapEx_64") });
        _f(handle, n, x, xType, incx, y, yType, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasSwapEx_64(handle: cublasHandle_t, n: i64, x: *mut ::core::ffi::c_void, xType: cudaDataType, incx: i64, y: *mut ::core::ffi::c_void, yType: cudaDataType, incy: i64) -> cublasStatus_t;
        }
        cublasSwapEx_64(handle, n, x, xType, incx, y, yType, incy)
    }
}
pub unsafe fn cublasUint8gemmBias(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, transc: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const ::core::ffi::c_uchar, A_bias: ::core::ffi::c_int, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_uchar, B_bias: ::core::ffi::c_int, ldb: ::core::ffi::c_int, C: *mut ::core::ffi::c_uchar, C_bias: ::core::ffi::c_int, ldc: ::core::ffi::c_int, C_mult: ::core::ffi::c_int, C_shift: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_uchar, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_uchar, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_uchar, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasUint8gemmBias") });
        _f(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasUint8gemmBias(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, transc: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const ::core::ffi::c_uchar, A_bias: ::core::ffi::c_int, lda: ::core::ffi::c_int, B: *const ::core::ffi::c_uchar, B_bias: ::core::ffi::c_int, ldb: ::core::ffi::c_int, C: *mut ::core::ffi::c_uchar, C_bias: ::core::ffi::c_int, ldc: ::core::ffi::c_int, C_mult: ::core::ffi::c_int, C_shift: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasUint8gemmBias(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias, ldb, C, C_bias, ldc, C_mult, C_shift)
    }
}
pub unsafe fn cublasXerbla(srName: *const ::core::ffi::c_char, info: ::core::ffi::c_int) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, ::core::ffi::c_int);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasXerbla") });
        _f(srName, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasXerbla(srName: *const ::core::ffi::c_char, info: ::core::ffi::c_int);
        }
        cublasXerbla(srName, info)
    }
}
pub unsafe fn cublasZaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZaxpy_v2") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZaxpy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZaxpy_v2_64") });
        _f(handle, n, alpha, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZaxpy_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZaxpy_v2_64(handle, n, alpha, x, incx, y, incy)
    }
}
pub unsafe fn cublasZcopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZcopy_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZcopy_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZcopy_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZcopy_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZcopy_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZcopy_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZcopy_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasZdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdgmm") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdgmm(handle: cublasHandle_t, mode: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZdgmm(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, i64, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdgmm_64") });
        _f(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdgmm_64(handle: cublasHandle_t, mode: cublasSideMode_t, m: i64, n: i64, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZdgmm_64(handle, mode, m, n, A, lda, x, incx, C, ldc)
    }
}
pub unsafe fn cublasZdotc_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, result: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdotc_v2") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdotc_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, result: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZdotc_v2(handle, n, x, incx, y, incy, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZdotc_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, result: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdotc_v2_64") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdotc_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, result: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZdotc_v2_64(handle, n, x, incx, y, incy, result)
    }
}
pub unsafe fn cublasZdotu_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, result: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdotu_v2") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdotu_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, result: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZdotu_v2(handle, n, x, incx, y, incy, result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZdotu_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, result: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdotu_v2_64") });
        _f(handle, n, x, incx, y, incy, result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdotu_v2_64(handle: cublasHandle_t, n: i64, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, result: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZdotu_v2_64(handle, n, x, incx, y, incy, result)
    }
}
pub unsafe fn cublasZdrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int, c: *const f64, s: *const f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const f64, *const f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdrot_v2") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int, c: *const f64, s: *const f64) -> cublasStatus_t;
        }
        cublasZdrot_v2(handle, n, x, incx, y, incy, c, s)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZdrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64, c: *const f64, s: *const f64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut cuDoubleComplex, i64, *mut cuDoubleComplex, i64, *const f64, *const f64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdrot_v2_64") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64, c: *const f64, s: *const f64) -> cublasStatus_t;
        }
        cublasZdrot_v2_64(handle, n, x, incx, y, incy, c, s)
    }
}
pub unsafe fn cublasZdscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f64, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const f64, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdscal_v2") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const f64, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZdscal_v2(handle, n, alpha, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZdscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const f64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZdscal_v2_64") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZdscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const f64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZdscal_v2_64(handle, n, alpha, x, incx)
    }
}
pub unsafe fn cublasZgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgbmv_v2") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgbmv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, kl: ::core::ffi::c_int, ku: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgbmv_v2(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgbmv_v2_64") });
        _f(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgbmv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, kl: i64, ku: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZgbmv_v2_64(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasZgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, beta: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgeam") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgeam(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, beta: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, beta: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: i64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgeam_64") });
        _f(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgeam_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, beta: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: i64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZgeam_64(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasZgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut cuDoubleComplex, lda: ::core::ffi::c_int, Carray: *const *mut cuDoubleComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgelsBatched") });
        _f(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgelsBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *mut cuDoubleComplex, lda: ::core::ffi::c_int, Carray: *const *mut cuDoubleComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, devInfoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
    }
}
pub unsafe fn cublasZgemm3m(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemm3m") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemm3m(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemm3m_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemm3m_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemm3m_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZgemm3m_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, Barray: *const *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, Carray: *const *mut cuDoubleComplex, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const *const cuDoubleComplex, ::core::ffi::c_int, *const *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *const *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemmBatched") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemmBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, Barray: *const *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, Carray: *const *mut cuDoubleComplex, ldc: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: i64, Barray: *const *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, Carray: *const *mut cuDoubleComplex, ldc: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuDoubleComplex, *const *const cuDoubleComplex, i64, *const *const cuDoubleComplex, i64, *const cuDoubleComplex, *const *mut cuDoubleComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemmBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemmBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: i64, Barray: *const *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, Carray: *const *mut cuDoubleComplex, ldc: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasZgemmBatched_64(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    }
}
pub unsafe fn cublasZgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemmStridedBatched") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemmStridedBatched(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, strideB: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int, strideC: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, strideA: ::core::ffi::c_longlong, B: *const cuDoubleComplex, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, ::core::ffi::c_longlong, *const cuDoubleComplex, i64, ::core::ffi::c_longlong, *const cuDoubleComplex, *mut cuDoubleComplex, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemmStridedBatched_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemmStridedBatched_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, strideA: ::core::ffi::c_longlong, B: *const cuDoubleComplex, ldb: i64, strideB: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64, strideC: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasZgemmStridedBatched_64(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)
    }
}
pub unsafe fn cublasZgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemm_v2") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemm_v2(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, cublasOperation_t, i64, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemm_v2_64") });
        _f(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemm_v2_64(handle: cublasHandle_t, transa: cublasOperation_t, transb: cublasOperation_t, m: i64, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZgemm_v2_64(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, xarray: *const *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, yarray: *const *mut cuDoubleComplex, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const *const cuDoubleComplex, ::core::ffi::c_int, *const *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *const *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemvBatched") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemvBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, xarray: *const *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, yarray: *const *mut cuDoubleComplex, incy: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemvBatched(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: i64, xarray: *const *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, yarray: *const *mut cuDoubleComplex, incy: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const *const cuDoubleComplex, i64, *const *const cuDoubleComplex, i64, *const cuDoubleComplex, *const *mut cuDoubleComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemvBatched_64") });
        _f(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemvBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, Aarray: *const *const cuDoubleComplex, lda: i64, xarray: *const *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, yarray: *const *mut cuDoubleComplex, incy: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasZgemvBatched_64(handle, trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemvStridedBatched") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemvStridedBatched(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, stridex: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int, stridey: ::core::ffi::c_longlong, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, strideA: ::core::ffi::c_longlong, x: *const cuDoubleComplex, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, ::core::ffi::c_longlong, *const cuDoubleComplex, i64, ::core::ffi::c_longlong, *const cuDoubleComplex, *mut cuDoubleComplex, i64, ::core::ffi::c_longlong, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemvStridedBatched_64") });
        _f(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemvStridedBatched_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, strideA: ::core::ffi::c_longlong, x: *const cuDoubleComplex, incx: i64, stridex: ::core::ffi::c_longlong, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64, stridey: ::core::ffi::c_longlong, batchCount: i64) -> cublasStatus_t;
        }
        cublasZgemvStridedBatched_64(handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
    }
}
pub unsafe fn cublasZgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemv_v2") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemv_v2(handle: cublasHandle_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgemv_v2_64") });
        _f(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgemv_v2_64(handle: cublasHandle_t, trans: cublasOperation_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZgemv_v2_64(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasZgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut cuDoubleComplex, lda: ::core::ffi::c_int, TauArray: *const *mut cuDoubleComplex, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *const *mut cuDoubleComplex, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgeqrfBatched") });
        _f(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgeqrfBatched(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Aarray: *const *mut cuDoubleComplex, lda: ::core::ffi::c_int, TauArray: *const *mut cuDoubleComplex, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize)
    }
}
pub unsafe fn cublasZgerc_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgerc_v2") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgerc_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgerc_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgerc_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgerc_v2_64") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgerc_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t;
        }
        cublasZgerc_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasZgeru_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgeru_v2") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgeru_v2(handle: cublasHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgeru_v2(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZgeru_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgeru_v2_64") });
        _f(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgeru_v2_64(handle: cublasHandle_t, m: i64, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t;
        }
        cublasZgeru_v2_64(handle, m, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasZgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut cuDoubleComplex, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgetrfBatched") });
        _f(handle, n, A, lda, P, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgetrfBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *mut cuDoubleComplex, lda: ::core::ffi::c_int, P: *mut ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize)
    }
}
pub unsafe fn cublasZgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut cuDoubleComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgetriBatched") });
        _f(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgetriBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, P: *const ::core::ffi::c_int, C: *const *mut cuDoubleComplex, ldc: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize)
    }
}
pub unsafe fn cublasZgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut cuDoubleComplex, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const *const cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZgetrsBatched") });
        _f(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZgetrsBatched(handle: cublasHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Aarray: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, Barray: *const *mut cuDoubleComplex, ldb: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize)
    }
}
pub unsafe fn cublasZhbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhbmv_v2") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZhbmv_v2(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZhbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhbmv_v2_64") });
        _f(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZhbmv_v2_64(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasZhemm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhemm_v2") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhemm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZhemm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZhemm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhemm_v2_64") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhemm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZhemm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZhemv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhemv_v2") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhemv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZhemv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZhemv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhemv_v2_64") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhemv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZhemv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasZher2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZher2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZher2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZher2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZher2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZher2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZher2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t;
        }
        cublasZher2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasZher2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZher2k_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZher2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZher2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZher2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const f64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const f64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZher2k_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZher2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const f64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZher2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZher_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZher_v2") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZher_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZher_v2(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZher_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const cuDoubleComplex, incx: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZher_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZher_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const cuDoubleComplex, incx: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t;
        }
        cublasZher_v2_64(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
pub unsafe fn cublasZherk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, beta: *const f64, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZherk_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZherk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, beta: *const f64, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZherk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZherk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const cuDoubleComplex, lda: i64, beta: *const f64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const f64, *const cuDoubleComplex, i64, *const f64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZherk_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZherk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const f64, A: *const cuDoubleComplex, lda: i64, beta: *const f64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZherk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
pub unsafe fn cublasZherkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZherkx") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZherkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZherkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZherkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const f64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const f64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZherkx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZherkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const f64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZherkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZhpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, AP: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhpmv_v2") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, AP: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZhpmv_v2(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZhpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, AP: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhpmv_v2_64") });
        _f(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, AP: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZhpmv_v2_64(handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasZhpr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, AP: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhpr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhpr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, AP: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZhpr2_v2(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZhpr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, AP: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhpr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhpr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, AP: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZhpr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, AP)
    }
}
pub unsafe fn cublasZhpr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, AP: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhpr_v2") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhpr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const f64, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, AP: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZhpr_v2(handle, uplo, n, alpha, x, incx, AP)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZhpr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const cuDoubleComplex, incx: i64, AP: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const f64, *const cuDoubleComplex, i64, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZhpr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZhpr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const f64, x: *const cuDoubleComplex, incx: i64, AP: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZhpr_v2_64(handle, uplo, n, alpha, x, incx, AP)
    }
}
pub unsafe fn cublasZmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, Ainv: *const *mut cuDoubleComplex, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const *const cuDoubleComplex, ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZmatinvBatched") });
        _f(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZmatinvBatched(handle: cublasHandle_t, n: ::core::ffi::c_int, A: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, Ainv: *const *mut cuDoubleComplex, lda_inv: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize)
    }
}
pub unsafe fn cublasZrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int, c: *const f64, s: *const cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const f64, *const cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZrot_v2") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZrot_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int, c: *const f64, s: *const cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZrot_v2(handle, n, x, incx, y, incy, c, s)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64, c: *const f64, s: *const cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut cuDoubleComplex, i64, *mut cuDoubleComplex, i64, *const f64, *const cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZrot_v2_64") });
        _f(handle, n, x, incx, y, incy, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZrot_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64, c: *const f64, s: *const cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZrot_v2_64(handle, n, x, incx, y, incy, c, s)
    }
}
pub unsafe fn cublasZrotg_v2(handle: cublasHandle_t, a: *mut cuDoubleComplex, b: *mut cuDoubleComplex, c: *mut f64, s: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut f64, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZrotg_v2") });
        _f(handle, a, b, c, s)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZrotg_v2(handle: cublasHandle_t, a: *mut cuDoubleComplex, b: *mut cuDoubleComplex, c: *mut f64, s: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZrotg_v2(handle, a, b, c, s)
    }
}
pub unsafe fn cublasZscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZscal_v2") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZscal_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZscal_v2(handle, n, alpha, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZscal_v2_64") });
        _f(handle, n, alpha, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZscal_v2_64(handle: cublasHandle_t, n: i64, alpha: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZscal_v2_64(handle, n, alpha, x, incx)
    }
}
pub unsafe fn cublasZswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZswap_v2") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZswap_v2(handle: cublasHandle_t, n: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZswap_v2(handle, n, x, incx, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, i64, *mut cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZswap_v2_64") });
        _f(handle, n, x, incx, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZswap_v2_64(handle: cublasHandle_t, n: i64, x: *mut cuDoubleComplex, incx: i64, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZswap_v2_64(handle, n, x, incx, y, incy)
    }
}
pub unsafe fn cublasZsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsymm_v2") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsymm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsymm_v2(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsymm_v2_64") });
        _f(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsymm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZsymm_v2_64(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsymv_v2") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsymv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsymv_v2(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsymv_v2_64") });
        _f(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsymv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, x: *const cuDoubleComplex, incx: i64, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, incy: i64) -> cublasStatus_t;
        }
        cublasZsymv_v2_64(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    }
}
pub unsafe fn cublasZsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyr2_v2") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyr2_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, y: *const cuDoubleComplex, incy: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsyr2_v2(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyr2_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyr2_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, y: *const cuDoubleComplex, incy: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t;
        }
        cublasZsyr2_v2_64(handle, uplo, n, alpha, x, incx, y, incy, A, lda)
    }
}
pub unsafe fn cublasZsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyr2k_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyr2k_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsyr2k_v2(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyr2k_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyr2k_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZsyr2k_v2_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyr_v2") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyr_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsyr_v2(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyr_v2_64") });
        _f(handle, uplo, n, alpha, x, incx, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyr_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, n: i64, alpha: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: i64, A: *mut cuDoubleComplex, lda: i64) -> cublasStatus_t;
        }
        cublasZsyr_v2_64(handle, uplo, n, alpha, x, incx, A, lda)
    }
}
pub unsafe fn cublasZsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyrk_v2") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyrk_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsyrk_v2(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyrk_v2_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyrk_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZsyrk_v2_64(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    }
}
pub unsafe fn cublasZsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyrkx") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyrkx(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZsyrkx_64") });
        _f(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZsyrkx_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, n: i64, k: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZsyrkx_64(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cublasZtbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtbmv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtbmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtbmv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtbmv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtbmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZtbmv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasZtbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtbsv_v2") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtbsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtbsv_v2(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtbsv_v2_64") });
        _f(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtbsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, k: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZtbsv_v2_64(handle, uplo, trans, diag, n, k, A, lda, x, incx)
    }
}
pub unsafe fn cublasZtpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtpmv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtpmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtpmv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtpmv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtpmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZtpmv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasZtpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtpsv_v2") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtpsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtpsv_v2(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuDoubleComplex, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtpsv_v2_64") });
        _f(handle, uplo, trans, diag, n, AP, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtpsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZtpsv_v2_64(handle, uplo, trans, diag, n, AP, x, incx)
    }
}
pub unsafe fn cublasZtpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const cuDoubleComplex, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtpttr") });
        _f(handle, uplo, n, AP, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtpttr(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, AP: *const cuDoubleComplex, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtpttr(handle, uplo, n, AP, A, lda)
    }
}
pub unsafe fn cublasZtrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrmm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrmm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrmm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrmm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *const cuDoubleComplex, ldb: i64, C: *mut cuDoubleComplex, ldc: i64) -> cublasStatus_t;
        }
        cublasZtrmm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    }
}
pub unsafe fn cublasZtrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrmv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrmv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtrmv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrmv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrmv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZtrmv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasZtrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const *mut cuDoubleComplex, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const *const cuDoubleComplex, ::core::ffi::c_int, *const *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrsmBatched") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrsmBatched(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const *mut cuDoubleComplex, ldb: ::core::ffi::c_int, batchCount: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const *const cuDoubleComplex, lda: i64, B: *const *mut cuDoubleComplex, ldb: i64, batchCount: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuDoubleComplex, *const *const cuDoubleComplex, i64, *const *mut cuDoubleComplex, i64, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrsmBatched_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrsmBatched_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const *const cuDoubleComplex, lda: i64, B: *const *mut cuDoubleComplex, ldb: i64, batchCount: i64) -> cublasStatus_t;
        }
        cublasZtrsmBatched_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
    }
}
pub unsafe fn cublasZtrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrsm_v2") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrsm_v2(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtrsm_v2(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *mut cuDoubleComplex, ldb: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, i64, *const cuDoubleComplex, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrsm_v2_64") });
        _f(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrsm_v2_64(handle: cublasHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, m: i64, n: i64, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: i64, B: *mut cuDoubleComplex, ldb: i64) -> cublasStatus_t;
        }
        cublasZtrsm_v2_64(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    }
}
pub unsafe fn cublasZtrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrsv_v2") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrsv_v2(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, x: *mut cuDoubleComplex, incx: ::core::ffi::c_int) -> cublasStatus_t;
        }
        cublasZtrsv_v2(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cublasZtrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, i64, *const cuDoubleComplex, i64, *mut cuDoubleComplex, i64) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrsv_v2_64") });
        _f(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrsv_v2_64(handle: cublasHandle_t, uplo: cublasFillMode_t, trans: cublasOperation_t, diag: cublasDiagType_t, n: i64, A: *const cuDoubleComplex, lda: i64, x: *mut cuDoubleComplex, incx: i64) -> cublasStatus_t;
        }
        cublasZtrsv_v2_64(handle, uplo, trans, diag, n, A, lda, x, incx)
    }
}
pub unsafe fn cublasZtrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, AP: *mut cuDoubleComplex) -> cublasStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cublasHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex) -> cublasStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cublasZtrttp") });
        _f(handle, uplo, n, A, lda, AP)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cublasZtrttp(handle: cublasHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, AP: *mut cuDoubleComplex) -> cublasStatus_t;
        }
        cublasZtrttp(handle, uplo, n, A, lda, AP)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cublas"];
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
        let lib_names = std::vec!["cublas"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
