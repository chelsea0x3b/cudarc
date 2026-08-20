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
pub use self::libraryPropertyType_t as libraryPropertyType;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type FILE = _IO_FILE;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type _IO_lock_t = ::core::ffi::c_void;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type __off64_t = ::core::ffi::c_long;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type __off_t = ::core::ffi::c_long;
pub type csrqrInfo_t = *mut csrqrInfo;
pub type cuComplex = cuFloatComplex;
pub type cuDoubleComplex = double2;
pub type cuFloatComplex = float2;
pub type cudaLibMgGrid_t = *mut ::core::ffi::c_void;
pub type cudaLibMgMatrixDesc_t = *mut ::core::ffi::c_void;
pub type cudaStream_t = *mut CUstream_st;
pub type cusolverDnHandle_t = *mut cusolverDnContext;
pub type cusolverDnIRSInfos_t = *mut cusolverDnIRSInfos;
pub type cusolverDnIRSParams_t = *mut cusolverDnIRSParams;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusolverDnLoggerCallback_t = ::core::option::Option<unsafe extern "C" fn(logLevel: ::core::ffi::c_int, functionName: *const ::core::ffi::c_char, message: *const ::core::ffi::c_char)>;
pub type cusolverDnParams_t = *mut cusolverDnParams;
pub type cusolverRfHandle_t = *mut cusolverRfCommon;
pub type cusolverSpHandle_t = *mut cusolverSpContext;
pub type cusolver_int_t = ::core::ffi::c_int;
pub type cusparseMatDescr_t = *mut cusparseMatDescr;
pub type gesvdjInfo_t = *mut gesvdjInfo;
pub type syevjInfo_t = *mut syevjInfo;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasDiagType_t {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1,
}
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
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1,
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
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaEmulationMantissaControl_t {
    CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC = 0,
    CUDA_EMULATION_MANTISSA_CONTROL_FIXED = 1,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaEmulationSpecialValuesSupport_t {
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT = 65535,
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE = 0,
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY = 1,
    CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN = 2,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaEmulationStrategy_t {
    CUDA_EMULATION_STRATEGY_DEFAULT = 0,
    CUDA_EMULATION_STRATEGY_PERFORMANT = 1,
    CUDA_EMULATION_STRATEGY_EAGER = 2,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverAlgMode_t {
    CUSOLVER_ALG_0 = 0,
    CUSOLVER_ALG_1 = 1,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverAlgMode_t {
    CUSOLVER_ALG_0 = 0,
    CUSOLVER_ALG_1 = 1,
    CUSOLVER_ALG_2 = 2,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDeterministicMode_t {
    CUSOLVER_DETERMINISTIC_RESULTS = 1,
    CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDirectMode_t {
    CUBLAS_DIRECT_FORWARD = 0,
    CUBLAS_DIRECT_BACKWARD = 1,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDnFunction_t {
    CUSOLVERDN_GETRF = 0,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDnFunction_t {
    CUSOLVERDN_GETRF = 0,
    CUSOLVERDN_POTRF = 1,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDnFunction_t {
    CUSOLVERDN_GETRF = 0,
    CUSOLVERDN_POTRF = 1,
    CUSOLVERDN_SYEVBATCHED = 2,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDnFunction_t {
    CUSOLVERDN_GETRF = 0,
    CUSOLVERDN_POTRF = 1,
    CUSOLVERDN_SYEVBATCHED = 2,
    CUSOLVERDN_GEQRF = 3,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigComp_t {
    CUSOLVER_EIG_COMP_N = 10,
    CUSOLVER_EIG_COMP_I = 11,
    CUSOLVER_EIG_COMP_V = 12,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigMode_t {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigRange_t {
    CUSOLVER_EIG_RANGE_ALL = 1001,
    CUSOLVER_EIG_RANGE_I = 1002,
    CUSOLVER_EIG_RANGE_V = 1003,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigType_t {
    CUSOLVER_EIG_TYPE_1 = 1,
    CUSOLVER_EIG_TYPE_2 = 2,
    CUSOLVER_EIG_TYPE_3 = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverIRSRefinement_t {
    CUSOLVER_IRS_REFINE_NOT_SET = 1100,
    CUSOLVER_IRS_REFINE_NONE = 1101,
    CUSOLVER_IRS_REFINE_CLASSICAL = 1102,
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES = 1103,
    CUSOLVER_IRS_REFINE_GMRES = 1104,
    CUSOLVER_IRS_REFINE_GMRES_GMRES = 1105,
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND = 1106,
    CUSOLVER_PREC_DD = 1150,
    CUSOLVER_PREC_SS = 1151,
    CUSOLVER_PREC_SHT = 1152,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverMathMode_t {
    CUSOLVER_DEFAULT_MATH = 1,
    CUSOLVER_FP32_EMULATED_BF16X9_MATH = 2,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverMathMode_t {
    CUSOLVER_DEFAULT_MATH = 1,
    CUSOLVER_FP32_EMULATED_BF16X9_MATH = 2,
    CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH = 4,
    CUSOLVER_FP32_FP64_EMULATED_MATH = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverNorm_t {
    CUSOLVER_INF_NORM = 104,
    CUSOLVER_MAX_NORM = 105,
    CUSOLVER_ONE_NORM = 106,
    CUSOLVER_FRO_NORM = 107,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverPrecType_t {
    CUSOLVER_R_8I = 1201,
    CUSOLVER_R_8U = 1202,
    CUSOLVER_R_64F = 1203,
    CUSOLVER_R_32F = 1204,
    CUSOLVER_R_16F = 1205,
    CUSOLVER_R_16BF = 1206,
    CUSOLVER_R_TF32 = 1207,
    CUSOLVER_R_AP = 1208,
    CUSOLVER_C_8I = 1211,
    CUSOLVER_C_8U = 1212,
    CUSOLVER_C_64F = 1213,
    CUSOLVER_C_32F = 1214,
    CUSOLVER_C_16F = 1215,
    CUSOLVER_C_16BF = 1216,
    CUSOLVER_C_TF32 = 1217,
    CUSOLVER_C_AP = 1218,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfFactorization_t {
    CUSOLVERRF_FACTORIZATION_ALG0 = 0,
    CUSOLVERRF_FACTORIZATION_ALG1 = 1,
    CUSOLVERRF_FACTORIZATION_ALG2 = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfMatrixFormat_t {
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0,
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfNumericBoostReport_t {
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0,
    CUSOLVERRF_NUMERIC_BOOST_USED = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfResetValuesFastMode_t {
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0,
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfTriangularSolve_t {
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfUnitDiagonal_t {
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0,
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1,
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3,
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
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverStorevMode_t {
    CUBLAS_STOREV_COLUMNWISE = 0,
    CUBLAS_STOREV_ROWWISE = 1,
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
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct _IO_FILE {
    pub _flags: ::core::ffi::c_int,
    pub _IO_read_ptr: *mut ::core::ffi::c_char,
    pub _IO_read_end: *mut ::core::ffi::c_char,
    pub _IO_read_base: *mut ::core::ffi::c_char,
    pub _IO_write_base: *mut ::core::ffi::c_char,
    pub _IO_write_ptr: *mut ::core::ffi::c_char,
    pub _IO_write_end: *mut ::core::ffi::c_char,
    pub _IO_buf_base: *mut ::core::ffi::c_char,
    pub _IO_buf_end: *mut ::core::ffi::c_char,
    pub _IO_save_base: *mut ::core::ffi::c_char,
    pub _IO_backup_base: *mut ::core::ffi::c_char,
    pub _IO_save_end: *mut ::core::ffi::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1usize],
    pub _lock: *mut _IO_lock_t,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: usize,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20usize],
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_codecvt {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_marker {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_wide_data {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csrqrInfo {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnIRSInfos {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnIRSParams {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnParams {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverRfCommon {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverSpContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseMatDescr {
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gesvdjInfo {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct syevjInfo {
    _unused: [u8; 0],
}
impl cublasOperation_t {
    pub const CUBLAS_OP_HERMITAN: cublasOperation_t = cublasOperation_t::CUBLAS_OP_C;
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
pub unsafe fn cusolverDnCCgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCCgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCCgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCCgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCCgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCCgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCCgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCCgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCCgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCCgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCCgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCCgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCEgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCEgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCEgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCEgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCEgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCEgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCEgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCEgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCEgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCEgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCEgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCEgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCKgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCKgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCKgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCKgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCKgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCKgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCKgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCKgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCKgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCKgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCKgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCKgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCYgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCYgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCYgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCYgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCYgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCYgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCYgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCYgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCYgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnCYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnCYgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut cuComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCYgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCYgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuComplex, lddb: cusolver_int_t, dX: *mut cuComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnCYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnCgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, D: *mut f32, E: *mut f32, TAUQ: *mut cuComplex, TAUP: *mut cuComplex, Work: *mut cuComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut f32, *mut cuComplex, *mut cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgebrd") });
        _f(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, D: *mut f32, E: *mut f32, TAUQ: *mut cuComplex, TAUP: *mut cuComplex, Work: *mut cuComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnCgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgebrd_bufferSize") });
        _f(handle, m, n, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgebrd_bufferSize(handle, m, n, Lwork)
    }
}
pub unsafe fn cusolverDnCgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, TAU: *mut cuComplex, Workspace: *mut cuComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgeqrf") });
        _f(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, TAU: *mut cuComplex, Workspace: *mut cuComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnCgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgeqrf_bufferSize") });
        _f(handle, m, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnCgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, S: *mut f32, U: *mut cuComplex, ldu: ::core::ffi::c_int, VT: *mut cuComplex, ldvt: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, rwork: *mut f32, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_schar, ::core::ffi::c_schar, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvd") });
        _f(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, S: *mut f32, U: *mut cuComplex, ldu: ::core::ffi::c_int, VT: *mut cuComplex, ldvt: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, rwork: *mut f32, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
}
pub unsafe fn cusolverDnCgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvd_bufferSize") });
        _f(handle, m, n, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgesvd_bufferSize(handle, m, n, lwork)
    }
}
pub unsafe fn cusolverDnCgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f32, strideS: ::core::ffi::c_longlong, d_U: *mut cuComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut cuComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut cuComplex, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f32, ::core::ffi::c_longlong, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvdaStridedBatched") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f32, strideS: ::core::ffi::c_longlong, d_U: *mut cuComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut cuComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut cuComplex, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
}
pub unsafe fn cusolverDnCgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f32, strideS: ::core::ffi::c_longlong, d_U: *const cuComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const cuComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, ::core::ffi::c_longlong, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvdaStridedBatched_bufferSize") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f32, strideS: ::core::ffi::c_longlong, d_U: *const cuComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const cuComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
}
pub unsafe fn cusolverDnCgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, S: *mut f32, U: *mut cuComplex, ldu: ::core::ffi::c_int, V: *mut cuComplex, ldv: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvdj") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, S: *mut f32, U: *mut cuComplex, ldu: ::core::ffi::c_int, V: *mut cuComplex, ldv: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnCgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, S: *mut f32, U: *mut cuComplex, ldu: ::core::ffi::c_int, V: *mut cuComplex, ldv: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvdjBatched") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, S: *mut f32, U: *mut cuComplex, ldu: ::core::ffi::c_int, V: *mut cuComplex, ldv: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnCgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, S: *const f32, U: *const cuComplex, ldu: ::core::ffi::c_int, V: *const cuComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvdjBatched_bufferSize") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, S: *const f32, U: *const cuComplex, ldu: ::core::ffi::c_int, V: *const cuComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnCgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, S: *const f32, U: *const cuComplex, ldu: ::core::ffi::c_int, V: *const cuComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgesvdj_bufferSize") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, S: *const f32, U: *const cuComplex, ldu: ::core::ffi::c_int, V: *const cuComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
}
pub unsafe fn cusolverDnCgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Workspace: *mut cuComplex, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgetrf") });
        _f(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Workspace: *mut cuComplex, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
}
pub unsafe fn cusolverDnCgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgetrf_bufferSize") });
        _f(handle, m, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnCgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCgetrs") });
        _f(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnCheevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevd") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCheevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevd_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    }
}
pub unsafe fn cusolverDnCheevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevdx") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCheevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevdx_bufferSize") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnCheevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevj") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnCheevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevjBatched") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnCheevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevjBatched_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnCheevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCheevj_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCheevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
}
pub unsafe fn cusolverDnChegvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChegvd") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChegvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnChegvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChegvd_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChegvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
}
pub unsafe fn cusolverDnChegvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChegvdx") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChegvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnChegvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChegvdx_bufferSize") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChegvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnChegvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChegvj") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChegvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnChegvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChegvj_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChegvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
}
pub unsafe fn cusolverDnChetrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, d: *mut f32, e: *mut f32, tau: *mut cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut f32, *mut f32, *mut cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChetrd") });
        _f(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChetrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, d: *mut f32, e: *mut f32, tau: *mut cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnChetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnChetrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, d: *const f32, e: *const f32, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const f32, *const f32, *const cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnChetrd_bufferSize") });
        _f(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnChetrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, d: *const f32, e: *const f32, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnChetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
}
pub unsafe fn cusolverDnClaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnClaswp") });
        _f(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnClaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnClaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
}
pub unsafe fn cusolverDnClauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnClauum") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnClauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnClauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnClauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnClauum_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnClauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnClauum_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnCpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Workspace: *mut cuComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotrf") });
        _f(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Workspace: *mut cuComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnCpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut cuComplex, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotrfBatched") });
        _f(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut cuComplex, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
}
pub unsafe fn cusolverDnCpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotrf_bufferSize") });
        _f(handle, uplo, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnCpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotri") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnCpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotri_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnCpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotrs") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnCpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut cuComplex, lda: ::core::ffi::c_int, B: *mut *mut cuComplex, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut cuComplex, ::core::ffi::c_int, *mut *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCpotrsBatched") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut cuComplex, lda: ::core::ffi::c_int, B: *mut *mut cuComplex, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
}
pub unsafe fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverDnHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t;
        }
        cusolverDnCreate(handle)
    }
}
pub unsafe fn cusolverDnCreateGesvdjInfo(info: *mut gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCreateGesvdjInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCreateGesvdjInfo(info: *mut gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnCreateGesvdjInfo(info)
    }
}
pub unsafe fn cusolverDnCreateParams(params: *mut cusolverDnParams_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverDnParams_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCreateParams") });
        _f(params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCreateParams(params: *mut cusolverDnParams_t) -> cusolverStatus_t;
        }
        cusolverDnCreateParams(params)
    }
}
pub unsafe fn cusolverDnCreateSyevjInfo(info: *mut syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCreateSyevjInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCreateSyevjInfo(info: *mut syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnCreateSyevjInfo(info)
    }
}
pub unsafe fn cusolverDnCsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCsytrf") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCsytrf_bufferSize") });
        _f(handle, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCsytrf_bufferSize(handle, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnCsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCsytri") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCsytri_bufferSize") });
        _f(handle, uplo, n, A, lda, ipiv, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    }
}
pub unsafe fn cusolverDnCungbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCungbr") });
        _f(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCungbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCungbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCungbr_bufferSize") });
        _f(handle, side, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCungbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnCungqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCungqr") });
        _f(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCungqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCungqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCungqr_bufferSize") });
        _f(handle, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCungqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnCungtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCungtr") });
        _f(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCungtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCungtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCungtr_bufferSize") });
        _f(handle, uplo, n, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCungtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnCunmqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCunmqr") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCunmqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnCunmqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, C: *const cuComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCunmqr_bufferSize") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCunmqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, C: *const cuComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnCunmtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *mut cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCunmtr") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCunmtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int, tau: *mut cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int, work: *mut cuComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
}
pub unsafe fn cusolverDnCunmtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, C: *const cuComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnCunmtr_bufferSize") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnCunmtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuComplex, lda: ::core::ffi::c_int, tau: *const cuComplex, C: *const cuComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnDBgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDBgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDBgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDBgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDBgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDBgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDBgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDBgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDBgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDBgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDBgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDBgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDDgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDDgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDDgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDDgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDDgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDDgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDDgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDDgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDDgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDDgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDDgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDDgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDDgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDDgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDDgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDDgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDHgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDHgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDHgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDHgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDHgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDHgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDHgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDHgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDHgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDHgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDHgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDHgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDSgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDSgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDSgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDSgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDSgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDSgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDSgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDSgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDSgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDSgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDSgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDSgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDXgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDXgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDXgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDXgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDXgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDXgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDXgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDXgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDXgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnDXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnDXgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f64, cusolver_int_t, *mut cusolver_int_t, *mut f64, cusolver_int_t, *mut f64, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDXgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDXgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f64, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f64, lddb: cusolver_int_t, dX: *mut f64, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnDXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDestroy") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> cusolverStatus_t;
        }
        cusolverDnDestroy(handle)
    }
}
pub unsafe fn cusolverDnDestroyGesvdjInfo(info: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDestroyGesvdjInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDestroyGesvdjInfo(info: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDestroyGesvdjInfo(info)
    }
}
pub unsafe fn cusolverDnDestroyParams(params: cusolverDnParams_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnParams_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDestroyParams") });
        _f(params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDestroyParams(params: cusolverDnParams_t) -> cusolverStatus_t;
        }
        cusolverDnDestroyParams(params)
    }
}
pub unsafe fn cusolverDnDestroySyevjInfo(info: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDestroySyevjInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDestroySyevjInfo(info: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDestroySyevjInfo(info)
    }
}
pub unsafe fn cusolverDnDgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, D: *mut f64, E: *mut f64, TAUQ: *mut f64, TAUP: *mut f64, Work: *mut f64, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, *mut f64, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgebrd") });
        _f(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, D: *mut f64, E: *mut f64, TAUQ: *mut f64, TAUP: *mut f64, Work: *mut f64, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnDgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgebrd_bufferSize") });
        _f(handle, m, n, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgebrd_bufferSize(handle, m, n, Lwork)
    }
}
pub unsafe fn cusolverDnDgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, TAU: *mut f64, Workspace: *mut f64, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgeqrf") });
        _f(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, TAU: *mut f64, Workspace: *mut f64, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnDgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgeqrf_bufferSize") });
        _f(handle, m, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnDgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, S: *mut f64, U: *mut f64, ldu: ::core::ffi::c_int, VT: *mut f64, ldvt: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, rwork: *mut f64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_schar, ::core::ffi::c_schar, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvd") });
        _f(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, S: *mut f64, U: *mut f64, ldu: ::core::ffi::c_int, VT: *mut f64, ldvt: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, rwork: *mut f64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
}
pub unsafe fn cusolverDnDgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvd_bufferSize") });
        _f(handle, m, n, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgesvd_bufferSize(handle, m, n, lwork)
    }
}
pub unsafe fn cusolverDnDgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f64, strideS: ::core::ffi::c_longlong, d_U: *mut f64, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut f64, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut f64, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f64, ::core::ffi::c_longlong, *mut f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvdaStridedBatched") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f64, strideS: ::core::ffi::c_longlong, d_U: *mut f64, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut f64, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut f64, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
}
pub unsafe fn cusolverDnDgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f64, strideS: ::core::ffi::c_longlong, d_U: *const f64, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const f64, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, ::core::ffi::c_longlong, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvdaStridedBatched_bufferSize") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f64, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f64, strideS: ::core::ffi::c_longlong, d_U: *const f64, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const f64, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
}
pub unsafe fn cusolverDnDgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, S: *mut f64, U: *mut f64, ldu: ::core::ffi::c_int, V: *mut f64, ldv: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvdj") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, S: *mut f64, U: *mut f64, ldu: ::core::ffi::c_int, V: *mut f64, ldv: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnDgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, S: *mut f64, U: *mut f64, ldu: ::core::ffi::c_int, V: *mut f64, ldv: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvdjBatched") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, S: *mut f64, U: *mut f64, ldu: ::core::ffi::c_int, V: *mut f64, ldv: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnDgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, S: *const f64, U: *const f64, ldu: ::core::ffi::c_int, V: *const f64, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvdjBatched_bufferSize") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, S: *const f64, U: *const f64, ldu: ::core::ffi::c_int, V: *const f64, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnDgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, S: *const f64, U: *const f64, ldu: ::core::ffi::c_int, V: *const f64, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgesvdj_bufferSize") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, S: *const f64, U: *const f64, ldu: ::core::ffi::c_int, V: *const f64, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
}
pub unsafe fn cusolverDnDgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Workspace: *mut f64, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgetrf") });
        _f(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Workspace: *mut f64, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
}
pub unsafe fn cusolverDnDgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgetrf_bufferSize") });
        _f(handle, m, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnDgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDgetrs") });
        _f(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnDlaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDlaswp") });
        _f(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDlaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
}
pub unsafe fn cusolverDnDlauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDlauum") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDlauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnDlauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDlauum_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDlauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnDorgbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *const f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDorgbr") });
        _f(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDorgbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *const f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDorgbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDorgbr_bufferSize") });
        _f(handle, side, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDorgbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnDorgqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *const f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDorgqr") });
        _f(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDorgqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *const f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDorgqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDorgqr_bufferSize") });
        _f(handle, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDorgqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnDorgtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *const f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDorgtr") });
        _f(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDorgtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *const f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDorgtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDorgtr_bufferSize") });
        _f(handle, uplo, n, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDorgtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnDormqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, C: *mut f64, ldc: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDormqr") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDormqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, C: *mut f64, ldc: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnDormqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, C: *const f64, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDormqr_bufferSize") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDormqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, C: *const f64, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnDormtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *mut f64, C: *mut f64, ldc: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDormtr") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDormtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, tau: *mut f64, C: *mut f64, ldc: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDormtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, C: *const f64, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDormtr_bufferSize") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDormtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, tau: *const f64, C: *const f64, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnDpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Workspace: *mut f64, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotrf") });
        _f(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Workspace: *mut f64, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnDpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut f64, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotrfBatched") });
        _f(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut f64, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
}
pub unsafe fn cusolverDnDpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotrf_bufferSize") });
        _f(handle, uplo, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnDpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotri") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnDpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotri_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnDpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotrs") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnDpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut f64, lda: ::core::ffi::c_int, B: *mut *mut f64, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut f64, ::core::ffi::c_int, *mut *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDpotrsBatched") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut f64, lda: ::core::ffi::c_int, B: *mut *mut f64, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
}
pub unsafe fn cusolverDnDsyevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevd") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsyevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevd_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    }
}
pub unsafe fn cusolverDnDsyevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevdx") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsyevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevdx_bufferSize") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnDsyevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevj") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnDsyevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevjBatched") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnDsyevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevjBatched_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnDsyevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsyevj_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsyevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
}
pub unsafe fn cusolverDnDsygvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsygvd") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsygvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsygvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsygvd_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsygvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
}
pub unsafe fn cusolverDnDsygvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsygvdx") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsygvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsygvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsygvdx_bufferSize") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsygvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnDsygvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsygvj") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsygvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnDsygvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsygvj_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsygvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
}
pub unsafe fn cusolverDnDsytrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, d: *mut f64, e: *mut f64, tau: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, *mut f64, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsytrd") });
        _f(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsytrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, d: *mut f64, e: *mut f64, tau: *mut f64, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsytrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, d: *const f64, e: *const f64, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *const f64, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsytrd_bufferSize") });
        _f(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsytrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, d: *const f64, e: *const f64, tau: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
}
pub unsafe fn cusolverDnDsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsytrf") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsytrf_bufferSize") });
        _f(handle, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsytrf_bufferSize(handle, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnDsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsytri") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut f64, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnDsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnDsytri_bufferSize") });
        _f(handle, uplo, n, A, lda, ipiv, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnDsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnDsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGeqrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *mut ::core::ffi::c_void, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGeqrf") });
        _f(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, pBuffer, workspaceInBytes, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGeqrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *mut ::core::ffi::c_void, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnGeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, pBuffer, workspaceInBytes, info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGeqrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGeqrf_bufferSize") });
        _f(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGeqrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnGeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGesvd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *mut ::core::ffi::c_void, dataTypeU: cudaDataType, U: *mut ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *mut ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, ::core::ffi::c_schar, ::core::ffi::c_schar, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGesvd") });
        _f(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, pBuffer, workspaceInBytes, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGesvd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *mut ::core::ffi::c_void, dataTypeU: cudaDataType, U: *mut ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *mut ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnGesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, pBuffer, workspaceInBytes, info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGesvd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *const ::core::ffi::c_void, dataTypeU: cudaDataType, U: *const ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *const ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, ::core::ffi::c_schar, ::core::ffi::c_schar, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGesvd_bufferSize") });
        _f(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGesvd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *const ::core::ffi::c_void, dataTypeU: cudaDataType, U: *const ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *const ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnGesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytes)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetDeterministicMode(handle: cusolverDnHandle_t, mode: *mut cusolverDeterministicMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut cusolverDeterministicMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetDeterministicMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetDeterministicMode(handle: cusolverDnHandle_t, mode: *mut cusolverDeterministicMode_t) -> cusolverStatus_t;
        }
        cusolverDnGetDeterministicMode(handle, mode)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetEmulationSpecialValuesSupport(handle: cusolverDnHandle_t, mask: *mut cudaEmulationSpecialValuesSupport_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut cudaEmulationSpecialValuesSupport_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetEmulationSpecialValuesSupport") });
        _f(handle, mask)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetEmulationSpecialValuesSupport(handle: cusolverDnHandle_t, mask: *mut cudaEmulationSpecialValuesSupport_t) -> cusolverStatus_t;
        }
        cusolverDnGetEmulationSpecialValuesSupport(handle, mask)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetEmulationStrategy(handle: cusolverDnHandle_t, strategy: *mut cudaEmulationStrategy_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut cudaEmulationStrategy_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetEmulationStrategy") });
        _f(handle, strategy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetEmulationStrategy(handle: cusolverDnHandle_t, strategy: *mut cudaEmulationStrategy_t) -> cusolverStatus_t;
        }
        cusolverDnGetEmulationStrategy(handle, strategy)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetFixedPointEmulationMantissaBitOffset(handle: cusolverDnHandle_t, mantissaBitOffset: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetFixedPointEmulationMantissaBitOffset") });
        _f(handle, mantissaBitOffset)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetFixedPointEmulationMantissaBitOffset(handle: cusolverDnHandle_t, mantissaBitOffset: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnGetFixedPointEmulationMantissaBitOffset(handle, mantissaBitOffset)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetFixedPointEmulationMantissaControl(handle: cusolverDnHandle_t, control: *mut cudaEmulationMantissaControl_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut cudaEmulationMantissaControl_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetFixedPointEmulationMantissaControl") });
        _f(handle, control)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetFixedPointEmulationMantissaControl(handle: cusolverDnHandle_t, control: *mut cudaEmulationMantissaControl_t) -> cusolverStatus_t;
        }
        cusolverDnGetFixedPointEmulationMantissaControl(handle, control)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetFixedPointEmulationMaxMantissaBitCount(handle: cusolverDnHandle_t, mantissaBitCount: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetFixedPointEmulationMaxMantissaBitCount") });
        _f(handle, mantissaBitCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetFixedPointEmulationMaxMantissaBitCount(handle: cusolverDnHandle_t, mantissaBitCount: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnGetFixedPointEmulationMaxMantissaBitCount(handle, mantissaBitCount)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnGetMathMode(handle: cusolverDnHandle_t, mode: *mut cusolverMathMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut cusolverMathMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetMathMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetMathMode(handle: cusolverDnHandle_t, mode: *mut cusolverMathMode_t) -> cusolverStatus_t;
        }
        cusolverDnGetMathMode(handle, mode)
    }
}
pub unsafe fn cusolverDnGetStream(handle: cusolverDnHandle_t, streamId: *mut cudaStream_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, *mut cudaStream_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetStream") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetStream(handle: cusolverDnHandle_t, streamId: *mut cudaStream_t) -> cusolverStatus_t;
        }
        cusolverDnGetStream(handle, streamId)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGetrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, ipiv: *mut i64, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetrf") });
        _f(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, pBuffer, workspaceInBytes, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, ipiv: *mut i64, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnGetrf(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, pBuffer, workspaceInBytes, info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGetrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetrf_bufferSize") });
        _f(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnGetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnGetrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, trans: cublasOperation_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, *const i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnGetrs") });
        _f(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnGetrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, trans: cublasOperation_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnGetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)
    }
}
pub unsafe fn cusolverDnIRSInfosCreate(infos_ptr: *mut cusolverDnIRSInfos_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverDnIRSInfos_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosCreate") });
        _f(infos_ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosCreate(infos_ptr: *mut cusolverDnIRSInfos_t) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosCreate(infos_ptr)
    }
}
pub unsafe fn cusolverDnIRSInfosDestroy(infos: cusolverDnIRSInfos_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSInfos_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosDestroy") });
        _f(infos)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosDestroy(infos: cusolverDnIRSInfos_t) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosDestroy(infos)
    }
}
pub unsafe fn cusolverDnIRSInfosGetMaxIters(infos: cusolverDnIRSInfos_t, maxiters: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSInfos_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosGetMaxIters") });
        _f(infos, maxiters)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosGetMaxIters(infos: cusolverDnIRSInfos_t, maxiters: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosGetMaxIters(infos, maxiters)
    }
}
pub unsafe fn cusolverDnIRSInfosGetNiters(infos: cusolverDnIRSInfos_t, niters: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSInfos_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosGetNiters") });
        _f(infos, niters)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosGetNiters(infos: cusolverDnIRSInfos_t, niters: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosGetNiters(infos, niters)
    }
}
pub unsafe fn cusolverDnIRSInfosGetOuterNiters(infos: cusolverDnIRSInfos_t, outer_niters: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSInfos_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosGetOuterNiters") });
        _f(infos, outer_niters)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosGetOuterNiters(infos: cusolverDnIRSInfos_t, outer_niters: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosGetOuterNiters(infos, outer_niters)
    }
}
pub unsafe fn cusolverDnIRSInfosGetResidualHistory(infos: cusolverDnIRSInfos_t, residual_history: *mut *mut ::core::ffi::c_void) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSInfos_t, *mut *mut ::core::ffi::c_void) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosGetResidualHistory") });
        _f(infos, residual_history)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosGetResidualHistory(infos: cusolverDnIRSInfos_t, residual_history: *mut *mut ::core::ffi::c_void) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosGetResidualHistory(infos, residual_history)
    }
}
pub unsafe fn cusolverDnIRSInfosRequestResidual(infos: cusolverDnIRSInfos_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSInfos_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSInfosRequestResidual") });
        _f(infos)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSInfosRequestResidual(infos: cusolverDnIRSInfos_t) -> cusolverStatus_t;
        }
        cusolverDnIRSInfosRequestResidual(infos)
    }
}
pub unsafe fn cusolverDnIRSParamsCreate(params_ptr: *mut cusolverDnIRSParams_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverDnIRSParams_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsCreate") });
        _f(params_ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsCreate(params_ptr: *mut cusolverDnIRSParams_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsCreate(params_ptr)
    }
}
pub unsafe fn cusolverDnIRSParamsDestroy(params: cusolverDnIRSParams_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsDestroy") });
        _f(params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsDestroy(params: cusolverDnIRSParams_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsDestroy(params)
    }
}
pub unsafe fn cusolverDnIRSParamsDisableFallback(params: cusolverDnIRSParams_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsDisableFallback") });
        _f(params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsDisableFallback(params: cusolverDnIRSParams_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsDisableFallback(params)
    }
}
pub unsafe fn cusolverDnIRSParamsEnableFallback(params: cusolverDnIRSParams_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsEnableFallback") });
        _f(params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsEnableFallback(params: cusolverDnIRSParams_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsEnableFallback(params)
    }
}
pub unsafe fn cusolverDnIRSParamsGetMaxIters(params: cusolverDnIRSParams_t, maxiters: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsGetMaxIters") });
        _f(params, maxiters)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsGetMaxIters(params: cusolverDnIRSParams_t, maxiters: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsGetMaxIters(params, maxiters)
    }
}
pub unsafe fn cusolverDnIRSParamsSetMaxIters(params: cusolverDnIRSParams_t, maxiters: cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetMaxIters") });
        _f(params, maxiters)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetMaxIters(params: cusolverDnIRSParams_t, maxiters: cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetMaxIters(params, maxiters)
    }
}
pub unsafe fn cusolverDnIRSParamsSetMaxItersInner(params: cusolverDnIRSParams_t, maxiters_inner: cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetMaxItersInner") });
        _f(params, maxiters_inner)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetMaxItersInner(params: cusolverDnIRSParams_t, maxiters_inner: cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner)
    }
}
pub unsafe fn cusolverDnIRSParamsSetRefinementSolver(params: cusolverDnIRSParams_t, refinement_solver: cusolverIRSRefinement_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, cusolverIRSRefinement_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetRefinementSolver") });
        _f(params, refinement_solver)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetRefinementSolver(params: cusolverDnIRSParams_t, refinement_solver: cusolverIRSRefinement_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetRefinementSolver(params, refinement_solver)
    }
}
pub unsafe fn cusolverDnIRSParamsSetSolverLowestPrecision(params: cusolverDnIRSParams_t, solver_lowest_precision: cusolverPrecType_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, cusolverPrecType_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetSolverLowestPrecision") });
        _f(params, solver_lowest_precision)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetSolverLowestPrecision(params: cusolverDnIRSParams_t, solver_lowest_precision: cusolverPrecType_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetSolverLowestPrecision(params, solver_lowest_precision)
    }
}
pub unsafe fn cusolverDnIRSParamsSetSolverMainPrecision(params: cusolverDnIRSParams_t, solver_main_precision: cusolverPrecType_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, cusolverPrecType_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetSolverMainPrecision") });
        _f(params, solver_main_precision)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetSolverMainPrecision(params: cusolverDnIRSParams_t, solver_main_precision: cusolverPrecType_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetSolverMainPrecision(params, solver_main_precision)
    }
}
pub unsafe fn cusolverDnIRSParamsSetSolverPrecisions(params: cusolverDnIRSParams_t, solver_main_precision: cusolverPrecType_t, solver_lowest_precision: cusolverPrecType_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, cusolverPrecType_t, cusolverPrecType_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetSolverPrecisions") });
        _f(params, solver_main_precision, solver_lowest_precision)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetSolverPrecisions(params: cusolverDnIRSParams_t, solver_main_precision: cusolverPrecType_t, solver_lowest_precision: cusolverPrecType_t) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetSolverPrecisions(params, solver_main_precision, solver_lowest_precision)
    }
}
pub unsafe fn cusolverDnIRSParamsSetTol(params: cusolverDnIRSParams_t, val: f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetTol") });
        _f(params, val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetTol(params: cusolverDnIRSParams_t, val: f64) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetTol(params, val)
    }
}
pub unsafe fn cusolverDnIRSParamsSetTolInner(params: cusolverDnIRSParams_t, val: f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnIRSParams_t, f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSParamsSetTolInner") });
        _f(params, val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSParamsSetTolInner(params: cusolverDnIRSParams_t, val: f64) -> cusolverStatus_t;
        }
        cusolverDnIRSParamsSetTolInner(params, val)
    }
}
pub unsafe fn cusolverDnIRSXgels(handle: cusolverDnHandle_t, gels_irs_params: cusolverDnIRSParams_t, gels_irs_infos: cusolverDnIRSInfos_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut ::core::ffi::c_void, ldda: cusolver_int_t, dB: *mut ::core::ffi::c_void, lddb: cusolver_int_t, dX: *mut ::core::ffi::c_void, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, niters: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut ::core::ffi::c_void, cusolver_int_t, *mut ::core::ffi::c_void, cusolver_int_t, *mut ::core::ffi::c_void, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSXgels") });
        _f(handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSXgels(handle: cusolverDnHandle_t, gels_irs_params: cusolverDnIRSParams_t, gels_irs_infos: cusolverDnIRSInfos_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut ::core::ffi::c_void, ldda: cusolver_int_t, dB: *mut ::core::ffi::c_void, lddb: cusolver_int_t, dX: *mut ::core::ffi::c_void, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, niters: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSXgels(handle, gels_irs_params, gels_irs_infos, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
    }
}
pub unsafe fn cusolverDnIRSXgels_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnIRSParams_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSXgels_bufferSize") });
        _f(handle, params, m, n, nrhs, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSXgels_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnIRSParams_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnIRSXgels_bufferSize(handle, params, m, n, nrhs, lwork_bytes)
    }
}
pub unsafe fn cusolverDnIRSXgesv(handle: cusolverDnHandle_t, gesv_irs_params: cusolverDnIRSParams_t, gesv_irs_infos: cusolverDnIRSInfos_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut ::core::ffi::c_void, ldda: cusolver_int_t, dB: *mut ::core::ffi::c_void, lddb: cusolver_int_t, dX: *mut ::core::ffi::c_void, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, niters: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolverDnIRSInfos_t, cusolver_int_t, cusolver_int_t, *mut ::core::ffi::c_void, cusolver_int_t, *mut ::core::ffi::c_void, cusolver_int_t, *mut ::core::ffi::c_void, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSXgesv") });
        _f(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSXgesv(handle: cusolverDnHandle_t, gesv_irs_params: cusolverDnIRSParams_t, gesv_irs_infos: cusolverDnIRSInfos_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut ::core::ffi::c_void, ldda: cusolver_int_t, dB: *mut ::core::ffi::c_void, lddb: cusolver_int_t, dX: *mut ::core::ffi::c_void, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, niters: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, niters, d_info)
    }
}
pub unsafe fn cusolverDnIRSXgesv_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnIRSParams_t, n: cusolver_int_t, nrhs: cusolver_int_t, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnIRSParams_t, cusolver_int_t, cusolver_int_t, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnIRSXgesv_bufferSize") });
        _f(handle, params, n, nrhs, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnIRSXgesv_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnIRSParams_t, n: cusolver_int_t, nrhs: cusolver_int_t, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnIRSXgesv_bufferSize(handle, params, n, nrhs, lwork_bytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnPotrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnPotrf") });
        _f(handle, params, uplo, n, dataTypeA, A, lda, computeType, pBuffer, workspaceInBytes, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnPotrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnPotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, pBuffer, workspaceInBytes, info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnPotrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnPotrf_bufferSize") });
        _f(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnPotrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnPotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnPotrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnPotrs") });
        _f(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnPotrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnPotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)
    }
}
pub unsafe fn cusolverDnSBgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSBgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSBgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSBgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSBgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSBgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSBgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSBgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSBgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSBgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSBgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSBgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSBgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSBgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSBgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSBgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSHgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSHgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSHgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSHgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSHgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSHgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSHgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSHgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSHgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSHgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSHgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSHgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSHgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSHgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSHgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSHgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSSgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSSgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSSgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSSgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSSgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSSgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSSgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSSgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSSgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSSgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSSgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSSgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSSgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSSgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSSgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSSgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSXgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSXgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSXgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSXgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSXgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSXgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSXgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSXgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSXgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSXgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSXgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnSXgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnSXgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut f32, cusolver_int_t, *mut cusolver_int_t, *mut f32, cusolver_int_t, *mut f32, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSXgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSXgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut f32, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut f32, lddb: cusolver_int_t, dX: *mut f32, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSXgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnSetAdvOptions(params: cusolverDnParams_t, function: cusolverDnFunction_t, algo: cusolverAlgMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnParams_t, cusolverDnFunction_t, cusolverAlgMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetAdvOptions") });
        _f(params, function, algo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetAdvOptions(params: cusolverDnParams_t, function: cusolverDnFunction_t, algo: cusolverAlgMode_t) -> cusolverStatus_t;
        }
        cusolverDnSetAdvOptions(params, function, algo)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetDeterministicMode(handle: cusolverDnHandle_t, mode: cusolverDeterministicMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDeterministicMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetDeterministicMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetDeterministicMode(handle: cusolverDnHandle_t, mode: cusolverDeterministicMode_t) -> cusolverStatus_t;
        }
        cusolverDnSetDeterministicMode(handle, mode)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetEmulationSpecialValuesSupport(handle: cusolverDnHandle_t, mask: cudaEmulationSpecialValuesSupport_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cudaEmulationSpecialValuesSupport_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetEmulationSpecialValuesSupport") });
        _f(handle, mask)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetEmulationSpecialValuesSupport(handle: cusolverDnHandle_t, mask: cudaEmulationSpecialValuesSupport_t) -> cusolverStatus_t;
        }
        cusolverDnSetEmulationSpecialValuesSupport(handle, mask)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetEmulationStrategy(handle: cusolverDnHandle_t, strategy: cudaEmulationStrategy_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cudaEmulationStrategy_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetEmulationStrategy") });
        _f(handle, strategy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetEmulationStrategy(handle: cusolverDnHandle_t, strategy: cudaEmulationStrategy_t) -> cusolverStatus_t;
        }
        cusolverDnSetEmulationStrategy(handle, strategy)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetFixedPointEmulationMantissaBitOffset(handle: cusolverDnHandle_t, mantissaBitOffset: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetFixedPointEmulationMantissaBitOffset") });
        _f(handle, mantissaBitOffset)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetFixedPointEmulationMantissaBitOffset(handle: cusolverDnHandle_t, mantissaBitOffset: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSetFixedPointEmulationMantissaBitOffset(handle, mantissaBitOffset)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetFixedPointEmulationMantissaControl(handle: cusolverDnHandle_t, control: cudaEmulationMantissaControl_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cudaEmulationMantissaControl_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetFixedPointEmulationMantissaControl") });
        _f(handle, control)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetFixedPointEmulationMantissaControl(handle: cusolverDnHandle_t, control: cudaEmulationMantissaControl_t) -> cusolverStatus_t;
        }
        cusolverDnSetFixedPointEmulationMantissaControl(handle, control)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetFixedPointEmulationMaxMantissaBitCount(handle: cusolverDnHandle_t, mantissaBitCount: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetFixedPointEmulationMaxMantissaBitCount") });
        _f(handle, mantissaBitCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetFixedPointEmulationMaxMantissaBitCount(handle: cusolverDnHandle_t, mantissaBitCount: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSetFixedPointEmulationMaxMantissaBitCount(handle, mantissaBitCount)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnSetMathMode(handle: cusolverDnHandle_t, mode: cusolverMathMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverMathMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetMathMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetMathMode(handle: cusolverDnHandle_t, mode: cusolverMathMode_t) -> cusolverStatus_t;
        }
        cusolverDnSetMathMode(handle, mode)
    }
}
pub unsafe fn cusolverDnSetStream(handle: cusolverDnHandle_t, streamId: cudaStream_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cudaStream_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSetStream") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSetStream(handle: cusolverDnHandle_t, streamId: cudaStream_t) -> cusolverStatus_t;
        }
        cusolverDnSetStream(handle, streamId)
    }
}
pub unsafe fn cusolverDnSgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, D: *mut f32, E: *mut f32, TAUQ: *mut f32, TAUP: *mut f32, Work: *mut f32, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgebrd") });
        _f(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, D: *mut f32, E: *mut f32, TAUQ: *mut f32, TAUP: *mut f32, Work: *mut f32, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnSgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgebrd_bufferSize") });
        _f(handle, m, n, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgebrd_bufferSize(handle, m, n, Lwork)
    }
}
pub unsafe fn cusolverDnSgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, TAU: *mut f32, Workspace: *mut f32, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgeqrf") });
        _f(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, TAU: *mut f32, Workspace: *mut f32, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnSgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgeqrf_bufferSize") });
        _f(handle, m, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnSgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, S: *mut f32, U: *mut f32, ldu: ::core::ffi::c_int, VT: *mut f32, ldvt: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, rwork: *mut f32, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_schar, ::core::ffi::c_schar, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvd") });
        _f(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, S: *mut f32, U: *mut f32, ldu: ::core::ffi::c_int, VT: *mut f32, ldvt: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, rwork: *mut f32, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
}
pub unsafe fn cusolverDnSgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvd_bufferSize") });
        _f(handle, m, n, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgesvd_bufferSize(handle, m, n, lwork)
    }
}
pub unsafe fn cusolverDnSgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f32, strideS: ::core::ffi::c_longlong, d_U: *mut f32, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut f32, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut f32, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f32, ::core::ffi::c_longlong, *mut f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvdaStridedBatched") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f32, strideS: ::core::ffi::c_longlong, d_U: *mut f32, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut f32, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut f32, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
}
pub unsafe fn cusolverDnSgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f32, strideS: ::core::ffi::c_longlong, d_U: *const f32, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const f32, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, ::core::ffi::c_longlong, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f32, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvdaStridedBatched_bufferSize") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const f32, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f32, strideS: ::core::ffi::c_longlong, d_U: *const f32, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const f32, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
}
pub unsafe fn cusolverDnSgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, S: *mut f32, U: *mut f32, ldu: ::core::ffi::c_int, V: *mut f32, ldv: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvdj") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, S: *mut f32, U: *mut f32, ldu: ::core::ffi::c_int, V: *mut f32, ldv: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnSgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, S: *mut f32, U: *mut f32, ldu: ::core::ffi::c_int, V: *mut f32, ldv: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvdjBatched") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, S: *mut f32, U: *mut f32, ldu: ::core::ffi::c_int, V: *mut f32, ldv: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnSgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, S: *const f32, U: *const f32, ldu: ::core::ffi::c_int, V: *const f32, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvdjBatched_bufferSize") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, S: *const f32, U: *const f32, ldu: ::core::ffi::c_int, V: *const f32, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnSgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, S: *const f32, U: *const f32, ldu: ::core::ffi::c_int, V: *const f32, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgesvdj_bufferSize") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, S: *const f32, U: *const f32, ldu: ::core::ffi::c_int, V: *const f32, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
}
pub unsafe fn cusolverDnSgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Workspace: *mut f32, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgetrf") });
        _f(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Workspace: *mut f32, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
}
pub unsafe fn cusolverDnSgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgetrf_bufferSize") });
        _f(handle, m, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnSgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSgetrs") });
        _f(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnSlaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSlaswp") });
        _f(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSlaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
}
pub unsafe fn cusolverDnSlauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSlauum") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSlauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnSlauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSlauum_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSlauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnSorgbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *const f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSorgbr") });
        _f(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSorgbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *const f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSorgbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSorgbr_bufferSize") });
        _f(handle, side, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSorgbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnSorgqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *const f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSorgqr") });
        _f(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSorgqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *const f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSorgqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSorgqr_bufferSize") });
        _f(handle, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSorgqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnSorgtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *const f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSorgtr") });
        _f(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSorgtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *const f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSorgtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSorgtr_bufferSize") });
        _f(handle, uplo, n, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSorgtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnSormqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, C: *mut f32, ldc: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSormqr") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSormqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, C: *mut f32, ldc: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnSormqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, C: *const f32, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSormqr_bufferSize") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSormqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, C: *const f32, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnSormtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *mut f32, C: *mut f32, ldc: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSormtr") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSormtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, tau: *mut f32, C: *mut f32, ldc: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSormtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, C: *const f32, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSormtr_bufferSize") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSormtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, tau: *const f32, C: *const f32, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnSpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Workspace: *mut f32, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotrf") });
        _f(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Workspace: *mut f32, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnSpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut f32, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotrfBatched") });
        _f(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut f32, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
}
pub unsafe fn cusolverDnSpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotrf_bufferSize") });
        _f(handle, uplo, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnSpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotri") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnSpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotri_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnSpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotrs") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnSpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut f32, lda: ::core::ffi::c_int, B: *mut *mut f32, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut f32, ::core::ffi::c_int, *mut *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSpotrsBatched") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut f32, lda: ::core::ffi::c_int, B: *mut *mut f32, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
}
pub unsafe fn cusolverDnSsyevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevd") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsyevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevd_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    }
}
pub unsafe fn cusolverDnSsyevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevdx") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsyevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevdx_bufferSize") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnSsyevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevj") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnSsyevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevjBatched") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnSsyevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevjBatched_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnSsyevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsyevj_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsyevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
}
pub unsafe fn cusolverDnSsygvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsygvd") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsygvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsygvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsygvd_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsygvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
}
pub unsafe fn cusolverDnSsygvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsygvdx") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsygvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsygvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, f32, f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsygvdx_bufferSize") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsygvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, vl: f32, vu: f32, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnSsygvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsygvj") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsygvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, W: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnSsygvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsygvj_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsygvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, W: *const f32, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
}
pub unsafe fn cusolverDnSsytrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, d: *mut f32, e: *mut f32, tau: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut f32, *mut f32, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsytrd") });
        _f(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsytrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, d: *mut f32, e: *mut f32, tau: *mut f32, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsytrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsytrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, d: *const f32, e: *const f32, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *const f32, *const f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsytrd_bufferSize") });
        _f(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsytrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, d: *const f32, e: *const f32, tau: *const f32, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
}
pub unsafe fn cusolverDnSsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsytrf") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsytrf_bufferSize") });
        _f(handle, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsytrf_bufferSize(handle, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnSsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsytri") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut f32, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnSsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSsytri_bufferSize") });
        _f(handle, uplo, n, A, lda, ipiv, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnSyevd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSyevd") });
        _f(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, pBuffer, workspaceInBytes, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSyevd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, pBuffer, workspaceInBytes, info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnSyevd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSyevd_bufferSize") });
        _f(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSyevd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnSyevdx(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig64: *mut i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, i64, i64, *mut i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSyevdx") });
        _f(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, pBuffer, workspaceInBytes, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSyevdx(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig64: *mut i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, pBuffer: *mut ::core::ffi::c_void, workspaceInBytes: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnSyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, pBuffer, workspaceInBytes, info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cusolverDnSyevdx_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, h_meig: *mut i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, i64, i64, *mut i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnSyevdx_bufferSize") });
        _f(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnSyevdx_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, h_meig: *mut i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnSyevdx_bufferSize(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytes)
    }
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXgeev(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobvl: cusolverEigMode_t, jobvr: cusolverEigMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, dataTypeVL: cudaDataType, VL: *mut ::core::ffi::c_void, ldvl: i64, dataTypeVR: cudaDataType, VR: *mut ::core::ffi::c_void, ldvr: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgeev") });
        _f(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgeev(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobvl: cusolverEigMode_t, jobvr: cusolverEigMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, dataTypeVL: cudaDataType, VL: *mut ::core::ffi::c_void, ldvl: i64, dataTypeVR: cudaDataType, VR: *mut ::core::ffi::c_void, ldvr: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgeev(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXgeev_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobvl: cusolverEigMode_t, jobvr: cusolverEigMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, dataTypeVL: cudaDataType, VL: *const ::core::ffi::c_void, ldvl: i64, dataTypeVR: cudaDataType, VR: *const ::core::ffi::c_void, ldvr: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgeev_bufferSize") });
        _f(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgeev_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobvl: cusolverEigMode_t, jobvr: cusolverEigMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, dataTypeVL: cudaDataType, VL: *const ::core::ffi::c_void, ldvl: i64, dataTypeVR: cudaDataType, VR: *const ::core::ffi::c_void, ldvr: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXgeev_bufferSize(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXgeqrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgeqrf") });
        _f(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgeqrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXgeqrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgeqrf_bufferSize") });
        _f(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgeqrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXgeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXgesvd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *mut ::core::ffi::c_void, dataTypeU: cudaDataType, U: *mut ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *mut ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, ::core::ffi::c_schar, ::core::ffi::c_schar, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvd") });
        _f(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *mut ::core::ffi::c_void, dataTypeU: cudaDataType, U: *mut ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *mut ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgesvd(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXgesvd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *const ::core::ffi::c_void, dataTypeU: cudaDataType, U: *const ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *const ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, ::core::ffi::c_schar, ::core::ffi::c_schar, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvd_bufferSize") });
        _f(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *const ::core::ffi::c_void, dataTypeU: cudaDataType, U: *const ::core::ffi::c_void, ldu: i64, dataTypeVT: cudaDataType, VT: *const ::core::ffi::c_void, ldvt: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXgesvd_bufferSize(handle, params, jobu, jobvt, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeVT, VT, ldvt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXgesvdjGetResidual(handle: cusolverDnHandle_t, info: gesvdjInfo_t, residual: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, gesvdjInfo_t, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdjGetResidual") });
        _f(handle, info, residual)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdjGetResidual(handle: cusolverDnHandle_t, info: gesvdjInfo_t, residual: *mut f64) -> cusolverStatus_t;
        }
        cusolverDnXgesvdjGetResidual(handle, info, residual)
    }
}
pub unsafe fn cusolverDnXgesvdjGetSweeps(handle: cusolverDnHandle_t, info: gesvdjInfo_t, executed_sweeps: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, gesvdjInfo_t, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdjGetSweeps") });
        _f(handle, info, executed_sweeps)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdjGetSweeps(handle: cusolverDnHandle_t, info: gesvdjInfo_t, executed_sweeps: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgesvdjGetSweeps(handle, info, executed_sweeps)
    }
}
pub unsafe fn cusolverDnXgesvdjSetMaxSweeps(info: gesvdjInfo_t, max_sweeps: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdjSetMaxSweeps") });
        _f(info, max_sweeps)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdjSetMaxSweeps(info: gesvdjInfo_t, max_sweeps: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
    }
}
pub unsafe fn cusolverDnXgesvdjSetSortEig(info: gesvdjInfo_t, sort_svd: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdjSetSortEig") });
        _f(info, sort_svd)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdjSetSortEig(info: gesvdjInfo_t, sort_svd: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgesvdjSetSortEig(info, sort_svd)
    }
}
pub unsafe fn cusolverDnXgesvdjSetTolerance(info: gesvdjInfo_t, tolerance: f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(gesvdjInfo_t, f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdjSetTolerance") });
        _f(info, tolerance)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdjSetTolerance(info: gesvdjInfo_t, tolerance: f64) -> cusolverStatus_t;
        }
        cusolverDnXgesvdjSetTolerance(info, tolerance)
    }
}
pub unsafe fn cusolverDnXgesvdp(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *mut ::core::ffi::c_void, dataTypeU: cudaDataType, U: *mut ::core::ffi::c_void, ldu: i64, dataTypeV: cudaDataType, V: *mut ::core::ffi::c_void, ldv: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int, h_err_sigma: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, ::core::ffi::c_int, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdp") });
        _f(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdp(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *mut ::core::ffi::c_void, dataTypeU: cudaDataType, U: *mut ::core::ffi::c_void, ldu: i64, dataTypeV: cudaDataType, V: *mut ::core::ffi::c_void, ldv: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int, h_err_sigma: *mut f64) -> cusolverStatus_t;
        }
        cusolverDnXgesvdp(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info, h_err_sigma)
    }
}
pub unsafe fn cusolverDnXgesvdp_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *const ::core::ffi::c_void, dataTypeU: cudaDataType, U: *const ::core::ffi::c_void, ldu: i64, dataTypeV: cudaDataType, V: *const ::core::ffi::c_void, ldv: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, ::core::ffi::c_int, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdp_bufferSize") });
        _f(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdp_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeS: cudaDataType, S: *const ::core::ffi::c_void, dataTypeU: cudaDataType, U: *const ::core::ffi::c_void, ldu: i64, dataTypeV: cudaDataType, V: *const ::core::ffi::c_void, ldv: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXgesvdp_bufferSize(handle, params, jobz, econ, m, n, dataTypeA, A, lda, dataTypeS, S, dataTypeU, U, ldu, dataTypeV, V, ldv, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXgesvdr(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobv: ::core::ffi::c_schar, m: i64, n: i64, k: i64, p: i64, niters: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeSrand: cudaDataType, Srand: *mut ::core::ffi::c_void, dataTypeUrand: cudaDataType, Urand: *mut ::core::ffi::c_void, ldUrand: i64, dataTypeVrand: cudaDataType, Vrand: *mut ::core::ffi::c_void, ldVrand: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, ::core::ffi::c_schar, ::core::ffi::c_schar, i64, i64, i64, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdr") });
        _f(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdr(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobv: ::core::ffi::c_schar, m: i64, n: i64, k: i64, p: i64, niters: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeSrand: cudaDataType, Srand: *mut ::core::ffi::c_void, dataTypeUrand: cudaDataType, Urand: *mut ::core::ffi::c_void, ldUrand: i64, dataTypeVrand: cudaDataType, Vrand: *mut ::core::ffi::c_void, ldVrand: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgesvdr(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    }
}
pub unsafe fn cusolverDnXgesvdr_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobv: ::core::ffi::c_schar, m: i64, n: i64, k: i64, p: i64, niters: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeSrand: cudaDataType, Srand: *const ::core::ffi::c_void, dataTypeUrand: cudaDataType, Urand: *const ::core::ffi::c_void, ldUrand: i64, dataTypeVrand: cudaDataType, Vrand: *const ::core::ffi::c_void, ldVrand: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, ::core::ffi::c_schar, ::core::ffi::c_schar, i64, i64, i64, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgesvdr_bufferSize") });
        _f(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgesvdr_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobu: ::core::ffi::c_schar, jobv: ::core::ffi::c_schar, m: i64, n: i64, k: i64, p: i64, niters: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeSrand: cudaDataType, Srand: *const ::core::ffi::c_void, dataTypeUrand: cudaDataType, Urand: *const ::core::ffi::c_void, ldUrand: i64, dataTypeVrand: cudaDataType, Vrand: *const ::core::ffi::c_void, ldVrand: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXgesvdr_bufferSize(handle, params, jobu, jobv, m, n, k, p, niters, dataTypeA, A, lda, dataTypeSrand, Srand, dataTypeUrand, Urand, ldUrand, dataTypeVrand, Vrand, ldVrand, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXgetrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, ipiv: *mut i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgetrf") });
        _f(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgetrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, ipiv: *mut i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgetrf(handle, params, m, n, dataTypeA, A, lda, ipiv, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXgetrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgetrf_bufferSize") });
        _f(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgetrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, m: i64, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXgetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXgetrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, trans: cublasOperation_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasOperation_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, *const i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXgetrs") });
        _f(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXgetrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, trans: cublasOperation_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXgetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, info)
    }
}
#[cfg(any(feature = "cuda-12040"))]
pub unsafe fn cusolverDnXlarft(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, N: i64, K: i64, dataTypeV: cudaDataType, d_V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, d_tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, d_T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXlarft") });
        _f(handle, params, direct, storev, N, K, dataTypeV, d_V, ldv, dataTypeTau, d_tau, dataTypeT, d_T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXlarft(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, N: i64, K: i64, dataTypeV: cudaDataType, d_V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, d_tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, d_T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize) -> cusolverStatus_t;
        }
        cusolverDnXlarft(handle, params, direct, storev, N, K, dataTypeV, d_V, ldv, dataTypeTau, d_tau, dataTypeT, d_T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXlarft(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, n: i64, k: i64, dataTypeV: cudaDataType, V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXlarft") });
        _f(handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXlarft(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, n: i64, k: i64, dataTypeV: cudaDataType, V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize) -> cusolverStatus_t;
        }
        cusolverDnXlarft(handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost)
    }
}
#[cfg(any(feature = "cuda-12040"))]
pub unsafe fn cusolverDnXlarft_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, N: i64, K: i64, dataTypeV: cudaDataType, d_V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, d_tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, d_T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXlarft_bufferSize") });
        _f(handle, params, direct, storev, N, K, dataTypeV, d_V, ldv, dataTypeTau, d_tau, dataTypeT, d_T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXlarft_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, N: i64, K: i64, dataTypeV: cudaDataType, d_V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, d_tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, d_T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXlarft_bufferSize(handle, params, direct, storev, N, K, dataTypeV, d_V, ldv, dataTypeTau, d_tau, dataTypeT, d_T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXlarft_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, n: i64, k: i64, dataTypeV: cudaDataType, V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverDirectMode_t, cusolverStorevMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXlarft_bufferSize") });
        _f(handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXlarft_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, direct: cusolverDirectMode_t, storev: cusolverStorevMode_t, n: i64, k: i64, dataTypeV: cudaDataType, V: *const ::core::ffi::c_void, ldv: i64, dataTypeTau: cudaDataType, tau: *const ::core::ffi::c_void, dataTypeT: cudaDataType, T: *mut ::core::ffi::c_void, ldt: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXlarft_bufferSize(handle, params, direct, storev, n, k, dataTypeV, V, ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXpolar(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, M: i64, N: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeH: cudaDataType, H: *mut ::core::ffi::c_void, ldh: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_res_nrm: *mut f64, d_A_nrmF: *mut f64, d_rcond: *mut f64, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut f64, *mut f64, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXpolar") });
        _f(handle, params, uplo, M, N, dataTypeA, A, lda, dataTypeH, H, ldh, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_res_nrm, d_A_nrmF, d_rcond, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXpolar(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, M: i64, N: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeH: cudaDataType, H: *mut ::core::ffi::c_void, ldh: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_res_nrm: *mut f64, d_A_nrmF: *mut f64, d_rcond: *mut f64, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXpolar(handle, params, uplo, M, N, dataTypeA, A, lda, dataTypeH, H, ldh, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_res_nrm, d_A_nrmF, d_rcond, d_info)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXpolar_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, M: i64, N: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeH: cudaDataType, H: *const ::core::ffi::c_void, ldh: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXpolar_bufferSize") });
        _f(handle, params, uplo, M, N, dataTypeA, A, lda, dataTypeH, H, ldh, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXpolar_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, M: i64, N: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeH: cudaDataType, H: *const ::core::ffi::c_void, ldh: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXpolar_bufferSize(handle, params, uplo, M, N, dataTypeA, A, lda, dataTypeH, H, ldh, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXpotrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXpotrf") });
        _f(handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXpotrf(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXpotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXpotrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXpotrf_bufferSize") });
        _f(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXpotrf_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXpotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXpotrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cublasFillMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXpotrs") });
        _f(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXpotrs(handle: cusolverDnHandle_t, params: cusolverDnParams_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, ldb, info)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXstedc(handle: cusolverDnHandle_t, params: cusolverDnParams_t, compz: cusolverEigComp_t, n: i64, dataTypeDE: cudaDataType, D: *mut ::core::ffi::c_void, E: *mut ::core::ffi::c_void, dataTypeZ: cudaDataType, Z: *mut ::core::ffi::c_void, ldz: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigComp_t, i64, cudaDataType, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXstedc") });
        _f(handle, params, compz, n, dataTypeDE, D, E, dataTypeZ, Z, ldz, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXstedc(handle: cusolverDnHandle_t, params: cusolverDnParams_t, compz: cusolverEigComp_t, n: i64, dataTypeDE: cudaDataType, D: *mut ::core::ffi::c_void, E: *mut ::core::ffi::c_void, dataTypeZ: cudaDataType, Z: *mut ::core::ffi::c_void, ldz: i64, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXstedc(handle, params, compz, n, dataTypeDE, D, E, dataTypeZ, Z, ldz, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXstedc_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, compz: cusolverEigComp_t, n: i64, dataTypeDE: cudaDataType, D: *const ::core::ffi::c_void, E: *const ::core::ffi::c_void, dataTypeZ: cudaDataType, Z: *const ::core::ffi::c_void, ldz: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigComp_t, i64, cudaDataType, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXstedc_bufferSize") });
        _f(handle, params, compz, n, dataTypeDE, D, E, dataTypeZ, Z, ldz, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXstedc_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, compz: cusolverEigComp_t, n: i64, dataTypeDE: cudaDataType, D: *const ::core::ffi::c_void, E: *const ::core::ffi::c_void, dataTypeZ: cudaDataType, Z: *const ::core::ffi::c_void, ldz: i64, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXstedc_bufferSize(handle, params, compz, n, dataTypeDE, D, E, dataTypeZ, Z, ldz, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXsyevBatched(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int, batchSize: i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int, i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevBatched") });
        _f(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevBatched(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int, batchSize: i64) -> cusolverStatus_t;
        }
        cusolverDnXsyevBatched(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info, batchSize)
    }
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXsyevBatched_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize, batchSize: i64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize, *mut usize, i64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevBatched_bufferSize") });
        _f(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevBatched_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize, batchSize: i64) -> cusolverStatus_t;
        }
        cusolverDnXsyevBatched_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost, batchSize)
    }
}
pub unsafe fn cusolverDnXsyevd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevd") });
        _f(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXsyevd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevd_bufferSize") });
        _f(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXsyevdx(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig64: *mut i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, i64, i64, *mut i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevdx") });
        _f(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevdx(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig64: *mut i64, dataTypeW: cudaDataType, W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsyevdx(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, meig64, dataTypeW, W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXsyevdx_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, h_meig: *mut i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, i64, i64, *mut i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevdx_bufferSize") });
        _f(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevdx_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, h_meig: *mut i64, dataTypeW: cudaDataType, W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXsyevdx_bufferSize(handle, params, jobz, range, uplo, n, dataTypeA, A, lda, vl, vu, il, iu, h_meig, dataTypeW, W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXsyevjGetResidual(handle: cusolverDnHandle_t, info: syevjInfo_t, residual: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, syevjInfo_t, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevjGetResidual") });
        _f(handle, info, residual)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevjGetResidual(handle: cusolverDnHandle_t, info: syevjInfo_t, residual: *mut f64) -> cusolverStatus_t;
        }
        cusolverDnXsyevjGetResidual(handle, info, residual)
    }
}
pub unsafe fn cusolverDnXsyevjGetSweeps(handle: cusolverDnHandle_t, info: syevjInfo_t, executed_sweeps: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, syevjInfo_t, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevjGetSweeps") });
        _f(handle, info, executed_sweeps)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevjGetSweeps(handle: cusolverDnHandle_t, info: syevjInfo_t, executed_sweeps: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsyevjGetSweeps(handle, info, executed_sweeps)
    }
}
pub unsafe fn cusolverDnXsyevjSetMaxSweeps(info: syevjInfo_t, max_sweeps: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevjSetMaxSweeps") });
        _f(info, max_sweeps)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevjSetMaxSweeps(info: syevjInfo_t, max_sweeps: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
    }
}
pub unsafe fn cusolverDnXsyevjSetSortEig(info: syevjInfo_t, sort_eig: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevjSetSortEig") });
        _f(info, sort_eig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevjSetSortEig(info: syevjInfo_t, sort_eig: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsyevjSetSortEig(info, sort_eig)
    }
}
pub unsafe fn cusolverDnXsyevjSetTolerance(info: syevjInfo_t, tolerance: f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(syevjInfo_t, f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsyevjSetTolerance") });
        _f(info, tolerance)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsyevjSetTolerance(info: syevjInfo_t, tolerance: f64) -> cusolverStatus_t;
        }
        cusolverDnXsyevjSetTolerance(info, tolerance)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXsygvd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *mut ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *mut ::core::ffi::c_void, ldb: i64, dataTypeW: cudaDataType, d_W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsygvd") });
        _f(handle, params, itype, jobz, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, dataTypeW, d_W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsygvd(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *mut ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *mut ::core::ffi::c_void, ldb: i64, dataTypeW: cudaDataType, d_W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsygvd(handle, params, itype, jobz, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, dataTypeW, d_W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXsygvd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *const ::core::ffi::c_void, ldb: i64, dataTypeW: cudaDataType, d_W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsygvd_bufferSize") });
        _f(handle, params, itype, jobz, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, dataTypeW, d_W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsygvd_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *const ::core::ffi::c_void, ldb: i64, dataTypeW: cudaDataType, d_W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXsygvd_bufferSize(handle, params, itype, jobz, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, dataTypeW, d_W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXsygvdx(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *mut ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *mut ::core::ffi::c_void, ldb: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig: *mut i64, dataTypeW: cudaDataType, d_W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, i64, i64, *mut i64, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsygvdx") });
        _f(handle, params, itype, jobz, range, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, vl, vu, il, iu, meig, dataTypeW, d_W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsygvdx(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *mut ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *mut ::core::ffi::c_void, ldb: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig: *mut i64, dataTypeW: cudaDataType, d_W: *mut ::core::ffi::c_void, computeType: cudaDataType, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, d_info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsygvdx(handle, params, itype, jobz, range, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, vl, vu, il, iu, meig, dataTypeW, d_W, computeType, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, d_info)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusolverDnXsygvdx_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *const ::core::ffi::c_void, ldb: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig: *mut i64, dataTypeW: cudaDataType, d_W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, i64, cudaDataType, *const ::core::ffi::c_void, i64, cudaDataType, *const ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, i64, i64, *mut i64, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsygvdx_bufferSize") });
        _f(handle, params, itype, jobz, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, vl, vu, il, iu, meig, dataTypeW, d_W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsygvdx_bufferSize(handle: cusolverDnHandle_t, params: cusolverDnParams_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: i64, dataTypeA: cudaDataType, d_A: *const ::core::ffi::c_void, lda: i64, dataTypeB: cudaDataType, d_B: *const ::core::ffi::c_void, ldb: i64, vl: *mut ::core::ffi::c_void, vu: *mut ::core::ffi::c_void, il: i64, iu: i64, meig: *mut i64, dataTypeW: cudaDataType, d_W: *const ::core::ffi::c_void, computeType: cudaDataType, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXsygvdx_bufferSize(handle, params, itype, jobz, uplo, n, dataTypeA, d_A, lda, dataTypeB, d_B, ldb, vl, vu, il, iu, meig, dataTypeW, d_W, computeType, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXsytrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, *const i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsytrs") });
        _f(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsytrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXsytrs(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info)
    }
}
pub unsafe fn cusolverDnXsytrs_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, i64, i64, cudaDataType, *const ::core::ffi::c_void, i64, *const i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXsytrs_bufferSize") });
        _f(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXsytrs_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: i64, nrhs: i64, dataTypeA: cudaDataType, A: *const ::core::ffi::c_void, lda: i64, ipiv: *const i64, dataTypeB: cudaDataType, B: *mut ::core::ffi::c_void, ldb: i64, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXsytrs_bufferSize(handle, uplo, n, nrhs, dataTypeA, A, lda, ipiv, dataTypeB, B, ldb, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnXtrtri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, diag: cublasDiagType_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_void, usize, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXtrtri") });
        _f(handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXtrtri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, diag: cublasDiagType_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, bufferOnDevice: *mut ::core::ffi::c_void, workspaceInBytesOnDevice: usize, bufferOnHost: *mut ::core::ffi::c_void, workspaceInBytesOnHost: usize, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnXtrtri(handle, uplo, diag, n, dataTypeA, A, lda, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, devInfo)
    }
}
pub unsafe fn cusolverDnXtrtri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, diag: cublasDiagType_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, cublasDiagType_t, i64, cudaDataType, *mut ::core::ffi::c_void, i64, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnXtrtri_bufferSize") });
        _f(handle, uplo, diag, n, dataTypeA, A, lda, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnXtrtri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, diag: cublasDiagType_t, n: i64, dataTypeA: cudaDataType, A: *mut ::core::ffi::c_void, lda: i64, workspaceInBytesOnDevice: *mut usize, workspaceInBytesOnHost: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnXtrtri_bufferSize(handle, uplo, diag, n, dataTypeA, A, lda, workspaceInBytesOnDevice, workspaceInBytesOnHost)
    }
}
pub unsafe fn cusolverDnZCgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZCgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZCgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZCgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZCgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZCgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZCgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZCgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZCgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZCgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZCgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZCgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZCgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZCgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZCgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZCgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZEgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZEgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZEgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZEgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZEgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZEgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZEgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZEgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZEgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZEgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZEgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZEgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZEgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZEgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZEgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZEgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZKgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZKgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZKgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZKgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZKgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZKgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZKgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZKgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZKgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZKgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZKgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZKgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZKgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZKgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZKgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZKgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZYgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZYgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZYgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZYgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZYgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZYgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZYgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZYgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZYgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZYgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZYgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZYgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZYgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZYgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZYgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZYgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZZgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZZgels") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZZgels(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZZgels(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZZgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZZgels_bufferSize") });
        _f(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZZgels_bufferSize(handle: cusolverDnHandle_t, m: cusolver_int_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZZgels_bufferSize(handle, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZZgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, usize, *mut cusolver_int_t, *mut cusolver_int_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZZgesv") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZZgesv(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: usize, iter: *mut cusolver_int_t, d_info: *mut cusolver_int_t) -> cusolverStatus_t;
        }
        cusolverDnZZgesv(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes, iter, d_info)
    }
}
pub unsafe fn cusolverDnZZgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolver_int_t, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut cuDoubleComplex, cusolver_int_t, *mut ::core::ffi::c_void, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZZgesv_bufferSize") });
        _f(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZZgesv_bufferSize(handle: cusolverDnHandle_t, n: cusolver_int_t, nrhs: cusolver_int_t, dA: *mut cuDoubleComplex, ldda: cusolver_int_t, dipiv: *mut cusolver_int_t, dB: *mut cuDoubleComplex, lddb: cusolver_int_t, dX: *mut cuDoubleComplex, lddx: cusolver_int_t, dWorkspace: *mut ::core::ffi::c_void, lwork_bytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverDnZZgesv_bufferSize(handle, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace, lwork_bytes)
    }
}
pub unsafe fn cusolverDnZgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, D: *mut f64, E: *mut f64, TAUQ: *mut cuDoubleComplex, TAUP: *mut cuDoubleComplex, Work: *mut cuDoubleComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut f64, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgebrd") });
        _f(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgebrd(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, D: *mut f64, E: *mut f64, TAUQ: *mut cuDoubleComplex, TAUP: *mut cuDoubleComplex, Work: *mut cuDoubleComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnZgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgebrd_bufferSize") });
        _f(handle, m, n, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgebrd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgebrd_bufferSize(handle, m, n, Lwork)
    }
}
pub unsafe fn cusolverDnZgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, TAU: *mut cuDoubleComplex, Workspace: *mut cuDoubleComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgeqrf") });
        _f(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgeqrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, TAU: *mut cuDoubleComplex, Workspace: *mut cuDoubleComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnZgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgeqrf_bufferSize") });
        _f(handle, m, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgeqrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnZgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, S: *mut f64, U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, VT: *mut cuDoubleComplex, ldvt: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, rwork: *mut f64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_schar, ::core::ffi::c_schar, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvd") });
        _f(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvd(handle: cusolverDnHandle_t, jobu: ::core::ffi::c_schar, jobvt: ::core::ffi::c_schar, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, S: *mut f64, U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, VT: *mut cuDoubleComplex, ldvt: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, rwork: *mut f64, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, info)
    }
}
pub unsafe fn cusolverDnZgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvd_bufferSize") });
        _f(handle, m, n, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvd_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgesvd_bufferSize(handle, m, n, lwork)
    }
}
pub unsafe fn cusolverDnZgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f64, strideS: ::core::ffi::c_longlong, d_U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut cuDoubleComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut f64, ::core::ffi::c_longlong, *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvdaStridedBatched") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvdaStridedBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *mut f64, strideS: ::core::ffi::c_longlong, d_U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *mut cuDoubleComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, d_work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, h_R_nrmF: *mut f64, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info, h_R_nrmF, batchSize)
    }
}
pub unsafe fn cusolverDnZgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f64, strideS: ::core::ffi::c_longlong, d_U: *const cuDoubleComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const cuDoubleComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const f64, ::core::ffi::c_longlong, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_longlong, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvdaStridedBatched_bufferSize") });
        _f(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvdaStridedBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, rank: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, d_A: *const cuDoubleComplex, lda: ::core::ffi::c_int, strideA: ::core::ffi::c_longlong, d_S: *const f64, strideS: ::core::ffi::c_longlong, d_U: *const cuDoubleComplex, ldu: ::core::ffi::c_int, strideU: ::core::ffi::c_longlong, d_V: *const cuDoubleComplex, ldv: ::core::ffi::c_int, strideV: ::core::ffi::c_longlong, lwork: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu, strideU, d_V, ldv, strideV, lwork, batchSize)
    }
}
pub unsafe fn cusolverDnZgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, S: *mut f64, U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, V: *mut cuDoubleComplex, ldv: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvdj") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvdj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, S: *mut f64, U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, V: *mut cuDoubleComplex, ldv: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnZgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, S: *mut f64, U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, V: *mut cuDoubleComplex, ldv: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvdjBatched") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvdjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, S: *mut f64, U: *mut cuDoubleComplex, ldu: ::core::ffi::c_int, V: *mut cuDoubleComplex, ldv: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnZgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, S: *const f64, U: *const cuDoubleComplex, ldu: ::core::ffi::c_int, V: *const cuDoubleComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvdjBatched_bufferSize") });
        _f(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvdjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, S: *const f64, U: *const cuDoubleComplex, ldu: ::core::ffi::c_int, V: *const cuDoubleComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnZgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, S: *const f64, U: *const cuDoubleComplex, ldu: ::core::ffi::c_int, V: *const cuDoubleComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, gesvdjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgesvdj_bufferSize") });
        _f(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgesvdj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, econ: ::core::ffi::c_int, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, S: *const f64, U: *const cuDoubleComplex, ldu: ::core::ffi::c_int, V: *const cuDoubleComplex, ldv: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int, params: gesvdjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params)
    }
}
pub unsafe fn cusolverDnZgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Workspace: *mut cuDoubleComplex, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgetrf") });
        _f(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgetrf(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Workspace: *mut cuDoubleComplex, devIpiv: *mut ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
}
pub unsafe fn cusolverDnZgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgetrf_bufferSize") });
        _f(handle, m, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgetrf_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnZgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZgetrs") });
        _f(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZgetrs(handle: cusolverDnHandle_t, trans: cublasOperation_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnZheevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevd") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevd(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZheevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevd_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevd_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)
    }
}
pub unsafe fn cusolverDnZheevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevdx") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevdx(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZheevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevdx_bufferSize") });
        _f(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevdx_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnZheevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevj") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevj(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnZheevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevjBatched") });
        _f(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevjBatched(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, batchSize)
    }
}
pub unsafe fn cusolverDnZheevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, syevjInfo_t, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevjBatched_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevjBatched_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, batchSize)
    }
}
pub unsafe fn cusolverDnZheevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZheevj_bufferSize") });
        _f(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZheevj_bufferSize(handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params)
    }
}
pub unsafe fn cusolverDnZhegvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhegvd") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhegvd(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZhegvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhegvd_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhegvd_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork)
    }
}
pub unsafe fn cusolverDnZhegvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhegvdx") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhegvdx(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZhegvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, f64, f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhegvdx_bufferSize") });
        _f(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhegvdx_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, range: cusolverEigRange_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, vl: f64, vu: f64, il: ::core::ffi::c_int, iu: ::core::ffi::c_int, meig: *mut ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, lwork)
    }
}
pub unsafe fn cusolverDnZhegvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhegvj") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhegvj(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, W: *mut f64, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params)
    }
}
pub unsafe fn cusolverDnZhegvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, syevjInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhegvj_bufferSize") });
        _f(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhegvj_bufferSize(handle: cusolverDnHandle_t, itype: cusolverEigType_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, W: *const f64, lwork: *mut ::core::ffi::c_int, params: syevjInfo_t) -> cusolverStatus_t;
        }
        cusolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params)
    }
}
pub unsafe fn cusolverDnZhetrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, d: *mut f64, e: *mut f64, tau: *mut cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut f64, *mut f64, *mut cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhetrd") });
        _f(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhetrd(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, d: *mut f64, e: *mut f64, tau: *mut cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZhetrd(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZhetrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, d: *const f64, e: *const f64, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const f64, *const f64, *const cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZhetrd_bufferSize") });
        _f(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZhetrd_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, d: *const f64, e: *const f64, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
}
pub unsafe fn cusolverDnZlaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZlaswp") });
        _f(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZlaswp(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, k1: ::core::ffi::c_int, k2: ::core::ffi::c_int, devIpiv: *const ::core::ffi::c_int, incx: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZlaswp(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
}
pub unsafe fn cusolverDnZlauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZlauum") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZlauum(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZlauum(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnZlauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZlauum_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZlauum_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZlauum_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnZpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Workspace: *mut cuDoubleComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotrf") });
        _f(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Workspace: *mut cuDoubleComplex, Lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
}
pub unsafe fn cusolverDnZpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut cuDoubleComplex, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotrfBatched") });
        _f(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotrfBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, Aarray: *mut *mut cuDoubleComplex, lda: ::core::ffi::c_int, infoArray: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
}
pub unsafe fn cusolverDnZpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotrf_bufferSize") });
        _f(handle, uplo, n, A, lda, Lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotrf_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, Lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, Lwork)
    }
}
pub unsafe fn cusolverDnZpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotri") });
        _f(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnZpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotri_bufferSize") });
        _f(handle, uplo, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnZpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotrs") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotrs(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
}
pub unsafe fn cusolverDnZpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut *mut cuDoubleComplex, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut *mut cuDoubleComplex, ::core::ffi::c_int, *mut *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZpotrsBatched") });
        _f(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZpotrsBatched(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, A: *mut *mut cuDoubleComplex, lda: ::core::ffi::c_int, B: *mut *mut cuDoubleComplex, ldb: ::core::ffi::c_int, d_info: *mut ::core::ffi::c_int, batchSize: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize)
    }
}
pub unsafe fn cusolverDnZsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZsytrf") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZsytrf(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, ipiv: *mut ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZsytrf_bufferSize") });
        _f(handle, n, A, lda, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZsytrf_bufferSize(handle: cusolverDnHandle_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZsytrf_bufferSize(handle, n, A, lda, lwork)
    }
}
pub unsafe fn cusolverDnZsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZsytri") });
        _f(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZsytri(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZsytri(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZsytri_bufferSize") });
        _f(handle, uplo, n, A, lda, ipiv, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZsytri_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, ipiv: *const ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZsytri_bufferSize(handle, uplo, n, A, lda, ipiv, lwork)
    }
}
pub unsafe fn cusolverDnZungbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZungbr") });
        _f(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZungbr(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZungbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZungbr_bufferSize") });
        _f(handle, side, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZungbr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnZungqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZungqr") });
        _f(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZungqr(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZungqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZungqr_bufferSize") });
        _f(handle, m, n, k, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZungqr_bufferSize(handle: cusolverDnHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnZungtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZungtr") });
        _f(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZungtr(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZungtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasFillMode_t, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZungtr_bufferSize") });
        _f(handle, uplo, n, A, lda, tau, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZungtr_bufferSize(handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork)
    }
}
pub unsafe fn cusolverDnZunmqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZunmqr") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZunmqr(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, devInfo: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo)
    }
}
pub unsafe fn cusolverDnZunmqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, C: *const cuDoubleComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZunmqr_bufferSize") });
        _f(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZunmqr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, C: *const cuDoubleComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverDnZunmtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *mut cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZunmtr") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZunmtr(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int, tau: *mut cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int, work: *mut cuDoubleComplex, lwork: ::core::ffi::c_int, info: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info)
    }
}
pub unsafe fn cusolverDnZunmtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, C: *const cuDoubleComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverDnHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverDnZunmtr_bufferSize") });
        _f(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverDnZunmtr_bufferSize(handle: cusolverDnHandle_t, side: cublasSideMode_t, uplo: cublasFillMode_t, trans: cublasOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, tau: *const cuDoubleComplex, C: *const cuDoubleComplex, ldc: ::core::ffi::c_int, lwork: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork)
    }
}
pub unsafe fn cusolverGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(libraryPropertyType, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverGetProperty") });
        _f(type_, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverGetProperty(type_, value)
    }
}
pub unsafe fn cusolverGetVersion(version: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverGetVersion") });
        _f(version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverGetVersion(version: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverGetVersion(version)
    }
}
pub unsafe fn cusolverRfAccessBundledFactorsDevice(handle: cusolverRfHandle_t, nnzM: *mut ::core::ffi::c_int, Mp: *mut *mut ::core::ffi::c_int, Mi: *mut *mut ::core::ffi::c_int, Mx: *mut *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfAccessBundledFactorsDevice") });
        _f(handle, nnzM, Mp, Mi, Mx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfAccessBundledFactorsDevice(handle: cusolverRfHandle_t, nnzM: *mut ::core::ffi::c_int, Mp: *mut *mut ::core::ffi::c_int, Mi: *mut *mut ::core::ffi::c_int, Mx: *mut *mut f64) -> cusolverStatus_t;
        }
        cusolverRfAccessBundledFactorsDevice(handle, nnzM, Mp, Mi, Mx)
    }
}
pub unsafe fn cusolverRfAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfAnalyze") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfAnalyze(handle)
    }
}
pub unsafe fn cusolverRfBatchAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfBatchAnalyze") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfBatchAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfBatchAnalyze(handle)
    }
}
pub unsafe fn cusolverRfBatchRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfBatchRefactor") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfBatchRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfBatchRefactor(handle)
    }
}
pub unsafe fn cusolverRfBatchResetValues(batchSize: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, csrValA_array: *mut *mut f64, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfBatchResetValues") });
        _f(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfBatchResetValues(batchSize: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, csrValA_array: *mut *mut f64, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle)
    }
}
pub unsafe fn cusolverRfBatchSetupHost(batchSize: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, h_csrRowPtrA: *mut ::core::ffi::c_int, h_csrColIndA: *mut ::core::ffi::c_int, h_csrValA_array: *mut *mut f64, nnzL: ::core::ffi::c_int, h_csrRowPtrL: *mut ::core::ffi::c_int, h_csrColIndL: *mut ::core::ffi::c_int, h_csrValL: *mut f64, nnzU: ::core::ffi::c_int, h_csrRowPtrU: *mut ::core::ffi::c_int, h_csrColIndU: *mut ::core::ffi::c_int, h_csrValU: *mut f64, h_P: *mut ::core::ffi::c_int, h_Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfBatchSetupHost") });
        _f(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfBatchSetupHost(batchSize: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, h_csrRowPtrA: *mut ::core::ffi::c_int, h_csrColIndA: *mut ::core::ffi::c_int, h_csrValA_array: *mut *mut f64, nnzL: ::core::ffi::c_int, h_csrRowPtrL: *mut ::core::ffi::c_int, h_csrColIndL: *mut ::core::ffi::c_int, h_csrValL: *mut f64, nnzU: ::core::ffi::c_int, h_csrRowPtrU: *mut ::core::ffi::c_int, h_csrColIndU: *mut ::core::ffi::c_int, h_csrValU: *mut f64, h_P: *mut ::core::ffi::c_int, h_Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    }
}
pub unsafe fn cusolverRfBatchSolve(handle: cusolverRfHandle_t, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Temp: *mut f64, ldt: ::core::ffi::c_int, XF_array: *mut *mut f64, ldxf: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut *mut f64, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfBatchSolve") });
        _f(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfBatchSolve(handle: cusolverRfHandle_t, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Temp: *mut f64, ldt: ::core::ffi::c_int, XF_array: *mut *mut f64, ldxf: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverRfBatchSolve(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
    }
}
pub unsafe fn cusolverRfBatchZeroPivot(handle: cusolverRfHandle_t, position: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfBatchZeroPivot") });
        _f(handle, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfBatchZeroPivot(handle: cusolverRfHandle_t, position: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverRfBatchZeroPivot(handle, position)
    }
}
pub unsafe fn cusolverRfCreate(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfCreate(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfCreate(handle)
    }
}
pub unsafe fn cusolverRfDestroy(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfDestroy") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfDestroy(handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfDestroy(handle)
    }
}
pub unsafe fn cusolverRfExtractBundledFactorsHost(handle: cusolverRfHandle_t, h_nnzM: *mut ::core::ffi::c_int, h_Mp: *mut *mut ::core::ffi::c_int, h_Mi: *mut *mut ::core::ffi::c_int, h_Mx: *mut *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfExtractBundledFactorsHost") });
        _f(handle, h_nnzM, h_Mp, h_Mi, h_Mx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfExtractBundledFactorsHost(handle: cusolverRfHandle_t, h_nnzM: *mut ::core::ffi::c_int, h_Mp: *mut *mut ::core::ffi::c_int, h_Mi: *mut *mut ::core::ffi::c_int, h_Mx: *mut *mut f64) -> cusolverStatus_t;
        }
        cusolverRfExtractBundledFactorsHost(handle, h_nnzM, h_Mp, h_Mi, h_Mx)
    }
}
pub unsafe fn cusolverRfExtractSplitFactorsHost(handle: cusolverRfHandle_t, h_nnzL: *mut ::core::ffi::c_int, h_csrRowPtrL: *mut *mut ::core::ffi::c_int, h_csrColIndL: *mut *mut ::core::ffi::c_int, h_csrValL: *mut *mut f64, h_nnzU: *mut ::core::ffi::c_int, h_csrRowPtrU: *mut *mut ::core::ffi::c_int, h_csrColIndU: *mut *mut ::core::ffi::c_int, h_csrValU: *mut *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut f64, *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut ::core::ffi::c_int, *mut *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfExtractSplitFactorsHost") });
        _f(handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfExtractSplitFactorsHost(handle: cusolverRfHandle_t, h_nnzL: *mut ::core::ffi::c_int, h_csrRowPtrL: *mut *mut ::core::ffi::c_int, h_csrColIndL: *mut *mut ::core::ffi::c_int, h_csrValL: *mut *mut f64, h_nnzU: *mut ::core::ffi::c_int, h_csrRowPtrU: *mut *mut ::core::ffi::c_int, h_csrColIndU: *mut *mut ::core::ffi::c_int, h_csrValU: *mut *mut f64) -> cusolverStatus_t;
        }
        cusolverRfExtractSplitFactorsHost(handle, h_nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, h_nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU)
    }
}
pub unsafe fn cusolverRfGetAlgs(handle: cusolverRfHandle_t, factAlg: *mut cusolverRfFactorization_t, solveAlg: *mut cusolverRfTriangularSolve_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut cusolverRfFactorization_t, *mut cusolverRfTriangularSolve_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfGetAlgs") });
        _f(handle, factAlg, solveAlg)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfGetAlgs(handle: cusolverRfHandle_t, factAlg: *mut cusolverRfFactorization_t, solveAlg: *mut cusolverRfTriangularSolve_t) -> cusolverStatus_t;
        }
        cusolverRfGetAlgs(handle, factAlg, solveAlg)
    }
}
pub unsafe fn cusolverRfGetMatrixFormat(handle: cusolverRfHandle_t, format: *mut cusolverRfMatrixFormat_t, diag: *mut cusolverRfUnitDiagonal_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut cusolverRfMatrixFormat_t, *mut cusolverRfUnitDiagonal_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfGetMatrixFormat") });
        _f(handle, format, diag)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfGetMatrixFormat(handle: cusolverRfHandle_t, format: *mut cusolverRfMatrixFormat_t, diag: *mut cusolverRfUnitDiagonal_t) -> cusolverStatus_t;
        }
        cusolverRfGetMatrixFormat(handle, format, diag)
    }
}
pub unsafe fn cusolverRfGetNumericBoostReport(handle: cusolverRfHandle_t, report: *mut cusolverRfNumericBoostReport_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut cusolverRfNumericBoostReport_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfGetNumericBoostReport") });
        _f(handle, report)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfGetNumericBoostReport(handle: cusolverRfHandle_t, report: *mut cusolverRfNumericBoostReport_t) -> cusolverStatus_t;
        }
        cusolverRfGetNumericBoostReport(handle, report)
    }
}
pub unsafe fn cusolverRfGetNumericProperties(handle: cusolverRfHandle_t, zero: *mut f64, boost: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut f64, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfGetNumericProperties") });
        _f(handle, zero, boost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfGetNumericProperties(handle: cusolverRfHandle_t, zero: *mut f64, boost: *mut f64) -> cusolverStatus_t;
        }
        cusolverRfGetNumericProperties(handle, zero, boost)
    }
}
pub unsafe fn cusolverRfGetResetValuesFastMode(handle: cusolverRfHandle_t, fastMode: *mut cusolverRfResetValuesFastMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut cusolverRfResetValuesFastMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfGetResetValuesFastMode") });
        _f(handle, fastMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfGetResetValuesFastMode(handle: cusolverRfHandle_t, fastMode: *mut cusolverRfResetValuesFastMode_t) -> cusolverStatus_t;
        }
        cusolverRfGetResetValuesFastMode(handle, fastMode)
    }
}
pub unsafe fn cusolverRfRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfRefactor") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfRefactor(handle)
    }
}
pub unsafe fn cusolverRfResetValues(n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, csrValA: *mut f64, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfResetValues") });
        _f(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfResetValues(n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, csrValA: *mut f64, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle)
    }
}
pub unsafe fn cusolverRfSetAlgs(handle: cusolverRfHandle_t, factAlg: cusolverRfFactorization_t, solveAlg: cusolverRfTriangularSolve_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, cusolverRfFactorization_t, cusolverRfTriangularSolve_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSetAlgs") });
        _f(handle, factAlg, solveAlg)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSetAlgs(handle: cusolverRfHandle_t, factAlg: cusolverRfFactorization_t, solveAlg: cusolverRfTriangularSolve_t) -> cusolverStatus_t;
        }
        cusolverRfSetAlgs(handle, factAlg, solveAlg)
    }
}
pub unsafe fn cusolverRfSetMatrixFormat(handle: cusolverRfHandle_t, format: cusolverRfMatrixFormat_t, diag: cusolverRfUnitDiagonal_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, cusolverRfMatrixFormat_t, cusolverRfUnitDiagonal_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSetMatrixFormat") });
        _f(handle, format, diag)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSetMatrixFormat(handle: cusolverRfHandle_t, format: cusolverRfMatrixFormat_t, diag: cusolverRfUnitDiagonal_t) -> cusolverStatus_t;
        }
        cusolverRfSetMatrixFormat(handle, format, diag)
    }
}
pub unsafe fn cusolverRfSetNumericProperties(handle: cusolverRfHandle_t, zero: f64, boost: f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, f64, f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSetNumericProperties") });
        _f(handle, zero, boost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSetNumericProperties(handle: cusolverRfHandle_t, zero: f64, boost: f64) -> cusolverStatus_t;
        }
        cusolverRfSetNumericProperties(handle, zero, boost)
    }
}
pub unsafe fn cusolverRfSetResetValuesFastMode(handle: cusolverRfHandle_t, fastMode: cusolverRfResetValuesFastMode_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, cusolverRfResetValuesFastMode_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSetResetValuesFastMode") });
        _f(handle, fastMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSetResetValuesFastMode(handle: cusolverRfHandle_t, fastMode: cusolverRfResetValuesFastMode_t) -> cusolverStatus_t;
        }
        cusolverRfSetResetValuesFastMode(handle, fastMode)
    }
}
pub unsafe fn cusolverRfSetupDevice(n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, csrValA: *mut f64, nnzL: ::core::ffi::c_int, csrRowPtrL: *mut ::core::ffi::c_int, csrColIndL: *mut ::core::ffi::c_int, csrValL: *mut f64, nnzU: ::core::ffi::c_int, csrRowPtrU: *mut ::core::ffi::c_int, csrColIndU: *mut ::core::ffi::c_int, csrValU: *mut f64, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSetupDevice") });
        _f(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSetupDevice(n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, csrValA: *mut f64, nnzL: ::core::ffi::c_int, csrRowPtrL: *mut ::core::ffi::c_int, csrColIndL: *mut ::core::ffi::c_int, csrValL: *mut f64, nnzU: ::core::ffi::c_int, csrRowPtrU: *mut ::core::ffi::c_int, csrColIndU: *mut ::core::ffi::c_int, csrValU: *mut f64, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, Q, handle)
    }
}
pub unsafe fn cusolverRfSetupHost(n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, h_csrRowPtrA: *mut ::core::ffi::c_int, h_csrColIndA: *mut ::core::ffi::c_int, h_csrValA: *mut f64, nnzL: ::core::ffi::c_int, h_csrRowPtrL: *mut ::core::ffi::c_int, h_csrColIndL: *mut ::core::ffi::c_int, h_csrValL: *mut f64, nnzU: ::core::ffi::c_int, h_csrRowPtrU: *mut ::core::ffi::c_int, h_csrColIndU: *mut ::core::ffi::c_int, h_csrValU: *mut f64, h_P: *mut ::core::ffi::c_int, h_Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusolverRfHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSetupHost") });
        _f(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSetupHost(n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, h_csrRowPtrA: *mut ::core::ffi::c_int, h_csrColIndA: *mut ::core::ffi::c_int, h_csrValA: *mut f64, nnzL: ::core::ffi::c_int, h_csrRowPtrL: *mut ::core::ffi::c_int, h_csrColIndL: *mut ::core::ffi::c_int, h_csrValL: *mut f64, nnzU: ::core::ffi::c_int, h_csrRowPtrU: *mut ::core::ffi::c_int, h_csrColIndU: *mut ::core::ffi::c_int, h_csrValU: *mut f64, h_P: *mut ::core::ffi::c_int, h_Q: *mut ::core::ffi::c_int, handle: cusolverRfHandle_t) -> cusolverStatus_t;
        }
        cusolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle)
    }
}
pub unsafe fn cusolverRfSolve(handle: cusolverRfHandle_t, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Temp: *mut f64, ldt: ::core::ffi::c_int, XF: *mut f64, ldxf: ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverRfHandle_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverRfSolve") });
        _f(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverRfSolve(handle: cusolverRfHandle_t, P: *mut ::core::ffi::c_int, Q: *mut ::core::ffi::c_int, nrhs: ::core::ffi::c_int, Temp: *mut f64, ldt: ::core::ffi::c_int, XF: *mut f64, ldxf: ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
    }
}
pub unsafe fn cusolverSpCcsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuComplex, right_upper_corner: cuComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuComplex, cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsreigsHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuComplex, right_upper_corner: cuComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
}
pub unsafe fn cusolverSpCcsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuComplex, x0: *const cuComplex, maxite: ::core::ffi::c_int, eps: f32, mu: *mut cuComplex, x: *mut cuComplex) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuComplex, *const cuComplex, ::core::ffi::c_int, f32, *mut cuComplex, *mut cuComplex) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsreigvsi") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuComplex, x0: *const cuComplex, maxite: ::core::ffi::c_int, eps: f32, mu: *mut cuComplex, x: *mut cuComplex) -> cusolverStatus_t;
        }
        cusolverSpCcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
}
pub unsafe fn cusolverSpCcsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuComplex, x0: *const cuComplex, maxite: ::core::ffi::c_int, tol: f32, mu: *mut cuComplex, x: *mut cuComplex) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuComplex, *const cuComplex, ::core::ffi::c_int, f32, *mut cuComplex, *mut cuComplex) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsreigvsiHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuComplex, x0: *const cuComplex, maxite: ::core::ffi::c_int, tol: f32, mu: *mut cuComplex, x: *mut cuComplex) -> cusolverStatus_t;
        }
        cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
}
pub unsafe fn cusolverSpCcsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, rankA: *mut ::core::ffi::c_int, x: *mut cuComplex, p: *mut ::core::ffi::c_int, min_norm: *mut f32) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, f32, *mut ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int, *mut f32) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrlsqvqrHost") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, rankA: *mut ::core::ffi::c_int, x: *mut cuComplex, p: *mut ::core::ffi::c_int, min_norm: *mut f32) -> cusolverStatus_t;
        }
        cusolverSpCcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
}
pub unsafe fn cusolverSpCcsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, f32, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrlsvchol") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpCcsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, f32, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrlsvcholHost") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpCcsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, f32, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrlsvluHost") });
        _f(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpCcsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, f32, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrlsvqr") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpCcsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, f32, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrlsvqrHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, tol: f32, reorder: ::core::ffi::c_int, x: *mut cuComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpCcsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, csrqrInfo_t, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrqrBufferInfoBatched") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverSpCcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
}
pub unsafe fn cusolverSpCcsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, x: *mut cuComplex, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, csrqrInfo_t, *mut ::core::ffi::c_void) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrqrsvBatched") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuComplex, x: *mut cuComplex, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t;
        }
        cusolverSpCcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
}
pub unsafe fn cusolverSpCcsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCcsrzfdHost") });
        _f(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCcsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpCcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
}
pub unsafe fn cusolverSpCreate(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusolverSpHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCreate(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t;
        }
        cusolverSpCreate(handle)
    }
}
pub unsafe fn cusolverSpCreateCsrqrInfo(info: *mut csrqrInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csrqrInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpCreateCsrqrInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpCreateCsrqrInfo(info: *mut csrqrInfo_t) -> cusolverStatus_t;
        }
        cusolverSpCreateCsrqrInfo(info)
    }
}
pub unsafe fn cusolverSpDcsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuDoubleComplex, right_upper_corner: cuDoubleComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuDoubleComplex, cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsreigsHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuDoubleComplex, right_upper_corner: cuDoubleComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
}
pub unsafe fn cusolverSpDcsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f64, x0: *const f64, maxite: ::core::ffi::c_int, eps: f64, mu: *mut f64, x: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f64, *const f64, ::core::ffi::c_int, f64, *mut f64, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsreigvsi") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f64, x0: *const f64, maxite: ::core::ffi::c_int, eps: f64, mu: *mut f64, x: *mut f64) -> cusolverStatus_t;
        }
        cusolverSpDcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
}
pub unsafe fn cusolverSpDcsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f64, x0: *const f64, maxite: ::core::ffi::c_int, tol: f64, mu: *mut f64, x: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f64, *const f64, ::core::ffi::c_int, f64, *mut f64, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsreigvsiHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f64, x0: *const f64, maxite: ::core::ffi::c_int, tol: f64, mu: *mut f64, x: *mut f64) -> cusolverStatus_t;
        }
        cusolverSpDcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
}
pub unsafe fn cusolverSpDcsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, tol: f64, rankA: *mut ::core::ffi::c_int, x: *mut f64, p: *mut ::core::ffi::c_int, min_norm: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, f64, *mut ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrlsqvqrHost") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, tol: f64, rankA: *mut ::core::ffi::c_int, x: *mut f64, p: *mut ::core::ffi::c_int, min_norm: *mut f64) -> cusolverStatus_t;
        }
        cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
}
pub unsafe fn cusolverSpDcsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrlsvchol") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpDcsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrlsvcholHost") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpDcsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrlsvluHost") });
        _f(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpDcsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrlsvqr") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpDcsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, f64, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrlsvqrHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, tol: f64, reorder: ::core::ffi::c_int, x: *mut f64, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpDcsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, csrqrInfo_t, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrqrBufferInfoBatched") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverSpDcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
}
pub unsafe fn cusolverSpDcsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, x: *mut f64, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int, csrqrInfo_t, *mut ::core::ffi::c_void) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrqrsvBatched") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f64, x: *mut f64, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t;
        }
        cusolverSpDcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
}
pub unsafe fn cusolverSpDcsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDcsrzfdHost") });
        _f(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDcsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f64, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpDcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
}
pub unsafe fn cusolverSpDestroy(handle: cusolverSpHandle_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDestroy") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDestroy(handle: cusolverSpHandle_t) -> cusolverStatus_t;
        }
        cusolverSpDestroy(handle)
    }
}
pub unsafe fn cusolverSpDestroyCsrqrInfo(info: csrqrInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csrqrInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpDestroyCsrqrInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpDestroyCsrqrInfo(info: csrqrInfo_t) -> cusolverStatus_t;
        }
        cusolverSpDestroyCsrqrInfo(info)
    }
}
pub unsafe fn cusolverSpGetStream(handle: cusolverSpHandle_t, streamId: *mut cudaStream_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, *mut cudaStream_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpGetStream") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpGetStream(handle: cusolverSpHandle_t, streamId: *mut cudaStream_t) -> cusolverStatus_t;
        }
        cusolverSpGetStream(handle, streamId)
    }
}
pub unsafe fn cusolverSpScsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuComplex, right_upper_corner: cuComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuComplex, cuComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsreigsHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuComplex, right_upper_corner: cuComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
}
pub unsafe fn cusolverSpScsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f32, x0: *const f32, maxite: ::core::ffi::c_int, eps: f32, mu: *mut f32, x: *mut f32) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, *const f32, ::core::ffi::c_int, f32, *mut f32, *mut f32) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsreigvsi") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f32, x0: *const f32, maxite: ::core::ffi::c_int, eps: f32, mu: *mut f32, x: *mut f32) -> cusolverStatus_t;
        }
        cusolverSpScsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
}
pub unsafe fn cusolverSpScsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f32, x0: *const f32, maxite: ::core::ffi::c_int, tol: f32, mu: *mut f32, x: *mut f32) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, *const f32, ::core::ffi::c_int, f32, *mut f32, *mut f32) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsreigvsiHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: f32, x0: *const f32, maxite: ::core::ffi::c_int, tol: f32, mu: *mut f32, x: *mut f32) -> cusolverStatus_t;
        }
        cusolverSpScsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
}
pub unsafe fn cusolverSpScsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, tol: f32, rankA: *mut ::core::ffi::c_int, x: *mut f32, p: *mut ::core::ffi::c_int, min_norm: *mut f32) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, f32, *mut ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int, *mut f32) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrlsqvqrHost") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, tol: f32, rankA: *mut ::core::ffi::c_int, x: *mut f32, p: *mut ::core::ffi::c_int, min_norm: *mut f32) -> cusolverStatus_t;
        }
        cusolverSpScsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
}
pub unsafe fn cusolverSpScsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrlsvchol") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpScsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrlsvcholHost") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpScsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrlsvluHost") });
        _f(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpScsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrlsvqr") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpScsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, f32, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrlsvqrHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, tol: f32, reorder: ::core::ffi::c_int, x: *mut f32, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpScsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, csrqrInfo_t, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrqrBufferInfoBatched") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
}
pub unsafe fn cusolverSpScsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, x: *mut f32, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int, csrqrInfo_t, *mut ::core::ffi::c_void) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrqrsvBatched") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const f32, x: *mut f32, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t;
        }
        cusolverSpScsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
}
pub unsafe fn cusolverSpScsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpScsrzfdHost") });
        _f(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpScsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const f32, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpScsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
}
pub unsafe fn cusolverSpSetStream(handle: cusolverSpHandle_t, streamId: cudaStream_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, cudaStream_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpSetStream") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpSetStream(handle: cusolverSpHandle_t, streamId: cudaStream_t) -> cusolverStatus_t;
        }
        cusolverSpSetStream(handle, streamId)
    }
}
pub unsafe fn cusolverSpXcsrissymHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrEndPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, issym: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrissymHost") });
        _f(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrissymHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrEndPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, issym: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpXcsrissymHost(handle, m, nnzA, descrA, csrRowPtrA, csrEndPtrA, csrColIndA, issym)
    }
}
pub unsafe fn cusolverSpXcsrmetisndHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, options: *const i64, p: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const i64, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrmetisndHost") });
        _f(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrmetisndHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, options: *const i64, p: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpXcsrmetisndHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, options, p)
    }
}
pub unsafe fn cusolverSpXcsrpermHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, p: *const ::core::ffi::c_int, q: *const ::core::ffi::c_int, map: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrpermHost") });
        _f(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrpermHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *mut ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, p: *const ::core::ffi::c_int, q: *const ::core::ffi::c_int, map: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t;
        }
        cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, map, pBuffer)
    }
}
pub unsafe fn cusolverSpXcsrperm_bufferSizeHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *const ::core::ffi::c_int, q: *const ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrperm_bufferSizeHost") });
        _f(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrperm_bufferSizeHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *const ::core::ffi::c_int, q: *const ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, p, q, bufferSizeInBytes)
    }
}
pub unsafe fn cusolverSpXcsrqrAnalysisBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, info: csrqrInfo_t) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrqrInfo_t) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrqrAnalysisBatched") });
        _f(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrqrAnalysisBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, info: csrqrInfo_t) -> cusolverStatus_t;
        }
        cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, csrRowPtrA, csrColIndA, info)
    }
}
pub unsafe fn cusolverSpXcsrsymamdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrsymamdHost") });
        _f(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrsymamdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    }
}
pub unsafe fn cusolverSpXcsrsymmdqHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrsymmdqHost") });
        _f(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrsymmdqHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpXcsrsymmdqHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    }
}
pub unsafe fn cusolverSpXcsrsymrcmHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpXcsrsymrcmHost") });
        _f(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpXcsrsymrcmHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpXcsrsymrcmHost(handle, n, nnzA, descrA, csrRowPtrA, csrColIndA, p)
    }
}
pub unsafe fn cusolverSpZcsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuDoubleComplex, right_upper_corner: cuDoubleComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuDoubleComplex, cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsreigsHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsreigsHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, left_bottom_corner: cuDoubleComplex, right_upper_corner: cuDoubleComplex, num_eigs: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsreigsHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, left_bottom_corner, right_upper_corner, num_eigs)
    }
}
pub unsafe fn cusolverSpZcsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuDoubleComplex, x0: *const cuDoubleComplex, maxite: ::core::ffi::c_int, eps: f64, mu: *mut cuDoubleComplex, x: *mut cuDoubleComplex) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, f64, *mut cuDoubleComplex, *mut cuDoubleComplex) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsreigvsi") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsreigvsi(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuDoubleComplex, x0: *const cuDoubleComplex, maxite: ::core::ffi::c_int, eps: f64, mu: *mut cuDoubleComplex, x: *mut cuDoubleComplex) -> cusolverStatus_t;
        }
        cusolverSpZcsreigvsi(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, eps, mu, x)
    }
}
pub unsafe fn cusolverSpZcsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuDoubleComplex, x0: *const cuDoubleComplex, maxite: ::core::ffi::c_int, tol: f64, mu: *mut cuDoubleComplex, x: *mut cuDoubleComplex) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, f64, *mut cuDoubleComplex, *mut cuDoubleComplex) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsreigvsiHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsreigvsiHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, mu0: cuDoubleComplex, x0: *const cuDoubleComplex, maxite: ::core::ffi::c_int, tol: f64, mu: *mut cuDoubleComplex, x: *mut cuDoubleComplex) -> cusolverStatus_t;
        }
        cusolverSpZcsreigvsiHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x)
    }
}
pub unsafe fn cusolverSpZcsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, rankA: *mut ::core::ffi::c_int, x: *mut cuDoubleComplex, p: *mut ::core::ffi::c_int, min_norm: *mut f64) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, f64, *mut ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut f64) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrlsqvqrHost") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrlsqvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, rankA: *mut ::core::ffi::c_int, x: *mut cuDoubleComplex, p: *mut ::core::ffi::c_int, min_norm: *mut f64) -> cusolverStatus_t;
        }
        cusolverSpZcsrlsqvqrHost(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, rankA, x, p, min_norm)
    }
}
pub unsafe fn cusolverSpZcsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, f64, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrlsvchol") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrlsvchol(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpZcsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, f64, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrlsvcholHost") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrlsvcholHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsrlsvcholHost(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpZcsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, f64, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrlsvluHost") });
        _f(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrlsvluHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpZcsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, f64, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrlsvqr") });
        _f(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrlsvqr(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsrlsvqr(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpZcsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, f64, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrlsvqrHost") });
        _f(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrlsvqrHost(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, tol: f64, reorder: ::core::ffi::c_int, x: *mut cuDoubleComplex, singularity: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsrlsvqrHost(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity)
    }
}
pub unsafe fn cusolverSpZcsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, csrqrInfo_t, *mut usize, *mut usize) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrqrBufferInfoBatched") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrqrBufferInfoBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *const cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, internalDataInBytes: *mut usize, workspaceInBytes: *mut usize) -> cusolverStatus_t;
        }
        cusolverSpZcsrqrBufferInfoBatched(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, batchSize, info, internalDataInBytes, workspaceInBytes)
    }
}
pub unsafe fn cusolverSpZcsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, x: *mut cuDoubleComplex, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, csrqrInfo_t, *mut ::core::ffi::c_void) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrqrsvBatched") });
        _f(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrqrsvBatched(handle: cusolverSpHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, b: *const cuDoubleComplex, x: *mut cuDoubleComplex, batchSize: ::core::ffi::c_int, info: csrqrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusolverStatus_t;
        }
        cusolverSpZcsrqrsvBatched(handle, m, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer)
    }
}
pub unsafe fn cusolverSpZcsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusolverSpHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusolverStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusolverSpZcsrzfdHost") });
        _f(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusolverSpZcsrzfdHost(handle: cusolverSpHandle_t, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrValA: *const cuDoubleComplex, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, P: *mut ::core::ffi::c_int, numnz: *mut ::core::ffi::c_int) -> cusolverStatus_t;
        }
        cusolverSpZcsrzfdHost(handle, n, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, P, numnz)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cusolver"];
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
        let lib_names = std::vec!["cusolver"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
