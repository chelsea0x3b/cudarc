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
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type FILE = _IO_FILE;
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type _IO_lock_t = ::core::ffi::c_void;
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type __off64_t = ::core::ffi::c_long;
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type __off_t = ::core::ffi::c_long;
pub type bsric02Info_t = *mut bsric02Info;
pub type bsrilu02Info_t = *mut bsrilu02Info;
pub type bsrsm2Info_t = *mut bsrsm2Info;
pub type bsrsv2Info_t = *mut bsrsv2Info;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub type csrgemm2Info_t = *mut csrgemm2Info;
pub type csric02Info_t = *mut csric02Info;
pub type csrilu02Info_t = *mut csrilu02Info;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub type csrsm2Info_t = *mut csrsm2Info;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub type csrsv2Info_t = *mut csrsv2Info;
pub type csru2csrInfo_t = *mut csru2csrInfo;
pub type cuComplex = cuFloatComplex;
pub type cuDoubleComplex = double2;
pub type cuFloatComplex = float2;
pub type cudaStream_t = *mut CUstream_st;
pub type cusparseColorInfo_t = *mut cusparseColorInfo;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusparseConstDnMatDescr_t = *const cusparseDnMatDescr;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusparseConstDnVecDescr_t = *const cusparseDnVecDescr;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusparseConstSpMatDescr_t = *const cusparseSpMatDescr;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusparseConstSpVecDescr_t = *const cusparseSpVecDescr;
pub type cusparseDnMatDescr_t = *mut cusparseDnMatDescr;
pub type cusparseDnVecDescr_t = *mut cusparseDnVecDescr;
pub type cusparseHandle_t = *mut cusparseContext;
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusparseLoggerCallback_t = ::core::option::Option<unsafe extern "C" fn(logLevel: ::core::ffi::c_int, functionName: *const ::core::ffi::c_char, message: *const ::core::ffi::c_char)>;
pub type cusparseMatDescr_t = *mut cusparseMatDescr;
#[cfg(any(feature = "cuda-13030"))]
pub type cusparseSpGEAMDescr_t = *mut cusparseSpGEAMDescr;
pub type cusparseSpGEMMDescr_t = *mut cusparseSpGEMMDescr;
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cusparseSpMMOpPlan_t = *mut cusparseSpMMOpPlan;
pub type cusparseSpMatDescr_t = *mut cusparseSpMatDescr;
pub type cusparseSpSMDescr_t = *mut cusparseSpSMDescr;
pub type cusparseSpSVDescr_t = *mut cusparseSpSVDescr;
pub type cusparseSpVecDescr_t = *mut cusparseSpVecDescr;
pub type pruneInfo_t = *mut pruneInfo;
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
pub enum cusparseAction_t {
    CUSPARSE_ACTION_SYMBOLIC = 0,
    CUSPARSE_ACTION_NUMERIC = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseAlgMode_t {
    CUSPARSE_ALG_MERGE_PATH = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseColorAlg_t {
    CUSPARSE_COLOR_ALG0 = 0,
    CUSPARSE_COLOR_ALG1 = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseCsr2CscAlg_t {
    CUSPARSE_CSR2CSC_ALG1 = 1,
    CUSPARSE_CSR2CSC_ALG2 = 2,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseCsr2CscAlg_t {
    CUSPARSE_CSR2CSC_ALG_DEFAULT = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseDenseToSparseAlg_t {
    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseDiagType_t {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
    CUSPARSE_DIAG_TYPE_UNIT = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseDirection_t {
    CUSPARSE_DIRECTION_ROW = 0,
    CUSPARSE_DIRECTION_COLUMN = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseFillMode_t {
    CUSPARSE_FILL_MODE_LOWER = 0,
    CUSPARSE_FILL_MODE_UPPER = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseFormat_t {
    CUSPARSE_FORMAT_CSR = 1,
    CUSPARSE_FORMAT_CSC = 2,
    CUSPARSE_FORMAT_COO = 3,
    CUSPARSE_FORMAT_COO_AOS = 4,
    CUSPARSE_FORMAT_BLOCKED_ELL = 5,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseFormat_t {
    CUSPARSE_FORMAT_CSR = 1,
    CUSPARSE_FORMAT_CSC = 2,
    CUSPARSE_FORMAT_COO = 3,
    CUSPARSE_FORMAT_BLOCKED_ELL = 5,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseFormat_t {
    CUSPARSE_FORMAT_CSR = 1,
    CUSPARSE_FORMAT_CSC = 2,
    CUSPARSE_FORMAT_COO = 3,
    CUSPARSE_FORMAT_BLOCKED_ELL = 5,
    CUSPARSE_FORMAT_BSR = 6,
    CUSPARSE_FORMAT_SLICED_ELLPACK = 7,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseIndexBase_t {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseIndexType_t {
    CUSPARSE_INDEX_16U = 1,
    CUSPARSE_INDEX_32I = 2,
    CUSPARSE_INDEX_64I = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseMatrixType_t {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseOperation_t {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseOrder_t {
    CUSPARSE_ORDER_COL = 1,
    CUSPARSE_ORDER_ROW = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparsePointerMode_t {
    CUSPARSE_POINTER_MODE_HOST = 0,
    CUSPARSE_POINTER_MODE_DEVICE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSDDMMAlg_t {
    CUSPARSE_SDDMM_ALG_DEFAULT = 0,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSideMode_t {
    CUSPARSE_SIDE_LEFT = 0,
    CUSPARSE_SIDE_RIGHT = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSolvePolicy_t {
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0,
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpGEAMAlg_t {
    CUSPARSE_SPGEAM_ALG_DEFAULT = 0,
    CUSPARSE_SPGEAM_ALG1 = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpGEMMAlg_t {
    CUSPARSE_SPGEMM_DEFAULT = 0,
    CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = 1,
    CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpGEMMAlg_t {
    CUSPARSE_SPGEMM_DEFAULT = 0,
    CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = 1,
    CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2,
    CUSPARSE_SPGEMM_ALG1 = 3,
    CUSPARSE_SPGEMM_ALG2 = 4,
    CUSPARSE_SPGEMM_ALG3 = 5,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMMAlg_t {
    CUSPARSE_MM_ALG_DEFAULT = 0,
    CUSPARSE_COOMM_ALG1 = 1,
    CUSPARSE_COOMM_ALG2 = 2,
    CUSPARSE_COOMM_ALG3 = 3,
    CUSPARSE_CSRMM_ALG1 = 4,
    CUSPARSE_SPMM_COO_ALG4 = 5,
    CUSPARSE_SPMM_CSR_ALG2 = 6,
    CUSPARSE_SPMM_CSR_ALG3 = 12,
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMMAlg_t {
    CUSPARSE_SPMM_ALG_DEFAULT = 0,
    CUSPARSE_SPMM_COO_ALG1 = 1,
    CUSPARSE_SPMM_COO_ALG2 = 2,
    CUSPARSE_SPMM_COO_ALG3 = 3,
    CUSPARSE_SPMM_COO_ALG4 = 5,
    CUSPARSE_SPMM_CSR_ALG1 = 4,
    CUSPARSE_SPMM_CSR_ALG2 = 6,
    CUSPARSE_SPMM_CSR_ALG3 = 12,
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMMAlg_t {
    CUSPARSE_SPMM_ALG_DEFAULT = 0,
    CUSPARSE_SPMM_COO_ALG1 = 1,
    CUSPARSE_SPMM_COO_ALG2 = 2,
    CUSPARSE_SPMM_COO_ALG3 = 3,
    CUSPARSE_SPMM_COO_ALG4 = 5,
    CUSPARSE_SPMM_CSR_ALG1 = 4,
    CUSPARSE_SPMM_CSR_ALG2 = 6,
    CUSPARSE_SPMM_CSR_ALG3 = 12,
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13,
    CUSPARSE_SPMM_BSR_ALG1 = 14,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMMOpAlg_t {
    CUSPARSE_SPMM_OP_ALG_DEFAULT = 0,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMVAlg_t {
    CUSPARSE_MV_ALG_DEFAULT = 0,
    CUSPARSE_COOMV_ALG = 1,
    CUSPARSE_CSRMV_ALG1 = 2,
    CUSPARSE_CSRMV_ALG2 = 3,
    CUSPARSE_SPMV_COO_ALG2 = 4,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMVAlg_t {
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_CSR_ALG1 = 2,
    CUSPARSE_SPMV_CSR_ALG2 = 3,
    CUSPARSE_SPMV_COO_ALG1 = 1,
    CUSPARSE_SPMV_COO_ALG2 = 4,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMVAlg_t {
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_CSR_ALG1 = 2,
    CUSPARSE_SPMV_CSR_ALG2 = 3,
    CUSPARSE_SPMV_COO_ALG1 = 1,
    CUSPARSE_SPMV_COO_ALG2 = 4,
    CUSPARSE_SPMV_SELL_ALG1 = 5,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMVAlg_t {
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_CSR_ALG1 = 2,
    CUSPARSE_SPMV_CSR_ALG2 = 3,
    CUSPARSE_SPMV_COO_ALG1 = 1,
    CUSPARSE_SPMV_COO_ALG2 = 4,
    CUSPARSE_SPMV_SELL_ALG1 = 5,
    CUSPARSE_SPMV_BSR_ALG1 = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpMatAttribute_t {
    CUSPARSE_SPMAT_FILL_MODE = 0,
    CUSPARSE_SPMAT_DIAG_TYPE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpSMAlg_t {
    CUSPARSE_SPSM_ALG_DEFAULT = 0,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpSMUpdate_t {
    CUSPARSE_SPSM_UPDATE_GENERAL = 0,
    CUSPARSE_SPSM_UPDATE_DIAGONAL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpSVAlg_t {
    CUSPARSE_SPSV_ALG_DEFAULT = 0,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSpSVUpdate_t {
    CUSPARSE_SPSV_UPDATE_GENERAL = 0,
    CUSPARSE_SPSV_UPDATE_DIAGONAL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseSparseToDenseAlg_t {
    CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusparseStatus_t {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11,
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
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
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
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_codecvt {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_marker {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_wide_data {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bsric02Info {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bsrilu02Info {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bsrsm2Info {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct bsrsv2Info {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csrgemm2Info {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csric02Info {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csrilu02Info {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csrsm2Info {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csrsv2Info {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csru2csrInfo {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseColorInfo {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseDnMatDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseDnVecDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseMatDescr {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpGEAMDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpGEMMDescr {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpMMOpPlan {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpMatDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpSMDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpSVDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseSpVecDescr {
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
pub struct pruneInfo {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cusparseCsr2CscAlg_t {
    pub const CUSPARSE_CSR2CSC_ALG1: cusparseCsr2CscAlg_t = cusparseCsr2CscAlg_t::CUSPARSE_CSR2CSC_ALG_DEFAULT;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMMAlg_t {
    pub const CUSPARSE_SPMM_ALG_DEFAULT: cusparseSpMMAlg_t = cusparseSpMMAlg_t::CUSPARSE_MM_ALG_DEFAULT;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMMAlg_t {
    pub const CUSPARSE_SPMM_COO_ALG1: cusparseSpMMAlg_t = cusparseSpMMAlg_t::CUSPARSE_COOMM_ALG1;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMMAlg_t {
    pub const CUSPARSE_SPMM_COO_ALG2: cusparseSpMMAlg_t = cusparseSpMMAlg_t::CUSPARSE_COOMM_ALG2;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMMAlg_t {
    pub const CUSPARSE_SPMM_COO_ALG3: cusparseSpMMAlg_t = cusparseSpMMAlg_t::CUSPARSE_COOMM_ALG3;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMMAlg_t {
    pub const CUSPARSE_SPMM_CSR_ALG1: cusparseSpMMAlg_t = cusparseSpMMAlg_t::CUSPARSE_CSRMM_ALG1;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMVAlg_t {
    pub const CUSPARSE_SPMV_ALG_DEFAULT: cusparseSpMVAlg_t = cusparseSpMVAlg_t::CUSPARSE_MV_ALG_DEFAULT;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMVAlg_t {
    pub const CUSPARSE_SPMV_COO_ALG1: cusparseSpMVAlg_t = cusparseSpMVAlg_t::CUSPARSE_COOMV_ALG;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMVAlg_t {
    pub const CUSPARSE_SPMV_CSR_ALG1: cusparseSpMVAlg_t = cusparseSpMVAlg_t::CUSPARSE_CSRMV_ALG1;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
impl cusparseSpMVAlg_t {
    pub const CUSPARSE_SPMV_CSR_ALG2: cusparseSpMVAlg_t = cusparseSpMVAlg_t::CUSPARSE_CSRMV_ALG2;
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseAxpby(handle: cusparseHandle_t, alpha: *const ::core::ffi::c_void, vecX: cusparseSpVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *const ::core::ffi::c_void, cusparseSpVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseAxpby") });
        _f(handle, alpha, vecX, beta, vecY)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseAxpby(handle: cusparseHandle_t, alpha: *const ::core::ffi::c_void, vecX: cusparseSpVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseAxpby(handle, alpha, vecX, beta, vecY)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseAxpby(handle: cusparseHandle_t, alpha: *const ::core::ffi::c_void, vecX: cusparseConstSpVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *const ::core::ffi::c_void, cusparseConstSpVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseAxpby") });
        _f(handle, alpha, vecX, beta, vecY)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseAxpby(handle: cusparseHandle_t, alpha: *const ::core::ffi::c_void, vecX: cusparseConstSpVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseAxpby(handle, alpha, vecX, beta, vecY)
    }
}
pub unsafe fn cusparseBlockedEllGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, ellBlockSize: *mut i64, ellCols: *mut i64, ellColInd: *mut *mut ::core::ffi::c_void, ellValue: *mut *mut ::core::ffi::c_void, ellIdxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseBlockedEllGet") });
        _f(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseBlockedEllGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, ellBlockSize: *mut i64, ellCols: *mut i64, ellColInd: *mut *mut ::core::ffi::c_void, ellValue: *mut *mut ::core::ffi::c_void, ellIdxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseBlockedEllGet(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12010"))]
pub unsafe fn cusparseBsrSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, offsetsBatchStride: i64, columnsValuesBatchStride: i64, ValuesBatchStride: i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, ::core::ffi::c_int, i64, i64, i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseBsrSetStridedBatch") });
        _f(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride, ValuesBatchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseBsrSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, offsetsBatchStride: i64, columnsValuesBatchStride: i64, ValuesBatchStride: i64) -> cusparseStatus_t;
        }
        cusparseBsrSetStridedBatch(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride, ValuesBatchStride)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseBsrSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, offsetsBatchStride: i64, columnsBatchStride: i64, ValuesBatchStride: i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, ::core::ffi::c_int, i64, i64, i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseBsrSetStridedBatch") });
        _f(spMatDescr, batchCount, offsetsBatchStride, columnsBatchStride, ValuesBatchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseBsrSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, offsetsBatchStride: i64, columnsBatchStride: i64, ValuesBatchStride: i64) -> cusparseStatus_t;
        }
        cusparseBsrSetStridedBatch(spMatDescr, batchCount, offsetsBatchStride, columnsBatchStride, ValuesBatchStride)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const cuComplex, xVal: *const cuComplex, xInd: *const ::core::ffi::c_int, y: *mut cuComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const ::core::ffi::c_int, *mut cuComplex, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCaxpyi") });
        _f(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const cuComplex, xVal: *const cuComplex, xInd: *const ::core::ffi::c_int, y: *mut cuComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
}
pub unsafe fn cusparseCbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseCbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsric02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsric02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
}
pub unsafe fn cusparseCbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsric02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrilu02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrilu02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrilu02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut cuComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuComplex) -> cusparseStatus_t;
        }
        cusparseCbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
pub unsafe fn cusparseCbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrmm") });
        _f(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cusparseCbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuComplex, beta: *const cuComplex, y: *mut cuComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *mut cuComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrmv") });
        _f(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuComplex, beta: *const cuComplex, y: *mut cuComplex) -> cusparseStatus_t;
        }
        cusparseCbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
pub unsafe fn cusparseCbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrsm2_analysis") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrsm2_bufferSize") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const cuComplex, ldb: ::core::ffi::c_int, X: *mut cuComplex, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *const cuComplex, ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrsm2_solve") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const cuComplex, ldb: ::core::ffi::c_int, X: *mut cuComplex, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrsv2_analysis") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrsv2_bufferSize") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const cuComplex, x: *mut cuComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *const cuComplex, *mut cuComplex, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrsv2_solve") });
        _f(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const cuComplex, x: *mut cuComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseCbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuComplex, beta: *const cuComplex, y: *mut cuComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *mut cuComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCbsrxmv") });
        _f(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuComplex, beta: *const cuComplex, y: *mut cuComplex) -> cusparseStatus_t;
        }
        cusparseCbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const cuComplex, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsc2dense") });
        _f(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const cuComplex, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
}
pub unsafe fn cusparseCcsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsr2bsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
}
pub unsafe fn cusparseCcsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut cuComplex, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: cuComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cuComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsr2csr_compress") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut cuComplex, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: cuComplex) -> cusparseStatus_t;
        }
        cusparseCcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
}
pub unsafe fn cusparseCcsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsr2csru") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsr2dense") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut cuComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
}
pub unsafe fn cusparseCcsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsr2gebsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
}
pub unsafe fn cusparseCcsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsr2gebsr_bufferSize") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCcsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f32, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseColorInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrcolor") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f32, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t;
        }
        cusparseCcsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
}
pub unsafe fn cusparseCcsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrgeam2") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseCcsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const cuComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrgeam2_bufferSizeExt") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const cuComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const cuComplex, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csrgemm2Info_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrgemm2") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const cuComplex, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrgemm2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrgemm2_bufferSizeExt") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCcsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsric02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCcsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsric02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCcsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsric02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCcsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrilu02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCcsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrilu02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseCcsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrilu02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCcsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut cuComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuComplex) -> cusparseStatus_t;
        }
        cusparseCcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsm2_analysis") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsm2_bufferSizeExt") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsm2_solve") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut cuComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsv2_analysis") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsv2_bufferSize") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsv2_bufferSizeExt") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseCcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCcsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const cuComplex, x: *mut cuComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *const cuComplex, *mut cuComplex, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsrsv2_solve") });
        _f(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const cuComplex, x: *mut cuComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseCcsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsru2csr") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCcsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
pub unsafe fn cusparseCcsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCcsru2csr_bufferSizeExt") });
        _f(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCcsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut cuComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCcsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuComplex, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut cuComplex, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCdense2csc") });
        _f(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuComplex, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut cuComplex, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuComplex, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCdense2csr") });
        _f(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuComplex, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut cuComplex, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
}
pub unsafe fn cusparseCgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgebsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseCgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut cuComplex, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseAction_t, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgebsr2gebsc") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut cuComplex, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseCgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgebsr2gebsc_bufferSize") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgebsr2gebsr") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
}
pub unsafe fn cusparseCgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgebsr2gebsr_bufferSize") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, cscValB: *const cuComplex, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuComplex, *mut cuComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgemmi") });
        _f(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, cscValB: *const cuComplex, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const cuComplex, C: *mut cuComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
}
pub unsafe fn cusparseCgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const cuComplex, xInd: *const ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *const cuComplex, *mut cuComplex, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgemvi") });
        _f(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuComplex, A: *const cuComplex, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const cuComplex, xInd: *const ::core::ffi::c_int, beta: *const cuComplex, y: *mut cuComplex, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseCgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgemvi_bufferSize") });
        _f(handle, transA, m, n, nnz, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    }
}
pub unsafe fn cusparseCgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut cuComplex, dl: *mut cuComplex, d: *mut cuComplex, du: *mut cuComplex, dw: *mut cuComplex, x: *mut cuComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, *mut cuComplex, *mut cuComplex, *mut cuComplex, *mut cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgpsvInterleavedBatch") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut cuComplex, dl: *mut cuComplex, d: *mut cuComplex, du: *mut cuComplex, dw: *mut cuComplex, x: *mut cuComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseCgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const cuComplex, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, dw: *const cuComplex, x: *const cuComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgpsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const cuComplex, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, dw: *const cuComplex, x: *const cuComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const cuComplex, xVal: *mut cuComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuComplex, *mut cuComplex, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgthr") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const cuComplex, xVal: *mut cuComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseCgthr(handle, nnz, y, xVal, xInd, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut cuComplex, xVal: *mut cuComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut cuComplex, *mut cuComplex, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgthrz") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut cuComplex, xVal: *mut cuComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseCgthrz(handle, nnz, y, xVal, xInd, idxBase)
    }
}
pub unsafe fn cusparseCgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *mut cuComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsv2") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *mut cuComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseCgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, x: *mut cuComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsv2StridedBatch") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, x: *mut cuComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
}
pub unsafe fn cusparseCgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, x: *const cuComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, ::core::ffi::c_int, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsv2StridedBatch_bufferSizeExt") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, x: *const cuComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseCgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *const cuComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsv2_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *const cuComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseCgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *mut cuComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsv2_nopivot") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *mut cuComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseCgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *const cuComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsv2_nopivot_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, B: *const cuComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseCgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut cuComplex, d: *mut cuComplex, du: *mut cuComplex, x: *mut cuComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuComplex, *mut cuComplex, *mut cuComplex, *mut cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsvInterleavedBatch") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut cuComplex, d: *mut cuComplex, du: *mut cuComplex, x: *mut cuComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseCgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, x: *const cuComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuComplex, *const cuComplex, *const cuComplex, *const cuComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCgtsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const cuComplex, d: *const cuComplex, du: *const cuComplex, x: *const cuComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseCnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuComplex, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCnnz") });
        _f(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuComplex, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
}
pub unsafe fn cusparseCnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: cuComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseMatDescr_t, *const cuComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cuComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCnnz_compress") });
        _f(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const cuComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: cuComplex) -> cusparseStatus_t;
        }
        cusparseCnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstBlockedEllGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, ellBlockSize: *mut i64, ellCols: *mut i64, ellColInd: *mut *const ::core::ffi::c_void, ellValue: *mut *const ::core::ffi::c_void, ellIdxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut i64, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstBlockedEllGet") });
        _f(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstBlockedEllGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, ellBlockSize: *mut i64, ellCols: *mut i64, ellColInd: *mut *const ::core::ffi::c_void, ellValue: *mut *const ::core::ffi::c_void, ellIdxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseConstBlockedEllGet(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstCooGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cooRowInd: *mut *const ::core::ffi::c_void, cooColInd: *mut *const ::core::ffi::c_void, cooValues: *mut *const ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstCooGet") });
        _f(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstCooGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cooRowInd: *mut *const ::core::ffi::c_void, cooColInd: *mut *const ::core::ffi::c_void, cooValues: *mut *const ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseConstCooGet(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstCscGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cscColOffsets: *mut *const ::core::ffi::c_void, cscRowInd: *mut *const ::core::ffi::c_void, cscValues: *mut *const ::core::ffi::c_void, cscColOffsetsType: *mut cusparseIndexType_t, cscRowIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstCscGet") });
        _f(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstCscGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cscColOffsets: *mut *const ::core::ffi::c_void, cscRowInd: *mut *const ::core::ffi::c_void, cscValues: *mut *const ::core::ffi::c_void, cscColOffsetsType: *mut cusparseIndexType_t, cscRowIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseConstCscGet(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstCsrGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, csrRowOffsets: *mut *const ::core::ffi::c_void, csrColInd: *mut *const ::core::ffi::c_void, csrValues: *mut *const ::core::ffi::c_void, csrRowOffsetsType: *mut cusparseIndexType_t, csrColIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstCsrGet") });
        _f(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstCsrGet(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, csrRowOffsets: *mut *const ::core::ffi::c_void, csrColInd: *mut *const ::core::ffi::c_void, csrValues: *mut *const ::core::ffi::c_void, csrRowOffsetsType: *mut cusparseIndexType_t, csrColIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseConstCsrGet(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstDnMatGet(dnMatDescr: cusparseConstDnMatDescr_t, rows: *mut i64, cols: *mut i64, ld: *mut i64, values: *mut *const ::core::ffi::c_void, type_: *mut cudaDataType, order: *mut cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *const ::core::ffi::c_void, *mut cudaDataType, *mut cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstDnMatGet") });
        _f(dnMatDescr, rows, cols, ld, values, type_, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstDnMatGet(dnMatDescr: cusparseConstDnMatDescr_t, rows: *mut i64, cols: *mut i64, ld: *mut i64, values: *mut *const ::core::ffi::c_void, type_: *mut cudaDataType, order: *mut cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseConstDnMatGet(dnMatDescr, rows, cols, ld, values, type_, order)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstDnMatGetValues(dnMatDescr: cusparseConstDnMatDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnMatDescr_t, *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstDnMatGetValues") });
        _f(dnMatDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstDnMatGetValues(dnMatDescr: cusparseConstDnMatDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseConstDnMatGetValues(dnMatDescr, values)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstDnVecGet(dnVecDescr: cusparseConstDnVecDescr_t, size: *mut i64, values: *mut *const ::core::ffi::c_void, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnVecDescr_t, *mut i64, *mut *const ::core::ffi::c_void, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstDnVecGet") });
        _f(dnVecDescr, size, values, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstDnVecGet(dnVecDescr: cusparseConstDnVecDescr_t, size: *mut i64, values: *mut *const ::core::ffi::c_void, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseConstDnVecGet(dnVecDescr, size, values, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstDnVecGetValues(dnVecDescr: cusparseConstDnVecDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnVecDescr_t, *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstDnVecGetValues") });
        _f(dnVecDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstDnVecGetValues(dnVecDescr: cusparseConstDnVecDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseConstDnVecGetValues(dnVecDescr, values)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstSpMatGetValues(spMatDescr: cusparseConstSpMatDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstSpMatGetValues") });
        _f(spMatDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstSpMatGetValues(spMatDescr: cusparseConstSpMatDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseConstSpMatGetValues(spMatDescr, values)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstSpVecGet(spVecDescr: cusparseConstSpVecDescr_t, size: *mut i64, nnz: *mut i64, indices: *mut *const ::core::ffi::c_void, values: *mut *const ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpVecDescr_t, *mut i64, *mut i64, *mut *const ::core::ffi::c_void, *mut *const ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstSpVecGet") });
        _f(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstSpVecGet(spVecDescr: cusparseConstSpVecDescr_t, size: *mut i64, nnz: *mut i64, indices: *mut *const ::core::ffi::c_void, values: *mut *const ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseConstSpVecGet(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseConstSpVecGetValues(spVecDescr: cusparseConstSpVecDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpVecDescr_t, *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstSpVecGetValues") });
        _f(spVecDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstSpVecGetValues(spVecDescr: cusparseConstSpVecDescr_t, values: *mut *const ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseConstSpVecGetValues(spVecDescr, values)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseConstrainedGeMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstrainedGeMM") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstrainedGeMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseConstrainedGeMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseConstrainedGeMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseConstrainedGeMM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseConstrainedGeMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseConstrainedGeMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, bufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCooAoSGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cooInd: *mut *mut ::core::ffi::c_void, cooValues: *mut *mut ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCooAoSGet") });
        _f(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCooAoSGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cooInd: *mut *mut ::core::ffi::c_void, cooValues: *mut *mut ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseCooAoSGet(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCooGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cooRowInd: *mut *mut ::core::ffi::c_void, cooColInd: *mut *mut ::core::ffi::c_void, cooValues: *mut *mut ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCooGet") });
        _f(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCooGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cooRowInd: *mut *mut ::core::ffi::c_void, cooColInd: *mut *mut ::core::ffi::c_void, cooValues: *mut *mut ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseCooGet(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCooSetPointers(spMatDescr: cusparseSpMatDescr_t, cooRows: *mut ::core::ffi::c_void, cooColumns: *mut ::core::ffi::c_void, cooValues: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCooSetPointers") });
        _f(spMatDescr, cooRows, cooColumns, cooValues)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCooSetPointers(spMatDescr: cusparseSpMatDescr_t, cooRows: *mut ::core::ffi::c_void, cooColumns: *mut ::core::ffi::c_void, cooValues: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCooSetPointers(spMatDescr, cooRows, cooColumns, cooValues)
    }
}
pub unsafe fn cusparseCooSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, batchStride: i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, ::core::ffi::c_int, i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCooSetStridedBatch") });
        _f(spMatDescr, batchCount, batchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCooSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, batchStride: i64) -> cusparseStatus_t;
        }
        cusparseCooSetStridedBatch(spMatDescr, batchCount, batchStride)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCopyMatDescr(dest: cusparseMatDescr_t, src: cusparseMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t, cusparseMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCopyMatDescr") });
        _f(dest, src)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCopyMatDescr(dest: cusparseMatDescr_t, src: cusparseMatDescr_t) -> cusparseStatus_t;
        }
        cusparseCopyMatDescr(dest, src)
    }
}
pub unsafe fn cusparseCreate(handle: *mut cusparseHandle_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseHandle_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreate(handle: *mut cusparseHandle_t) -> cusparseStatus_t;
        }
        cusparseCreate(handle)
    }
}
pub unsafe fn cusparseCreateBlockedEll(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, ellBlockSize: i64, ellCols: i64, ellColInd: *mut ::core::ffi::c_void, ellValue: *mut ::core::ffi::c_void, ellIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBlockedEll") });
        _f(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBlockedEll(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, ellBlockSize: i64, ellCols: i64, ellColInd: *mut ::core::ffi::c_void, ellValue: *mut ::core::ffi::c_void, ellIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateBlockedEll(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12010"))]
pub unsafe fn cusparseCreateBsr(spMatDescr: *mut cusparseSpMatDescr_t, brows: i64, bcols: i64, bnnz: i64, rowBlockDim: i64, colBlockDim: i64, bsrRowOffsets: *mut ::core::ffi::c_void, bsrColInd: *mut ::core::ffi::c_void, bsrValues: *mut ::core::ffi::c_void, bsrRowOffsetsType: cusparseIndexType_t, bsrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType, cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBsr") });
        _f(spMatDescr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBsr(spMatDescr: *mut cusparseSpMatDescr_t, brows: i64, bcols: i64, bnnz: i64, rowBlockDim: i64, colBlockDim: i64, bsrRowOffsets: *mut ::core::ffi::c_void, bsrColInd: *mut ::core::ffi::c_void, bsrValues: *mut ::core::ffi::c_void, bsrRowOffsetsType: cusparseIndexType_t, bsrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseCreateBsr(spMatDescr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateBsr(spMatDescr: *mut cusparseSpMatDescr_t, brows: i64, bcols: i64, bnnz: i64, rowBlockSize: i64, colBlockSize: i64, bsrRowOffsets: *mut ::core::ffi::c_void, bsrColInd: *mut ::core::ffi::c_void, bsrValues: *mut ::core::ffi::c_void, bsrRowOffsetsType: cusparseIndexType_t, bsrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType, cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBsr") });
        _f(spMatDescr, brows, bcols, bnnz, rowBlockSize, colBlockSize, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBsr(spMatDescr: *mut cusparseSpMatDescr_t, brows: i64, bcols: i64, bnnz: i64, rowBlockSize: i64, colBlockSize: i64, bsrRowOffsets: *mut ::core::ffi::c_void, bsrColInd: *mut ::core::ffi::c_void, bsrValues: *mut ::core::ffi::c_void, bsrRowOffsetsType: cusparseIndexType_t, bsrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseCreateBsr(spMatDescr, brows, bcols, bnnz, rowBlockSize, colBlockSize, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)
    }
}
pub unsafe fn cusparseCreateBsric02Info(info: *mut bsric02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut bsric02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBsric02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBsric02Info(info: *mut bsric02Info_t) -> cusparseStatus_t;
        }
        cusparseCreateBsric02Info(info)
    }
}
pub unsafe fn cusparseCreateBsrilu02Info(info: *mut bsrilu02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut bsrilu02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBsrilu02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBsrilu02Info(info: *mut bsrilu02Info_t) -> cusparseStatus_t;
        }
        cusparseCreateBsrilu02Info(info)
    }
}
pub unsafe fn cusparseCreateBsrsm2Info(info: *mut bsrsm2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut bsrsm2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBsrsm2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBsrsm2Info(info: *mut bsrsm2Info_t) -> cusparseStatus_t;
        }
        cusparseCreateBsrsm2Info(info)
    }
}
pub unsafe fn cusparseCreateBsrsv2Info(info: *mut bsrsv2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut bsrsv2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateBsrsv2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateBsrsv2Info(info: *mut bsrsv2Info_t) -> cusparseStatus_t;
        }
        cusparseCreateBsrsv2Info(info)
    }
}
pub unsafe fn cusparseCreateColorInfo(info: *mut cusparseColorInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseColorInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateColorInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateColorInfo(info: *mut cusparseColorInfo_t) -> cusparseStatus_t;
        }
        cusparseCreateColorInfo(info)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstBlockedEll(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, ellBlockSize: i64, ellCols: i64, ellColInd: *const ::core::ffi::c_void, ellValue: *const ::core::ffi::c_void, ellIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpMatDescr_t, i64, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstBlockedEll") });
        _f(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstBlockedEll(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, ellBlockSize: i64, ellCols: i64, ellColInd: *const ::core::ffi::c_void, ellValue: *const ::core::ffi::c_void, ellIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstBlockedEll(spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstBsr(spMatDescr: *mut cusparseConstSpMatDescr_t, brows: i64, bcols: i64, bnnz: i64, rowBlockDim: i64, colBlockDim: i64, bsrRowOffsets: *const ::core::ffi::c_void, bsrColInd: *const ::core::ffi::c_void, bsrValues: *const ::core::ffi::c_void, bsrRowOffsetsType: cusparseIndexType_t, bsrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpMatDescr_t, i64, i64, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType, cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstBsr") });
        _f(spMatDescr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstBsr(spMatDescr: *mut cusparseConstSpMatDescr_t, brows: i64, bcols: i64, bnnz: i64, rowBlockDim: i64, colBlockDim: i64, bsrRowOffsets: *const ::core::ffi::c_void, bsrColInd: *const ::core::ffi::c_void, bsrValues: *const ::core::ffi::c_void, bsrRowOffsetsType: cusparseIndexType_t, bsrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseCreateConstBsr(spMatDescr, brows, bcols, bnnz, rowBlockDim, colBlockDim, bsrRowOffsets, bsrColInd, bsrValues, bsrRowOffsetsType, bsrColIndType, idxBase, valueType, order)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstCoo(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cooRowInd: *const ::core::ffi::c_void, cooColInd: *const ::core::ffi::c_void, cooValues: *const ::core::ffi::c_void, cooIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpMatDescr_t, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstCoo") });
        _f(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstCoo(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cooRowInd: *const ::core::ffi::c_void, cooColInd: *const ::core::ffi::c_void, cooValues: *const ::core::ffi::c_void, cooIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstCsc(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cscColOffsets: *const ::core::ffi::c_void, cscRowInd: *const ::core::ffi::c_void, cscValues: *const ::core::ffi::c_void, cscColOffsetsType: cusparseIndexType_t, cscRowIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpMatDescr_t, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstCsc") });
        _f(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstCsc(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cscColOffsets: *const ::core::ffi::c_void, cscRowInd: *const ::core::ffi::c_void, cscValues: *const ::core::ffi::c_void, cscColOffsetsType: cusparseIndexType_t, cscRowIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstCsc(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstCsr(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, csrRowOffsets: *const ::core::ffi::c_void, csrColInd: *const ::core::ffi::c_void, csrValues: *const ::core::ffi::c_void, csrRowOffsetsType: cusparseIndexType_t, csrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpMatDescr_t, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstCsr") });
        _f(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstCsr(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, csrRowOffsets: *const ::core::ffi::c_void, csrColInd: *const ::core::ffi::c_void, csrValues: *const ::core::ffi::c_void, csrRowOffsetsType: cusparseIndexType_t, csrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstDnMat(dnMatDescr: *mut cusparseConstDnMatDescr_t, rows: i64, cols: i64, ld: i64, values: *const ::core::ffi::c_void, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstDnMatDescr_t, i64, i64, i64, *const ::core::ffi::c_void, cudaDataType, cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstDnMat") });
        _f(dnMatDescr, rows, cols, ld, values, valueType, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstDnMat(dnMatDescr: *mut cusparseConstDnMatDescr_t, rows: i64, cols: i64, ld: i64, values: *const ::core::ffi::c_void, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseCreateConstDnMat(dnMatDescr, rows, cols, ld, values, valueType, order)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstDnVec(dnVecDescr: *mut cusparseConstDnVecDescr_t, size: i64, values: *const ::core::ffi::c_void, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstDnVecDescr_t, i64, *const ::core::ffi::c_void, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstDnVec") });
        _f(dnVecDescr, size, values, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstDnVec(dnVecDescr: *mut cusparseConstDnVecDescr_t, size: i64, values: *const ::core::ffi::c_void, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstDnVec(dnVecDescr, size, values, valueType)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstSlicedEll(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, sellValuesSize: i64, sliceSize: i64, sellSliceOffsets: *const ::core::ffi::c_void, sellColInd: *const ::core::ffi::c_void, sellValues: *const ::core::ffi::c_void, sellSliceOffsetsType: cusparseIndexType_t, sellColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpMatDescr_t, i64, i64, i64, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstSlicedEll") });
        _f(spMatDescr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues, sellSliceOffsetsType, sellColIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstSlicedEll(spMatDescr: *mut cusparseConstSpMatDescr_t, rows: i64, cols: i64, nnz: i64, sellValuesSize: i64, sliceSize: i64, sellSliceOffsets: *const ::core::ffi::c_void, sellColInd: *const ::core::ffi::c_void, sellValues: *const ::core::ffi::c_void, sellSliceOffsetsType: cusparseIndexType_t, sellColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstSlicedEll(spMatDescr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues, sellSliceOffsetsType, sellColIndType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateConstSpVec(spVecDescr: *mut cusparseConstSpVecDescr_t, size: i64, nnz: i64, indices: *const ::core::ffi::c_void, values: *const ::core::ffi::c_void, idxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseConstSpVecDescr_t, i64, i64, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateConstSpVec") });
        _f(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateConstSpVec(spVecDescr: *mut cusparseConstSpVecDescr_t, size: i64, nnz: i64, indices: *const ::core::ffi::c_void, values: *const ::core::ffi::c_void, idxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateConstSpVec(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCreateCoo(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cooRowInd: *mut ::core::ffi::c_void, cooColInd: *mut ::core::ffi::c_void, cooValues: *mut ::core::ffi::c_void, cooIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCoo") });
        _f(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCoo(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cooRowInd: *mut ::core::ffi::c_void, cooColInd: *mut ::core::ffi::c_void, cooValues: *mut ::core::ffi::c_void, cooIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCreateCooAoS(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cooInd: *mut ::core::ffi::c_void, cooValues: *mut ::core::ffi::c_void, cooIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCooAoS") });
        _f(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCooAoS(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cooInd: *mut ::core::ffi::c_void, cooValues: *mut ::core::ffi::c_void, cooIdxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateCooAoS(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCreateCsc(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cscColOffsets: *mut ::core::ffi::c_void, cscRowInd: *mut ::core::ffi::c_void, cscValues: *mut ::core::ffi::c_void, cscColOffsetsType: cusparseIndexType_t, cscRowIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsc") });
        _f(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsc(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, cscColOffsets: *mut ::core::ffi::c_void, cscRowInd: *mut ::core::ffi::c_void, cscValues: *mut ::core::ffi::c_void, cscColOffsetsType: cusparseIndexType_t, cscRowIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateCsc(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCreateCsr(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, csrRowOffsets: *mut ::core::ffi::c_void, csrColInd: *mut ::core::ffi::c_void, csrValues: *mut ::core::ffi::c_void, csrRowOffsetsType: cusparseIndexType_t, csrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsr") });
        _f(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsr(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, csrRowOffsets: *mut ::core::ffi::c_void, csrColInd: *mut ::core::ffi::c_void, csrValues: *mut ::core::ffi::c_void, csrRowOffsetsType: cusparseIndexType_t, csrColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCreateCsrgemm2Info(info: *mut csrgemm2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csrgemm2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsrgemm2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsrgemm2Info(info: *mut csrgemm2Info_t) -> cusparseStatus_t;
        }
        cusparseCreateCsrgemm2Info(info)
    }
}
pub unsafe fn cusparseCreateCsric02Info(info: *mut csric02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csric02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsric02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsric02Info(info: *mut csric02Info_t) -> cusparseStatus_t;
        }
        cusparseCreateCsric02Info(info)
    }
}
pub unsafe fn cusparseCreateCsrilu02Info(info: *mut csrilu02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csrilu02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsrilu02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsrilu02Info(info: *mut csrilu02Info_t) -> cusparseStatus_t;
        }
        cusparseCreateCsrilu02Info(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCreateCsrsm2Info(info: *mut csrsm2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csrsm2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsrsm2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsrsm2Info(info: *mut csrsm2Info_t) -> cusparseStatus_t;
        }
        cusparseCreateCsrsm2Info(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCreateCsrsv2Info(info: *mut csrsv2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csrsv2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsrsv2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsrsv2Info(info: *mut csrsv2Info_t) -> cusparseStatus_t;
        }
        cusparseCreateCsrsv2Info(info)
    }
}
pub unsafe fn cusparseCreateCsru2csrInfo(info: *mut csru2csrInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut csru2csrInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateCsru2csrInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateCsru2csrInfo(info: *mut csru2csrInfo_t) -> cusparseStatus_t;
        }
        cusparseCreateCsru2csrInfo(info)
    }
}
pub unsafe fn cusparseCreateDnMat(dnMatDescr: *mut cusparseDnMatDescr_t, rows: i64, cols: i64, ld: i64, values: *mut ::core::ffi::c_void, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseDnMatDescr_t, i64, i64, i64, *mut ::core::ffi::c_void, cudaDataType, cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateDnMat") });
        _f(dnMatDescr, rows, cols, ld, values, valueType, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateDnMat(dnMatDescr: *mut cusparseDnMatDescr_t, rows: i64, cols: i64, ld: i64, values: *mut ::core::ffi::c_void, valueType: cudaDataType, order: cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseCreateDnMat(dnMatDescr, rows, cols, ld, values, valueType, order)
    }
}
pub unsafe fn cusparseCreateDnVec(dnVecDescr: *mut cusparseDnVecDescr_t, size: i64, values: *mut ::core::ffi::c_void, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseDnVecDescr_t, i64, *mut ::core::ffi::c_void, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateDnVec") });
        _f(dnVecDescr, size, values, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateDnVec(dnVecDescr: *mut cusparseDnVecDescr_t, size: i64, values: *mut ::core::ffi::c_void, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateDnVec(dnVecDescr, size, values, valueType)
    }
}
pub unsafe fn cusparseCreateIdentityPermutation(handle: cusparseHandle_t, n: ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateIdentityPermutation") });
        _f(handle, n, p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateIdentityPermutation(handle: cusparseHandle_t, n: ::core::ffi::c_int, p: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseCreateIdentityPermutation(handle, n, p)
    }
}
pub unsafe fn cusparseCreateMatDescr(descrA: *mut cusparseMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateMatDescr") });
        _f(descrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateMatDescr(descrA: *mut cusparseMatDescr_t) -> cusparseStatus_t;
        }
        cusparseCreateMatDescr(descrA)
    }
}
pub unsafe fn cusparseCreatePruneInfo(info: *mut pruneInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut pruneInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreatePruneInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreatePruneInfo(info: *mut pruneInfo_t) -> cusparseStatus_t;
        }
        cusparseCreatePruneInfo(info)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCreateSlicedEll(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, sellValuesSize: i64, sliceSize: i64, sellSliceOffsets: *mut ::core::ffi::c_void, sellColInd: *mut ::core::ffi::c_void, sellValues: *mut ::core::ffi::c_void, sellSliceOffsetsType: cusparseIndexType_t, sellColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpMatDescr_t, i64, i64, i64, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateSlicedEll") });
        _f(spMatDescr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues, sellSliceOffsetsType, sellColIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateSlicedEll(spMatDescr: *mut cusparseSpMatDescr_t, rows: i64, cols: i64, nnz: i64, sellValuesSize: i64, sliceSize: i64, sellSliceOffsets: *mut ::core::ffi::c_void, sellColInd: *mut ::core::ffi::c_void, sellValues: *mut ::core::ffi::c_void, sellSliceOffsetsType: cusparseIndexType_t, sellColIndType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateSlicedEll(spMatDescr, rows, cols, nnz, sellValuesSize, sliceSize, sellSliceOffsets, sellColInd, sellValues, sellSliceOffsetsType, sellColIndType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCreateSpVec(spVecDescr: *mut cusparseSpVecDescr_t, size: i64, nnz: i64, indices: *mut ::core::ffi::c_void, values: *mut ::core::ffi::c_void, idxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpVecDescr_t, i64, i64, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCreateSpVec") });
        _f(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCreateSpVec(spVecDescr: *mut cusparseSpVecDescr_t, size: i64, nnz: i64, indices: *mut ::core::ffi::c_void, values: *mut ::core::ffi::c_void, idxType: cusparseIndexType_t, idxBase: cusparseIndexBase_t, valueType: cudaDataType) -> cusparseStatus_t;
        }
        cusparseCreateSpVec(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseCscGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cscColOffsets: *mut *mut ::core::ffi::c_void, cscRowInd: *mut *mut ::core::ffi::c_void, cscValues: *mut *mut ::core::ffi::c_void, cscColOffsetsType: *mut cusparseIndexType_t, cscRowIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCscGet") });
        _f(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCscGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, cscColOffsets: *mut *mut ::core::ffi::c_void, cscRowInd: *mut *mut ::core::ffi::c_void, cscValues: *mut *mut ::core::ffi::c_void, cscColOffsetsType: *mut cusparseIndexType_t, cscRowIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseCscGet(spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCscSetPointers(spMatDescr: cusparseSpMatDescr_t, cscColOffsets: *mut ::core::ffi::c_void, cscRowInd: *mut ::core::ffi::c_void, cscValues: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCscSetPointers") });
        _f(spMatDescr, cscColOffsets, cscRowInd, cscValues)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCscSetPointers(spMatDescr: cusparseSpMatDescr_t, cscColOffsets: *mut ::core::ffi::c_void, cscRowInd: *mut ::core::ffi::c_void, cscValues: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCscSetPointers(spMatDescr, cscColOffsets, cscRowInd, cscValues)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const cuComplex, xInd: *const ::core::ffi::c_int, y: *mut cuComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuComplex, *const ::core::ffi::c_int, *mut cuComplex, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsctr") });
        _f(handle, nnz, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const cuComplex, xInd: *const ::core::ffi::c_int, y: *mut cuComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseCsctr(handle, nnz, xVal, xInd, y, idxBase)
    }
}
pub unsafe fn cusparseCsr2cscEx2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *const ::core::ffi::c_void, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, cscVal: *mut ::core::ffi::c_void, cscColPtr: *mut ::core::ffi::c_int, cscRowInd: *mut ::core::ffi::c_int, valType: cudaDataType, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, alg: cusparseCsr2CscAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_void, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsr2cscEx2") });
        _f(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsr2cscEx2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *const ::core::ffi::c_void, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, cscVal: *mut ::core::ffi::c_void, cscColPtr: *mut ::core::ffi::c_int, cscRowInd: *mut ::core::ffi::c_int, valType: cudaDataType, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, alg: cusparseCsr2CscAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer)
    }
}
pub unsafe fn cusparseCsr2cscEx2_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *const ::core::ffi::c_void, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, cscVal: *mut ::core::ffi::c_void, cscColPtr: *mut ::core::ffi::c_int, cscRowInd: *mut ::core::ffi::c_int, valType: cudaDataType, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, alg: cusparseCsr2CscAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut ::core::ffi::c_void, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cudaDataType, cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsr2cscEx2_bufferSize") });
        _f(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsr2cscEx2_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *const ::core::ffi::c_void, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *const ::core::ffi::c_int, cscVal: *mut ::core::ffi::c_void, cscColPtr: *mut ::core::ffi::c_int, cscRowInd: *mut ::core::ffi::c_int, valType: cudaDataType, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, alg: cusparseCsr2CscAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, bufferSize)
    }
}
pub unsafe fn cusparseCsrGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, csrRowOffsets: *mut *mut ::core::ffi::c_void, csrColInd: *mut *mut ::core::ffi::c_void, csrValues: *mut *mut ::core::ffi::c_void, csrRowOffsetsType: *mut cusparseIndexType_t, csrColIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsrGet") });
        _f(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsrGet(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64, csrRowOffsets: *mut *mut ::core::ffi::c_void, csrColInd: *mut *mut ::core::ffi::c_void, csrValues: *mut *mut ::core::ffi::c_void, csrRowOffsetsType: *mut cusparseIndexType_t, csrColIndType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseCsrGet(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    }
}
pub unsafe fn cusparseCsrSetPointers(spMatDescr: cusparseSpMatDescr_t, csrRowOffsets: *mut ::core::ffi::c_void, csrColInd: *mut ::core::ffi::c_void, csrValues: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsrSetPointers") });
        _f(spMatDescr, csrRowOffsets, csrColInd, csrValues)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsrSetPointers(spMatDescr: cusparseSpMatDescr_t, csrRowOffsets: *mut ::core::ffi::c_void, csrColInd: *mut ::core::ffi::c_void, csrValues: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCsrSetPointers(spMatDescr, csrRowOffsets, csrColInd, csrValues)
    }
}
pub unsafe fn cusparseCsrSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, offsetsBatchStride: i64, columnsValuesBatchStride: i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, ::core::ffi::c_int, i64, i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsrSetStridedBatch") });
        _f(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsrSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int, offsetsBatchStride: i64, columnsValuesBatchStride: i64) -> cusparseStatus_t;
        }
        cusparseCsrSetStridedBatch(spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCsrmvEx(handle: cusparseHandle_t, alg: cusparseAlgMode_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphatype: cudaDataType, descrA: cusparseMatDescr_t, csrValA: *const ::core::ffi::c_void, csrValAtype: cudaDataType, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, x: *const ::core::ffi::c_void, xtype: cudaDataType, beta: *const ::core::ffi::c_void, betatype: cudaDataType, y: *mut ::core::ffi::c_void, ytype: cudaDataType, executiontype: cudaDataType, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, cusparseMatDescr_t, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, cudaDataType, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsrmvEx") });
        _f(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, buffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsrmvEx(handle: cusparseHandle_t, alg: cusparseAlgMode_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphatype: cudaDataType, descrA: cusparseMatDescr_t, csrValA: *const ::core::ffi::c_void, csrValAtype: cudaDataType, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, x: *const ::core::ffi::c_void, xtype: cudaDataType, beta: *const ::core::ffi::c_void, betatype: cudaDataType, y: *mut ::core::ffi::c_void, ytype: cudaDataType, executiontype: cudaDataType, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseCsrmvEx(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, buffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseCsrmvEx_bufferSize(handle: cusparseHandle_t, alg: cusparseAlgMode_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphatype: cudaDataType, descrA: cusparseMatDescr_t, csrValA: *const ::core::ffi::c_void, csrValAtype: cudaDataType, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, x: *const ::core::ffi::c_void, xtype: cudaDataType, beta: *const ::core::ffi::c_void, betatype: cudaDataType, y: *mut ::core::ffi::c_void, ytype: cudaDataType, executiontype: cudaDataType, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, cusparseMatDescr_t, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_void, cudaDataType, *const ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void, cudaDataType, cudaDataType, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseCsrmvEx_bufferSize") });
        _f(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseCsrmvEx_bufferSize(handle: cusparseHandle_t, alg: cusparseAlgMode_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const ::core::ffi::c_void, alphatype: cudaDataType, descrA: cusparseMatDescr_t, csrValA: *const ::core::ffi::c_void, csrValAtype: cudaDataType, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, x: *const ::core::ffi::c_void, xtype: cudaDataType, beta: *const ::core::ffi::c_void, betatype: cudaDataType, y: *mut ::core::ffi::c_void, ytype: cudaDataType, executiontype: cudaDataType, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseCsrmvEx_bufferSize(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype, executiontype, bufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const f64, xVal: *const f64, xInd: *const ::core::ffi::c_int, y: *mut f64, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f64, *const f64, *const ::core::ffi::c_int, *mut f64, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDaxpyi") });
        _f(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const f64, xVal: *const f64, xInd: *const ::core::ffi::c_int, y: *mut f64, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
}
pub unsafe fn cusparseDbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseDbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsric02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsric02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
}
pub unsafe fn cusparseDbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsric02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrilu02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrilu02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrilu02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut f64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f64) -> cusparseStatus_t;
        }
        cusparseDbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
pub unsafe fn cusparseDbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrmm") });
        _f(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cusparseDbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f64, beta: *const f64, y: *mut f64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *mut f64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrmv") });
        _f(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f64, beta: *const f64, y: *mut f64) -> cusparseStatus_t;
        }
        cusparseDbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
pub unsafe fn cusparseDbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrsm2_analysis") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrsm2_bufferSize") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const f64, ldb: ::core::ffi::c_int, X: *mut f64, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *const f64, ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrsm2_solve") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const f64, ldb: ::core::ffi::c_int, X: *mut f64, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrsv2_analysis") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrsv2_bufferSize") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const f64, x: *mut f64, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *const f64, *mut f64, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrsv2_solve") });
        _f(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const f64, x: *mut f64, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseDbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f64, beta: *const f64, y: *mut f64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *mut f64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDbsrxmv") });
        _f(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f64, beta: *const f64, y: *mut f64) -> cusparseStatus_t;
        }
        cusparseDbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const f64, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsc2dense") });
        _f(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const f64, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
}
pub unsafe fn cusparseDcsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f64, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsr2bsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f64, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
}
pub unsafe fn cusparseDcsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut f64, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: f64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, f64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsr2csr_compress") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut f64, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: f64) -> cusparseStatus_t;
        }
        cusparseDcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
}
pub unsafe fn cusparseDcsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsr2csru") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsr2dense") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut f64, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
}
pub unsafe fn cusparseDcsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f64, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsr2gebsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f64, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
}
pub unsafe fn cusparseDcsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsr2gebsr_bufferSize") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDcsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f64, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseColorInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrcolor") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f64, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t;
        }
        cusparseDcsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
}
pub unsafe fn cusparseDcsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f64, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f64, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrgeam2") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f64, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f64, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseDcsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f64, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f64, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrgeam2_bufferSizeExt") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f64, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f64, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f64, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f64, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const f64, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csrgemm2Info_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrgemm2") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f64, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f64, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const f64, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f64, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrgemm2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrgemm2_bufferSizeExt") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f64, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDcsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsric02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDcsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsric02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDcsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsric02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDcsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrilu02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDcsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrilu02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseDcsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrilu02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDcsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut f64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f64) -> cusparseStatus_t;
        }
        cusparseDcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsm2_analysis") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsm2_bufferSizeExt") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f64, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsm2_solve") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut f64, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsv2_analysis") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsv2_bufferSize") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsv2_bufferSizeExt") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseDcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDcsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const f64, x: *mut f64, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *const f64, *mut f64, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsrsv2_solve") });
        _f(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const f64, x: *mut f64, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseDcsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsru2csr") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDcsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
pub unsafe fn cusparseDcsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDcsru2csr_bufferSizeExt") });
        _f(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDcsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut f64, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDcsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f64, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut f64, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDdense2csc") });
        _f(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f64, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut f64, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f64, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut f64, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDdense2csr") });
        _f(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f64, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut f64, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
}
#[cfg(any(feature = "cuda-11040"))]
pub unsafe fn cusparseDenseToSparse_analysis(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_analysis") });
        _f(handle, matA, matB, alg, buffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_analysis(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_analysis(handle, matA, matB, alg, buffer)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDenseToSparse_analysis(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_analysis") });
        _f(handle, matA, matB, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_analysis(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_analysis(handle, matA, matB, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDenseToSparse_analysis(handle: cusparseHandle_t, matA: cusparseConstDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_analysis") });
        _f(handle, matA, matB, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_analysis(handle: cusparseHandle_t, matA: cusparseConstDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_analysis(handle, matA, matB, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDenseToSparse_bufferSize(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_bufferSize") });
        _f(handle, matA, matB, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_bufferSize(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_bufferSize(handle, matA, matB, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDenseToSparse_bufferSize(handle: cusparseHandle_t, matA: cusparseConstDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_bufferSize") });
        _f(handle, matA, matB, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_bufferSize(handle: cusparseHandle_t, matA: cusparseConstDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_bufferSize(handle, matA, matB, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-11040"))]
pub unsafe fn cusparseDenseToSparse_convert(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_convert") });
        _f(handle, matA, matB, alg, buffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_convert(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_convert(handle, matA, matB, alg, buffer)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDenseToSparse_convert(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_convert") });
        _f(handle, matA, matB, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_convert(handle: cusparseHandle_t, matA: cusparseDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_convert(handle, matA, matB, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDenseToSparse_convert(handle: cusparseHandle_t, matA: cusparseConstDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstDnMatDescr_t, cusparseSpMatDescr_t, cusparseDenseToSparseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDenseToSparse_convert") });
        _f(handle, matA, matB, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDenseToSparse_convert(handle: cusparseHandle_t, matA: cusparseConstDnMatDescr_t, matB: cusparseSpMatDescr_t, alg: cusparseDenseToSparseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDenseToSparse_convert(handle, matA, matB, alg, externalBuffer)
    }
}
pub unsafe fn cusparseDestroy(handle: cusparseHandle_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroy") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroy(handle: cusparseHandle_t) -> cusparseStatus_t;
        }
        cusparseDestroy(handle)
    }
}
pub unsafe fn cusparseDestroyBsric02Info(info: bsric02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(bsric02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyBsric02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyBsric02Info(info: bsric02Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyBsric02Info(info)
    }
}
pub unsafe fn cusparseDestroyBsrilu02Info(info: bsrilu02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(bsrilu02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyBsrilu02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyBsrilu02Info(info: bsrilu02Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyBsrilu02Info(info)
    }
}
pub unsafe fn cusparseDestroyBsrsm2Info(info: bsrsm2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(bsrsm2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyBsrsm2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyBsrsm2Info(info: bsrsm2Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyBsrsm2Info(info)
    }
}
pub unsafe fn cusparseDestroyBsrsv2Info(info: bsrsv2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(bsrsv2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyBsrsv2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyBsrsv2Info(info: bsrsv2Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyBsrsv2Info(info)
    }
}
pub unsafe fn cusparseDestroyColorInfo(info: cusparseColorInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseColorInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyColorInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyColorInfo(info: cusparseColorInfo_t) -> cusparseStatus_t;
        }
        cusparseDestroyColorInfo(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroyCsrgemm2Info(info: csrgemm2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csrgemm2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyCsrgemm2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyCsrgemm2Info(info: csrgemm2Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyCsrgemm2Info(info)
    }
}
pub unsafe fn cusparseDestroyCsric02Info(info: csric02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csric02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyCsric02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyCsric02Info(info: csric02Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyCsric02Info(info)
    }
}
pub unsafe fn cusparseDestroyCsrilu02Info(info: csrilu02Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csrilu02Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyCsrilu02Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyCsrilu02Info(info: csrilu02Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyCsrilu02Info(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroyCsrsm2Info(info: csrsm2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csrsm2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyCsrsm2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyCsrsm2Info(info: csrsm2Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyCsrsm2Info(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroyCsrsv2Info(info: csrsv2Info_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csrsv2Info_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyCsrsv2Info") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyCsrsv2Info(info: csrsv2Info_t) -> cusparseStatus_t;
        }
        cusparseDestroyCsrsv2Info(info)
    }
}
pub unsafe fn cusparseDestroyCsru2csrInfo(info: csru2csrInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(csru2csrInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyCsru2csrInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyCsru2csrInfo(info: csru2csrInfo_t) -> cusparseStatus_t;
        }
        cusparseDestroyCsru2csrInfo(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroyDnMat(dnMatDescr: cusparseDnMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyDnMat") });
        _f(dnMatDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyDnMat(dnMatDescr: cusparseDnMatDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroyDnMat(dnMatDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDestroyDnMat(dnMatDescr: cusparseConstDnMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyDnMat") });
        _f(dnMatDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyDnMat(dnMatDescr: cusparseConstDnMatDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroyDnMat(dnMatDescr)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroyDnVec(dnVecDescr: cusparseDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyDnVec") });
        _f(dnVecDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyDnVec(dnVecDescr: cusparseDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroyDnVec(dnVecDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDestroyDnVec(dnVecDescr: cusparseConstDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyDnVec") });
        _f(dnVecDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyDnVec(dnVecDescr: cusparseConstDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroyDnVec(dnVecDescr)
    }
}
pub unsafe fn cusparseDestroyMatDescr(descrA: cusparseMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyMatDescr") });
        _f(descrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyMatDescr(descrA: cusparseMatDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroyMatDescr(descrA)
    }
}
pub unsafe fn cusparseDestroyPruneInfo(info: pruneInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(pruneInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroyPruneInfo") });
        _f(info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroyPruneInfo(info: pruneInfo_t) -> cusparseStatus_t;
        }
        cusparseDestroyPruneInfo(info)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroySpMat(spMatDescr: cusparseSpMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroySpMat") });
        _f(spMatDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroySpMat(spMatDescr: cusparseSpMatDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroySpMat(spMatDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDestroySpMat(spMatDescr: cusparseConstSpMatDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroySpMat") });
        _f(spMatDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroySpMat(spMatDescr: cusparseConstSpMatDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroySpMat(spMatDescr)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDestroySpVec(spVecDescr: cusparseSpVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroySpVec") });
        _f(spVecDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroySpVec(spVecDescr: cusparseSpVecDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroySpVec(spVecDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDestroySpVec(spVecDescr: cusparseConstSpVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDestroySpVec") });
        _f(spVecDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDestroySpVec(spVecDescr: cusparseConstSpVecDescr_t) -> cusparseStatus_t;
        }
        cusparseDestroySpVec(spVecDescr)
    }
}
pub unsafe fn cusparseDgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgebsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseDgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut f64, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseAction_t, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgebsr2gebsc") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut f64, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseDgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgebsr2gebsc_bufferSize") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f64, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f64, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgebsr2gebsr") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f64, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
}
pub unsafe fn cusparseDgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgebsr2gebsr_bufferSize") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f64, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, cscValB: *const f64, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, *mut f64, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgemmi") });
        _f(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, cscValB: *const f64, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const f64, C: *mut f64, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
}
pub unsafe fn cusparseDgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const f64, xInd: *const ::core::ffi::c_int, beta: *const f64, y: *mut f64, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *const f64, *mut f64, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgemvi") });
        _f(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f64, A: *const f64, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const f64, xInd: *const ::core::ffi::c_int, beta: *const f64, y: *mut f64, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseDgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgemvi_bufferSize") });
        _f(handle, transA, m, n, nnz, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    }
}
pub unsafe fn cusparseDgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut f64, dl: *mut f64, d: *mut f64, du: *mut f64, dw: *mut f64, x: *mut f64, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, *mut f64, *mut f64, *mut f64, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgpsvInterleavedBatch") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut f64, dl: *mut f64, d: *mut f64, du: *mut f64, dw: *mut f64, x: *mut f64, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseDgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const f64, dl: *const f64, d: *const f64, du: *const f64, dw: *const f64, x: *const f64, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *const f64, *const f64, *const f64, *const f64, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgpsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const f64, dl: *const f64, d: *const f64, du: *const f64, dw: *const f64, x: *const f64, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const f64, xVal: *mut f64, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f64, *mut f64, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgthr") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const f64, xVal: *mut f64, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut f64, xVal: *mut f64, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut f64, *mut f64, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgthrz") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut f64, xVal: *mut f64, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseDgthrz(handle, nnz, y, xVal, xInd, idxBase)
    }
}
pub unsafe fn cusparseDgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *mut f64, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *const f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsv2") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *mut f64, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseDgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, x: *mut f64, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f64, *const f64, *const f64, *mut f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsv2StridedBatch") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, x: *mut f64, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
}
pub unsafe fn cusparseDgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, x: *const f64, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f64, *const f64, *const f64, *const f64, ::core::ffi::c_int, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsv2StridedBatch_bufferSizeExt") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, x: *const f64, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseDgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *const f64, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *const f64, *const f64, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsv2_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *const f64, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseDgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *mut f64, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *const f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsv2_nopivot") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *mut f64, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseDgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *const f64, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *const f64, *const f64, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsv2_nopivot_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, B: *const f64, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseDgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut f64, d: *mut f64, du: *mut f64, x: *mut f64, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f64, *mut f64, *mut f64, *mut f64, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsvInterleavedBatch") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut f64, d: *mut f64, du: *mut f64, x: *mut f64, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseDgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, x: *const f64, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, *const f64, *const f64, *const f64, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDgtsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const f64, d: *const f64, du: *const f64, x: *const f64, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDnMatGet(dnMatDescr: cusparseDnMatDescr_t, rows: *mut i64, cols: *mut i64, ld: *mut i64, values: *mut *mut ::core::ffi::c_void, type_: *mut cudaDataType, order: *mut cusparseOrder_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnMatDescr_t, *mut i64, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut cudaDataType, *mut cusparseOrder_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnMatGet") });
        _f(dnMatDescr, rows, cols, ld, values, type_, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnMatGet(dnMatDescr: cusparseDnMatDescr_t, rows: *mut i64, cols: *mut i64, ld: *mut i64, values: *mut *mut ::core::ffi::c_void, type_: *mut cudaDataType, order: *mut cusparseOrder_t) -> cusparseStatus_t;
        }
        cusparseDnMatGet(dnMatDescr, rows, cols, ld, values, type_, order)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDnMatGetStridedBatch(dnMatDescr: cusparseDnMatDescr_t, batchCount: *mut ::core::ffi::c_int, batchStride: *mut i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnMatDescr_t, *mut ::core::ffi::c_int, *mut i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnMatGetStridedBatch") });
        _f(dnMatDescr, batchCount, batchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnMatGetStridedBatch(dnMatDescr: cusparseDnMatDescr_t, batchCount: *mut ::core::ffi::c_int, batchStride: *mut i64) -> cusparseStatus_t;
        }
        cusparseDnMatGetStridedBatch(dnMatDescr, batchCount, batchStride)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseDnMatGetStridedBatch(dnMatDescr: cusparseConstDnMatDescr_t, batchCount: *mut ::core::ffi::c_int, batchStride: *mut i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstDnMatDescr_t, *mut ::core::ffi::c_int, *mut i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnMatGetStridedBatch") });
        _f(dnMatDescr, batchCount, batchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnMatGetStridedBatch(dnMatDescr: cusparseConstDnMatDescr_t, batchCount: *mut ::core::ffi::c_int, batchStride: *mut i64) -> cusparseStatus_t;
        }
        cusparseDnMatGetStridedBatch(dnMatDescr, batchCount, batchStride)
    }
}
pub unsafe fn cusparseDnMatGetValues(dnMatDescr: cusparseDnMatDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnMatDescr_t, *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnMatGetValues") });
        _f(dnMatDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnMatGetValues(dnMatDescr: cusparseDnMatDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDnMatGetValues(dnMatDescr, values)
    }
}
pub unsafe fn cusparseDnMatSetStridedBatch(dnMatDescr: cusparseDnMatDescr_t, batchCount: ::core::ffi::c_int, batchStride: i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnMatDescr_t, ::core::ffi::c_int, i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnMatSetStridedBatch") });
        _f(dnMatDescr, batchCount, batchStride)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnMatSetStridedBatch(dnMatDescr: cusparseDnMatDescr_t, batchCount: ::core::ffi::c_int, batchStride: i64) -> cusparseStatus_t;
        }
        cusparseDnMatSetStridedBatch(dnMatDescr, batchCount, batchStride)
    }
}
pub unsafe fn cusparseDnMatSetValues(dnMatDescr: cusparseDnMatDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnMatDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnMatSetValues") });
        _f(dnMatDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnMatSetValues(dnMatDescr: cusparseDnMatDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDnMatSetValues(dnMatDescr, values)
    }
}
pub unsafe fn cusparseDnVecGet(dnVecDescr: cusparseDnVecDescr_t, size: *mut i64, values: *mut *mut ::core::ffi::c_void, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnVecDescr_t, *mut i64, *mut *mut ::core::ffi::c_void, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnVecGet") });
        _f(dnVecDescr, size, values, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnVecGet(dnVecDescr: cusparseDnVecDescr_t, size: *mut i64, values: *mut *mut ::core::ffi::c_void, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseDnVecGet(dnVecDescr, size, values, valueType)
    }
}
pub unsafe fn cusparseDnVecGetValues(dnVecDescr: cusparseDnVecDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnVecDescr_t, *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnVecGetValues") });
        _f(dnVecDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnVecGetValues(dnVecDescr: cusparseDnVecDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDnVecGetValues(dnVecDescr, values)
    }
}
pub unsafe fn cusparseDnVecSetValues(dnVecDescr: cusparseDnVecDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseDnVecDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnVecSetValues") });
        _f(dnVecDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnVecSetValues(dnVecDescr: cusparseDnVecDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDnVecSetValues(dnVecDescr, values)
    }
}
pub unsafe fn cusparseDnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f64, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnnz") });
        _f(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f64, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
}
pub unsafe fn cusparseDnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: f64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, f64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDnnz_compress") });
        _f(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: f64) -> cusparseStatus_t;
        }
        cusparseDnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
}
pub unsafe fn cusparseDpruneCsr2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneCsr2csr") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneCsr2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneCsr2csr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseDpruneCsr2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneCsr2csrByPercentage") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneCsr2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneCsr2csrByPercentage(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
pub unsafe fn cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, pruneInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneCsr2csrByPercentage_bufferSizeExt") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDpruneCsr2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneCsr2csrNnz") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneCsr2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneCsr2csrNnz(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
}
pub unsafe fn cusparseDpruneCsr2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneCsr2csrNnzByPercentage") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneCsr2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneCsr2csrNnzByPercentage(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
}
pub unsafe fn cusparseDpruneCsr2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneCsr2csr_bufferSizeExt") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneCsr2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f64, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDpruneCsr2csr_bufferSizeExt(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDpruneDense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneDense2csr") });
        _f(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneDense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneDense2csr(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseDpruneDense2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut f64, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneDense2csrByPercentage") });
        _f(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneDense2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneDense2csrByPercentage(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
pub unsafe fn cusparseDpruneDense2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, f32, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, pruneInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneDense2csrByPercentage_bufferSizeExt") });
        _f(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneDense2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDpruneDense2csrByPercentage_bufferSizeExt(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseDpruneDense2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneDense2csrNnz") });
        _f(handle, m, n, A, lda, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneDense2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneDense2csrNnz(handle, m, n, A, lda, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
}
pub unsafe fn cusparseDpruneDense2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneDense2csrNnzByPercentage") });
        _f(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneDense2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseDpruneDense2csrNnzByPercentage(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
}
pub unsafe fn cusparseDpruneDense2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f64, ::core::ffi::c_int, *const f64, cusparseMatDescr_t, *const f64, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDpruneDense2csr_bufferSizeExt") });
        _f(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDpruneDense2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f64, lda: ::core::ffi::c_int, threshold: *const f64, descrC: cusparseMatDescr_t, csrSortedValC: *const f64, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseDpruneDense2csr_bufferSizeExt(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDroti(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *mut f64, xInd: *const ::core::ffi::c_int, y: *mut f64, c: *const f64, s: *const f64, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut f64, *const ::core::ffi::c_int, *mut f64, *const f64, *const f64, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDroti") });
        _f(handle, nnz, xVal, xInd, y, c, s, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDroti(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *mut f64, xInd: *const ::core::ffi::c_int, y: *mut f64, c: *const f64, s: *const f64, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseDsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const f64, xInd: *const ::core::ffi::c_int, y: *mut f64, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f64, *const ::core::ffi::c_int, *mut f64, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseDsctr") });
        _f(handle, nnz, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseDsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const f64, xInd: *const ::core::ffi::c_int, y: *mut f64, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseDsctr(handle, nnz, xVal, xInd, y, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseGather(handle: cusparseHandle_t, vecY: cusparseDnVecDescr_t, vecX: cusparseSpVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDnVecDescr_t, cusparseSpVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGather") });
        _f(handle, vecY, vecX)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGather(handle: cusparseHandle_t, vecY: cusparseDnVecDescr_t, vecX: cusparseSpVecDescr_t) -> cusparseStatus_t;
        }
        cusparseGather(handle, vecY, vecX)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseGather(handle: cusparseHandle_t, vecY: cusparseConstDnVecDescr_t, vecX: cusparseSpVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstDnVecDescr_t, cusparseSpVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGather") });
        _f(handle, vecY, vecX)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGather(handle: cusparseHandle_t, vecY: cusparseConstDnVecDescr_t, vecX: cusparseSpVecDescr_t) -> cusparseStatus_t;
        }
        cusparseGather(handle, vecY, vecX)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000"))]
pub unsafe fn cusparseGetColorAlgs(info: cusparseColorInfo_t, alg: *mut cusparseColorAlg_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseColorInfo_t, *mut cusparseColorAlg_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetColorAlgs") });
        _f(info, alg)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetColorAlgs(info: cusparseColorInfo_t, alg: *mut cusparseColorAlg_t) -> cusparseStatus_t;
        }
        cusparseGetColorAlgs(info, alg)
    }
}
pub unsafe fn cusparseGetErrorName(status: cusparseStatus_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseStatus_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetErrorName") });
        _f(status)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetErrorName(status: cusparseStatus_t) -> *const ::core::ffi::c_char;
        }
        cusparseGetErrorName(status)
    }
}
pub unsafe fn cusparseGetErrorString(status: cusparseStatus_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseStatus_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetErrorString") });
        _f(status)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetErrorString(status: cusparseStatus_t) -> *const ::core::ffi::c_char;
        }
        cusparseGetErrorString(status)
    }
}
pub unsafe fn cusparseGetMatDiagType(descrA: cusparseMatDescr_t) -> cusparseDiagType_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t) -> cusparseDiagType_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetMatDiagType") });
        _f(descrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetMatDiagType(descrA: cusparseMatDescr_t) -> cusparseDiagType_t;
        }
        cusparseGetMatDiagType(descrA)
    }
}
pub unsafe fn cusparseGetMatFillMode(descrA: cusparseMatDescr_t) -> cusparseFillMode_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t) -> cusparseFillMode_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetMatFillMode") });
        _f(descrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetMatFillMode(descrA: cusparseMatDescr_t) -> cusparseFillMode_t;
        }
        cusparseGetMatFillMode(descrA)
    }
}
pub unsafe fn cusparseGetMatIndexBase(descrA: cusparseMatDescr_t) -> cusparseIndexBase_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t) -> cusparseIndexBase_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetMatIndexBase") });
        _f(descrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetMatIndexBase(descrA: cusparseMatDescr_t) -> cusparseIndexBase_t;
        }
        cusparseGetMatIndexBase(descrA)
    }
}
pub unsafe fn cusparseGetMatType(descrA: cusparseMatDescr_t) -> cusparseMatrixType_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t) -> cusparseMatrixType_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetMatType") });
        _f(descrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetMatType(descrA: cusparseMatDescr_t) -> cusparseMatrixType_t;
        }
        cusparseGetMatType(descrA)
    }
}
pub unsafe fn cusparseGetPointerMode(handle: cusparseHandle_t, mode: *mut cusparsePointerMode_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *mut cusparsePointerMode_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetPointerMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetPointerMode(handle: cusparseHandle_t, mode: *mut cusparsePointerMode_t) -> cusparseStatus_t;
        }
        cusparseGetPointerMode(handle, mode)
    }
}
pub unsafe fn cusparseGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(libraryPropertyType, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetProperty") });
        _f(type_, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseGetProperty(type_, value)
    }
}
pub unsafe fn cusparseGetStream(handle: cusparseHandle_t, streamId: *mut cudaStream_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *mut cudaStream_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetStream") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetStream(handle: cusparseHandle_t, streamId: *mut cudaStream_t) -> cusparseStatus_t;
        }
        cusparseGetStream(handle, streamId)
    }
}
pub unsafe fn cusparseGetVersion(handle: cusparseHandle_t, version: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseGetVersion") });
        _f(handle, version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseGetVersion(handle: cusparseHandle_t, version: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseGetVersion(handle, version)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseLoggerForceDisable() -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseLoggerForceDisable") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseLoggerForceDisable() -> cusparseStatus_t;
        }
        cusparseLoggerForceDisable()
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseLoggerOpenFile(logFile: *const ::core::ffi::c_char) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseLoggerOpenFile") });
        _f(logFile)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseLoggerOpenFile(logFile: *const ::core::ffi::c_char) -> cusparseStatus_t;
        }
        cusparseLoggerOpenFile(logFile)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseLoggerSetCallback(callback: cusparseLoggerCallback_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseLoggerCallback_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseLoggerSetCallback") });
        _f(callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseLoggerSetCallback(callback: cusparseLoggerCallback_t) -> cusparseStatus_t;
        }
        cusparseLoggerSetCallback(callback)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseLoggerSetFile(file: *mut FILE) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut FILE) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseLoggerSetFile") });
        _f(file)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseLoggerSetFile(file: *mut FILE) -> cusparseStatus_t;
        }
        cusparseLoggerSetFile(file)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseLoggerSetLevel(level: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseLoggerSetLevel") });
        _f(level)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseLoggerSetLevel(level: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseLoggerSetLevel(level)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseLoggerSetMask(mask: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseLoggerSetMask") });
        _f(mask)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseLoggerSetMask(mask: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseLoggerSetMask(mask)
    }
}
pub unsafe fn cusparseRot(handle: cusparseHandle_t, c_coeff: *const ::core::ffi::c_void, s_coeff: *const ::core::ffi::c_void, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, cusparseSpVecDescr_t, cusparseDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseRot") });
        _f(handle, c_coeff, s_coeff, vecX, vecY)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseRot(handle: cusparseHandle_t, c_coeff: *const ::core::ffi::c_void, s_coeff: *const ::core::ffi::c_void, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseRot(handle, c_coeff, s_coeff, vecX, vecY)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSDDMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSDDMM") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSDDMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSDDMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSDDMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstDnMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSDDMM") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSDDMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstDnMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSDDMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSDDMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSDDMM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSDDMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSDDMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSDDMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstDnMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSDDMM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSDDMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstDnMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSDDMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSDDMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSDDMM_preprocess") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSDDMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseDnMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSDDMM_preprocess(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSDDMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstDnMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstDnMatDescr_t, cusparseConstDnMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSDDMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSDDMM_preprocess") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSDDMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstDnMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSDDMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSDDMM_preprocess(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const f32, xVal: *const f32, xInd: *const ::core::ffi::c_int, y: *mut f32, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f32, *const f32, *const ::core::ffi::c_int, *mut f32, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSaxpyi") });
        _f(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const f32, xVal: *const f32, xInd: *const ::core::ffi::c_int, y: *mut f32, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
}
pub unsafe fn cusparseSbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseSbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsric02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsric02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
}
pub unsafe fn cusparseSbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsric02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrilu02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrilu02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrilu02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f32) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut f32) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f32) -> cusparseStatus_t;
        }
        cusparseSbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
pub unsafe fn cusparseSbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrmm") });
        _f(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cusparseSbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f32, beta: *const f32, y: *mut f32) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *mut f32) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrmv") });
        _f(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f32, beta: *const f32, y: *mut f32) -> cusparseStatus_t;
        }
        cusparseSbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
pub unsafe fn cusparseSbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrsm2_analysis") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrsm2_bufferSize") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const f32, ldb: ::core::ffi::c_int, X: *mut f32, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *const f32, ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrsm2_solve") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const f32, ldb: ::core::ffi::c_int, X: *mut f32, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrsv2_analysis") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrsv2_bufferSize") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const f32, x: *mut f32, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *const f32, *mut f32, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrsv2_solve") });
        _f(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const f32, x: *mut f32, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseSbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f32, beta: *const f32, y: *mut f32) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *mut f32) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSbsrxmv") });
        _f(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const f32, beta: *const f32, y: *mut f32) -> cusparseStatus_t;
        }
        cusparseSbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScatter(handle: cusparseHandle_t, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScatter") });
        _f(handle, vecX, vecY)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScatter(handle: cusparseHandle_t, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseScatter(handle, vecX, vecY)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseScatter(handle: cusparseHandle_t, vecX: cusparseConstSpVecDescr_t, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstSpVecDescr_t, cusparseDnVecDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScatter") });
        _f(handle, vecX, vecY)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScatter(handle: cusparseHandle_t, vecX: cusparseConstSpVecDescr_t, vecY: cusparseDnVecDescr_t) -> cusparseStatus_t;
        }
        cusparseScatter(handle, vecX, vecY)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const f32, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsc2dense") });
        _f(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const f32, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
}
pub unsafe fn cusparseScsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f32, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsr2bsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f32, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
}
pub unsafe fn cusparseScsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut f32, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: f32) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, f32) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsr2csr_compress") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut f32, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: f32) -> cusparseStatus_t;
        }
        cusparseScsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
}
pub unsafe fn cusparseScsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsr2csru") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsr2dense") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut f32, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
}
pub unsafe fn cusparseScsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f32, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsr2gebsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f32, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
}
pub unsafe fn cusparseScsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsr2gebsr_bufferSize") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseScsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f32, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseColorInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrcolor") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f32, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t;
        }
        cusparseScsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
}
pub unsafe fn cusparseScsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f32, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f32, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrgeam2") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f32, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f32, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseScsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f32, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f32, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrgeam2_bufferSizeExt") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const f32, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f32, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f32, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f32, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const f32, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csrgemm2Info_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrgemm2") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const f32, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f32, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const f32, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f32, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrgemm2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrgemm2_bufferSizeExt") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const f32, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseScsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsric02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseScsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsric02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseScsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsric02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseScsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrilu02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseScsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrilu02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseScsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrilu02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseScsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f32) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut f32) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut f32) -> cusparseStatus_t;
        }
        cusparseScsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsm2_analysis") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsm2_bufferSizeExt") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const f32, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsm2_solve") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut f32, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsv2_analysis") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsv2_bufferSize") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsv2_bufferSizeExt") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseScsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseScsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const f32, x: *mut f32, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *const f32, *mut f32, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsrsv2_solve") });
        _f(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const f32, x: *mut f32, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseScsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsru2csr") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseScsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
pub unsafe fn cusparseScsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseScsru2csr_bufferSizeExt") });
        _f(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseScsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut f32, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseScsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f32, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut f32, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSdense2csc") });
        _f(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f32, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut f32, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f32, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut f32, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSdense2csr") });
        _f(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f32, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut f32, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000"))]
pub unsafe fn cusparseSetColorAlgs(info: cusparseColorInfo_t, alg: cusparseColorAlg_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseColorInfo_t, cusparseColorAlg_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetColorAlgs") });
        _f(info, alg)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetColorAlgs(info: cusparseColorInfo_t, alg: cusparseColorAlg_t) -> cusparseStatus_t;
        }
        cusparseSetColorAlgs(info, alg)
    }
}
pub unsafe fn cusparseSetMatDiagType(descrA: cusparseMatDescr_t, diagType: cusparseDiagType_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t, cusparseDiagType_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetMatDiagType") });
        _f(descrA, diagType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetMatDiagType(descrA: cusparseMatDescr_t, diagType: cusparseDiagType_t) -> cusparseStatus_t;
        }
        cusparseSetMatDiagType(descrA, diagType)
    }
}
pub unsafe fn cusparseSetMatFillMode(descrA: cusparseMatDescr_t, fillMode: cusparseFillMode_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t, cusparseFillMode_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetMatFillMode") });
        _f(descrA, fillMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetMatFillMode(descrA: cusparseMatDescr_t, fillMode: cusparseFillMode_t) -> cusparseStatus_t;
        }
        cusparseSetMatFillMode(descrA, fillMode)
    }
}
pub unsafe fn cusparseSetMatIndexBase(descrA: cusparseMatDescr_t, base: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetMatIndexBase") });
        _f(descrA, base)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetMatIndexBase(descrA: cusparseMatDescr_t, base: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSetMatIndexBase(descrA, base)
    }
}
pub unsafe fn cusparseSetMatType(descrA: cusparseMatDescr_t, type_: cusparseMatrixType_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseMatDescr_t, cusparseMatrixType_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetMatType") });
        _f(descrA, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetMatType(descrA: cusparseMatDescr_t, type_: cusparseMatrixType_t) -> cusparseStatus_t;
        }
        cusparseSetMatType(descrA, type_)
    }
}
pub unsafe fn cusparseSetPointerMode(handle: cusparseHandle_t, mode: cusparsePointerMode_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparsePointerMode_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetPointerMode") });
        _f(handle, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetPointerMode(handle: cusparseHandle_t, mode: cusparsePointerMode_t) -> cusparseStatus_t;
        }
        cusparseSetPointerMode(handle, mode)
    }
}
pub unsafe fn cusparseSetStream(handle: cusparseHandle_t, streamId: cudaStream_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cudaStream_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSetStream") });
        _f(handle, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSetStream(handle: cusparseHandle_t, streamId: cudaStream_t) -> cusparseStatus_t;
        }
        cusparseSetStream(handle, streamId)
    }
}
pub unsafe fn cusparseSgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgebsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseSgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut f32, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseAction_t, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgebsr2gebsc") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut f32, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseSgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgebsr2gebsc_bufferSize") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const f32, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f32, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut f32, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgebsr2gebsr") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut f32, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
}
pub unsafe fn cusparseSgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgebsr2gebsr_bufferSize") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const f32, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, cscValB: *const f32, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, *mut f32, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgemmi") });
        _f(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, cscValB: *const f32, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const f32, C: *mut f32, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
}
pub unsafe fn cusparseSgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const f32, xInd: *const ::core::ffi::c_int, beta: *const f32, y: *mut f32, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *const f32, *mut f32, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgemvi") });
        _f(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const f32, A: *const f32, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const f32, xInd: *const ::core::ffi::c_int, beta: *const f32, y: *mut f32, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseSgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgemvi_bufferSize") });
        _f(handle, transA, m, n, nnz, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    }
}
pub unsafe fn cusparseSgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut f32, dl: *mut f32, d: *mut f32, du: *mut f32, dw: *mut f32, x: *mut f32, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgpsvInterleavedBatch") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut f32, dl: *mut f32, d: *mut f32, du: *mut f32, dw: *mut f32, x: *mut f32, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseSgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const f32, dl: *const f32, d: *const f32, du: *const f32, dw: *const f32, x: *const f32, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *const f32, *const f32, *const f32, *const f32, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgpsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const f32, dl: *const f32, d: *const f32, du: *const f32, dw: *const f32, x: *const f32, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const f32, xVal: *mut f32, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f32, *mut f32, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgthr") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const f32, xVal: *mut f32, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut f32, xVal: *mut f32, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut f32, *mut f32, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgthrz") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut f32, xVal: *mut f32, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSgthrz(handle, nnz, y, xVal, xInd, idxBase)
    }
}
pub unsafe fn cusparseSgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *mut f32, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *const f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsv2") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *mut f32, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseSgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, x: *mut f32, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f32, *const f32, *const f32, *mut f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsv2StridedBatch") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, x: *mut f32, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
}
pub unsafe fn cusparseSgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, x: *const f32, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f32, *const f32, *const f32, *const f32, ::core::ffi::c_int, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsv2StridedBatch_bufferSizeExt") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, x: *const f32, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseSgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *const f32, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *const f32, *const f32, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsv2_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *const f32, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseSgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *mut f32, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *const f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsv2_nopivot") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *mut f32, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseSgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *const f32, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *const f32, *const f32, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsv2_nopivot_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, B: *const f32, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseSgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut f32, d: *mut f32, du: *mut f32, x: *mut f32, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut f32, *mut f32, *mut f32, *mut f32, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsvInterleavedBatch") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut f32, d: *mut f32, du: *mut f32, x: *mut f32, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseSgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, x: *const f32, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, *const f32, *const f32, *const f32, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSgtsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const f32, d: *const f32, du: *const f32, x: *const f32, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f32, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSnnz") });
        _f(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const f32, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
}
pub unsafe fn cusparseSnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: f32) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, f32) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSnnz_compress") });
        _f(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: f32) -> cusparseStatus_t;
        }
        cusparseSnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEAM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEAMAlg_t, spgeamDescr: cusparseSpGEAMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEAMAlg_t, cusparseSpGEAMDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEAM") });
        _f(handle, opA, opB, alpha, matA, beta, matB, matC, computeType, alg, spgeamDescr, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEAM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEAMAlg_t, spgeamDescr: cusparseSpGEAMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEAM(handle, opA, opB, alpha, matA, beta, matB, matC, computeType, alg, spgeamDescr, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEAM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEAMAlg_t, spgeamDescr: cusparseSpGEAMDescr_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEAMAlg_t, cusparseSpGEAMDescr_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEAM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, beta, matB, matC, computeType, alg, spgeamDescr, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEAM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEAMAlg_t, spgeamDescr: cusparseSpGEAMDescr_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpGEAM_bufferSize(handle, opA, opB, alpha, matA, beta, matB, matC, computeType, alg, spgeamDescr, bufferSize)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEAM_createDescr(descr: *mut cusparseSpGEAMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpGEAMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEAM_createDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEAM_createDescr(descr: *mut cusparseSpGEAMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEAM_createDescr(descr)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEAM_destroyDescr(descr: cusparseSpGEAMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpGEAMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEAM_destroyDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEAM_destroyDescr(descr: cusparseSpGEAMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEAM_destroyDescr(descr)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEAM_nnz(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEAMAlg_t, spgeamDescr: cusparseSpGEAMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEAMAlg_t, cusparseSpGEAMDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEAM_nnz") });
        _f(handle, opA, opB, alpha, matA, beta, matB, matC, computeType, alg, spgeamDescr, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEAM_nnz(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEAMAlg_t, spgeamDescr: cusparseSpGEAMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEAM_nnz(handle, opA, opB, alpha, matA, beta, matB, matC, computeType, alg, spgeamDescr, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMM_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_compute") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMM_compute(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMM_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_compute") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMM_compute(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize2, externalBuffer2)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMM_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_copy") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEMM_copy(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMM_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_copy") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEMM_copy(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
}
pub unsafe fn cusparseSpGEMM_createDescr(descr: *mut cusparseSpGEMMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_createDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_createDescr(descr: *mut cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEMM_createDescr(descr)
    }
}
pub unsafe fn cusparseSpGEMM_destroyDescr(descr: cusparseSpGEMMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_destroyDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_destroyDescr(descr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEMM_destroyDescr(descr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMM_estimateMemory(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, chunk_fraction: f32, bufferSize3: *mut usize, externalBuffer3: *mut ::core::ffi::c_void, bufferSize2: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, f32, *mut usize, *mut ::core::ffi::c_void, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_estimateMemory") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, chunk_fraction, bufferSize3, externalBuffer3, bufferSize2)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_estimateMemory(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, chunk_fraction: f32, bufferSize3: *mut usize, externalBuffer3: *mut ::core::ffi::c_void, bufferSize2: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpGEMM_estimateMemory(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, chunk_fraction, bufferSize3, externalBuffer3, bufferSize2)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMM_getNumProducts(spgemmDescr: cusparseSpGEMMDescr_t, num_prods: *mut i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpGEMMDescr_t, *mut i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_getNumProducts") });
        _f(spgemmDescr, num_prods)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_getNumProducts(spgemmDescr: cusparseSpGEMMDescr_t, num_prods: *mut i64) -> cusparseStatus_t;
        }
        cusparseSpGEMM_getNumProducts(spgemmDescr, num_prods)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMM_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_workEstimation") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMM_workEstimation(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMM_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMM_workEstimation") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMM_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMM_workEstimation(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMMreuse_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_compute") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_compute(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMMreuse_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cudaDataType, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_compute") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_compute(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseSpMatDescr_t, computeType: cudaDataType, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_compute(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMMreuse_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize5: *mut usize, externalBuffer5: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_copy") });
        _f(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize5: *mut usize, externalBuffer5: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMMreuse_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize5: *mut usize, externalBuffer5: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_copy") });
        _f(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_copy(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize5: *mut usize, externalBuffer5: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize5, externalBuffer5)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMMreuse_nnz(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void, bufferSize3: *mut usize, externalBuffer3: *mut ::core::ffi::c_void, bufferSize4: *mut usize, externalBuffer4: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void, *mut usize, *mut ::core::ffi::c_void, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_nnz") });
        _f(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2, externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_nnz(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void, bufferSize3: *mut usize, externalBuffer3: *mut ::core::ffi::c_void, bufferSize4: *mut usize, externalBuffer4: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2, externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMMreuse_nnz(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void, bufferSize3: *mut usize, externalBuffer3: *mut ::core::ffi::c_void, bufferSize4: *mut usize, externalBuffer4: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void, *mut usize, *mut ::core::ffi::c_void, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_nnz") });
        _f(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2, externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_nnz(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize2: *mut usize, externalBuffer2: *mut ::core::ffi::c_void, bufferSize3: *mut usize, externalBuffer3: *mut ::core::ffi::c_void, bufferSize4: *mut usize, externalBuffer4: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize2, externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpGEMMreuse_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_workEstimation") });
        _f(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpGEMMreuse_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t, cusparseSpMatDescr_t, cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t, *mut usize, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpGEMMreuse_workEstimation") });
        _f(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpGEMMreuse_workEstimation(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstSpMatDescr_t, matC: cusparseSpMatDescr_t, alg: cusparseSpGEMMAlg_t, spgemmDescr: cusparseSpGEMMDescr_t, bufferSize1: *mut usize, externalBuffer1: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC, alg, spgemmDescr, bufferSize1, externalBuffer1)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMM") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMM") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMM(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMMOp(plan: cusparseSpMMOpPlan_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMMOpPlan_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMMOp") });
        _f(plan, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMMOp(plan: cusparseSpMMOpPlan_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMMOp(plan, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMMOp_createPlan(handle: cusparseHandle_t, plan: *mut cusparseSpMMOpPlan_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMOpAlg_t, addOperationNvvmBuffer: *const ::core::ffi::c_void, addOperationBufferSize: usize, mulOperationNvvmBuffer: *const ::core::ffi::c_void, mulOperationBufferSize: usize, epilogueNvvmBuffer: *const ::core::ffi::c_void, epilogueBufferSize: usize, SpMMWorkspaceSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *mut cusparseSpMMOpPlan_t, cusparseOperation_t, cusparseOperation_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMOpAlg_t, *const ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMMOp_createPlan") });
        _f(handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationNvvmBuffer, addOperationBufferSize, mulOperationNvvmBuffer, mulOperationBufferSize, epilogueNvvmBuffer, epilogueBufferSize, SpMMWorkspaceSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMMOp_createPlan(handle: cusparseHandle_t, plan: *mut cusparseSpMMOpPlan_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMOpAlg_t, addOperationNvvmBuffer: *const ::core::ffi::c_void, addOperationBufferSize: usize, mulOperationNvvmBuffer: *const ::core::ffi::c_void, mulOperationBufferSize: usize, epilogueNvvmBuffer: *const ::core::ffi::c_void, epilogueBufferSize: usize, SpMMWorkspaceSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMMOp_createPlan(handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationNvvmBuffer, addOperationBufferSize, mulOperationNvvmBuffer, mulOperationBufferSize, epilogueNvvmBuffer, epilogueBufferSize, SpMMWorkspaceSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
pub unsafe fn cusparseSpMMOp_createPlan(handle: cusparseHandle_t, plan: *mut cusparseSpMMOpPlan_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMOpAlg_t, addOperationNvvmBuffer: *const ::core::ffi::c_void, addOperationBufferSize: usize, mulOperationNvvmBuffer: *const ::core::ffi::c_void, mulOperationBufferSize: usize, epilogueNvvmBuffer: *const ::core::ffi::c_void, epilogueBufferSize: usize, SpMMWorkspaceSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *mut cusparseSpMMOpPlan_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMOpAlg_t, *const ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMMOp_createPlan") });
        _f(handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationNvvmBuffer, addOperationBufferSize, mulOperationNvvmBuffer, mulOperationBufferSize, epilogueNvvmBuffer, epilogueBufferSize, SpMMWorkspaceSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMMOp_createPlan(handle: cusparseHandle_t, plan: *mut cusparseSpMMOpPlan_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMOpAlg_t, addOperationNvvmBuffer: *const ::core::ffi::c_void, addOperationBufferSize: usize, mulOperationNvvmBuffer: *const ::core::ffi::c_void, mulOperationBufferSize: usize, epilogueNvvmBuffer: *const ::core::ffi::c_void, epilogueBufferSize: usize, SpMMWorkspaceSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMMOp_createPlan(handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationNvvmBuffer, addOperationBufferSize, mulOperationNvvmBuffer, mulOperationBufferSize, epilogueNvvmBuffer, epilogueBufferSize, SpMMWorkspaceSize)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMMOp_createPlan(handle: cusparseHandle_t, plan: *mut cusparseSpMMOpPlan_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMOpAlg_t, addOperationLtoirBuffer: *const ::core::ffi::c_void, addOperationBufferSize: usize, mulOperationLtoirBuffer: *const ::core::ffi::c_void, mulOperationBufferSize: usize, epilogueLtoirBuffer: *const ::core::ffi::c_void, epilogueBufferSize: usize, SpMMWorkspaceSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *mut cusparseSpMMOpPlan_t, cusparseOperation_t, cusparseOperation_t, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMOpAlg_t, *const ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMMOp_createPlan") });
        _f(handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationLtoirBuffer, addOperationBufferSize, mulOperationLtoirBuffer, mulOperationBufferSize, epilogueLtoirBuffer, epilogueBufferSize, SpMMWorkspaceSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMMOp_createPlan(handle: cusparseHandle_t, plan: *mut cusparseSpMMOpPlan_t, opA: cusparseOperation_t, opB: cusparseOperation_t, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMOpAlg_t, addOperationLtoirBuffer: *const ::core::ffi::c_void, addOperationBufferSize: usize, mulOperationLtoirBuffer: *const ::core::ffi::c_void, mulOperationBufferSize: usize, epilogueLtoirBuffer: *const ::core::ffi::c_void, epilogueBufferSize: usize, SpMMWorkspaceSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMMOp_createPlan(handle, plan, opA, opB, matA, matB, matC, computeType, alg, addOperationLtoirBuffer, addOperationBufferSize, mulOperationLtoirBuffer, mulOperationBufferSize, epilogueLtoirBuffer, epilogueBufferSize, SpMMWorkspaceSize)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMMOp_destroyPlan(plan: cusparseSpMMOpPlan_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMMOpPlan_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMMOp_destroyPlan") });
        _f(plan)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMMOp_destroyPlan(plan: cusparseSpMMOpPlan_t) -> cusparseStatus_t;
        }
        cusparseSpMMOp_destroyPlan(plan)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnMatDescr_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMM_preprocess") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMM_preprocess(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, *const ::core::ffi::c_void, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMM_preprocess") });
        _f(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMM_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, beta: *const ::core::ffi::c_void, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpMMAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMM_preprocess(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMV(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMV") });
        _f(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMV(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMV(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMV") });
        _f(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMV(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMV_bufferSize") });
        _f(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMV_bufferSize") });
        _f(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMV_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, *const ::core::ffi::c_void, cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMV_preprocess") });
        _f(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMV_preprocess(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, beta: *const ::core::ffi::c_void, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpMVAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMV_preprocess(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMatGetAttribute(spMatDescr: cusparseSpMatDescr_t, attribute: cusparseSpMatAttribute_t, data: *mut ::core::ffi::c_void, dataSize: usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, cusparseSpMatAttribute_t, *mut ::core::ffi::c_void, usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetAttribute") });
        _f(spMatDescr, attribute, data, dataSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetAttribute(spMatDescr: cusparseSpMatDescr_t, attribute: cusparseSpMatAttribute_t, data: *mut ::core::ffi::c_void, dataSize: usize) -> cusparseStatus_t;
        }
        cusparseSpMatGetAttribute(spMatDescr, attribute, data, dataSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMatGetAttribute(spMatDescr: cusparseConstSpMatDescr_t, attribute: cusparseSpMatAttribute_t, data: *mut ::core::ffi::c_void, dataSize: usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, cusparseSpMatAttribute_t, *mut ::core::ffi::c_void, usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetAttribute") });
        _f(spMatDescr, attribute, data, dataSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetAttribute(spMatDescr: cusparseConstSpMatDescr_t, attribute: cusparseSpMatAttribute_t, data: *mut ::core::ffi::c_void, dataSize: usize) -> cusparseStatus_t;
        }
        cusparseSpMatGetAttribute(spMatDescr, attribute, data, dataSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMatGetFormat(spMatDescr: cusparseSpMatDescr_t, format: *mut cusparseFormat_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut cusparseFormat_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetFormat") });
        _f(spMatDescr, format)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetFormat(spMatDescr: cusparseSpMatDescr_t, format: *mut cusparseFormat_t) -> cusparseStatus_t;
        }
        cusparseSpMatGetFormat(spMatDescr, format)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMatGetFormat(spMatDescr: cusparseConstSpMatDescr_t, format: *mut cusparseFormat_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut cusparseFormat_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetFormat") });
        _f(spMatDescr, format)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetFormat(spMatDescr: cusparseConstSpMatDescr_t, format: *mut cusparseFormat_t) -> cusparseStatus_t;
        }
        cusparseSpMatGetFormat(spMatDescr, format)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMatGetIndexBase(spMatDescr: cusparseSpMatDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetIndexBase") });
        _f(spMatDescr, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetIndexBase(spMatDescr: cusparseSpMatDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSpMatGetIndexBase(spMatDescr, idxBase)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMatGetIndexBase(spMatDescr: cusparseConstSpMatDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetIndexBase") });
        _f(spMatDescr, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetIndexBase(spMatDescr: cusparseConstSpMatDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSpMatGetIndexBase(spMatDescr, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMatGetSize(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut i64, *mut i64, *mut i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetSize") });
        _f(spMatDescr, rows, cols, nnz)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetSize(spMatDescr: cusparseSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64) -> cusparseStatus_t;
        }
        cusparseSpMatGetSize(spMatDescr, rows, cols, nnz)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMatGetSize(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut i64, *mut i64, *mut i64) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetSize") });
        _f(spMatDescr, rows, cols, nnz)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetSize(spMatDescr: cusparseConstSpMatDescr_t, rows: *mut i64, cols: *mut i64, nnz: *mut i64) -> cusparseStatus_t;
        }
        cusparseSpMatGetSize(spMatDescr, rows, cols, nnz)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMatGetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetStridedBatch") });
        _f(spMatDescr, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSpMatGetStridedBatch(spMatDescr, batchCount)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpMatGetStridedBatch(spMatDescr: cusparseConstSpMatDescr_t, batchCount: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpMatDescr_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetStridedBatch") });
        _f(spMatDescr, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetStridedBatch(spMatDescr: cusparseConstSpMatDescr_t, batchCount: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSpMatGetStridedBatch(spMatDescr, batchCount)
    }
}
pub unsafe fn cusparseSpMatGetValues(spMatDescr: cusparseSpMatDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatGetValues") });
        _f(spMatDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatGetValues(spMatDescr: cusparseSpMatDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMatGetValues(spMatDescr, values)
    }
}
pub unsafe fn cusparseSpMatSetAttribute(spMatDescr: cusparseSpMatDescr_t, attribute: cusparseSpMatAttribute_t, data: *mut ::core::ffi::c_void, dataSize: usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, cusparseSpMatAttribute_t, *mut ::core::ffi::c_void, usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatSetAttribute") });
        _f(spMatDescr, attribute, data, dataSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatSetAttribute(spMatDescr: cusparseSpMatDescr_t, attribute: cusparseSpMatAttribute_t, data: *mut ::core::ffi::c_void, dataSize: usize) -> cusparseStatus_t;
        }
        cusparseSpMatSetAttribute(spMatDescr, attribute, data, dataSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpMatSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatSetStridedBatch") });
        _f(spMatDescr, batchCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatSetStridedBatch(spMatDescr: cusparseSpMatDescr_t, batchCount: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseSpMatSetStridedBatch(spMatDescr, batchCount)
    }
}
pub unsafe fn cusparseSpMatSetValues(spMatDescr: cusparseSpMatDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpMatDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpMatSetValues") });
        _f(spMatDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpMatSetValues(spMatDescr: cusparseSpMatDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpMatSetValues(spMatDescr, values)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpSM_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_analysis") });
        _f(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpSM_analysis(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSM_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_analysis") });
        _f(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpSM_analysis(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpSM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpSM_bufferSize(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_bufferSize") });
        _f(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpSM_bufferSize(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr, bufferSize)
    }
}
pub unsafe fn cusparseSpSM_createDescr(descr: *mut cusparseSpSMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpSMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_createDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_createDescr(descr: *mut cusparseSpSMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSM_createDescr(descr)
    }
}
pub unsafe fn cusparseSpSM_destroyDescr(descr: cusparseSpSMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpSMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_destroyDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_destroyDescr(descr: cusparseSpSMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSM_destroyDescr(descr)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpSM_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_solve") });
        _f(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSM_solve(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSM_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t, cusparseSpSMDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_solve") });
        _f(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, opB: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, matB: cusparseConstDnMatDescr_t, matC: cusparseDnMatDescr_t, computeType: cudaDataType, alg: cusparseSpSMAlg_t, spsmDescr: cusparseSpSMDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSM_solve(handle, opA, opB, alpha, matA, matB, matC, computeType, alg, spsmDescr)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSM_updateMatrix(handle: cusparseHandle_t, spsmDescr: cusparseSpSMDescr_t, newValues: *mut ::core::ffi::c_void, updatePart: cusparseSpSMUpdate_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseSpSMDescr_t, *mut ::core::ffi::c_void, cusparseSpSMUpdate_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSM_updateMatrix") });
        _f(handle, spsmDescr, newValues, updatePart)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSM_updateMatrix(handle: cusparseHandle_t, spsmDescr: cusparseSpSMDescr_t, newValues: *mut ::core::ffi::c_void, updatePart: cusparseSpSMUpdate_t) -> cusparseStatus_t;
        }
        cusparseSpSM_updateMatrix(handle, spsmDescr, newValues, updatePart)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpSV_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_analysis") });
        _f(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpSV_analysis(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSV_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_analysis") });
        _f(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_analysis(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpSV_analysis(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpSV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_bufferSize") });
        _f(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpSV_bufferSize(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_bufferSize") });
        _f(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_bufferSize(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpSV_bufferSize(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr, bufferSize)
    }
}
pub unsafe fn cusparseSpSV_createDescr(descr: *mut cusparseSpSVDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cusparseSpSVDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_createDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_createDescr(descr: *mut cusparseSpSVDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSV_createDescr(descr)
    }
}
pub unsafe fn cusparseSpSV_destroyDescr(descr: cusparseSpSVDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpSVDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_destroyDescr") });
        _f(descr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_destroyDescr(descr: cusparseSpSVDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSV_destroyDescr(descr)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpSV_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseSpMatDescr_t, cusparseDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_solve") });
        _f(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseSpMatDescr_t, vecX: cusparseDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSV_solve(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSV_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, *const ::core::ffi::c_void, cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t, cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t, cusparseSpSVDescr_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_solve") });
        _f(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_solve(handle: cusparseHandle_t, opA: cusparseOperation_t, alpha: *const ::core::ffi::c_void, matA: cusparseConstSpMatDescr_t, vecX: cusparseConstDnVecDescr_t, vecY: cusparseDnVecDescr_t, computeType: cudaDataType, alg: cusparseSpSVAlg_t, spsvDescr: cusparseSpSVDescr_t) -> cusparseStatus_t;
        }
        cusparseSpSV_solve(handle, opA, alpha, matA, vecX, vecY, computeType, alg, spsvDescr)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpSV_updateMatrix(handle: cusparseHandle_t, spsvDescr: cusparseSpSVDescr_t, newValues: *mut ::core::ffi::c_void, updatePart: cusparseSpSVUpdate_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseSpSVDescr_t, *mut ::core::ffi::c_void, cusparseSpSVUpdate_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpSV_updateMatrix") });
        _f(handle, spsvDescr, newValues, updatePart)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpSV_updateMatrix(handle: cusparseHandle_t, spsvDescr: cusparseSpSVDescr_t, newValues: *mut ::core::ffi::c_void, updatePart: cusparseSpSVUpdate_t) -> cusparseStatus_t;
        }
        cusparseSpSV_updateMatrix(handle, spsvDescr, newValues, updatePart)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpVV(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t, result: *mut ::core::ffi::c_void, computeType: cudaDataType, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVV") });
        _f(handle, opX, vecX, vecY, result, computeType, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVV(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t, result: *mut ::core::ffi::c_void, computeType: cudaDataType, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpVV(handle, opX, vecX, vecY, result, computeType, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpVV(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseConstSpVecDescr_t, vecY: cusparseConstDnVecDescr_t, result: *mut ::core::ffi::c_void, computeType: cudaDataType, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseConstSpVecDescr_t, cusparseConstDnVecDescr_t, *mut ::core::ffi::c_void, cudaDataType, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVV") });
        _f(handle, opX, vecX, vecY, result, computeType, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVV(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseConstSpVecDescr_t, vecY: cusparseConstDnVecDescr_t, result: *mut ::core::ffi::c_void, computeType: cudaDataType, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpVV(handle, opX, vecX, vecY, result, computeType, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpVV_bufferSize(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t, result: *const ::core::ffi::c_void, computeType: cudaDataType, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t, cusparseDnVecDescr_t, *const ::core::ffi::c_void, cudaDataType, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVV_bufferSize") });
        _f(handle, opX, vecX, vecY, result, computeType, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVV_bufferSize(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseSpVecDescr_t, vecY: cusparseDnVecDescr_t, result: *const ::core::ffi::c_void, computeType: cudaDataType, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpVV_bufferSize(handle, opX, vecX, vecY, result, computeType, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpVV_bufferSize(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseConstSpVecDescr_t, vecY: cusparseConstDnVecDescr_t, result: *const ::core::ffi::c_void, computeType: cudaDataType, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, cusparseConstSpVecDescr_t, cusparseConstDnVecDescr_t, *const ::core::ffi::c_void, cudaDataType, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVV_bufferSize") });
        _f(handle, opX, vecX, vecY, result, computeType, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVV_bufferSize(handle: cusparseHandle_t, opX: cusparseOperation_t, vecX: cusparseConstSpVecDescr_t, vecY: cusparseConstDnVecDescr_t, result: *const ::core::ffi::c_void, computeType: cudaDataType, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpVV_bufferSize(handle, opX, vecX, vecY, result, computeType, bufferSize)
    }
}
pub unsafe fn cusparseSpVecGet(spVecDescr: cusparseSpVecDescr_t, size: *mut i64, nnz: *mut i64, indices: *mut *mut ::core::ffi::c_void, values: *mut *mut ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpVecDescr_t, *mut i64, *mut i64, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut cusparseIndexType_t, *mut cusparseIndexBase_t, *mut cudaDataType) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVecGet") });
        _f(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVecGet(spVecDescr: cusparseSpVecDescr_t, size: *mut i64, nnz: *mut i64, indices: *mut *mut ::core::ffi::c_void, values: *mut *mut ::core::ffi::c_void, idxType: *mut cusparseIndexType_t, idxBase: *mut cusparseIndexBase_t, valueType: *mut cudaDataType) -> cusparseStatus_t;
        }
        cusparseSpVecGet(spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSpVecGetIndexBase(spVecDescr: cusparseSpVecDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpVecDescr_t, *mut cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVecGetIndexBase") });
        _f(spVecDescr, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVecGetIndexBase(spVecDescr: cusparseSpVecDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSpVecGetIndexBase(spVecDescr, idxBase)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSpVecGetIndexBase(spVecDescr: cusparseConstSpVecDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseConstSpVecDescr_t, *mut cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVecGetIndexBase") });
        _f(spVecDescr, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVecGetIndexBase(spVecDescr: cusparseConstSpVecDescr_t, idxBase: *mut cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSpVecGetIndexBase(spVecDescr, idxBase)
    }
}
pub unsafe fn cusparseSpVecGetValues(spVecDescr: cusparseSpVecDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpVecDescr_t, *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVecGetValues") });
        _f(spVecDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVecGetValues(spVecDescr: cusparseSpVecDescr_t, values: *mut *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpVecGetValues(spVecDescr, values)
    }
}
pub unsafe fn cusparseSpVecSetValues(spVecDescr: cusparseSpVecDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseSpVecDescr_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpVecSetValues") });
        _f(spVecDescr, values)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpVecSetValues(spVecDescr: cusparseSpVecDescr_t, values: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpVecSetValues(spVecDescr, values)
    }
}
#[cfg(any(feature = "cuda-11040"))]
pub unsafe fn cusparseSparseToDense(handle: cusparseHandle_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSparseToDense") });
        _f(handle, matA, matB, alg, buffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSparseToDense(handle: cusparseHandle_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, buffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSparseToDense(handle, matA, matB, alg, buffer)
    }
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSparseToDense(handle: cusparseHandle_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSparseToDense") });
        _f(handle, matA, matB, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSparseToDense(handle: cusparseHandle_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSparseToDense(handle, matA, matB, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSparseToDense(handle: cusparseHandle_t, matA: cusparseConstSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSparseToDense") });
        _f(handle, matA, matB, alg, externalBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSparseToDense(handle: cusparseHandle_t, matA: cusparseConstSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, externalBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSparseToDense(handle, matA, matB, alg, externalBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSparseToDense_bufferSize(handle: cusparseHandle_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSparseToDense_bufferSize") });
        _f(handle, matA, matB, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSparseToDense_bufferSize(handle: cusparseHandle_t, matA: cusparseSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSparseToDense_bufferSize(handle, matA, matB, alg, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cusparseSparseToDense_bufferSize(handle: cusparseHandle_t, matA: cusparseConstSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseConstSpMatDescr_t, cusparseDnMatDescr_t, cusparseSparseToDenseAlg_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSparseToDense_bufferSize") });
        _f(handle, matA, matB, alg, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSparseToDense_bufferSize(handle: cusparseHandle_t, matA: cusparseConstSpMatDescr_t, matB: cusparseDnMatDescr_t, alg: cusparseSparseToDenseAlg_t, bufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseSparseToDense_bufferSize(handle, matA, matB, alg, bufferSize)
    }
}
pub unsafe fn cusparseSpruneCsr2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneCsr2csr") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneCsr2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneCsr2csr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseSpruneCsr2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneCsr2csrByPercentage") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneCsr2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneCsr2csrByPercentage(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
pub unsafe fn cusparseSpruneCsr2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, pruneInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneCsr2csrByPercentage_bufferSizeExt") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneCsr2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpruneCsr2csrByPercentage_bufferSizeExt(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSpruneCsr2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneCsr2csrNnz") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneCsr2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneCsr2csrNnz(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
}
pub unsafe fn cusparseSpruneCsr2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneCsr2csrNnzByPercentage") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneCsr2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneCsr2csrNnzByPercentage(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
}
pub unsafe fn cusparseSpruneCsr2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneCsr2csr_bufferSizeExt") });
        _f(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneCsr2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzA: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const f32, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpruneCsr2csr_bufferSizeExt(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSpruneDense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneDense2csr") });
        _f(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneDense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneDense2csr(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseSpruneDense2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut f32, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneDense2csrByPercentage") });
        _f(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneDense2csrByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *mut f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneDense2csrByPercentage(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
pub unsafe fn cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, pruneInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneDense2csrByPercentage_bufferSizeExt") });
        _f(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, info: pruneInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseSpruneDense2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneDense2csrNnz") });
        _f(handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneDense2csrNnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneDense2csrNnz(handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr, pBuffer)
    }
}
pub unsafe fn cusparseSpruneDense2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, f32, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, pruneInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneDense2csrNnzByPercentage") });
        _f(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneDense2csrNnzByPercentage(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, percentage: f32, descrC: cusparseMatDescr_t, csrRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: pruneInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseSpruneDense2csrNnzByPercentage(handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
}
pub unsafe fn cusparseSpruneDense2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const f32, ::core::ffi::c_int, *const f32, cusparseMatDescr_t, *const f32, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSpruneDense2csr_bufferSizeExt") });
        _f(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSpruneDense2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, A: *const f32, lda: ::core::ffi::c_int, threshold: *const f32, descrC: cusparseMatDescr_t, csrSortedValC: *const f32, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseSpruneDense2csr_bufferSizeExt(handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSroti(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *mut f32, xInd: *const ::core::ffi::c_int, y: *mut f32, c: *const f32, s: *const f32, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut f32, *const ::core::ffi::c_int, *mut f32, *const f32, *const f32, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSroti") });
        _f(handle, nnz, xVal, xInd, y, c, s, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSroti(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *mut f32, xInd: *const ::core::ffi::c_int, y: *mut f32, c: *const f32, s: *const f32, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseSsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const f32, xInd: *const ::core::ffi::c_int, y: *mut f32, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const f32, *const ::core::ffi::c_int, *mut f32, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseSsctr") });
        _f(handle, nnz, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseSsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const f32, xInd: *const ::core::ffi::c_int, y: *mut f32, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseSsctr(handle, nnz, xVal, xInd, y, idxBase)
    }
}
pub unsafe fn cusparseXbsric02_zeroPivot(handle: cusparseHandle_t, info: bsric02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXbsric02_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXbsric02_zeroPivot(handle: cusparseHandle_t, info: bsric02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXbsric02_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXbsrilu02_zeroPivot(handle: cusparseHandle_t, info: bsrilu02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXbsrilu02_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXbsrilu02_zeroPivot(handle: cusparseHandle_t, info: bsrilu02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXbsrilu02_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXbsrsm2_zeroPivot(handle: cusparseHandle_t, info: bsrsm2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrsm2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXbsrsm2_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXbsrsm2_zeroPivot(handle: cusparseHandle_t, info: bsrsm2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXbsrsm2_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXbsrsv2_zeroPivot(handle: cusparseHandle_t, info: bsrsv2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXbsrsv2_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXbsrsv2_zeroPivot(handle: cusparseHandle_t, info: bsrsv2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXbsrsv2_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXcoo2csr(handle: cusparseHandle_t, cooRowInd: *const ::core::ffi::c_int, nnz: ::core::ffi::c_int, m: ::core::ffi::c_int, csrSortedRowPtr: *mut ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcoo2csr") });
        _f(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcoo2csr(handle: cusparseHandle_t, cooRowInd: *const ::core::ffi::c_int, nnz: ::core::ffi::c_int, m: ::core::ffi::c_int, csrSortedRowPtr: *mut ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase)
    }
}
pub unsafe fn cusparseXcoosortByColumn(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cooRowsA: *mut ::core::ffi::c_int, cooColsA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcoosortByColumn") });
        _f(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcoosortByColumn(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cooRowsA: *mut ::core::ffi::c_int, cooColsA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcoosortByColumn(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
    }
}
pub unsafe fn cusparseXcoosortByRow(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cooRowsA: *mut ::core::ffi::c_int, cooColsA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcoosortByRow") });
        _f(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcoosortByRow(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cooRowsA: *mut ::core::ffi::c_int, cooColsA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcoosortByRow(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
    }
}
pub unsafe fn cusparseXcoosort_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cooRowsA: *const ::core::ffi::c_int, cooColsA: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcoosort_bufferSizeExt") });
        _f(handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcoosort_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cooRowsA: *const ::core::ffi::c_int, cooColsA: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseXcscsort(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscColPtrA: *const ::core::ffi::c_int, cscRowIndA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcscsort") });
        _f(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcscsort(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscColPtrA: *const ::core::ffi::c_int, cscRowIndA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcscsort(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer)
    }
}
pub unsafe fn cusparseXcscsort_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cscColPtrA: *const ::core::ffi::c_int, cscRowIndA: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcscsort_bufferSizeExt") });
        _f(handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcscsort_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, cscColPtrA: *const ::core::ffi::c_int, cscRowIndA: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseXcscsort_bufferSizeExt(handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseXcsr2bsrNnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsr2bsrNnz") });
        _f(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedRowPtrC, nnzTotalDevHostPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsr2bsrNnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXcsr2bsrNnz(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedRowPtrC, nnzTotalDevHostPtr)
    }
}
pub unsafe fn cusparseXcsr2coo(handle: cusparseHandle_t, csrSortedRowPtr: *const ::core::ffi::c_int, nnz: ::core::ffi::c_int, m: ::core::ffi::c_int, cooRowInd: *mut ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsr2coo") });
        _f(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsr2coo(handle: cusparseHandle_t, csrSortedRowPtr: *const ::core::ffi::c_int, nnz: ::core::ffi::c_int, m: ::core::ffi::c_int, cooRowInd: *mut ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseXcsr2coo(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase)
    }
}
pub unsafe fn cusparseXcsr2gebsrNnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedRowPtrC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsr2gebsrNnz") });
        _f(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsr2gebsrNnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedRowPtrC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcsr2gebsrNnz(handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer)
    }
}
pub unsafe fn cusparseXcsrgeam2Nnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, workspace: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrgeam2Nnz") });
        _f(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrgeam2Nnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, workspace: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseXcsrgemm2Nnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, csrgemm2Info_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrgemm2Nnz") });
        _f(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrgemm2Nnz(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedRowPtrC: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info, pBuffer)
    }
}
pub unsafe fn cusparseXcsric02_zeroPivot(handle: cusparseHandle_t, info: csric02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsric02_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsric02_zeroPivot(handle: cusparseHandle_t, info: csric02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXcsric02_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXcsrilu02_zeroPivot(handle: cusparseHandle_t, info: csrilu02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrilu02_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrilu02_zeroPivot(handle: cusparseHandle_t, info: csrilu02Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXcsrilu02_zeroPivot(handle, info, position)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseXcsrsm2_zeroPivot(handle: cusparseHandle_t, info: csrsm2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrsm2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrsm2_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrsm2_zeroPivot(handle: cusparseHandle_t, info: csrsm2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXcsrsm2_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXcsrsort(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrsort") });
        _f(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrsort(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *mut ::core::ffi::c_int, P: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer)
    }
}
pub unsafe fn cusparseXcsrsort_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrsort_bufferSizeExt") });
        _f(handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrsort_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrRowPtrA: *const ::core::ffi::c_int, csrColIndA: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseXcsrsv2_zeroPivot(handle: cusparseHandle_t, info: csrsv2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXcsrsv2_zeroPivot") });
        _f(handle, info, position)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXcsrsv2_zeroPivot(handle: cusparseHandle_t, info: csrsv2Info_t, position: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseXcsrsv2_zeroPivot(handle, info, position)
    }
}
pub unsafe fn cusparseXgebsr2gebsrNnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedRowPtrC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseXgebsr2gebsrNnz") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseXgebsr2gebsrNnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedRowPtrC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseXgebsr2gebsrNnz(handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, xVal: *const cuDoubleComplex, xInd: *const ::core::ffi::c_int, y: *mut cuDoubleComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const ::core::ffi::c_int, *mut cuDoubleComplex, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZaxpyi") });
        _f(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZaxpyi(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, xVal: *const cuDoubleComplex, xInd: *const ::core::ffi::c_int, y: *mut cuDoubleComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    }
}
pub unsafe fn cusparseZbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseZbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsric02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsric02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsric02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsric02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, policy: cusparseSolvePolicy_t, pInputBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pInputBuffer)
    }
}
pub unsafe fn cusparseZbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsric02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsric02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrilu02") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrilu02(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrilu02_analysis") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrilu02_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrilu02_bufferSize") });
        _f(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrilu02_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuDoubleComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, bsrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrilu02_numericBoost(handle: cusparseHandle_t, info: bsrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuDoubleComplex) -> cusparseStatus_t;
        }
        cusparseZbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
pub unsafe fn cusparseZbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrmm") });
        _f(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrmm(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transB: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, kb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc)
    }
}
pub unsafe fn cusparseZbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuDoubleComplex, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *mut cuDoubleComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrmv") });
        _f(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuDoubleComplex, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex) -> cusparseStatus_t;
        }
        cusparseZbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
pub unsafe fn cusparseZbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrsm2_analysis") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrsm2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrsm2_bufferSize") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrsm2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedVal: *mut cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, X: *mut cuDoubleComplex, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsm2Info_t, *const cuDoubleComplex, ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrsm2_solve") });
        _f(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrsm2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, transXY: cusparseOperation_t, mb: ::core::ffi::c_int, n: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, blockSize: ::core::ffi::c_int, info: bsrsm2Info_t, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, X: *mut cuDoubleComplex, ldx: ::core::ffi::c_int, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrsv2_analysis") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrsv2_analysis(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrsv2_bufferSize") });
        _f(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrsv2_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *mut cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const cuDoubleComplex, x: *mut cuDoubleComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, bsrsv2Info_t, *const cuDoubleComplex, *mut cuDoubleComplex, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrsv2_solve") });
        _f(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrsv2_solve(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, mb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, info: bsrsv2Info_t, f: *const cuDoubleComplex, x: *mut cuDoubleComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseZbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuDoubleComplex, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *mut cuDoubleComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZbsrxmv") });
        _f(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZbsrxmv(handle: cusparseHandle_t, dirA: cusparseDirection_t, transA: cusparseOperation_t, sizeOfMask: ::core::ffi::c_int, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedMaskPtrA: *const ::core::ffi::c_int, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedEndPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, x: *const cuDoubleComplex, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex) -> cusparseStatus_t;
        }
        cusparseZbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const cuDoubleComplex, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsc2dense") });
        _f(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsc2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, cscSortedValA: *const cuDoubleComplex, cscSortedRowIndA: *const ::core::ffi::c_int, cscSortedColPtrA: *const ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA, A, lda)
    }
}
pub unsafe fn cusparseZcsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuDoubleComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsr2bsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsr2bsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, blockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuDoubleComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC)
    }
}
pub unsafe fn cusparseZcsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut cuDoubleComplex, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: cuDoubleComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cuDoubleComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsr2csr_compress") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsr2csr_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedColIndA: *const ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzA: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValC: *mut cuDoubleComplex, csrSortedColIndC: *mut ::core::ffi::c_int, csrSortedRowPtrC: *mut ::core::ffi::c_int, tol: cuDoubleComplex) -> cusparseStatus_t;
        }
        cusparseZcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
    }
}
pub unsafe fn cusparseZcsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsr2csru") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsr2csru(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsr2dense") });
        _f(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsr2dense(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, A: *mut cuDoubleComplex, lda: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, A, lda)
    }
}
pub unsafe fn cusparseZcsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuDoubleComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsr2gebsr") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuDoubleComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    }
}
pub unsafe fn cusparseZcsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsr2gebsr_bufferSize") });
        _f(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZcsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f64, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const f64, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseColorInfo_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrcolor") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrcolor(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, fractionToColor: *const f64, ncolors: *mut ::core::ffi::c_int, coloring: *mut ::core::ffi::c_int, reordering: *mut ::core::ffi::c_int, info: cusparseColorInfo_t) -> cusparseStatus_t;
        }
        cusparseZcsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, info)
    }
}
pub unsafe fn cusparseZcsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuDoubleComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrgeam2") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrgeam2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuDoubleComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    }
}
pub unsafe fn cusparseZcsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuDoubleComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const cuDoubleComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrgeam2_bufferSizeExt") });
        _f(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrgeam2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuDoubleComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *const cuDoubleComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *const ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuDoubleComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const cuDoubleComplex, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csrgemm2Info_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrgemm2") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrgemm2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedValB: *const cuDoubleComplex, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedValD: *const cuDoubleComplex, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *const ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int, info: csrgemm2Info_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, ::core::ffi::c_int, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrgemm2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrgemm2_bufferSizeExt") });
        _f(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrgemm2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, nnzA: ::core::ffi::c_int, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, descrB: cusparseMatDescr_t, nnzB: ::core::ffi::c_int, csrSortedRowPtrB: *const ::core::ffi::c_int, csrSortedColIndB: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, descrD: cusparseMatDescr_t, nnzD: ::core::ffi::c_int, csrSortedRowPtrD: *const ::core::ffi::c_int, csrSortedColIndD: *const ::core::ffi::c_int, info: csrgemm2Info_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD, csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZcsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsric02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsric02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZcsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsric02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsric02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZcsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csric02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsric02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsric02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csric02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZcsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrilu02") });
        _f(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrilu02(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA_valM: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZcsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrilu02_analysis") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrilu02_analysis(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
pub unsafe fn cusparseZcsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrilu02Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrilu02_bufferSize") });
        _f(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrilu02_bufferSize(handle: cusparseHandle_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrilu02Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZcsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuDoubleComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, csrilu02Info_t, ::core::ffi::c_int, *mut f64, *mut cuDoubleComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrilu02_numericBoost") });
        _f(handle, info, enable_boost, tol, boost_val)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrilu02_numericBoost(handle: cusparseHandle_t, info: csrilu02Info_t, enable_boost: ::core::ffi::c_int, tol: *mut f64, boost_val: *mut cuDoubleComplex) -> cusparseStatus_t;
        }
        cusparseZcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsm2_analysis") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsm2_analysis(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsm2_bufferSizeExt") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsm2_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseOperation_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, ::core::ffi::c_int, csrsm2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsm2_solve") });
        _f(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsm2_solve(handle: cusparseHandle_t, algo: ::core::ffi::c_int, transA: cusparseOperation_t, transB: cusparseOperation_t, m: ::core::ffi::c_int, nrhs: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, info: csrsm2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsv2_analysis") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsv2_analysis(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsv2_bufferSize") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsv2_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsv2_bufferSizeExt") });
        _f(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsv2_bufferSizeExt(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, pBufferSize: *mut usize) -> cusparseStatus_t;
        }
        cusparseZcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZcsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const cuDoubleComplex, x: *mut cuDoubleComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, csrsv2Info_t, *const cuDoubleComplex, *mut cuDoubleComplex, cusparseSolvePolicy_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsrsv2_solve") });
        _f(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsrsv2_solve(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, descrA: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, csrSortedColIndA: *const ::core::ffi::c_int, info: csrsv2Info_t, f: *const cuDoubleComplex, x: *mut cuDoubleComplex, policy: cusparseSolvePolicy_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy, pBuffer)
    }
}
pub unsafe fn cusparseZcsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsru2csr") });
        _f(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsru2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, descrA: cusparseMatDescr_t, csrVal: *mut cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZcsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
    }
}
pub unsafe fn cusparseZcsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, csru2csrInfo_t, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZcsru2csr_bufferSizeExt") });
        _f(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZcsru2csr_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, csrVal: *mut cuDoubleComplex, csrRowPtr: *const ::core::ffi::c_int, csrColInd: *mut ::core::ffi::c_int, info: csru2csrInfo_t, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZcsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut cuDoubleComplex, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZdense2csc") });
        _f(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZdense2csc(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnzPerCol: *const ::core::ffi::c_int, cscSortedValA: *mut cuDoubleComplex, cscSortedRowIndA: *mut ::core::ffi::c_int, cscSortedColPtrA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, ::core::ffi::c_int, *const ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZdense2csr") });
        _f(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZdense2csr(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnzPerRow: *const ::core::ffi::c_int, csrSortedValA: *mut cuDoubleComplex, csrSortedRowPtrA: *mut ::core::ffi::c_int, csrSortedColIndA: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
    }
}
pub unsafe fn cusparseZgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgebsr2csr") });
        _f(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgebsr2csr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, descrC: cusparseMatDescr_t, csrSortedValC: *mut cuDoubleComplex, csrSortedRowPtrC: *mut ::core::ffi::c_int, csrSortedColIndC: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    }
}
pub unsafe fn cusparseZgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut cuDoubleComplex, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cusparseAction_t, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgebsr2gebsc") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgebsr2gebsc(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, bscVal: *mut cuDoubleComplex, bscRowInd: *mut ::core::ffi::c_int, bscColPtr: *mut ::core::ffi::c_int, copyValues: cusparseAction_t, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseZgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgebsr2gebsc_bufferSize") });
        _f(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgebsr2gebsc_bufferSize(handle: cusparseHandle_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, bsrSortedVal: *const cuDoubleComplex, bsrSortedRowPtr: *const ::core::ffi::c_int, bsrSortedColInd: *const ::core::ffi::c_int, rowBlockDim: ::core::ffi::c_int, colBlockDim: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuDoubleComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *mut cuDoubleComplex, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgebsr2gebsr") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgebsr2gebsr(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, descrC: cusparseMatDescr_t, bsrSortedValC: *mut cuDoubleComplex, bsrSortedRowPtrC: *mut ::core::ffi::c_int, bsrSortedColIndC: *mut ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    }
}
pub unsafe fn cusparseZgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgebsr2gebsr_bufferSize") });
        _f(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgebsr2gebsr_bufferSize(handle: cusparseHandle_t, dirA: cusparseDirection_t, mb: ::core::ffi::c_int, nb: ::core::ffi::c_int, nnzb: ::core::ffi::c_int, descrA: cusparseMatDescr_t, bsrSortedValA: *const cuDoubleComplex, bsrSortedRowPtrA: *const ::core::ffi::c_int, bsrSortedColIndA: *const ::core::ffi::c_int, rowBlockDimA: ::core::ffi::c_int, colBlockDimA: ::core::ffi::c_int, rowBlockDimC: ::core::ffi::c_int, colBlockDimC: ::core::ffi::c_int, pBufferSizeInBytes: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, cscValB: *const cuDoubleComplex, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgemmi") });
        _f(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgemmi(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, k: ::core::ffi::c_int, nnz: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, cscValB: *const cuDoubleComplex, cscColPtrB: *const ::core::ffi::c_int, cscRowIndB: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, C: *mut cuDoubleComplex, ldc: ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB, beta, C, ldc)
    }
}
pub unsafe fn cusparseZgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const cuDoubleComplex, xInd: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, cusparseIndexBase_t, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgemvi") });
        _f(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgemvi(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, alpha: *const cuDoubleComplex, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnz: ::core::ffi::c_int, xVal: *const cuDoubleComplex, xInd: *const ::core::ffi::c_int, beta: *const cuDoubleComplex, y: *mut cuDoubleComplex, idxBase: cusparseIndexBase_t, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase, pBuffer)
    }
}
pub unsafe fn cusparseZgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseOperation_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgemvi_bufferSize") });
        _f(handle, transA, m, n, nnz, pBufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgemvi_bufferSize(handle: cusparseHandle_t, transA: cusparseOperation_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, nnz: ::core::ffi::c_int, pBufferSize: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    }
}
pub unsafe fn cusparseZgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut cuDoubleComplex, dl: *mut cuDoubleComplex, d: *mut cuDoubleComplex, du: *mut cuDoubleComplex, dw: *mut cuDoubleComplex, x: *mut cuDoubleComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgpsvInterleavedBatch") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgpsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *mut cuDoubleComplex, dl: *mut cuDoubleComplex, d: *mut cuDoubleComplex, du: *mut cuDoubleComplex, dw: *mut cuDoubleComplex, x: *mut cuDoubleComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseZgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const cuDoubleComplex, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, dw: *const cuDoubleComplex, x: *const cuDoubleComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgpsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgpsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, ds: *const cuDoubleComplex, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, dw: *const cuDoubleComplex, x: *const cuDoubleComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const cuDoubleComplex, xVal: *mut cuDoubleComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *mut cuDoubleComplex, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgthr") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgthr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *const cuDoubleComplex, xVal: *mut cuDoubleComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseZgthr(handle, nnz, y, xVal, xInd, idxBase)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut cuDoubleComplex, xVal: *mut cuDoubleComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *mut cuDoubleComplex, *mut cuDoubleComplex, *const ::core::ffi::c_int, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgthrz") });
        _f(handle, nnz, y, xVal, xInd, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgthrz(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, y: *mut cuDoubleComplex, xVal: *mut cuDoubleComplex, xInd: *const ::core::ffi::c_int, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseZgthrz(handle, nnz, y, xVal, xInd, idxBase)
    }
}
pub unsafe fn cusparseZgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsv2") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsv2(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseZgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, x: *mut cuDoubleComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsv2StridedBatch") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsv2StridedBatch(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, x: *mut cuDoubleComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
    }
}
pub unsafe fn cusparseZgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, x: *const cuDoubleComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsv2StridedBatch_bufferSizeExt") });
        _f(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsv2StridedBatch_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, x: *const cuDoubleComplex, batchCount: ::core::ffi::c_int, batchStride: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseZgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsv2_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsv2_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseZgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsv2_nopivot") });
        _f(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsv2_nopivot(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *mut cuDoubleComplex, ldb: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    }
}
pub unsafe fn cusparseZgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsv2_nopivot_bufferSizeExt") });
        _f(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsv2_nopivot_bufferSizeExt(handle: cusparseHandle_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, B: *const cuDoubleComplex, ldb: ::core::ffi::c_int, bufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    }
}
pub unsafe fn cusparseZgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut cuDoubleComplex, d: *mut cuDoubleComplex, du: *mut cuDoubleComplex, x: *mut cuDoubleComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, *mut cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_void) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsvInterleavedBatch") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsvInterleavedBatch(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *mut cuDoubleComplex, d: *mut cuDoubleComplex, du: *mut cuDoubleComplex, x: *mut cuDoubleComplex, batchCount: ::core::ffi::c_int, pBuffer: *mut ::core::ffi::c_void) -> cusparseStatus_t;
        }
        cusparseZgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    }
}
pub unsafe fn cusparseZgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, x: *const cuDoubleComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, ::core::ffi::c_int, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, *const cuDoubleComplex, ::core::ffi::c_int, *mut usize) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZgtsvInterleavedBatch_bufferSizeExt") });
        _f(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZgtsvInterleavedBatch_bufferSizeExt(handle: cusparseHandle_t, algo: ::core::ffi::c_int, m: ::core::ffi::c_int, dl: *const cuDoubleComplex, d: *const cuDoubleComplex, du: *const cuDoubleComplex, x: *const cuDoubleComplex, batchCount: ::core::ffi::c_int, pBufferSizeInBytes: *mut usize) -> cusparseStatus_t;
        }
        cusparseZgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
    }
}
pub unsafe fn cusparseZnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, cusparseDirection_t, ::core::ffi::c_int, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZnnz") });
        _f(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZnnz(handle: cusparseHandle_t, dirA: cusparseDirection_t, m: ::core::ffi::c_int, n: ::core::ffi::c_int, descrA: cusparseMatDescr_t, A: *const cuDoubleComplex, lda: ::core::ffi::c_int, nnzPerRowCol: *mut ::core::ffi::c_int, nnzTotalDevHostPtr: *mut ::core::ffi::c_int) -> cusparseStatus_t;
        }
        cusparseZnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    }
}
pub unsafe fn cusparseZnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: cuDoubleComplex) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, cusparseMatDescr_t, *const cuDoubleComplex, *const ::core::ffi::c_int, *mut ::core::ffi::c_int, *mut ::core::ffi::c_int, cuDoubleComplex) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZnnz_compress") });
        _f(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZnnz_compress(handle: cusparseHandle_t, m: ::core::ffi::c_int, descr: cusparseMatDescr_t, csrSortedValA: *const cuDoubleComplex, csrSortedRowPtrA: *const ::core::ffi::c_int, nnzPerRow: *mut ::core::ffi::c_int, nnzC: *mut ::core::ffi::c_int, tol: cuDoubleComplex) -> cusparseStatus_t;
        }
        cusparseZnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cusparseZsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const cuDoubleComplex, xInd: *const ::core::ffi::c_int, y: *mut cuDoubleComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cusparseHandle_t, ::core::ffi::c_int, *const cuDoubleComplex, *const ::core::ffi::c_int, *mut cuDoubleComplex, cusparseIndexBase_t) -> cusparseStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cusparseZsctr") });
        _f(handle, nnz, xVal, xInd, y, idxBase)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cusparseZsctr(handle: cusparseHandle_t, nnz: ::core::ffi::c_int, xVal: *const cuDoubleComplex, xInd: *const ::core::ffi::c_int, y: *mut cuDoubleComplex, idxBase: cusparseIndexBase_t) -> cusparseStatus_t;
        }
        cusparseZsctr(handle, nnz, xVal, xInd, y, idxBase)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cusparse"];
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
        let lib_names = std::vec!["cusparse"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
