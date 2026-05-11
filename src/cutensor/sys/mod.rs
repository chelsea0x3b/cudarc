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
pub use self::cudaDataType_t as cutensorDataType_t;
pub type FILE = _IO_FILE;
pub type _IO_lock_t = ::core::ffi::c_void;
pub type __off64_t = ::core::ffi::c_long;
pub type __off_t = ::core::ffi::c_long;
pub type cudaStream_t = *mut CUstream_st;
pub type cutensorBlockSparseTensorDescriptor_t = *mut cutensorBlockSparseTensorDescriptor;
pub type cutensorComputeDescriptor_t = *mut cutensorComputeDescriptor;
pub type cutensorHandle_t = *mut cutensorHandle;
pub type cutensorLoggerCallback_t = ::core::option::Option<unsafe extern "C" fn(logLevel: i32, functionName: *const ::core::ffi::c_char, message: *const ::core::ffi::c_char)>;
pub type cutensorOperationDescriptor_t = *mut cutensorOperationDescriptor;
pub type cutensorPlanPreference_t = *mut cutensorPlanPreference;
pub type cutensorPlan_t = *mut cutensorPlan;
pub type cutensorTensorDescriptor_t = *mut cutensorTensorDescriptor;
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
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorAlgo_t {
    CUTENSOR_ALGO_DEFAULT_PATIENT = -6,
    CUTENSOR_ALGO_GETT = -4,
    CUTENSOR_ALGO_TGETT = -3,
    CUTENSOR_ALGO_TTGT = -2,
    CUTENSOR_ALGO_DEFAULT = -1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorAutotuneMode_t {
    CUTENSOR_AUTOTUNE_MODE_NONE = 0,
    CUTENSOR_AUTOTUNE_MODE_INCREMENTAL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorCacheMode_t {
    CUTENSOR_CACHE_MODE_NONE = 0,
    CUTENSOR_CACHE_MODE_PEDANTIC = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorJitMode_t {
    CUTENSOR_JIT_MODE_NONE = 0,
    CUTENSOR_JIT_MODE_DEFAULT = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorOperationDescriptorAttribute_t {
    CUTENSOR_OPERATION_DESCRIPTOR_TAG = 0,
    CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1,
    CUTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2,
    CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5,
    CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorOperator_t {
    CUTENSOR_OP_IDENTITY = 1,
    CUTENSOR_OP_SQRT = 2,
    CUTENSOR_OP_RELU = 8,
    CUTENSOR_OP_CONJ = 9,
    CUTENSOR_OP_RCP = 10,
    CUTENSOR_OP_SIGMOID = 11,
    CUTENSOR_OP_TANH = 12,
    CUTENSOR_OP_EXP = 22,
    CUTENSOR_OP_LOG = 23,
    CUTENSOR_OP_ABS = 24,
    CUTENSOR_OP_NEG = 25,
    CUTENSOR_OP_SIN = 26,
    CUTENSOR_OP_COS = 27,
    CUTENSOR_OP_TAN = 28,
    CUTENSOR_OP_SINH = 29,
    CUTENSOR_OP_COSH = 30,
    CUTENSOR_OP_ASIN = 31,
    CUTENSOR_OP_ACOS = 32,
    CUTENSOR_OP_ATAN = 33,
    CUTENSOR_OP_ASINH = 34,
    CUTENSOR_OP_ACOSH = 35,
    CUTENSOR_OP_ATANH = 36,
    CUTENSOR_OP_CEIL = 37,
    CUTENSOR_OP_FLOOR = 38,
    CUTENSOR_OP_MISH = 39,
    CUTENSOR_OP_SWISH = 40,
    CUTENSOR_OP_SOFT_PLUS = 41,
    CUTENSOR_OP_SOFT_SIGN = 42,
    CUTENSOR_OP_ADD = 3,
    CUTENSOR_OP_MUL = 5,
    CUTENSOR_OP_MAX = 6,
    CUTENSOR_OP_MIN = 7,
    CUTENSOR_OP_UNKNOWN = 126,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorPlanAttribute_t {
    CUTENSOR_PLAN_REQUIRED_WORKSPACE = 0,
}
#[cfg(any(feature = "cutensor-02003", feature = "cutensor-02004", feature = "cutensor-02005"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorPlanPreferenceAttribute_t {
    CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = 0,
    CUTENSOR_PLAN_PREFERENCE_CACHE_MODE = 1,
    CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2,
    CUTENSOR_PLAN_PREFERENCE_ALGO = 3,
    CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 4,
    CUTENSOR_PLAN_PREFERENCE_JIT = 5,
}
#[cfg(any(feature = "cutensor-02006"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorPlanPreferenceAttribute_t {
    CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = 0,
    CUTENSOR_PLAN_PREFERENCE_CACHE_MODE = 1,
    CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = 2,
    CUTENSOR_PLAN_PREFERENCE_ALGO = 3,
    CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 4,
    CUTENSOR_PLAN_PREFERENCE_JIT = 5,
    CUTENSOR_PLAN_PREFERENCE_GPU_ARCH = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorStatus_t {
    CUTENSOR_STATUS_SUCCESS = 0,
    CUTENSOR_STATUS_NOT_INITIALIZED = 1,
    CUTENSOR_STATUS_ALLOC_FAILED = 3,
    CUTENSOR_STATUS_INVALID_VALUE = 7,
    CUTENSOR_STATUS_ARCH_MISMATCH = 8,
    CUTENSOR_STATUS_MAPPING_ERROR = 11,
    CUTENSOR_STATUS_EXECUTION_FAILED = 13,
    CUTENSOR_STATUS_INTERNAL_ERROR = 14,
    CUTENSOR_STATUS_NOT_SUPPORTED = 15,
    CUTENSOR_STATUS_LICENSE_ERROR = 16,
    CUTENSOR_STATUS_CUBLAS_ERROR = 17,
    CUTENSOR_STATUS_CUDA_ERROR = 18,
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    CUTENSOR_STATUS_IO_ERROR = 21,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cutensorWorksizePreference_t {
    CUTENSOR_WORKSPACE_MIN = 1,
    CUTENSOR_WORKSPACE_DEFAULT = 2,
    CUTENSOR_WORKSPACE_MAX = 3,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_codecvt {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_marker {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_wide_data {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorBlockSparseTensorDescriptor {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorComputeDescriptor {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorHandle {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorOperationDescriptor {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorPlan {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorPlanPreference {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cutensorTensorDescriptor {
    _unused: [u8; 0],
}
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
pub unsafe fn cutensorBlockSparseContract(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const *const ::core::ffi::c_void, B: *const *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, C: *const *const ::core::ffi::c_void, D: *const *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const *const ::core::ffi::c_void, *const *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, u64, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorBlockSparseContract") });
        _f(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorBlockSparseContract(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const *const ::core::ffi::c_void, B: *const *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, C: *const *const ::core::ffi::c_void, D: *const *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorBlockSparseContract(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
    }
}
pub unsafe fn cutensorContract(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, B: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, u64, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorContract") });
        _f(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorContract(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, B: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorContract(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
    }
}
pub unsafe fn cutensorContractTrinary(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, B: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, D: *const ::core::ffi::c_void, E: *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, u64, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorContractTrinary") });
        _f(handle, plan, alpha, A, B, C, beta, D, E, workspace, workspaceSize, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorContractTrinary(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, B: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, D: *const ::core::ffi::c_void, E: *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorContractTrinary(handle, plan, alpha, A, B, C, beta, D, E, workspace, workspaceSize, stream)
    }
}
pub unsafe fn cutensorCreate(handle: *mut cutensorHandle_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cutensorHandle_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreate") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreate(handle: *mut cutensorHandle_t) -> cutensorStatus_t;
        }
        cutensorCreate(handle)
    }
}
pub unsafe fn cutensorCreateBlockSparseContraction(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorBlockSparseTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorBlockSparseTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorBlockSparseTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorBlockSparseTensorDescriptor_t, modeD: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorBlockSparseTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorBlockSparseTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorBlockSparseTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorBlockSparseTensorDescriptor_t, *const i32, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateBlockSparseContraction") });
        _f(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateBlockSparseContraction(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorBlockSparseTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorBlockSparseTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorBlockSparseTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorBlockSparseTensorDescriptor_t, modeD: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreateBlockSparseContraction(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)
    }
}
pub unsafe fn cutensorCreateBlockSparseTensorDescriptor(handle: cutensorHandle_t, desc: *mut cutensorBlockSparseTensorDescriptor_t, numModes: u32, numNonZeroBlocks: u64, numSectionsPerMode: *const u32, extent: *const i64, nonZeroCoordinates: *const i32, stride: *const i64, dataType: cudaDataType_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorBlockSparseTensorDescriptor_t, u32, u64, *const u32, *const i64, *const i32, *const i64, cudaDataType_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateBlockSparseTensorDescriptor") });
        _f(handle, desc, numModes, numNonZeroBlocks, numSectionsPerMode, extent, nonZeroCoordinates, stride, dataType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateBlockSparseTensorDescriptor(handle: cutensorHandle_t, desc: *mut cutensorBlockSparseTensorDescriptor_t, numModes: u32, numNonZeroBlocks: u64, numSectionsPerMode: *const u32, extent: *const i64, nonZeroCoordinates: *const i32, stride: *const i64, dataType: cudaDataType_t) -> cutensorStatus_t;
        }
        cutensorCreateBlockSparseTensorDescriptor(handle, desc, numModes, numNonZeroBlocks, numSectionsPerMode, extent, nonZeroCoordinates, stride, dataType)
    }
}
pub unsafe fn cutensorCreateContraction(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateContraction") });
        _f(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateContraction(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreateContraction(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, descCompute)
    }
}
pub unsafe fn cutensorCreateContractionTrinary(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opD: cutensorOperator_t, descE: cutensorTensorDescriptor_t, modeE: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateContractionTrinary") });
        _f(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opD, descE, modeE, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateContractionTrinary(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opD: cutensorOperator_t, descE: cutensorTensorDescriptor_t, modeE: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreateContractionTrinary(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opD, descE, modeE, descCompute)
    }
}
pub unsafe fn cutensorCreateElementwiseBinary(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opAC: cutensorOperator_t, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateElementwiseBinary") });
        _f(handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opAC, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateElementwiseBinary(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opAC: cutensorOperator_t, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreateElementwiseBinary(handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opAC, descCompute)
    }
}
pub unsafe fn cutensorCreateElementwiseTrinary(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opAB: cutensorOperator_t, opABC: cutensorOperator_t, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorOperator_t, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateElementwiseTrinary") });
        _f(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opAB, opABC, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateElementwiseTrinary(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, opB: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opAB: cutensorOperator_t, opABC: cutensorOperator_t, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreateElementwiseTrinary(handle, desc, descA, modeA, opA, descB, modeB, opB, descC, modeC, opC, descD, modeD, opAB, opABC, descCompute)
    }
}
pub unsafe fn cutensorCreatePermutation(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreatePermutation") });
        _f(handle, desc, descA, modeA, opA, descB, modeB, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreatePermutation(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descB: cutensorTensorDescriptor_t, modeB: *const i32, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreatePermutation(handle, desc, descA, modeA, opA, descB, modeB, descCompute)
    }
}
pub unsafe fn cutensorCreatePlan(handle: cutensorHandle_t, plan: *mut cutensorPlan_t, desc: cutensorOperationDescriptor_t, pref: cutensorPlanPreference_t, workspaceSizeLimit: u64) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorPlan_t, cutensorOperationDescriptor_t, cutensorPlanPreference_t, u64) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreatePlan") });
        _f(handle, plan, desc, pref, workspaceSizeLimit)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreatePlan(handle: cutensorHandle_t, plan: *mut cutensorPlan_t, desc: cutensorOperationDescriptor_t, pref: cutensorPlanPreference_t, workspaceSizeLimit: u64) -> cutensorStatus_t;
        }
        cutensorCreatePlan(handle, plan, desc, pref, workspaceSizeLimit)
    }
}
pub unsafe fn cutensorCreatePlanPreference(handle: cutensorHandle_t, pref: *mut cutensorPlanPreference_t, algo: cutensorAlgo_t, jitMode: cutensorJitMode_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorPlanPreference_t, cutensorAlgo_t, cutensorJitMode_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreatePlanPreference") });
        _f(handle, pref, algo, jitMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreatePlanPreference(handle: cutensorHandle_t, pref: *mut cutensorPlanPreference_t, algo: cutensorAlgo_t, jitMode: cutensorJitMode_t) -> cutensorStatus_t;
        }
        cutensorCreatePlanPreference(handle, pref, algo, jitMode)
    }
}
pub unsafe fn cutensorCreateReduction(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opReduce: cutensorOperator_t, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorOperationDescriptor_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorTensorDescriptor_t, *const i32, cutensorOperator_t, cutensorComputeDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateReduction") });
        _f(handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opReduce, descCompute)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateReduction(handle: cutensorHandle_t, desc: *mut cutensorOperationDescriptor_t, descA: cutensorTensorDescriptor_t, modeA: *const i32, opA: cutensorOperator_t, descC: cutensorTensorDescriptor_t, modeC: *const i32, opC: cutensorOperator_t, descD: cutensorTensorDescriptor_t, modeD: *const i32, opReduce: cutensorOperator_t, descCompute: cutensorComputeDescriptor_t) -> cutensorStatus_t;
        }
        cutensorCreateReduction(handle, desc, descA, modeA, opA, descC, modeC, opC, descD, modeD, opReduce, descCompute)
    }
}
pub unsafe fn cutensorCreateTensorDescriptor(handle: cutensorHandle_t, desc: *mut cutensorTensorDescriptor_t, numModes: u32, extent: *const i64, stride: *const i64, dataType: cudaDataType_t, alignmentRequirement: u32) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *mut cutensorTensorDescriptor_t, u32, *const i64, *const i64, cudaDataType_t, u32) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorCreateTensorDescriptor") });
        _f(handle, desc, numModes, extent, stride, dataType, alignmentRequirement)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorCreateTensorDescriptor(handle: cutensorHandle_t, desc: *mut cutensorTensorDescriptor_t, numModes: u32, extent: *const i64, stride: *const i64, dataType: cudaDataType_t, alignmentRequirement: u32) -> cutensorStatus_t;
        }
        cutensorCreateTensorDescriptor(handle, desc, numModes, extent, stride, dataType, alignmentRequirement)
    }
}
pub unsafe fn cutensorDestroy(handle: cutensorHandle_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorDestroy") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorDestroy(handle: cutensorHandle_t) -> cutensorStatus_t;
        }
        cutensorDestroy(handle)
    }
}
pub unsafe fn cutensorDestroyBlockSparseTensorDescriptor(desc: cutensorBlockSparseTensorDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorBlockSparseTensorDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorDestroyBlockSparseTensorDescriptor") });
        _f(desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorDestroyBlockSparseTensorDescriptor(desc: cutensorBlockSparseTensorDescriptor_t) -> cutensorStatus_t;
        }
        cutensorDestroyBlockSparseTensorDescriptor(desc)
    }
}
pub unsafe fn cutensorDestroyOperationDescriptor(desc: cutensorOperationDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorOperationDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorDestroyOperationDescriptor") });
        _f(desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorDestroyOperationDescriptor(desc: cutensorOperationDescriptor_t) -> cutensorStatus_t;
        }
        cutensorDestroyOperationDescriptor(desc)
    }
}
pub unsafe fn cutensorDestroyPlan(plan: cutensorPlan_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorPlan_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorDestroyPlan") });
        _f(plan)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorDestroyPlan(plan: cutensorPlan_t) -> cutensorStatus_t;
        }
        cutensorDestroyPlan(plan)
    }
}
pub unsafe fn cutensorDestroyPlanPreference(pref: cutensorPlanPreference_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorPlanPreference_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorDestroyPlanPreference") });
        _f(pref)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorDestroyPlanPreference(pref: cutensorPlanPreference_t) -> cutensorStatus_t;
        }
        cutensorDestroyPlanPreference(pref)
    }
}
pub unsafe fn cutensorDestroyTensorDescriptor(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorTensorDescriptor_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorDestroyTensorDescriptor") });
        _f(desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorDestroyTensorDescriptor(desc: cutensorTensorDescriptor_t) -> cutensorStatus_t;
        }
        cutensorDestroyTensorDescriptor(desc)
    }
}
pub unsafe fn cutensorElementwiseBinaryExecute(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, gamma: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorElementwiseBinaryExecute") });
        _f(handle, plan, alpha, A, gamma, C, D, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorElementwiseBinaryExecute(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, gamma: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorElementwiseBinaryExecute(handle, plan, alpha, A, gamma, C, D, stream)
    }
}
pub unsafe fn cutensorElementwiseTrinaryExecute(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, B: *const ::core::ffi::c_void, gamma: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorElementwiseTrinaryExecute") });
        _f(handle, plan, alpha, A, beta, B, gamma, C, D, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorElementwiseTrinaryExecute(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, B: *const ::core::ffi::c_void, gamma: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorElementwiseTrinaryExecute(handle, plan, alpha, A, beta, B, gamma, C, D, stream)
    }
}
pub unsafe fn cutensorEstimateWorkspaceSize(handle: cutensorHandle_t, desc: cutensorOperationDescriptor_t, planPref: cutensorPlanPreference_t, workspacePref: cutensorWorksizePreference_t, workspaceSizeEstimate: *mut u64) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorOperationDescriptor_t, cutensorPlanPreference_t, cutensorWorksizePreference_t, *mut u64) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorEstimateWorkspaceSize") });
        _f(handle, desc, planPref, workspacePref, workspaceSizeEstimate)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorEstimateWorkspaceSize(handle: cutensorHandle_t, desc: cutensorOperationDescriptor_t, planPref: cutensorPlanPreference_t, workspacePref: cutensorWorksizePreference_t, workspaceSizeEstimate: *mut u64) -> cutensorStatus_t;
        }
        cutensorEstimateWorkspaceSize(handle, desc, planPref, workspacePref, workspaceSizeEstimate)
    }
}
pub unsafe fn cutensorGetCudartVersion() -> usize {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> usize;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorGetCudartVersion") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorGetCudartVersion() -> usize;
        }
        cutensorGetCudartVersion()
    }
}
pub unsafe fn cutensorGetErrorString(error: cutensorStatus_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorStatus_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorGetErrorString") });
        _f(error)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorGetErrorString(error: cutensorStatus_t) -> *const ::core::ffi::c_char;
        }
        cutensorGetErrorString(error)
    }
}
pub unsafe fn cutensorGetVersion() -> usize {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> usize;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorGetVersion") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorGetVersion() -> usize;
        }
        cutensorGetVersion()
    }
}
pub unsafe fn cutensorHandleReadPlanCacheFromFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char, numCachelinesRead: *mut u32) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *const ::core::ffi::c_char, *mut u32) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorHandleReadPlanCacheFromFile") });
        _f(handle, filename, numCachelinesRead)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorHandleReadPlanCacheFromFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char, numCachelinesRead: *mut u32) -> cutensorStatus_t;
        }
        cutensorHandleReadPlanCacheFromFile(handle, filename, numCachelinesRead)
    }
}
pub unsafe fn cutensorHandleResizePlanCache(handle: cutensorHandle_t, numEntries: u32) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, u32) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorHandleResizePlanCache") });
        _f(handle, numEntries)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorHandleResizePlanCache(handle: cutensorHandle_t, numEntries: u32) -> cutensorStatus_t;
        }
        cutensorHandleResizePlanCache(handle, numEntries)
    }
}
pub unsafe fn cutensorHandleWritePlanCacheToFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *const ::core::ffi::c_char) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorHandleWritePlanCacheToFile") });
        _f(handle, filename)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorHandleWritePlanCacheToFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char) -> cutensorStatus_t;
        }
        cutensorHandleWritePlanCacheToFile(handle, filename)
    }
}
pub unsafe fn cutensorLoggerForceDisable() -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorLoggerForceDisable") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorLoggerForceDisable() -> cutensorStatus_t;
        }
        cutensorLoggerForceDisable()
    }
}
pub unsafe fn cutensorLoggerOpenFile(logFile: *const ::core::ffi::c_char) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorLoggerOpenFile") });
        _f(logFile)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorLoggerOpenFile(logFile: *const ::core::ffi::c_char) -> cutensorStatus_t;
        }
        cutensorLoggerOpenFile(logFile)
    }
}
pub unsafe fn cutensorLoggerSetCallback(callback: cutensorLoggerCallback_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorLoggerCallback_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorLoggerSetCallback") });
        _f(callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorLoggerSetCallback(callback: cutensorLoggerCallback_t) -> cutensorStatus_t;
        }
        cutensorLoggerSetCallback(callback)
    }
}
pub unsafe fn cutensorLoggerSetFile(file: *mut FILE) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut FILE) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorLoggerSetFile") });
        _f(file)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorLoggerSetFile(file: *mut FILE) -> cutensorStatus_t;
        }
        cutensorLoggerSetFile(file)
    }
}
pub unsafe fn cutensorLoggerSetLevel(level: i32) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i32) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorLoggerSetLevel") });
        _f(level)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorLoggerSetLevel(level: i32) -> cutensorStatus_t;
        }
        cutensorLoggerSetLevel(level)
    }
}
pub unsafe fn cutensorLoggerSetMask(mask: i32) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(i32) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorLoggerSetMask") });
        _f(mask)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorLoggerSetMask(mask: i32) -> cutensorStatus_t;
        }
        cutensorLoggerSetMask(mask)
    }
}
pub unsafe fn cutensorOperationDescriptorGetAttribute(handle: cutensorHandle_t, desc: cutensorOperationDescriptor_t, attr: cutensorOperationDescriptorAttribute_t, buf: *mut ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorOperationDescriptor_t, cutensorOperationDescriptorAttribute_t, *mut ::core::ffi::c_void, usize) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorOperationDescriptorGetAttribute") });
        _f(handle, desc, attr, buf, sizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorOperationDescriptorGetAttribute(handle: cutensorHandle_t, desc: cutensorOperationDescriptor_t, attr: cutensorOperationDescriptorAttribute_t, buf: *mut ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t;
        }
        cutensorOperationDescriptorGetAttribute(handle, desc, attr, buf, sizeInBytes)
    }
}
pub unsafe fn cutensorOperationDescriptorSetAttribute(handle: cutensorHandle_t, desc: cutensorOperationDescriptor_t, attr: cutensorOperationDescriptorAttribute_t, buf: *const ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorOperationDescriptor_t, cutensorOperationDescriptorAttribute_t, *const ::core::ffi::c_void, usize) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorOperationDescriptorSetAttribute") });
        _f(handle, desc, attr, buf, sizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorOperationDescriptorSetAttribute(handle: cutensorHandle_t, desc: cutensorOperationDescriptor_t, attr: cutensorOperationDescriptorAttribute_t, buf: *const ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t;
        }
        cutensorOperationDescriptorSetAttribute(handle, desc, attr, buf, sizeInBytes)
    }
}
pub unsafe fn cutensorPermute(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, B: *mut ::core::ffi::c_void, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorPermute") });
        _f(handle, plan, alpha, A, B, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorPermute(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, B: *mut ::core::ffi::c_void, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorPermute(handle, plan, alpha, A, B, stream)
    }
}
pub unsafe fn cutensorPlanGetAttribute(handle: cutensorHandle_t, plan: cutensorPlan_t, attr: cutensorPlanAttribute_t, buf: *mut ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, cutensorPlanAttribute_t, *mut ::core::ffi::c_void, usize) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorPlanGetAttribute") });
        _f(handle, plan, attr, buf, sizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorPlanGetAttribute(handle: cutensorHandle_t, plan: cutensorPlan_t, attr: cutensorPlanAttribute_t, buf: *mut ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t;
        }
        cutensorPlanGetAttribute(handle, plan, attr, buf, sizeInBytes)
    }
}
#[cfg(any(feature = "cutensor-02004", feature = "cutensor-02005", feature = "cutensor-02006"))]
pub unsafe fn cutensorPlanPreferenceGetAttribute(handle: cutensorHandle_t, pref: cutensorPlanPreference_t, attr: cutensorPlanPreferenceAttribute_t, buf: *mut ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlanPreference_t, cutensorPlanPreferenceAttribute_t, *mut ::core::ffi::c_void, usize) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorPlanPreferenceGetAttribute") });
        _f(handle, pref, attr, buf, sizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorPlanPreferenceGetAttribute(handle: cutensorHandle_t, pref: cutensorPlanPreference_t, attr: cutensorPlanPreferenceAttribute_t, buf: *mut ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t;
        }
        cutensorPlanPreferenceGetAttribute(handle, pref, attr, buf, sizeInBytes)
    }
}
pub unsafe fn cutensorPlanPreferenceSetAttribute(handle: cutensorHandle_t, pref: cutensorPlanPreference_t, attr: cutensorPlanPreferenceAttribute_t, buf: *const ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlanPreference_t, cutensorPlanPreferenceAttribute_t, *const ::core::ffi::c_void, usize) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorPlanPreferenceSetAttribute") });
        _f(handle, pref, attr, buf, sizeInBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorPlanPreferenceSetAttribute(handle: cutensorHandle_t, pref: cutensorPlanPreference_t, attr: cutensorPlanPreferenceAttribute_t, buf: *const ::core::ffi::c_void, sizeInBytes: usize) -> cutensorStatus_t;
        }
        cutensorPlanPreferenceSetAttribute(handle, pref, attr, buf, sizeInBytes)
    }
}
pub unsafe fn cutensorReadKernelCacheFromFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *const ::core::ffi::c_char) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorReadKernelCacheFromFile") });
        _f(handle, filename)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorReadKernelCacheFromFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char) -> cutensorStatus_t;
        }
        cutensorReadKernelCacheFromFile(handle, filename)
    }
}
pub unsafe fn cutensorReduce(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, cutensorPlan_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *const ::core::ffi::c_void, *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, u64, cudaStream_t) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorReduce") });
        _f(handle, plan, alpha, A, beta, C, D, workspace, workspaceSize, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorReduce(handle: cutensorHandle_t, plan: cutensorPlan_t, alpha: *const ::core::ffi::c_void, A: *const ::core::ffi::c_void, beta: *const ::core::ffi::c_void, C: *const ::core::ffi::c_void, D: *mut ::core::ffi::c_void, workspace: *mut ::core::ffi::c_void, workspaceSize: u64, stream: cudaStream_t) -> cutensorStatus_t;
        }
        cutensorReduce(handle, plan, alpha, A, beta, C, D, workspace, workspaceSize, stream)
    }
}
pub unsafe fn cutensorWriteKernelCacheToFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char) -> cutensorStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cutensorHandle_t, *const ::core::ffi::c_char) -> cutensorStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cutensorWriteKernelCacheToFile") });
        _f(handle, filename)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cutensorWriteKernelCacheToFile(handle: cutensorHandle_t, filename: *const ::core::ffi::c_char) -> cutensorStatus_t;
        }
        cutensorWriteKernelCacheToFile(handle, filename)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cutensor"];
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
        let lib_names = std::vec!["cutensor"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
