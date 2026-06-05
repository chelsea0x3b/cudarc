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
pub type nvrtcProgram = *mut _nvrtcProgram;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
    NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
    NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
    NVRTC_ERROR_PCH_CREATE = 15,
    NVRTC_ERROR_CANCELLED = 16,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
    NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
    NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
    NVRTC_ERROR_PCH_CREATE = 15,
    NVRTC_ERROR_CANCELLED = 16,
    NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED = 17,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
    NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
    NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
    NVRTC_ERROR_PCH_CREATE = 15,
    NVRTC_ERROR_CANCELLED = 16,
    NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED = 17,
    NVRTC_ERROR_BUSY = 18,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _nvrtcProgram {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct nvrtcBundledHeadersInfo {
    pub available: ::core::ffi::c_int,
    pub compressedSize: usize,
    pub uncompressedSize: usize,
    pub cudaVersionMajor: ::core::ffi::c_int,
    pub cudaVersionMinor: ::core::ffi::c_int,
    pub numFiles: ::core::ffi::c_uint,
}
pub unsafe fn nvrtcAddNameExpression(prog: nvrtcProgram, name_expression: *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcAddNameExpression") });
        _f(prog, name_expression)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcAddNameExpression(prog: nvrtcProgram, name_expression: *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcAddNameExpression(prog, name_expression)
    }
}
pub unsafe fn nvrtcCompileProgram(prog: nvrtcProgram, numOptions: ::core::ffi::c_int, options: *const *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, ::core::ffi::c_int, *const *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcCompileProgram") });
        _f(prog, numOptions, options)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcCompileProgram(prog: nvrtcProgram, numOptions: ::core::ffi::c_int, options: *const *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcCompileProgram(prog, numOptions, options)
    }
}
pub unsafe fn nvrtcCreateProgram(prog: *mut nvrtcProgram, src: *const ::core::ffi::c_char, name: *const ::core::ffi::c_char, numHeaders: ::core::ffi::c_int, headers: *const *const ::core::ffi::c_char, includeNames: *const *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut nvrtcProgram, *const ::core::ffi::c_char, *const ::core::ffi::c_char, ::core::ffi::c_int, *const *const ::core::ffi::c_char, *const *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcCreateProgram") });
        _f(prog, src, name, numHeaders, headers, includeNames)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcCreateProgram(prog: *mut nvrtcProgram, src: *const ::core::ffi::c_char, name: *const ::core::ffi::c_char, numHeaders: ::core::ffi::c_int, headers: *const *const ::core::ffi::c_char, includeNames: *const *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcCreateProgram(prog, src, name, numHeaders, headers, includeNames)
    }
}
pub unsafe fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut nvrtcProgram) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcDestroyProgram") });
        _f(prog)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
        }
        nvrtcDestroyProgram(prog)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn nvrtcGetBundledHeadersInfo(info: *mut nvrtcBundledHeadersInfo, errorLog: *mut *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut nvrtcBundledHeadersInfo, *mut *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetBundledHeadersInfo") });
        _f(info, errorLog)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetBundledHeadersInfo(info: *mut nvrtcBundledHeadersInfo, errorLog: *mut *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetBundledHeadersInfo(info, errorLog)
    }
}
pub unsafe fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetCUBIN") });
        _f(prog, cubin)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetCUBIN(prog, cubin)
    }
}
pub unsafe fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetCUBINSize") });
        _f(prog, cubinSizeRet)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult;
        }
        nvrtcGetCUBINSize(prog, cubinSizeRet)
    }
}
pub unsafe fn nvrtcGetErrorString(result: nvrtcResult) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcResult) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetErrorString") });
        _f(result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetErrorString(result: nvrtcResult) -> *const ::core::ffi::c_char;
        }
        nvrtcGetErrorString(result)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn nvrtcGetLTOIR(prog: nvrtcProgram, LTOIR: *mut ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetLTOIR") });
        _f(prog, LTOIR)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetLTOIR(prog: nvrtcProgram, LTOIR: *mut ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetLTOIR(prog, LTOIR)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetLTOIRSize") });
        _f(prog, LTOIRSizeRet)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult;
        }
        nvrtcGetLTOIRSize(prog, LTOIRSizeRet)
    }
}
pub unsafe fn nvrtcGetLoweredName(prog: nvrtcProgram, name_expression: *const ::core::ffi::c_char, lowered_name: *mut *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *const ::core::ffi::c_char, *mut *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetLoweredName") });
        _f(prog, name_expression, lowered_name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetLoweredName(prog: nvrtcProgram, name_expression: *const ::core::ffi::c_char, lowered_name: *mut *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetLoweredName(prog, name_expression, lowered_name)
    }
}
pub unsafe fn nvrtcGetNumSupportedArchs(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetNumSupportedArchs") });
        _f(numArchs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetNumSupportedArchs(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult;
        }
        nvrtcGetNumSupportedArchs(numArchs)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetOptiXIR") });
        _f(prog, optixir)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetOptiXIR(prog, optixir)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn nvrtcGetOptiXIRSize(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetOptiXIRSize") });
        _f(prog, optixirSizeRet)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetOptiXIRSize(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult;
        }
        nvrtcGetOptiXIRSize(prog, optixirSizeRet)
    }
}
pub unsafe fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetPTX") });
        _f(prog, ptx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetPTX(prog, ptx)
    }
}
pub unsafe fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetPTXSize") });
        _f(prog, ptxSizeRet)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult;
        }
        nvrtcGetPTXSize(prog, ptxSizeRet)
    }
}
pub unsafe fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetProgramLog") });
        _f(prog, log)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetProgramLog(prog, log)
    }
}
pub unsafe fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetProgramLogSize") });
        _f(prog, logSizeRet)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult;
        }
        nvrtcGetProgramLogSize(prog, logSizeRet)
    }
}
pub unsafe fn nvrtcGetSupportedArchs(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetSupportedArchs") });
        _f(supportedArchs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetSupportedArchs(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult;
        }
        nvrtcGetSupportedArchs(supportedArchs)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn nvrtcGetTileIR(prog: nvrtcProgram, TileIR: *mut ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetTileIR") });
        _f(prog, TileIR)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetTileIR(prog: nvrtcProgram, TileIR: *mut ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcGetTileIR(prog, TileIR)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn nvrtcGetTileIRSize(prog: nvrtcProgram, TileIRSizeRet: *mut usize) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcGetTileIRSize") });
        _f(prog, TileIRSizeRet)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcGetTileIRSize(prog: nvrtcProgram, TileIRSizeRet: *mut usize) -> nvrtcResult;
        }
        nvrtcGetTileIRSize(prog, TileIRSizeRet)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn nvrtcInstallBundledHeaders(installPath: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint, errorLog: *mut *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, ::core::ffi::c_uint, *mut *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcInstallBundledHeaders") });
        _f(installPath, flags, errorLog)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcInstallBundledHeaders(installPath: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint, errorLog: *mut *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcInstallBundledHeaders(installPath, flags, errorLog)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn nvrtcRemoveBundledHeaders(installPath: *const ::core::ffi::c_char, errorLog: *mut *const ::core::ffi::c_char) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *const ::core::ffi::c_char) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcRemoveBundledHeaders") });
        _f(installPath, errorLog)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcRemoveBundledHeaders(installPath: *const ::core::ffi::c_char, errorLog: *mut *const ::core::ffi::c_char) -> nvrtcResult;
        }
        nvrtcRemoveBundledHeaders(installPath, errorLog)
    }
}
pub unsafe fn nvrtcVersion(major: *mut ::core::ffi::c_int, minor: *mut ::core::ffi::c_int) -> nvrtcResult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> nvrtcResult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvrtcVersion") });
        _f(major, minor)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvrtcVersion(major: *mut ::core::ffi::c_int, minor: *mut ::core::ffi::c_int) -> nvrtcResult;
        }
        nvrtcVersion(major, minor)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["nvrtc"];
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
        let lib_names = std::vec!["nvrtc"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
