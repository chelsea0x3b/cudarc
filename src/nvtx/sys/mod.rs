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
pub type nvtxDomainCreateA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> nvtxDomainHandle_t>;
pub type nvtxDomainCreateW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t) -> nvtxDomainHandle_t>;
pub type nvtxDomainDestroy_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t)>;
pub type nvtxDomainHandle_t = *mut nvtxDomainRegistration;
pub type nvtxDomainMarkEx_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t)>;
pub type nvtxDomainNameCategoryA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, category: u32, name: *const ::core::ffi::c_char)>;
pub type nvtxDomainNameCategoryW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, category: u32, name: *const wchar_t)>;
pub type nvtxDomainRangeEnd_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, id: nvtxRangeId_t)>;
pub type nvtxDomainRangePop_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int>;
pub type nvtxDomainRangePushEx_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int>;
pub type nvtxDomainRangeStartEx_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t>;
pub type nvtxDomainRegisterStringA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, string: *const ::core::ffi::c_char) -> nvtxStringHandle_t>;
pub type nvtxDomainRegisterStringW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, string: *const wchar_t) -> nvtxStringHandle_t>;
pub type nvtxDomainRegistration = nvtxDomainRegistration_st;
pub type nvtxDomainResourceCreate_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, attribs: *mut nvtxResourceAttributes_t) -> nvtxResourceHandle_t>;
pub type nvtxDomainResourceDestroy_impl_fntype = ::core::option::Option<unsafe extern "C" fn(resource: nvtxResourceHandle_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxDomainSyncUserAcquireFailed_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxDomainSyncUserAcquireFailed_impl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxDomainSyncUserAcquireStart_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxDomainSyncUserAcquireStart_impl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxDomainSyncUserAcquireSuccess_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxDomainSyncUserAcquireSuccess_impl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxDomainSyncUserCreate_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, attribs: *const nvtx_nvtxSyncUserAttributes_t) -> nvtx_nvtxSyncUser_t>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxDomainSyncUserCreate_impl_fntype = ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, attribs: *const nvtxSyncUserAttributes_t) -> nvtxSyncUser_t>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxDomainSyncUserDestroy_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxDomainSyncUserDestroy_impl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxDomainSyncUserReleasing_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxDomainSyncUserReleasing_impl_fntype = ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
pub type nvtxEventAttributes_t = nvtxEventAttributes_v2;
pub type nvtxInitialize_impl_fntype = ::core::option::Option<unsafe extern "C" fn(reserved: *const ::core::ffi::c_void)>;
pub type nvtxMarkA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const ::core::ffi::c_char)>;
pub type nvtxMarkEx_impl_fntype = ::core::option::Option<unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t)>;
pub type nvtxMarkW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t)>;
pub type nvtxNameCategoryA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(category: u32, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCategoryW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(category: u32, name: *const wchar_t)>;
pub type nvtxNameClCommandQueueA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(command_queue: nvtx_cl_command_queue, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClCommandQueueW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(command_queue: nvtx_cl_command_queue, name: *const wchar_t)>;
pub type nvtxNameClContextA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(context: nvtx_cl_context, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClContextW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(context: nvtx_cl_context, name: *const wchar_t)>;
pub type nvtxNameClDeviceA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(device: nvtx_cl_device_id, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClDeviceW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(device: nvtx_cl_device_id, name: *const wchar_t)>;
pub type nvtxNameClEventA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(evnt: nvtx_cl_event, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClEventW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(evnt: nvtx_cl_event, name: *const wchar_t)>;
pub type nvtxNameClMemObjectA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(memobj: nvtx_cl_mem, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClMemObjectW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(memobj: nvtx_cl_mem, name: *const wchar_t)>;
pub type nvtxNameClProgramA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(program: nvtx_cl_program, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClProgramW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(program: nvtx_cl_program, name: *const wchar_t)>;
pub type nvtxNameClSamplerA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(sampler: nvtx_cl_sampler, name: *const ::core::ffi::c_char)>;
pub type nvtxNameClSamplerW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(sampler: nvtx_cl_sampler, name: *const wchar_t)>;
pub type nvtxNameCuContextA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(context: nvtx_CUcontext, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCuContextW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(context: nvtx_CUcontext, name: *const wchar_t)>;
pub type nvtxNameCuDeviceA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(device: nvtx_CUdevice, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCuDeviceW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(device: nvtx_CUdevice, name: *const wchar_t)>;
pub type nvtxNameCuEventA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(event: nvtx_CUevent, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCuEventW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(event: nvtx_CUevent, name: *const wchar_t)>;
pub type nvtxNameCuStreamA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(stream: nvtx_CUstream, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCuStreamW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(stream: nvtx_CUstream, name: *const wchar_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxNameCudaDeviceA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const ::core::ffi::c_char)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxNameCudaDeviceA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const ::core::ffi::c_char)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtxNameCudaDeviceW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const wchar_t)>;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxNameCudaDeviceW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const wchar_t)>;
pub type nvtxNameCudaEventA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(event: nvtx_cudaEvent_t, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCudaEventW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(event: nvtx_cudaEvent_t, name: *const wchar_t)>;
pub type nvtxNameCudaStreamA_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(stream: nvtx_cudaStream_t, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCudaStreamW_fakeimpl_fntype = ::core::option::Option<unsafe extern "C" fn(stream: nvtx_cudaStream_t, name: *const wchar_t)>;
pub type nvtxNameOsThreadA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(threadId: u32, name: *const ::core::ffi::c_char)>;
pub type nvtxNameOsThreadW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(threadId: u32, name: *const wchar_t)>;
pub type nvtxRangeEnd_impl_fntype = ::core::option::Option<unsafe extern "C" fn(id: nvtxRangeId_t)>;
pub type nvtxRangeId_t = u64;
pub type nvtxRangePop_impl_fntype = ::core::option::Option<unsafe extern "C" fn() -> ::core::ffi::c_int>;
pub type nvtxRangePushA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int>;
pub type nvtxRangePushEx_impl_fntype = ::core::option::Option<unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int>;
pub type nvtxRangePushW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t) -> ::core::ffi::c_int>;
pub type nvtxRangeStartA_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> nvtxRangeId_t>;
pub type nvtxRangeStartEx_impl_fntype = ::core::option::Option<unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t>;
pub type nvtxRangeStartW_impl_fntype = ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t) -> nvtxRangeId_t>;
pub type nvtxResourceAttributes_t = nvtxResourceAttributes_v0;
pub type nvtxResourceHandle_t = *mut nvtxResourceHandle;
pub type nvtxStringHandle_t = *mut nvtxStringRegistration;
pub type nvtxStringRegistration = nvtxStringRegistration_st;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxSyncUserAttributes_t = nvtxSyncUserAttributes_v0;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type nvtxSyncUser_t = *mut nvtxSyncUser;
pub type nvtx_CUcontext = *mut ::core::ffi::c_void;
pub type nvtx_CUdevice = ::core::ffi::c_int;
pub type nvtx_CUevent = *mut ::core::ffi::c_void;
pub type nvtx_CUstream = *mut ::core::ffi::c_void;
pub type nvtx_cl_command_queue = *mut ::core::ffi::c_void;
pub type nvtx_cl_context = *mut ::core::ffi::c_void;
pub type nvtx_cl_device_id = *mut ::core::ffi::c_void;
pub type nvtx_cl_event = *mut ::core::ffi::c_void;
pub type nvtx_cl_kernel = *mut ::core::ffi::c_void;
pub type nvtx_cl_mem = *mut ::core::ffi::c_void;
pub type nvtx_cl_platform_id = *mut ::core::ffi::c_void;
pub type nvtx_cl_program = *mut ::core::ffi::c_void;
pub type nvtx_cl_sampler = *mut ::core::ffi::c_void;
pub type nvtx_cudaEvent_t = *mut ::core::ffi::c_void;
pub type nvtx_cudaStream_t = *mut ::core::ffi::c_void;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtx_nvtxSyncUserAttributes_t = ::core::ffi::c_void;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type nvtx_nvtxSyncUser_t = *mut ::core::ffi::c_void;
pub type wchar_t = ::core::ffi::c_int;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxColorType_t {
    NVTX_COLOR_UNKNOWN = 0,
    NVTX_COLOR_ARGB = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxMessageType_t {
    NVTX_MESSAGE_UNKNOWN = 0,
    NVTX_MESSAGE_TYPE_ASCII = 1,
    NVTX_MESSAGE_TYPE_UNICODE = 2,
    NVTX_MESSAGE_TYPE_REGISTERED = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxPayloadType_t {
    NVTX_PAYLOAD_UNKNOWN = 0,
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1,
    NVTX_PAYLOAD_TYPE_INT64 = 2,
    NVTX_PAYLOAD_TYPE_DOUBLE = 3,
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 = 4,
    NVTX_PAYLOAD_TYPE_INT32 = 5,
    NVTX_PAYLOAD_TYPE_FLOAT = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxResourceGenericType_t {
    NVTX_RESOURCE_TYPE_UNKNOWN = 0,
    NVTX_RESOURCE_TYPE_GENERIC_POINTER = 65537,
    NVTX_RESOURCE_TYPE_GENERIC_HANDLE = 65538,
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE = 65539,
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX = 65540,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxDomainRegistration_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvtxEventAttributes_v2 {
    pub version: u16,
    pub size: u16,
    pub category: u32,
    pub colorType: i32,
    pub color: u32,
    pub payloadType: i32,
    pub reserved0: i32,
    pub payload: nvtxEventAttributes_v2_payload_t,
    pub messageType: i32,
    pub message: nvtxMessageValue_t,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvtxResourceAttributes_v0 {
    pub version: u16,
    pub size: u16,
    pub identifierType: i32,
    pub identifier: nvtxResourceAttributes_v0_identifier_t,
    pub messageType: i32,
    pub message: nvtxMessageValue_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxResourceHandle {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxStringRegistration_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxSyncUser {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxSyncUserAttributes_v0 {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union nvtxEventAttributes_v2_payload_t {
    pub ullValue: u64,
    pub llValue: i64,
    pub dValue: f64,
    pub uiValue: u32,
    pub iValue: i32,
    pub fValue: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union nvtxMessageValue_t {
    pub ascii: *const ::core::ffi::c_char,
    pub unicode: *const wchar_t,
    pub registered: nvtxStringHandle_t,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union nvtxResourceAttributes_v0_identifier_t {
    pub pValue: *const ::core::ffi::c_void,
    pub ullValue: u64,
}
pub unsafe fn nvtxDomainCreateA(name: *const ::core::ffi::c_char) -> nvtxDomainHandle_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char) -> nvtxDomainHandle_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainCreateA") });
        _f(name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainCreateA(name: *const ::core::ffi::c_char) -> nvtxDomainHandle_t;
        }
        nvtxDomainCreateA(name)
    }
}
pub unsafe fn nvtxDomainCreateW(name: *const wchar_t) -> nvtxDomainHandle_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const wchar_t) -> nvtxDomainHandle_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainCreateW") });
        _f(name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainCreateW(name: *const wchar_t) -> nvtxDomainHandle_t;
        }
        nvtxDomainCreateW(name)
    }
}
pub unsafe fn nvtxDomainDestroy(domain: nvtxDomainHandle_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainDestroy") });
        _f(domain)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainDestroy(domain: nvtxDomainHandle_t);
        }
        nvtxDomainDestroy(domain)
    }
}
pub unsafe fn nvtxDomainMarkEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, *const nvtxEventAttributes_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainMarkEx") });
        _f(domain, eventAttrib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainMarkEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t);
        }
        nvtxDomainMarkEx(domain, eventAttrib)
    }
}
pub unsafe fn nvtxDomainNameCategoryA(domain: nvtxDomainHandle_t, category: u32, name: *const ::core::ffi::c_char) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, u32, *const ::core::ffi::c_char);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainNameCategoryA") });
        _f(domain, category, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainNameCategoryA(domain: nvtxDomainHandle_t, category: u32, name: *const ::core::ffi::c_char);
        }
        nvtxDomainNameCategoryA(domain, category, name)
    }
}
pub unsafe fn nvtxDomainNameCategoryW(domain: nvtxDomainHandle_t, category: u32, name: *const wchar_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, u32, *const wchar_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainNameCategoryW") });
        _f(domain, category, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainNameCategoryW(domain: nvtxDomainHandle_t, category: u32, name: *const wchar_t);
        }
        nvtxDomainNameCategoryW(domain, category, name)
    }
}
pub unsafe fn nvtxDomainRangeEnd(domain: nvtxDomainHandle_t, id: nvtxRangeId_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, nvtxRangeId_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainRangeEnd") });
        _f(domain, id)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainRangeEnd(domain: nvtxDomainHandle_t, id: nvtxRangeId_t);
        }
        nvtxDomainRangeEnd(domain, id)
    }
}
pub unsafe fn nvtxDomainRangePop(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t) -> ::core::ffi::c_int;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainRangePop") });
        _f(domain)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainRangePop(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int;
        }
        nvtxDomainRangePop(domain)
    }
}
pub unsafe fn nvtxDomainRangePushEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, *const nvtxEventAttributes_t) -> ::core::ffi::c_int;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainRangePushEx") });
        _f(domain, eventAttrib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainRangePushEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int;
        }
        nvtxDomainRangePushEx(domain, eventAttrib)
    }
}
pub unsafe fn nvtxDomainRangeStartEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, *const nvtxEventAttributes_t) -> nvtxRangeId_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainRangeStartEx") });
        _f(domain, eventAttrib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainRangeStartEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t;
        }
        nvtxDomainRangeStartEx(domain, eventAttrib)
    }
}
pub unsafe fn nvtxDomainRegisterStringA(domain: nvtxDomainHandle_t, string: *const ::core::ffi::c_char) -> nvtxStringHandle_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, *const ::core::ffi::c_char) -> nvtxStringHandle_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainRegisterStringA") });
        _f(domain, string)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainRegisterStringA(domain: nvtxDomainHandle_t, string: *const ::core::ffi::c_char) -> nvtxStringHandle_t;
        }
        nvtxDomainRegisterStringA(domain, string)
    }
}
pub unsafe fn nvtxDomainRegisterStringW(domain: nvtxDomainHandle_t, string: *const wchar_t) -> nvtxStringHandle_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, *const wchar_t) -> nvtxStringHandle_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainRegisterStringW") });
        _f(domain, string)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainRegisterStringW(domain: nvtxDomainHandle_t, string: *const wchar_t) -> nvtxStringHandle_t;
        }
        nvtxDomainRegisterStringW(domain, string)
    }
}
pub unsafe fn nvtxDomainResourceCreate(domain: nvtxDomainHandle_t, attribs: *mut nvtxResourceAttributes_t) -> nvtxResourceHandle_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxDomainHandle_t, *mut nvtxResourceAttributes_t) -> nvtxResourceHandle_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainResourceCreate") });
        _f(domain, attribs)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainResourceCreate(domain: nvtxDomainHandle_t, attribs: *mut nvtxResourceAttributes_t) -> nvtxResourceHandle_t;
        }
        nvtxDomainResourceCreate(domain, attribs)
    }
}
pub unsafe fn nvtxDomainResourceDestroy(resource: nvtxResourceHandle_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxResourceHandle_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxDomainResourceDestroy") });
        _f(resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxDomainResourceDestroy(resource: nvtxResourceHandle_t);
        }
        nvtxDomainResourceDestroy(resource)
    }
}
pub unsafe fn nvtxMarkA(message: *const ::core::ffi::c_char) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxMarkA") });
        _f(message)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxMarkA(message: *const ::core::ffi::c_char);
        }
        nvtxMarkA(message)
    }
}
pub unsafe fn nvtxMarkEx(eventAttrib: *const nvtxEventAttributes_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const nvtxEventAttributes_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxMarkEx") });
        _f(eventAttrib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxMarkEx(eventAttrib: *const nvtxEventAttributes_t);
        }
        nvtxMarkEx(eventAttrib)
    }
}
pub unsafe fn nvtxMarkW(message: *const wchar_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const wchar_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxMarkW") });
        _f(message)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxMarkW(message: *const wchar_t);
        }
        nvtxMarkW(message)
    }
}
pub unsafe fn nvtxNameCategoryA(category: u32, name: *const ::core::ffi::c_char) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(u32, *const ::core::ffi::c_char);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxNameCategoryA") });
        _f(category, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxNameCategoryA(category: u32, name: *const ::core::ffi::c_char);
        }
        nvtxNameCategoryA(category, name)
    }
}
pub unsafe fn nvtxNameCategoryW(category: u32, name: *const wchar_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(u32, *const wchar_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxNameCategoryW") });
        _f(category, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxNameCategoryW(category: u32, name: *const wchar_t);
        }
        nvtxNameCategoryW(category, name)
    }
}
pub unsafe fn nvtxNameOsThreadA(threadId: u32, name: *const ::core::ffi::c_char) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(u32, *const ::core::ffi::c_char);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxNameOsThreadA") });
        _f(threadId, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxNameOsThreadA(threadId: u32, name: *const ::core::ffi::c_char);
        }
        nvtxNameOsThreadA(threadId, name)
    }
}
pub unsafe fn nvtxNameOsThreadW(threadId: u32, name: *const wchar_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(u32, *const wchar_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxNameOsThreadW") });
        _f(threadId, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxNameOsThreadW(threadId: u32, name: *const wchar_t);
        }
        nvtxNameOsThreadW(threadId, name)
    }
}
pub unsafe fn nvtxRangeEnd(id: nvtxRangeId_t) {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(nvtxRangeId_t);
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangeEnd") });
        _f(id)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangeEnd(id: nvtxRangeId_t);
        }
        nvtxRangeEnd(id)
    }
}
pub unsafe fn nvtxRangePop() -> ::core::ffi::c_int {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> ::core::ffi::c_int;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangePop") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangePop() -> ::core::ffi::c_int;
        }
        nvtxRangePop()
    }
}
pub unsafe fn nvtxRangePushA(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char) -> ::core::ffi::c_int;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangePushA") });
        _f(message)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangePushA(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int;
        }
        nvtxRangePushA(message)
    }
}
pub unsafe fn nvtxRangePushEx(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const nvtxEventAttributes_t) -> ::core::ffi::c_int;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangePushEx") });
        _f(eventAttrib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangePushEx(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int;
        }
        nvtxRangePushEx(eventAttrib)
    }
}
pub unsafe fn nvtxRangePushW(message: *const wchar_t) -> ::core::ffi::c_int {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const wchar_t) -> ::core::ffi::c_int;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangePushW") });
        _f(message)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangePushW(message: *const wchar_t) -> ::core::ffi::c_int;
        }
        nvtxRangePushW(message)
    }
}
pub unsafe fn nvtxRangeStartA(message: *const ::core::ffi::c_char) -> nvtxRangeId_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char) -> nvtxRangeId_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangeStartA") });
        _f(message)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangeStartA(message: *const ::core::ffi::c_char) -> nvtxRangeId_t;
        }
        nvtxRangeStartA(message)
    }
}
pub unsafe fn nvtxRangeStartEx(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const nvtxEventAttributes_t) -> nvtxRangeId_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangeStartEx") });
        _f(eventAttrib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangeStartEx(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t;
        }
        nvtxRangeStartEx(eventAttrib)
    }
}
pub unsafe fn nvtxRangeStartW(message: *const wchar_t) -> nvtxRangeId_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const wchar_t) -> nvtxRangeId_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("nvtxRangeStartW") });
        _f(message)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn nvtxRangeStartW(message: *const wchar_t) -> nvtxRangeId_t;
        }
        nvtxRangeStartW(message)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["nvToolsExt"];
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
        let lib_names = std::vec!["nvToolsExt"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
