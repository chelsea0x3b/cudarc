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
pub type cudaStream_t = *mut CUstream_st;
pub type ncclComm_t = *mut ncclComm;
#[cfg(any(feature = "nccl-02022", feature = "nccl-02024", feature = "nccl-02025", feature = "nccl-02026"))]
pub type ncclConfig_t = ncclConfig_v21700;
#[cfg(any(feature = "nccl-02027"))]
pub type ncclConfig_t = ncclConfig_v22700;
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub type ncclConfig_t = ncclConfig_v22800;
#[cfg(any(feature = "nccl-02030"))]
pub type ncclParamHandle_t = ncclParamHandle;
pub type ncclSimInfo_t = ncclSimInfo_v22200;
#[cfg(any(feature = "nccl-02027"))]
pub type ncclWindow_t = *mut ncclWindow;
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub type ncclWindow_t = *mut ncclWindow_vidmem;
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclCommMemStat_t {
    ncclStatGpuMemSuspend = 0,
    ncclStatGpuMemSuspended = 1,
    ncclStatGpuMemPersist = 2,
    ncclStatGpuMemTotal = 3,
}
#[cfg(any(feature = "nccl-02022"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclDataType_t {
    ncclInt8 = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclFloat32 = 7,
    ncclFloat64 = 8,
    ncclNumTypes = 9,
}
#[cfg(any(feature = "nccl-02024", feature = "nccl-02025", feature = "nccl-02026", feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclDataType_t {
    ncclInt8 = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclFloat32 = 7,
    ncclFloat64 = 8,
    ncclBfloat16 = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes = 12,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclRedOp_dummy_t {
    ncclNumOps_dummy = 5,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclRedOp_t {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5,
    ncclMaxRedOp = 2147483647,
}
#[cfg(any(feature = "nccl-02022", feature = "nccl-02024", feature = "nccl-02025", feature = "nccl-02026", feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclResult_t {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8,
}
#[cfg(any(feature = "nccl-02030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclResult_t {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclTimeout = 8,
    ncclNumResults = 9,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclScalarResidence_t {
    ncclScalarDevice = 0,
    ncclScalarHostImmediate = 1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ncclComm {
    _unused: [u8; 0],
}
#[cfg(any(feature = "nccl-02022", feature = "nccl-02024", feature = "nccl-02025"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v21700 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02026"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v21700 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
    pub trafficClass: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02027"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v22700 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
    pub trafficClass: ::core::ffi::c_int,
    pub commName: *const ::core::ffi::c_char,
    pub collnetEnable: ::core::ffi::c_int,
    pub CTAPolicy: ::core::ffi::c_int,
    pub shrinkShare: ::core::ffi::c_int,
    pub nvlsCTAs: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02028"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v22800 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
    pub trafficClass: ::core::ffi::c_int,
    pub commName: *const ::core::ffi::c_char,
    pub collnetEnable: ::core::ffi::c_int,
    pub CTAPolicy: ::core::ffi::c_int,
    pub shrinkShare: ::core::ffi::c_int,
    pub nvlsCTAs: ::core::ffi::c_int,
    pub nChannelsPerNetPeer: ::core::ffi::c_int,
    pub nvlinkCentricSched: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02029"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v22800 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
    pub trafficClass: ::core::ffi::c_int,
    pub commName: *const ::core::ffi::c_char,
    pub collnetEnable: ::core::ffi::c_int,
    pub CTAPolicy: ::core::ffi::c_int,
    pub shrinkShare: ::core::ffi::c_int,
    pub nvlsCTAs: ::core::ffi::c_int,
    pub nChannelsPerNetPeer: ::core::ffi::c_int,
    pub nvlinkCentricSched: ::core::ffi::c_int,
    pub graphUsageMode: ::core::ffi::c_int,
    pub numRmaCtx: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v22800 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
    pub trafficClass: ::core::ffi::c_int,
    pub commName: *const ::core::ffi::c_char,
    pub collnetEnable: ::core::ffi::c_int,
    pub CTAPolicy: ::core::ffi::c_int,
    pub shrinkShare: ::core::ffi::c_int,
    pub nvlsCTAs: ::core::ffi::c_int,
    pub nChannelsPerNetPeer: ::core::ffi::c_int,
    pub nvlinkCentricSched: ::core::ffi::c_int,
    pub graphUsageMode: ::core::ffi::c_int,
    pub numRmaCtx: ::core::ffi::c_int,
    pub maxP2pPeers: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ncclParamHandle {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct ncclSimInfo_v22200 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub estimatedTime: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclUniqueId {
    pub internal: [::core::ffi::c_char; 128usize],
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclWaitSignalDesc_t {
    pub opCnt: ::core::ffi::c_int,
    pub peer: ::core::ffi::c_int,
    pub sigIdx: ::core::ffi::c_int,
    pub ctx: ::core::ffi::c_int,
}
#[cfg(any(feature = "nccl-02027"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ncclWindow {
    _unused: [u8; 0],
}
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ncclWindow_vidmem {
    _unused: [u8; 0],
}
impl ncclDataType_t {
    pub const ncclChar: ncclDataType_t = ncclDataType_t::ncclInt8;
}
impl ncclDataType_t {
    pub const ncclDouble: ncclDataType_t = ncclDataType_t::ncclFloat64;
}
impl ncclDataType_t {
    pub const ncclFloat: ncclDataType_t = ncclDataType_t::ncclFloat32;
}
impl ncclDataType_t {
    pub const ncclHalf: ncclDataType_t = ncclDataType_t::ncclFloat16;
}
impl ncclDataType_t {
    pub const ncclInt: ncclDataType_t = ncclDataType_t::ncclInt32;
}
pub unsafe fn ncclAllGather(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclAllGather") });
        _f(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclAllGather(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, sendcount: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    }
}
pub unsafe fn ncclAllReduce(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclAllReduce") });
        _f(sendbuff, recvbuff, count, datatype, op, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclAllReduce(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)
    }
}
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclAlltoAll(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclAlltoAll") });
        _f(sendbuff, recvbuff, count, datatype, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclAlltoAll(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream)
    }
}
pub unsafe fn ncclBcast(buff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclBcast") });
        _f(buff, count, datatype, root, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclBcast(buff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclBcast(buff, count, datatype, root, comm, stream)
    }
}
pub unsafe fn ncclBroadcast(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclBroadcast") });
        _f(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclBroadcast(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
}
pub unsafe fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommAbort") });
        _f(comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
        }
        ncclCommAbort(comm)
    }
}
pub unsafe fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommCount") });
        _f(comm, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommCount(comm, count)
    }
}
pub unsafe fn ncclCommCuDevice(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommCuDevice") });
        _f(comm, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommCuDevice(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommCuDevice(comm, device)
    }
}
pub unsafe fn ncclCommDeregister(comm: ncclComm_t, handle: *mut ::core::ffi::c_void) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_void) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommDeregister") });
        _f(comm, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommDeregister(comm: ncclComm_t, handle: *mut ::core::ffi::c_void) -> ncclResult_t;
        }
        ncclCommDeregister(comm, handle)
    }
}
pub unsafe fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommDestroy") });
        _f(comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
        }
        ncclCommDestroy(comm)
    }
}
pub unsafe fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommFinalize") });
        _f(comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;
        }
        ncclCommFinalize(comm)
    }
}
pub unsafe fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ncclResult_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommGetAsyncError") });
        _f(comm, asyncError)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;
        }
        ncclCommGetAsyncError(comm, asyncError)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommGetUniqueId(comm: ncclComm_t, uniqueId: *mut ncclUniqueId) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ncclUniqueId) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommGetUniqueId") });
        _f(comm, uniqueId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommGetUniqueId(comm: ncclComm_t, uniqueId: *mut ncclUniqueId) -> ncclResult_t;
        }
        ncclCommGetUniqueId(comm, uniqueId)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommGrow(comm: ncclComm_t, nRanks: ::core::ffi::c_int, uniqueId: *const ncclUniqueId, rank: ::core::ffi::c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ::core::ffi::c_int, *const ncclUniqueId, ::core::ffi::c_int, *mut ncclComm_t, *mut ncclConfig_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommGrow") });
        _f(comm, nRanks, uniqueId, rank, newcomm, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommGrow(comm: ncclComm_t, nRanks: ::core::ffi::c_int, uniqueId: *const ncclUniqueId, rank: ::core::ffi::c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t;
        }
        ncclCommGrow(comm, nRanks, uniqueId, rank, newcomm, config)
    }
}
pub unsafe fn ncclCommInitAll(comm: *mut ncclComm_t, ndev: ::core::ffi::c_int, devlist: *const ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclComm_t, ::core::ffi::c_int, *const ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommInitAll") });
        _f(comm, ndev, devlist)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommInitAll(comm: *mut ncclComm_t, ndev: ::core::ffi::c_int, devlist: *const ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommInitAll(comm, ndev, devlist)
    }
}
pub unsafe fn ncclCommInitRank(comm: *mut ncclComm_t, nranks: ::core::ffi::c_int, commId: ncclUniqueId, rank: ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclComm_t, ::core::ffi::c_int, ncclUniqueId, ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommInitRank") });
        _f(comm, nranks, commId, rank)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommInitRank(comm: *mut ncclComm_t, nranks: ::core::ffi::c_int, commId: ncclUniqueId, rank: ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommInitRank(comm, nranks, commId, rank)
    }
}
pub unsafe fn ncclCommInitRankConfig(comm: *mut ncclComm_t, nranks: ::core::ffi::c_int, commId: ncclUniqueId, rank: ::core::ffi::c_int, config: *mut ncclConfig_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclComm_t, ::core::ffi::c_int, ncclUniqueId, ::core::ffi::c_int, *mut ncclConfig_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommInitRankConfig") });
        _f(comm, nranks, commId, rank, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommInitRankConfig(comm: *mut ncclComm_t, nranks: ::core::ffi::c_int, commId: ncclUniqueId, rank: ::core::ffi::c_int, config: *mut ncclConfig_t) -> ncclResult_t;
        }
        ncclCommInitRankConfig(comm, nranks, commId, rank, config)
    }
}
#[cfg(any(feature = "nccl-02024", feature = "nccl-02025", feature = "nccl-02026", feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommInitRankScalable(newcomm: *mut ncclComm_t, nranks: ::core::ffi::c_int, myrank: ::core::ffi::c_int, nId: ::core::ffi::c_int, commIds: *mut ncclUniqueId, config: *mut ncclConfig_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclComm_t, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, *mut ncclUniqueId, *mut ncclConfig_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommInitRankScalable") });
        _f(newcomm, nranks, myrank, nId, commIds, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommInitRankScalable(newcomm: *mut ncclComm_t, nranks: ::core::ffi::c_int, myrank: ::core::ffi::c_int, nId: ::core::ffi::c_int, commIds: *mut ncclUniqueId, config: *mut ncclConfig_t) -> ncclResult_t;
        }
        ncclCommInitRankScalable(newcomm, nranks, myrank, nId, commIds, config)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommMemStats(comm: ncclComm_t, stat: ncclCommMemStat_t, value: *mut u64) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ncclCommMemStat_t, *mut u64) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommMemStats") });
        _f(comm, stat, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommMemStats(comm: ncclComm_t, stat: ncclCommMemStat_t, value: *mut u64) -> ncclResult_t;
        }
        ncclCommMemStats(comm, stat, value)
    }
}
pub unsafe fn ncclCommRegister(comm: ncclComm_t, buff: *mut ::core::ffi::c_void, size: usize, handle: *mut *mut ::core::ffi::c_void) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_void, usize, *mut *mut ::core::ffi::c_void) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommRegister") });
        _f(comm, buff, size, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommRegister(comm: ncclComm_t, buff: *mut ::core::ffi::c_void, size: usize, handle: *mut *mut ::core::ffi::c_void) -> ncclResult_t;
        }
        ncclCommRegister(comm, buff, size, handle)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommResume(comm: ncclComm_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommResume") });
        _f(comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommResume(comm: ncclComm_t) -> ncclResult_t;
        }
        ncclCommResume(comm)
    }
}
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommRevoke(comm: ncclComm_t, revokeFlags: ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommRevoke") });
        _f(comm, revokeFlags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommRevoke(comm: ncclComm_t, revokeFlags: ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommRevoke(comm, revokeFlags)
    }
}
#[cfg(any(feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommShrink(comm: ncclComm_t, excludeRanksList: *mut ::core::ffi::c_int, excludeRanksCount: ::core::ffi::c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t, shrinkFlags: ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_int, ::core::ffi::c_int, *mut ncclComm_t, *mut ncclConfig_t, ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommShrink") });
        _f(comm, excludeRanksList, excludeRanksCount, newcomm, config, shrinkFlags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommShrink(comm: ncclComm_t, excludeRanksList: *mut ::core::ffi::c_int, excludeRanksCount: ::core::ffi::c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t, shrinkFlags: ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommShrink(comm, excludeRanksList, excludeRanksCount, newcomm, config, shrinkFlags)
    }
}
pub unsafe fn ncclCommSplit(comm: ncclComm_t, color: ::core::ffi::c_int, key: ::core::ffi::c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ::core::ffi::c_int, ::core::ffi::c_int, *mut ncclComm_t, *mut ncclConfig_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommSplit") });
        _f(comm, color, key, newcomm, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommSplit(comm: ncclComm_t, color: ::core::ffi::c_int, key: ::core::ffi::c_int, newcomm: *mut ncclComm_t, config: *mut ncclConfig_t) -> ncclResult_t;
        }
        ncclCommSplit(comm, color, key, newcomm, config)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommSuspend(comm: ncclComm_t, flags: ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommSuspend") });
        _f(comm, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommSuspend(comm: ncclComm_t, flags: ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommSuspend(comm, flags)
    }
}
pub unsafe fn ncclCommUserRank(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommUserRank") });
        _f(comm, rank)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommUserRank(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommUserRank(comm, rank)
    }
}
#[cfg(any(feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ncclWindow_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommWindowDeregister") });
        _f(comm, win)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t;
        }
        ncclCommWindowDeregister(comm, win)
    }
}
#[cfg(any(feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclCommWindowRegister(comm: ncclComm_t, buff: *mut ::core::ffi::c_void, size: usize, win: *mut ncclWindow_t, winFlags: ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, *mut ::core::ffi::c_void, usize, *mut ncclWindow_t, ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclCommWindowRegister") });
        _f(comm, buff, size, win, winFlags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclCommWindowRegister(comm: ncclComm_t, buff: *mut ::core::ffi::c_void, size: usize, win: *mut ncclWindow_t, winFlags: ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclCommWindowRegister(comm, buff, size, win, winFlags)
    }
}
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclGather(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGather") });
        _f(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGather(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclGather(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
}
pub unsafe fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclResult_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGetErrorString") });
        _f(result)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char;
        }
        ncclGetErrorString(result)
    }
}
pub unsafe fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGetLastError") });
        _f(comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char;
        }
        ncclGetLastError(comm)
    }
}
pub unsafe fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclUniqueId) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGetUniqueId") });
        _f(uniqueId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;
        }
        ncclGetUniqueId(uniqueId)
    }
}
pub unsafe fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGetVersion") });
        _f(version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclGetVersion(version)
    }
}
pub unsafe fn ncclGroupEnd() -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGroupEnd") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGroupEnd() -> ncclResult_t;
        }
        ncclGroupEnd()
    }
}
pub unsafe fn ncclGroupSimulateEnd(simInfo: *mut ncclSimInfo_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclSimInfo_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGroupSimulateEnd") });
        _f(simInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGroupSimulateEnd(simInfo: *mut ncclSimInfo_t) -> ncclResult_t;
        }
        ncclGroupSimulateEnd(simInfo)
    }
}
pub unsafe fn ncclGroupStart() -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclGroupStart") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclGroupStart() -> ncclResult_t;
        }
        ncclGroupStart()
    }
}
pub unsafe fn ncclMemAlloc(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclMemAlloc") });
        _f(ptr, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclMemAlloc(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t;
        }
        ncclMemAlloc(ptr, size)
    }
}
pub unsafe fn ncclMemFree(ptr: *mut ::core::ffi::c_void) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclMemFree") });
        _f(ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclMemFree(ptr: *mut ::core::ffi::c_void) -> ncclResult_t;
        }
        ncclMemFree(ptr)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamBind(out: *mut *mut ncclParamHandle_t, key: *const ::core::ffi::c_char) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ncclParamHandle_t, *const ::core::ffi::c_char) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamBind") });
        _f(out, key)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamBind(out: *mut *mut ncclParamHandle_t, key: *const ::core::ffi::c_char) -> ncclResult_t;
        }
        ncclParamBind(out, key)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamDumpAll() {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn();
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamDumpAll") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamDumpAll();
        }
        ncclParamDumpAll()
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGet(h: *mut ncclParamHandle_t, out: *mut ::core::ffi::c_void, maxLen: ::core::ffi::c_int, len: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut ::core::ffi::c_void, ::core::ffi::c_int, *mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGet") });
        _f(h, out, maxLen, len)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGet(h: *mut ncclParamHandle_t, out: *mut ::core::ffi::c_void, maxLen: ::core::ffi::c_int, len: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclParamGet(h, out, maxLen, len)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetAllParameterKeys(table: *mut *mut *const ::core::ffi::c_char, tableLen: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut *const ::core::ffi::c_char, *mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetAllParameterKeys") });
        _f(table, tableLen)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetAllParameterKeys(table: *mut *mut *const ::core::ffi::c_char, tableLen: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclParamGetAllParameterKeys(table, tableLen)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetI16(h: *mut ncclParamHandle_t, out: *mut i16) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut i16) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetI16") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetI16(h: *mut ncclParamHandle_t, out: *mut i16) -> ncclResult_t;
        }
        ncclParamGetI16(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetI32(h: *mut ncclParamHandle_t, out: *mut i32) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut i32) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetI32") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetI32(h: *mut ncclParamHandle_t, out: *mut i32) -> ncclResult_t;
        }
        ncclParamGetI32(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetI64(h: *mut ncclParamHandle_t, out: *mut i64) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut i64) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetI64") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetI64(h: *mut ncclParamHandle_t, out: *mut i64) -> ncclResult_t;
        }
        ncclParamGetI64(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetI8(h: *mut ncclParamHandle_t, out: *mut i8) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut i8) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetI8") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetI8(h: *mut ncclParamHandle_t, out: *mut i8) -> ncclResult_t;
        }
        ncclParamGetI8(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetParameter(key: *const ::core::ffi::c_char, value: *mut *const ::core::ffi::c_char, valueLen: *mut ::core::ffi::c_int) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *const ::core::ffi::c_char, *mut ::core::ffi::c_int) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetParameter") });
        _f(key, value, valueLen)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetParameter(key: *const ::core::ffi::c_char, value: *mut *const ::core::ffi::c_char, valueLen: *mut ::core::ffi::c_int) -> ncclResult_t;
        }
        ncclParamGetParameter(key, value, valueLen)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetStr(h: *mut ncclParamHandle_t, out: *mut *const ::core::ffi::c_char) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut *const ::core::ffi::c_char) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetStr") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetStr(h: *mut ncclParamHandle_t, out: *mut *const ::core::ffi::c_char) -> ncclResult_t;
        }
        ncclParamGetStr(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetU16(h: *mut ncclParamHandle_t, out: *mut u16) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut u16) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetU16") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetU16(h: *mut ncclParamHandle_t, out: *mut u16) -> ncclResult_t;
        }
        ncclParamGetU16(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetU32(h: *mut ncclParamHandle_t, out: *mut u32) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut u32) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetU32") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetU32(h: *mut ncclParamHandle_t, out: *mut u32) -> ncclResult_t;
        }
        ncclParamGetU32(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetU64(h: *mut ncclParamHandle_t, out: *mut u64) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut u64) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetU64") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetU64(h: *mut ncclParamHandle_t, out: *mut u64) -> ncclResult_t;
        }
        ncclParamGetU64(h, out)
    }
}
#[cfg(any(feature = "nccl-02030"))]
pub unsafe fn ncclParamGetU8(h: *mut ncclParamHandle_t, out: *mut u8) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclParamHandle_t, *mut u8) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclParamGetU8") });
        _f(h, out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclParamGetU8(h: *mut ncclParamHandle_t, out: *mut u8) -> ncclResult_t;
        }
        ncclParamGetU8(h, out)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclPutSignal(localbuff: *const ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, peer: ::core::ffi::c_int, peerWin: ncclWindow_t, peerWinOffset: usize, sigIdx: ::core::ffi::c_int, ctx: ::core::ffi::c_int, flags: ::core::ffi::c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclWindow_t, usize, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_uint, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclPutSignal") });
        _f(localbuff, count, datatype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclPutSignal(localbuff: *const ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, peer: ::core::ffi::c_int, peerWin: ncclWindow_t, peerWinOffset: usize, sigIdx: ::core::ffi::c_int, ctx: ::core::ffi::c_int, flags: ::core::ffi::c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclPutSignal(localbuff, count, datatype, peer, peerWin, peerWinOffset, sigIdx, ctx, flags, comm, stream)
    }
}
pub unsafe fn ncclRecv(recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, peer: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclRecv") });
        _f(recvbuff, count, datatype, peer, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclRecv(recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, peer: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclRecv(recvbuff, count, datatype, peer, comm, stream)
    }
}
pub unsafe fn ncclRedOpCreatePreMulSum(op: *mut ncclRedOp_t, scalar: *mut ::core::ffi::c_void, datatype: ncclDataType_t, residence: ncclScalarResidence_t, comm: ncclComm_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ncclRedOp_t, *mut ::core::ffi::c_void, ncclDataType_t, ncclScalarResidence_t, ncclComm_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclRedOpCreatePreMulSum") });
        _f(op, scalar, datatype, residence, comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclRedOpCreatePreMulSum(op: *mut ncclRedOp_t, scalar: *mut ::core::ffi::c_void, datatype: ncclDataType_t, residence: ncclScalarResidence_t, comm: ncclComm_t) -> ncclResult_t;
        }
        ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm)
    }
}
pub unsafe fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclRedOp_t, ncclComm_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclRedOpDestroy") });
        _f(op, comm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;
        }
        ncclRedOpDestroy(op, comm)
    }
}
pub unsafe fn ncclReduce(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ncclRedOp_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclReduce") });
        _f(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclReduce(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, op: ncclRedOp_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    }
}
pub unsafe fn ncclReduceScatter(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclReduceScatter") });
        _f(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclReduceScatter(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, recvcount: usize, datatype: ncclDataType_t, op: ncclRedOp_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    }
}
#[cfg(any(feature = "nccl-02024", feature = "nccl-02025", feature = "nccl-02026", feature = "nccl-02027", feature = "nccl-02028", feature = "nccl-02029"))]
pub unsafe fn ncclResetDebugInit() {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn();
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclResetDebugInit") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclResetDebugInit();
        }
        ncclResetDebugInit()
    }
}
#[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclScatter(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclScatter") });
        _f(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclScatter(sendbuff: *const ::core::ffi::c_void, recvbuff: *mut ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, root: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclScatter(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
}
pub unsafe fn ncclSend(sendbuff: *const ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, peer: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, ncclDataType_t, ::core::ffi::c_int, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclSend") });
        _f(sendbuff, count, datatype, peer, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclSend(sendbuff: *const ::core::ffi::c_void, count: usize, datatype: ncclDataType_t, peer: ::core::ffi::c_int, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclSend(sendbuff, count, datatype, peer, comm, stream)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclSignal(peer: ::core::ffi::c_int, sigIdx: ::core::ffi::c_int, ctx: ::core::ffi::c_int, flags: ::core::ffi::c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_uint, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclSignal") });
        _f(peer, sigIdx, ctx, flags, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclSignal(peer: ::core::ffi::c_int, sigIdx: ::core::ffi::c_int, ctx: ::core::ffi::c_int, flags: ::core::ffi::c_uint, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclSignal(peer, sigIdx, ctx, flags, comm, stream)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclWaitSignal(nDesc: ::core::ffi::c_int, signalDescs: *mut ncclWaitSignalDesc_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, *mut ncclWaitSignalDesc_t, ncclComm_t, cudaStream_t) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclWaitSignal") });
        _f(nDesc, signalDescs, comm, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclWaitSignal(nDesc: ::core::ffi::c_int, signalDescs: *mut ncclWaitSignalDesc_t, comm: ncclComm_t, stream: cudaStream_t) -> ncclResult_t;
        }
        ncclWaitSignal(nDesc, signalDescs, comm, stream)
    }
}
#[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
pub unsafe fn ncclWinGetUserPtr(comm: ncclComm_t, win: ncclWindow_t, outUserPtr: *mut *mut ::core::ffi::c_void) -> ncclResult_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(ncclComm_t, ncclWindow_t, *mut *mut ::core::ffi::c_void) -> ncclResult_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("ncclWinGetUserPtr") });
        _f(comm, win, outUserPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn ncclWinGetUserPtr(comm: ncclComm_t, win: ncclWindow_t, outUserPtr: *mut *mut ::core::ffi::c_void) -> ncclResult_t;
        }
        ncclWinGetUserPtr(comm, win, outUserPtr)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["nccl"];
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
        let lib_names = std::vec!["nccl"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
