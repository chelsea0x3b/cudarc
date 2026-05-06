#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub type cudaStream_t = *mut CUstream_st;
pub type ncclComm_t = *mut ncclComm;
#[cfg(any(
    feature = "nccl-02022",
    feature = "nccl-02024",
    feature = "nccl-02025",
    feature = "nccl-02026"
))]
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
#[cfg(any(
    feature = "nccl-02024",
    feature = "nccl-02025",
    feature = "nccl-02026",
    feature = "nccl-02027",
    feature = "nccl-02028",
    feature = "nccl-02029",
    feature = "nccl-02030"
))]
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
#[cfg(any(
    feature = "nccl-02022",
    feature = "nccl-02024",
    feature = "nccl-02025",
    feature = "nccl-02026",
    feature = "nccl-02027",
    feature = "nccl-02028",
    feature = "nccl-02029"
))]
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
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn ncclAllGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclAllReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclAlltoAll(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclBcast(
        buff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclBroadcast(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclCommCuDevice(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclCommDeregister(comm: ncclComm_t, handle: *mut ::core::ffi::c_void) -> ncclResult_t;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclCommGetUniqueId(comm: ncclComm_t, uniqueId: *mut ncclUniqueId) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclCommGrow(
        comm: ncclComm_t,
        nRanks: ::core::ffi::c_int,
        uniqueId: *const ncclUniqueId,
        rank: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    pub fn ncclCommInitAll(
        comm: *mut ncclComm_t,
        ndev: ::core::ffi::c_int,
        devlist: *const ::core::ffi::c_int,
    ) -> ncclResult_t;
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
    ) -> ncclResult_t;
    pub fn ncclCommInitRankConfig(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    #[cfg(any(
        feature = "nccl-02024",
        feature = "nccl-02025",
        feature = "nccl-02026",
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub fn ncclCommInitRankScalable(
        newcomm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        myrank: ::core::ffi::c_int,
        nId: ::core::ffi::c_int,
        commIds: *mut ncclUniqueId,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclCommMemStats(
        comm: ncclComm_t,
        stat: ncclCommMemStat_t,
        value: *mut u64,
    ) -> ncclResult_t;
    pub fn ncclCommRegister(
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        handle: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclCommResume(comm: ncclComm_t) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclCommRevoke(comm: ncclComm_t, revokeFlags: ::core::ffi::c_int) -> ncclResult_t;
    #[cfg(any(
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub fn ncclCommShrink(
        comm: ncclComm_t,
        excludeRanksList: *mut ::core::ffi::c_int,
        excludeRanksCount: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
        shrinkFlags: ::core::ffi::c_int,
    ) -> ncclResult_t;
    pub fn ncclCommSplit(
        comm: ncclComm_t,
        color: ::core::ffi::c_int,
        key: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclCommSuspend(comm: ncclComm_t, flags: ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclCommUserRank(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t;
    #[cfg(any(
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t;
    #[cfg(any(
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub fn ncclCommWindowRegister(
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        win: *mut ncclWindow_t,
        winFlags: ::core::ffi::c_int,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char;
    pub fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char;
    pub fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;
    pub fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclGroupEnd() -> ncclResult_t;
    pub fn ncclGroupSimulateEnd(simInfo: *mut ncclSimInfo_t) -> ncclResult_t;
    pub fn ncclGroupStart() -> ncclResult_t;
    pub fn ncclMemAlloc(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t;
    pub fn ncclMemFree(ptr: *mut ::core::ffi::c_void) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamBind(
        out: *mut *mut ncclParamHandle_t,
        key: *const ::core::ffi::c_char,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamDumpAll();
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGet(
        h: *mut ncclParamHandle_t,
        out: *mut ::core::ffi::c_void,
        maxLen: ::core::ffi::c_int,
        len: *mut ::core::ffi::c_int,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetAllParameterKeys(
        table: *mut *mut *const ::core::ffi::c_char,
        tableLen: *mut ::core::ffi::c_int,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetI16(h: *mut ncclParamHandle_t, out: *mut i16) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetI32(h: *mut ncclParamHandle_t, out: *mut i32) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetI64(h: *mut ncclParamHandle_t, out: *mut i64) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetI8(h: *mut ncclParamHandle_t, out: *mut i8) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetParameter(
        key: *const ::core::ffi::c_char,
        value: *mut *const ::core::ffi::c_char,
        valueLen: *mut ::core::ffi::c_int,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetStr(
        h: *mut ncclParamHandle_t,
        out: *mut *const ::core::ffi::c_char,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetU16(h: *mut ncclParamHandle_t, out: *mut u16) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetU32(h: *mut ncclParamHandle_t, out: *mut u32) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetU64(h: *mut ncclParamHandle_t, out: *mut u64) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02030"))]
    pub fn ncclParamGetU8(h: *mut ncclParamHandle_t, out: *mut u8) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclPutSignal(
        localbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        peerWin: ncclWindow_t,
        peerWinOffset: usize,
        sigIdx: ::core::ffi::c_int,
        ctx: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclRecv(
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclRedOpCreatePreMulSum(
        op: *mut ncclRedOp_t,
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> ncclResult_t;
    pub fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclReduceScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        recvcount: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    #[cfg(any(
        feature = "nccl-02024",
        feature = "nccl-02025",
        feature = "nccl-02026",
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029"
    ))]
    pub fn ncclResetDebugInit();
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclSend(
        sendbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclSignal(
        peer: ::core::ffi::c_int,
        sigIdx: ::core::ffi::c_int,
        ctx: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclWaitSignal(
        nDesc: ::core::ffi::c_int,
        signalDescs: *mut ncclWaitSignalDesc_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub fn ncclWinGetUserPtr(
        comm: ncclComm_t,
        win: ncclWindow_t,
        outUserPtr: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn ncclAllGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclAllGather)(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    }
    pub unsafe fn ncclAllReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclAllReduce)(sendbuff, recvbuff, count, datatype, op, comm, stream)
    }
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclAlltoAll(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclAlltoAll)(sendbuff, recvbuff, count, datatype, comm, stream)
    }
    pub unsafe fn ncclBcast(
        buff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclBcast)(buff, count, datatype, root, comm, stream)
    }
    pub unsafe fn ncclBroadcast(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclBroadcast)(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    pub unsafe fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommAbort)(comm)
    }
    pub unsafe fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t {
        (culib().ncclCommCount)(comm, count)
    }
    pub unsafe fn ncclCommCuDevice(
        comm: ncclComm_t,
        device: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommCuDevice)(comm, device)
    }
    pub unsafe fn ncclCommDeregister(
        comm: ncclComm_t,
        handle: *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (culib().ncclCommDeregister)(comm, handle)
    }
    pub unsafe fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommDestroy)(comm)
    }
    pub unsafe fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommFinalize)(comm)
    }
    pub unsafe fn ncclCommGetAsyncError(
        comm: ncclComm_t,
        asyncError: *mut ncclResult_t,
    ) -> ncclResult_t {
        (culib().ncclCommGetAsyncError)(comm, asyncError)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclCommGetUniqueId(
        comm: ncclComm_t,
        uniqueId: *mut ncclUniqueId,
    ) -> ncclResult_t {
        (culib().ncclCommGetUniqueId)(comm, uniqueId)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclCommGrow(
        comm: ncclComm_t,
        nRanks: ::core::ffi::c_int,
        uniqueId: *const ncclUniqueId,
        rank: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommGrow)(comm, nRanks, uniqueId, rank, newcomm, config)
    }
    pub unsafe fn ncclCommInitAll(
        comm: *mut ncclComm_t,
        ndev: ::core::ffi::c_int,
        devlist: *const ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommInitAll)(comm, ndev, devlist)
    }
    pub unsafe fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommInitRank)(comm, nranks, commId, rank)
    }
    pub unsafe fn ncclCommInitRankConfig(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommInitRankConfig)(comm, nranks, commId, rank, config)
    }
    #[cfg(any(
        feature = "nccl-02024",
        feature = "nccl-02025",
        feature = "nccl-02026",
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub unsafe fn ncclCommInitRankScalable(
        newcomm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        myrank: ::core::ffi::c_int,
        nId: ::core::ffi::c_int,
        commIds: *mut ncclUniqueId,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommInitRankScalable)(newcomm, nranks, myrank, nId, commIds, config)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclCommMemStats(
        comm: ncclComm_t,
        stat: ncclCommMemStat_t,
        value: *mut u64,
    ) -> ncclResult_t {
        (culib().ncclCommMemStats)(comm, stat, value)
    }
    pub unsafe fn ncclCommRegister(
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        handle: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (culib().ncclCommRegister)(comm, buff, size, handle)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclCommResume(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommResume)(comm)
    }
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclCommRevoke(
        comm: ncclComm_t,
        revokeFlags: ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommRevoke)(comm, revokeFlags)
    }
    #[cfg(any(
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub unsafe fn ncclCommShrink(
        comm: ncclComm_t,
        excludeRanksList: *mut ::core::ffi::c_int,
        excludeRanksCount: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
        shrinkFlags: ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommShrink)(
            comm,
            excludeRanksList,
            excludeRanksCount,
            newcomm,
            config,
            shrinkFlags,
        )
    }
    pub unsafe fn ncclCommSplit(
        comm: ncclComm_t,
        color: ::core::ffi::c_int,
        key: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommSplit)(comm, color, key, newcomm, config)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclCommSuspend(comm: ncclComm_t, flags: ::core::ffi::c_int) -> ncclResult_t {
        (culib().ncclCommSuspend)(comm, flags)
    }
    pub unsafe fn ncclCommUserRank(
        comm: ncclComm_t,
        rank: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommUserRank)(comm, rank)
    }
    #[cfg(any(
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub unsafe fn ncclCommWindowDeregister(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t {
        (culib().ncclCommWindowDeregister)(comm, win)
    }
    #[cfg(any(
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029",
        feature = "nccl-02030"
    ))]
    pub unsafe fn ncclCommWindowRegister(
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        win: *mut ncclWindow_t,
        winFlags: ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommWindowRegister)(comm, buff, size, win, winFlags)
    }
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclGather)(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    pub unsafe fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char {
        (culib().ncclGetErrorString)(result)
    }
    pub unsafe fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char {
        (culib().ncclGetLastError)(comm)
    }
    pub unsafe fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t {
        (culib().ncclGetUniqueId)(uniqueId)
    }
    pub unsafe fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t {
        (culib().ncclGetVersion)(version)
    }
    pub unsafe fn ncclGroupEnd() -> ncclResult_t {
        (culib().ncclGroupEnd)()
    }
    pub unsafe fn ncclGroupSimulateEnd(simInfo: *mut ncclSimInfo_t) -> ncclResult_t {
        (culib().ncclGroupSimulateEnd)(simInfo)
    }
    pub unsafe fn ncclGroupStart() -> ncclResult_t {
        (culib().ncclGroupStart)()
    }
    pub unsafe fn ncclMemAlloc(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t {
        (culib().ncclMemAlloc)(ptr, size)
    }
    pub unsafe fn ncclMemFree(ptr: *mut ::core::ffi::c_void) -> ncclResult_t {
        (culib().ncclMemFree)(ptr)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamBind(
        out: *mut *mut ncclParamHandle_t,
        key: *const ::core::ffi::c_char,
    ) -> ncclResult_t {
        (culib().ncclParamBind)(out, key)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamDumpAll() {
        (culib().ncclParamDumpAll)()
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGet(
        h: *mut ncclParamHandle_t,
        out: *mut ::core::ffi::c_void,
        maxLen: ::core::ffi::c_int,
        len: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclParamGet)(h, out, maxLen, len)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetAllParameterKeys(
        table: *mut *mut *const ::core::ffi::c_char,
        tableLen: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclParamGetAllParameterKeys)(table, tableLen)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetI16(h: *mut ncclParamHandle_t, out: *mut i16) -> ncclResult_t {
        (culib().ncclParamGetI16)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetI32(h: *mut ncclParamHandle_t, out: *mut i32) -> ncclResult_t {
        (culib().ncclParamGetI32)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetI64(h: *mut ncclParamHandle_t, out: *mut i64) -> ncclResult_t {
        (culib().ncclParamGetI64)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetI8(h: *mut ncclParamHandle_t, out: *mut i8) -> ncclResult_t {
        (culib().ncclParamGetI8)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetParameter(
        key: *const ::core::ffi::c_char,
        value: *mut *const ::core::ffi::c_char,
        valueLen: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclParamGetParameter)(key, value, valueLen)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetStr(
        h: *mut ncclParamHandle_t,
        out: *mut *const ::core::ffi::c_char,
    ) -> ncclResult_t {
        (culib().ncclParamGetStr)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetU16(h: *mut ncclParamHandle_t, out: *mut u16) -> ncclResult_t {
        (culib().ncclParamGetU16)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetU32(h: *mut ncclParamHandle_t, out: *mut u32) -> ncclResult_t {
        (culib().ncclParamGetU32)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetU64(h: *mut ncclParamHandle_t, out: *mut u64) -> ncclResult_t {
        (culib().ncclParamGetU64)(h, out)
    }
    #[cfg(any(feature = "nccl-02030"))]
    pub unsafe fn ncclParamGetU8(h: *mut ncclParamHandle_t, out: *mut u8) -> ncclResult_t {
        (culib().ncclParamGetU8)(h, out)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclPutSignal(
        localbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        peerWin: ncclWindow_t,
        peerWinOffset: usize,
        sigIdx: ::core::ffi::c_int,
        ctx: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclPutSignal)(
            localbuff,
            count,
            datatype,
            peer,
            peerWin,
            peerWinOffset,
            sigIdx,
            ctx,
            flags,
            comm,
            stream,
        )
    }
    pub unsafe fn ncclRecv(
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclRecv)(recvbuff, count, datatype, peer, comm, stream)
    }
    pub unsafe fn ncclRedOpCreatePreMulSum(
        op: *mut ncclRedOp_t,
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> ncclResult_t {
        (culib().ncclRedOpCreatePreMulSum)(op, scalar, datatype, residence, comm)
    }
    pub unsafe fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclRedOpDestroy)(op, comm)
    }
    pub unsafe fn ncclReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclReduce)(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    }
    pub unsafe fn ncclReduceScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        recvcount: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclReduceScatter)(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    }
    #[cfg(any(
        feature = "nccl-02024",
        feature = "nccl-02025",
        feature = "nccl-02026",
        feature = "nccl-02027",
        feature = "nccl-02028",
        feature = "nccl-02029"
    ))]
    pub unsafe fn ncclResetDebugInit() {
        (culib().ncclResetDebugInit)()
    }
    #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclScatter)(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    pub unsafe fn ncclSend(
        sendbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclSend)(sendbuff, count, datatype, peer, comm, stream)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclSignal(
        peer: ::core::ffi::c_int,
        sigIdx: ::core::ffi::c_int,
        ctx: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclSignal)(peer, sigIdx, ctx, flags, comm, stream)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclWaitSignal(
        nDesc: ::core::ffi::c_int,
        signalDescs: *mut ncclWaitSignalDesc_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclWaitSignal)(nDesc, signalDescs, comm, stream)
    }
    #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
    pub unsafe fn ncclWinGetUserPtr(
        comm: ncclComm_t,
        win: ncclWindow_t,
        outUserPtr: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (culib().ncclWinGetUserPtr)(comm, win, outUserPtr)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub ncclAllGather: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            sendcount: usize,
            datatype: ncclDataType_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclAllReduce: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclAlltoAll: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclBcast: unsafe extern "C" fn(
            buff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclBroadcast: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclCommAbort: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        pub ncclCommCount:
            unsafe extern "C" fn(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclCommCuDevice:
            unsafe extern "C" fn(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclCommDeregister: unsafe extern "C" fn(
            comm: ncclComm_t,
            handle: *mut ::core::ffi::c_void,
        ) -> ncclResult_t,
        pub ncclCommDestroy: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        pub ncclCommFinalize: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        pub ncclCommGetAsyncError:
            unsafe extern "C" fn(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclCommGetUniqueId:
            unsafe extern "C" fn(comm: ncclComm_t, uniqueId: *mut ncclUniqueId) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclCommGrow: unsafe extern "C" fn(
            comm: ncclComm_t,
            nRanks: ::core::ffi::c_int,
            uniqueId: *const ncclUniqueId,
            rank: ::core::ffi::c_int,
            newcomm: *mut ncclComm_t,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        pub ncclCommInitAll: unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            ndev: ::core::ffi::c_int,
            devlist: *const ::core::ffi::c_int,
        ) -> ncclResult_t,
        pub ncclCommInitRank: unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            commId: ncclUniqueId,
            rank: ::core::ffi::c_int,
        ) -> ncclResult_t,
        pub ncclCommInitRankConfig: unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            commId: ncclUniqueId,
            rank: ::core::ffi::c_int,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        #[cfg(any(
            feature = "nccl-02024",
            feature = "nccl-02025",
            feature = "nccl-02026",
            feature = "nccl-02027",
            feature = "nccl-02028",
            feature = "nccl-02029",
            feature = "nccl-02030"
        ))]
        pub ncclCommInitRankScalable: unsafe extern "C" fn(
            newcomm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            myrank: ::core::ffi::c_int,
            nId: ::core::ffi::c_int,
            commIds: *mut ncclUniqueId,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclCommMemStats: unsafe extern "C" fn(
            comm: ncclComm_t,
            stat: ncclCommMemStat_t,
            value: *mut u64,
        ) -> ncclResult_t,
        pub ncclCommRegister: unsafe extern "C" fn(
            comm: ncclComm_t,
            buff: *mut ::core::ffi::c_void,
            size: usize,
            handle: *mut *mut ::core::ffi::c_void,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclCommResume: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclCommRevoke:
            unsafe extern "C" fn(comm: ncclComm_t, revokeFlags: ::core::ffi::c_int) -> ncclResult_t,
        #[cfg(any(
            feature = "nccl-02027",
            feature = "nccl-02028",
            feature = "nccl-02029",
            feature = "nccl-02030"
        ))]
        pub ncclCommShrink: unsafe extern "C" fn(
            comm: ncclComm_t,
            excludeRanksList: *mut ::core::ffi::c_int,
            excludeRanksCount: ::core::ffi::c_int,
            newcomm: *mut ncclComm_t,
            config: *mut ncclConfig_t,
            shrinkFlags: ::core::ffi::c_int,
        ) -> ncclResult_t,
        pub ncclCommSplit: unsafe extern "C" fn(
            comm: ncclComm_t,
            color: ::core::ffi::c_int,
            key: ::core::ffi::c_int,
            newcomm: *mut ncclComm_t,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclCommSuspend:
            unsafe extern "C" fn(comm: ncclComm_t, flags: ::core::ffi::c_int) -> ncclResult_t,
        pub ncclCommUserRank:
            unsafe extern "C" fn(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t,
        #[cfg(any(
            feature = "nccl-02027",
            feature = "nccl-02028",
            feature = "nccl-02029",
            feature = "nccl-02030"
        ))]
        pub ncclCommWindowDeregister:
            unsafe extern "C" fn(comm: ncclComm_t, win: ncclWindow_t) -> ncclResult_t,
        #[cfg(any(
            feature = "nccl-02027",
            feature = "nccl-02028",
            feature = "nccl-02029",
            feature = "nccl-02030"
        ))]
        pub ncclCommWindowRegister: unsafe extern "C" fn(
            comm: ncclComm_t,
            buff: *mut ::core::ffi::c_void,
            size: usize,
            win: *mut ncclWindow_t,
            winFlags: ::core::ffi::c_int,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclGather: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclGetErrorString:
            unsafe extern "C" fn(result: ncclResult_t) -> *const ::core::ffi::c_char,
        pub ncclGetLastError: unsafe extern "C" fn(comm: ncclComm_t) -> *const ::core::ffi::c_char,
        pub ncclGetUniqueId: unsafe extern "C" fn(uniqueId: *mut ncclUniqueId) -> ncclResult_t,
        pub ncclGetVersion: unsafe extern "C" fn(version: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclGroupEnd: unsafe extern "C" fn() -> ncclResult_t,
        pub ncclGroupSimulateEnd: unsafe extern "C" fn(simInfo: *mut ncclSimInfo_t) -> ncclResult_t,
        pub ncclGroupStart: unsafe extern "C" fn() -> ncclResult_t,
        pub ncclMemAlloc:
            unsafe extern "C" fn(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t,
        pub ncclMemFree: unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamBind: unsafe extern "C" fn(
            out: *mut *mut ncclParamHandle_t,
            key: *const ::core::ffi::c_char,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamDumpAll: unsafe extern "C" fn(),
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGet: unsafe extern "C" fn(
            h: *mut ncclParamHandle_t,
            out: *mut ::core::ffi::c_void,
            maxLen: ::core::ffi::c_int,
            len: *mut ::core::ffi::c_int,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetAllParameterKeys: unsafe extern "C" fn(
            table: *mut *mut *const ::core::ffi::c_char,
            tableLen: *mut ::core::ffi::c_int,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetI16:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut i16) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetI32:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut i32) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetI64:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut i64) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetI8:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut i8) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetParameter: unsafe extern "C" fn(
            key: *const ::core::ffi::c_char,
            value: *mut *const ::core::ffi::c_char,
            valueLen: *mut ::core::ffi::c_int,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetStr: unsafe extern "C" fn(
            h: *mut ncclParamHandle_t,
            out: *mut *const ::core::ffi::c_char,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetU16:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut u16) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetU32:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut u32) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetU64:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut u64) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02030"))]
        pub ncclParamGetU8:
            unsafe extern "C" fn(h: *mut ncclParamHandle_t, out: *mut u8) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclPutSignal: unsafe extern "C" fn(
            localbuff: *const ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            peerWin: ncclWindow_t,
            peerWinOffset: usize,
            sigIdx: ::core::ffi::c_int,
            ctx: ::core::ffi::c_int,
            flags: ::core::ffi::c_uint,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclRecv: unsafe extern "C" fn(
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclRedOpCreatePreMulSum: unsafe extern "C" fn(
            op: *mut ncclRedOp_t,
            scalar: *mut ::core::ffi::c_void,
            datatype: ncclDataType_t,
            residence: ncclScalarResidence_t,
            comm: ncclComm_t,
        ) -> ncclResult_t,
        pub ncclRedOpDestroy:
            unsafe extern "C" fn(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t,
        pub ncclReduce: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclReduceScatter: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            recvcount: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        #[cfg(any(
            feature = "nccl-02024",
            feature = "nccl-02025",
            feature = "nccl-02026",
            feature = "nccl-02027",
            feature = "nccl-02028",
            feature = "nccl-02029"
        ))]
        pub ncclResetDebugInit: unsafe extern "C" fn(),
        #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclScatter: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclSend: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclSignal: unsafe extern "C" fn(
            peer: ::core::ffi::c_int,
            sigIdx: ::core::ffi::c_int,
            ctx: ::core::ffi::c_int,
            flags: ::core::ffi::c_uint,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclWaitSignal: unsafe extern "C" fn(
            nDesc: ::core::ffi::c_int,
            signalDescs: *mut ncclWaitSignalDesc_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
        pub ncclWinGetUserPtr: unsafe extern "C" fn(
            comm: ncclComm_t,
            win: ncclWindow_t,
            outUserPtr: *mut *mut ::core::ffi::c_void,
        ) -> ncclResult_t,
    }
    impl Lib {
        pub unsafe fn new<P>(path: P) -> Result<Self, ::libloading::Error>
        where
            P: AsRef<::std::ffi::OsStr>,
        {
            let library = ::libloading::Library::new(path.as_ref())?;
            Self::from_library(library)
        }
        pub unsafe fn from_library<L>(library: L) -> Result<Self, ::libloading::Error>
        where
            L: Into<::libloading::Library>,
        {
            let __library = library.into();
            let ncclAllGather = __library
                .get(b"ncclAllGather\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclAllReduce = __library
                .get(b"ncclAllReduce\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclAlltoAll = __library
                .get(b"ncclAlltoAll\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclBcast = __library
                .get(b"ncclBcast\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclBroadcast = __library
                .get(b"ncclBroadcast\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommAbort = __library
                .get(b"ncclCommAbort\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommCount = __library
                .get(b"ncclCommCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommCuDevice = __library
                .get(b"ncclCommCuDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommDeregister = __library
                .get(b"ncclCommDeregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommDestroy = __library
                .get(b"ncclCommDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommFinalize = __library
                .get(b"ncclCommFinalize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommGetAsyncError = __library
                .get(b"ncclCommGetAsyncError\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclCommGetUniqueId = __library
                .get(b"ncclCommGetUniqueId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclCommGrow = __library
                .get(b"ncclCommGrow\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitAll = __library
                .get(b"ncclCommInitAll\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitRank = __library
                .get(b"ncclCommInitRank\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitRankConfig = __library
                .get(b"ncclCommInitRankConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "nccl-02024",
                feature = "nccl-02025",
                feature = "nccl-02026",
                feature = "nccl-02027",
                feature = "nccl-02028",
                feature = "nccl-02029",
                feature = "nccl-02030"
            ))]
            let ncclCommInitRankScalable = __library
                .get(b"ncclCommInitRankScalable\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclCommMemStats = __library
                .get(b"ncclCommMemStats\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommRegister = __library
                .get(b"ncclCommRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclCommResume = __library
                .get(b"ncclCommResume\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclCommRevoke = __library
                .get(b"ncclCommRevoke\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "nccl-02027",
                feature = "nccl-02028",
                feature = "nccl-02029",
                feature = "nccl-02030"
            ))]
            let ncclCommShrink = __library
                .get(b"ncclCommShrink\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommSplit = __library
                .get(b"ncclCommSplit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclCommSuspend = __library
                .get(b"ncclCommSuspend\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommUserRank = __library
                .get(b"ncclCommUserRank\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "nccl-02027",
                feature = "nccl-02028",
                feature = "nccl-02029",
                feature = "nccl-02030"
            ))]
            let ncclCommWindowDeregister = __library
                .get(b"ncclCommWindowDeregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "nccl-02027",
                feature = "nccl-02028",
                feature = "nccl-02029",
                feature = "nccl-02030"
            ))]
            let ncclCommWindowRegister = __library
                .get(b"ncclCommWindowRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclGather = __library
                .get(b"ncclGather\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetErrorString = __library
                .get(b"ncclGetErrorString\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetLastError = __library
                .get(b"ncclGetLastError\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetUniqueId = __library
                .get(b"ncclGetUniqueId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetVersion = __library
                .get(b"ncclGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGroupEnd = __library
                .get(b"ncclGroupEnd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGroupSimulateEnd = __library
                .get(b"ncclGroupSimulateEnd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGroupStart = __library
                .get(b"ncclGroupStart\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclMemAlloc = __library
                .get(b"ncclMemAlloc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclMemFree = __library
                .get(b"ncclMemFree\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamBind = __library
                .get(b"ncclParamBind\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamDumpAll = __library
                .get(b"ncclParamDumpAll\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGet = __library
                .get(b"ncclParamGet\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetAllParameterKeys = __library
                .get(b"ncclParamGetAllParameterKeys\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetI16 = __library
                .get(b"ncclParamGetI16\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetI32 = __library
                .get(b"ncclParamGetI32\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetI64 = __library
                .get(b"ncclParamGetI64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetI8 = __library
                .get(b"ncclParamGetI8\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetParameter = __library
                .get(b"ncclParamGetParameter\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetStr = __library
                .get(b"ncclParamGetStr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetU16 = __library
                .get(b"ncclParamGetU16\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetU32 = __library
                .get(b"ncclParamGetU32\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetU64 = __library
                .get(b"ncclParamGetU64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02030"))]
            let ncclParamGetU8 = __library
                .get(b"ncclParamGetU8\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclPutSignal = __library
                .get(b"ncclPutSignal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclRecv = __library
                .get(b"ncclRecv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclRedOpCreatePreMulSum = __library
                .get(b"ncclRedOpCreatePreMulSum\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclRedOpDestroy = __library
                .get(b"ncclRedOpDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclReduce = __library
                .get(b"ncclReduce\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclReduceScatter = __library
                .get(b"ncclReduceScatter\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "nccl-02024",
                feature = "nccl-02025",
                feature = "nccl-02026",
                feature = "nccl-02027",
                feature = "nccl-02028",
                feature = "nccl-02029"
            ))]
            let ncclResetDebugInit = __library
                .get(b"ncclResetDebugInit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02028", feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclScatter = __library
                .get(b"ncclScatter\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclSend = __library
                .get(b"ncclSend\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclSignal = __library
                .get(b"ncclSignal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclWaitSignal = __library
                .get(b"ncclWaitSignal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
            let ncclWinGetUserPtr = __library
                .get(b"ncclWinGetUserPtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                ncclAllGather,
                ncclAllReduce,
                #[cfg(any(
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclAlltoAll,
                ncclBcast,
                ncclBroadcast,
                ncclCommAbort,
                ncclCommCount,
                ncclCommCuDevice,
                ncclCommDeregister,
                ncclCommDestroy,
                ncclCommFinalize,
                ncclCommGetAsyncError,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclCommGetUniqueId,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclCommGrow,
                ncclCommInitAll,
                ncclCommInitRank,
                ncclCommInitRankConfig,
                #[cfg(any(
                    feature = "nccl-02024",
                    feature = "nccl-02025",
                    feature = "nccl-02026",
                    feature = "nccl-02027",
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclCommInitRankScalable,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclCommMemStats,
                ncclCommRegister,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclCommResume,
                #[cfg(any(
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclCommRevoke,
                #[cfg(any(
                    feature = "nccl-02027",
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclCommShrink,
                ncclCommSplit,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclCommSuspend,
                ncclCommUserRank,
                #[cfg(any(
                    feature = "nccl-02027",
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclCommWindowDeregister,
                #[cfg(any(
                    feature = "nccl-02027",
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclCommWindowRegister,
                #[cfg(any(
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclGather,
                ncclGetErrorString,
                ncclGetLastError,
                ncclGetUniqueId,
                ncclGetVersion,
                ncclGroupEnd,
                ncclGroupSimulateEnd,
                ncclGroupStart,
                ncclMemAlloc,
                ncclMemFree,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamBind,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamDumpAll,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGet,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetAllParameterKeys,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetI16,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetI32,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetI64,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetI8,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetParameter,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetStr,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetU16,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetU32,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetU64,
                #[cfg(any(feature = "nccl-02030"))]
                ncclParamGetU8,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclPutSignal,
                ncclRecv,
                ncclRedOpCreatePreMulSum,
                ncclRedOpDestroy,
                ncclReduce,
                ncclReduceScatter,
                #[cfg(any(
                    feature = "nccl-02024",
                    feature = "nccl-02025",
                    feature = "nccl-02026",
                    feature = "nccl-02027",
                    feature = "nccl-02028",
                    feature = "nccl-02029"
                ))]
                ncclResetDebugInit,
                #[cfg(any(
                    feature = "nccl-02028",
                    feature = "nccl-02029",
                    feature = "nccl-02030"
                ))]
                ncclScatter,
                ncclSend,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclSignal,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclWaitSignal,
                #[cfg(any(feature = "nccl-02029", feature = "nccl-02030"))]
                ncclWinGetUserPtr,
            })
        }
    }
    pub unsafe fn is_culib_present() -> bool {
        let lib_names = ["nccl"];
        let choices = lib_names
            .iter()
            .map(|l| crate::get_lib_name_candidates(l))
            .flatten();
        for choice in choices {
            if Lib::new(choice).is_ok() {
                return true;
            }
        }
        false
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["nccl"];
            let choices: std::vec::Vec<_> = lib_names
                .iter()
                .map(|l| crate::get_lib_name_candidates(l))
                .flatten()
                .collect();
            for choice in choices.iter() {
                if let Ok(lib) = Lib::new(choice) {
                    return lib;
                }
            }
            crate::panic_no_lib_found(lib_names[0], &choices);
        })
    }
}
#[cfg(feature = "dynamic-loading")]
pub use loaded::*;
