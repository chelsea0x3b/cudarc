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
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaAsyncNotificationType_enum as cudaAsyncNotificationType;
pub use self::cudaDataType_t as cudaDataType;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaEmulationMantissaControl_t as cudaEmulationMantissaControl;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaEmulationSpecialValuesSupport_t as cudaEmulationSpecialValuesSupport;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaEmulationStrategy_t as cudaEmulationStrategy;
pub use self::cudaError as cudaError_t;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::cudaGraphDependencyType_enum as cudaGraphDependencyType;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub use self::cudaOutputMode as cudaOutputMode_t;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUDAlogLevel_enum as cudaLogLevel;
#[cfg(any(feature = "cuda-11040"))]
pub const CUDART_VERSION: u32 = 11040;
#[cfg(any(feature = "cuda-11050"))]
pub const CUDART_VERSION: u32 = 11050;
#[cfg(any(feature = "cuda-11060"))]
pub const CUDART_VERSION: u32 = 11060;
#[cfg(any(feature = "cuda-11070"))]
pub const CUDART_VERSION: u32 = 11070;
#[cfg(any(feature = "cuda-11080"))]
pub const CUDART_VERSION: u32 = 11080;
#[cfg(any(feature = "cuda-12000"))]
pub const CUDART_VERSION: u32 = 12000;
#[cfg(any(feature = "cuda-12010"))]
pub const CUDART_VERSION: u32 = 12010;
#[cfg(any(feature = "cuda-12020"))]
pub const CUDART_VERSION: u32 = 12020;
#[cfg(any(feature = "cuda-12030"))]
pub const CUDART_VERSION: u32 = 12030;
#[cfg(any(feature = "cuda-12040"))]
pub const CUDART_VERSION: u32 = 12040;
#[cfg(any(feature = "cuda-12050"))]
pub const CUDART_VERSION: u32 = 12050;
#[cfg(any(feature = "cuda-12060"))]
pub const CUDART_VERSION: u32 = 12060;
#[cfg(any(feature = "cuda-12080"))]
pub const CUDART_VERSION: u32 = 12080;
#[cfg(any(feature = "cuda-12090"))]
pub const CUDART_VERSION: u32 = 12090;
#[cfg(any(feature = "cuda-13000"))]
pub const CUDART_VERSION: u32 = 13000;
#[cfg(any(feature = "cuda-13010"))]
pub const CUDART_VERSION: u32 = 13010;
#[cfg(any(feature = "cuda-13020"))]
pub const CUDART_VERSION: u32 = 13020;
#[cfg(any(feature = "cuda-13030"))]
pub const CUDART_VERSION: u32 = 13030;
pub const CUDA_IPC_HANDLE_SIZE: u32 = 64;
pub const cudaArrayColorAttachment: u32 = 32;
pub const cudaArrayCubemap: u32 = 4;
pub const cudaArrayDefault: u32 = 0;
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaArrayDeferredMapping: u32 = 128;
pub const cudaArrayLayered: u32 = 1;
pub const cudaArraySparse: u32 = 64;
pub const cudaArraySparsePropertiesSingleMipTail: u32 = 1;
pub const cudaArraySurfaceLoadStore: u32 = 2;
pub const cudaArrayTextureGather: u32 = 8;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub const cudaCooperativeLaunchMultiDeviceNoPostSync: u32 = 2;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub const cudaCooperativeLaunchMultiDeviceNoPreSync: u32 = 1;
pub const cudaDeviceBlockingSync: u32 = 4;
pub const cudaDeviceLmemResizeToMax: u32 = 16;
pub const cudaDeviceMapHost: u32 = 8;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000"))]
pub const cudaDeviceMask: u32 = 31;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaDeviceMask: u32 = 255;
pub const cudaDeviceScheduleAuto: u32 = 0;
pub const cudaDeviceScheduleBlockingSync: u32 = 4;
pub const cudaDeviceScheduleMask: u32 = 7;
pub const cudaDeviceScheduleSpin: u32 = 1;
pub const cudaDeviceScheduleYield: u32 = 2;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaDeviceSyncMemops: u32 = 128;
pub const cudaEventBlockingSync: u32 = 1;
pub const cudaEventDefault: u32 = 0;
pub const cudaEventDisableTiming: u32 = 2;
pub const cudaEventInterprocess: u32 = 4;
pub const cudaEventRecordDefault: u32 = 0;
pub const cudaEventRecordExternal: u32 = 1;
pub const cudaEventWaitDefault: u32 = 0;
pub const cudaEventWaitExternal: u32 = 1;
pub const cudaExternalMemoryDedicated: u32 = 1;
pub const cudaExternalSemaphoreSignalSkipNvSciBufMemSync: u32 = 1;
pub const cudaExternalSemaphoreWaitSkipNvSciBufMemSync: u32 = 2;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaGraphKernelNodePortDefault: u32 = 0;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaGraphKernelNodePortLaunchCompletion: u32 = 2;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaGraphKernelNodePortProgrammatic: u32 = 1;
pub const cudaHostAllocDefault: u32 = 0;
pub const cudaHostAllocMapped: u32 = 2;
pub const cudaHostAllocPortable: u32 = 1;
pub const cudaHostAllocWriteCombined: u32 = 4;
pub const cudaHostRegisterDefault: u32 = 0;
pub const cudaHostRegisterIoMemory: u32 = 4;
pub const cudaHostRegisterMapped: u32 = 2;
pub const cudaHostRegisterPortable: u32 = 1;
pub const cudaHostRegisterReadOnly: u32 = 8;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaInitDeviceFlagsAreValid: u32 = 1;
pub const cudaIpcMemLazyEnablePeerAccess: u32 = 1;
pub const cudaMemAttachGlobal: u32 = 1;
pub const cudaMemAttachHost: u32 = 2;
pub const cudaMemAttachSingle: u32 = 4;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const cudaMemPoolCreateUsageHwDecompress: u32 = 2;
pub const cudaNvSciSyncAttrSignal: u32 = 1;
pub const cudaNvSciSyncAttrWait: u32 = 2;
pub const cudaOccupancyDefault: u32 = 0;
pub const cudaOccupancyDisableCachingOverride: u32 = 1;
pub const cudaPeerAccessDefault: u32 = 0;
pub const cudaStreamDefault: u32 = 0;
pub const cudaStreamNonBlocking: u32 = 1;
pub const cudaSurfaceType1D: u32 = 1;
pub const cudaSurfaceType1DLayered: u32 = 241;
pub const cudaSurfaceType2D: u32 = 2;
pub const cudaSurfaceType2DLayered: u32 = 242;
pub const cudaSurfaceType3D: u32 = 3;
pub const cudaSurfaceTypeCubemap: u32 = 12;
pub const cudaSurfaceTypeCubemapLayered: u32 = 252;
pub const cudaTextureType1D: u32 = 1;
pub const cudaTextureType1DLayered: u32 = 241;
pub const cudaTextureType2D: u32 = 2;
pub const cudaTextureType2DLayered: u32 = 242;
pub const cudaTextureType3D: u32 = 3;
pub const cudaTextureTypeCubemap: u32 = 12;
pub const cudaTextureTypeCubemapLayered: u32 = 252;
pub type cudaArray_const_t = *const cudaArray;
pub type cudaArray_t = *mut cudaArray;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaAsyncCallback = ::core::option::Option<unsafe extern "C" fn(arg1: *mut cudaAsyncNotificationInfo_t, arg2: *mut ::core::ffi::c_void, arg3: cudaAsyncCallbackHandle_t)>;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaAsyncCallbackHandle_t = *mut cudaAsyncCallbackEntry;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaAsyncNotificationInfo_t = cudaAsyncNotificationInfo;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaDevResource = cudaDevResource_st;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaDevResourceDesc_t = *mut CUdevResourceDesc_st;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaDevSmResourceGroupParams = cudaDevSmResourceGroupParams_st;
pub type cudaEvent_t = *mut CUevent_st;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaExecutionContext_t = *mut cudaExecutionContext_st;
pub type cudaExternalMemory_t = *mut CUexternalMemory_st;
pub type cudaExternalSemaphore_t = *mut CUexternalSemaphore_st;
pub type cudaFunction_t = *mut CUfunc_st;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaGraphConditionalHandle = ::core::ffi::c_ulonglong;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaGraphDeviceNode_t = *mut CUgraphDeviceUpdatableNode_st;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaGraphEdgeData = cudaGraphEdgeData_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaGraphExecUpdateResultInfo = cudaGraphExecUpdateResultInfo_st;
pub type cudaGraphExec_t = *mut CUgraphExec_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaGraphInstantiateParams = cudaGraphInstantiateParams_st;
pub type cudaGraphNode_t = *mut CUgraphNode_st;
#[cfg(any(feature = "cuda-13030"))]
pub type cudaGraphRecaptureCallback_t = ::core::option::Option<unsafe extern "C" fn(data: *mut ::core::ffi::c_void, node: cudaGraphNode_t, originalParams: *const cudaGraphNodeParams, recaptureParams: *const cudaGraphNodeParams, status: cudaGraphRecaptureStatus) -> cudaError_t>;
pub type cudaGraph_t = *mut CUgraph_st;
pub type cudaGraphicsResource_t = *mut cudaGraphicsResource;
pub type cudaHostFn_t = ::core::option::Option<unsafe extern "C" fn(userData: *mut ::core::ffi::c_void)>;
pub type cudaIpcEventHandle_t = cudaIpcEventHandle_st;
pub type cudaIpcMemHandle_t = cudaIpcMemHandle_st;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaKernel_t = *mut CUkern_st;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLaunchAttribute = cudaLaunchAttribute_st;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLaunchConfig_t = cudaLaunchConfig_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLaunchMemSyncDomainMap = cudaLaunchMemSyncDomainMap_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLibrary_t = *mut CUlib_st;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLogIterator = ::core::ffi::c_uint;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLogsCallbackHandle = *mut CUlogsCallbackEntry_st;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaLogsCallback_t = ::core::option::Option<unsafe extern "C" fn(data: *mut ::core::ffi::c_void, logLevel: cudaLogLevel, message: *mut ::core::ffi::c_char, length: usize)>;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type cudaMemFabricHandle_t = cudaMemFabricHandle_st;
pub type cudaMemPool_t = *mut CUmemPoolHandle_st;
pub type cudaMipmappedArray_const_t = *const cudaMipmappedArray;
pub type cudaMipmappedArray_t = *mut cudaMipmappedArray;
pub type cudaStreamCallback_t = ::core::option::Option<unsafe extern "C" fn(stream: cudaStream_t, status: cudaError_t, userData: *mut ::core::ffi::c_void)>;
pub type cudaStream_t = *mut CUstream_st;
pub type cudaSurfaceObject_t = ::core::ffi::c_ulonglong;
pub type cudaTextureObject_t = ::core::ffi::c_ulonglong;
pub type cudaUUID_t = CUuuid_st;
pub type cudaUserObject_t = *mut CUuserObject_st;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUDAlogLevel_enum {
    cudaLogLevelError = 0,
    cudaLogLevelWarning = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaAccessProperty {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaAsyncNotificationType_enum {
    cudaAsyncNotificationTypeOverBudget = 1,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaAtomicOperation {
    cudaAtomicOperationIntegerAdd = 0,
    cudaAtomicOperationIntegerMin = 1,
    cudaAtomicOperationIntegerMax = 2,
    cudaAtomicOperationIntegerIncrement = 3,
    cudaAtomicOperationIntegerDecrement = 4,
    cudaAtomicOperationAnd = 5,
    cudaAtomicOperationOr = 6,
    cudaAtomicOperationXOR = 7,
    cudaAtomicOperationExchange = 8,
    cudaAtomicOperationCAS = 9,
    cudaAtomicOperationFloatAdd = 10,
    cudaAtomicOperationFloatMin = 11,
    cudaAtomicOperationFloatMax = 12,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaAtomicOperationCapability {
    cudaAtomicCapabilitySigned = 1,
    cudaAtomicCapabilityUnsigned = 2,
    cudaAtomicCapabilityReduction = 4,
    cudaAtomicCapabilityScalar32 = 8,
    cudaAtomicCapabilityScalar64 = 16,
    cudaAtomicCapabilityScalar128 = 32,
    cudaAtomicCapabilityVector32x4 = 64,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaCGScope {
    cudaCGScopeInvalid = 0,
    cudaCGScopeGrid = 1,
    cudaCGScopeMultiGrid = 2,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaCGScope {
    cudaCGScopeInvalid = 0,
    cudaCGScopeGrid = 1,
    cudaCGScopeReserved = 2,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
    cudaChannelFormatKindUnsignedNormalized8X1 = 5,
    cudaChannelFormatKindUnsignedNormalized8X2 = 6,
    cudaChannelFormatKindUnsignedNormalized8X4 = 7,
    cudaChannelFormatKindUnsignedNormalized16X1 = 8,
    cudaChannelFormatKindUnsignedNormalized16X2 = 9,
    cudaChannelFormatKindUnsignedNormalized16X4 = 10,
    cudaChannelFormatKindSignedNormalized8X1 = 11,
    cudaChannelFormatKindSignedNormalized8X2 = 12,
    cudaChannelFormatKindSignedNormalized8X4 = 13,
    cudaChannelFormatKindSignedNormalized16X1 = 14,
    cudaChannelFormatKindSignedNormalized16X2 = 15,
    cudaChannelFormatKindSignedNormalized16X4 = 16,
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
    cudaChannelFormatKindSignedBlockCompressed4 = 24,
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
    cudaChannelFormatKindSignedBlockCompressed5 = 26,
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
    cudaChannelFormatKindSignedBlockCompressed6H = 28,
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
    cudaChannelFormatKindUnsignedNormalized8X1 = 5,
    cudaChannelFormatKindUnsignedNormalized8X2 = 6,
    cudaChannelFormatKindUnsignedNormalized8X4 = 7,
    cudaChannelFormatKindUnsignedNormalized16X1 = 8,
    cudaChannelFormatKindUnsignedNormalized16X2 = 9,
    cudaChannelFormatKindUnsignedNormalized16X4 = 10,
    cudaChannelFormatKindSignedNormalized8X1 = 11,
    cudaChannelFormatKindSignedNormalized8X2 = 12,
    cudaChannelFormatKindSignedNormalized8X4 = 13,
    cudaChannelFormatKindSignedNormalized16X1 = 14,
    cudaChannelFormatKindSignedNormalized16X2 = 15,
    cudaChannelFormatKindSignedNormalized16X4 = 16,
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
    cudaChannelFormatKindSignedBlockCompressed4 = 24,
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
    cudaChannelFormatKindSignedBlockCompressed5 = 26,
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
    cudaChannelFormatKindSignedBlockCompressed6H = 28,
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30,
    cudaChannelFormatKindUnsignedNormalized1010102 = 31,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
    cudaChannelFormatKindUnsignedNormalized8X1 = 5,
    cudaChannelFormatKindUnsignedNormalized8X2 = 6,
    cudaChannelFormatKindUnsignedNormalized8X4 = 7,
    cudaChannelFormatKindUnsignedNormalized16X1 = 8,
    cudaChannelFormatKindUnsignedNormalized16X2 = 9,
    cudaChannelFormatKindUnsignedNormalized16X4 = 10,
    cudaChannelFormatKindSignedNormalized8X1 = 11,
    cudaChannelFormatKindSignedNormalized8X2 = 12,
    cudaChannelFormatKindSignedNormalized8X4 = 13,
    cudaChannelFormatKindSignedNormalized16X1 = 14,
    cudaChannelFormatKindSignedNormalized16X2 = 15,
    cudaChannelFormatKindSignedNormalized16X4 = 16,
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
    cudaChannelFormatKindSignedBlockCompressed4 = 24,
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
    cudaChannelFormatKindSignedBlockCompressed5 = 26,
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
    cudaChannelFormatKindSignedBlockCompressed6H = 28,
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30,
    cudaChannelFormatKindUnsignedNormalized1010102 = 31,
    cudaChannelFormatKindUnsigned8Packed422 = 32,
    cudaChannelFormatKindUnsigned8Packed444 = 33,
    cudaChannelFormatKindUnsigned8SemiPlanar420 = 34,
    cudaChannelFormatKindUnsigned16SemiPlanar420 = 35,
    cudaChannelFormatKindUnsigned8SemiPlanar422 = 36,
    cudaChannelFormatKindUnsigned16SemiPlanar422 = 37,
    cudaChannelFormatKindUnsigned8SemiPlanar444 = 38,
    cudaChannelFormatKindUnsigned16SemiPlanar444 = 39,
    cudaChannelFormatKindUnsigned8Planar420 = 40,
    cudaChannelFormatKindUnsigned16Planar420 = 41,
    cudaChannelFormatKindUnsigned8Planar422 = 42,
    cudaChannelFormatKindUnsigned16Planar422 = 43,
    cudaChannelFormatKindUnsigned8Planar444 = 44,
    cudaChannelFormatKindUnsigned16Planar444 = 45,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault = 0,
    cudaClusterSchedulingPolicySpread = 1,
    cudaClusterSchedulingPolicyLoadBalancing = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaComputeMode {
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3,
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
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDevResourceType {
    cudaDevResourceTypeInvalid = 0,
    cudaDevResourceTypeSm = 1,
    cudaDevResourceTypeWorkqueueConfig = 1000,
    cudaDevResourceTypeWorkqueue = 10000,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDevSmResourceGroup_flags {
    cudaDevSmResourceGroupDefault = 0,
    cudaDevSmResourceGroupBackfill = 1,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDevSmResourceSplitByCount_flags {
    cudaDevSmResourceSplitIgnoreSmCoscheduling = 1,
    cudaDevSmResourceSplitMaxPotentialClusterSize = 2,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDevWorkqueueConfigScope {
    cudaDevWorkqueueConfigScopeDeviceCtx = 0,
    cudaDevWorkqueueConfigScopeGreenCtxBalanced = 1,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrMax = 120,
}
#[cfg(any(feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrMax = 120,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrMax = 122,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrMax = 122,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrMax = 127,
}
#[cfg(any(feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMax = 133,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrMax = 135,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrMax = 135,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrMax = 136,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrGpuPciDeviceId = 139,
    cudaDevAttrGpuPciSubsystemId = 140,
    cudaDevAttrHostNumaMultinodeIpcSupported = 143,
    cudaDevAttrMax = 144,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrVulkanCigSupported = 138,
    cudaDevAttrGpuPciDeviceId = 139,
    cudaDevAttrGpuPciSubsystemId = 140,
    cudaDevAttrReserved141 = 141,
    cudaDevAttrHostNumaMemoryPoolsSupported = 142,
    cudaDevAttrHostNumaMultinodeIpcSupported = 143,
    cudaDevAttrMax = 144,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrReserved96 = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrVulkanCigSupported = 138,
    cudaDevAttrGpuPciDeviceId = 139,
    cudaDevAttrGpuPciSubsystemId = 140,
    cudaDevAttrReserved141 = 141,
    cudaDevAttrHostNumaMemoryPoolsSupported = 142,
    cudaDevAttrHostNumaMultinodeIpcSupported = 143,
    cudaDevAttrHostMemoryPoolsSupported = 144,
    cudaDevAttrReserved145 = 145,
    cudaDevAttrOnlyPartialHostNativeAtomicSupported = 147,
    cudaDevAttrMax = 148,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrReserved96 = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrVulkanCigSupported = 138,
    cudaDevAttrGpuPciDeviceId = 139,
    cudaDevAttrGpuPciSubsystemId = 140,
    cudaDevAttrReserved141 = 141,
    cudaDevAttrHostNumaMemoryPoolsSupported = 142,
    cudaDevAttrHostNumaMultinodeIpcSupported = 143,
    cudaDevAttrHostMemoryPoolsSupported = 144,
    cudaDevAttrReserved145 = 145,
    cudaDevAttrOnlyPartialHostNativeAtomicSupported = 147,
    cudaDevAttrAtomicReductionSupported = 148,
    cudaDevAttrCigStreamsSupported = 151,
    cudaDevAttrMax = 152,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceNumaConfig {
    cudaDeviceNumaConfigNone = 0,
    cudaDeviceNumaConfigNumaNode = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3,
    cudaDevP2PAttrCudaArrayAccessSupported = 4,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3,
    cudaDevP2PAttrCudaArrayAccessSupported = 4,
    cudaDevP2PAttrOnlyPartialNativeAtomicSupported = 5,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1,
    cudaDriverEntryPointVersionNotSufficent = 2,
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
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaEmulationStrategy_t {
    CUDA_EMULATION_STRATEGY_DEFAULT = 0,
    CUDA_EMULATION_STRATEGY_PERFORMANT = 1,
    CUDA_EMULATION_STRATEGY_EAGER = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorContained = 226,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorTensorMemoryLeak = 721,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorContained = 226,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorTensorMemoryLeak = 721,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorStreamDetached = 917,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorVersionTranslation = 10,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorContained = 226,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorTensorMemoryLeak = 721,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorStreamDetached = 917,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorVersionTranslation = 10,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorContained = 226,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorTensorMemoryLeak = 721,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorStreamDetached = 917,
    cudaErrorGraphRecaptureFailure = 918,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaExternalMemoryHandleType {
    cudaExternalMemoryHandleTypeOpaqueFd = 1,
    cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalMemoryHandleTypeD3D12Heap = 4,
    cudaExternalMemoryHandleTypeD3D12Resource = 5,
    cudaExternalMemoryHandleTypeD3D11Resource = 6,
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    cudaExternalMemoryHandleTypeNvSciBuf = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaExternalSemaphoreHandleType {
    cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalSemaphoreHandleTypeD3D12Fence = 4,
    cudaExternalSemaphoreHandleTypeD3D11Fence = 5,
    cudaExternalSemaphoreHandleTypeNvSciSync = 6,
    cudaExternalSemaphoreHandleTypeKeyedMutex = 7,
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFabricOpStatusInfo {
    cudaFabricOpStatusInfoSuccess = 0,
    cudaFabricOpStatusInfoMax = 2147483647,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFabricOpStatusSource {
    cudaFabricOpStatusSourceMbarrierV1 = 0,
    cudaFabricOpStatusSourceMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFlushGPUDirectRDMAWritesOptions {
    cudaFlushGPUDirectRDMAWritesOptionHost = 1,
    cudaFlushGPUDirectRDMAWritesOptionMemOps = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFlushGPUDirectRDMAWritesScope {
    cudaFlushGPUDirectRDMAWritesToOwner = 100,
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFlushGPUDirectRDMAWritesTarget {
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice = 0,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeMax = 10,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeClusterDimMustBeSet = 10,
    cudaFuncAttributeRequiredClusterWidth = 11,
    cudaFuncAttributeRequiredClusterHeight = 12,
    cudaFuncAttributeRequiredClusterDepth = 13,
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
    cudaFuncAttributeMax = 16,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGPUDirectRDMAWritesOrdering {
    cudaGPUDirectRDMAWritesOrderingNone = 0,
    cudaGPUDirectRDMAWritesOrderingOwner = 100,
    cudaGPUDirectRDMAWritesOrderingAllDevices = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGetDriverEntryPointFlags {
    cudaEnableDefault = 0,
    cudaEnableLegacyStream = 1,
    cudaEnablePerThreadDefaultStream = 2,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphChildGraphNodeOwnership {
    cudaGraphChildGraphOwnershipClone = 0,
    cudaGraphChildGraphOwnershipMove = 1,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphChildGraphNodeOwnership {
    cudaGraphChildGraphOwnershipClone = 0,
    cudaGraphChildGraphOwnershipMove = 1,
    cudaGraphChildGraphOwnershipInvalid = -1,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphConditionalHandleFlags {
    cudaGraphCondAssignDefault = 1,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf = 0,
    cudaGraphCondTypeWhile = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf = 0,
    cudaGraphCondTypeWhile = 1,
    cudaGraphCondTypeSwitch = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose = 1,
    cudaGraphDebugDotFlagsKernelNodeParams = 4,
    cudaGraphDebugDotFlagsMemcpyNodeParams = 8,
    cudaGraphDebugDotFlagsMemsetNodeParams = 16,
    cudaGraphDebugDotFlagsHostNodeParams = 32,
    cudaGraphDebugDotFlagsEventNodeParams = 64,
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 128,
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 256,
    cudaGraphDebugDotFlagsKernelNodeAttributes = 512,
    cudaGraphDebugDotFlagsHandles = 1024,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose = 1,
    cudaGraphDebugDotFlagsKernelNodeParams = 4,
    cudaGraphDebugDotFlagsMemcpyNodeParams = 8,
    cudaGraphDebugDotFlagsMemsetNodeParams = 16,
    cudaGraphDebugDotFlagsHostNodeParams = 32,
    cudaGraphDebugDotFlagsEventNodeParams = 64,
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 128,
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 256,
    cudaGraphDebugDotFlagsKernelNodeAttributes = 512,
    cudaGraphDebugDotFlagsHandles = 1024,
    cudaGraphDebugDotFlagsConditionalNodeParams = 32768,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphDependencyType_enum {
    cudaGraphDependencyTypeDefault = 0,
    cudaGraphDependencyTypeProgrammatic = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
    cudaGraphExecUpdateErrorAttributesChanged = 8,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUseNodePriority = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUpload = 2,
    cudaGraphInstantiateFlagDeviceLaunch = 4,
    cudaGraphInstantiateFlagUseNodePriority = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess = 0,
    cudaGraphInstantiateError = 1,
    cudaGraphInstantiateInvalidStructure = 2,
    cudaGraphInstantiateNodeOperationNotSupported = 3,
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess = 0,
    cudaGraphInstantiateError = 1,
    cudaGraphInstantiateInvalidStructure = 2,
    cudaGraphInstantiateNodeOperationNotSupported = 3,
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,
    cudaGraphInstantiateConditionalHandleUnused = 5,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphKernelNodeField {
    cudaGraphKernelNodeFieldInvalid = 0,
    cudaGraphKernelNodeFieldGridDim = 1,
    cudaGraphKernelNodeFieldParam = 2,
    cudaGraphKernelNodeFieldEnabled = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphMemAttributeType {
    cudaGraphMemAttrUsedMemCurrent = 1,
    cudaGraphMemAttrUsedMemHigh = 2,
    cudaGraphMemAttrReservedMemCurrent = 3,
    cudaGraphMemAttrReservedMemHigh = 4,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphMemAttributeType {
    cudaGraphMemAttrUsedMemCurrent = 0,
    cudaGraphMemAttrUsedMemHigh = 1,
    cudaGraphMemAttrReservedMemCurrent = 2,
    cudaGraphMemAttrReservedMemHigh = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeCount = 12,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeConditional = 13,
    cudaGraphNodeTypeCount = 14,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeConditional = 13,
    cudaGraphNodeTypeReserved16 = 16,
    cudaGraphNodeTypeCount = 17,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphRecaptureStatus {
    cudaGraphRecaptureEligibleForUpdate = 0,
    cudaGraphRecaptureIneligibleForUpdate = 1,
    cudaGraphRecaptureError = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphicsCubeFace {
    cudaGraphicsCubeFacePositiveX = 0,
    cudaGraphicsCubeFaceNegativeX = 1,
    cudaGraphicsCubeFacePositiveY = 2,
    cudaGraphicsCubeFaceNegativeY = 3,
    cudaGraphicsCubeFacePositiveZ = 4,
    cudaGraphicsCubeFaceNegativeZ = 5,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphicsMapFlags {
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphicsRegisterFlags {
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaHostTaskSyncMode {
    cudaHostTaskBlocking = 0,
    cudaHostTaskSpinWait = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaJitOption {
    cudaJitMaxRegisters = 0,
    cudaJitThreadsPerBlock = 1,
    cudaJitWallTime = 2,
    cudaJitInfoLogBuffer = 3,
    cudaJitInfoLogBufferSizeBytes = 4,
    cudaJitErrorLogBuffer = 5,
    cudaJitErrorLogBufferSizeBytes = 6,
    cudaJitOptimizationLevel = 7,
    cudaJitFallbackStrategy = 10,
    cudaJitGenerateDebugInfo = 11,
    cudaJitLogVerbose = 12,
    cudaJitGenerateLineInfo = 13,
    cudaJitCacheMode = 14,
    cudaJitPositionIndependentCode = 30,
    cudaJitMinCtaPerSm = 31,
    cudaJitMaxThreadsPerBlock = 32,
    cudaJitOverrideDirectiveValues = 33,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaJit_CacheMode {
    cudaJitCacheOptionNone = 0,
    cudaJitCacheOptionCG = 1,
    cudaJitCacheOptionCA = 2,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaJit_Fallback {
    cudaPreferPtx = 0,
    cudaPreferBinary = 1,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaKernelFunctionType {
    cudaKernelFunctionTypeUnspecified = 0,
    cudaKernelFunctionTypeDeviceEntry = 1,
    cudaKernelFunctionTypeKernel = 2,
    cudaKernelFunctionTypeFunction = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaKernelNodeAttrID {
    cudaKernelNodeAttributeAccessPolicyWindow = 1,
    cudaKernelNodeAttributeCooperative = 2,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaKernelNodeAttrID {
    cudaKernelNodeAttributeAccessPolicyWindow = 1,
    cudaKernelNodeAttributeCooperative = 2,
    cudaKernelNodeAttributePriority = 8,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributePreferredClusterDimension = 11,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributePreferredClusterDimension = 11,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
    cudaLaunchAttributeNvlinkUtilCentricScheduling = 16,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributePreferredClusterDimension = 11,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
    cudaLaunchAttributeNvlinkUtilCentricScheduling = 16,
    cudaLaunchAttributePortableClusterSizeMode = 17,
    cudaLaunchAttributeSharedMemoryMode = 18,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributePortableClusterMode {
    cudaLaunchPortableClusterModeDefault = 0,
    cudaLaunchPortableClusterModeRequirePortable = 1,
    cudaLaunchPortableClusterModeAllowNonPortable = 2,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchMemSyncDomain {
    cudaLaunchMemSyncDomainDefault = 0,
    cudaLaunchMemSyncDomainRemote = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLibraryOption {
    cudaLibraryHostUniversalFunctionAndDataTable = 0,
    cudaLibraryBinaryIsPreserved = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLimit {
    cudaLimitStackSize = 0,
    cudaLimitPrintfFifoSize = 1,
    cudaLimitMallocHeapSize = 2,
    cudaLimitDevRuntimeSyncDepth = 3,
    cudaLimitDevRuntimePendingLaunchCount = 4,
    cudaLimitMaxL2FetchGranularity = 5,
    cudaLimitPersistingL2CacheSize = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone = 0,
    cudaMemAccessFlagsProtRead = 1,
    cudaMemAccessFlagsProtReadWrite = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationHandleType {
    cudaMemHandleTypeNone = 0,
    cudaMemHandleTypePosixFileDescriptor = 1,
    cudaMemHandleTypeWin32 = 2,
    cudaMemHandleTypeWin32Kmt = 4,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationHandleType {
    cudaMemHandleTypeNone = 0,
    cudaMemHandleTypePosixFileDescriptor = 1,
    cudaMemHandleTypeWin32 = 2,
    cudaMemHandleTypeWin32Kmt = 4,
    cudaMemHandleTypeFabric = 8,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0,
    cudaMemAllocationTypePinned = 1,
    cudaMemAllocationTypeMax = 2147483647,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0,
    cudaMemAllocationTypePinned = 1,
    cudaMemAllocationTypeManaged = 2,
    cudaMemAllocationTypeMax = 2147483647,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
    cudaMemLocationTypeHostNuma = 3,
    cudaMemLocationTypeHostNumaCurrent = 4,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
    cudaMemLocationTypeHostNuma = 3,
    cudaMemLocationTypeHostNumaCurrent = 4,
    cudaMemLocationTypeInvisible = 5,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies = 1,
    cudaMemPoolReuseAllowOpportunistic = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold = 4,
    cudaMemPoolAttrReservedMemCurrent = 5,
    cudaMemPoolAttrReservedMemHigh = 6,
    cudaMemPoolAttrUsedMemCurrent = 7,
    cudaMemPoolAttrUsedMemHigh = 8,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies = 1,
    cudaMemPoolReuseAllowOpportunistic = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold = 4,
    cudaMemPoolAttrReservedMemCurrent = 5,
    cudaMemPoolAttrReservedMemHigh = 6,
    cudaMemPoolAttrUsedMemCurrent = 7,
    cudaMemPoolAttrUsedMemHigh = 8,
    cudaMemPoolAttrAllocationType = 9,
    cudaMemPoolAttrExportHandleTypes = 10,
    cudaMemPoolAttrLocationId = 11,
    cudaMemPoolAttrLocationType = 12,
    cudaMemPoolAttrMaxPoolSize = 13,
    cudaMemPoolAttrHwDecompressEnabled = 14,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemRangeAttribute {
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemRangeAttribute {
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4,
    cudaMemRangeAttributePreferredLocationType = 5,
    cudaMemRangeAttributePreferredLocationId = 6,
    cudaMemRangeAttributeLastPrefetchLocationType = 7,
    cudaMemRangeAttributeLastPrefetchLocationId = 8,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpy3DOperandType {
    cudaMemcpyOperandTypePointer = 1,
    cudaMemcpyOperandTypeArray = 2,
    cudaMemcpyOperandTypeMax = 2147483647,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpyFlags {
    cudaMemcpyFlagDefault = 0,
    cudaMemcpyFlagPreferOverlapWithCompute = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpySrcAccessOrder {
    cudaMemcpySrcAccessOrderInvalid = 0,
    cudaMemcpySrcAccessOrderStream = 1,
    cudaMemcpySrcAccessOrderDuringApiCall = 2,
    cudaMemcpySrcAccessOrderAny = 3,
    cudaMemcpySrcAccessOrderMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemoryAdvise {
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaOutputMode {
    cudaKeyValuePair = 0,
    cudaCSV = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaResourceType {
    cudaResourceTypeArray = 0,
    cudaResourceTypeMipmappedArray = 1,
    cudaResourceTypeLinear = 2,
    cudaResourceTypePitch2D = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaResourceViewFormat {
    cudaResViewFormatNone = 0,
    cudaResViewFormatUnsignedChar1 = 1,
    cudaResViewFormatUnsignedChar2 = 2,
    cudaResViewFormatUnsignedChar4 = 3,
    cudaResViewFormatSignedChar1 = 4,
    cudaResViewFormatSignedChar2 = 5,
    cudaResViewFormatSignedChar4 = 6,
    cudaResViewFormatUnsignedShort1 = 7,
    cudaResViewFormatUnsignedShort2 = 8,
    cudaResViewFormatUnsignedShort4 = 9,
    cudaResViewFormatSignedShort1 = 10,
    cudaResViewFormatSignedShort2 = 11,
    cudaResViewFormatSignedShort4 = 12,
    cudaResViewFormatUnsignedInt1 = 13,
    cudaResViewFormatUnsignedInt2 = 14,
    cudaResViewFormatUnsignedInt4 = 15,
    cudaResViewFormatSignedInt1 = 16,
    cudaResViewFormatSignedInt2 = 17,
    cudaResViewFormatSignedInt4 = 18,
    cudaResViewFormatHalf1 = 19,
    cudaResViewFormatHalf2 = 20,
    cudaResViewFormatHalf4 = 21,
    cudaResViewFormatFloat1 = 22,
    cudaResViewFormatFloat2 = 23,
    cudaResViewFormatFloat4 = 24,
    cudaResViewFormatUnsignedBlockCompressed1 = 25,
    cudaResViewFormatUnsignedBlockCompressed2 = 26,
    cudaResViewFormatUnsignedBlockCompressed3 = 27,
    cudaResViewFormatUnsignedBlockCompressed4 = 28,
    cudaResViewFormatSignedBlockCompressed4 = 29,
    cudaResViewFormatUnsignedBlockCompressed5 = 30,
    cudaResViewFormatSignedBlockCompressed5 = 31,
    cudaResViewFormatUnsignedBlockCompressed6H = 32,
    cudaResViewFormatSignedBlockCompressed6H = 33,
    cudaResViewFormatUnsignedBlockCompressed7 = 34,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaRoundMode {
    cudaRoundNearest = 0,
    cudaRoundZero = 1,
    cudaRoundPosInf = 2,
    cudaRoundMinInf = 3,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSharedCarveout {
    cudaSharedmemCarveoutDefault = -1,
    cudaSharedmemCarveoutMaxShared = 100,
    cudaSharedmemCarveoutMaxL1 = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSharedMemoryMode {
    cudaSharedMemoryModeDefault = 0,
    cudaSharedMemoryModeRequirePortable = 1,
    cudaSharedMemoryModeAllowNonPortable = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamAttrID {
    cudaStreamAttributeAccessPolicyWindow = 1,
    cudaStreamAttributeSynchronizationPolicy = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamUpdateCaptureDependenciesFlags {
    cudaStreamAddCaptureDependencies = 0,
    cudaStreamSetCaptureDependencies = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSurfaceBoundaryMode {
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSurfaceFormatMode {
    cudaFormatModeForced = 0,
    cudaFormatModeAuto = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaTextureAddressMode {
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaTextureFilterMode {
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaTextureReadMode {
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaUserObjectFlags {
    cudaUserObjectNoDestructorSync = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaUserObjectRetainFlags {
    cudaGraphUserObjectMove = 1,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUdevResourceDesc_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUexternalMemory_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUexternalSemaphore_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUfunc_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphDeviceUpdatableNode_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphNode_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraph_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUkern_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlib_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlogsCallbackEntry_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmemPoolHandle_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUuserObject_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUuuid_st {
    pub bytes: [::core::ffi::c_char; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaAccessPolicyWindow {
    pub base_ptr: *mut ::core::ffi::c_void,
    pub num_bytes: usize,
    pub hitRatio: f32,
    pub hitProp: cudaAccessProperty,
    pub missProp: cudaAccessProperty,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaArray {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaArrayMemoryRequirements {
    pub size: usize,
    pub alignment: usize,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaArraySparseProperties {
    pub tileExtent: cudaArraySparseProperties__bindgen_ty_1,
    pub miptailFirstLevel: ::core::ffi::c_uint,
    pub miptailSize: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaArraySparseProperties__bindgen_ty_1 {
    pub width: ::core::ffi::c_uint,
    pub height: ::core::ffi::c_uint,
    pub depth: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaAsyncCallbackEntry {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaAsyncNotificationInfo {
    pub type_: cudaAsyncNotificationType,
    pub info: cudaAsyncNotificationInfo__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaAsyncNotificationInfo__bindgen_ty_1__bindgen_ty_1 {
    pub bytesOverBudget: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaChannelFormatDesc {
    pub x: ::core::ffi::c_int,
    pub y: ::core::ffi::c_int,
    pub z: ::core::ffi::c_int,
    pub w: ::core::ffi::c_int,
    pub f: cudaChannelFormatKind,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaChildGraphNodeParams {
    pub graph: cudaGraph_t,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaChildGraphNodeParams {
    pub graph: cudaGraph_t,
    pub ownership: cudaGraphChildGraphNodeOwnership,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaConditionalNodeParams {
    pub handle: cudaGraphConditionalHandle,
    pub type_: cudaGraphConditionalNodeType,
    pub size: ::core::ffi::c_uint,
    pub phGraph_out: *mut cudaGraph_t,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaConditionalNodeParams {
    pub handle: cudaGraphConditionalHandle,
    pub type_: cudaGraphConditionalNodeType,
    pub size: ::core::ffi::c_uint,
    pub phGraph_out: *mut cudaGraph_t,
    pub ctx: cudaExecutionContext_t,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaDevResource_st {
    pub type_: cudaDevResourceType,
    pub _internal_padding: [::core::ffi::c_uchar; 92usize],
    pub __bindgen_anon_1: cudaDevResource_st__bindgen_ty_1,
    pub nextResource: *mut cudaDevResource_st,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDevSmResource {
    pub smCount: ::core::ffi::c_uint,
    pub minSmPartitionSize: ::core::ffi::c_uint,
    pub smCoscheduledAlignment: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDevSmResourceGroupParams_st {
    pub smCount: ::core::ffi::c_uint,
    pub coscheduledSmCount: ::core::ffi::c_uint,
    pub preferredCoscheduledSmCount: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDevWorkqueueConfigResource {
    pub device: ::core::ffi::c_int,
    pub wqConcurrencyLimit: ::core::ffi::c_uint,
    pub sharingScope: cudaDevWorkqueueConfigScope,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDevWorkqueueResource {
    pub reserved: [::core::ffi::c_uchar; 40usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 63usize],
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved2: [::core::ffi::c_int; 2usize],
    pub reserved: [::core::ffi::c_int; 61usize],
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved2: [::core::ffi::c_int; 2usize],
    pub reserved1: [::core::ffi::c_int; 1usize],
    pub reserved: [::core::ffi::c_int; 60usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 63usize],
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub deviceNumaConfig: ::core::ffi::c_int,
    pub deviceNumaId: ::core::ffi::c_int,
    pub mpsEnabled: ::core::ffi::c_int,
    pub hostNumaId: ::core::ffi::c_int,
    pub gpuPciDeviceID: ::core::ffi::c_uint,
    pub gpuPciSubsystemID: ::core::ffi::c_uint,
    pub hostNumaMultinodeIpcSupported: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 56usize],
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaEventRecordNodeParams {
    pub event: cudaEvent_t,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaEventWaitNodeParams {
    pub event: cudaEvent_t,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaExecutionContext_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryBufferDesc {
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryBufferDesc {
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalMemoryHandleDesc {
    pub type_: cudaExternalMemoryHandleType,
    pub handle: cudaExternalMemoryHandleDesc__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalMemoryHandleDesc {
    pub type_: cudaExternalMemoryHandleType,
    pub handle: cudaExternalMemoryHandleDesc__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryMipmappedArrayDesc {
    pub offset: ::core::ffi::c_ulonglong,
    pub formatDesc: cudaChannelFormatDesc,
    pub extent: cudaExtent,
    pub flags: ::core::ffi::c_uint,
    pub numLevels: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryMipmappedArrayDesc {
    pub offset: ::core::ffi::c_ulonglong,
    pub formatDesc: cudaChannelFormatDesc,
    pub extent: cudaExtent,
    pub flags: ::core::ffi::c_uint,
    pub numLevels: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreHandleDesc {
    pub type_: cudaExternalSemaphoreHandleType,
    pub handle: cudaExternalSemaphoreHandleDesc__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreHandleDesc {
    pub type_: cudaExternalSemaphoreHandleType,
    pub handle: cudaExternalSemaphoreHandleDesc__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalNodeParams {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreSignalParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalNodeParamsV2 {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreSignalParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams {
    pub params: cudaExternalSemaphoreSignalParams__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams_v1 {
    pub params: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_3,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitNodeParams {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreWaitParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitNodeParamsV2 {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreWaitParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams {
    pub params: cudaExternalSemaphoreWaitParams__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 10usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams_v1 {
    pub params: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_3,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
    pub clusterDimMustBeSet: ::core::ffi::c_int,
    pub requiredClusterWidth: ::core::ffi::c_int,
    pub requiredClusterHeight: ::core::ffi::c_int,
    pub requiredClusterDepth: ::core::ffi::c_int,
    pub clusterSchedulingPolicyPreference: ::core::ffi::c_int,
    pub nonPortableClusterSizeAllowed: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 16usize],
}
#[cfg(any(feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
    pub clusterDimMustBeSet: ::core::ffi::c_int,
    pub requiredClusterWidth: ::core::ffi::c_int,
    pub requiredClusterHeight: ::core::ffi::c_int,
    pub requiredClusterDepth: ::core::ffi::c_int,
    pub clusterSchedulingPolicyPreference: ::core::ffi::c_int,
    pub nonPortableClusterSizeAllowed: ::core::ffi::c_int,
    pub reserved0: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 15usize],
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
    pub clusterDimMustBeSet: ::core::ffi::c_int,
    pub requiredClusterWidth: ::core::ffi::c_int,
    pub requiredClusterHeight: ::core::ffi::c_int,
    pub requiredClusterDepth: ::core::ffi::c_int,
    pub clusterSchedulingPolicyPreference: ::core::ffi::c_int,
    pub nonPortableClusterSizeAllowed: ::core::ffi::c_int,
    pub deviceNodeUpdateStatus: ::core::ffi::c_int,
    pub reserved1: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 14usize],
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphEdgeData_st {
    pub from_port: ::core::ffi::c_uchar,
    pub to_port: ::core::ffi::c_uchar,
    pub type_: ::core::ffi::c_uchar,
    pub reserved: [::core::ffi::c_uchar; 5usize],
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphExecUpdateResultInfo_st {
    pub result: cudaGraphExecUpdateResult,
    pub errorNode: cudaGraphNode_t,
    pub errorFromNode: cudaGraphNode_t,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphInstantiateParams_st {
    pub flags: ::core::ffi::c_ulonglong,
    pub uploadStream: cudaStream_t,
    pub errNode_out: cudaGraphNode_t,
    pub result_out: cudaGraphInstantiateResult,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaGraphKernelNodeUpdate {
    pub node: cudaGraphDeviceNode_t,
    pub field: cudaGraphKernelNodeField,
    pub updateData: cudaGraphKernelNodeUpdate__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphKernelNodeUpdate__bindgen_ty_1__bindgen_ty_1 {
    pub pValue: *const ::core::ffi::c_void,
    pub offset: usize,
    pub size: usize,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaGraphNodeParams {
    pub type_: cudaGraphNodeType,
    pub reserved0: [::core::ffi::c_int; 3usize],
    pub __bindgen_anon_1: cudaGraphNodeParams__bindgen_ty_1,
    pub reserved2: ::core::ffi::c_longlong,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphRecaptureCallbackData {
    pub callbackFunc: cudaGraphRecaptureCallback_t,
    pub userData: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaGraphicsResource {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaHostNodeParams {
    pub fn_: cudaHostFn_t,
    pub userData: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaHostNodeParamsV2 {
    pub fn_: cudaHostFn_t,
    pub userData: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaHostNodeParamsV2 {
    pub fn_: cudaHostFn_t,
    pub userData: *mut ::core::ffi::c_void,
    pub syncMode: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaIpcEventHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaIpcMemHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaKernelNodeParams {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaKernelNodeParamsV2 {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaKernelNodeParamsV2 {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub ctx: cudaExecutionContext_t,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaKernelNodeParamsV2 {
    pub __bindgen_anon_1: cudaKernelNodeParamsV2__bindgen_ty_1,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub ctx: cudaExecutionContext_t,
    pub functionType: cudaKernelFunctionType,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_1 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_2 {
    pub event: cudaEvent_t,
    pub flags: ::core::ffi::c_int,
    pub triggerAtBlockStart: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_3 {
    pub event: cudaEvent_t,
    pub flags: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_3 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_4 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: cudaGraphDeviceNode_t,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_4 {
    pub event: cudaEvent_t,
    pub flags: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_5 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: cudaGraphDeviceNode_t,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaLaunchAttribute_st {
    pub id: cudaLaunchAttributeID,
    pub pad: [::core::ffi::c_char; 4usize],
    pub val: cudaLaunchAttributeValue,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchConfig_st {
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub dynamicSmemBytes: usize,
    pub stream: cudaStream_t,
    pub attrs: *mut cudaLaunchAttribute,
    pub numAttrs: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchMemSyncDomainMap_st {
    pub default_: ::core::ffi::c_uchar,
    pub remote: ::core::ffi::c_uchar,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchParams {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub args: *mut *mut ::core::ffi::c_void,
    pub sharedMem: usize,
    pub stream: cudaStream_t,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemAccessDesc {
    pub location: cudaMemLocation,
    pub flags: cudaMemAccessFlags,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemAccessDesc {
    pub location: cudaMemLocation,
    pub flags: cudaMemAccessFlags,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemAllocNodeParams {
    pub poolProps: cudaMemPoolProps,
    pub accessDescs: *const cudaMemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemAllocNodeParams {
    pub poolProps: cudaMemPoolProps,
    pub accessDescs: *const cudaMemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemAllocNodeParamsV2 {
    pub poolProps: cudaMemPoolProps,
    pub accessDescs: *const cudaMemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemAllocNodeParamsV2 {
    pub poolProps: cudaMemPoolProps,
    pub accessDescs: *const cudaMemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemFabricHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemFreeNodeParams {
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemLocation {
    pub type_: cudaMemLocationType,
    pub id: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemLocation {
    pub type_: cudaMemLocationType,
    pub __bindgen_anon_1: cudaMemLocation__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub reserved: [::core::ffi::c_uchar; 56usize],
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 54usize],
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 54usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolPtrExportData {
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemcpy3DBatchOp {
    pub src: cudaMemcpy3DOperand,
    pub dst: cudaMemcpy3DOperand,
    pub extent: cudaExtent,
    pub srcAccessOrder: cudaMemcpySrcAccessOrder,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemcpy3DOperand {
    pub type_: cudaMemcpy3DOperandType,
    pub op: cudaMemcpy3DOperand__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: *mut ::core::ffi::c_void,
    pub rowLength: usize,
    pub layerHeight: usize,
    pub locHint: cudaMemLocation,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: *mut ::core::ffi::c_void,
    pub rowLength: usize,
    pub layerHeight: usize,
    pub locHint: cudaMemLocation,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2 {
    pub array: cudaArray_t,
    pub offset: cudaOffset3D,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DParms {
    pub srcArray: cudaArray_t,
    pub srcPos: cudaPos,
    pub srcPtr: cudaPitchedPtr,
    pub dstArray: cudaArray_t,
    pub dstPos: cudaPos,
    pub dstPtr: cudaPitchedPtr,
    pub extent: cudaExtent,
    pub kind: cudaMemcpyKind,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DPeerParms {
    pub srcArray: cudaArray_t,
    pub srcPos: cudaPos,
    pub srcPtr: cudaPitchedPtr,
    pub srcDevice: ::core::ffi::c_int,
    pub dstArray: cudaArray_t,
    pub dstPos: cudaPos,
    pub dstPtr: cudaPitchedPtr,
    pub dstDevice: ::core::ffi::c_int,
    pub extent: cudaExtent,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpyAttributes {
    pub srcAccessOrder: cudaMemcpySrcAccessOrder,
    pub srcLocHint: cudaMemLocation,
    pub dstLocHint: cudaMemLocation,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemcpyAttributes {
    pub srcAccessOrder: cudaMemcpySrcAccessOrder,
    pub srcLocHint: cudaMemLocation,
    pub dstLocHint: cudaMemLocation,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpyNodeParams {
    pub flags: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 3usize],
    pub copyParams: cudaMemcpy3DParms,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpyNodeParams {
    pub flags: ::core::ffi::c_int,
    pub reserved: ::core::ffi::c_int,
    pub ctx: cudaExecutionContext_t,
    pub copyParams: cudaMemcpy3DParms,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemsetParams {
    pub dst: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemsetParamsV2 {
    pub dst: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemsetParamsV2 {
    pub dst: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
    pub ctx: cudaExecutionContext_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaMipmappedArray {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaOffset3D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPitchedPtr {
    pub ptr: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub xsize: usize,
    pub ysize: usize,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPointerAttributes {
    pub type_: cudaMemoryType,
    pub device: ::core::ffi::c_int,
    pub devicePointer: *mut ::core::ffi::c_void,
    pub hostPointer: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPointerAttributes {
    pub type_: cudaMemoryType,
    pub device: ::core::ffi::c_int,
    pub devicePointer: *mut ::core::ffi::c_void,
    pub hostPointer: *mut ::core::ffi::c_void,
    pub reserved: [::core::ffi::c_long; 8usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPos {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaResourceDesc {
    pub resType: cudaResourceType,
    pub res: cudaResourceDesc__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaResourceDesc {
    pub resType: cudaResourceType,
    pub res: cudaResourceDesc__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 {
    pub array: cudaArray_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_2 {
    pub mipmap: cudaMipmappedArray_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_3 {
    pub devPtr: *mut ::core::ffi::c_void,
    pub desc: cudaChannelFormatDesc,
    pub sizeInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_4 {
    pub devPtr: *mut ::core::ffi::c_void,
    pub desc: cudaChannelFormatDesc,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_5 {
    pub reserved: [::core::ffi::c_int; 32usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceViewDesc {
    pub format: cudaResourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::core::ffi::c_uint,
    pub lastMipmapLevel: ::core::ffi::c_uint,
    pub firstLayer: ::core::ffi::c_uint,
    pub lastLayer: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceViewDesc {
    pub format: cudaResourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::core::ffi::c_uint,
    pub lastMipmapLevel: ::core::ffi::c_uint,
    pub firstLayer: ::core::ffi::c_uint,
    pub lastLayer: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub seamlessCubemap: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub seamlessCubemap: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc_v2 {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub seamlessCubemap: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudalibraryHostUniversalFunctionAndDataTable {
    pub functionTable: *mut ::core::ffi::c_void,
    pub functionWindowSize: usize,
    pub dataTable: *mut ::core::ffi::c_void,
    pub dataWindowSize: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct dim3 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct surfaceReference {
    pub channelDesc: cudaChannelFormatDesc,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct textureReference {
    pub normalized: ::core::ffi::c_int,
    pub filterMode: cudaTextureFilterMode,
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub channelDesc: cudaChannelFormatDesc,
    pub sRGB: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub __cudaReserved: [::core::ffi::c_int; 14usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
impl cudaDeviceAttr {
    pub const cudaDevAttrMaxTimelineSemaphoreInteropSupported: cudaDeviceAttr = cudaDeviceAttr::cudaDevAttrTimelineSemaphoreInteropSupported;
}
#[cfg(any(feature = "cuda-13030"))]
impl cudaFabricOpStatusInfo {
    pub const cudaFabricOpStatusInfoLast: cudaFabricOpStatusInfo = cudaFabricOpStatusInfo::cudaFabricOpStatusInfoSuccess;
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl cudaMemLocationType {
    pub const cudaMemLocationTypeNone: cudaMemLocationType = cudaMemLocationType::cudaMemLocationTypeInvalid;
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaAsyncNotificationInfo__bindgen_ty_1 {
    pub overBudget: cudaAsyncNotificationInfo__bindgen_ty_1__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaDevResource_st__bindgen_ty_1 {
    pub sm: cudaDevSmResource,
    pub wqConfig: cudaDevWorkqueueConfigResource,
    pub wq: cudaDevWorkqueueResource,
    pub _oversize: [::core::ffi::c_uchar; 40usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalMemoryHandleDesc__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1,
    pub nvSciBufObject: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreHandleDesc__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSyncObj: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaGraphKernelNodeUpdate__bindgen_ty_1 {
    pub gridDim: dim3,
    pub param: cudaGraphKernelNodeUpdate__bindgen_ty_1__bindgen_ty_1,
    pub isEnabled: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaGraphNodeParams__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: cudaKernelNodeParamsV2,
    pub memcpy: cudaMemcpyNodeParams,
    pub memset: cudaMemsetParamsV2,
    pub host: cudaHostNodeParamsV2,
    pub graph: cudaChildGraphNodeParams,
    pub eventWait: cudaEventWaitNodeParams,
    pub eventRecord: cudaEventRecordNodeParams,
    pub extSemSignal: cudaExternalSemaphoreSignalNodeParamsV2,
    pub extSemWait: cudaExternalSemaphoreWaitNodeParamsV2,
    pub alloc: cudaMemAllocNodeParamsV2,
    pub free: cudaMemFreeNodeParams,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaGraphNodeParams__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: cudaKernelNodeParamsV2,
    pub memcpy: cudaMemcpyNodeParams,
    pub memset: cudaMemsetParamsV2,
    pub host: cudaHostNodeParamsV2,
    pub graph: cudaChildGraphNodeParams,
    pub eventWait: cudaEventWaitNodeParams,
    pub eventRecord: cudaEventRecordNodeParams,
    pub extSemSignal: cudaExternalSemaphoreSignalNodeParamsV2,
    pub extSemWait: cudaExternalSemaphoreWaitNodeParamsV2,
    pub alloc: cudaMemAllocNodeParamsV2,
    pub free: cudaMemFreeNodeParams,
    pub conditional: cudaConditionalNodeParams,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaKernelNodeAttrValue {
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaKernelNodeAttrValue {
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaKernelNodeParamsV2__bindgen_ty_1 {
    pub func: *mut ::core::ffi::c_void,
    pub kern: cudaKernel_t,
    pub cuFunc: cudaFunction_t,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_3,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_3,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_4,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_3,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_4,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub preferredClusterDim: cudaLaunchAttributeValue__bindgen_ty_3,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_4,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub preferredClusterDim: cudaLaunchAttributeValue__bindgen_ty_3,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_4,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
    pub nvlinkUtilCentricScheduling: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub preferredClusterDim: cudaLaunchAttributeValue__bindgen_ty_3,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_4,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
    pub nvlinkUtilCentricScheduling: ::core::ffi::c_uint,
    pub portableClusterSizeMode: cudaLaunchAttributePortableClusterMode,
    pub sharedMemoryMode: cudaSharedMemoryMode,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaMemLocation__bindgen_ty_1 {
    pub id: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaMemcpy3DOperand__bindgen_ty_1 {
    pub ptr: cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1,
    pub array: cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaResourceDesc__bindgen_ty_1 {
    pub array: cudaResourceDesc__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: cudaResourceDesc__bindgen_ty_1__bindgen_ty_2,
    pub linear: cudaResourceDesc__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: cudaResourceDesc__bindgen_ty_1__bindgen_ty_4,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaResourceDesc__bindgen_ty_1 {
    pub array: cudaResourceDesc__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: cudaResourceDesc__bindgen_ty_1__bindgen_ty_2,
    pub linear: cudaResourceDesc__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: cudaResourceDesc__bindgen_ty_1__bindgen_ty_4,
    pub reserved: cudaResourceDesc__bindgen_ty_1__bindgen_ty_5,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaStreamAttrValue {
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub syncPolicy: cudaSynchronizationPolicy,
}
pub unsafe fn cudaArrayGetInfo(desc: *mut cudaChannelFormatDesc, extent: *mut cudaExtent, flags: *mut ::core::ffi::c_uint, array: cudaArray_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaChannelFormatDesc, *mut cudaExtent, *mut ::core::ffi::c_uint, cudaArray_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaArrayGetInfo") });
        _f(desc, extent, flags, array)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaArrayGetInfo(desc: *mut cudaChannelFormatDesc, extent: *mut cudaExtent, flags: *mut ::core::ffi::c_uint, array: cudaArray_t) -> cudaError_t;
        }
        cudaArrayGetInfo(desc, extent, flags, array)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaArrayGetMemoryRequirements(memoryRequirements: *mut cudaArrayMemoryRequirements, array: cudaArray_t, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArrayMemoryRequirements, cudaArray_t, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaArrayGetMemoryRequirements") });
        _f(memoryRequirements, array, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaArrayGetMemoryRequirements(memoryRequirements: *mut cudaArrayMemoryRequirements, array: cudaArray_t, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaArrayGetMemoryRequirements(memoryRequirements, array, device)
    }
}
pub unsafe fn cudaArrayGetPlane(pPlaneArray: *mut cudaArray_t, hArray: cudaArray_t, planeIdx: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArray_t, cudaArray_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaArrayGetPlane") });
        _f(pPlaneArray, hArray, planeIdx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaArrayGetPlane(pPlaneArray: *mut cudaArray_t, hArray: cudaArray_t, planeIdx: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaArrayGetPlane(pPlaneArray, hArray, planeIdx)
    }
}
pub unsafe fn cudaArrayGetSparseProperties(sparseProperties: *mut cudaArraySparseProperties, array: cudaArray_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArraySparseProperties, cudaArray_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaArrayGetSparseProperties") });
        _f(sparseProperties, array)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaArrayGetSparseProperties(sparseProperties: *mut cudaArraySparseProperties, array: cudaArray_t) -> cudaError_t;
        }
        cudaArrayGetSparseProperties(sparseProperties, array)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaBindSurfaceToArray(surfref: *const surfaceReference, array: cudaArray_const_t, desc: *const cudaChannelFormatDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const surfaceReference, cudaArray_const_t, *const cudaChannelFormatDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaBindSurfaceToArray") });
        _f(surfref, array, desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaBindSurfaceToArray(surfref: *const surfaceReference, array: cudaArray_const_t, desc: *const cudaChannelFormatDesc) -> cudaError_t;
        }
        cudaBindSurfaceToArray(surfref, array, desc)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaBindTexture(offset: *mut usize, texref: *const textureReference, devPtr: *const ::core::ffi::c_void, desc: *const cudaChannelFormatDesc, size: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const textureReference, *const ::core::ffi::c_void, *const cudaChannelFormatDesc, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaBindTexture") });
        _f(offset, texref, devPtr, desc, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaBindTexture(offset: *mut usize, texref: *const textureReference, devPtr: *const ::core::ffi::c_void, desc: *const cudaChannelFormatDesc, size: usize) -> cudaError_t;
        }
        cudaBindTexture(offset, texref, devPtr, desc, size)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaBindTexture2D(offset: *mut usize, texref: *const textureReference, devPtr: *const ::core::ffi::c_void, desc: *const cudaChannelFormatDesc, width: usize, height: usize, pitch: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const textureReference, *const ::core::ffi::c_void, *const cudaChannelFormatDesc, usize, usize, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaBindTexture2D") });
        _f(offset, texref, devPtr, desc, width, height, pitch)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaBindTexture2D(offset: *mut usize, texref: *const textureReference, devPtr: *const ::core::ffi::c_void, desc: *const cudaChannelFormatDesc, width: usize, height: usize, pitch: usize) -> cudaError_t;
        }
        cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaBindTextureToArray(texref: *const textureReference, array: cudaArray_const_t, desc: *const cudaChannelFormatDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const textureReference, cudaArray_const_t, *const cudaChannelFormatDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaBindTextureToArray") });
        _f(texref, array, desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaBindTextureToArray(texref: *const textureReference, array: cudaArray_const_t, desc: *const cudaChannelFormatDesc) -> cudaError_t;
        }
        cudaBindTextureToArray(texref, array, desc)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaBindTextureToMipmappedArray(texref: *const textureReference, mipmappedArray: cudaMipmappedArray_const_t, desc: *const cudaChannelFormatDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const textureReference, cudaMipmappedArray_const_t, *const cudaChannelFormatDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaBindTextureToMipmappedArray") });
        _f(texref, mipmappedArray, desc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaBindTextureToMipmappedArray(texref: *const textureReference, mipmappedArray: cudaMipmappedArray_const_t, desc: *const cudaChannelFormatDesc) -> cudaError_t;
        }
        cudaBindTextureToMipmappedArray(texref, mipmappedArray, desc)
    }
}
pub unsafe fn cudaChooseDevice(device: *mut ::core::ffi::c_int, prop: *const cudaDeviceProp) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *const cudaDeviceProp) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaChooseDevice") });
        _f(device, prop)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaChooseDevice(device: *mut ::core::ffi::c_int, prop: *const cudaDeviceProp) -> cudaError_t;
        }
        cudaChooseDevice(device, prop)
    }
}
pub unsafe fn cudaCreateChannelDesc(x: ::core::ffi::c_int, y: ::core::ffi::c_int, z: ::core::ffi::c_int, w: ::core::ffi::c_int, f: cudaChannelFormatKind) -> cudaChannelFormatDesc {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int, cudaChannelFormatKind) -> cudaChannelFormatDesc;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaCreateChannelDesc") });
        _f(x, y, z, w, f)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaCreateChannelDesc(x: ::core::ffi::c_int, y: ::core::ffi::c_int, z: ::core::ffi::c_int, w: ::core::ffi::c_int, f: cudaChannelFormatKind) -> cudaChannelFormatDesc;
        }
        cudaCreateChannelDesc(x, y, z, w, f)
    }
}
pub unsafe fn cudaCreateSurfaceObject(pSurfObject: *mut cudaSurfaceObject_t, pResDesc: *const cudaResourceDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaSurfaceObject_t, *const cudaResourceDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaCreateSurfaceObject") });
        _f(pSurfObject, pResDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaCreateSurfaceObject(pSurfObject: *mut cudaSurfaceObject_t, pResDesc: *const cudaResourceDesc) -> cudaError_t;
        }
        cudaCreateSurfaceObject(pSurfObject, pResDesc)
    }
}
pub unsafe fn cudaCreateTextureObject(pTexObject: *mut cudaTextureObject_t, pResDesc: *const cudaResourceDesc, pTexDesc: *const cudaTextureDesc, pResViewDesc: *const cudaResourceViewDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaTextureObject_t, *const cudaResourceDesc, *const cudaTextureDesc, *const cudaResourceViewDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaCreateTextureObject") });
        _f(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaCreateTextureObject(pTexObject: *mut cudaTextureObject_t, pResDesc: *const cudaResourceDesc, pTexDesc: *const cudaTextureDesc, pResViewDesc: *const cudaResourceViewDesc) -> cudaError_t;
        }
        cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
}
#[cfg(any(feature = "cuda-11080"))]
pub unsafe fn cudaCreateTextureObject_v2(pTexObject: *mut cudaTextureObject_t, pResDesc: *const cudaResourceDesc, pTexDesc: *const cudaTextureDesc_v2, pResViewDesc: *const cudaResourceViewDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaTextureObject_t, *const cudaResourceDesc, *const cudaTextureDesc_v2, *const cudaResourceViewDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaCreateTextureObject_v2") });
        _f(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaCreateTextureObject_v2(pTexObject: *mut cudaTextureObject_t, pResDesc: *const cudaResourceDesc, pTexDesc: *const cudaTextureDesc_v2, pResViewDesc: *const cudaResourceViewDesc) -> cudaError_t;
        }
        cudaCreateTextureObject_v2(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
}
pub unsafe fn cudaCtxResetPersistingL2Cache() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaCtxResetPersistingL2Cache") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaCtxResetPersistingL2Cache() -> cudaError_t;
        }
        cudaCtxResetPersistingL2Cache()
    }
}
pub unsafe fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExternalMemory_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDestroyExternalMemory") });
        _f(extMem)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t;
        }
        cudaDestroyExternalMemory(extMem)
    }
}
pub unsafe fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExternalSemaphore_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDestroyExternalSemaphore") });
        _f(extSem)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t;
        }
        cudaDestroyExternalSemaphore(extSem)
    }
}
pub unsafe fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaSurfaceObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDestroySurfaceObject") });
        _f(surfObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t;
        }
        cudaDestroySurfaceObject(surfObject)
    }
}
pub unsafe fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaTextureObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDestroyTextureObject") });
        _f(texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t;
        }
        cudaDestroyTextureObject(texObject)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDevResourceGenerateDesc(phDesc: *mut cudaDevResourceDesc_t, resources: *mut cudaDevResource, nbResources: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaDevResourceDesc_t, *mut cudaDevResource, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDevResourceGenerateDesc") });
        _f(phDesc, resources, nbResources)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDevResourceGenerateDesc(phDesc: *mut cudaDevResourceDesc_t, resources: *mut cudaDevResource, nbResources: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaDevResourceGenerateDesc(phDesc, resources, nbResources)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDevSmResourceSplit(result: *mut cudaDevResource, nbGroups: ::core::ffi::c_uint, input: *const cudaDevResource, remainder: *mut cudaDevResource, flags: ::core::ffi::c_uint, groupParams: *mut cudaDevSmResourceGroupParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaDevResource, ::core::ffi::c_uint, *const cudaDevResource, *mut cudaDevResource, ::core::ffi::c_uint, *mut cudaDevSmResourceGroupParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDevSmResourceSplit") });
        _f(result, nbGroups, input, remainder, flags, groupParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDevSmResourceSplit(result: *mut cudaDevResource, nbGroups: ::core::ffi::c_uint, input: *const cudaDevResource, remainder: *mut cudaDevResource, flags: ::core::ffi::c_uint, groupParams: *mut cudaDevSmResourceGroupParams) -> cudaError_t;
        }
        cudaDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDevSmResourceSplitByCount(result: *mut cudaDevResource, nbGroups: *mut ::core::ffi::c_uint, input: *const cudaDevResource, remaining: *mut cudaDevResource, flags: ::core::ffi::c_uint, minCount: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaDevResource, *mut ::core::ffi::c_uint, *const cudaDevResource, *mut cudaDevResource, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDevSmResourceSplitByCount") });
        _f(result, nbGroups, input, remaining, flags, minCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDevSmResourceSplitByCount(result: *mut cudaDevResource, nbGroups: *mut ::core::ffi::c_uint, input: *const cudaDevResource, remaining: *mut cudaDevResource, flags: ::core::ffi::c_uint, minCount: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaDevSmResourceSplitByCount(result, nbGroups, input, remaining, flags, minCount)
    }
}
pub unsafe fn cudaDeviceCanAccessPeer(canAccessPeer: *mut ::core::ffi::c_int, device: ::core::ffi::c_int, peerDevice: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceCanAccessPeer") });
        _f(canAccessPeer, device, peerDevice)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceCanAccessPeer(canAccessPeer: *mut ::core::ffi::c_int, device: ::core::ffi::c_int, peerDevice: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice)
    }
}
pub unsafe fn cudaDeviceDisablePeerAccess(peerDevice: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceDisablePeerAccess") });
        _f(peerDevice)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceDisablePeerAccess(peerDevice: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceDisablePeerAccess(peerDevice)
    }
}
pub unsafe fn cudaDeviceEnablePeerAccess(peerDevice: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceEnablePeerAccess") });
        _f(peerDevice, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceEnablePeerAccess(peerDevice: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaDeviceEnablePeerAccess(peerDevice, flags)
    }
}
pub unsafe fn cudaDeviceFlushGPUDirectRDMAWrites(target: cudaFlushGPUDirectRDMAWritesTarget, scope: cudaFlushGPUDirectRDMAWritesScope) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaFlushGPUDirectRDMAWritesTarget, cudaFlushGPUDirectRDMAWritesScope) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceFlushGPUDirectRDMAWrites") });
        _f(target, scope)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceFlushGPUDirectRDMAWrites(target: cudaFlushGPUDirectRDMAWritesTarget, scope: cudaFlushGPUDirectRDMAWritesScope) -> cudaError_t;
        }
        cudaDeviceFlushGPUDirectRDMAWrites(target, scope)
    }
}
pub unsafe fn cudaDeviceGetAttribute(value: *mut ::core::ffi::c_int, attr: cudaDeviceAttr, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, cudaDeviceAttr, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetAttribute") });
        _f(value, attr, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetAttribute(value: *mut ::core::ffi::c_int, attr: cudaDeviceAttr, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetAttribute(value, attr, device)
    }
}
pub unsafe fn cudaDeviceGetByPCIBusId(device: *mut ::core::ffi::c_int, pciBusId: *const ::core::ffi::c_char) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *const ::core::ffi::c_char) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetByPCIBusId") });
        _f(device, pciBusId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetByPCIBusId(device: *mut ::core::ffi::c_int, pciBusId: *const ::core::ffi::c_char) -> cudaError_t;
        }
        cudaDeviceGetByPCIBusId(device, pciBusId)
    }
}
pub unsafe fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaFuncCache) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetCacheConfig") });
        _f(pCacheConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
        }
        cudaDeviceGetCacheConfig(pCacheConfig)
    }
}
pub unsafe fn cudaDeviceGetDefaultMemPool(memPool: *mut cudaMemPool_t, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPool_t, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetDefaultMemPool") });
        _f(memPool, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetDefaultMemPool(memPool: *mut cudaMemPool_t, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetDefaultMemPool(memPool, device)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDeviceGetDevResource(device: ::core::ffi::c_int, resource: *mut cudaDevResource, type_: cudaDevResourceType) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, *mut cudaDevResource, cudaDevResourceType) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetDevResource") });
        _f(device, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetDevResource(device: ::core::ffi::c_int, resource: *mut cudaDevResource, type_: cudaDevResourceType) -> cudaError_t;
        }
        cudaDeviceGetDevResource(device, resource, type_)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDeviceGetExecutionCtx(ctx: *mut cudaExecutionContext_t, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaExecutionContext_t, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetExecutionCtx") });
        _f(ctx, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetExecutionCtx(ctx: *mut cudaExecutionContext_t, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetExecutionCtx(ctx, device)
    }
}
pub unsafe fn cudaDeviceGetGraphMemAttribute(device: ::core::ffi::c_int, attr: cudaGraphMemAttributeType, value: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, cudaGraphMemAttributeType, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetGraphMemAttribute") });
        _f(device, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetGraphMemAttribute(device: ::core::ffi::c_int, attr: cudaGraphMemAttributeType, value: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaDeviceGetGraphMemAttribute(device, attr, value)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDeviceGetHostAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const cudaAtomicOperation, count: ::core::ffi::c_uint, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, *const cudaAtomicOperation, ::core::ffi::c_uint, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetHostAtomicCapabilities") });
        _f(capabilities, operations, count, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetHostAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const cudaAtomicOperation, count: ::core::ffi::c_uint, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetHostAtomicCapabilities(capabilities, operations, count, device)
    }
}
pub unsafe fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, cudaLimit) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetLimit") });
        _f(pValue, limit)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
        }
        cudaDeviceGetLimit(pValue, limit)
    }
}
pub unsafe fn cudaDeviceGetMemPool(memPool: *mut cudaMemPool_t, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPool_t, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetMemPool") });
        _f(memPool, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetMemPool(memPool: *mut cudaMemPool_t, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetMemPool(memPool, device)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDeviceGetP2PAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const cudaAtomicOperation, count: ::core::ffi::c_uint, srcDevice: ::core::ffi::c_int, dstDevice: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, *const cudaAtomicOperation, ::core::ffi::c_uint, ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetP2PAtomicCapabilities") });
        _f(capabilities, operations, count, srcDevice, dstDevice)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetP2PAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const cudaAtomicOperation, count: ::core::ffi::c_uint, srcDevice: ::core::ffi::c_int, dstDevice: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)
    }
}
pub unsafe fn cudaDeviceGetP2PAttribute(value: *mut ::core::ffi::c_int, attr: cudaDeviceP2PAttr, srcDevice: ::core::ffi::c_int, dstDevice: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, cudaDeviceP2PAttr, ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetP2PAttribute") });
        _f(value, attr, srcDevice, dstDevice)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetP2PAttribute(value: *mut ::core::ffi::c_int, attr: cudaDeviceP2PAttr, srcDevice: ::core::ffi::c_int, dstDevice: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetP2PAttribute(value, attr, srcDevice, dstDevice)
    }
}
pub unsafe fn cudaDeviceGetPCIBusId(pciBusId: *mut ::core::ffi::c_char, len: ::core::ffi::c_int, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_char, ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetPCIBusId") });
        _f(pciBusId, len, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetPCIBusId(pciBusId: *mut ::core::ffi::c_char, len: ::core::ffi::c_int, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetPCIBusId(pciBusId, len, device)
    }
}
pub unsafe fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaSharedMemConfig) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetSharedMemConfig") });
        _f(pConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;
        }
        cudaDeviceGetSharedMemConfig(pConfig)
    }
}
pub unsafe fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut ::core::ffi::c_int, greatestPriority: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetStreamPriorityRange") });
        _f(leastPriority, greatestPriority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetStreamPriorityRange(leastPriority: *mut ::core::ffi::c_int, greatestPriority: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority)
    }
}
pub unsafe fn cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: *mut usize, fmtDesc: *const cudaChannelFormatDesc, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const cudaChannelFormatDesc, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGetTexture1DLinearMaxWidth") });
        _f(maxWidthInElements, fmtDesc, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: *mut usize, fmtDesc: *const cudaChannelFormatDesc, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, fmtDesc, device)
    }
}
pub unsafe fn cudaDeviceGraphMemTrim(device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceGraphMemTrim") });
        _f(device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceGraphMemTrim(device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDeviceGraphMemTrim(device)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDeviceRegisterAsyncNotification(device: ::core::ffi::c_int, callbackFunc: cudaAsyncCallback, userData: *mut ::core::ffi::c_void, callback: *mut cudaAsyncCallbackHandle_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, cudaAsyncCallback, *mut ::core::ffi::c_void, *mut cudaAsyncCallbackHandle_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceRegisterAsyncNotification") });
        _f(device, callbackFunc, userData, callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceRegisterAsyncNotification(device: ::core::ffi::c_int, callbackFunc: cudaAsyncCallback, userData: *mut ::core::ffi::c_void, callback: *mut cudaAsyncCallbackHandle_t) -> cudaError_t;
        }
        cudaDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)
    }
}
pub unsafe fn cudaDeviceReset() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceReset") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceReset() -> cudaError_t;
        }
        cudaDeviceReset()
    }
}
pub unsafe fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaFuncCache) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceSetCacheConfig") });
        _f(cacheConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
        }
        cudaDeviceSetCacheConfig(cacheConfig)
    }
}
pub unsafe fn cudaDeviceSetGraphMemAttribute(device: ::core::ffi::c_int, attr: cudaGraphMemAttributeType, value: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, cudaGraphMemAttributeType, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceSetGraphMemAttribute") });
        _f(device, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceSetGraphMemAttribute(device: ::core::ffi::c_int, attr: cudaGraphMemAttributeType, value: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaDeviceSetGraphMemAttribute(device, attr, value)
    }
}
pub unsafe fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLimit, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceSetLimit") });
        _f(limit, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;
        }
        cudaDeviceSetLimit(limit, value)
    }
}
pub unsafe fn cudaDeviceSetMemPool(device: ::core::ffi::c_int, memPool: cudaMemPool_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, cudaMemPool_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceSetMemPool") });
        _f(device, memPool)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceSetMemPool(device: ::core::ffi::c_int, memPool: cudaMemPool_t) -> cudaError_t;
        }
        cudaDeviceSetMemPool(device, memPool)
    }
}
pub unsafe fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaSharedMemConfig) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceSetSharedMemConfig") });
        _f(config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;
        }
        cudaDeviceSetSharedMemConfig(config)
    }
}
pub unsafe fn cudaDeviceSynchronize() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceSynchronize") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceSynchronize() -> cudaError_t;
        }
        cudaDeviceSynchronize()
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaDeviceUnregisterAsyncNotification(device: ::core::ffi::c_int, callback: cudaAsyncCallbackHandle_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, cudaAsyncCallbackHandle_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDeviceUnregisterAsyncNotification") });
        _f(device, callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDeviceUnregisterAsyncNotification(device: ::core::ffi::c_int, callback: cudaAsyncCallbackHandle_t) -> cudaError_t;
        }
        cudaDeviceUnregisterAsyncNotification(device, callback)
    }
}
pub unsafe fn cudaDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaDriverGetVersion") });
        _f(driverVersion)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaDriverGetVersion(driverVersion)
    }
}
pub unsafe fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventCreate") });
        _f(event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
        }
        cudaEventCreate(event)
    }
}
pub unsafe fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaEvent_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventCreateWithFlags") });
        _f(event, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaEventCreateWithFlags(event, flags)
    }
}
pub unsafe fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventDestroy") });
        _f(event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
        }
        cudaEventDestroy(event)
    }
}
pub unsafe fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, cudaEvent_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventElapsedTime") });
        _f(ms, start, end)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
        }
        cudaEventElapsedTime(ms, start, end)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaEventElapsedTime_v2(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, cudaEvent_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventElapsedTime_v2") });
        _f(ms, start, end)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventElapsedTime_v2(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
        }
        cudaEventElapsedTime_v2(ms, start, end)
    }
}
pub unsafe fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventQuery") });
        _f(event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
        }
        cudaEventQuery(event)
    }
}
pub unsafe fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaEvent_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventRecord") });
        _f(event, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
        }
        cudaEventRecord(event, stream)
    }
}
pub unsafe fn cudaEventRecordWithFlags(event: cudaEvent_t, stream: cudaStream_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaEvent_t, cudaStream_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventRecordWithFlags") });
        _f(event, stream, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventRecordWithFlags(event: cudaEvent_t, stream: cudaStream_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaEventRecordWithFlags(event, stream, flags)
    }
}
pub unsafe fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaEventSynchronize") });
        _f(event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
        }
        cudaEventSynchronize(event)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxDestroy(ctx: cudaExecutionContext_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExecutionContext_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxDestroy") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxDestroy(ctx: cudaExecutionContext_t) -> cudaError_t;
        }
        cudaExecutionCtxDestroy(ctx)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxGetDevResource(ctx: cudaExecutionContext_t, resource: *mut cudaDevResource, type_: cudaDevResourceType) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExecutionContext_t, *mut cudaDevResource, cudaDevResourceType) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxGetDevResource") });
        _f(ctx, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxGetDevResource(ctx: cudaExecutionContext_t, resource: *mut cudaDevResource, type_: cudaDevResourceType) -> cudaError_t;
        }
        cudaExecutionCtxGetDevResource(ctx, resource, type_)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxGetDevice(device: *mut ::core::ffi::c_int, ctx: cudaExecutionContext_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, cudaExecutionContext_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxGetDevice") });
        _f(device, ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxGetDevice(device: *mut ::core::ffi::c_int, ctx: cudaExecutionContext_t) -> cudaError_t;
        }
        cudaExecutionCtxGetDevice(device, ctx)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxGetId(ctx: cudaExecutionContext_t, ctxId: *mut ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExecutionContext_t, *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxGetId") });
        _f(ctx, ctxId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxGetId(ctx: cudaExecutionContext_t, ctxId: *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaExecutionCtxGetId(ctx, ctxId)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxRecordEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExecutionContext_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxRecordEvent") });
        _f(ctx, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxRecordEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaExecutionCtxRecordEvent(ctx, event)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxStreamCreate(phStream: *mut cudaStream_t, ctx: cudaExecutionContext_t, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaStream_t, cudaExecutionContext_t, ::core::ffi::c_uint, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxStreamCreate") });
        _f(phStream, ctx, flags, priority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxStreamCreate(phStream: *mut cudaStream_t, ctx: cudaExecutionContext_t, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaExecutionCtxStreamCreate(phStream, ctx, flags, priority)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxSynchronize(ctx: cudaExecutionContext_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExecutionContext_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxSynchronize") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxSynchronize(ctx: cudaExecutionContext_t) -> cudaError_t;
        }
        cudaExecutionCtxSynchronize(ctx)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaExecutionCtxWaitEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaExecutionContext_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExecutionCtxWaitEvent") });
        _f(ctx, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExecutionCtxWaitEvent(ctx: cudaExecutionContext_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaExecutionCtxWaitEvent(ctx, event)
    }
}
pub unsafe fn cudaExternalMemoryGetMappedBuffer(devPtr: *mut *mut ::core::ffi::c_void, extMem: cudaExternalMemory_t, bufferDesc: *const cudaExternalMemoryBufferDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, cudaExternalMemory_t, *const cudaExternalMemoryBufferDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExternalMemoryGetMappedBuffer") });
        _f(devPtr, extMem, bufferDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExternalMemoryGetMappedBuffer(devPtr: *mut *mut ::core::ffi::c_void, extMem: cudaExternalMemory_t, bufferDesc: *const cudaExternalMemoryBufferDesc) -> cudaError_t;
        }
        cudaExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
    }
}
pub unsafe fn cudaExternalMemoryGetMappedMipmappedArray(mipmap: *mut cudaMipmappedArray_t, extMem: cudaExternalMemory_t, mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMipmappedArray_t, cudaExternalMemory_t, *const cudaExternalMemoryMipmappedArrayDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaExternalMemoryGetMappedMipmappedArray") });
        _f(mipmap, extMem, mipmapDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaExternalMemoryGetMappedMipmappedArray(mipmap: *mut cudaMipmappedArray_t, extMem: cudaExternalMemory_t, mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc) -> cudaError_t;
        }
        cudaExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
    }
}
pub unsafe fn cudaFree(devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFree") });
        _f(devPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFree(devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaFree(devPtr)
    }
}
pub unsafe fn cudaFreeArray(array: cudaArray_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFreeArray") });
        _f(array)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFreeArray(array: cudaArray_t) -> cudaError_t;
        }
        cudaFreeArray(array)
    }
}
pub unsafe fn cudaFreeAsync(devPtr: *mut ::core::ffi::c_void, hStream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFreeAsync") });
        _f(devPtr, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFreeAsync(devPtr: *mut ::core::ffi::c_void, hStream: cudaStream_t) -> cudaError_t;
        }
        cudaFreeAsync(devPtr, hStream)
    }
}
pub unsafe fn cudaFreeHost(ptr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFreeHost") });
        _f(ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFreeHost(ptr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaFreeHost(ptr)
    }
}
pub unsafe fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaMipmappedArray_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFreeMipmappedArray") });
        _f(mipmappedArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;
        }
        cudaFreeMipmappedArray(mipmappedArray)
    }
}
pub unsafe fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaFuncAttributes, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncGetAttributes") });
        _f(attr, func)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaFuncGetAttributes(attr, func)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaFuncGetName(name: *mut *const ::core::ffi::c_char, func: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const ::core::ffi::c_char, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncGetName") });
        _f(name, func)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncGetName(name: *mut *const ::core::ffi::c_char, func: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaFuncGetName(name, func)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaFuncGetParamCount(func: *const ::core::ffi::c_void, paramCount: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncGetParamCount") });
        _f(func, paramCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncGetParamCount(func: *const ::core::ffi::c_void, paramCount: *mut usize) -> cudaError_t;
        }
        cudaFuncGetParamCount(func, paramCount)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaFuncGetParamInfo(func: *const ::core::ffi::c_void, paramIndex: usize, paramOffset: *mut usize, paramSize: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, *mut usize, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncGetParamInfo") });
        _f(func, paramIndex, paramOffset, paramSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncGetParamInfo(func: *const ::core::ffi::c_void, paramIndex: usize, paramOffset: *mut usize, paramSize: *mut usize) -> cudaError_t;
        }
        cudaFuncGetParamInfo(func, paramIndex, paramOffset, paramSize)
    }
}
pub unsafe fn cudaFuncSetAttribute(func: *const ::core::ffi::c_void, attr: cudaFuncAttribute, value: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, cudaFuncAttribute, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncSetAttribute") });
        _f(func, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncSetAttribute(func: *const ::core::ffi::c_void, attr: cudaFuncAttribute, value: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaFuncSetAttribute(func, attr, value)
    }
}
pub unsafe fn cudaFuncSetCacheConfig(func: *const ::core::ffi::c_void, cacheConfig: cudaFuncCache) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, cudaFuncCache) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncSetCacheConfig") });
        _f(func, cacheConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncSetCacheConfig(func: *const ::core::ffi::c_void, cacheConfig: cudaFuncCache) -> cudaError_t;
        }
        cudaFuncSetCacheConfig(func, cacheConfig)
    }
}
pub unsafe fn cudaFuncSetSharedMemConfig(func: *const ::core::ffi::c_void, config: cudaSharedMemConfig) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, cudaSharedMemConfig) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaFuncSetSharedMemConfig") });
        _f(func, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaFuncSetSharedMemConfig(func: *const ::core::ffi::c_void, config: cudaSharedMemConfig) -> cudaError_t;
        }
        cudaFuncSetSharedMemConfig(func, config)
    }
}
pub unsafe fn cudaGetChannelDesc(desc: *mut cudaChannelFormatDesc, array: cudaArray_const_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaChannelFormatDesc, cudaArray_const_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetChannelDesc") });
        _f(desc, array)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetChannelDesc(desc: *mut cudaChannelFormatDesc, array: cudaArray_const_t) -> cudaError_t;
        }
        cudaGetChannelDesc(desc, array)
    }
}
pub unsafe fn cudaGetDevice(device: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDevice") });
        _f(device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDevice(device: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaGetDevice(device)
    }
}
pub unsafe fn cudaGetDeviceCount(count: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDeviceCount") });
        _f(count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDeviceCount(count: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaGetDeviceCount(count)
    }
}
pub unsafe fn cudaGetDeviceFlags(flags: *mut ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDeviceFlags") });
        _f(flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDeviceFlags(flags: *mut ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGetDeviceFlags(flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaDeviceProp, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDeviceProperties") });
        _f(prop, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaGetDeviceProperties(prop, device)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaDeviceProp, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDeviceProperties_v2") });
        _f(prop, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDeviceProperties_v2(prop: *mut cudaDeviceProp, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaGetDeviceProperties_v2(prop, device)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaGetDriverEntryPoint(symbol: *const ::core::ffi::c_char, funcPtr: *mut *mut ::core::ffi::c_void, flags: ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *mut ::core::ffi::c_void, ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDriverEntryPoint") });
        _f(symbol, funcPtr, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDriverEntryPoint(symbol: *const ::core::ffi::c_char, funcPtr: *mut *mut ::core::ffi::c_void, flags: ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaGetDriverEntryPoint(symbol, funcPtr, flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGetDriverEntryPoint(symbol: *const ::core::ffi::c_char, funcPtr: *mut *mut ::core::ffi::c_void, flags: ::core::ffi::c_ulonglong, driverStatus: *mut cudaDriverEntryPointQueryResult) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *mut ::core::ffi::c_void, ::core::ffi::c_ulonglong, *mut cudaDriverEntryPointQueryResult) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDriverEntryPoint") });
        _f(symbol, funcPtr, flags, driverStatus)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDriverEntryPoint(symbol: *const ::core::ffi::c_char, funcPtr: *mut *mut ::core::ffi::c_void, flags: ::core::ffi::c_ulonglong, driverStatus: *mut cudaDriverEntryPointQueryResult) -> cudaError_t;
        }
        cudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGetDriverEntryPointByVersion(symbol: *const ::core::ffi::c_char, funcPtr: *mut *mut ::core::ffi::c_void, cudaVersion: ::core::ffi::c_uint, flags: ::core::ffi::c_ulonglong, driverStatus: *mut cudaDriverEntryPointQueryResult) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint, ::core::ffi::c_ulonglong, *mut cudaDriverEntryPointQueryResult) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetDriverEntryPointByVersion") });
        _f(symbol, funcPtr, cudaVersion, flags, driverStatus)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetDriverEntryPointByVersion(symbol: *const ::core::ffi::c_char, funcPtr: *mut *mut ::core::ffi::c_void, cudaVersion: ::core::ffi::c_uint, flags: ::core::ffi::c_ulonglong, driverStatus: *mut cudaDriverEntryPointQueryResult) -> cudaError_t;
        }
        cudaGetDriverEntryPointByVersion(symbol, funcPtr, cudaVersion, flags, driverStatus)
    }
}
pub unsafe fn cudaGetErrorName(error: cudaError_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaError_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetErrorName") });
        _f(error)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetErrorName(error: cudaError_t) -> *const ::core::ffi::c_char;
        }
        cudaGetErrorName(error)
    }
}
pub unsafe fn cudaGetErrorString(error: cudaError_t) -> *const ::core::ffi::c_char {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaError_t) -> *const ::core::ffi::c_char;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetErrorString") });
        _f(error)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetErrorString(error: cudaError_t) -> *const ::core::ffi::c_char;
        }
        cudaGetErrorString(error)
    }
}
pub unsafe fn cudaGetExportTable(ppExportTable: *mut *const ::core::ffi::c_void, pExportTableId: *const cudaUUID_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const ::core::ffi::c_void, *const cudaUUID_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetExportTable") });
        _f(ppExportTable, pExportTableId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetExportTable(ppExportTable: *mut *const ::core::ffi::c_void, pExportTableId: *const cudaUUID_t) -> cudaError_t;
        }
        cudaGetExportTable(ppExportTable, pExportTableId)
    }
}
pub unsafe fn cudaGetFuncBySymbol(functionPtr: *mut cudaFunction_t, symbolPtr: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaFunction_t, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetFuncBySymbol") });
        _f(functionPtr, symbolPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetFuncBySymbol(functionPtr: *mut cudaFunction_t, symbolPtr: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGetFuncBySymbol(functionPtr, symbolPtr)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGetKernel(kernelPtr: *mut cudaKernel_t, entryFuncAddr: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaKernel_t, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetKernel") });
        _f(kernelPtr, entryFuncAddr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetKernel(kernelPtr: *mut cudaKernel_t, entryFuncAddr: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGetKernel(kernelPtr, entryFuncAddr)
    }
}
pub unsafe fn cudaGetLastError() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetLastError") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetLastError() -> cudaError_t;
        }
        cudaGetLastError()
    }
}
pub unsafe fn cudaGetMipmappedArrayLevel(levelArray: *mut cudaArray_t, mipmappedArray: cudaMipmappedArray_const_t, level: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArray_t, cudaMipmappedArray_const_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetMipmappedArrayLevel") });
        _f(levelArray, mipmappedArray, level)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetMipmappedArrayLevel(levelArray: *mut cudaArray_t, mipmappedArray: cudaMipmappedArray_const_t, level: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGetMipmappedArrayLevel(levelArray, mipmappedArray, level)
    }
}
pub unsafe fn cudaGetSurfaceObjectResourceDesc(pResDesc: *mut cudaResourceDesc, surfObject: cudaSurfaceObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaResourceDesc, cudaSurfaceObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetSurfaceObjectResourceDesc") });
        _f(pResDesc, surfObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetSurfaceObjectResourceDesc(pResDesc: *mut cudaResourceDesc, surfObject: cudaSurfaceObject_t) -> cudaError_t;
        }
        cudaGetSurfaceObjectResourceDesc(pResDesc, surfObject)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaGetSurfaceReference(surfref: *mut *const surfaceReference, symbol: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const surfaceReference, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetSurfaceReference") });
        _f(surfref, symbol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetSurfaceReference(surfref: *mut *const surfaceReference, symbol: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGetSurfaceReference(surfref, symbol)
    }
}
pub unsafe fn cudaGetSymbolAddress(devPtr: *mut *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetSymbolAddress") });
        _f(devPtr, symbol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetSymbolAddress(devPtr: *mut *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGetSymbolAddress(devPtr, symbol)
    }
}
pub unsafe fn cudaGetSymbolSize(size: *mut usize, symbol: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetSymbolSize") });
        _f(size, symbol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetSymbolSize(size: *mut usize, symbol: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGetSymbolSize(size, symbol)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaGetTextureAlignmentOffset(offset: *mut usize, texref: *const textureReference) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const textureReference) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetTextureAlignmentOffset") });
        _f(offset, texref)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetTextureAlignmentOffset(offset: *mut usize, texref: *const textureReference) -> cudaError_t;
        }
        cudaGetTextureAlignmentOffset(offset, texref)
    }
}
pub unsafe fn cudaGetTextureObjectResourceDesc(pResDesc: *mut cudaResourceDesc, texObject: cudaTextureObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaResourceDesc, cudaTextureObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetTextureObjectResourceDesc") });
        _f(pResDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetTextureObjectResourceDesc(pResDesc: *mut cudaResourceDesc, texObject: cudaTextureObject_t) -> cudaError_t;
        }
        cudaGetTextureObjectResourceDesc(pResDesc, texObject)
    }
}
pub unsafe fn cudaGetTextureObjectResourceViewDesc(pResViewDesc: *mut cudaResourceViewDesc, texObject: cudaTextureObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaResourceViewDesc, cudaTextureObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetTextureObjectResourceViewDesc") });
        _f(pResViewDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetTextureObjectResourceViewDesc(pResViewDesc: *mut cudaResourceViewDesc, texObject: cudaTextureObject_t) -> cudaError_t;
        }
        cudaGetTextureObjectResourceViewDesc(pResViewDesc, texObject)
    }
}
pub unsafe fn cudaGetTextureObjectTextureDesc(pTexDesc: *mut cudaTextureDesc, texObject: cudaTextureObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaTextureDesc, cudaTextureObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetTextureObjectTextureDesc") });
        _f(pTexDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetTextureObjectTextureDesc(pTexDesc: *mut cudaTextureDesc, texObject: cudaTextureObject_t) -> cudaError_t;
        }
        cudaGetTextureObjectTextureDesc(pTexDesc, texObject)
    }
}
#[cfg(any(feature = "cuda-11080"))]
pub unsafe fn cudaGetTextureObjectTextureDesc_v2(pTexDesc: *mut cudaTextureDesc_v2, texObject: cudaTextureObject_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaTextureDesc_v2, cudaTextureObject_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetTextureObjectTextureDesc_v2") });
        _f(pTexDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetTextureObjectTextureDesc_v2(pTexDesc: *mut cudaTextureDesc_v2, texObject: cudaTextureObject_t) -> cudaError_t;
        }
        cudaGetTextureObjectTextureDesc_v2(pTexDesc, texObject)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaGetTextureReference(texref: *mut *const textureReference, symbol: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const textureReference, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGetTextureReference") });
        _f(texref, symbol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGetTextureReference(texref: *mut *const textureReference, symbol: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGetTextureReference(texref, symbol)
    }
}
pub unsafe fn cudaGraphAddChildGraphNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, childGraph: cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddChildGraphNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, childGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddChildGraphNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, childGraph: cudaGraph_t) -> cudaError_t;
        }
        cudaGraphAddChildGraphNode(pGraphNode, graph, pDependencies, numDependencies, childGraph)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphAddDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphNode_t, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddDependencies") });
        _f(graph, from, to, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphAddDependencies(graph, from, to, numDependencies)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphAddDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddDependencies") });
        _f(graph, from, to, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphAddDependencies(graph, from, to, edgeData, numDependencies)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphAddDependencies_v2(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddDependencies_v2") });
        _f(graph, from, to, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddDependencies_v2(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphAddDependencies_v2(graph, from, to, edgeData, numDependencies)
    }
}
pub unsafe fn cudaGraphAddEmptyNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddEmptyNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddEmptyNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphAddEmptyNode(pGraphNode, graph, pDependencies, numDependencies)
    }
}
pub unsafe fn cudaGraphAddEventRecordNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddEventRecordNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddEventRecordNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) -> cudaError_t;
        }
        cudaGraphAddEventRecordNode(pGraphNode, graph, pDependencies, numDependencies, event)
    }
}
pub unsafe fn cudaGraphAddEventWaitNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddEventWaitNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddEventWaitNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, event: cudaEvent_t) -> cudaError_t;
        }
        cudaGraphAddEventWaitNode(pGraphNode, graph, pDependencies, numDependencies, event)
    }
}
pub unsafe fn cudaGraphAddExternalSemaphoresSignalNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddExternalSemaphoresSignalNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddExternalSemaphoresSignalNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        }
        cudaGraphAddExternalSemaphoresSignalNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cudaGraphAddExternalSemaphoresWaitNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddExternalSemaphoresWaitNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddExternalSemaphoresWaitNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        }
        cudaGraphAddExternalSemaphoresWaitNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cudaGraphAddHostNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pNodeParams: *const cudaHostNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const cudaHostNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddHostNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddHostNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pNodeParams: *const cudaHostNodeParams) -> cudaError_t;
        }
        cudaGraphAddHostNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    }
}
pub unsafe fn cudaGraphAddKernelNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const cudaKernelNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddKernelNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddKernelNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t;
        }
        cudaGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams)
    }
}
pub unsafe fn cudaGraphAddMemAllocNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *mut cudaMemAllocNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *mut cudaMemAllocNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemAllocNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemAllocNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *mut cudaMemAllocNodeParams) -> cudaError_t;
        }
        cudaGraphAddMemAllocNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cudaGraphAddMemFreeNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dptr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemFreeNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, dptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemFreeNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dptr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGraphAddMemFreeNode(pGraphNode, graph, pDependencies, numDependencies, dptr)
    }
}
pub unsafe fn cudaGraphAddMemcpyNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pCopyParams: *const cudaMemcpy3DParms) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const cudaMemcpy3DParms) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemcpyNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemcpyNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pCopyParams: *const cudaMemcpy3DParms) -> cudaError_t;
        }
        cudaGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams)
    }
}
pub unsafe fn cudaGraphAddMemcpyNode1D(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemcpyNode1D") });
        _f(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemcpyNode1D(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphAddMemcpyNode1D(pGraphNode, graph, pDependencies, numDependencies, dst, src, count, kind)
    }
}
pub unsafe fn cudaGraphAddMemcpyNodeFromSymbol(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemcpyNodeFromSymbol") });
        _f(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemcpyNodeFromSymbol(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphAddMemcpyNodeFromSymbol(pGraphNode, graph, pDependencies, numDependencies, dst, symbol, count, offset, kind)
    }
}
pub unsafe fn cudaGraphAddMemcpyNodeToSymbol(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemcpyNodeToSymbol") });
        _f(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemcpyNodeToSymbol(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphAddMemcpyNodeToSymbol(pGraphNode, graph, pDependencies, numDependencies, symbol, src, count, offset, kind)
    }
}
pub unsafe fn cudaGraphAddMemsetNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pMemsetParams: *const cudaMemsetParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *const cudaMemsetParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddMemsetNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddMemsetNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, pMemsetParams: *const cudaMemsetParams) -> cudaError_t;
        }
        cudaGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphAddNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, usize, *mut cudaGraphNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddNode") });
        _f(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t;
        }
        cudaGraphAddNode(pGraphNode, graph, pDependencies, numDependencies, nodeParams)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphAddNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize, *mut cudaGraphNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddNode") });
        _f(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddNode(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t;
        }
        cudaGraphAddNode(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphAddNode_v2(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize, *mut cudaGraphNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphAddNode_v2") });
        _f(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphAddNode_v2(pGraphNode: *mut cudaGraphNode_t, graph: cudaGraph_t, pDependencies: *const cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t;
        }
        cudaGraphAddNode_v2(pGraphNode, graph, pDependencies, dependencyData, numDependencies, nodeParams)
    }
}
pub unsafe fn cudaGraphChildGraphNodeGetGraph(node: cudaGraphNode_t, pGraph: *mut cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphChildGraphNodeGetGraph") });
        _f(node, pGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphChildGraphNodeGetGraph(node: cudaGraphNode_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
        }
        cudaGraphChildGraphNodeGetGraph(node, pGraph)
    }
}
pub unsafe fn cudaGraphClone(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraph_t, cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphClone") });
        _f(pGraphClone, originalGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphClone(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t) -> cudaError_t;
        }
        cudaGraphClone(pGraphClone, originalGraph)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphConditionalHandleCreate(pHandle_out: *mut cudaGraphConditionalHandle, graph: cudaGraph_t, defaultLaunchValue: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphConditionalHandle, cudaGraph_t, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphConditionalHandleCreate") });
        _f(pHandle_out, graph, defaultLaunchValue, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphConditionalHandleCreate(pHandle_out: *mut cudaGraphConditionalHandle, graph: cudaGraph_t, defaultLaunchValue: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphConditionalHandleCreate(pHandle_out, graph, defaultLaunchValue, flags)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphConditionalHandleCreate_v2(pHandle_out: *mut cudaGraphConditionalHandle, graph: cudaGraph_t, ctx: cudaExecutionContext_t, defaultLaunchValue: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphConditionalHandle, cudaGraph_t, cudaExecutionContext_t, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphConditionalHandleCreate_v2") });
        _f(pHandle_out, graph, ctx, defaultLaunchValue, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphConditionalHandleCreate_v2(pHandle_out: *mut cudaGraphConditionalHandle, graph: cudaGraph_t, ctx: cudaExecutionContext_t, defaultLaunchValue: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphConditionalHandleCreate_v2(pHandle_out, graph, ctx, defaultLaunchValue, flags)
    }
}
pub unsafe fn cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraph_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphCreate") });
        _f(pGraph, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphCreate(pGraph, flags)
    }
}
pub unsafe fn cudaGraphDebugDotPrint(graph: cudaGraph_t, path: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const ::core::ffi::c_char, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphDebugDotPrint") });
        _f(graph, path, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphDebugDotPrint(graph: cudaGraph_t, path: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphDebugDotPrint(graph, path, flags)
    }
}
pub unsafe fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphDestroy") });
        _f(graph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
        }
        cudaGraphDestroy(graph)
    }
}
pub unsafe fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphDestroyNode") });
        _f(node)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t;
        }
        cudaGraphDestroyNode(node)
    }
}
pub unsafe fn cudaGraphEventRecordNodeGetEvent(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphEventRecordNodeGetEvent") });
        _f(node, event_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphEventRecordNodeGetEvent(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t;
        }
        cudaGraphEventRecordNodeGetEvent(node, event_out)
    }
}
pub unsafe fn cudaGraphEventRecordNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphEventRecordNodeSetEvent") });
        _f(node, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphEventRecordNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaGraphEventRecordNodeSetEvent(node, event)
    }
}
pub unsafe fn cudaGraphEventWaitNodeGetEvent(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphEventWaitNodeGetEvent") });
        _f(node, event_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphEventWaitNodeGetEvent(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t;
        }
        cudaGraphEventWaitNodeGetEvent(node, event_out)
    }
}
pub unsafe fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphEventWaitNodeSetEvent") });
        _f(node, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaGraphEventWaitNodeSetEvent(node, event)
    }
}
pub unsafe fn cudaGraphExecChildGraphNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, childGraph: cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecChildGraphNodeSetParams") });
        _f(hGraphExec, node, childGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecChildGraphNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, childGraph: cudaGraph_t) -> cudaError_t;
        }
        cudaGraphExecChildGraphNodeSetParams(hGraphExec, node, childGraph)
    }
}
pub unsafe fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecDestroy") });
        _f(graphExec)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;
        }
        cudaGraphExecDestroy(graphExec)
    }
}
pub unsafe fn cudaGraphExecEventRecordNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecEventRecordNodeSetEvent") });
        _f(hGraphExec, hNode, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecEventRecordNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)
    }
}
pub unsafe fn cudaGraphExecEventWaitNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecEventWaitNodeSetEvent") });
        _f(hGraphExec, hNode, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecEventWaitNodeSetEvent(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)
    }
}
pub unsafe fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecExternalSemaphoresSignalNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        }
        cudaGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
pub unsafe fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecExternalSemaphoresWaitNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        }
        cudaGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphExecGetFlags(graphExec: cudaGraphExec_t, flags: *mut ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecGetFlags") });
        _f(graphExec, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecGetFlags(graphExec: cudaGraphExec_t, flags: *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaGraphExecGetFlags(graphExec, flags)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphExecGetId(hGraphExec: cudaGraphExec_t, graphID: *mut ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, *mut ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecGetId") });
        _f(hGraphExec, graphID)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecGetId(hGraphExec: cudaGraphExec_t, graphID: *mut ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphExecGetId(hGraphExec, graphID)
    }
}
pub unsafe fn cudaGraphExecHostNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaHostNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const cudaHostNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecHostNodeSetParams") });
        _f(hGraphExec, node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecHostNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaHostNodeParams) -> cudaError_t;
        }
        cudaGraphExecHostNodeSetParams(hGraphExec, node, pNodeParams)
    }
}
pub unsafe fn cudaGraphExecKernelNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const cudaKernelNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecKernelNodeSetParams") });
        _f(hGraphExec, node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecKernelNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t;
        }
        cudaGraphExecKernelNodeSetParams(hGraphExec, node, pNodeParams)
    }
}
pub unsafe fn cudaGraphExecMemcpyNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaMemcpy3DParms) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const cudaMemcpy3DParms) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecMemcpyNodeSetParams") });
        _f(hGraphExec, node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecMemcpyNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaMemcpy3DParms) -> cudaError_t;
        }
        cudaGraphExecMemcpyNodeSetParams(hGraphExec, node, pNodeParams)
    }
}
pub unsafe fn cudaGraphExecMemcpyNodeSetParams1D(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecMemcpyNodeSetParams1D") });
        _f(hGraphExec, node, dst, src, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecMemcpyNodeSetParams1D(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphExecMemcpyNodeSetParams1D(hGraphExec, node, dst, src, count, kind)
    }
}
pub unsafe fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecMemcpyNodeSetParamsFromSymbol") });
        _f(hGraphExec, node, dst, symbol, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphExecMemcpyNodeSetParamsFromSymbol(hGraphExec, node, dst, symbol, count, offset, kind)
    }
}
pub unsafe fn cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecMemcpyNodeSetParamsToSymbol") });
        _f(hGraphExec, node, symbol, src, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphExecMemcpyNodeSetParamsToSymbol(hGraphExec, node, symbol, src, count, offset, kind)
    }
}
pub unsafe fn cudaGraphExecMemsetNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaMemsetParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *const cudaMemsetParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecMemsetNodeSetParams") });
        _f(hGraphExec, node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecMemsetNodeSetParams(hGraphExec: cudaGraphExec_t, node: cudaGraphNode_t, pNodeParams: *const cudaMemsetParams) -> cudaError_t;
        }
        cudaGraphExecMemsetNodeSetParams(hGraphExec, node, pNodeParams)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphExecNodeSetParams(graphExec: cudaGraphExec_t, node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *mut cudaGraphNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecNodeSetParams") });
        _f(graphExec, node, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecNodeSetParams(graphExec: cudaGraphExec_t, node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t;
        }
        cudaGraphExecNodeSetParams(graphExec, node, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, hErrorNode_out: *mut cudaGraphNode_t, updateResult_out: *mut cudaGraphExecUpdateResult) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraph_t, *mut cudaGraphNode_t, *mut cudaGraphExecUpdateResult) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecUpdate") });
        _f(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, hErrorNode_out: *mut cudaGraphNode_t, updateResult_out: *mut cudaGraphExecUpdateResult) -> cudaError_t;
        }
        cudaGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, resultInfo: *mut cudaGraphExecUpdateResultInfo) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraph_t, *mut cudaGraphExecUpdateResultInfo) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExecUpdate") });
        _f(hGraphExec, hGraph, resultInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExecUpdate(hGraphExec: cudaGraphExec_t, hGraph: cudaGraph_t, resultInfo: *mut cudaGraphExecUpdateResultInfo) -> cudaError_t;
        }
        cudaGraphExecUpdate(hGraphExec, hGraph, resultInfo)
    }
}
pub unsafe fn cudaGraphExternalSemaphoresSignalNodeGetParams(hNode: cudaGraphNode_t, params_out: *mut cudaExternalSemaphoreSignalNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExternalSemaphoresSignalNodeGetParams") });
        _f(hNode, params_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExternalSemaphoresSignalNodeGetParams(hNode: cudaGraphNode_t, params_out: *mut cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        }
        cudaGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)
    }
}
pub unsafe fn cudaGraphExternalSemaphoresSignalNodeSetParams(hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExternalSemaphoresSignalNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExternalSemaphoresSignalNodeSetParams(hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreSignalNodeParams) -> cudaError_t;
        }
        cudaGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cudaGraphExternalSemaphoresWaitNodeGetParams(hNode: cudaGraphNode_t, params_out: *mut cudaExternalSemaphoreWaitNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExternalSemaphoresWaitNodeGetParams") });
        _f(hNode, params_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExternalSemaphoresWaitNodeGetParams(hNode: cudaGraphNode_t, params_out: *mut cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        }
        cudaGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)
    }
}
pub unsafe fn cudaGraphExternalSemaphoresWaitNodeSetParams(hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphExternalSemaphoresWaitNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphExternalSemaphoresWaitNodeSetParams(hNode: cudaGraphNode_t, nodeParams: *const cudaExternalSemaphoreWaitNodeParams) -> cudaError_t;
        }
        cudaGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphGetEdges(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, numEdges: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *mut cudaGraphNode_t, *mut cudaGraphNode_t, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphGetEdges") });
        _f(graph, from, to, numEdges)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphGetEdges(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, numEdges: *mut usize) -> cudaError_t;
        }
        cudaGraphGetEdges(graph, from, to, numEdges)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphGetEdges(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, numEdges: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *mut cudaGraphNode_t, *mut cudaGraphNode_t, *mut cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphGetEdges") });
        _f(graph, from, to, edgeData, numEdges)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphGetEdges(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, numEdges: *mut usize) -> cudaError_t;
        }
        cudaGraphGetEdges(graph, from, to, edgeData, numEdges)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphGetEdges_v2(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, numEdges: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *mut cudaGraphNode_t, *mut cudaGraphNode_t, *mut cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphGetEdges_v2") });
        _f(graph, from, to, edgeData, numEdges)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphGetEdges_v2(graph: cudaGraph_t, from: *mut cudaGraphNode_t, to: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, numEdges: *mut usize) -> cudaError_t;
        }
        cudaGraphGetEdges_v2(graph, from, to, edgeData, numEdges)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphGetId(hGraph: cudaGraph_t, graphID: *mut ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *mut ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphGetId") });
        _f(hGraph, graphID)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphGetId(hGraph: cudaGraph_t, graphID: *mut ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphGetId(hGraph, graphID)
    }
}
pub unsafe fn cudaGraphGetNodes(graph: cudaGraph_t, nodes: *mut cudaGraphNode_t, numNodes: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *mut cudaGraphNode_t, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphGetNodes") });
        _f(graph, nodes, numNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphGetNodes(graph: cudaGraph_t, nodes: *mut cudaGraphNode_t, numNodes: *mut usize) -> cudaError_t;
        }
        cudaGraphGetNodes(graph, nodes, numNodes)
    }
}
pub unsafe fn cudaGraphGetRootNodes(graph: cudaGraph_t, pRootNodes: *mut cudaGraphNode_t, pNumRootNodes: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *mut cudaGraphNode_t, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphGetRootNodes") });
        _f(graph, pRootNodes, pNumRootNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphGetRootNodes(graph: cudaGraph_t, pRootNodes: *mut cudaGraphNode_t, pNumRootNodes: *mut usize) -> cudaError_t;
        }
        cudaGraphGetRootNodes(graph, pRootNodes, pNumRootNodes)
    }
}
pub unsafe fn cudaGraphHostNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaHostNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaHostNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphHostNodeGetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphHostNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaHostNodeParams) -> cudaError_t;
        }
        cudaGraphHostNodeGetParams(node, pNodeParams)
    }
}
pub unsafe fn cudaGraphHostNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaHostNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const cudaHostNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphHostNodeSetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphHostNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaHostNodeParams) -> cudaError_t;
        }
        cudaGraphHostNodeSetParams(node, pNodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaGraphInstantiate(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, pErrorNode: *mut cudaGraphNode_t, pLogBuffer: *mut ::core::ffi::c_char, bufferSize: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphExec_t, cudaGraph_t, *mut cudaGraphNode_t, *mut ::core::ffi::c_char, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphInstantiate") });
        _f(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphInstantiate(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, pErrorNode: *mut cudaGraphNode_t, pLogBuffer: *mut ::core::ffi::c_char, bufferSize: usize) -> cudaError_t;
        }
        cudaGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphInstantiate(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphExec_t, cudaGraph_t, ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphInstantiate") });
        _f(pGraphExec, graph, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphInstantiate(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaGraphInstantiate(pGraphExec, graph, flags)
    }
}
pub unsafe fn cudaGraphInstantiateWithFlags(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphExec_t, cudaGraph_t, ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphInstantiateWithFlags") });
        _f(pGraphExec, graph, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphInstantiateWithFlags(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaGraphInstantiateWithFlags(pGraphExec, graph, flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphInstantiateWithParams(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, instantiateParams: *mut cudaGraphInstantiateParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphExec_t, cudaGraph_t, *mut cudaGraphInstantiateParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphInstantiateWithParams") });
        _f(pGraphExec, graph, instantiateParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphInstantiateWithParams(pGraphExec: *mut cudaGraphExec_t, graph: cudaGraph_t, instantiateParams: *mut cudaGraphInstantiateParams) -> cudaError_t;
        }
        cudaGraphInstantiateWithParams(pGraphExec, graph, instantiateParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphKernelNodeCopyAttributes(hSrc: cudaGraphNode_t, hDst: cudaGraphNode_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaGraphNode_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeCopyAttributes") });
        _f(hSrc, hDst)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeCopyAttributes(hSrc: cudaGraphNode_t, hDst: cudaGraphNode_t) -> cudaError_t;
        }
        cudaGraphKernelNodeCopyAttributes(hSrc, hDst)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphKernelNodeCopyAttributes(hDst: cudaGraphNode_t, hSrc: cudaGraphNode_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaGraphNode_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeCopyAttributes") });
        _f(hDst, hSrc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeCopyAttributes(hDst: cudaGraphNode_t, hSrc: cudaGraphNode_t) -> cudaError_t;
        }
        cudaGraphKernelNodeCopyAttributes(hDst, hSrc)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub unsafe fn cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaKernelNodeAttrID, value_out: *mut cudaKernelNodeAttrValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaKernelNodeAttrID, *mut cudaKernelNodeAttrValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeGetAttribute") });
        _f(hNode, attr, value_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaKernelNodeAttrID, value_out: *mut cudaKernelNodeAttrValue) -> cudaError_t;
        }
        cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value_out: *mut cudaLaunchAttributeValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaLaunchAttributeID, *mut cudaLaunchAttributeValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeGetAttribute") });
        _f(hNode, attr, value_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeGetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value_out: *mut cudaLaunchAttributeValue) -> cudaError_t;
        }
        cudaGraphKernelNodeGetAttribute(hNode, attr, value_out)
    }
}
pub unsafe fn cudaGraphKernelNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaKernelNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaKernelNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeGetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaKernelNodeParams) -> cudaError_t;
        }
        cudaGraphKernelNodeGetParams(node, pNodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub unsafe fn cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaKernelNodeAttrID, value: *const cudaKernelNodeAttrValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaKernelNodeAttrID, *const cudaKernelNodeAttrValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeSetAttribute") });
        _f(hNode, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaKernelNodeAttrID, value: *const cudaKernelNodeAttrValue) -> cudaError_t;
        }
        cudaGraphKernelNodeSetAttribute(hNode, attr, value)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value: *const cudaLaunchAttributeValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, cudaLaunchAttributeID, *const cudaLaunchAttributeValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeSetAttribute") });
        _f(hNode, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeSetAttribute(hNode: cudaGraphNode_t, attr: cudaLaunchAttributeID, value: *const cudaLaunchAttributeValue) -> cudaError_t;
        }
        cudaGraphKernelNodeSetAttribute(hNode, attr, value)
    }
}
pub unsafe fn cudaGraphKernelNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const cudaKernelNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphKernelNodeSetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphKernelNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaKernelNodeParams) -> cudaError_t;
        }
        cudaGraphKernelNodeSetParams(node, pNodeParams)
    }
}
pub unsafe fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphLaunch") });
        _f(graphExec, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
        }
        cudaGraphLaunch(graphExec, stream)
    }
}
pub unsafe fn cudaGraphMemAllocNodeGetParams(node: cudaGraphNode_t, params_out: *mut cudaMemAllocNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaMemAllocNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemAllocNodeGetParams") });
        _f(node, params_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemAllocNodeGetParams(node: cudaGraphNode_t, params_out: *mut cudaMemAllocNodeParams) -> cudaError_t;
        }
        cudaGraphMemAllocNodeGetParams(node, params_out)
    }
}
pub unsafe fn cudaGraphMemFreeNodeGetParams(node: cudaGraphNode_t, dptr_out: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemFreeNodeGetParams") });
        _f(node, dptr_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemFreeNodeGetParams(node: cudaGraphNode_t, dptr_out: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaGraphMemFreeNodeGetParams(node, dptr_out)
    }
}
pub unsafe fn cudaGraphMemcpyNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaMemcpy3DParms) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaMemcpy3DParms) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemcpyNodeGetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemcpyNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaMemcpy3DParms) -> cudaError_t;
        }
        cudaGraphMemcpyNodeGetParams(node, pNodeParams)
    }
}
pub unsafe fn cudaGraphMemcpyNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaMemcpy3DParms) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const cudaMemcpy3DParms) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemcpyNodeSetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemcpyNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaMemcpy3DParms) -> cudaError_t;
        }
        cudaGraphMemcpyNodeSetParams(node, pNodeParams)
    }
}
pub unsafe fn cudaGraphMemcpyNodeSetParams1D(node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemcpyNodeSetParams1D") });
        _f(node, dst, src, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemcpyNodeSetParams1D(node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphMemcpyNodeSetParams1D(node, dst, src, count, kind)
    }
}
pub unsafe fn cudaGraphMemcpyNodeSetParamsFromSymbol(node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemcpyNodeSetParamsFromSymbol") });
        _f(node, dst, symbol, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemcpyNodeSetParamsFromSymbol(node: cudaGraphNode_t, dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphMemcpyNodeSetParamsFromSymbol(node, dst, symbol, count, offset, kind)
    }
}
pub unsafe fn cudaGraphMemcpyNodeSetParamsToSymbol(node: cudaGraphNode_t, symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemcpyNodeSetParamsToSymbol") });
        _f(node, symbol, src, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemcpyNodeSetParamsToSymbol(node: cudaGraphNode_t, symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaGraphMemcpyNodeSetParamsToSymbol(node, symbol, src, count, offset, kind)
    }
}
pub unsafe fn cudaGraphMemsetNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaMemsetParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaMemsetParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemsetNodeGetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemsetNodeGetParams(node: cudaGraphNode_t, pNodeParams: *mut cudaMemsetParams) -> cudaError_t;
        }
        cudaGraphMemsetNodeGetParams(node, pNodeParams)
    }
}
pub unsafe fn cudaGraphMemsetNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaMemsetParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *const cudaMemsetParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphMemsetNodeSetParams") });
        _f(node, pNodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphMemsetNodeSetParams(node: cudaGraphNode_t, pNodeParams: *const cudaMemsetParams) -> cudaError_t;
        }
        cudaGraphMemsetNodeSetParams(node, pNodeParams)
    }
}
pub unsafe fn cudaGraphNodeFindInClone(pNode: *mut cudaGraphNode_t, originalNode: cudaGraphNode_t, clonedGraph: cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaGraphNode_t, cudaGraphNode_t, cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeFindInClone") });
        _f(pNode, originalNode, clonedGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeFindInClone(pNode: *mut cudaGraphNode_t, originalNode: cudaGraphNode_t, clonedGraph: cudaGraph_t) -> cudaError_t;
        }
        cudaGraphNodeFindInClone(pNode, originalNode, clonedGraph)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetContainingGraph(hNode: cudaGraphNode_t, phGraph: *mut cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetContainingGraph") });
        _f(hNode, phGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetContainingGraph(hNode: cudaGraphNode_t, phGraph: *mut cudaGraph_t) -> cudaError_t;
        }
        cudaGraphNodeGetContainingGraph(hNode, phGraph)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, pNumDependencies: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNode_t, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetDependencies") });
        _f(node, pDependencies, pNumDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, pNumDependencies: *mut usize) -> cudaError_t;
        }
        cudaGraphNodeGetDependencies(node, pDependencies, pNumDependencies)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependencies: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNode_t, *mut cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetDependencies") });
        _f(node, pDependencies, edgeData, pNumDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetDependencies(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependencies: *mut usize) -> cudaError_t;
        }
        cudaGraphNodeGetDependencies(node, pDependencies, edgeData, pNumDependencies)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphNodeGetDependencies_v2(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependencies: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNode_t, *mut cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetDependencies_v2") });
        _f(node, pDependencies, edgeData, pNumDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetDependencies_v2(node: cudaGraphNode_t, pDependencies: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependencies: *mut usize) -> cudaError_t;
        }
        cudaGraphNodeGetDependencies_v2(node, pDependencies, edgeData, pNumDependencies)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, pNumDependentNodes: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNode_t, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetDependentNodes") });
        _f(node, pDependentNodes, pNumDependentNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, pNumDependentNodes: *mut usize) -> cudaError_t;
        }
        cudaGraphNodeGetDependentNodes(node, pDependentNodes, pNumDependentNodes)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependentNodes: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNode_t, *mut cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetDependentNodes") });
        _f(node, pDependentNodes, edgeData, pNumDependentNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetDependentNodes(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependentNodes: *mut usize) -> cudaError_t;
        }
        cudaGraphNodeGetDependentNodes(node, pDependentNodes, edgeData, pNumDependentNodes)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphNodeGetDependentNodes_v2(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependentNodes: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNode_t, *mut cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetDependentNodes_v2") });
        _f(node, pDependentNodes, edgeData, pNumDependentNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetDependentNodes_v2(node: cudaGraphNode_t, pDependentNodes: *mut cudaGraphNode_t, edgeData: *mut cudaGraphEdgeData, pNumDependentNodes: *mut usize) -> cudaError_t;
        }
        cudaGraphNodeGetDependentNodes_v2(node, pDependentNodes, edgeData, pNumDependentNodes)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: *mut ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, *mut ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetEnabled") });
        _f(hGraphExec, hNode, isEnabled)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: *mut ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetLocalId(hNode: cudaGraphNode_t, nodeId: *mut ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetLocalId") });
        _f(hNode, nodeId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetLocalId(hNode: cudaGraphNode_t, nodeId: *mut ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphNodeGetLocalId(hNode, nodeId)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetParams(node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetParams") });
        _f(node, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetParams(node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t;
        }
        cudaGraphNodeGetParams(node, nodeParams)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeGetToolsId(hNode: cudaGraphNode_t, toolsNodeId: *mut ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetToolsId") });
        _f(hNode, toolsNodeId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetToolsId(hNode: cudaGraphNode_t, toolsNodeId: *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaGraphNodeGetToolsId(hNode, toolsNodeId)
    }
}
pub unsafe fn cudaGraphNodeGetType(node: cudaGraphNode_t, pType: *mut cudaGraphNodeType) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNodeType) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeGetType") });
        _f(node, pType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeGetType(node: cudaGraphNode_t, pType: *mut cudaGraphNodeType) -> cudaError_t;
        }
        cudaGraphNodeGetType(node, pType)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeSetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaGraphNode_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeSetEnabled") });
        _f(hGraphExec, hNode, isEnabled)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeSetEnabled(hGraphExec: cudaGraphExec_t, hNode: cudaGraphNode_t, isEnabled: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphNodeSetParams(node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphNode_t, *mut cudaGraphNodeParams) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphNodeSetParams") });
        _f(node, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphNodeSetParams(node: cudaGraphNode_t, nodeParams: *mut cudaGraphNodeParams) -> cudaError_t;
        }
        cudaGraphNodeSetParams(node, nodeParams)
    }
}
pub unsafe fn cudaGraphReleaseUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, cudaUserObject_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphReleaseUserObject") });
        _f(graph, object, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphReleaseUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphReleaseUserObject(graph, object, count)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphRemoveDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphNode_t, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphRemoveDependencies") });
        _f(graph, from, to, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphRemoveDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphRemoveDependencies(graph, from, to, numDependencies)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGraphRemoveDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphRemoveDependencies") });
        _f(graph, from, to, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphRemoveDependencies(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphRemoveDependencies(graph, from, to, edgeData, numDependencies)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaGraphRemoveDependencies_v2(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphRemoveDependencies_v2") });
        _f(graph, from, to, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphRemoveDependencies_v2(graph: cudaGraph_t, from: *const cudaGraphNode_t, to: *const cudaGraphNode_t, edgeData: *const cudaGraphEdgeData, numDependencies: usize) -> cudaError_t;
        }
        cudaGraphRemoveDependencies_v2(graph, from, to, edgeData, numDependencies)
    }
}
pub unsafe fn cudaGraphRetainUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraph_t, cudaUserObject_t, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphRetainUserObject") });
        _f(graph, object, count, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphRetainUserObject(graph: cudaGraph_t, object: cudaUserObject_t, count: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphRetainUserObject(graph, object, count, flags)
    }
}
pub unsafe fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphExec_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphUpload") });
        _f(graphExec, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
        }
        cudaGraphUpload(graphExec, stream)
    }
}
pub unsafe fn cudaGraphicsMapResources(count: ::core::ffi::c_int, resources: *mut cudaGraphicsResource_t, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, *mut cudaGraphicsResource_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsMapResources") });
        _f(count, resources, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsMapResources(count: ::core::ffi::c_int, resources: *mut cudaGraphicsResource_t, stream: cudaStream_t) -> cudaError_t;
        }
        cudaGraphicsMapResources(count, resources, stream)
    }
}
pub unsafe fn cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, resource: cudaGraphicsResource_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMipmappedArray_t, cudaGraphicsResource_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsResourceGetMappedMipmappedArray") });
        _f(mipmappedArray, resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, resource: cudaGraphicsResource_t) -> cudaError_t;
        }
        cudaGraphicsResourceGetMappedMipmappedArray(mipmappedArray, resource)
    }
}
pub unsafe fn cudaGraphicsResourceGetMappedPointer(devPtr: *mut *mut ::core::ffi::c_void, size: *mut usize, resource: cudaGraphicsResource_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, cudaGraphicsResource_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsResourceGetMappedPointer") });
        _f(devPtr, size, resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsResourceGetMappedPointer(devPtr: *mut *mut ::core::ffi::c_void, size: *mut usize, resource: cudaGraphicsResource_t) -> cudaError_t;
        }
        cudaGraphicsResourceGetMappedPointer(devPtr, size, resource)
    }
}
pub unsafe fn cudaGraphicsResourceSetMapFlags(resource: cudaGraphicsResource_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphicsResource_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsResourceSetMapFlags") });
        _f(resource, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsResourceSetMapFlags(resource: cudaGraphicsResource_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphicsResourceSetMapFlags(resource, flags)
    }
}
pub unsafe fn cudaGraphicsSubResourceGetMappedArray(array: *mut cudaArray_t, resource: cudaGraphicsResource_t, arrayIndex: ::core::ffi::c_uint, mipLevel: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArray_t, cudaGraphicsResource_t, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsSubResourceGetMappedArray") });
        _f(array, resource, arrayIndex, mipLevel)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsSubResourceGetMappedArray(array: *mut cudaArray_t, resource: cudaGraphicsResource_t, arrayIndex: ::core::ffi::c_uint, mipLevel: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGraphicsSubResourceGetMappedArray(array, resource, arrayIndex, mipLevel)
    }
}
pub unsafe fn cudaGraphicsUnmapResources(count: ::core::ffi::c_int, resources: *mut cudaGraphicsResource_t, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, *mut cudaGraphicsResource_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsUnmapResources") });
        _f(count, resources, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsUnmapResources(count: ::core::ffi::c_int, resources: *mut cudaGraphicsResource_t, stream: cudaStream_t) -> cudaError_t;
        }
        cudaGraphicsUnmapResources(count, resources, stream)
    }
}
pub unsafe fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaGraphicsResource_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGraphicsUnregisterResource") });
        _f(resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t;
        }
        cudaGraphicsUnregisterResource(resource)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaGreenCtxCreate(phCtx: *mut cudaExecutionContext_t, desc: cudaDevResourceDesc_t, device: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaExecutionContext_t, cudaDevResourceDesc_t, ::core::ffi::c_int, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaGreenCtxCreate") });
        _f(phCtx, desc, device, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaGreenCtxCreate(phCtx: *mut cudaExecutionContext_t, desc: cudaDevResourceDesc_t, device: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaGreenCtxCreate(phCtx, desc, device, flags)
    }
}
pub unsafe fn cudaHostAlloc(pHost: *mut *mut ::core::ffi::c_void, size: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaHostAlloc") });
        _f(pHost, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaHostAlloc(pHost: *mut *mut ::core::ffi::c_void, size: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaHostAlloc(pHost, size, flags)
    }
}
pub unsafe fn cudaHostGetDevicePointer(pDevice: *mut *mut ::core::ffi::c_void, pHost: *mut ::core::ffi::c_void, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaHostGetDevicePointer") });
        _f(pDevice, pHost, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaHostGetDevicePointer(pDevice: *mut *mut ::core::ffi::c_void, pHost: *mut ::core::ffi::c_void, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaHostGetDevicePointer(pDevice, pHost, flags)
    }
}
pub unsafe fn cudaHostGetFlags(pFlags: *mut ::core::ffi::c_uint, pHost: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaHostGetFlags") });
        _f(pFlags, pHost)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaHostGetFlags(pFlags: *mut ::core::ffi::c_uint, pHost: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaHostGetFlags(pFlags, pHost)
    }
}
pub unsafe fn cudaHostRegister(ptr: *mut ::core::ffi::c_void, size: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaHostRegister") });
        _f(ptr, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaHostRegister(ptr: *mut ::core::ffi::c_void, size: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaHostRegister(ptr, size, flags)
    }
}
pub unsafe fn cudaHostUnregister(ptr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaHostUnregister") });
        _f(ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaHostUnregister(ptr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaHostUnregister(ptr)
    }
}
pub unsafe fn cudaImportExternalMemory(extMem_out: *mut cudaExternalMemory_t, memHandleDesc: *const cudaExternalMemoryHandleDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaExternalMemory_t, *const cudaExternalMemoryHandleDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaImportExternalMemory") });
        _f(extMem_out, memHandleDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaImportExternalMemory(extMem_out: *mut cudaExternalMemory_t, memHandleDesc: *const cudaExternalMemoryHandleDesc) -> cudaError_t;
        }
        cudaImportExternalMemory(extMem_out, memHandleDesc)
    }
}
pub unsafe fn cudaImportExternalSemaphore(extSem_out: *mut cudaExternalSemaphore_t, semHandleDesc: *const cudaExternalSemaphoreHandleDesc) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaExternalSemaphore_t, *const cudaExternalSemaphoreHandleDesc) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaImportExternalSemaphore") });
        _f(extSem_out, semHandleDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaImportExternalSemaphore(extSem_out: *mut cudaExternalSemaphore_t, semHandleDesc: *const cudaExternalSemaphoreHandleDesc) -> cudaError_t;
        }
        cudaImportExternalSemaphore(extSem_out, semHandleDesc)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaInitDevice(device: ::core::ffi::c_int, deviceFlags: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaInitDevice") });
        _f(device, deviceFlags, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaInitDevice(device: ::core::ffi::c_int, deviceFlags: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaInitDevice(device, deviceFlags, flags)
    }
}
pub unsafe fn cudaIpcCloseMemHandle(devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaIpcCloseMemHandle") });
        _f(devPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaIpcCloseMemHandle(devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaIpcCloseMemHandle(devPtr)
    }
}
pub unsafe fn cudaIpcGetEventHandle(handle: *mut cudaIpcEventHandle_t, event: cudaEvent_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaIpcEventHandle_t, cudaEvent_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaIpcGetEventHandle") });
        _f(handle, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaIpcGetEventHandle(handle: *mut cudaIpcEventHandle_t, event: cudaEvent_t) -> cudaError_t;
        }
        cudaIpcGetEventHandle(handle, event)
    }
}
pub unsafe fn cudaIpcGetMemHandle(handle: *mut cudaIpcMemHandle_t, devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaIpcMemHandle_t, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaIpcGetMemHandle") });
        _f(handle, devPtr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaIpcGetMemHandle(handle: *mut cudaIpcMemHandle_t, devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaIpcGetMemHandle(handle, devPtr)
    }
}
pub unsafe fn cudaIpcOpenEventHandle(event: *mut cudaEvent_t, handle: cudaIpcEventHandle_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaEvent_t, cudaIpcEventHandle_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaIpcOpenEventHandle") });
        _f(event, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaIpcOpenEventHandle(event: *mut cudaEvent_t, handle: cudaIpcEventHandle_t) -> cudaError_t;
        }
        cudaIpcOpenEventHandle(event, handle)
    }
}
pub unsafe fn cudaIpcOpenMemHandle(devPtr: *mut *mut ::core::ffi::c_void, handle: cudaIpcMemHandle_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, cudaIpcMemHandle_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaIpcOpenMemHandle") });
        _f(devPtr, handle, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaIpcOpenMemHandle(devPtr: *mut *mut ::core::ffi::c_void, handle: cudaIpcMemHandle_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaIpcOpenMemHandle(devPtr, handle, flags)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaKernelSetAttributeForDevice(kernel: cudaKernel_t, attr: cudaFuncAttribute, value: ::core::ffi::c_int, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaKernel_t, cudaFuncAttribute, ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaKernelSetAttributeForDevice") });
        _f(kernel, attr, value, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaKernelSetAttributeForDevice(kernel: cudaKernel_t, attr: cudaFuncAttribute, value: ::core::ffi::c_int, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaKernelSetAttributeForDevice(kernel, attr, value, device)
    }
}
pub unsafe fn cudaLaunchCooperativeKernel(func: *const ::core::ffi::c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut ::core::ffi::c_void, sharedMem: usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, dim3, dim3, *mut *mut ::core::ffi::c_void, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLaunchCooperativeKernel") });
        _f(func, gridDim, blockDim, args, sharedMem, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLaunchCooperativeKernel(func: *const ::core::ffi::c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut ::core::ffi::c_void, sharedMem: usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem, stream)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaLaunchCooperativeKernelMultiDevice(launchParamsList: *mut cudaLaunchParams, numDevices: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLaunchParams, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLaunchCooperativeKernelMultiDevice") });
        _f(launchParamsList, numDevices, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLaunchCooperativeKernelMultiDevice(launchParamsList: *mut cudaLaunchParams, numDevices: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
    }
}
pub unsafe fn cudaLaunchHostFunc(stream: cudaStream_t, fn_: cudaHostFn_t, userData: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaHostFn_t, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLaunchHostFunc") });
        _f(stream, fn_, userData)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLaunchHostFunc(stream: cudaStream_t, fn_: cudaHostFn_t, userData: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaLaunchHostFunc(stream, fn_, userData)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLaunchHostFunc_v2(stream: cudaStream_t, fn_: cudaHostFn_t, userData: *mut ::core::ffi::c_void, syncMode: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaHostFn_t, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLaunchHostFunc_v2") });
        _f(stream, fn_, userData, syncMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLaunchHostFunc_v2(stream: cudaStream_t, fn_: cudaHostFn_t, userData: *mut ::core::ffi::c_void, syncMode: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLaunchHostFunc_v2(stream, fn_, userData, syncMode)
    }
}
pub unsafe fn cudaLaunchKernel(func: *const ::core::ffi::c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut ::core::ffi::c_void, sharedMem: usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, dim3, dim3, *mut *mut ::core::ffi::c_void, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLaunchKernel") });
        _f(func, gridDim, blockDim, args, sharedMem, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLaunchKernel(func: *const ::core::ffi::c_void, gridDim: dim3, blockDim: dim3, args: *mut *mut ::core::ffi::c_void, sharedMem: usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLaunchKernelExC(config: *const cudaLaunchConfig_t, func: *const ::core::ffi::c_void, args: *mut *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaLaunchConfig_t, *const ::core::ffi::c_void, *mut *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLaunchKernelExC") });
        _f(config, func, args)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLaunchKernelExC(config: *const cudaLaunchConfig_t, func: *const ::core::ffi::c_void, args: *mut *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaLaunchKernelExC(config, func, args)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryEnumerateKernels(kernels: *mut cudaKernel_t, numKernels: ::core::ffi::c_uint, lib: cudaLibrary_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaKernel_t, ::core::ffi::c_uint, cudaLibrary_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryEnumerateKernels") });
        _f(kernels, numKernels, lib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryEnumerateKernels(kernels: *mut cudaKernel_t, numKernels: ::core::ffi::c_uint, lib: cudaLibrary_t) -> cudaError_t;
        }
        cudaLibraryEnumerateKernels(kernels, numKernels, lib)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryGetGlobal(dptr: *mut *mut ::core::ffi::c_void, bytes: *mut usize, library: cudaLibrary_t, name: *const ::core::ffi::c_char) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, cudaLibrary_t, *const ::core::ffi::c_char) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryGetGlobal") });
        _f(dptr, bytes, library, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryGetGlobal(dptr: *mut *mut ::core::ffi::c_void, bytes: *mut usize, library: cudaLibrary_t, name: *const ::core::ffi::c_char) -> cudaError_t;
        }
        cudaLibraryGetGlobal(dptr, bytes, library, name)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryGetKernel(pKernel: *mut cudaKernel_t, library: cudaLibrary_t, name: *const ::core::ffi::c_char) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaKernel_t, cudaLibrary_t, *const ::core::ffi::c_char) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryGetKernel") });
        _f(pKernel, library, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryGetKernel(pKernel: *mut cudaKernel_t, library: cudaLibrary_t, name: *const ::core::ffi::c_char) -> cudaError_t;
        }
        cudaLibraryGetKernel(pKernel, library, name)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryGetKernelCount(count: *mut ::core::ffi::c_uint, lib: cudaLibrary_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, cudaLibrary_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryGetKernelCount") });
        _f(count, lib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryGetKernelCount(count: *mut ::core::ffi::c_uint, lib: cudaLibrary_t) -> cudaError_t;
        }
        cudaLibraryGetKernelCount(count, lib)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryGetManaged(dptr: *mut *mut ::core::ffi::c_void, bytes: *mut usize, library: cudaLibrary_t, name: *const ::core::ffi::c_char) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, cudaLibrary_t, *const ::core::ffi::c_char) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryGetManaged") });
        _f(dptr, bytes, library, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryGetManaged(dptr: *mut *mut ::core::ffi::c_void, bytes: *mut usize, library: cudaLibrary_t, name: *const ::core::ffi::c_char) -> cudaError_t;
        }
        cudaLibraryGetManaged(dptr, bytes, library, name)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryGetUnifiedFunction(fptr: *mut *mut ::core::ffi::c_void, library: cudaLibrary_t, symbol: *const ::core::ffi::c_char) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, cudaLibrary_t, *const ::core::ffi::c_char) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryGetUnifiedFunction") });
        _f(fptr, library, symbol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryGetUnifiedFunction(fptr: *mut *mut ::core::ffi::c_void, library: cudaLibrary_t, symbol: *const ::core::ffi::c_char) -> cudaError_t;
        }
        cudaLibraryGetUnifiedFunction(fptr, library, symbol)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryLoadData(library: *mut cudaLibrary_t, code: *const ::core::ffi::c_void, jitOptions: *mut cudaJitOption, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut cudaLibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLibrary_t, *const ::core::ffi::c_void, *mut cudaJitOption, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint, *mut cudaLibraryOption, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryLoadData") });
        _f(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryLoadData(library: *mut cudaLibrary_t, code: *const ::core::ffi::c_void, jitOptions: *mut cudaJitOption, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut cudaLibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryLoadFromFile(library: *mut cudaLibrary_t, fileName: *const ::core::ffi::c_char, jitOptions: *mut cudaJitOption, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut cudaLibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLibrary_t, *const ::core::ffi::c_char, *mut cudaJitOption, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint, *mut cudaLibraryOption, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryLoadFromFile") });
        _f(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryLoadFromFile(library: *mut cudaLibrary_t, fileName: *const ::core::ffi::c_char, jitOptions: *mut cudaJitOption, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut cudaLibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLibraryUnload(library: cudaLibrary_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLibrary_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLibraryUnload") });
        _f(library)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLibraryUnload(library: cudaLibrary_t) -> cudaError_t;
        }
        cudaLibraryUnload(library)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLogsCurrent(iterator_out: *mut cudaLogIterator, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLogIterator, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLogsCurrent") });
        _f(iterator_out, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLogsCurrent(iterator_out: *mut cudaLogIterator, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLogsCurrent(iterator_out, flags)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLogsDumpToFile(iterator: *mut cudaLogIterator, pathToFile: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLogIterator, *const ::core::ffi::c_char, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLogsDumpToFile") });
        _f(iterator, pathToFile, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLogsDumpToFile(iterator: *mut cudaLogIterator, pathToFile: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLogsDumpToFile(iterator, pathToFile, flags)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLogsDumpToMemory(iterator: *mut cudaLogIterator, buffer: *mut ::core::ffi::c_char, size: *mut usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaLogIterator, *mut ::core::ffi::c_char, *mut usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLogsDumpToMemory") });
        _f(iterator, buffer, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLogsDumpToMemory(iterator: *mut cudaLogIterator, buffer: *mut ::core::ffi::c_char, size: *mut usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaLogsDumpToMemory(iterator, buffer, size, flags)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLogsRegisterCallback(callbackFunc: cudaLogsCallback_t, userData: *mut ::core::ffi::c_void, callback_out: *mut cudaLogsCallbackHandle) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLogsCallback_t, *mut ::core::ffi::c_void, *mut cudaLogsCallbackHandle) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLogsRegisterCallback") });
        _f(callbackFunc, userData, callback_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLogsRegisterCallback(callbackFunc: cudaLogsCallback_t, userData: *mut ::core::ffi::c_void, callback_out: *mut cudaLogsCallbackHandle) -> cudaError_t;
        }
        cudaLogsRegisterCallback(callbackFunc, userData, callback_out)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaLogsUnregisterCallback(callback: cudaLogsCallbackHandle) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLogsCallbackHandle) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaLogsUnregisterCallback") });
        _f(callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaLogsUnregisterCallback(callback: cudaLogsCallbackHandle) -> cudaError_t;
        }
        cudaLogsUnregisterCallback(callback)
    }
}
pub unsafe fn cudaMalloc(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMalloc") });
        _f(devPtr, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMalloc(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t;
        }
        cudaMalloc(devPtr, size)
    }
}
pub unsafe fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaPitchedPtr, cudaExtent) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMalloc3D") });
        _f(pitchedDevPtr, extent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;
        }
        cudaMalloc3D(pitchedDevPtr, extent)
    }
}
pub unsafe fn cudaMalloc3DArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArray_t, *const cudaChannelFormatDesc, cudaExtent, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMalloc3DArray") });
        _f(array, desc, extent, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMalloc3DArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaMalloc3DArray(array, desc, extent, flags)
    }
}
pub unsafe fn cudaMallocArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, width: usize, height: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArray_t, *const cudaChannelFormatDesc, usize, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocArray") });
        _f(array, desc, width, height, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocArray(array: *mut cudaArray_t, desc: *const cudaChannelFormatDesc, width: usize, height: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaMallocArray(array, desc, width, height, flags)
    }
}
pub unsafe fn cudaMallocAsync(devPtr: *mut *mut ::core::ffi::c_void, size: usize, hStream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocAsync") });
        _f(devPtr, size, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocAsync(devPtr: *mut *mut ::core::ffi::c_void, size: usize, hStream: cudaStream_t) -> cudaError_t;
        }
        cudaMallocAsync(devPtr, size, hStream)
    }
}
pub unsafe fn cudaMallocFromPoolAsync(ptr: *mut *mut ::core::ffi::c_void, size: usize, memPool: cudaMemPool_t, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize, cudaMemPool_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocFromPoolAsync") });
        _f(ptr, size, memPool, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocFromPoolAsync(ptr: *mut *mut ::core::ffi::c_void, size: usize, memPool: cudaMemPool_t, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMallocFromPoolAsync(ptr, size, memPool, stream)
    }
}
pub unsafe fn cudaMallocHost(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocHost") });
        _f(ptr, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocHost(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t;
        }
        cudaMallocHost(ptr, size)
    }
}
pub unsafe fn cudaMallocManaged(devPtr: *mut *mut ::core::ffi::c_void, size: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocManaged") });
        _f(devPtr, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocManaged(devPtr: *mut *mut ::core::ffi::c_void, size: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaMallocManaged(devPtr, size, flags)
    }
}
pub unsafe fn cudaMallocMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, numLevels: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMipmappedArray_t, *const cudaChannelFormatDesc, cudaExtent, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocMipmappedArray") });
        _f(mipmappedArray, desc, extent, numLevels, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocMipmappedArray(mipmappedArray: *mut cudaMipmappedArray_t, desc: *const cudaChannelFormatDesc, extent: cudaExtent, numLevels: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaMallocMipmappedArray(mipmappedArray, desc, extent, numLevels, flags)
    }
}
pub unsafe fn cudaMallocPitch(devPtr: *mut *mut ::core::ffi::c_void, pitch: *mut usize, width: usize, height: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, usize, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMallocPitch") });
        _f(devPtr, pitch, width, height)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMallocPitch(devPtr: *mut *mut ::core::ffi::c_void, pitch: *mut usize, width: usize, height: usize) -> cudaError_t;
        }
        cudaMallocPitch(devPtr, pitch, width, height)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaMemAdvise(devPtr: *const ::core::ffi::c_void, count: usize, advice: cudaMemoryAdvise, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, cudaMemoryAdvise, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemAdvise") });
        _f(devPtr, count, advice, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemAdvise(devPtr: *const ::core::ffi::c_void, count: usize, advice: cudaMemoryAdvise, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaMemAdvise(devPtr, count, advice, device)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemAdvise(devPtr: *const ::core::ffi::c_void, count: usize, advice: cudaMemoryAdvise, location: cudaMemLocation) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, cudaMemoryAdvise, cudaMemLocation) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemAdvise") });
        _f(devPtr, count, advice, location)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemAdvise(devPtr: *const ::core::ffi::c_void, count: usize, advice: cudaMemoryAdvise, location: cudaMemLocation) -> cudaError_t;
        }
        cudaMemAdvise(devPtr, count, advice, location)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaMemAdvise_v2(devPtr: *const ::core::ffi::c_void, count: usize, advice: cudaMemoryAdvise, location: cudaMemLocation) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, cudaMemoryAdvise, cudaMemLocation) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemAdvise_v2") });
        _f(devPtr, count, advice, location)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemAdvise_v2(devPtr: *const ::core::ffi::c_void, count: usize, advice: cudaMemoryAdvise, location: cudaMemLocation) -> cudaError_t;
        }
        cudaMemAdvise_v2(devPtr, count, advice, location)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemDiscardAndPrefetchBatchAsync(dptrs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, prefetchLocs: *mut cudaMemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, usize, *mut cudaMemLocation, *mut usize, usize, ::core::ffi::c_ulonglong, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemDiscardAndPrefetchBatchAsync") });
        _f(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemDiscardAndPrefetchBatchAsync(dptrs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, prefetchLocs: *mut cudaMemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemDiscardBatchAsync(dptrs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, usize, ::core::ffi::c_ulonglong, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemDiscardBatchAsync") });
        _f(dptrs, sizes, count, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemDiscardBatchAsync(dptrs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemDiscardBatchAsync(dptrs, sizes, count, flags, stream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemGetDefaultMemPool(memPool: *mut cudaMemPool_t, location: *mut cudaMemLocation, type_: cudaMemAllocationType) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPool_t, *mut cudaMemLocation, cudaMemAllocationType) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemGetDefaultMemPool") });
        _f(memPool, location, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemGetDefaultMemPool(memPool: *mut cudaMemPool_t, location: *mut cudaMemLocation, type_: cudaMemAllocationType) -> cudaError_t;
        }
        cudaMemGetDefaultMemPool(memPool, location, type_)
    }
}
pub unsafe fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemGetInfo") });
        _f(free, total)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
        }
        cudaMemGetInfo(free, total)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemGetMemPool(memPool: *mut cudaMemPool_t, location: *mut cudaMemLocation, type_: cudaMemAllocationType) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPool_t, *mut cudaMemLocation, cudaMemAllocationType) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemGetMemPool") });
        _f(memPool, location, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemGetMemPool(memPool: *mut cudaMemPool_t, location: *mut cudaMemLocation, type_: cudaMemAllocationType) -> cudaError_t;
        }
        cudaMemGetMemPool(memPool, location, type_)
    }
}
pub unsafe fn cudaMemPoolCreate(memPool: *mut cudaMemPool_t, poolProps: *const cudaMemPoolProps) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPool_t, *const cudaMemPoolProps) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolCreate") });
        _f(memPool, poolProps)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolCreate(memPool: *mut cudaMemPool_t, poolProps: *const cudaMemPoolProps) -> cudaError_t;
        }
        cudaMemPoolCreate(memPool, poolProps)
    }
}
pub unsafe fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaMemPool_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolDestroy") });
        _f(memPool)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t;
        }
        cudaMemPoolDestroy(memPool)
    }
}
pub unsafe fn cudaMemPoolExportPointer(exportData: *mut cudaMemPoolPtrExportData, ptr: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPoolPtrExportData, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolExportPointer") });
        _f(exportData, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolExportPointer(exportData: *mut cudaMemPoolPtrExportData, ptr: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaMemPoolExportPointer(exportData, ptr)
    }
}
pub unsafe fn cudaMemPoolExportToShareableHandle(shareableHandle: *mut ::core::ffi::c_void, memPool: cudaMemPool_t, handleType: cudaMemAllocationHandleType, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, cudaMemPool_t, cudaMemAllocationHandleType, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolExportToShareableHandle") });
        _f(shareableHandle, memPool, handleType, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolExportToShareableHandle(shareableHandle: *mut ::core::ffi::c_void, memPool: cudaMemPool_t, handleType: cudaMemAllocationHandleType, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaMemPoolExportToShareableHandle(shareableHandle, memPool, handleType, flags)
    }
}
pub unsafe fn cudaMemPoolGetAccess(flags: *mut cudaMemAccessFlags, memPool: cudaMemPool_t, location: *mut cudaMemLocation) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemAccessFlags, cudaMemPool_t, *mut cudaMemLocation) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolGetAccess") });
        _f(flags, memPool, location)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolGetAccess(flags: *mut cudaMemAccessFlags, memPool: cudaMemPool_t, location: *mut cudaMemLocation) -> cudaError_t;
        }
        cudaMemPoolGetAccess(flags, memPool, location)
    }
}
pub unsafe fn cudaMemPoolGetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaMemPool_t, cudaMemPoolAttr, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolGetAttribute") });
        _f(memPool, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolGetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaMemPoolGetAttribute(memPool, attr, value)
    }
}
pub unsafe fn cudaMemPoolImportFromShareableHandle(memPool: *mut cudaMemPool_t, shareableHandle: *mut ::core::ffi::c_void, handleType: cudaMemAllocationHandleType, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemPool_t, *mut ::core::ffi::c_void, cudaMemAllocationHandleType, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolImportFromShareableHandle") });
        _f(memPool, shareableHandle, handleType, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolImportFromShareableHandle(memPool: *mut cudaMemPool_t, shareableHandle: *mut ::core::ffi::c_void, handleType: cudaMemAllocationHandleType, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaMemPoolImportFromShareableHandle(memPool, shareableHandle, handleType, flags)
    }
}
pub unsafe fn cudaMemPoolImportPointer(ptr: *mut *mut ::core::ffi::c_void, memPool: cudaMemPool_t, exportData: *mut cudaMemPoolPtrExportData) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, cudaMemPool_t, *mut cudaMemPoolPtrExportData) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolImportPointer") });
        _f(ptr, memPool, exportData)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolImportPointer(ptr: *mut *mut ::core::ffi::c_void, memPool: cudaMemPool_t, exportData: *mut cudaMemPoolPtrExportData) -> cudaError_t;
        }
        cudaMemPoolImportPointer(ptr, memPool, exportData)
    }
}
pub unsafe fn cudaMemPoolSetAccess(memPool: cudaMemPool_t, descList: *const cudaMemAccessDesc, count: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaMemPool_t, *const cudaMemAccessDesc, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolSetAccess") });
        _f(memPool, descList, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolSetAccess(memPool: cudaMemPool_t, descList: *const cudaMemAccessDesc, count: usize) -> cudaError_t;
        }
        cudaMemPoolSetAccess(memPool, descList, count)
    }
}
pub unsafe fn cudaMemPoolSetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *mut ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaMemPool_t, cudaMemPoolAttr, *mut ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolSetAttribute") });
        _f(memPool, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolSetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *mut ::core::ffi::c_void) -> cudaError_t;
        }
        cudaMemPoolSetAttribute(memPool, attr, value)
    }
}
pub unsafe fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaMemPool_t, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPoolTrimTo") });
        _f(memPool, minBytesToKeep)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t;
        }
        cudaMemPoolTrimTo(memPool, minBytesToKeep)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaMemPrefetchAsync(devPtr: *const ::core::ffi::c_void, count: usize, dstDevice: ::core::ffi::c_int, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, ::core::ffi::c_int, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPrefetchAsync") });
        _f(devPtr, count, dstDevice, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPrefetchAsync(devPtr: *const ::core::ffi::c_void, count: usize, dstDevice: ::core::ffi::c_int, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemPrefetchAsync(devPtr, count, dstDevice, stream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemPrefetchAsync(devPtr: *const ::core::ffi::c_void, count: usize, location: cudaMemLocation, flags: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, cudaMemLocation, ::core::ffi::c_uint, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPrefetchAsync") });
        _f(devPtr, count, location, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPrefetchAsync(devPtr: *const ::core::ffi::c_void, count: usize, location: cudaMemLocation, flags: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemPrefetchAsync(devPtr, count, location, flags, stream)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaMemPrefetchAsync_v2(devPtr: *const ::core::ffi::c_void, count: usize, location: cudaMemLocation, flags: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, usize, cudaMemLocation, ::core::ffi::c_uint, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPrefetchAsync_v2") });
        _f(devPtr, count, location, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPrefetchAsync_v2(devPtr: *const ::core::ffi::c_void, count: usize, location: cudaMemLocation, flags: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemPrefetchAsync_v2(devPtr, count, location, flags, stream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemPrefetchBatchAsync(dptrs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, prefetchLocs: *mut cudaMemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, usize, *mut cudaMemLocation, *mut usize, usize, ::core::ffi::c_ulonglong, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemPrefetchBatchAsync") });
        _f(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemPrefetchBatchAsync(dptrs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, prefetchLocs: *mut cudaMemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, stream)
    }
}
pub unsafe fn cudaMemRangeGetAttribute(data: *mut ::core::ffi::c_void, dataSize: usize, attribute: cudaMemRangeAttribute, devPtr: *const ::core::ffi::c_void, count: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, cudaMemRangeAttribute, *const ::core::ffi::c_void, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemRangeGetAttribute") });
        _f(data, dataSize, attribute, devPtr, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemRangeGetAttribute(data: *mut ::core::ffi::c_void, dataSize: usize, attribute: cudaMemRangeAttribute, devPtr: *const ::core::ffi::c_void, count: usize) -> cudaError_t;
        }
        cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
    }
}
pub unsafe fn cudaMemRangeGetAttributes(data: *mut *mut ::core::ffi::c_void, dataSizes: *mut usize, attributes: *mut cudaMemRangeAttribute, numAttributes: usize, devPtr: *const ::core::ffi::c_void, count: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, *mut cudaMemRangeAttribute, usize, *const ::core::ffi::c_void, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemRangeGetAttributes") });
        _f(data, dataSizes, attributes, numAttributes, devPtr, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemRangeGetAttributes(data: *mut *mut ::core::ffi::c_void, dataSizes: *mut usize, attributes: *mut cudaMemRangeAttribute, numAttributes: usize, devPtr: *const ::core::ffi::c_void, count: usize) -> cudaError_t;
        }
        cudaMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemSetMemPool(location: *mut cudaMemLocation, type_: cudaMemAllocationType, memPool: cudaMemPool_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemLocation, cudaMemAllocationType, cudaMemPool_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemSetMemPool") });
        _f(location, type_, memPool)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemSetMemPool(location: *mut cudaMemLocation, type_: cudaMemAllocationType, memPool: cudaMemPool_t) -> cudaError_t;
        }
        cudaMemSetMemPool(location, type_, memPool)
    }
}
pub unsafe fn cudaMemcpy(dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy") });
        _f(dst, src, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy(dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpy(dst, src, count, kind)
    }
}
pub unsafe fn cudaMemcpy2D(dst: *mut ::core::ffi::c_void, dpitch: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2D") });
        _f(dst, dpitch, src, spitch, width, height, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2D(dst: *mut ::core::ffi::c_void, dpitch: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
    }
}
pub unsafe fn cudaMemcpy2DArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t, usize, usize, cudaArray_const_t, usize, usize, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2DArrayToArray") });
        _f(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2DArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpy2DArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind)
    }
}
pub unsafe fn cudaMemcpy2DAsync(dst: *mut ::core::ffi::c_void, dpitch: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, *const ::core::ffi::c_void, usize, usize, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2DAsync") });
        _f(dst, dpitch, src, spitch, width, height, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2DAsync(dst: *mut ::core::ffi::c_void, dpitch: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
    }
}
pub unsafe fn cudaMemcpy2DFromArray(dst: *mut ::core::ffi::c_void, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, cudaArray_const_t, usize, usize, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2DFromArray") });
        _f(dst, dpitch, src, wOffset, hOffset, width, height, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2DFromArray(dst: *mut ::core::ffi::c_void, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind)
    }
}
pub unsafe fn cudaMemcpy2DFromArrayAsync(dst: *mut ::core::ffi::c_void, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, cudaArray_const_t, usize, usize, usize, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2DFromArrayAsync") });
        _f(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2DFromArrayAsync(dst: *mut ::core::ffi::c_void, dpitch: usize, src: cudaArray_const_t, wOffset: usize, hOffset: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy2DFromArrayAsync(dst, dpitch, src, wOffset, hOffset, width, height, kind, stream)
    }
}
pub unsafe fn cudaMemcpy2DToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t, usize, usize, *const ::core::ffi::c_void, usize, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2DToArray") });
        _f(dst, wOffset, hOffset, src, spitch, width, height, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2DToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind)
    }
}
pub unsafe fn cudaMemcpy2DToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t, usize, usize, *const ::core::ffi::c_void, usize, usize, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy2DToArrayAsync") });
        _f(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy2DToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, spitch: usize, width: usize, height: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, src, spitch, width, height, kind, stream)
    }
}
pub unsafe fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaMemcpy3DParms) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3D") });
        _f(p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t;
        }
        cudaMemcpy3D(p)
    }
}
pub unsafe fn cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaMemcpy3DParms, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3DAsync") });
        _f(p, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy3DAsync(p, stream)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaMemcpy3DBatchAsync(numOps: usize, opList: *mut cudaMemcpy3DBatchOp, failIdx: *mut usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(usize, *mut cudaMemcpy3DBatchOp, *mut usize, ::core::ffi::c_ulonglong, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3DBatchAsync") });
        _f(numOps, opList, failIdx, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3DBatchAsync(numOps: usize, opList: *mut cudaMemcpy3DBatchOp, failIdx: *mut usize, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy3DBatchAsync(numOps, opList, failIdx, flags, stream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemcpy3DBatchAsync(numOps: usize, opList: *mut cudaMemcpy3DBatchOp, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(usize, *mut cudaMemcpy3DBatchOp, ::core::ffi::c_ulonglong, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3DBatchAsync") });
        _f(numOps, opList, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3DBatchAsync(numOps: usize, opList: *mut cudaMemcpy3DBatchOp, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy3DBatchAsync(numOps, opList, flags, stream)
    }
}
pub unsafe fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaMemcpy3DPeerParms) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3DPeer") });
        _f(p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t;
        }
        cudaMemcpy3DPeer(p)
    }
}
pub unsafe fn cudaMemcpy3DPeerAsync(p: *const cudaMemcpy3DPeerParms, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaMemcpy3DPeerParms, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3DPeerAsync") });
        _f(p, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3DPeerAsync(p: *const cudaMemcpy3DPeerParms, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy3DPeerAsync(p, stream)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemcpy3DWithAttributesAsync(op: *mut cudaMemcpy3DBatchOp, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaMemcpy3DBatchOp, ::core::ffi::c_ulonglong, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpy3DWithAttributesAsync") });
        _f(op, flags, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpy3DWithAttributesAsync(op: *mut cudaMemcpy3DBatchOp, flags: ::core::ffi::c_ulonglong, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpy3DWithAttributesAsync(op, flags, stream)
    }
}
pub unsafe fn cudaMemcpyArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t, usize, usize, cudaArray_const_t, usize, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyArrayToArray") });
        _f(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyArrayToArray(dst: cudaArray_t, wOffsetDst: usize, hOffsetDst: usize, src: cudaArray_const_t, wOffsetSrc: usize, hOffsetSrc: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind)
    }
}
pub unsafe fn cudaMemcpyAsync(dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyAsync") });
        _f(dst, src, count, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyAsync(dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyAsync(dst, src, count, kind, stream)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaMemcpyBatchAsync(dsts: *mut *mut ::core::ffi::c_void, srcs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, attrs: *mut cudaMemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, failIdx: *mut usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void, *mut usize, usize, *mut cudaMemcpyAttributes, *mut usize, usize, *mut usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyBatchAsync") });
        _f(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyBatchAsync(dsts: *mut *mut ::core::ffi::c_void, srcs: *mut *mut ::core::ffi::c_void, sizes: *mut usize, count: usize, attrs: *mut cudaMemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, failIdx: *mut usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemcpyBatchAsync(dsts: *const *mut ::core::ffi::c_void, srcs: *const *const ::core::ffi::c_void, sizes: *const usize, count: usize, attrs: *mut cudaMemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const *mut ::core::ffi::c_void, *const *const ::core::ffi::c_void, *const usize, usize, *mut cudaMemcpyAttributes, *mut usize, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyBatchAsync") });
        _f(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyBatchAsync(dsts: *const *mut ::core::ffi::c_void, srcs: *const *const ::core::ffi::c_void, sizes: *const usize, count: usize, attrs: *mut cudaMemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, stream)
    }
}
pub unsafe fn cudaMemcpyFromArray(dst: *mut ::core::ffi::c_void, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, cudaArray_const_t, usize, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyFromArray") });
        _f(dst, src, wOffset, hOffset, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyFromArray(dst: *mut ::core::ffi::c_void, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind)
    }
}
pub unsafe fn cudaMemcpyFromArrayAsync(dst: *mut ::core::ffi::c_void, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, cudaArray_const_t, usize, usize, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyFromArrayAsync") });
        _f(dst, src, wOffset, hOffset, count, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyFromArrayAsync(dst: *mut ::core::ffi::c_void, src: cudaArray_const_t, wOffset: usize, hOffset: usize, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyFromArrayAsync(dst, src, wOffset, hOffset, count, kind, stream)
    }
}
pub unsafe fn cudaMemcpyFromSymbol(dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyFromSymbol") });
        _f(dst, symbol, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyFromSymbol(dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpyFromSymbol(dst, symbol, count, offset, kind)
    }
}
pub unsafe fn cudaMemcpyFromSymbolAsync(dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyFromSymbolAsync") });
        _f(dst, symbol, count, offset, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyFromSymbolAsync(dst: *mut ::core::ffi::c_void, symbol: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyFromSymbolAsync(dst, symbol, count, offset, kind, stream)
    }
}
pub unsafe fn cudaMemcpyPeer(dst: *mut ::core::ffi::c_void, dstDevice: ::core::ffi::c_int, src: *const ::core::ffi::c_void, srcDevice: ::core::ffi::c_int, count: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyPeer") });
        _f(dst, dstDevice, src, srcDevice, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyPeer(dst: *mut ::core::ffi::c_void, dstDevice: ::core::ffi::c_int, src: *const ::core::ffi::c_void, srcDevice: ::core::ffi::c_int, count: usize) -> cudaError_t;
        }
        cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)
    }
}
pub unsafe fn cudaMemcpyPeerAsync(dst: *mut ::core::ffi::c_void, dstDevice: ::core::ffi::c_int, src: *const ::core::ffi::c_void, srcDevice: ::core::ffi::c_int, count: usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyPeerAsync") });
        _f(dst, dstDevice, src, srcDevice, count, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyPeerAsync(dst: *mut ::core::ffi::c_void, dstDevice: ::core::ffi::c_int, src: *const ::core::ffi::c_void, srcDevice: ::core::ffi::c_int, count: usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream)
    }
}
pub unsafe fn cudaMemcpyToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t, usize, usize, *const ::core::ffi::c_void, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyToArray") });
        _f(dst, wOffset, hOffset, src, count, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyToArray(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind)
    }
}
pub unsafe fn cudaMemcpyToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaArray_t, usize, usize, *const ::core::ffi::c_void, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyToArrayAsync") });
        _f(dst, wOffset, hOffset, src, count, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyToArrayAsync(dst: cudaArray_t, wOffset: usize, hOffset: usize, src: *const ::core::ffi::c_void, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyToArrayAsync(dst, wOffset, hOffset, src, count, kind, stream)
    }
}
pub unsafe fn cudaMemcpyToSymbol(symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyToSymbol") });
        _f(symbol, src, count, offset, kind)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyToSymbol(symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind) -> cudaError_t;
        }
        cudaMemcpyToSymbol(symbol, src, count, offset, kind)
    }
}
pub unsafe fn cudaMemcpyToSymbolAsync(symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, *const ::core::ffi::c_void, usize, usize, cudaMemcpyKind, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyToSymbolAsync") });
        _f(symbol, src, count, offset, kind, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyToSymbolAsync(symbol: *const ::core::ffi::c_void, src: *const ::core::ffi::c_void, count: usize, offset: usize, kind: cudaMemcpyKind, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMemcpyWithAttributesAsync(dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, size: usize, attr: *mut cudaMemcpyAttributes, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, *const ::core::ffi::c_void, usize, *mut cudaMemcpyAttributes, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemcpyWithAttributesAsync") });
        _f(dst, src, size, attr, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemcpyWithAttributesAsync(dst: *mut ::core::ffi::c_void, src: *const ::core::ffi::c_void, size: usize, attr: *mut cudaMemcpyAttributes, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemcpyWithAttributesAsync(dst, src, size, attr, stream)
    }
}
pub unsafe fn cudaMemset(devPtr: *mut ::core::ffi::c_void, value: ::core::ffi::c_int, count: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, ::core::ffi::c_int, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemset") });
        _f(devPtr, value, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemset(devPtr: *mut ::core::ffi::c_void, value: ::core::ffi::c_int, count: usize) -> cudaError_t;
        }
        cudaMemset(devPtr, value, count)
    }
}
pub unsafe fn cudaMemset2D(devPtr: *mut ::core::ffi::c_void, pitch: usize, value: ::core::ffi::c_int, width: usize, height: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, ::core::ffi::c_int, usize, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemset2D") });
        _f(devPtr, pitch, value, width, height)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemset2D(devPtr: *mut ::core::ffi::c_void, pitch: usize, value: ::core::ffi::c_int, width: usize, height: usize) -> cudaError_t;
        }
        cudaMemset2D(devPtr, pitch, value, width, height)
    }
}
pub unsafe fn cudaMemset2DAsync(devPtr: *mut ::core::ffi::c_void, pitch: usize, value: ::core::ffi::c_int, width: usize, height: usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, ::core::ffi::c_int, usize, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemset2DAsync") });
        _f(devPtr, pitch, value, width, height, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemset2DAsync(devPtr: *mut ::core::ffi::c_void, pitch: usize, value: ::core::ffi::c_int, width: usize, height: usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)
    }
}
pub unsafe fn cudaMemset3D(pitchedDevPtr: cudaPitchedPtr, value: ::core::ffi::c_int, extent: cudaExtent) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaPitchedPtr, ::core::ffi::c_int, cudaExtent) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemset3D") });
        _f(pitchedDevPtr, value, extent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemset3D(pitchedDevPtr: cudaPitchedPtr, value: ::core::ffi::c_int, extent: cudaExtent) -> cudaError_t;
        }
        cudaMemset3D(pitchedDevPtr, value, extent)
    }
}
pub unsafe fn cudaMemset3DAsync(pitchedDevPtr: cudaPitchedPtr, value: ::core::ffi::c_int, extent: cudaExtent, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaPitchedPtr, ::core::ffi::c_int, cudaExtent, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemset3DAsync") });
        _f(pitchedDevPtr, value, extent, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemset3DAsync(pitchedDevPtr: cudaPitchedPtr, value: ::core::ffi::c_int, extent: cudaExtent, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)
    }
}
pub unsafe fn cudaMemsetAsync(devPtr: *mut ::core::ffi::c_void, value: ::core::ffi::c_int, count: usize, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, ::core::ffi::c_int, usize, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMemsetAsync") });
        _f(devPtr, value, count, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMemsetAsync(devPtr: *mut ::core::ffi::c_void, value: ::core::ffi::c_int, count: usize, stream: cudaStream_t) -> cudaError_t;
        }
        cudaMemsetAsync(devPtr, value, count, stream)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaMipmappedArrayGetMemoryRequirements(memoryRequirements: *mut cudaArrayMemoryRequirements, mipmap: cudaMipmappedArray_t, device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArrayMemoryRequirements, cudaMipmappedArray_t, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMipmappedArrayGetMemoryRequirements") });
        _f(memoryRequirements, mipmap, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMipmappedArrayGetMemoryRequirements(memoryRequirements: *mut cudaArrayMemoryRequirements, mipmap: cudaMipmappedArray_t, device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)
    }
}
pub unsafe fn cudaMipmappedArrayGetSparseProperties(sparseProperties: *mut cudaArraySparseProperties, mipmap: cudaMipmappedArray_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaArraySparseProperties, cudaMipmappedArray_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaMipmappedArrayGetSparseProperties") });
        _f(sparseProperties, mipmap)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaMipmappedArrayGetSparseProperties(sparseProperties: *mut cudaArraySparseProperties, mipmap: cudaMipmappedArray_t) -> cudaError_t;
        }
        cudaMipmappedArrayGetSparseProperties(sparseProperties, mipmap)
    }
}
pub unsafe fn cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: *mut usize, func: *const ::core::ffi::c_void, numBlocks: ::core::ffi::c_int, blockSize: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const ::core::ffi::c_void, ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaOccupancyAvailableDynamicSMemPerBlock") });
        _f(dynamicSmemSize, func, numBlocks, blockSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: *mut usize, func: *const ::core::ffi::c_void, numBlocks: ::core::ffi::c_int, blockSize: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)
    }
}
pub unsafe fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaOccupancyMaxActiveBlocksPerMultiprocessor") });
        _f(numBlocks, func, blockSize, dynamicSMemSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize) -> cudaError_t;
        }
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)
    }
}
pub unsafe fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *const ::core::ffi::c_void, ::core::ffi::c_int, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") });
        _f(numBlocks, func, blockSize, dynamicSMemSize, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaOccupancyMaxActiveClusters(numClusters: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, launchConfig: *const cudaLaunchConfig_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *const ::core::ffi::c_void, *const cudaLaunchConfig_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaOccupancyMaxActiveClusters") });
        _f(numClusters, func, launchConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaOccupancyMaxActiveClusters(numClusters: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, launchConfig: *const cudaLaunchConfig_t) -> cudaError_t;
        }
        cudaOccupancyMaxActiveClusters(numClusters, func, launchConfig)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaOccupancyMaxPotentialClusterSize(clusterSize: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, launchConfig: *const cudaLaunchConfig_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *const ::core::ffi::c_void, *const cudaLaunchConfig_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaOccupancyMaxPotentialClusterSize") });
        _f(clusterSize, func, launchConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaOccupancyMaxPotentialClusterSize(clusterSize: *mut ::core::ffi::c_int, func: *const ::core::ffi::c_void, launchConfig: *const cudaLaunchConfig_t) -> cudaError_t;
        }
        cudaOccupancyMaxPotentialClusterSize(clusterSize, func, launchConfig)
    }
}
pub unsafe fn cudaPeekAtLastError() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaPeekAtLastError") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaPeekAtLastError() -> cudaError_t;
        }
        cudaPeekAtLastError()
    }
}
pub unsafe fn cudaPointerGetAttributes(attributes: *mut cudaPointerAttributes, ptr: *const ::core::ffi::c_void) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaPointerAttributes, *const ::core::ffi::c_void) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaPointerGetAttributes") });
        _f(attributes, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaPointerGetAttributes(attributes: *mut cudaPointerAttributes, ptr: *const ::core::ffi::c_void) -> cudaError_t;
        }
        cudaPointerGetAttributes(attributes, ptr)
    }
}
pub unsafe fn cudaProfilerStop() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaProfilerStop") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaProfilerStop() -> cudaError_t;
        }
        cudaProfilerStop()
    }
}
pub unsafe fn cudaRuntimeGetVersion(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaRuntimeGetVersion") });
        _f(runtimeVersion)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaRuntimeGetVersion(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaRuntimeGetVersion(runtimeVersion)
    }
}
pub unsafe fn cudaSetDevice(device: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSetDevice") });
        _f(device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSetDevice(device: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaSetDevice(device)
    }
}
pub unsafe fn cudaSetDeviceFlags(flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSetDeviceFlags") });
        _f(flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSetDeviceFlags(flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaSetDeviceFlags(flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f64) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSetDoubleForDevice") });
        _f(d)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t;
        }
        cudaSetDoubleForDevice(d)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaSetDoubleForHost(d: *mut f64) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f64) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSetDoubleForHost") });
        _f(d)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSetDoubleForHost(d: *mut f64) -> cudaError_t;
        }
        cudaSetDoubleForHost(d)
    }
}
pub unsafe fn cudaSetValidDevices(device_arr: *mut ::core::ffi::c_int, len: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSetValidDevices") });
        _f(device_arr, len)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSetValidDevices(device_arr: *mut ::core::ffi::c_int, len: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaSetValidDevices(device_arr, len)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaSignalExternalSemaphoresAsync(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreSignalParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaExternalSemaphore_t, *const cudaExternalSemaphoreSignalParams, ::core::ffi::c_uint, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSignalExternalSemaphoresAsync") });
        _f(extSemArray, paramsArray, numExtSems, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSignalExternalSemaphoresAsync(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreSignalParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t;
        }
        cudaSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaSignalExternalSemaphoresAsync_v2(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreSignalParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaExternalSemaphore_t, *const cudaExternalSemaphoreSignalParams, ::core::ffi::c_uint, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaSignalExternalSemaphoresAsync_v2") });
        _f(extSemArray, paramsArray, numExtSems, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaSignalExternalSemaphoresAsync_v2(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreSignalParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t;
        }
        cudaSignalExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream)
    }
}
pub unsafe fn cudaStreamAddCallback(stream: cudaStream_t, callback: cudaStreamCallback_t, userData: *mut ::core::ffi::c_void, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaStreamCallback_t, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamAddCallback") });
        _f(stream, callback, userData, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamAddCallback(stream: cudaStream_t, callback: cudaStreamCallback_t, userData: *mut ::core::ffi::c_void, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamAddCallback(stream, callback, userData, flags)
    }
}
pub unsafe fn cudaStreamAttachMemAsync(stream: cudaStream_t, devPtr: *mut ::core::ffi::c_void, length: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut ::core::ffi::c_void, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamAttachMemAsync") });
        _f(stream, devPtr, length, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamAttachMemAsync(stream: cudaStream_t, devPtr: *mut ::core::ffi::c_void, length: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamAttachMemAsync(stream, devPtr, length, flags)
    }
}
pub unsafe fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaStreamCaptureMode) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamBeginCapture") });
        _f(stream, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t;
        }
        cudaStreamBeginCapture(stream, mode)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamBeginCaptureToGraph(stream: cudaStream_t, graph: cudaGraph_t, dependencies: *const cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, mode: cudaStreamCaptureMode) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaGraph_t, *const cudaGraphNode_t, *const cudaGraphEdgeData, usize, cudaStreamCaptureMode) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamBeginCaptureToGraph") });
        _f(stream, graph, dependencies, dependencyData, numDependencies, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamBeginCaptureToGraph(stream: cudaStream_t, graph: cudaGraph_t, dependencies: *const cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, mode: cudaStreamCaptureMode) -> cudaError_t;
        }
        cudaStreamBeginCaptureToGraph(stream, graph, dependencies, dependencyData, numDependencies, mode)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cudaStreamBeginRecaptureToGraph(stream: cudaStream_t, mode: cudaStreamCaptureMode, graph: cudaGraph_t, callbackData: *mut cudaGraphRecaptureCallbackData) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaStreamCaptureMode, cudaGraph_t, *mut cudaGraphRecaptureCallbackData) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamBeginRecaptureToGraph") });
        _f(stream, mode, graph, callbackData)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamBeginRecaptureToGraph(stream: cudaStream_t, mode: cudaStreamCaptureMode, graph: cudaGraph_t, callbackData: *mut cudaGraphRecaptureCallbackData) -> cudaError_t;
        }
        cudaStreamBeginRecaptureToGraph(stream, mode, graph, callbackData)
    }
}
pub unsafe fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamCopyAttributes") });
        _f(dst, src)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t;
        }
        cudaStreamCopyAttributes(dst, src)
    }
}
pub unsafe fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamCreate") });
        _f(pStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
        }
        cudaStreamCreate(pStream)
    }
}
pub unsafe fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaStream_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamCreateWithFlags") });
        _f(pStream, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamCreateWithFlags(pStream, flags)
    }
}
pub unsafe fn cudaStreamCreateWithPriority(pStream: *mut cudaStream_t, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaStream_t, ::core::ffi::c_uint, ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamCreateWithPriority") });
        _f(pStream, flags, priority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamCreateWithPriority(pStream: *mut cudaStream_t, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> cudaError_t;
        }
        cudaStreamCreateWithPriority(pStream, flags, priority)
    }
}
pub unsafe fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamDestroy") });
        _f(stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
        }
        cudaStreamDestroy(stream)
    }
}
pub unsafe fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaGraph_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamEndCapture") });
        _f(stream, pGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
        }
        cudaStreamEndCapture(stream, pGraph)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub unsafe fn cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaStreamAttrID, value_out: *mut cudaStreamAttrValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaStreamAttrID, *mut cudaStreamAttrValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetAttribute") });
        _f(hStream, attr, value_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaStreamAttrID, value_out: *mut cudaStreamAttrValue) -> cudaError_t;
        }
        cudaStreamGetAttribute(hStream, attr, value_out)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value_out: *mut cudaLaunchAttributeValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaLaunchAttributeID, *mut cudaLaunchAttributeValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetAttribute") });
        _f(hStream, attr, value_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value_out: *mut cudaLaunchAttributeValue) -> cudaError_t;
        }
        cudaStreamGetAttribute(hStream, attr, value_out)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaStreamGetCaptureInfo(stream: cudaStream_t, pCaptureStatus: *mut cudaStreamCaptureStatus, pId: *mut ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaStreamCaptureStatus, *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetCaptureInfo") });
        _f(stream, pCaptureStatus, pId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetCaptureInfo(stream: cudaStream_t, pCaptureStatus: *mut cudaStreamCaptureStatus, pId: *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamGetCaptureInfo(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut ::core::ffi::c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, edgeData_out: *mut *const cudaGraphEdgeData, numDependencies_out: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaStreamCaptureStatus, *mut ::core::ffi::c_ulonglong, *mut cudaGraph_t, *mut *const cudaGraphNode_t, *mut *const cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetCaptureInfo") });
        _f(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetCaptureInfo(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut ::core::ffi::c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, edgeData_out: *mut *const cudaGraphEdgeData, numDependencies_out: *mut usize) -> cudaError_t;
        }
        cudaStreamGetCaptureInfo(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaStreamGetCaptureInfo_v2(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut ::core::ffi::c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, numDependencies_out: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaStreamCaptureStatus, *mut ::core::ffi::c_ulonglong, *mut cudaGraph_t, *mut *const cudaGraphNode_t, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetCaptureInfo_v2") });
        _f(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetCaptureInfo_v2(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut ::core::ffi::c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, numDependencies_out: *mut usize) -> cudaError_t;
        }
        cudaStreamGetCaptureInfo_v2(stream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaStreamGetCaptureInfo_v3(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut ::core::ffi::c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, edgeData_out: *mut *const cudaGraphEdgeData, numDependencies_out: *mut usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaStreamCaptureStatus, *mut ::core::ffi::c_ulonglong, *mut cudaGraph_t, *mut *const cudaGraphNode_t, *mut *const cudaGraphEdgeData, *mut usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetCaptureInfo_v3") });
        _f(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetCaptureInfo_v3(stream: cudaStream_t, captureStatus_out: *mut cudaStreamCaptureStatus, id_out: *mut ::core::ffi::c_ulonglong, graph_out: *mut cudaGraph_t, dependencies_out: *mut *const cudaGraphNode_t, edgeData_out: *mut *const cudaGraphEdgeData, numDependencies_out: *mut usize) -> cudaError_t;
        }
        cudaStreamGetCaptureInfo_v3(stream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamGetDevResource(hStream: cudaStream_t, resource: *mut cudaDevResource, type_: cudaDevResourceType) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaDevResource, cudaDevResourceType) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetDevResource") });
        _f(hStream, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetDevResource(hStream: cudaStream_t, resource: *mut cudaDevResource, type_: cudaDevResourceType) -> cudaError_t;
        }
        cudaStreamGetDevResource(hStream, resource, type_)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamGetDevice(hStream: cudaStream_t, device: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetDevice") });
        _f(hStream, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetDevice(hStream: cudaStream_t, device: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaStreamGetDevice(hStream, device)
    }
}
pub unsafe fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetFlags") });
        _f(hStream, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamGetFlags(hStream, flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamGetId(hStream: cudaStream_t, streamId: *mut ::core::ffi::c_ulonglong) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetId") });
        _f(hStream, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetId(hStream: cudaStream_t, streamId: *mut ::core::ffi::c_ulonglong) -> cudaError_t;
        }
        cudaStreamGetId(hStream, streamId)
    }
}
pub unsafe fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut ::core::ffi::c_int) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut ::core::ffi::c_int) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamGetPriority") });
        _f(hStream, priority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut ::core::ffi::c_int) -> cudaError_t;
        }
        cudaStreamGetPriority(hStream, priority)
    }
}
pub unsafe fn cudaStreamIsCapturing(stream: cudaStream_t, pCaptureStatus: *mut cudaStreamCaptureStatus) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaStreamCaptureStatus) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamIsCapturing") });
        _f(stream, pCaptureStatus)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamIsCapturing(stream: cudaStream_t, pCaptureStatus: *mut cudaStreamCaptureStatus) -> cudaError_t;
        }
        cudaStreamIsCapturing(stream, pCaptureStatus)
    }
}
pub unsafe fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamQuery") });
        _f(stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
        }
        cudaStreamQuery(stream)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub unsafe fn cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaStreamAttrID, value: *const cudaStreamAttrValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaStreamAttrID, *const cudaStreamAttrValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamSetAttribute") });
        _f(hStream, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaStreamAttrID, value: *const cudaStreamAttrValue) -> cudaError_t;
        }
        cudaStreamSetAttribute(hStream, attr, value)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value: *const cudaLaunchAttributeValue) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaLaunchAttributeID, *const cudaLaunchAttributeValue) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamSetAttribute") });
        _f(hStream, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamSetAttribute(hStream: cudaStream_t, attr: cudaLaunchAttributeID, value: *const cudaLaunchAttributeValue) -> cudaError_t;
        }
        cudaStreamSetAttribute(hStream, attr, value)
    }
}
pub unsafe fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamSynchronize") });
        _f(stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
        }
        cudaStreamSynchronize(stream)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, numDependencies: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaGraphNode_t, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamUpdateCaptureDependencies") });
        _f(stream, dependencies, numDependencies, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, numDependencies: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamUpdateCaptureDependencies(stream, dependencies, numDependencies, flags)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaGraphNode_t, *const cudaGraphEdgeData, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamUpdateCaptureDependencies") });
        _f(stream, dependencies, dependencyData, numDependencies, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamUpdateCaptureDependencies(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamUpdateCaptureDependencies(stream, dependencies, dependencyData, numDependencies, flags)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaStreamUpdateCaptureDependencies_v2(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, *mut cudaGraphNode_t, *const cudaGraphEdgeData, usize, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamUpdateCaptureDependencies_v2") });
        _f(stream, dependencies, dependencyData, numDependencies, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamUpdateCaptureDependencies_v2(stream: cudaStream_t, dependencies: *mut cudaGraphNode_t, dependencyData: *const cudaGraphEdgeData, numDependencies: usize, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamUpdateCaptureDependencies_v2(stream, dependencies, dependencyData, numDependencies, flags)
    }
}
pub unsafe fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaStream_t, cudaEvent_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaStreamWaitEvent") });
        _f(stream, event, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaStreamWaitEvent(stream, event, flags)
    }
}
pub unsafe fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaStreamCaptureMode) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadExchangeStreamCaptureMode") });
        _f(mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t;
        }
        cudaThreadExchangeStreamCaptureMode(mode)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaThreadExit() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadExit") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadExit() -> cudaError_t;
        }
        cudaThreadExit()
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaFuncCache) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadGetCacheConfig") });
        _f(pCacheConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
        }
        cudaThreadGetCacheConfig(pCacheConfig)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, cudaLimit) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadGetLimit") });
        _f(pValue, limit)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
        }
        cudaThreadGetLimit(pValue, limit)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaFuncCache) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadSetCacheConfig") });
        _f(cacheConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
        }
        cudaThreadSetCacheConfig(cacheConfig)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaLimit, usize) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadSetLimit") });
        _f(limit, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;
        }
        cudaThreadSetLimit(limit, value)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaThreadSynchronize() -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaThreadSynchronize") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaThreadSynchronize() -> cudaError_t;
        }
        cudaThreadSynchronize()
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cudaUnbindTexture(texref: *const textureReference) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const textureReference) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaUnbindTexture") });
        _f(texref)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaUnbindTexture(texref: *const textureReference) -> cudaError_t;
        }
        cudaUnbindTexture(texref)
    }
}
pub unsafe fn cudaUserObjectCreate(object_out: *mut cudaUserObject_t, ptr: *mut ::core::ffi::c_void, destroy: cudaHostFn_t, initialRefcount: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cudaUserObject_t, *mut ::core::ffi::c_void, cudaHostFn_t, ::core::ffi::c_uint, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaUserObjectCreate") });
        _f(object_out, ptr, destroy, initialRefcount, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaUserObjectCreate(object_out: *mut cudaUserObject_t, ptr: *mut ::core::ffi::c_void, destroy: cudaHostFn_t, initialRefcount: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)
    }
}
pub unsafe fn cudaUserObjectRelease(object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaUserObject_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaUserObjectRelease") });
        _f(object, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaUserObjectRelease(object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaUserObjectRelease(object, count)
    }
}
pub unsafe fn cudaUserObjectRetain(object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(cudaUserObject_t, ::core::ffi::c_uint) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaUserObjectRetain") });
        _f(object, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaUserObjectRetain(object: cudaUserObject_t, count: ::core::ffi::c_uint) -> cudaError_t;
        }
        cudaUserObjectRetain(object, count)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cudaWaitExternalSemaphoresAsync(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreWaitParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaExternalSemaphore_t, *const cudaExternalSemaphoreWaitParams, ::core::ffi::c_uint, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaWaitExternalSemaphoresAsync") });
        _f(extSemArray, paramsArray, numExtSems, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaWaitExternalSemaphoresAsync(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreWaitParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t;
        }
        cudaWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cudaWaitExternalSemaphoresAsync_v2(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreWaitParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const cudaExternalSemaphore_t, *const cudaExternalSemaphoreWaitParams, ::core::ffi::c_uint, cudaStream_t) -> cudaError_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cudaWaitExternalSemaphoresAsync_v2") });
        _f(extSemArray, paramsArray, numExtSems, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cudaWaitExternalSemaphoresAsync_v2(extSemArray: *const cudaExternalSemaphore_t, paramsArray: *const cudaExternalSemaphoreWaitParams, numExtSems: ::core::ffi::c_uint, stream: cudaStream_t) -> cudaError_t;
        }
        cudaWaitExternalSemaphoresAsync_v2(extSemArray, paramsArray, numExtSems, stream)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cudart"];
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
        let lib_names = std::vec!["cudart"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
