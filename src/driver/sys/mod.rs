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
pub use self::cudaError_enum as CUresult;
pub use self::CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum as CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS;
pub use self::CUGPUDirectRDMAWritesOrdering_enum as CUGPUDirectRDMAWritesOrdering;
pub use self::CUaccessProperty_enum as CUaccessProperty;
pub use self::CUaddress_mode_enum as CUaddress_mode;
pub use self::CUarraySparseSubresourceType_enum as CUarraySparseSubresourceType;
pub use self::CUarray_cubemap_face_enum as CUarray_cubemap_face;
pub use self::CUarray_format_enum as CUarray_format;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUasyncNotificationType_enum as CUasyncNotificationType;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUatomicOperationCapability_enum as CUatomicOperationCapability;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUatomicOperation_enum as CUatomicOperation;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUcigDataType_enum as CUcigDataType;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUclusterSchedulingPolicy_enum as CUclusterSchedulingPolicy;
pub use self::CUcomputemode_enum as CUcomputemode;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUcoredumpSettings_enum as CUcoredumpSettings;
pub use self::CUctx_flags_enum as CUctx_flags;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUdeviceNumaConfig_enum as CUdeviceNumaConfig;
pub use self::CUdevice_P2PAttribute_enum as CUdevice_P2PAttribute;
pub use self::CUdevice_attribute_enum as CUdevice_attribute;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUdriverProcAddressQueryResult_enum as CUdriverProcAddressQueryResult;
pub use self::CUdriverProcAddress_flags_enum as CUdriverProcAddress_flags;
pub use self::CUevent_flags_enum as CUevent_flags;
pub use self::CUevent_record_flags_enum as CUevent_record_flags;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUevent_sched_flags_enum as CUevent_sched_flags;
pub use self::CUevent_wait_flags_enum as CUevent_wait_flags;
pub use self::CUexecAffinityType_enum as CUexecAffinityType;
pub use self::CUexternalMemoryHandleType_enum as CUexternalMemoryHandleType;
pub use self::CUexternalSemaphoreHandleType_enum as CUexternalSemaphoreHandleType;
pub use self::CUfilter_mode_enum as CUfilter_mode;
pub use self::CUflushGPUDirectRDMAWritesOptions_enum as CUflushGPUDirectRDMAWritesOptions;
pub use self::CUflushGPUDirectRDMAWritesScope_enum as CUflushGPUDirectRDMAWritesScope;
pub use self::CUflushGPUDirectRDMAWritesTarget_enum as CUflushGPUDirectRDMAWritesTarget;
pub use self::CUfunc_cache_enum as CUfunc_cache;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUfunctionLoadingState_enum as CUfunctionLoadingState;
pub use self::CUfunction_attribute_enum as CUfunction_attribute;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUgraphChildGraphNodeOwnership_enum as CUgraphChildGraphNodeOwnership;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUgraphConditionalNodeType_enum as CUgraphConditionalNodeType;
pub use self::CUgraphDebugDot_flags_enum as CUgraphDebugDot_flags;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUgraphDependencyType_enum as CUgraphDependencyType;
pub use self::CUgraphExecUpdateResult_enum as CUgraphExecUpdateResult;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUgraphInstantiateResult_enum as CUgraphInstantiateResult;
pub use self::CUgraphInstantiate_flags_enum as CUgraphInstantiate_flags;
pub use self::CUgraphMem_attribute_enum as CUgraphMem_attribute;
pub use self::CUgraphNodeType_enum as CUgraphNodeType;
#[cfg(any(feature = "cuda-13030"))]
pub use self::CUgraphRecaptureStatus_enum as CUgraphRecaptureStatus;
pub use self::CUgraphicsMapResourceFlags_enum as CUgraphicsMapResourceFlags;
pub use self::CUgraphicsRegisterFlags_enum as CUgraphicsRegisterFlags;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUhostTaskSyncMode_enum as CUhostTaskSyncMode;
pub use self::CUipcMem_flags_enum as CUipcMem_flags;
pub use self::CUjitInputType_enum as CUjitInputType;
pub use self::CUjit_cacheMode_enum as CUjit_cacheMode;
pub use self::CUjit_fallback_enum as CUjit_fallback;
pub use self::CUjit_option_enum as CUjit_option;
pub use self::CUjit_target_enum as CUjit_target;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub use self::CUkernelNodeAttrID_enum as CUkernelNodeAttrID;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlaunchAttributeID as CUkernelNodeAttrID;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlaunchAttributeID as CUstreamAttrID;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlaunchAttributeID_enum as CUlaunchAttributeID;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlaunchAttributePortableClusterMode_enum as CUlaunchAttributePortableClusterMode;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlaunchMemSyncDomain_enum as CUlaunchMemSyncDomain;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlibraryOption_enum as CUlibraryOption;
pub use self::CUlimit_enum as CUlimit;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUlogLevel_enum as CUlogLevel;
#[cfg(any(feature = "cuda-13030"))]
pub use self::CUlogicalEndpointFlag_enum as CUlogicalEndpointFlag;
#[cfg(any(feature = "cuda-13030"))]
pub use self::CUlogicalEndpointIpcHandleType_enum as CUlogicalEndpointIpcHandleType;
#[cfg(any(feature = "cuda-13030"))]
pub use self::CUlogicalEndpointType_enum as CUlogicalEndpointType;
pub use self::CUmemAccess_flags_enum as CUmemAccess_flags;
pub use self::CUmemAllocationCompType_enum as CUmemAllocationCompType;
pub use self::CUmemAllocationGranularity_flags_enum as CUmemAllocationGranularity_flags;
pub use self::CUmemAllocationHandleType_enum as CUmemAllocationHandleType;
pub use self::CUmemAllocationType_enum as CUmemAllocationType;
pub use self::CUmemAttach_flags_enum as CUmemAttach_flags;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmemDecompressAlgorithm_enum as CUmemDecompressAlgorithm;
pub use self::CUmemHandleType_enum as CUmemHandleType;
pub use self::CUmemLocationType_enum as CUmemLocationType;
pub use self::CUmemOperationType_enum as CUmemOperationType;
pub use self::CUmemPool_attribute_enum as CUmemPool_attribute;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmemRangeFlags_enum as CUmemRangeFlags;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmemRangeHandleType_enum as CUmemRangeHandleType;
pub use self::CUmem_advise_enum as CUmem_advise;
pub use self::CUmem_range_attribute_enum as CUmem_range_attribute;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmemcpy3DOperandType_enum as CUmemcpy3DOperandType;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmemcpyFlags_enum as CUmemcpyFlags;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmemcpySrcAccessOrder_enum as CUmemcpySrcAccessOrder;
pub use self::CUmemorytype_enum as CUmemorytype;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmoduleLoadingMode_enum as CUmoduleLoadingMode;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUmulticastGranularity_flags_enum as CUmulticastGranularity_flags;
pub use self::CUoccupancy_flags_enum as CUoccupancy_flags;
pub use self::CUoutput_mode_enum as CUoutput_mode;
pub use self::CUpointer_attribute_enum as CUpointer_attribute;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUprocessState_enum as CUprocessState;
pub use self::CUresourceViewFormat_enum as CUresourceViewFormat;
pub use self::CUresourcetype_enum as CUresourcetype;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUsharedMemoryMode_enum as CUsharedMemoryMode;
pub use self::CUshared_carveout_enum as CUshared_carveout;
pub use self::CUsharedconfig_enum as CUsharedconfig;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUstreamAtomicReductionDataType_enum as CUstreamAtomicReductionDataType;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUstreamAtomicReductionOpType_enum as CUstreamAtomicReductionOpType;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub use self::CUstreamAttrID_enum as CUstreamAttrID;
pub use self::CUstreamBatchMemOpType_enum as CUstreamBatchMemOpType;
pub use self::CUstreamCaptureMode_enum as CUstreamCaptureMode;
pub use self::CUstreamCaptureStatus_enum as CUstreamCaptureStatus;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUstreamCigDataType_enum as CUstreamCigDataType;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUstreamMemoryBarrier_flags_enum as CUstreamMemoryBarrier_flags;
pub use self::CUstreamUpdateCaptureDependencies_flags_enum as CUstreamUpdateCaptureDependencies_flags;
pub use self::CUstreamWaitValue_flags_enum as CUstreamWaitValue_flags;
pub use self::CUstreamWriteValue_flags_enum as CUstreamWriteValue_flags;
pub use self::CUstream_flags_enum as CUstream_flags;
pub use self::CUsynchronizationPolicy_enum as CUsynchronizationPolicy;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUtensorMapDataType_enum as CUtensorMapDataType;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUtensorMapFloatOOBfill_enum as CUtensorMapFloatOOBfill;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUtensorMapIm2ColWideMode_enum as CUtensorMapIm2ColWideMode;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUtensorMapInterleave_enum as CUtensorMapInterleave;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUtensorMapL2promotion_enum as CUtensorMapL2promotion;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub use self::CUtensorMapSwizzle_enum as CUtensorMapSwizzle;
pub use self::CUuserObjectRetain_flags_enum as CUuserObjectRetain_flags;
pub use self::CUuserObject_flags_enum as CUuserObject_flags;
pub const CUDA_ARRAY3D_2DARRAY: u32 = 1;
pub const CUDA_ARRAY3D_COLOR_ATTACHMENT: u32 = 32;
pub const CUDA_ARRAY3D_CUBEMAP: u32 = 4;
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CUDA_ARRAY3D_DEFERRED_MAPPING: u32 = 128;
pub const CUDA_ARRAY3D_DEPTH_TEXTURE: u32 = 16;
pub const CUDA_ARRAY3D_LAYERED: u32 = 1;
pub const CUDA_ARRAY3D_SPARSE: u32 = 64;
pub const CUDA_ARRAY3D_SURFACE_LDST: u32 = 2;
pub const CUDA_ARRAY3D_TEXTURE_GATHER: u32 = 8;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CUDA_ARRAY3D_VIDEO_ENCODE_DECODE: u32 = 256;
pub const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC: u32 = 2;
pub const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC: u32 = 1;
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 1;
pub const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC: u32 = 1;
pub const CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC: u32 = 2;
pub const CUDA_NVSCISYNC_ATTR_SIGNAL: u32 = 1;
pub const CUDA_NVSCISYNC_ATTR_WAIT: u32 = 2;
#[cfg(any(feature = "cuda-11040"))]
pub const CUDA_VERSION: u32 = 11040;
#[cfg(any(feature = "cuda-11050"))]
pub const CUDA_VERSION: u32 = 11050;
#[cfg(any(feature = "cuda-11060"))]
pub const CUDA_VERSION: u32 = 11060;
#[cfg(any(feature = "cuda-11070"))]
pub const CUDA_VERSION: u32 = 11070;
#[cfg(any(feature = "cuda-11080"))]
pub const CUDA_VERSION: u32 = 11080;
#[cfg(any(feature = "cuda-12000"))]
pub const CUDA_VERSION: u32 = 12000;
#[cfg(any(feature = "cuda-12010"))]
pub const CUDA_VERSION: u32 = 12010;
#[cfg(any(feature = "cuda-12020"))]
pub const CUDA_VERSION: u32 = 12020;
#[cfg(any(feature = "cuda-12030"))]
pub const CUDA_VERSION: u32 = 12030;
#[cfg(any(feature = "cuda-12040"))]
pub const CUDA_VERSION: u32 = 12040;
#[cfg(any(feature = "cuda-12050"))]
pub const CUDA_VERSION: u32 = 12050;
#[cfg(any(feature = "cuda-12060"))]
pub const CUDA_VERSION: u32 = 12060;
#[cfg(any(feature = "cuda-12080"))]
pub const CUDA_VERSION: u32 = 12080;
#[cfg(any(feature = "cuda-12090"))]
pub const CUDA_VERSION: u32 = 12090;
#[cfg(any(feature = "cuda-13000"))]
pub const CUDA_VERSION: u32 = 13000;
#[cfg(any(feature = "cuda-13010"))]
pub const CUDA_VERSION: u32 = 13010;
#[cfg(any(feature = "cuda-13020"))]
pub const CUDA_VERSION: u32 = 13020;
#[cfg(any(feature = "cuda-13030"))]
pub const CUDA_VERSION: u32 = 13030;
pub const CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL: u32 = 1;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_COMPUTE_ACCELERATED_TARGET_BASE: u32 = 65536;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_COMPUTE_FAMILY_TARGET_BASE: u32 = 131072;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_GRAPH_COND_ASSIGN_DEFAULT: u32 = 1;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_GRAPH_KERNEL_NODE_PORT_DEFAULT: u32 = 0;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER: u32 = 2;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC: u32 = 1;
pub const CU_IPC_HANDLE_SIZE: u32 = 64;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_LAUNCH_KERNEL_REQUIRED_BLOCK_DIM: u32 = 1;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT: u32 = 1;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT: u32 = 2;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_LAUNCH_PARAM_END_AS_INT: u32 = 0;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 4;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 4;
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;
pub const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 8;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_MEM_CREATE_USAGE_HW_DECOMPRESS: u32 = 2;
pub const CU_MEM_CREATE_USAGE_TILE_POOL: u32 = 1;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS: u32 = 2;
pub const CU_PARAM_TR_DEFAULT: i32 = -1;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_TENSOR_MAP_NUM_QWORDS: u32 = 16;
pub const CU_TRSA_OVERRIDE_FORMAT: u32 = 1;
pub const CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION: u32 = 32;
pub const CU_TRSF_NORMALIZED_COORDINATES: u32 = 2;
pub const CU_TRSF_READ_AS_INTEGER: u32 = 1;
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub const CU_TRSF_SEAMLESS_CUBEMAP: u32 = 64;
pub const CU_TRSF_SRGB: u32 = 16;
pub type CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_v2;
pub type CUDA_ARRAY3D_DESCRIPTOR_v2 = CUDA_ARRAY3D_DESCRIPTOR_st;
pub type CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_v2;
pub type CUDA_ARRAY_DESCRIPTOR_v2 = CUDA_ARRAY_DESCRIPTOR_st;
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_ARRAY_MEMORY_REQUIREMENTS = CUDA_ARRAY_MEMORY_REQUIREMENTS_v1;
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = CUDA_ARRAY_MEMORY_REQUIREMENTS_st;
pub type CUDA_ARRAY_SPARSE_PROPERTIES = CUDA_ARRAY_SPARSE_PROPERTIES_v1;
pub type CUDA_ARRAY_SPARSE_PROPERTIES_v1 = CUDA_ARRAY_SPARSE_PROPERTIES_st;
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_v1;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS_v1 = CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 = CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_CHILD_GRAPH_NODE_PARAMS = CUDA_CHILD_GRAPH_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_EVENT_RECORD_NODE_PARAMS = CUDA_EVENT_RECORD_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_EVENT_WAIT_NODE_PARAMS = CUDA_EVENT_WAIT_NODE_PARAMS_st;
pub type CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1;
pub type CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
pub type CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1;
pub type CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st;
pub type CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1;
pub type CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
pub type CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1;
pub type CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st;
pub type CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1;
pub type CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st;
pub type CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1;
pub type CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st;
pub type CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1;
pub type CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st;
pub type CUDA_EXT_SEM_WAIT_NODE_PARAMS = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1;
pub type CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = CUDA_EXT_SEM_WAIT_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_GRAPH_INSTANTIATE_PARAMS = CUDA_GRAPH_INSTANTIATE_PARAMS_st;
pub type CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_v1;
pub type CUDA_HOST_NODE_PARAMS_v1 = CUDA_HOST_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_HOST_NODE_PARAMS_v2 = CUDA_HOST_NODE_PARAMS_v2_st;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub type CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v1;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v2;
pub type CUDA_KERNEL_NODE_PARAMS_v1 = CUDA_KERNEL_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_KERNEL_NODE_PARAMS_v2 = CUDA_KERNEL_NODE_PARAMS_v2_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_KERNEL_NODE_PARAMS_v3 = CUDA_KERNEL_NODE_PARAMS_v3_st;
pub type CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_v1;
pub type CUDA_LAUNCH_PARAMS_v1 = CUDA_LAUNCH_PARAMS_st;
pub type CUDA_MEMCPY2D = CUDA_MEMCPY2D_v2;
pub type CUDA_MEMCPY2D_v2 = CUDA_MEMCPY2D_st;
pub type CUDA_MEMCPY3D = CUDA_MEMCPY3D_v2;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEMCPY3D_BATCH_OP = CUDA_MEMCPY3D_BATCH_OP_v1;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEMCPY3D_BATCH_OP_v1 = CUDA_MEMCPY3D_BATCH_OP_st;
pub type CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_v1;
pub type CUDA_MEMCPY3D_PEER_v1 = CUDA_MEMCPY3D_PEER_st;
pub type CUDA_MEMCPY3D_v2 = CUDA_MEMCPY3D_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEMCPY_NODE_PARAMS = CUDA_MEMCPY_NODE_PARAMS_st;
pub type CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_v1;
pub type CUDA_MEMSET_NODE_PARAMS_v1 = CUDA_MEMSET_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEMSET_NODE_PARAMS_v2 = CUDA_MEMSET_NODE_PARAMS_v2_st;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
pub type CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_v1;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEM_ALLOC_NODE_PARAMS_v1 = CUDA_MEM_ALLOC_NODE_PARAMS_v1_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEM_ALLOC_NODE_PARAMS_v2 = CUDA_MEM_ALLOC_NODE_PARAMS_v2_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUDA_MEM_FREE_NODE_PARAMS = CUDA_MEM_FREE_NODE_PARAMS_st;
pub type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1;
pub type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
pub type CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_v1;
pub type CUDA_RESOURCE_DESC_v1 = CUDA_RESOURCE_DESC_st;
pub type CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_v1;
pub type CUDA_RESOURCE_VIEW_DESC_v1 = CUDA_RESOURCE_VIEW_DESC_st;
pub type CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_v1;
pub type CUDA_TEXTURE_DESC_v1 = CUDA_TEXTURE_DESC_st;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CU_DEV_SM_RESOURCE_GROUP_PARAMS = CU_DEV_SM_RESOURCE_GROUP_PARAMS_st;
pub type CUaccessPolicyWindow = CUaccessPolicyWindow_v1;
pub type CUaccessPolicyWindow_v1 = CUaccessPolicyWindow_st;
pub type CUarray = *mut CUarray_st;
pub type CUarrayMapInfo = CUarrayMapInfo_v1;
pub type CUarrayMapInfo_v1 = CUarrayMapInfo_st;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUasyncCallback = ::core::option::Option<unsafe extern "C" fn(info: *mut CUasyncNotificationInfo, userData: *mut ::core::ffi::c_void, callback: CUasyncCallbackHandle)>;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUasyncCallbackHandle = *mut CUasyncCallbackEntry_st;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUasyncNotificationInfo = CUasyncNotificationInfo_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcheckpointCheckpointArgs = CUcheckpointCheckpointArgs_st;
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcheckpointGpuPair = CUcheckpointGpuPair_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcheckpointLockArgs = CUcheckpointLockArgs_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcheckpointRestoreArgs = CUcheckpointRestoreArgs_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcheckpointUnlockArgs = CUcheckpointUnlockArgs_st;
pub type CUcontext = *mut CUctx_st;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcoredumpCallbackHandle = *mut CUcoredumpCallbackEntry_st;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUcoredumpStatusCallback = ::core::option::Option<unsafe extern "C" fn(userData: *mut ::core::ffi::c_void, pid: ::core::ffi::c_int, dev: CUdevice)>;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUctxCigParam = CUctxCigParam_st;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUctxCreateParams = CUctxCreateParams_st;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUdevResource = CUdevResource_v1;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUdevResourceDesc = *mut CUdevResourceDesc_st;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUdevResource_v1 = CUdevResource_st;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUdevSmResource = CUdevSmResource_st;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUdevWorkqueueConfigResource = CUdevWorkqueueConfigResource_st;
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUdevWorkqueueResource = CUdevWorkqueueResource_st;
pub type CUdevice = CUdevice_v1;
pub type CUdevice_v1 = ::core::ffi::c_int;
pub type CUdeviceptr = CUdeviceptr_v2;
pub type CUdeviceptr_v2 = ::core::ffi::c_ulonglong;
pub type CUdevprop = CUdevprop_v1;
pub type CUdevprop_v1 = CUdevprop_st;
pub type CUevent = *mut CUevent_st;
pub type CUexecAffinityParam = CUexecAffinityParam_v1;
pub type CUexecAffinityParam_v1 = CUexecAffinityParam_st;
pub type CUexecAffinitySmCount = CUexecAffinitySmCount_v1;
pub type CUexecAffinitySmCount_v1 = CUexecAffinitySmCount_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUextent3D = CUextent3D_v1;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUextent3D_v1 = CUextent3D_st;
pub type CUexternalMemory = *mut CUextMemory_st;
pub type CUexternalSemaphore = *mut CUextSemaphore_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUgraph = *mut CUgraph_st;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgraphConditionalHandle = cuuint64_t;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgraphDeviceNode = *mut CUgraphDeviceUpdatableNode_st;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgraphEdgeData = CUgraphEdgeData_st;
pub type CUgraphExec = *mut CUgraphExec_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgraphExecUpdateResultInfo = CUgraphExecUpdateResultInfo_v1;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgraphExecUpdateResultInfo_v1 = CUgraphExecUpdateResultInfo_st;
pub type CUgraphNode = *mut CUgraphNode_st;
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgraphNodeParams = CUgraphNodeParams_st;
#[cfg(any(feature = "cuda-13030"))]
pub type CUgraphRecaptureCallback = ::core::option::Option<unsafe extern "C" fn(data: *mut ::core::ffi::c_void, node: CUgraphNode, originalParams: *const CUgraphNodeParams, recaptureParams: *const CUgraphNodeParams, status: CUgraphRecaptureStatus) -> CUresult>;
pub type CUgraphicsResource = *mut CUgraphicsResource_st;
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUgreenCtx = *mut CUgreenCtx_st;
pub type CUhostFn = ::core::option::Option<unsafe extern "C" fn(userData: *mut ::core::ffi::c_void)>;
pub type CUipcEventHandle = CUipcEventHandle_v1;
pub type CUipcEventHandle_v1 = CUipcEventHandle_st;
pub type CUipcMemHandle = CUipcMemHandle_v1;
pub type CUipcMemHandle_v1 = CUipcMemHandle_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUkernel = *mut CUkern_st;
pub type CUkernelNodeAttrValue = CUkernelNodeAttrValue_v1;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub type CUkernelNodeAttrValue_v1 = CUkernelNodeAttrValue_union;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUkernelNodeAttrValue_v1 = CUlaunchAttributeValue;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlaunchAttribute = CUlaunchAttribute_st;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlaunchAttributeValue = CUlaunchAttributeValue_union;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlaunchConfig = CUlaunchConfig_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlaunchMemSyncDomainMap = CUlaunchMemSyncDomainMap_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlibrary = *mut CUlib_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlibraryHostUniversalFunctionAndDataTable = CUlibraryHostUniversalFunctionAndDataTable_st;
pub type CUlinkState = *mut CUlinkState_st;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlogIterator = ::core::ffi::c_uint;
#[cfg(any(feature = "cuda-13030"))]
pub type CUlogicalEndpointFabricHandle = CUlogicalEndpointFabricHandle_st;
#[cfg(any(feature = "cuda-13030"))]
pub type CUlogicalEndpointId = cuuint32_t;
#[cfg(any(feature = "cuda-13030"))]
pub type CUlogicalEndpointProp = CUlogicalEndpointProp_struct;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlogsCallback = ::core::option::Option<unsafe extern "C" fn(data: *mut ::core::ffi::c_void, logLevel: CUlogLevel, message: *mut ::core::ffi::c_char, length: usize)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUlogsCallbackHandle = *mut CUlogsCallbackEntry_st;
pub type CUmemAccessDesc = CUmemAccessDesc_v1;
pub type CUmemAccessDesc_v1 = CUmemAccessDesc_st;
pub type CUmemAllocationProp = CUmemAllocationProp_v1;
pub type CUmemAllocationProp_v1 = CUmemAllocationProp_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemDecompressParams = CUmemDecompressParams_st;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemFabricHandle = CUmemFabricHandle_v1;
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemFabricHandle_v1 = CUmemFabricHandle_st;
pub type CUmemGenericAllocationHandle = CUmemGenericAllocationHandle_v1;
pub type CUmemGenericAllocationHandle_v1 = ::core::ffi::c_ulonglong;
pub type CUmemLocation = CUmemLocation_v1;
pub type CUmemLocation_v1 = CUmemLocation_st;
pub type CUmemPoolProps = CUmemPoolProps_v1;
pub type CUmemPoolProps_v1 = CUmemPoolProps_st;
pub type CUmemPoolPtrExportData = CUmemPoolPtrExportData_v1;
pub type CUmemPoolPtrExportData_v1 = CUmemPoolPtrExportData_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemcpy3DOperand = CUmemcpy3DOperand_v1;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemcpy3DOperand_v1 = CUmemcpy3DOperand_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemcpyAttributes = CUmemcpyAttributes_v1;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmemcpyAttributes_v1 = CUmemcpyAttributes_st;
pub type CUmemoryPool = *mut CUmemPoolHandle_st;
pub type CUmipmappedArray = *mut CUmipmappedArray_st;
pub type CUmodule = *mut CUmod_st;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmulticastObjectProp = CUmulticastObjectProp_v1;
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUmulticastObjectProp_v1 = CUmulticastObjectProp_st;
pub type CUoccupancyB2DSize = ::core::option::Option<unsafe extern "C" fn(blockSize: ::core::ffi::c_int) -> usize>;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUoffset3D = CUoffset3D_v1;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUoffset3D_v1 = CUoffset3D_st;
pub type CUstream = *mut CUstream_st;
pub type CUstreamAttrValue = CUstreamAttrValue_v1;
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
pub type CUstreamAttrValue_v1 = CUstreamAttrValue_union;
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUstreamAttrValue_v1 = CUlaunchAttributeValue;
pub type CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_v1;
pub type CUstreamBatchMemOpParams_v1 = CUstreamBatchMemOpParams_union;
pub type CUstreamCallback = ::core::option::Option<unsafe extern "C" fn(hStream: CUstream, status: CUresult, userData: *mut ::core::ffi::c_void)>;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUstreamCigCaptureParams = CUstreamCigCaptureParams_st;
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUstreamCigParam = CUstreamCigParam_st;
pub type CUsurfObject = CUsurfObject_v1;
pub type CUsurfObject_v1 = ::core::ffi::c_ulonglong;
pub type CUsurfref = *mut CUsurfref_st;
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub type CUtensorMap = CUtensorMap_st;
pub type CUtexObject = CUtexObject_v1;
pub type CUtexObject_v1 = ::core::ffi::c_ulonglong;
pub type CUtexref = *mut CUtexref_st;
pub type CUuserObject = *mut CUuserObject_st;
pub type CUuuid = CUuuid_st;
pub type cuuint32_t = u32;
pub type cuuint64_t = u64;
#[cfg(any(feature = "cuda-12050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUCoredumpGenerationFlags {
    CU_COREDUMP_DEFAULT_FLAGS = 0,
    CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = 1,
    CU_COREDUMP_SKIP_GLOBAL_MEMORY = 2,
    CU_COREDUMP_SKIP_SHARED_MEMORY = 4,
    CU_COREDUMP_SKIP_LOCAL_MEMORY = 8,
    CU_COREDUMP_SKIP_ABORT = 16,
    CU_COREDUMP_LIGHTWEIGHT_FLAGS = 15,
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUCoredumpGenerationFlags {
    CU_COREDUMP_DEFAULT_FLAGS = 0,
    CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = 1,
    CU_COREDUMP_SKIP_GLOBAL_MEMORY = 2,
    CU_COREDUMP_SKIP_SHARED_MEMORY = 4,
    CU_COREDUMP_SKIP_LOCAL_MEMORY = 8,
    CU_COREDUMP_SKIP_ABORT = 16,
    CU_COREDUMP_SKIP_CONSTBANK_MEMORY = 32,
    CU_COREDUMP_LIGHTWEIGHT_FLAGS = 47,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUCoredumpGenerationFlags {
    CU_COREDUMP_DEFAULT_FLAGS = 0,
    CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = 1,
    CU_COREDUMP_SKIP_GLOBAL_MEMORY = 2,
    CU_COREDUMP_SKIP_SHARED_MEMORY = 4,
    CU_COREDUMP_SKIP_LOCAL_MEMORY = 8,
    CU_COREDUMP_SKIP_ABORT = 16,
    CU_COREDUMP_SKIP_CONSTBANK_MEMORY = 32,
    CU_COREDUMP_GZIP_COMPRESS = 64,
    CU_COREDUMP_LIGHTWEIGHT_FLAGS = 47,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUCoredumpGenerationFlags {
    CU_COREDUMP_DEFAULT_FLAGS = 0,
    CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = 1,
    CU_COREDUMP_SKIP_GLOBAL_MEMORY = 2,
    CU_COREDUMP_SKIP_SHARED_MEMORY = 4,
    CU_COREDUMP_SKIP_LOCAL_MEMORY = 8,
    CU_COREDUMP_SKIP_ABORT = 16,
    CU_COREDUMP_SKIP_CONSTBANK_MEMORY = 32,
    CU_COREDUMP_GZIP_COMPRESS = 64,
    CU_COREDUMP_FAULTED_CONTEXTS_ONLY = 128,
    CU_COREDUMP_NO_ERRBAR_AT_EXIT = 1073741824,
    CU_COREDUMP_LOG_ONLY = -2147483648,
    CU_COREDUMP_LIGHTWEIGHT_FLAGS = 47,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum {
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 1,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUGPUDirectRDMAWritesOrdering_enum {
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0,
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100,
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUaccessProperty_enum {
    CU_ACCESS_PROPERTY_NORMAL = 0,
    CU_ACCESS_PROPERTY_STREAMING = 1,
    CU_ACCESS_PROPERTY_PERSISTING = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0,
    CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2,
    CU_TR_ADDRESS_MODE_BORDER = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarraySparseSubresourceType_enum {
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_cubemap_face_enum {
    CU_CUBEMAP_FACE_POSITIVE_X = 0,
    CU_CUBEMAP_FACE_NEGATIVE_X = 1,
    CU_CUBEMAP_FACE_POSITIVE_Y = 2,
    CU_CUBEMAP_FACE_NEGATIVE_Y = 3,
    CU_CUBEMAP_FACE_POSITIVE_Z = 4,
    CU_CUBEMAP_FACE_NEGATIVE_Z = 5,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
    CU_AD_FORMAT_P010 = 159,
    CU_AD_FORMAT_P016 = 161,
    CU_AD_FORMAT_NV16 = 162,
    CU_AD_FORMAT_P210 = 163,
    CU_AD_FORMAT_P216 = 164,
    CU_AD_FORMAT_YUY2 = 165,
    CU_AD_FORMAT_Y210 = 166,
    CU_AD_FORMAT_Y216 = 167,
    CU_AD_FORMAT_AYUV = 168,
    CU_AD_FORMAT_Y410 = 169,
    CU_AD_FORMAT_Y416 = 177,
    CU_AD_FORMAT_Y444_PLANAR8 = 178,
    CU_AD_FORMAT_Y444_PLANAR10 = 179,
    CU_AD_FORMAT_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
    CU_AD_FORMAT_P010 = 159,
    CU_AD_FORMAT_P016 = 161,
    CU_AD_FORMAT_NV16 = 162,
    CU_AD_FORMAT_P210 = 163,
    CU_AD_FORMAT_P216 = 164,
    CU_AD_FORMAT_YUY2 = 165,
    CU_AD_FORMAT_Y210 = 166,
    CU_AD_FORMAT_Y216 = 167,
    CU_AD_FORMAT_AYUV = 168,
    CU_AD_FORMAT_Y410 = 169,
    CU_AD_FORMAT_Y416 = 177,
    CU_AD_FORMAT_Y444_PLANAR8 = 178,
    CU_AD_FORMAT_Y444_PLANAR10 = 179,
    CU_AD_FORMAT_YUV444_8bit_SemiPlanar = 180,
    CU_AD_FORMAT_YUV444_16bit_SemiPlanar = 181,
    CU_AD_FORMAT_UNORM_INT_101010_2 = 80,
    CU_AD_FORMAT_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
    CU_AD_FORMAT_P010 = 159,
    CU_AD_FORMAT_P016 = 161,
    CU_AD_FORMAT_NV16 = 162,
    CU_AD_FORMAT_P210 = 163,
    CU_AD_FORMAT_P216 = 164,
    CU_AD_FORMAT_YUY2 = 165,
    CU_AD_FORMAT_Y210 = 166,
    CU_AD_FORMAT_Y216 = 167,
    CU_AD_FORMAT_AYUV = 168,
    CU_AD_FORMAT_Y410 = 169,
    CU_AD_FORMAT_Y416 = 177,
    CU_AD_FORMAT_Y444_PLANAR8 = 178,
    CU_AD_FORMAT_Y444_PLANAR10 = 179,
    CU_AD_FORMAT_YUV444_8bit_SemiPlanar = 180,
    CU_AD_FORMAT_YUV444_16bit_SemiPlanar = 181,
    CU_AD_FORMAT_UNORM_INT_101010_2 = 80,
    CU_AD_FORMAT_UINT8_PACKED_422 = 81,
    CU_AD_FORMAT_UINT8_PACKED_444 = 82,
    CU_AD_FORMAT_UINT8_SEMIPLANAR_420 = 83,
    CU_AD_FORMAT_UINT16_SEMIPLANAR_420 = 84,
    CU_AD_FORMAT_UINT8_SEMIPLANAR_422 = 85,
    CU_AD_FORMAT_UINT16_SEMIPLANAR_422 = 86,
    CU_AD_FORMAT_UINT8_SEMIPLANAR_444 = 87,
    CU_AD_FORMAT_UINT16_SEMIPLANAR_444 = 88,
    CU_AD_FORMAT_UINT8_PLANAR_420 = 89,
    CU_AD_FORMAT_UINT16_PLANAR_420 = 90,
    CU_AD_FORMAT_UINT8_PLANAR_422 = 91,
    CU_AD_FORMAT_UINT16_PLANAR_422 = 92,
    CU_AD_FORMAT_UINT8_PLANAR_444 = 93,
    CU_AD_FORMAT_UINT16_PLANAR_444 = 94,
    CU_AD_FORMAT_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUasyncNotificationType_enum {
    CU_ASYNC_NOTIFICATION_TYPE_OVER_BUDGET = 1,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUatomicOperationCapability_enum {
    CU_ATOMIC_CAPABILITY_SIGNED = 1,
    CU_ATOMIC_CAPABILITY_UNSIGNED = 2,
    CU_ATOMIC_CAPABILITY_REDUCTION = 4,
    CU_ATOMIC_CAPABILITY_SCALAR_32 = 8,
    CU_ATOMIC_CAPABILITY_SCALAR_64 = 16,
    CU_ATOMIC_CAPABILITY_SCALAR_128 = 32,
    CU_ATOMIC_CAPABILITY_VECTOR_32x4 = 64,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUatomicOperation_enum {
    CU_ATOMIC_OPERATION_INTEGER_ADD = 0,
    CU_ATOMIC_OPERATION_INTEGER_MIN = 1,
    CU_ATOMIC_OPERATION_INTEGER_MAX = 2,
    CU_ATOMIC_OPERATION_INTEGER_INCREMENT = 3,
    CU_ATOMIC_OPERATION_INTEGER_DECREMENT = 4,
    CU_ATOMIC_OPERATION_AND = 5,
    CU_ATOMIC_OPERATION_OR = 6,
    CU_ATOMIC_OPERATION_XOR = 7,
    CU_ATOMIC_OPERATION_EXCHANGE = 8,
    CU_ATOMIC_OPERATION_CAS = 9,
    CU_ATOMIC_OPERATION_FLOAT_ADD = 10,
    CU_ATOMIC_OPERATION_FLOAT_MIN = 11,
    CU_ATOMIC_OPERATION_FLOAT_MAX = 12,
    CU_ATOMIC_OPERATION_MAX = 13,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcigDataType_enum {
    CIG_DATA_TYPE_D3D12_COMMAND_QUEUE = 1,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcigDataType_enum {
    CIG_DATA_TYPE_D3D12_COMMAND_QUEUE = 1,
    CIG_DATA_TYPE_NV_BLOB = 2,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUclusterSchedulingPolicy_enum {
    CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0,
    CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
    CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcomputemode_enum {
    CU_COMPUTEMODE_DEFAULT = 0,
    CU_COMPUTEMODE_PROHIBITED = 2,
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcoredumpSettings_enum {
    CU_COREDUMP_ENABLE_ON_EXCEPTION = 1,
    CU_COREDUMP_TRIGGER_HOST = 2,
    CU_COREDUMP_LIGHTWEIGHT = 3,
    CU_COREDUMP_ENABLE_USER_TRIGGER = 4,
    CU_COREDUMP_FILE = 5,
    CU_COREDUMP_PIPE = 6,
    CU_COREDUMP_MAX = 7,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcoredumpSettings_enum {
    CU_COREDUMP_ENABLE_ON_EXCEPTION = 1,
    CU_COREDUMP_TRIGGER_HOST = 2,
    CU_COREDUMP_LIGHTWEIGHT = 3,
    CU_COREDUMP_ENABLE_USER_TRIGGER = 4,
    CU_COREDUMP_FILE = 5,
    CU_COREDUMP_PIPE = 6,
    CU_COREDUMP_GENERATION_FLAGS = 7,
    CU_COREDUMP_MAX = 8,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0,
    CU_CTX_SCHED_SPIN = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_BLOCKING_SYNC = 4,
    CU_CTX_SCHED_MASK = 7,
    CU_CTX_MAP_HOST = 8,
    CU_CTX_LMEM_RESIZE_TO_MAX = 16,
    CU_CTX_FLAGS_MASK = 31,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0,
    CU_CTX_SCHED_SPIN = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_BLOCKING_SYNC = 4,
    CU_CTX_SCHED_MASK = 7,
    CU_CTX_MAP_HOST = 8,
    CU_CTX_LMEM_RESIZE_TO_MAX = 16,
    CU_CTX_COREDUMP_ENABLE = 32,
    CU_CTX_USER_COREDUMP_ENABLE = 64,
    CU_CTX_SYNC_MEMOPS = 128,
    CU_CTX_FLAGS_MASK = 255,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevResourceType {
    CU_DEV_RESOURCE_TYPE_INVALID = 0,
    CU_DEV_RESOURCE_TYPE_SM = 1,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevResourceType {
    CU_DEV_RESOURCE_TYPE_INVALID = 0,
    CU_DEV_RESOURCE_TYPE_SM = 1,
    CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000,
    CU_DEV_RESOURCE_TYPE_WORKQUEUE = 10000,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevSmResourceGroup_flags {
    CU_DEV_SM_RESOURCE_GROUP_DEFAULT = 0,
    CU_DEV_SM_RESOURCE_GROUP_BACKFILL = 1,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevSmResourceSplitByCount_flags {
    CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING = 1,
    CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE = 2,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevSmResourceSplit_flags {
    CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING = 1,
    CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE = 2,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevWorkqueueConfigScope {
    CU_WORKQUEUE_SCOPE_DEVICE_CTX = 0,
    CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED = 1,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdeviceNumaConfig_enum {
    CU_DEVICE_NUMA_CONFIG_NONE = 0,
    CU_DEVICE_NUMA_CONFIG_NUMA_NODE = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_P2PAttribute_enum {
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2,
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_P2PAttribute_enum {
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2,
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4,
    CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED = 5,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_MAX = 120,
}
#[cfg(any(feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_MAX = 122,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_MAX = 125,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_MAX = 125,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_MAX = 130,
}
#[cfg(any(feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MAX = 133,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_MAX = 135,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_MAX = 135,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MAX = 136,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,
    CU_DEVICE_ATTRIBUTE_MAX = 144,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,
    CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED = 138,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 141,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED = 142,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,
    CU_DEVICE_ATTRIBUTE_MAX = 144,
}
#[cfg(any(feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,
    CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED = 138,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 141,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED = 142,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,
    CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED = 144,
    CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 145,
    CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED = 146,
    CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED = 147,
    CU_DEVICE_ATTRIBUTE_MAX = 148,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,
    CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED = 138,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 141,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED = 142,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,
    CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED = 144,
    CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 145,
    CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED = 146,
    CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED = 147,
    CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED = 148,
    CU_DEVICE_ATTRIBUTE_MAX = 149,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,
    CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED = 138,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 141,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED = 142,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,
    CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED = 144,
    CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 145,
    CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED = 146,
    CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED = 147,
    CU_DEVICE_ATTRIBUTE_ATOMIC_REDUCTION_SUPPORTED = 148,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_STREAMS_SUPPORTED = 151,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED = 152,
    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED = 153,
    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED = 154,
    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED = 155,
    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED = 156,
    CU_DEVICE_ATTRIBUTE_MAX = 157,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdriverProcAddress_flags_enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1,
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_flags_enum {
    CU_EVENT_DEFAULT = 0,
    CU_EVENT_BLOCKING_SYNC = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_EVENT_INTERPROCESS = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_record_flags_enum {
    CU_EVENT_RECORD_DEFAULT = 0,
    CU_EVENT_RECORD_EXTERNAL = 1,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_sched_flags_enum {
    CU_EVENT_SCHED_AUTO = 0,
    CU_EVENT_SCHED_SPIN = 1,
    CU_EVENT_SCHED_YIELD = 2,
    CU_EVENT_SCHED_BLOCKING_SYNC = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_wait_flags_enum {
    CU_EVENT_WAIT_DEFAULT = 0,
    CU_EVENT_WAIT_EXTERNAL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexecAffinityType_enum {
    CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0,
    CU_EXEC_AFFINITY_TYPE_MAX = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexternalMemoryHandleType_enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexternalMemoryHandleType_enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD = 9,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexternalSemaphoreHandleType_enum {
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfilter_mode_enum {
    CU_TR_FILTER_MODE_POINT = 0,
    CU_TR_FILTER_MODE_LINEAR = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUflushGPUDirectRDMAWritesOptions_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1,
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUflushGPUDirectRDMAWritesScope_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100,
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUflushGPUDirectRDMAWritesTarget_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunc_cache_enum {
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunctionLoadingState_enum {
    CU_FUNCTION_LOADING_STATE_UNLOADED = 0,
    CU_FUNCTION_LOADING_STATE_LOADED = 1,
    CU_FUNCTION_LOADING_STATE_MAX = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_MAX = 10,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
    CU_FUNC_ATTRIBUTE_MAX = 16,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
    CU_FUNC_ATTRIBUTE_DEVICE_NODE_UPDATE_SUPPORTED = 16,
    CU_FUNC_ATTRIBUTE_MAX = 17,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphChildGraphNodeOwnership_enum {
    CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE = 0,
    CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE = 1,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphChildGraphNodeOwnership_enum {
    CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE = 0,
    CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE = 1,
    CU_GRAPH_CHILD_GRAPH_OWNERSHIP_INVALID = -1,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphConditionalNodeType_enum {
    CU_GRAPH_COND_TYPE_IF = 0,
    CU_GRAPH_COND_TYPE_WHILE = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphConditionalNodeType_enum {
    CU_GRAPH_COND_TYPE_IF = 0,
    CU_GRAPH_COND_TYPE_WHILE = 1,
    CU_GRAPH_COND_TYPE_SWITCH = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = 16384,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = 16384,
    CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS = 32768,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDependencyType_enum {
    CU_GRAPH_DEPENDENCY_TYPE_DEFAULT = 0,
    CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiateResult_enum {
    CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
    CUDA_GRAPH_INSTANTIATE_ERROR = 1,
    CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
    CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
    CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiateResult_enum {
    CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
    CUDA_GRAPH_INSTANTIATE_ERROR = 1,
    CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
    CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
    CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
    CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED = 5,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiate_flags_enum {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiate_flags_enum {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiate_flags_enum {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
    CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
    CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphMem_attribute_enum {
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0,
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
    CU_GRAPH_NODE_TYPE_CONDITIONAL = 13,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
    CU_GRAPH_NODE_TYPE_CONDITIONAL = 13,
    CU_GRAPH_NODE_TYPE_RESERVED_16 = 16,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphRecaptureStatus_enum {
    CU_GRAPH_RECAPTURE_ELIGIBLE_FOR_UPDATE = 0,
    CU_GRAPH_RECAPTURE_INELIGIBLE_FOR_UPDATE = 1,
    CU_GRAPH_RECAPTURE_ERROR = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphicsMapResourceFlags_enum {
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphicsRegisterFlags_enum {
    CU_GRAPHICS_REGISTER_FLAGS_NONE = 0,
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1,
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2,
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgreenCtxCreate_flags {
    CU_GREEN_CTX_DEFAULT_STREAM = 1,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgreenCtxCreate_flags {
    CU_GREEN_CTX_NONE = 0,
    CU_GREEN_CTX_DEFAULT_STREAM = 1,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUhostTaskSyncMode_enum {
    CU_HOST_TASK_BLOCKING = 0,
    CU_HOST_TASK_SPINWAIT = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUipcMem_flags_enum {
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjitInputType_enum {
    CU_JIT_INPUT_CUBIN = 0,
    CU_JIT_INPUT_PTX = 1,
    CU_JIT_INPUT_FATBINARY = 2,
    CU_JIT_INPUT_OBJECT = 3,
    CU_JIT_INPUT_LIBRARY = 4,
    CU_JIT_INPUT_NVVM = 5,
    CU_JIT_NUM_INPUT_TYPES = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_cacheMode_enum {
    CU_JIT_CACHE_OPTION_NONE = 0,
    CU_JIT_CACHE_OPTION_CG = 1,
    CU_JIT_CACHE_OPTION_CA = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_fallback_enum {
    CU_PREFER_PTX = 0,
    CU_PREFER_BINARY = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_NUM_OPTIONS = 25,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_NUM_OPTIONS = 30,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_NUM_OPTIONS = 31,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_NUM_OPTIONS = 32,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_MAX_THREADS_PER_BLOCK = 32,
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    CU_JIT_NUM_OPTIONS = 34,
}
#[cfg(any(feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_MAX_THREADS_PER_BLOCK = 32,
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    CU_JIT_SPLIT_COMPILE = 34,
    CU_JIT_NUM_OPTIONS = 35,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_MAX_THREADS_PER_BLOCK = 32,
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    CU_JIT_SPLIT_COMPILE = 34,
    CU_JIT_BINARY_LOADER_THREAD_COUNT = 35,
    CU_JIT_NUM_OPTIONS = 36,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
    CU_TARGET_COMPUTE_90A = 65626,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
    CU_TARGET_COMPUTE_100 = 100,
    CU_TARGET_COMPUTE_101 = 101,
    CU_TARGET_COMPUTE_120 = 120,
    CU_TARGET_COMPUTE_90A = 65626,
    CU_TARGET_COMPUTE_100A = 65636,
    CU_TARGET_COMPUTE_101A = 65637,
    CU_TARGET_COMPUTE_120A = 65656,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
    CU_TARGET_COMPUTE_100 = 100,
    CU_TARGET_COMPUTE_101 = 101,
    CU_TARGET_COMPUTE_103 = 103,
    CU_TARGET_COMPUTE_120 = 120,
    CU_TARGET_COMPUTE_121 = 121,
    CU_TARGET_COMPUTE_90A = 65626,
    CU_TARGET_COMPUTE_100A = 65636,
    CU_TARGET_COMPUTE_101A = 65637,
    CU_TARGET_COMPUTE_103A = 65639,
    CU_TARGET_COMPUTE_120A = 65656,
    CU_TARGET_COMPUTE_121A = 65657,
    CU_TARGET_COMPUTE_100F = 131172,
    CU_TARGET_COMPUTE_101F = 131173,
    CU_TARGET_COMPUTE_103F = 131175,
    CU_TARGET_COMPUTE_120F = 131192,
    CU_TARGET_COMPUTE_121F = 131193,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
    CU_TARGET_COMPUTE_100 = 100,
    CU_TARGET_COMPUTE_110 = 110,
    CU_TARGET_COMPUTE_103 = 103,
    CU_TARGET_COMPUTE_120 = 120,
    CU_TARGET_COMPUTE_121 = 121,
    CU_TARGET_COMPUTE_90A = 65626,
    CU_TARGET_COMPUTE_100A = 65636,
    CU_TARGET_COMPUTE_110A = 65646,
    CU_TARGET_COMPUTE_103A = 65639,
    CU_TARGET_COMPUTE_120A = 65656,
    CU_TARGET_COMPUTE_121A = 65657,
    CU_TARGET_COMPUTE_100F = 131172,
    CU_TARGET_COMPUTE_110F = 131182,
    CU_TARGET_COMPUTE_103F = 131175,
    CU_TARGET_COMPUTE_120F = 131192,
    CU_TARGET_COMPUTE_121F = 131193,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUkernelNodeAttrID_enum {
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUkernelNodeAttrID_enum {
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2,
    CU_KERNEL_NODE_ATTRIBUTE_PRIORITY = 8,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
    CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING = 16,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
    CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING = 16,
    CU_LAUNCH_ATTRIBUTE_PORTABLE_CLUSTER_SIZE_MODE = 17,
    CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE = 18,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributePortableClusterMode_enum {
    CU_LAUNCH_PORTABLE_CLUSTER_MODE_DEFAULT = 0,
    CU_LAUNCH_PORTABLE_CLUSTER_MODE_REQUIRE_PORTABLE = 1,
    CU_LAUNCH_PORTABLE_CLUSTER_MODE_ALLOW_NON_PORTABLE = 2,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchMemSyncDomain_enum {
    CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = 0,
    CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = 1,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlibraryOption_enum {
    CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0,
    CU_LIBRARY_BINARY_IS_PRESERVED = 1,
    CU_LIBRARY_NUM_OPTIONS = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6,
    CU_LIMIT_MAX = 7,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6,
    CU_LIMIT_SHMEM_SIZE = 7,
    CU_LIMIT_CIG_ENABLED = 8,
    CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED = 9,
    CU_LIMIT_MAX = 10,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlogLevel_enum {
    CU_LOG_LEVEL_ERROR = 0,
    CU_LOG_LEVEL_WARNING = 1,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlogicalEndpointFlag_enum {
    CU_LOGICAL_ENDPOINT_FLAG_NONE = 0,
    CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS = 1,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlogicalEndpointIpcHandleType_enum {
    CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_NONE = 0,
    CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC = 1,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlogicalEndpointType_enum {
    CU_LOGICAL_ENDPOINT_TYPE_INVALID = 0,
    CU_LOGICAL_ENDPOINT_TYPE_UNICAST = 1,
    CU_LOGICAL_ENDPOINT_TYPE_MULTICAST = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
    CU_MEM_ACCESS_FLAGS_PROT_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationCompType_enum {
    CU_MEM_ALLOCATION_COMP_NONE = 0,
    CU_MEM_ALLOCATION_COMP_GENERIC = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 1,
    CU_MEM_ALLOCATION_TYPE_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 1,
    CU_MEM_ALLOCATION_TYPE_MANAGED = 2,
    CU_MEM_ALLOCATION_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAttach_flags_enum {
    CU_MEM_ATTACH_GLOBAL = 1,
    CU_MEM_ATTACH_HOST = 2,
    CU_MEM_ATTACH_SINGLE = 4,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemDecompressAlgorithm_enum {
    CU_MEM_DECOMPRESS_UNSUPPORTED = 0,
    CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE = 1,
    CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY = 2,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemDecompressAlgorithm_enum {
    CU_MEM_DECOMPRESS_UNSUPPORTED = 0,
    CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE = 1,
    CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY = 2,
    CU_MEM_DECOMPRESS_ALGORITHM_LZ4 = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemHandleType_enum {
    CU_MEM_HANDLE_TYPE_GENERIC = 0,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_HOST = 2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4,
    CU_MEM_LOCATION_TYPE_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_HOST = 2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4,
    CU_MEM_LOCATION_TYPE_INVISIBLE = 5,
    CU_MEM_LOCATION_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemOperationType_enum {
    CU_MEM_OPERATION_TYPE_MAP = 1,
    CU_MEM_OPERATION_TYPE_UNMAP = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8,
    CU_MEMPOOL_ATTR_ALLOCATION_TYPE = 9,
    CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES = 10,
    CU_MEMPOOL_ATTR_LOCATION_ID = 11,
    CU_MEMPOOL_ATTR_LOCATION_TYPE = 12,
    CU_MEMPOOL_ATTR_MAX_POOL_SIZE = 13,
    CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED = 14,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemRangeFlags_enum {
    CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE = 1,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemRangeHandleType_enum {
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1,
    CU_MEM_RANGE_HANDLE_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmem_advise_enum {
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmem_range_attribute_enum {
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmem_range_attribute_enum {
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE = 5,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID = 6,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE = 7,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID = 8,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemcpy3DOperandType_enum {
    CU_MEMCPY_OPERAND_TYPE_POINTER = 1,
    CU_MEMCPY_OPERAND_TYPE_ARRAY = 2,
    CU_MEMCPY_OPERAND_TYPE_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemcpyFlags_enum {
    CU_MEMCPY_FLAG_DEFAULT = 0,
    CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemcpySrcAccessOrder_enum {
    CU_MEMCPY_SRC_ACCESS_ORDER_INVALID = 0,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1,
    CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL = 2,
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3,
    CU_MEMCPY_SRC_ACCESS_ORDER_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmoduleLoadingMode_enum {
    CU_MODULE_EAGER_LOADING = 1,
    CU_MODULE_LAZY_LOADING = 2,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmulticastGranularity_flags_enum {
    CU_MULTICAST_GRANULARITY_MINIMUM = 0,
    CU_MULTICAST_GRANULARITY_RECOMMENDED = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUoccupancy_flags_enum {
    CU_OCCUPANCY_DEFAULT = 0,
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUoutput_mode_enum {
    CU_OUT_KEY_VALUE_PAIR = 0,
    CU_OUT_CSV = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
    CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUprocessState_enum {
    CU_PROCESS_STATE_RUNNING = 0,
    CU_PROCESS_STATE_LOCKED = 1,
    CU_PROCESS_STATE_CHECKPOINTED = 2,
    CU_PROCESS_STATE_FAILED = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUresourceViewFormat_enum {
    CU_RES_VIEW_FORMAT_NONE = 0,
    CU_RES_VIEW_FORMAT_UINT_1X8 = 1,
    CU_RES_VIEW_FORMAT_UINT_2X8 = 2,
    CU_RES_VIEW_FORMAT_UINT_4X8 = 3,
    CU_RES_VIEW_FORMAT_SINT_1X8 = 4,
    CU_RES_VIEW_FORMAT_SINT_2X8 = 5,
    CU_RES_VIEW_FORMAT_SINT_4X8 = 6,
    CU_RES_VIEW_FORMAT_UINT_1X16 = 7,
    CU_RES_VIEW_FORMAT_UINT_2X16 = 8,
    CU_RES_VIEW_FORMAT_UINT_4X16 = 9,
    CU_RES_VIEW_FORMAT_SINT_1X16 = 10,
    CU_RES_VIEW_FORMAT_SINT_2X16 = 11,
    CU_RES_VIEW_FORMAT_SINT_4X16 = 12,
    CU_RES_VIEW_FORMAT_UINT_1X32 = 13,
    CU_RES_VIEW_FORMAT_UINT_2X32 = 14,
    CU_RES_VIEW_FORMAT_UINT_4X32 = 15,
    CU_RES_VIEW_FORMAT_SINT_1X32 = 16,
    CU_RES_VIEW_FORMAT_SINT_2X32 = 17,
    CU_RES_VIEW_FORMAT_SINT_4X32 = 18,
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19,
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20,
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21,
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22,
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23,
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28,
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30,
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32,
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2,
    CU_RESOURCE_TYPE_PITCH2D = 3,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUsharedMemoryMode_enum {
    CU_SHARED_MEMORY_MODE_DEFAULT = 0,
    CU_SHARED_MEMORY_MODE_REQUIRE_PORTABLE = 1,
    CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE = 2,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUshared_carveout_enum {
    CU_SHAREDMEM_CARVEOUT_DEFAULT = -1,
    CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100,
    CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUsharedconfig_enum {
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamAtomicReductionDataType_enum {
    CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_32 = 14,
    CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_64 = 22,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamAtomicReductionOpType_enum {
    CU_STREAM_ATOMIC_REDUCTION_OP_OR = 6,
    CU_STREAM_ATOMIC_REDUCTION_OP_AND = 5,
    CU_STREAM_ATOMIC_REDUCTION_OP_ADD = 0,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamAttrID_enum {
    CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
    CU_STREAM_MEM_OP_BARRIER = 6,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
    CU_STREAM_MEM_OP_BARRIER = 6,
    CU_STREAM_MEM_OP_ATOMIC_REDUCTION = 8,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureMode_enum {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureStatus_enum {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCigDataType_enum {
    STREAM_CIG_DATA_TYPE_D3D12_COMMAND_LIST = 1,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamMemoryBarrier_flags_enum {
    CU_STREAM_MEMORY_BARRIER_TYPE_SYS = 0,
    CU_STREAM_MEMORY_BARRIER_TYPE_GPU = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamUpdateCaptureDependencies_flags_enum {
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = 0,
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamWaitValue_flags_enum {
    CU_STREAM_WAIT_VALUE_GEQ = 0,
    CU_STREAM_WAIT_VALUE_EQ = 1,
    CU_STREAM_WAIT_VALUE_AND = 2,
    CU_STREAM_WAIT_VALUE_NOR = 3,
    CU_STREAM_WAIT_VALUE_FLUSH = 1073741824,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamWriteValue_flags_enum {
    CU_STREAM_WRITE_VALUE_DEFAULT = 0,
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstream_flags_enum {
    CU_STREAM_DEFAULT = 0,
    CU_STREAM_NON_BLOCKING = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUsynchronizationPolicy_enum {
    CU_SYNC_POLICY_AUTO = 1,
    CU_SYNC_POLICY_SPIN = 2,
    CU_SYNC_POLICY_YIELD = 3,
    CU_SYNC_POLICY_BLOCKING_SYNC = 4,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
    CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
    CU_TENSOR_MAP_DATA_TYPE_INT32 = 3,
    CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4,
    CU_TENSOR_MAP_DATA_TYPE_INT64 = 5,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 10,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
    CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
    CU_TENSOR_MAP_DATA_TYPE_INT32 = 3,
    CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4,
    CU_TENSOR_MAP_DATA_TYPE_INT64 = 5,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 10,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B = 13,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B = 14,
    CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B = 15,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapFloatOOBfill_enum {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapIm2ColWideMode_enum {
    CU_TENSOR_MAP_IM2COL_WIDE_MODE_W = 0,
    CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128 = 1,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapInterleave_enum {
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    CU_TENSOR_MAP_INTERLEAVE_16B = 1,
    CU_TENSOR_MAP_INTERLEAVE_32B = 2,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapL2promotion_enum {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 1,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B = 1,
    CU_TENSOR_MAP_SWIZZLE_64B = 2,
    CU_TENSOR_MAP_SWIZZLE_128B = 3,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B = 1,
    CU_TENSOR_MAP_SWIZZLE_64B = 2,
    CU_TENSOR_MAP_SWIZZLE_128B = 3,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B = 4,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B = 5,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUuserObjectRetain_flags_enum {
    CU_GRAPH_USER_OBJECT_MOVE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUuserObject_flags_enum {
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_CONTAINED = 226,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
    CUDA_ERROR_KEY_ROTATION = 916,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER = 36,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_CONTAINED = 226,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
    CUDA_ERROR_KEY_ROTATION = 916,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER = 36,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_CONTAINED = 226,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
    CUDA_ERROR_KEY_ROTATION = 916,
    CUDA_ERROR_STREAM_DETACHED = 917,
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER = 36,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_CONTAINED = 226,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
    CUDA_ERROR_KEY_ROTATION = 916,
    CUDA_ERROR_STREAM_DETACHED = 917,
    CUDA_ERROR_GRAPH_RECAPTURE_FAILURE = 918,
    CUDA_ERROR_UNKNOWN = 999,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY3D_DESCRIPTOR_st {
    pub Width: usize,
    pub Height: usize,
    pub Depth: usize,
    pub Format: CUarray_format,
    pub NumChannels: ::core::ffi::c_uint,
    pub Flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_DESCRIPTOR_st {
    pub Width: usize,
    pub Height: usize,
    pub Format: CUarray_format,
    pub NumChannels: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_MEMORY_REQUIREMENTS_st {
    pub size: usize,
    pub alignment: usize,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_SPARSE_PROPERTIES_st {
    pub tileExtent: CUDA_ARRAY_SPARSE_PROPERTIES_st__bindgen_ty_1,
    pub miptailFirstLevel: ::core::ffi::c_uint,
    pub miptailSize: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_SPARSE_PROPERTIES_st__bindgen_ty_1 {
    pub width: ::core::ffi::c_uint,
    pub height: ::core::ffi::c_uint,
    pub depth: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS_st {
    pub ctx: CUcontext,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut CUstreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st {
    pub ctx: CUcontext,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut CUstreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st {
    pub ctx: CUcontext,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut CUstreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_CHILD_GRAPH_NODE_PARAMS_st {
    pub graph: CUgraph,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_CHILD_GRAPH_NODE_PARAMS_st {
    pub graph: CUgraph,
    pub ownership: CUgraphChildGraphNodeOwnership,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_CONDITIONAL_NODE_PARAMS {
    pub handle: CUgraphConditionalHandle,
    pub type_: CUgraphConditionalNodeType,
    pub size: ::core::ffi::c_uint,
    pub phGraph_out: *mut CUgraph,
    pub ctx: CUcontext,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EVENT_RECORD_NODE_PARAMS_st {
    pub event: CUevent,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EVENT_WAIT_NODE_PARAMS_st {
    pub event: CUevent,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
    pub type_: CUexternalMemoryHandleType,
    pub handle: CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
    pub offset: ::core::ffi::c_ulonglong,
    pub arrayDesc: CUDA_ARRAY3D_DESCRIPTOR,
    pub numLevels: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
    pub type_: CUexternalSemaphoreHandleType,
    pub handle: CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st {
    pub params: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
    pub fence: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st {
    pub params: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
    pub fence: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 10usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_GRAPH_INSTANTIATE_PARAMS_st {
    pub flags: cuuint64_t,
    pub hUploadStream: CUstream,
    pub hErrNode_out: CUgraphNode,
    pub result_out: CUgraphInstantiateResult,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_HOST_NODE_PARAMS_st {
    pub fn_: CUhostFn,
    pub userData: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_HOST_NODE_PARAMS_v2_st {
    pub fn_: CUhostFn,
    pub userData: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_HOST_NODE_PARAMS_v2_st {
    pub fn_: CUhostFn,
    pub userData: *mut ::core::ffi::c_void,
    pub syncMode: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_KERNEL_NODE_PARAMS_st {
    pub func: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_KERNEL_NODE_PARAMS_v2_st {
    pub func: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_KERNEL_NODE_PARAMS_v3_st {
    pub func: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_LAUNCH_PARAMS_st {
    pub function: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub hStream: CUstream,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY2D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::core::ffi::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::core::ffi::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_MEMCPY3D_BATCH_OP_st {
    pub src: CUmemcpy3DOperand,
    pub dst: CUmemcpy3DOperand,
    pub extent: CUextent3D,
    pub srcAccessOrder: CUmemcpySrcAccessOrder,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY3D_PEER_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::core::ffi::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcContext: CUcontext,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::core::ffi::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstContext: CUcontext,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY3D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::core::ffi::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub reserved0: *mut ::core::ffi::c_void,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::core::ffi::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub reserved1: *mut ::core::ffi::c_void,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY_NODE_PARAMS_st {
    pub flags: ::core::ffi::c_int,
    pub reserved: ::core::ffi::c_int,
    pub copyCtx: CUcontext,
    pub copyParams: CUDA_MEMCPY3D,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMSET_NODE_PARAMS_st {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMSET_NODE_PARAMS_v2_st {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
    pub ctx: CUcontext,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_v1_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_v1_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_v2_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_v2_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_FREE_NODE_PARAMS_st {
    pub dptr: CUdeviceptr,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    pub p2pToken: ::core::ffi::c_ulonglong,
    pub vaSpaceToken: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_RESOURCE_DESC_st {
    pub resType: CUresourcetype,
    pub res: CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub hArray: CUarray,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    pub hMipmappedArray: CUmipmappedArray,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::core::ffi::c_uint,
    pub sizeInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5 {
    pub reserved: [::core::ffi::c_int; 32usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_VIEW_DESC_st {
    pub format: CUresourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::core::ffi::c_uint,
    pub lastMipmapLevel: ::core::ffi::c_uint,
    pub firstLayer: ::core::ffi::c_uint,
    pub lastLayer: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct CUDA_TEXTURE_DESC_st {
    pub addressMode: [CUaddress_mode; 3usize],
    pub filterMode: CUfilter_mode,
    pub flags: ::core::ffi::c_uint,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: CUfilter_mode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub borderColor: [f32; 4usize],
    pub reserved: [::core::ffi::c_int; 12usize],
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CU_DEV_SM_RESOURCE_GROUP_PARAMS_st {
    pub smCount: ::core::ffi::c_uint,
    pub coscheduledSmCount: ::core::ffi::c_uint,
    pub preferredCoscheduledSmCount: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct CUaccessPolicyWindow_st {
    pub base_ptr: *mut ::core::ffi::c_void,
    pub num_bytes: usize,
    pub hitRatio: f32,
    pub hitProp: CUaccessProperty,
    pub missProp: CUaccessProperty,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUarrayMapInfo_st {
    pub resourceType: CUresourcetype,
    pub resource: CUarrayMapInfo_st__bindgen_ty_1,
    pub subresourceType: CUarraySparseSubresourceType,
    pub subresource: CUarrayMapInfo_st__bindgen_ty_2,
    pub memOperationType: CUmemOperationType,
    pub memHandleType: CUmemHandleType,
    pub memHandle: CUarrayMapInfo_st__bindgen_ty_3,
    pub offset: ::core::ffi::c_ulonglong,
    pub deviceBitMask: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 2usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_1 {
    pub level: ::core::ffi::c_uint,
    pub layer: ::core::ffi::c_uint,
    pub offsetX: ::core::ffi::c_uint,
    pub offsetY: ::core::ffi::c_uint,
    pub offsetZ: ::core::ffi::c_uint,
    pub extentWidth: ::core::ffi::c_uint,
    pub extentHeight: ::core::ffi::c_uint,
    pub extentDepth: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_2 {
    pub layer: ::core::ffi::c_uint,
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUarray_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUasyncCallbackEntry_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUasyncNotificationInfo_st {
    pub type_: CUasyncNotificationType,
    pub info: CUasyncNotificationInfo_st__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUasyncNotificationInfo_st__bindgen_ty_1__bindgen_ty_1 {
    pub bytesOverBudget: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointCheckpointArgs_st {
    pub reserved: [cuuint64_t; 8usize],
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointGpuPair_st {
    pub oldUuid: CUuuid,
    pub newUuid: CUuuid,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointLockArgs_st {
    pub timeoutMs: ::core::ffi::c_uint,
    pub reserved0: ::core::ffi::c_uint,
    pub reserved1: [cuuint64_t; 7usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointRestoreArgs_st {
    pub reserved: [cuuint64_t; 8usize],
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointRestoreArgs_st {
    pub gpuPairs: *mut CUcheckpointGpuPair,
    pub gpuPairsCount: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_char; 44usize],
    pub reserved1: cuuint64_t,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointRestoreArgs_st {
    pub gpuPairs: *mut CUcheckpointGpuPair,
    pub gpuPairsCount: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_char; 52usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointUnlockArgs_st {
    pub reserved: [cuuint64_t; 8usize],
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUcoredumpCallbackEntry_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUctxCigParam_st {
    pub sharedDataType: CUcigDataType,
    pub sharedData: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUctxCreateParams_st {
    pub execAffinityParams: *mut CUexecAffinityParam,
    pub numExecAffinityParams: ::core::ffi::c_int,
    pub cigParams: *mut CUctxCigParam,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUctx_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUdevResourceDesc_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUdevResource_st {
    pub type_: CUdevResourceType,
    pub _internal_padding: [::core::ffi::c_uchar; 92usize],
    pub __bindgen_anon_1: CUdevResource_st__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUdevResource_st {
    pub type_: CUdevResourceType,
    pub _internal_padding: [::core::ffi::c_uchar; 92usize],
    pub __bindgen_anon_1: CUdevResource_st__bindgen_ty_1,
    pub nextResource: *mut CUdevResource_st,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevSmResource_st {
    pub smCount: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevSmResource_st {
    pub smCount: ::core::ffi::c_uint,
    pub minSmPartitionSize: ::core::ffi::c_uint,
    pub smCoscheduledAlignment: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevSmResource_st {
    pub smCount: ::core::ffi::c_uint,
    pub minSmPartitionSize: ::core::ffi::c_uint,
    pub smCoscheduledAlignment: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevWorkqueueConfigResource_st {
    pub device: CUdevice,
    pub wqConcurrencyLimit: ::core::ffi::c_uint,
    pub sharingScope: CUdevWorkqueueConfigScope,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevWorkqueueResource_st {
    pub reserved: [::core::ffi::c_uchar; 40usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevprop_st {
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub sharedMemPerBlock: ::core::ffi::c_int,
    pub totalConstantMemory: ::core::ffi::c_int,
    pub SIMDWidth: ::core::ffi::c_int,
    pub memPitch: ::core::ffi::c_int,
    pub regsPerBlock: ::core::ffi::c_int,
    pub clockRate: ::core::ffi::c_int,
    pub textureAlign: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUexecAffinityParam_st {
    pub type_: CUexecAffinityType,
    pub param: CUexecAffinityParam_st__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUexecAffinitySmCount_st {
    pub val: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUextMemory_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUextSemaphore_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUextent3D_st {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
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
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphEdgeData_st {
    pub from_port: ::core::ffi::c_uchar,
    pub to_port: ::core::ffi::c_uchar,
    pub type_: ::core::ffi::c_uchar,
    pub reserved: [::core::ffi::c_uchar; 5usize],
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphExecUpdateResultInfo_st {
    pub result: CUgraphExecUpdateResult,
    pub errorNode: CUgraphNode,
    pub errorFromNode: CUgraphNode,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUgraphNodeParams_st {
    pub type_: CUgraphNodeType,
    pub reserved0: [::core::ffi::c_int; 3usize],
    pub __bindgen_anon_1: CUgraphNodeParams_st__bindgen_ty_1,
    pub reserved2: ::core::ffi::c_longlong,
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphicsResource_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgreenCtx_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUipcEventHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUipcMemHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUkern_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_1 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_2 {
    pub event: CUevent,
    pub flags: ::core::ffi::c_int,
    pub triggerAtBlockStart: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_3 {
    pub event: CUevent,
    pub flags: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_4 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: CUgraphDeviceNode,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_4 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_5 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: CUgraphDeviceNode,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUlaunchAttribute_st {
    pub id: CUlaunchAttributeID,
    pub pad: [::core::ffi::c_char; 4usize],
    pub value: CUlaunchAttributeValue,
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchConfig_st {
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub hStream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub numAttrs: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchMemSyncDomainMap_st {
    pub default_: ::core::ffi::c_uchar,
    pub remote: ::core::ffi::c_uchar,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlib_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlibraryHostUniversalFunctionAndDataTable_st {
    pub functionTable: *mut ::core::ffi::c_void,
    pub functionWindowSize: usize,
    pub dataTable: *mut ::core::ffi::c_void,
    pub dataWindowSize: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlinkState_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlogicalEndpointFabricHandle_st {
    pub data: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUlogicalEndpointProp_struct {
    pub type_: CUlogicalEndpointType,
    pub __bindgen_anon_1: CUlogicalEndpointProp_struct__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub ipcHandleTypes: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlogicalEndpointProp_struct__bindgen_ty_1__bindgen_ty_1 {
    pub device: CUdevice,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlogicalEndpointProp_struct__bindgen_ty_1__bindgen_ty_2 {
    pub numDevices: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlogsCallbackEntry_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAccessDesc_st {
    pub location: CUmemLocation,
    pub flags: CUmemAccess_flags,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemAccessDesc_st {
    pub location: CUmemLocation,
    pub flags: CUmemAccess_flags,
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAllocationHandleType_enum(pub ::core::ffi::c_uint);
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAllocationProp_st {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut ::core::ffi::c_void,
    pub allocFlags: CUmemAllocationProp_st__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemAllocationProp_st {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut ::core::ffi::c_void,
    pub allocFlags: CUmemAllocationProp_st__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAllocationProp_st__bindgen_ty_1 {
    pub compressionType: ::core::ffi::c_uchar,
    pub gpuDirectRDMACapable: ::core::ffi::c_uchar,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 4usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemDecompressParams_st {
    pub srcNumBytes: usize,
    pub dstNumBytes: usize,
    pub dstActBytes: *mut cuuint32_t,
    pub src: *const ::core::ffi::c_void,
    pub dst: *mut ::core::ffi::c_void,
    pub algo: CUmemDecompressAlgorithm,
    pub padding: [::core::ffi::c_uchar; 20usize],
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemFabricHandle_st {
    pub data: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemLocation_st {
    pub type_: CUmemLocationType,
    pub id: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemLocation_st {
    pub type_: CUmemLocationType,
    pub __bindgen_anon_1: CUmemLocation_st__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmemPoolHandle_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub reserved: [::core::ffi::c_uchar; 56usize],
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 54usize],
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 54usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolPtrExportData_st {
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemcpy3DOperand_st {
    pub type_: CUmemcpy3DOperandType,
    pub op: CUmemcpy3DOperand_st__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: CUdeviceptr,
    pub rowLength: usize,
    pub layerHeight: usize,
    pub locHint: CUmemLocation,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: CUdeviceptr,
    pub rowLength: usize,
    pub layerHeight: usize,
    pub locHint: CUmemLocation,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_2 {
    pub array: CUarray,
    pub offset: CUoffset3D,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemcpyAttributes_st {
    pub srcAccessOrder: CUmemcpySrcAccessOrder,
    pub srcLocHint: CUmemLocation,
    pub dstLocHint: CUmemLocation,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemcpyAttributes_st {
    pub srcAccessOrder: CUmemcpySrcAccessOrder,
    pub srcLocHint: CUmemLocation,
    pub dstLocHint: CUmemLocation,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmipmappedArray_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmod_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmulticastObjectProp_st {
    pub numDevices: ::core::ffi::c_uint,
    pub size: usize,
    pub handleTypes: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUoffset3D_st {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpAtomicReductionParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
    pub reductionOp: CUstreamAtomicReductionOpType,
    pub dataType: CUstreamAtomicReductionDataType,
    pub address: CUdeviceptr,
    pub value: cuuint64_t,
    pub alias: CUdeviceptr,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub address: CUdeviceptr,
    pub __bindgen_anon_1: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub alias: CUdeviceptr,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub address: CUdeviceptr,
    pub __bindgen_anon_1: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub alias: CUdeviceptr,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamCigCaptureParams_st {
    pub streamCigParams: *mut CUstreamCigParam,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamCigParam_st {
    pub streamSharedDataType: CUstreamCigDataType,
    pub streamSharedData: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUsurfref_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[repr(align(64))]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUtensorMap_st {
    pub opaque: [cuuint64_t; 16usize],
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[repr(align(128))]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUtensorMap_st {
    pub opaque: [cuuint64_t; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUtexref_st {
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
impl CUctx_flags_enum {
    pub const CU_CTX_BLOCKING_SYNC: CUctx_flags_enum = CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC;
}
impl CUdevice_P2PAttribute_enum {
    pub const CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED: CUdevice_P2PAttribute_enum = CUdevice_P2PAttribute_enum::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED;
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl CUmemAllocationHandleType_enum {
    pub const CU_MEM_HANDLE_TYPE_FABRIC: CUmemAllocationHandleType_enum = CUmemAllocationHandleType_enum(8);
}
impl CUmemAllocationHandleType_enum {
    pub const CU_MEM_HANDLE_TYPE_MAX: CUmemAllocationHandleType_enum = CUmemAllocationHandleType_enum(2147483647);
}
impl CUmemAllocationHandleType_enum {
    pub const CU_MEM_HANDLE_TYPE_NONE: CUmemAllocationHandleType_enum = CUmemAllocationHandleType_enum(0);
}
impl CUmemAllocationHandleType_enum {
    pub const CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: CUmemAllocationHandleType_enum = CUmemAllocationHandleType_enum(1);
}
impl CUmemAllocationHandleType_enum {
    pub const CU_MEM_HANDLE_TYPE_WIN32: CUmemAllocationHandleType_enum = CUmemAllocationHandleType_enum(2);
}
impl CUmemAllocationHandleType_enum {
    pub const CU_MEM_HANDLE_TYPE_WIN32_KMT: CUmemAllocationHandleType_enum = CUmemAllocationHandleType_enum(4);
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
impl CUmemLocationType_enum {
    pub const CU_MEM_LOCATION_TYPE_NONE: CUmemLocationType_enum = CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID;
}
impl ::core::ops::BitAnd<CUmemAllocationHandleType_enum> for CUmemAllocationHandleType_enum {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        CUmemAllocationHandleType_enum(self.0 & other.0)
    }
}
impl ::core::ops::BitAndAssign for CUmemAllocationHandleType_enum {
    #[inline]
    fn bitand_assign(&mut self, rhs: CUmemAllocationHandleType_enum) {
        self.0 &= rhs.0;
    }
}
impl ::core::ops::BitOr<CUmemAllocationHandleType_enum> for CUmemAllocationHandleType_enum {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        CUmemAllocationHandleType_enum(self.0 | other.0)
    }
}
impl ::core::ops::BitOrAssign for CUmemAllocationHandleType_enum {
    #[inline]
    fn bitor_assign(&mut self, rhs: CUmemAllocationHandleType_enum) {
        self.0 |= rhs.0;
    }
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciBufObject: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSyncObj: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUarrayMapInfo_st__bindgen_ty_1 {
    pub mipmap: CUmipmappedArray,
    pub array: CUarray,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUarrayMapInfo_st__bindgen_ty_2 {
    pub sparseLevel: CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_1,
    pub miptail: CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_2,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUarrayMapInfo_st__bindgen_ty_3 {
    pub memHandle: CUmemGenericAllocationHandle,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUasyncNotificationInfo_st__bindgen_ty_1 {
    pub overBudget: CUasyncNotificationInfo_st__bindgen_ty_1__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUdevResource_st__bindgen_ty_1 {
    pub sm: CUdevSmResource,
    pub _oversize: [::core::ffi::c_uchar; 48usize],
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUdevResource_st__bindgen_ty_1 {
    pub sm: CUdevSmResource,
    pub wqConfig: CUdevWorkqueueConfigResource,
    pub wq: CUdevWorkqueueResource,
    pub _oversize: [::core::ffi::c_uchar; 40usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUexecAffinityParam_st__bindgen_ty_1 {
    pub smCount: CUexecAffinitySmCount,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUgraphNodeParams_st__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: CUDA_KERNEL_NODE_PARAMS_v3,
    pub memcpy: CUDA_MEMCPY_NODE_PARAMS,
    pub memset: CUDA_MEMSET_NODE_PARAMS_v2,
    pub host: CUDA_HOST_NODE_PARAMS_v2,
    pub graph: CUDA_CHILD_GRAPH_NODE_PARAMS,
    pub eventWait: CUDA_EVENT_WAIT_NODE_PARAMS,
    pub eventRecord: CUDA_EVENT_RECORD_NODE_PARAMS,
    pub extSemSignal: CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2,
    pub extSemWait: CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2,
    pub alloc: CUDA_MEM_ALLOC_NODE_PARAMS_v2,
    pub free: CUDA_MEM_FREE_NODE_PARAMS,
    pub memOp: CUDA_BATCH_MEM_OP_NODE_PARAMS_v2,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUgraphNodeParams_st__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: CUDA_KERNEL_NODE_PARAMS_v3,
    pub memcpy: CUDA_MEMCPY_NODE_PARAMS,
    pub memset: CUDA_MEMSET_NODE_PARAMS_v2,
    pub host: CUDA_HOST_NODE_PARAMS_v2,
    pub graph: CUDA_CHILD_GRAPH_NODE_PARAMS,
    pub eventWait: CUDA_EVENT_WAIT_NODE_PARAMS,
    pub eventRecord: CUDA_EVENT_RECORD_NODE_PARAMS,
    pub extSemSignal: CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2,
    pub extSemWait: CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2,
    pub alloc: CUDA_MEM_ALLOC_NODE_PARAMS_v2,
    pub free: CUDA_MEM_FREE_NODE_PARAMS,
    pub memOp: CUDA_BATCH_MEM_OP_NODE_PARAMS_v2,
    pub conditional: CUDA_CONDITIONAL_NODE_PARAMS,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUgraphNodeParams_st__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: CUDA_KERNEL_NODE_PARAMS_v3,
    pub memcpy: CUDA_MEMCPY_NODE_PARAMS,
    pub memset: CUDA_MEMSET_NODE_PARAMS_v2,
    pub host: CUDA_HOST_NODE_PARAMS_v2,
    pub graph: CUDA_CHILD_GRAPH_NODE_PARAMS,
    pub eventWait: CUDA_EVENT_WAIT_NODE_PARAMS,
    pub eventRecord: CUDA_EVENT_RECORD_NODE_PARAMS,
    pub extSemSignal: CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2,
    pub extSemWait: CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2,
    pub alloc: CUDA_MEM_ALLOC_NODE_PARAMS_v2,
    pub free: CUDA_MEM_FREE_NODE_PARAMS,
    pub memOp: CUDA_BATCH_MEM_OP_NODE_PARAMS_v2,
    pub conditional: CUDA_CONDITIONAL_NODE_PARAMS,
    pub asBytes: [::core::ffi::c_char; 232usize],
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUkernelNodeAttrValue_union {
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUkernelNodeAttrValue_union {
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_4,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_4,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub preferredClusterDim: CUlaunchAttributeValue_union__bindgen_ty_4,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub preferredClusterDim: CUlaunchAttributeValue_union__bindgen_ty_4,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
    pub nvlinkUtilCentricScheduling: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub preferredClusterDim: CUlaunchAttributeValue_union__bindgen_ty_4,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
    pub nvlinkUtilCentricScheduling: ::core::ffi::c_uint,
    pub portableClusterSizeMode: CUlaunchAttributePortableClusterMode,
    pub sharedMemoryMode: CUsharedMemoryMode,
}
#[cfg(any(feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlogicalEndpointProp_struct__bindgen_ty_1 {
    pub unicast: CUlogicalEndpointProp_struct__bindgen_ty_1__bindgen_ty_1,
    pub multicast: CUlogicalEndpointProp_struct__bindgen_ty_1__bindgen_ty_2,
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUmemLocation_st__bindgen_ty_1 {
    pub id: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUmemcpy3DOperand_st__bindgen_ty_1 {
    pub ptr: CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_1,
    pub array: CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamAttrValue_union {
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub syncPolicy: CUsynchronizationPolicy,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union {
    pub operation: CUstreamBatchMemOpType,
    pub waitValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st,
    pub writeValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st,
    pub flushRemoteWrites: CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st,
    pub pad: [cuuint64_t; 6usize],
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union {
    pub operation: CUstreamBatchMemOpType,
    pub waitValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st,
    pub writeValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st,
    pub flushRemoteWrites: CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st,
    pub memoryBarrier: CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st,
    pub pad: [cuuint64_t; 6usize],
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union {
    pub operation: CUstreamBatchMemOpType,
    pub waitValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st,
    pub writeValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st,
    pub flushRemoteWrites: CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st,
    pub memoryBarrier: CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st,
    pub atomicReduction: CUstreamBatchMemOpParams_union_CUstreamMemOpAtomicReductionParams_st,
    pub pad: [cuuint64_t; 6usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
    pub value: cuuint32_t,
    pub value64: cuuint64_t,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
    pub value: cuuint32_t,
    pub value64: cuuint64_t,
}
pub unsafe fn cuArray3DCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, *const CUDA_ARRAY3D_DESCRIPTOR) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArray3DCreate_v2") });
        _f(pHandle, pAllocateArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArray3DCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR) -> CUresult;
        }
        cuArray3DCreate_v2(pHandle, pAllocateArray)
    }
}
pub unsafe fn cuArray3DGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR, hArray: CUarray) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_ARRAY3D_DESCRIPTOR, CUarray) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArray3DGetDescriptor_v2") });
        _f(pArrayDescriptor, hArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArray3DGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR, hArray: CUarray) -> CUresult;
        }
        cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray)
    }
}
pub unsafe fn cuArrayCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, *const CUDA_ARRAY_DESCRIPTOR) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArrayCreate_v2") });
        _f(pHandle, pAllocateArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArrayCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR) -> CUresult;
        }
        cuArrayCreate_v2(pHandle, pAllocateArray)
    }
}
pub unsafe fn cuArrayDestroy(hArray: CUarray) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUarray) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArrayDestroy") });
        _f(hArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArrayDestroy(hArray: CUarray) -> CUresult;
        }
        cuArrayDestroy(hArray)
    }
}
pub unsafe fn cuArrayGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR, hArray: CUarray) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_ARRAY_DESCRIPTOR, CUarray) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArrayGetDescriptor_v2") });
        _f(pArrayDescriptor, hArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArrayGetDescriptor_v2(pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR, hArray: CUarray) -> CUresult;
        }
        cuArrayGetDescriptor_v2(pArrayDescriptor, hArray)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuArrayGetMemoryRequirements(memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS, array: CUarray, device: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_ARRAY_MEMORY_REQUIREMENTS, CUarray, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArrayGetMemoryRequirements") });
        _f(memoryRequirements, array, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArrayGetMemoryRequirements(memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS, array: CUarray, device: CUdevice) -> CUresult;
        }
        cuArrayGetMemoryRequirements(memoryRequirements, array, device)
    }
}
pub unsafe fn cuArrayGetPlane(pPlaneArray: *mut CUarray, hArray: CUarray, planeIdx: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, CUarray, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArrayGetPlane") });
        _f(pPlaneArray, hArray, planeIdx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArrayGetPlane(pPlaneArray: *mut CUarray, hArray: CUarray, planeIdx: ::core::ffi::c_uint) -> CUresult;
        }
        cuArrayGetPlane(pPlaneArray, hArray, planeIdx)
    }
}
pub unsafe fn cuArrayGetSparseProperties(sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES, array: CUarray) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_ARRAY_SPARSE_PROPERTIES, CUarray) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuArrayGetSparseProperties") });
        _f(sparseProperties, array)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuArrayGetSparseProperties(sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES, array: CUarray) -> CUresult;
        }
        cuArrayGetSparseProperties(sparseProperties, array)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpDeregisterCompleteCallback(callback: CUcoredumpCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpDeregisterCompleteCallback") });
        _f(callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpDeregisterCompleteCallback(callback: CUcoredumpCallbackHandle) -> CUresult;
        }
        cuCoredumpDeregisterCompleteCallback(callback)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpDeregisterStartCallback(callback: CUcoredumpCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpDeregisterStartCallback") });
        _f(callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpDeregisterStartCallback(callback: CUcoredumpCallbackHandle) -> CUresult;
        }
        cuCoredumpDeregisterStartCallback(callback)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpGetAttribute(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpSettings, *mut ::core::ffi::c_void, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpGetAttribute") });
        _f(attrib, value, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpGetAttribute(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult;
        }
        cuCoredumpGetAttribute(attrib, value, size)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpGetAttributeGlobal(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpSettings, *mut ::core::ffi::c_void, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpGetAttributeGlobal") });
        _f(attrib, value, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpGetAttributeGlobal(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult;
        }
        cuCoredumpGetAttributeGlobal(attrib, value, size)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpRegisterCompleteCallback(callback: CUcoredumpStatusCallback, userData: *mut ::core::ffi::c_void, callbackOut: *mut CUcoredumpCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpStatusCallback, *mut ::core::ffi::c_void, *mut CUcoredumpCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpRegisterCompleteCallback") });
        _f(callback, userData, callbackOut)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpRegisterCompleteCallback(callback: CUcoredumpStatusCallback, userData: *mut ::core::ffi::c_void, callbackOut: *mut CUcoredumpCallbackHandle) -> CUresult;
        }
        cuCoredumpRegisterCompleteCallback(callback, userData, callbackOut)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpRegisterStartCallback(callback: CUcoredumpStatusCallback, userData: *mut ::core::ffi::c_void, callbackOut: *mut CUcoredumpCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpStatusCallback, *mut ::core::ffi::c_void, *mut CUcoredumpCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpRegisterStartCallback") });
        _f(callback, userData, callbackOut)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpRegisterStartCallback(callback: CUcoredumpStatusCallback, userData: *mut ::core::ffi::c_void, callbackOut: *mut CUcoredumpCallbackHandle) -> CUresult;
        }
        cuCoredumpRegisterStartCallback(callback, userData, callbackOut)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpSetAttribute(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpSettings, *mut ::core::ffi::c_void, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpSetAttribute") });
        _f(attrib, value, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpSetAttribute(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult;
        }
        cuCoredumpSetAttribute(attrib, value, size)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCoredumpSetAttributeGlobal(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcoredumpSettings, *mut ::core::ffi::c_void, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCoredumpSetAttributeGlobal") });
        _f(attrib, value, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCoredumpSetAttributeGlobal(attrib: CUcoredumpSettings, value: *mut ::core::ffi::c_void, size: *mut usize) -> CUresult;
        }
        cuCoredumpSetAttributeGlobal(attrib, value, size)
    }
}
pub unsafe fn cuCtxAttach(pctx: *mut CUcontext, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxAttach") });
        _f(pctx, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxAttach(pctx: *mut CUcontext, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuCtxAttach(pctx, flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: ::core::ffi::c_uint, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext, ::core::ffi::c_uint, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxCreate_v2") });
        _f(pctx, flags, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: ::core::ffi::c_uint, dev: CUdevice) -> CUresult;
        }
        cuCtxCreate_v2(pctx, flags, dev)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuCtxCreate_v3(pctx: *mut CUcontext, paramsArray: *mut CUexecAffinityParam, numParams: ::core::ffi::c_int, flags: ::core::ffi::c_uint, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext, *mut CUexecAffinityParam, ::core::ffi::c_int, ::core::ffi::c_uint, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxCreate_v3") });
        _f(pctx, paramsArray, numParams, flags, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxCreate_v3(pctx: *mut CUcontext, paramsArray: *mut CUexecAffinityParam, numParams: ::core::ffi::c_int, flags: ::core::ffi::c_uint, dev: CUdevice) -> CUresult;
        }
        cuCtxCreate_v3(pctx, paramsArray, numParams, flags, dev)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxCreate_v4(pctx: *mut CUcontext, ctxCreateParams: *mut CUctxCreateParams, flags: ::core::ffi::c_uint, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext, *mut CUctxCreateParams, ::core::ffi::c_uint, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxCreate_v4") });
        _f(pctx, ctxCreateParams, flags, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxCreate_v4(pctx: *mut CUcontext, ctxCreateParams: *mut CUctxCreateParams, flags: ::core::ffi::c_uint, dev: CUdevice) -> CUresult;
        }
        cuCtxCreate_v4(pctx, ctxCreateParams, flags, dev)
    }
}
pub unsafe fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxDestroy_v2") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
        }
        cuCtxDestroy_v2(ctx)
    }
}
pub unsafe fn cuCtxDetach(ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxDetach") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxDetach(ctx: CUcontext) -> CUresult;
        }
        cuCtxDetach(ctx)
    }
}
pub unsafe fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxDisablePeerAccess") });
        _f(peerContext)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult;
        }
        cuCtxDisablePeerAccess(peerContext)
    }
}
pub unsafe fn cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxEnablePeerAccess") });
        _f(peerContext, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuCtxEnablePeerAccess(peerContext, Flags)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxFromGreenCtx(pContext: *mut CUcontext, hCtx: CUgreenCtx) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext, CUgreenCtx) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxFromGreenCtx") });
        _f(pContext, hCtx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxFromGreenCtx(pContext: *mut CUcontext, hCtx: CUgreenCtx) -> CUresult;
        }
        cuCtxFromGreenCtx(pContext, hCtx)
    }
}
pub unsafe fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext, *mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetApiVersion") });
        _f(ctx, version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuCtxGetApiVersion(ctx, version)
    }
}
pub unsafe fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfunc_cache) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetCacheConfig") });
        _f(pconfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult;
        }
        cuCtxGetCacheConfig(pconfig)
    }
}
pub unsafe fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetCurrent") });
        _f(pctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
        }
        cuCtxGetCurrent(pctx)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxGetDevResource(hCtx: CUcontext, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext, *mut CUdevResource, CUdevResourceType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetDevResource") });
        _f(hCtx, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetDevResource(hCtx: CUcontext, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult;
        }
        cuCtxGetDevResource(hCtx, resource, type_)
    }
}
pub unsafe fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetDevice") });
        _f(device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
        }
        cuCtxGetDevice(device)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxGetDevice_v2(device: *mut CUdevice, ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevice, CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetDevice_v2") });
        _f(device, ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetDevice_v2(device: *mut CUdevice, ctx: CUcontext) -> CUresult;
        }
        cuCtxGetDevice_v2(device, ctx)
    }
}
pub unsafe fn cuCtxGetExecAffinity(pExecAffinity: *mut CUexecAffinityParam, type_: CUexecAffinityType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUexecAffinityParam, CUexecAffinityType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetExecAffinity") });
        _f(pExecAffinity, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetExecAffinity(pExecAffinity: *mut CUexecAffinityParam, type_: CUexecAffinityType) -> CUresult;
        }
        cuCtxGetExecAffinity(pExecAffinity, type_)
    }
}
pub unsafe fn cuCtxGetFlags(flags: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetFlags") });
        _f(flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetFlags(flags: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuCtxGetFlags(flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxGetId(ctx: CUcontext, ctxId: *mut ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext, *mut ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetId") });
        _f(ctx, ctxId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetId(ctx: CUcontext, ctxId: *mut ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuCtxGetId(ctx, ctxId)
    }
}
pub unsafe fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, CUlimit) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetLimit") });
        _f(pvalue, limit)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult;
        }
        cuCtxGetLimit(pvalue, limit)
    }
}
pub unsafe fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUsharedconfig) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetSharedMemConfig") });
        _f(pConfig)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult;
        }
        cuCtxGetSharedMemConfig(pConfig)
    }
}
pub unsafe fn cuCtxGetStreamPriorityRange(leastPriority: *mut ::core::ffi::c_int, greatestPriority: *mut ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *mut ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxGetStreamPriorityRange") });
        _f(leastPriority, greatestPriority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxGetStreamPriorityRange(leastPriority: *mut ::core::ffi::c_int, greatestPriority: *mut ::core::ffi::c_int) -> CUresult;
        }
        cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)
    }
}
pub unsafe fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxPopCurrent_v2") });
        _f(pctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;
        }
        cuCtxPopCurrent_v2(pctx)
    }
}
pub unsafe fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxPushCurrent_v2") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
        }
        cuCtxPushCurrent_v2(ctx)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxRecordEvent") });
        _f(hCtx, hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult;
        }
        cuCtxRecordEvent(hCtx, hEvent)
    }
}
pub unsafe fn cuCtxResetPersistingL2Cache() -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxResetPersistingL2Cache") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxResetPersistingL2Cache() -> CUresult;
        }
        cuCtxResetPersistingL2Cache()
    }
}
pub unsafe fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunc_cache) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSetCacheConfig") });
        _f(config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult;
        }
        cuCtxSetCacheConfig(config)
    }
}
pub unsafe fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSetCurrent") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
        }
        cuCtxSetCurrent(ctx)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxSetFlags(flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSetFlags") });
        _f(flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSetFlags(flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuCtxSetFlags(flags)
    }
}
pub unsafe fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlimit, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSetLimit") });
        _f(limit, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult;
        }
        cuCtxSetLimit(limit, value)
    }
}
pub unsafe fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUsharedconfig) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSetSharedMemConfig") });
        _f(config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult;
        }
        cuCtxSetSharedMemConfig(config)
    }
}
pub unsafe fn cuCtxSynchronize() -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSynchronize") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSynchronize() -> CUresult;
        }
        cuCtxSynchronize()
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxSynchronize_v2(ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxSynchronize_v2") });
        _f(ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxSynchronize_v2(ctx: CUcontext) -> CUresult;
        }
        cuCtxSynchronize_v2(ctx)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUcontext, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuCtxWaitEvent") });
        _f(hCtx, hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult;
        }
        cuCtxWaitEvent(hCtx, hEvent)
    }
}
pub unsafe fn cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUexternalMemory) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDestroyExternalMemory") });
        _f(extMem)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult;
        }
        cuDestroyExternalMemory(extMem)
    }
}
pub unsafe fn cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUexternalSemaphore) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDestroyExternalSemaphore") });
        _f(extSem)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) -> CUresult;
        }
        cuDestroyExternalSemaphore(extSem)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDevResourceGenerateDesc(phDesc: *mut CUdevResourceDesc, resources: *mut CUdevResource, nbResources: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevResourceDesc, *mut CUdevResource, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevResourceGenerateDesc") });
        _f(phDesc, resources, nbResources)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevResourceGenerateDesc(phDesc: *mut CUdevResourceDesc, resources: *mut CUdevResource, nbResources: ::core::ffi::c_uint) -> CUresult;
        }
        cuDevResourceGenerateDesc(phDesc, resources, nbResources)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDevSmResourceSplit(result: *mut CUdevResource, nbGroups: ::core::ffi::c_uint, input: *const CUdevResource, remainder: *mut CUdevResource, flags: ::core::ffi::c_uint, groupParams: *mut CU_DEV_SM_RESOURCE_GROUP_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevResource, ::core::ffi::c_uint, *const CUdevResource, *mut CUdevResource, ::core::ffi::c_uint, *mut CU_DEV_SM_RESOURCE_GROUP_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevSmResourceSplit") });
        _f(result, nbGroups, input, remainder, flags, groupParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevSmResourceSplit(result: *mut CUdevResource, nbGroups: ::core::ffi::c_uint, input: *const CUdevResource, remainder: *mut CUdevResource, flags: ::core::ffi::c_uint, groupParams: *mut CU_DEV_SM_RESOURCE_GROUP_PARAMS) -> CUresult;
        }
        cuDevSmResourceSplit(result, nbGroups, input, remainder, flags, groupParams)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
pub unsafe fn cuDevSmResourceSplitByCount(result: *mut CUdevResource, nbGroups: *mut ::core::ffi::c_uint, input: *const CUdevResource, remaining: *mut CUdevResource, useFlags: ::core::ffi::c_uint, minCount: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevResource, *mut ::core::ffi::c_uint, *const CUdevResource, *mut CUdevResource, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevSmResourceSplitByCount") });
        _f(result, nbGroups, input, remaining, useFlags, minCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevSmResourceSplitByCount(result: *mut CUdevResource, nbGroups: *mut ::core::ffi::c_uint, input: *const CUdevResource, remaining: *mut CUdevResource, useFlags: ::core::ffi::c_uint, minCount: ::core::ffi::c_uint) -> CUresult;
        }
        cuDevSmResourceSplitByCount(result, nbGroups, input, remaining, useFlags, minCount)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDevSmResourceSplitByCount(result: *mut CUdevResource, nbGroups: *mut ::core::ffi::c_uint, input: *const CUdevResource, remainder: *mut CUdevResource, flags: ::core::ffi::c_uint, minCount: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevResource, *mut ::core::ffi::c_uint, *const CUdevResource, *mut CUdevResource, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevSmResourceSplitByCount") });
        _f(result, nbGroups, input, remainder, flags, minCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevSmResourceSplitByCount(result: *mut CUdevResource, nbGroups: *mut ::core::ffi::c_uint, input: *const CUdevResource, remainder: *mut CUdevResource, flags: ::core::ffi::c_uint, minCount: ::core::ffi::c_uint) -> CUresult;
        }
        cuDevSmResourceSplitByCount(result, nbGroups, input, remainder, flags, minCount)
    }
}
pub unsafe fn cuDeviceCanAccessPeer(canAccessPeer: *mut ::core::ffi::c_int, dev: CUdevice, peerDev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUdevice, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceCanAccessPeer") });
        _f(canAccessPeer, dev, peerDev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceCanAccessPeer(canAccessPeer: *mut ::core::ffi::c_int, dev: CUdevice, peerDev: CUdevice) -> CUresult;
        }
        cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev)
    }
}
pub unsafe fn cuDeviceComputeCapability(major: *mut ::core::ffi::c_int, minor: *mut ::core::ffi::c_int, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *mut ::core::ffi::c_int, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceComputeCapability") });
        _f(major, minor, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceComputeCapability(major: *mut ::core::ffi::c_int, minor: *mut ::core::ffi::c_int, dev: CUdevice) -> CUresult;
        }
        cuDeviceComputeCapability(major, minor, dev)
    }
}
pub unsafe fn cuDeviceGet(device: *mut CUdevice, ordinal: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevice, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGet") });
        _f(device, ordinal)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGet(device: *mut CUdevice, ordinal: ::core::ffi::c_int) -> CUresult;
        }
        cuDeviceGet(device, ordinal)
    }
}
pub unsafe fn cuDeviceGetAttribute(pi: *mut ::core::ffi::c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUdevice_attribute, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetAttribute") });
        _f(pi, attrib, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetAttribute(pi: *mut ::core::ffi::c_int, attrib: CUdevice_attribute, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetAttribute(pi, attrib, dev)
    }
}
pub unsafe fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pciBusId: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevice, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetByPCIBusId") });
        _f(dev, pciBusId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pciBusId: *const ::core::ffi::c_char) -> CUresult;
        }
        cuDeviceGetByPCIBusId(dev, pciBusId)
    }
}
pub unsafe fn cuDeviceGetCount(count: *mut ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetCount") });
        _f(count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetCount(count: *mut ::core::ffi::c_int) -> CUresult;
        }
        cuDeviceGetCount(count)
    }
}
pub unsafe fn cuDeviceGetDefaultMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemoryPool, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetDefaultMemPool") });
        _f(pool_out, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetDefaultMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetDefaultMemPool(pool_out, dev)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDeviceGetDevResource(device: CUdevice, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, *mut CUdevResource, CUdevResourceType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetDevResource") });
        _f(device, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetDevResource(device: CUdevice, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult;
        }
        cuDeviceGetDevResource(device, resource, type_)
    }
}
pub unsafe fn cuDeviceGetExecAffinitySupport(pi: *mut ::core::ffi::c_int, type_: CUexecAffinityType, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUexecAffinityType, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetExecAffinitySupport") });
        _f(pi, type_, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetExecAffinitySupport(pi: *mut ::core::ffi::c_int, type_: CUexecAffinityType, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetExecAffinitySupport(pi, type_, dev)
    }
}
pub unsafe fn cuDeviceGetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, CUgraphMem_attribute, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetGraphMemAttribute") });
        _f(device, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuDeviceGetGraphMemAttribute(device, attr, value)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDeviceGetHostAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const CUatomicOperation, count: ::core::ffi::c_uint, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, *const CUatomicOperation, ::core::ffi::c_uint, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetHostAtomicCapabilities") });
        _f(capabilities, operations, count, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetHostAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const CUatomicOperation, count: ::core::ffi::c_uint, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetHostAtomicCapabilities(capabilities, operations, count, dev)
    }
}
pub unsafe fn cuDeviceGetLuid(luid: *mut ::core::ffi::c_char, deviceNodeMask: *mut ::core::ffi::c_uint, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_char, *mut ::core::ffi::c_uint, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetLuid") });
        _f(luid, deviceNodeMask, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetLuid(luid: *mut ::core::ffi::c_char, deviceNodeMask: *mut ::core::ffi::c_uint, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetLuid(luid, deviceNodeMask, dev)
    }
}
pub unsafe fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemoryPool, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetMemPool") });
        _f(pool, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetMemPool(pool, dev)
    }
}
pub unsafe fn cuDeviceGetName(name: *mut ::core::ffi::c_char, len: ::core::ffi::c_int, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_char, ::core::ffi::c_int, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetName") });
        _f(name, len, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetName(name: *mut ::core::ffi::c_char, len: ::core::ffi::c_int, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetName(name, len, dev)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDeviceGetP2PAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const CUatomicOperation, count: ::core::ffi::c_uint, srcDevice: CUdevice, dstDevice: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, *const CUatomicOperation, ::core::ffi::c_uint, CUdevice, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetP2PAtomicCapabilities") });
        _f(capabilities, operations, count, srcDevice, dstDevice)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetP2PAtomicCapabilities(capabilities: *mut ::core::ffi::c_uint, operations: *const CUatomicOperation, count: ::core::ffi::c_uint, srcDevice: CUdevice, dstDevice: CUdevice) -> CUresult;
        }
        cuDeviceGetP2PAtomicCapabilities(capabilities, operations, count, srcDevice, dstDevice)
    }
}
pub unsafe fn cuDeviceGetP2PAttribute(value: *mut ::core::ffi::c_int, attrib: CUdevice_P2PAttribute, srcDevice: CUdevice, dstDevice: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUdevice_P2PAttribute, CUdevice, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetP2PAttribute") });
        _f(value, attrib, srcDevice, dstDevice)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetP2PAttribute(value: *mut ::core::ffi::c_int, attrib: CUdevice_P2PAttribute, srcDevice: CUdevice, dstDevice: CUdevice) -> CUresult;
        }
        cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice)
    }
}
pub unsafe fn cuDeviceGetPCIBusId(pciBusId: *mut ::core::ffi::c_char, len: ::core::ffi::c_int, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_char, ::core::ffi::c_int, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetPCIBusId") });
        _f(pciBusId, len, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetPCIBusId(pciBusId: *mut ::core::ffi::c_char, len: ::core::ffi::c_int, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetPCIBusId(pciBusId, len, dev)
    }
}
pub unsafe fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdevprop, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetProperties") });
        _f(prop, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetProperties(prop, dev)
    }
}
pub unsafe fn cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: *mut usize, format: CUarray_format, numChannels: ::core::ffi::c_uint, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, CUarray_format, ::core::ffi::c_uint, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetTexture1DLinearMaxWidth") });
        _f(maxWidthInElements, format, numChannels, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements: *mut usize, format: CUarray_format, numChannels: ::core::ffi::c_uint, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetTexture1DLinearMaxWidth(maxWidthInElements, format, numChannels, dev)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetUuid") });
        _f(uuid, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetUuid(uuid, dev)
    }
}
pub unsafe fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGetUuid_v2") });
        _f(uuid, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
        }
        cuDeviceGetUuid_v2(uuid, dev)
    }
}
pub unsafe fn cuDeviceGraphMemTrim(device: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceGraphMemTrim") });
        _f(device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceGraphMemTrim(device: CUdevice) -> CUresult;
        }
        cuDeviceGraphMemTrim(device)
    }
}
pub unsafe fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut ::core::ffi::c_uint, active: *mut ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, *mut ::core::ffi::c_uint, *mut ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevicePrimaryCtxGetState") });
        _f(dev, flags, active)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevicePrimaryCtxGetState(dev: CUdevice, flags: *mut ::core::ffi::c_uint, active: *mut ::core::ffi::c_int) -> CUresult;
        }
        cuDevicePrimaryCtxGetState(dev, flags, active)
    }
}
pub unsafe fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevicePrimaryCtxRelease_v2") });
        _f(dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
        }
        cuDevicePrimaryCtxRelease_v2(dev)
    }
}
pub unsafe fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevicePrimaryCtxReset_v2") });
        _f(dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult;
        }
        cuDevicePrimaryCtxReset_v2(dev)
    }
}
pub unsafe fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevicePrimaryCtxRetain") });
        _f(pctx, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
        }
        cuDevicePrimaryCtxRetain(pctx, dev)
    }
}
pub unsafe fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDevicePrimaryCtxSetFlags_v2") });
        _f(dev, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuDevicePrimaryCtxSetFlags_v2(dev, flags)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDeviceRegisterAsyncNotification(device: CUdevice, callbackFunc: CUasyncCallback, userData: *mut ::core::ffi::c_void, callback: *mut CUasyncCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, CUasyncCallback, *mut ::core::ffi::c_void, *mut CUasyncCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceRegisterAsyncNotification") });
        _f(device, callbackFunc, userData, callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceRegisterAsyncNotification(device: CUdevice, callbackFunc: CUasyncCallback, userData: *mut ::core::ffi::c_void, callback: *mut CUasyncCallbackHandle) -> CUresult;
        }
        cuDeviceRegisterAsyncNotification(device, callbackFunc, userData, callback)
    }
}
pub unsafe fn cuDeviceSetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, CUgraphMem_attribute, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceSetGraphMemAttribute") });
        _f(device, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceSetGraphMemAttribute(device: CUdevice, attr: CUgraphMem_attribute, value: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuDeviceSetGraphMemAttribute(device, attr, value)
    }
}
pub unsafe fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, CUmemoryPool) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceSetMemPool") });
        _f(dev, pool)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult;
        }
        cuDeviceSetMemPool(dev, pool)
    }
}
pub unsafe fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceTotalMem_v2") });
        _f(bytes, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
        }
        cuDeviceTotalMem_v2(bytes, dev)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuDeviceUnregisterAsyncNotification(device: CUdevice, callback: CUasyncCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdevice, CUasyncCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDeviceUnregisterAsyncNotification") });
        _f(device, callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDeviceUnregisterAsyncNotification(device: CUdevice, callback: CUasyncCallbackHandle) -> CUresult;
        }
        cuDeviceUnregisterAsyncNotification(device, callback)
    }
}
pub unsafe fn cuDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuDriverGetVersion") });
        _f(driverVersion)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> CUresult;
        }
        cuDriverGetVersion(driverVersion)
    }
}
pub unsafe fn cuEventCreate(phEvent: *mut CUevent, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUevent, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventCreate") });
        _f(phEvent, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventCreate(phEvent: *mut CUevent, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuEventCreate(phEvent, Flags)
    }
}
pub unsafe fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventDestroy_v2") });
        _f(hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
        }
        cuEventDestroy_v2(hEvent)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventElapsedTime") });
        _f(pMilliseconds, hStart, hEnd)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;
        }
        cuEventElapsedTime(pMilliseconds, hStart, hEnd)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuEventElapsedTime_v2(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventElapsedTime_v2") });
        _f(pMilliseconds, hStart, hEnd)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventElapsedTime_v2(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;
        }
        cuEventElapsedTime_v2(pMilliseconds, hStart, hEnd)
    }
}
pub unsafe fn cuEventQuery(hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventQuery") });
        _f(hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventQuery(hEvent: CUevent) -> CUresult;
        }
        cuEventQuery(hEvent)
    }
}
pub unsafe fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventRecord") });
        _f(hEvent, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
        }
        cuEventRecord(hEvent, hStream)
    }
}
pub unsafe fn cuEventRecordWithFlags(hEvent: CUevent, hStream: CUstream, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUevent, CUstream, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventRecordWithFlags") });
        _f(hEvent, hStream, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventRecordWithFlags(hEvent: CUevent, hStream: CUstream, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuEventRecordWithFlags(hEvent, hStream, flags)
    }
}
pub unsafe fn cuEventSynchronize(hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuEventSynchronize") });
        _f(hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
        }
        cuEventSynchronize(hEvent)
    }
}
pub unsafe fn cuExternalMemoryGetMappedBuffer(devPtr: *mut CUdeviceptr, extMem: CUexternalMemory, bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, CUexternalMemory, *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuExternalMemoryGetMappedBuffer") });
        _f(devPtr, extMem, bufferDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuExternalMemoryGetMappedBuffer(devPtr: *mut CUdeviceptr, extMem: CUexternalMemory, bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC) -> CUresult;
        }
        cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
    }
}
pub unsafe fn cuExternalMemoryGetMappedMipmappedArray(mipmap: *mut CUmipmappedArray, extMem: CUexternalMemory, mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmipmappedArray, CUexternalMemory, *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuExternalMemoryGetMappedMipmappedArray") });
        _f(mipmap, extMem, mipmapDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuExternalMemoryGetMappedMipmappedArray(mipmap: *mut CUmipmappedArray, extMem: CUexternalMemory, mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC) -> CUresult;
        }
        cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
    }
}
pub unsafe fn cuFlushGPUDirectRDMAWrites(target: CUflushGPUDirectRDMAWritesTarget, scope: CUflushGPUDirectRDMAWritesScope) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFlushGPUDirectRDMAWrites") });
        _f(target, scope)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFlushGPUDirectRDMAWrites(target: CUflushGPUDirectRDMAWritesTarget, scope: CUflushGPUDirectRDMAWritesScope) -> CUresult;
        }
        cuFlushGPUDirectRDMAWrites(target, scope)
    }
}
pub unsafe fn cuFuncGetAttribute(pi: *mut ::core::ffi::c_int, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUfunction_attribute, CUfunction) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncGetAttribute") });
        _f(pi, attrib, hfunc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncGetAttribute(pi: *mut ::core::ffi::c_int, attrib: CUfunction_attribute, hfunc: CUfunction) -> CUresult;
        }
        cuFuncGetAttribute(pi, attrib, hfunc)
    }
}
pub unsafe fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmodule, CUfunction) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncGetModule") });
        _f(hmod, hfunc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult;
        }
        cuFuncGetModule(hmod, hfunc)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuFuncGetName(name: *mut *const ::core::ffi::c_char, hfunc: CUfunction) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const ::core::ffi::c_char, CUfunction) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncGetName") });
        _f(name, hfunc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncGetName(name: *mut *const ::core::ffi::c_char, hfunc: CUfunction) -> CUresult;
        }
        cuFuncGetName(name, hfunc)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuFuncGetParamCount(func: CUfunction, paramCount: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncGetParamCount") });
        _f(func, paramCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncGetParamCount(func: CUfunction, paramCount: *mut usize) -> CUresult;
        }
        cuFuncGetParamCount(func, paramCount)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuFuncGetParamInfo(func: CUfunction, paramIndex: usize, paramOffset: *mut usize, paramSize: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, usize, *mut usize, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncGetParamInfo") });
        _f(func, paramIndex, paramOffset, paramSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncGetParamInfo(func: CUfunction, paramIndex: usize, paramOffset: *mut usize, paramSize: *mut usize) -> CUresult;
        }
        cuFuncGetParamInfo(func, paramIndex, paramOffset, paramSize)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuFuncIsLoaded(state: *mut CUfunctionLoadingState, function: CUfunction) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfunctionLoadingState, CUfunction) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncIsLoaded") });
        _f(state, function)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncIsLoaded(state: *mut CUfunctionLoadingState, function: CUfunction) -> CUresult;
        }
        cuFuncIsLoaded(state, function)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuFuncLoad(function: CUfunction) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncLoad") });
        _f(function)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncLoad(function: CUfunction) -> CUresult;
        }
        cuFuncLoad(function)
    }
}
pub unsafe fn cuFuncSetAttribute(hfunc: CUfunction, attrib: CUfunction_attribute, value: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, CUfunction_attribute, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncSetAttribute") });
        _f(hfunc, attrib, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncSetAttribute(hfunc: CUfunction, attrib: CUfunction_attribute, value: ::core::ffi::c_int) -> CUresult;
        }
        cuFuncSetAttribute(hfunc, attrib, value)
    }
}
pub unsafe fn cuFuncSetBlockShape(hfunc: CUfunction, x: ::core::ffi::c_int, y: ::core::ffi::c_int, z: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, ::core::ffi::c_int, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncSetBlockShape") });
        _f(hfunc, x, y, z)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncSetBlockShape(hfunc: CUfunction, x: ::core::ffi::c_int, y: ::core::ffi::c_int, z: ::core::ffi::c_int) -> CUresult;
        }
        cuFuncSetBlockShape(hfunc, x, y, z)
    }
}
pub unsafe fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, CUfunc_cache) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncSetCacheConfig") });
        _f(hfunc, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult;
        }
        cuFuncSetCacheConfig(hfunc, config)
    }
}
pub unsafe fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, CUsharedconfig) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncSetSharedMemConfig") });
        _f(hfunc, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: CUsharedconfig) -> CUresult;
        }
        cuFuncSetSharedMemConfig(hfunc, config)
    }
}
pub unsafe fn cuFuncSetSharedSize(hfunc: CUfunction, bytes: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuFuncSetSharedSize") });
        _f(hfunc, bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuFuncSetSharedSize(hfunc: CUfunction, bytes: ::core::ffi::c_uint) -> CUresult;
        }
        cuFuncSetSharedSize(hfunc, bytes)
    }
}
pub unsafe fn cuGetErrorName(error: CUresult, pStr: *mut *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUresult, *mut *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGetErrorName") });
        _f(error, pStr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGetErrorName(error: CUresult, pStr: *mut *const ::core::ffi::c_char) -> CUresult;
        }
        cuGetErrorName(error, pStr)
    }
}
pub unsafe fn cuGetErrorString(error: CUresult, pStr: *mut *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUresult, *mut *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGetErrorString") });
        _f(error, pStr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGetErrorString(error: CUresult, pStr: *mut *const ::core::ffi::c_char) -> CUresult;
        }
        cuGetErrorString(error, pStr)
    }
}
pub unsafe fn cuGetExportTable(ppExportTable: *mut *const ::core::ffi::c_void, pExportTableId: *const CUuuid) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const ::core::ffi::c_void, *const CUuuid) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGetExportTable") });
        _f(ppExportTable, pExportTableId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGetExportTable(ppExportTable: *mut *const ::core::ffi::c_void, pExportTableId: *const CUuuid) -> CUresult;
        }
        cuGetExportTable(ppExportTable, pExportTableId)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGetProcAddress(symbol: *const ::core::ffi::c_char, pfn: *mut *mut ::core::ffi::c_void, cudaVersion: ::core::ffi::c_int, flags: cuuint64_t) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, cuuint64_t) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGetProcAddress") });
        _f(symbol, pfn, cudaVersion, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGetProcAddress(symbol: *const ::core::ffi::c_char, pfn: *mut *mut ::core::ffi::c_void, cudaVersion: ::core::ffi::c_int, flags: cuuint64_t) -> CUresult;
        }
        cuGetProcAddress(symbol, pfn, cudaVersion, flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGetProcAddress_v2(symbol: *const ::core::ffi::c_char, pfn: *mut *mut ::core::ffi::c_void, cudaVersion: ::core::ffi::c_int, flags: cuuint64_t, symbolStatus: *mut CUdriverProcAddressQueryResult) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *mut *mut ::core::ffi::c_void, ::core::ffi::c_int, cuuint64_t, *mut CUdriverProcAddressQueryResult) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGetProcAddress_v2") });
        _f(symbol, pfn, cudaVersion, flags, symbolStatus)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGetProcAddress_v2(symbol: *const ::core::ffi::c_char, pfn: *mut *mut ::core::ffi::c_void, cudaVersion: ::core::ffi::c_int, flags: cuuint64_t, symbolStatus: *mut CUdriverProcAddressQueryResult) -> CUresult;
        }
        cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphAddBatchMemOpNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddBatchMemOpNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddBatchMemOpNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddBatchMemOpNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cuGraphAddChildGraphNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, childGraph: CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddChildGraphNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, childGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddChildGraphNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, childGraph: CUgraph) -> CUresult;
        }
        cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies, childGraph)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuGraphAddDependencies(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, numDependencies: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *const CUgraphNode, *const CUgraphNode, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddDependencies") });
        _f(hGraph, from, to, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddDependencies(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, numDependencies: usize) -> CUresult;
        }
        cuGraphAddDependencies(hGraph, from, to, numDependencies)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphAddDependencies_v2(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, edgeData: *const CUgraphEdgeData, numDependencies: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *const CUgraphNode, *const CUgraphNode, *const CUgraphEdgeData, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddDependencies_v2") });
        _f(hGraph, from, to, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddDependencies_v2(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, edgeData: *const CUgraphEdgeData, numDependencies: usize) -> CUresult;
        }
        cuGraphAddDependencies_v2(hGraph, from, to, edgeData, numDependencies)
    }
}
pub unsafe fn cuGraphAddEmptyNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddEmptyNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddEmptyNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize) -> CUresult;
        }
        cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies)
    }
}
pub unsafe fn cuGraphAddEventRecordNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddEventRecordNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddEventRecordNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, event: CUevent) -> CUresult;
        }
        cuGraphAddEventRecordNode(phGraphNode, hGraph, dependencies, numDependencies, event)
    }
}
pub unsafe fn cuGraphAddEventWaitNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddEventWaitNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddEventWaitNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, event: CUevent) -> CUresult;
        }
        cuGraphAddEventWaitNode(phGraphNode, hGraph, dependencies, numDependencies, event)
    }
}
pub unsafe fn cuGraphAddExternalSemaphoresSignalNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddExternalSemaphoresSignalNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddExternalSemaphoresSignalNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddExternalSemaphoresSignalNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cuGraphAddExternalSemaphoresWaitNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddExternalSemaphoresWaitNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddExternalSemaphoresWaitNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddExternalSemaphoresWaitNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cuGraphAddHostNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_HOST_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddHostNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddHostNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGraphAddKernelNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddKernelNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddKernelNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphAddKernelNode_v2(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddKernelNode_v2") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddKernelNode_v2(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cuGraphAddMemAllocNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddMemAllocNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddMemAllocNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult;
        }
        cuGraphAddMemAllocNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
pub unsafe fn cuGraphAddMemFreeNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, dptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddMemFreeNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, dptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddMemFreeNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, dptr: CUdeviceptr) -> CUresult;
        }
        cuGraphAddMemFreeNode(phGraphNode, hGraph, dependencies, numDependencies, dptr)
    }
}
pub unsafe fn cuGraphAddMemcpyNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, copyParams: *const CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_MEMCPY3D, CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddMemcpyNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddMemcpyNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, copyParams: *const CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult;
        }
        cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
    }
}
pub unsafe fn cuGraphAddMemsetNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, memsetParams: *const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *const CUDA_MEMSET_NODE_PARAMS, CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddMemsetNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddMemsetNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, memsetParams: *const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult;
        }
        cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuGraphAddNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUgraphNodeParams) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, usize, *mut CUgraphNodeParams) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddNode") });
        _f(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddNode(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, numDependencies: usize, nodeParams: *mut CUgraphNodeParams) -> CUresult;
        }
        cuGraphAddNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphAddNode_v2(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, dependencyData: *const CUgraphEdgeData, numDependencies: usize, nodeParams: *mut CUgraphNodeParams) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraph, *const CUgraphNode, *const CUgraphEdgeData, usize, *mut CUgraphNodeParams) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphAddNode_v2") });
        _f(phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphAddNode_v2(phGraphNode: *mut CUgraphNode, hGraph: CUgraph, dependencies: *const CUgraphNode, dependencyData: *const CUgraphEdgeData, numDependencies: usize, nodeParams: *mut CUgraphNodeParams) -> CUresult;
        }
        cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData, numDependencies, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphBatchMemOpNodeGetParams(hNode: CUgraphNode, nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphBatchMemOpNodeGetParams") });
        _f(hNode, nodeParams_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphBatchMemOpNodeGetParams(hNode: CUgraphNode, nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        }
        cuGraphBatchMemOpNodeGetParams(hNode, nodeParams_out)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphBatchMemOpNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphBatchMemOpNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphBatchMemOpNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        }
        cuGraphBatchMemOpNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphChildGraphNodeGetGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphChildGraphNodeGetGraph") });
        _f(hNode, phGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphChildGraphNodeGetGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult;
        }
        cuGraphChildGraphNodeGetGraph(hNode, phGraph)
    }
}
pub unsafe fn cuGraphClone(phGraphClone: *mut CUgraph, originalGraph: CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraph, CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphClone") });
        _f(phGraphClone, originalGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphClone(phGraphClone: *mut CUgraph, originalGraph: CUgraph) -> CUresult;
        }
        cuGraphClone(phGraphClone, originalGraph)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphConditionalHandleCreate(pHandle_out: *mut CUgraphConditionalHandle, hGraph: CUgraph, ctx: CUcontext, defaultLaunchValue: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphConditionalHandle, CUgraph, CUcontext, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphConditionalHandleCreate") });
        _f(pHandle_out, hGraph, ctx, defaultLaunchValue, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphConditionalHandleCreate(pHandle_out: *mut CUgraphConditionalHandle, hGraph: CUgraph, ctx: CUcontext, defaultLaunchValue: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphConditionalHandleCreate(pHandle_out, hGraph, ctx, defaultLaunchValue, flags)
    }
}
pub unsafe fn cuGraphCreate(phGraph: *mut CUgraph, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraph, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphCreate") });
        _f(phGraph, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphCreate(phGraph: *mut CUgraph, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphCreate(phGraph, flags)
    }
}
pub unsafe fn cuGraphDebugDotPrint(hGraph: CUgraph, path: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *const ::core::ffi::c_char, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphDebugDotPrint") });
        _f(hGraph, path, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphDebugDotPrint(hGraph: CUgraph, path: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphDebugDotPrint(hGraph, path, flags)
    }
}
pub unsafe fn cuGraphDestroy(hGraph: CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphDestroy") });
        _f(hGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphDestroy(hGraph: CUgraph) -> CUresult;
        }
        cuGraphDestroy(hGraph)
    }
}
pub unsafe fn cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphDestroyNode") });
        _f(hNode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult;
        }
        cuGraphDestroyNode(hNode)
    }
}
pub unsafe fn cuGraphEventRecordNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphEventRecordNodeGetEvent") });
        _f(hNode, event_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphEventRecordNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult;
        }
        cuGraphEventRecordNodeGetEvent(hNode, event_out)
    }
}
pub unsafe fn cuGraphEventRecordNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphEventRecordNodeSetEvent") });
        _f(hNode, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphEventRecordNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult;
        }
        cuGraphEventRecordNodeSetEvent(hNode, event)
    }
}
pub unsafe fn cuGraphEventWaitNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphEventWaitNodeGetEvent") });
        _f(hNode, event_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphEventWaitNodeGetEvent(hNode: CUgraphNode, event_out: *mut CUevent) -> CUresult;
        }
        cuGraphEventWaitNodeGetEvent(hNode, event_out)
    }
}
pub unsafe fn cuGraphEventWaitNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphEventWaitNodeSetEvent") });
        _f(hNode, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphEventWaitNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult;
        }
        cuGraphEventWaitNodeSetEvent(hNode, event)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphExecBatchMemOpNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecBatchMemOpNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecBatchMemOpNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS) -> CUresult;
        }
        cuGraphExecBatchMemOpNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
pub unsafe fn cuGraphExecChildGraphNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, childGraph: CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecChildGraphNodeSetParams") });
        _f(hGraphExec, hNode, childGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecChildGraphNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, childGraph: CUgraph) -> CUresult;
        }
        cuGraphExecChildGraphNodeSetParams(hGraphExec, hNode, childGraph)
    }
}
pub unsafe fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecDestroy") });
        _f(hGraphExec)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult;
        }
        cuGraphExecDestroy(hGraphExec)
    }
}
pub unsafe fn cuGraphExecEventRecordNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecEventRecordNodeSetEvent") });
        _f(hGraphExec, hNode, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecEventRecordNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult;
        }
        cuGraphExecEventRecordNodeSetEvent(hGraphExec, hNode, event)
    }
}
pub unsafe fn cuGraphExecEventWaitNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecEventWaitNodeSetEvent") });
        _f(hGraphExec, hNode, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecEventWaitNodeSetEvent(hGraphExec: CUgraphExec, hNode: CUgraphNode, event: CUevent) -> CUresult;
        }
        cuGraphExecEventWaitNodeSetEvent(hGraphExec, hNode, event)
    }
}
pub unsafe fn cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecExternalSemaphoresSignalNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        }
        cuGraphExecExternalSemaphoresSignalNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
pub unsafe fn cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecExternalSemaphoresWaitNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        }
        cuGraphExecExternalSemaphoresWaitNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphExecGetFlags(hGraphExec: CUgraphExec, flags: *mut cuuint64_t) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, *mut cuuint64_t) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecGetFlags") });
        _f(hGraphExec, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecGetFlags(hGraphExec: CUgraphExec, flags: *mut cuuint64_t) -> CUresult;
        }
        cuGraphExecGetFlags(hGraphExec, flags)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphExecGetId(hGraphExec: CUgraphExec, graphId: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, *mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecGetId") });
        _f(hGraphExec, graphId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecGetId(hGraphExec: CUgraphExec, graphId: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphExecGetId(hGraphExec, graphId)
    }
}
pub unsafe fn cuGraphExecHostNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_HOST_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecHostNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecHostNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult;
        }
        cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGraphExecKernelNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecKernelNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecKernelNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphExecKernelNodeSetParams_v2(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecKernelNodeSetParams_v2") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecKernelNodeSetParams_v2(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams)
    }
}
pub unsafe fn cuGraphExecMemcpyNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, copyParams: *const CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_MEMCPY3D, CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecMemcpyNodeSetParams") });
        _f(hGraphExec, hNode, copyParams, ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecMemcpyNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, copyParams: *const CUDA_MEMCPY3D, ctx: CUcontext) -> CUresult;
        }
        cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)
    }
}
pub unsafe fn cuGraphExecMemsetNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, memsetParams: *const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_MEMSET_NODE_PARAMS, CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecMemsetNodeSetParams") });
        _f(hGraphExec, hNode, memsetParams, ctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecMemsetNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, memsetParams: *const CUDA_MEMSET_NODE_PARAMS, ctx: CUcontext) -> CUresult;
        }
        cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphExecNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *mut CUgraphNodeParams) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecNodeSetParams") });
        _f(hGraphExec, hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecNodeSetParams(hGraphExec: CUgraphExec, hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult;
        }
        cuGraphExecNodeSetParams(hGraphExec, hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGraphExecUpdate(hGraphExec: CUgraphExec, hGraph: CUgraph, hErrorNode_out: *mut CUgraphNode, updateResult_out: *mut CUgraphExecUpdateResult) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraph, *mut CUgraphNode, *mut CUgraphExecUpdateResult) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecUpdate") });
        _f(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecUpdate(hGraphExec: CUgraphExec, hGraph: CUgraph, hErrorNode_out: *mut CUgraphNode, updateResult_out: *mut CUgraphExecUpdateResult) -> CUresult;
        }
        cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphExecUpdate_v2(hGraphExec: CUgraphExec, hGraph: CUgraph, resultInfo: *mut CUgraphExecUpdateResultInfo) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraph, *mut CUgraphExecUpdateResultInfo) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExecUpdate_v2") });
        _f(hGraphExec, hGraph, resultInfo)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExecUpdate_v2(hGraphExec: CUgraphExec, hGraph: CUgraph, resultInfo: *mut CUgraphExecUpdateResultInfo) -> CUresult;
        }
        cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo)
    }
}
pub unsafe fn cuGraphExternalSemaphoresSignalNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExternalSemaphoresSignalNodeGetParams") });
        _f(hNode, params_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExternalSemaphoresSignalNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        }
        cuGraphExternalSemaphoresSignalNodeGetParams(hNode, params_out)
    }
}
pub unsafe fn cuGraphExternalSemaphoresSignalNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExternalSemaphoresSignalNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExternalSemaphoresSignalNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS) -> CUresult;
        }
        cuGraphExternalSemaphoresSignalNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphExternalSemaphoresWaitNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExternalSemaphoresWaitNodeGetParams") });
        _f(hNode, params_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExternalSemaphoresWaitNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        }
        cuGraphExternalSemaphoresWaitNodeGetParams(hNode, params_out)
    }
}
pub unsafe fn cuGraphExternalSemaphoresWaitNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphExternalSemaphoresWaitNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphExternalSemaphoresWaitNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS) -> CUresult;
        }
        cuGraphExternalSemaphoresWaitNodeSetParams(hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuGraphGetEdges(hGraph: CUgraph, from: *mut CUgraphNode, to: *mut CUgraphNode, numEdges: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *mut CUgraphNode, *mut CUgraphNode, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphGetEdges") });
        _f(hGraph, from, to, numEdges)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphGetEdges(hGraph: CUgraph, from: *mut CUgraphNode, to: *mut CUgraphNode, numEdges: *mut usize) -> CUresult;
        }
        cuGraphGetEdges(hGraph, from, to, numEdges)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphGetEdges_v2(hGraph: CUgraph, from: *mut CUgraphNode, to: *mut CUgraphNode, edgeData: *mut CUgraphEdgeData, numEdges: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *mut CUgraphNode, *mut CUgraphNode, *mut CUgraphEdgeData, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphGetEdges_v2") });
        _f(hGraph, from, to, edgeData, numEdges)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphGetEdges_v2(hGraph: CUgraph, from: *mut CUgraphNode, to: *mut CUgraphNode, edgeData: *mut CUgraphEdgeData, numEdges: *mut usize) -> CUresult;
        }
        cuGraphGetEdges_v2(hGraph, from, to, edgeData, numEdges)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphGetId(hGraph: CUgraph, graphId: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphGetId") });
        _f(hGraph, graphId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphGetId(hGraph: CUgraph, graphId: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphGetId(hGraph, graphId)
    }
}
pub unsafe fn cuGraphGetNodes(hGraph: CUgraph, nodes: *mut CUgraphNode, numNodes: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *mut CUgraphNode, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphGetNodes") });
        _f(hGraph, nodes, numNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphGetNodes(hGraph: CUgraph, nodes: *mut CUgraphNode, numNodes: *mut usize) -> CUresult;
        }
        cuGraphGetNodes(hGraph, nodes, numNodes)
    }
}
pub unsafe fn cuGraphGetRootNodes(hGraph: CUgraph, rootNodes: *mut CUgraphNode, numRootNodes: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *mut CUgraphNode, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphGetRootNodes") });
        _f(hGraph, rootNodes, numRootNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphGetRootNodes(hGraph: CUgraph, rootNodes: *mut CUgraphNode, numRootNodes: *mut usize) -> CUresult;
        }
        cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes)
    }
}
pub unsafe fn cuGraphHostNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_HOST_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_HOST_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphHostNodeGetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphHostNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_HOST_NODE_PARAMS) -> CUresult;
        }
        cuGraphHostNodeGetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphHostNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_HOST_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphHostNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphHostNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_HOST_NODE_PARAMS) -> CUresult;
        }
        cuGraphHostNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphInstantiateWithFlags(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphInstantiateWithFlags") });
        _f(phGraphExec, hGraph, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphInstantiateWithFlags(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuGraphInstantiateWithFlags(phGraphExec, hGraph, flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphInstantiateWithParams(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, *mut CUDA_GRAPH_INSTANTIATE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphInstantiateWithParams") });
        _f(phGraphExec, hGraph, instantiateParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphInstantiateWithParams(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS) -> CUresult;
        }
        cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGraphInstantiate_v2(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, phErrorNode: *mut CUgraphNode, logBuffer: *mut ::core::ffi::c_char, bufferSize: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphExec, CUgraph, *mut CUgraphNode, *mut ::core::ffi::c_char, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphInstantiate_v2") });
        _f(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphInstantiate_v2(phGraphExec: *mut CUgraphExec, hGraph: CUgraph, phErrorNode: *mut CUgraphNode, logBuffer: *mut ::core::ffi::c_char, bufferSize: usize) -> CUresult;
        }
        cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
    }
}
pub unsafe fn cuGraphKernelNodeCopyAttributes(dst: CUgraphNode, src: CUgraphNode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, CUgraphNode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeCopyAttributes") });
        _f(dst, src)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeCopyAttributes(dst: CUgraphNode, src: CUgraphNode) -> CUresult;
        }
        cuGraphKernelNodeCopyAttributes(dst, src)
    }
}
pub unsafe fn cuGraphKernelNodeGetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value_out: *mut CUkernelNodeAttrValue) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, CUkernelNodeAttrID, *mut CUkernelNodeAttrValue) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeGetAttribute") });
        _f(hNode, attr, value_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeGetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value_out: *mut CUkernelNodeAttrValue) -> CUresult;
        }
        cuGraphKernelNodeGetAttribute(hNode, attr, value_out)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGraphKernelNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeGetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphKernelNodeGetParams(hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphKernelNodeGetParams_v2(hNode: CUgraphNode, nodeParams: *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeGetParams_v2") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeGetParams_v2(hNode: CUgraphNode, nodeParams: *mut CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphKernelNodeGetParams_v2(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphKernelNodeSetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value: *const CUkernelNodeAttrValue) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, CUkernelNodeAttrID, *const CUkernelNodeAttrValue) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeSetAttribute") });
        _f(hNode, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeSetAttribute(hNode: CUgraphNode, attr: CUkernelNodeAttrID, value: *const CUkernelNodeAttrValue) -> CUresult;
        }
        cuGraphKernelNodeSetAttribute(hNode, attr, value)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuGraphKernelNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphKernelNodeSetParams(hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphKernelNodeSetParams_v2(hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphKernelNodeSetParams_v2") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphKernelNodeSetParams_v2(hNode: CUgraphNode, nodeParams: *const CUDA_KERNEL_NODE_PARAMS) -> CUresult;
        }
        cuGraphKernelNodeSetParams_v2(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphLaunch") });
        _f(hGraphExec, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;
        }
        cuGraphLaunch(hGraphExec, hStream)
    }
}
pub unsafe fn cuGraphMemAllocNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphMemAllocNodeGetParams") });
        _f(hNode, params_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphMemAllocNodeGetParams(hNode: CUgraphNode, params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS) -> CUresult;
        }
        cuGraphMemAllocNodeGetParams(hNode, params_out)
    }
}
pub unsafe fn cuGraphMemFreeNodeGetParams(hNode: CUgraphNode, dptr_out: *mut CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphMemFreeNodeGetParams") });
        _f(hNode, dptr_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphMemFreeNodeGetParams(hNode: CUgraphNode, dptr_out: *mut CUdeviceptr) -> CUresult;
        }
        cuGraphMemFreeNodeGetParams(hNode, dptr_out)
    }
}
pub unsafe fn cuGraphMemcpyNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_MEMCPY3D) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_MEMCPY3D) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphMemcpyNodeGetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphMemcpyNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_MEMCPY3D) -> CUresult;
        }
        cuGraphMemcpyNodeGetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphMemcpyNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_MEMCPY3D) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_MEMCPY3D) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphMemcpyNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphMemcpyNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_MEMCPY3D) -> CUresult;
        }
        cuGraphMemcpyNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphMemsetNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_MEMSET_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUDA_MEMSET_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphMemsetNodeGetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphMemsetNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUDA_MEMSET_NODE_PARAMS) -> CUresult;
        }
        cuGraphMemsetNodeGetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphMemsetNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_MEMSET_NODE_PARAMS) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *const CUDA_MEMSET_NODE_PARAMS) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphMemsetNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphMemsetNodeSetParams(hNode: CUgraphNode, nodeParams: *const CUDA_MEMSET_NODE_PARAMS) -> CUresult;
        }
        cuGraphMemsetNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphNodeFindInClone(phNode: *mut CUgraphNode, hOriginalNode: CUgraphNode, hClonedGraph: CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgraphNode, CUgraphNode, CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeFindInClone") });
        _f(phNode, hOriginalNode, hClonedGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeFindInClone(phNode: *mut CUgraphNode, hOriginalNode: CUgraphNode, hClonedGraph: CUgraph) -> CUresult;
        }
        cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetContainingGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetContainingGraph") });
        _f(hNode, phGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetContainingGraph(hNode: CUgraphNode, phGraph: *mut CUgraph) -> CUresult;
        }
        cuGraphNodeGetContainingGraph(hNode, phGraph)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuGraphNodeGetDependencies(hNode: CUgraphNode, dependencies: *mut CUgraphNode, numDependencies: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNode, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetDependencies") });
        _f(hNode, dependencies, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetDependencies(hNode: CUgraphNode, dependencies: *mut CUgraphNode, numDependencies: *mut usize) -> CUresult;
        }
        cuGraphNodeGetDependencies(hNode, dependencies, numDependencies)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetDependencies_v2(hNode: CUgraphNode, dependencies: *mut CUgraphNode, edgeData: *mut CUgraphEdgeData, numDependencies: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNode, *mut CUgraphEdgeData, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetDependencies_v2") });
        _f(hNode, dependencies, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetDependencies_v2(hNode: CUgraphNode, dependencies: *mut CUgraphNode, edgeData: *mut CUgraphEdgeData, numDependencies: *mut usize) -> CUresult;
        }
        cuGraphNodeGetDependencies_v2(hNode, dependencies, edgeData, numDependencies)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuGraphNodeGetDependentNodes(hNode: CUgraphNode, dependentNodes: *mut CUgraphNode, numDependentNodes: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNode, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetDependentNodes") });
        _f(hNode, dependentNodes, numDependentNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetDependentNodes(hNode: CUgraphNode, dependentNodes: *mut CUgraphNode, numDependentNodes: *mut usize) -> CUresult;
        }
        cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetDependentNodes_v2(hNode: CUgraphNode, dependentNodes: *mut CUgraphNode, edgeData: *mut CUgraphEdgeData, numDependentNodes: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNode, *mut CUgraphEdgeData, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetDependentNodes_v2") });
        _f(hNode, dependentNodes, edgeData, numDependentNodes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetDependentNodes_v2(hNode: CUgraphNode, dependentNodes: *mut CUgraphNode, edgeData: *mut CUgraphEdgeData, numDependentNodes: *mut usize) -> CUresult;
        }
        cuGraphNodeGetDependentNodes_v2(hNode, dependentNodes, edgeData, numDependentNodes)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, *mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetEnabled") });
        _f(hGraphExec, hNode, isEnabled)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphNodeGetEnabled(hGraphExec, hNode, isEnabled)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetLocalId(hNode: CUgraphNode, nodeId: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetLocalId") });
        _f(hNode, nodeId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetLocalId(hNode: CUgraphNode, nodeId: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphNodeGetLocalId(hNode, nodeId)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNodeParams) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetParams(hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult;
        }
        cuGraphNodeGetParams(hNode, nodeParams)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeGetToolsId(hNode: CUgraphNode, toolsNodeId: *mut ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetToolsId") });
        _f(hNode, toolsNodeId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetToolsId(hNode: CUgraphNode, toolsNodeId: *mut ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuGraphNodeGetToolsId(hNode, toolsNodeId)
    }
}
pub unsafe fn cuGraphNodeGetType(hNode: CUgraphNode, type_: *mut CUgraphNodeType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNodeType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeGetType") });
        _f(hNode, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeGetType(hNode: CUgraphNode, type_: *mut CUgraphNodeType) -> CUresult;
        }
        cuGraphNodeGetType(hNode, type_)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeSetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUgraphNode, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeSetEnabled") });
        _f(hGraphExec, hNode, isEnabled)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeSetEnabled(hGraphExec: CUgraphExec, hNode: CUgraphNode, isEnabled: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphNodeSetEnabled(hGraphExec, hNode, isEnabled)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphNodeSetParams(hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphNode, *mut CUgraphNodeParams) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphNodeSetParams") });
        _f(hNode, nodeParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphNodeSetParams(hNode: CUgraphNode, nodeParams: *mut CUgraphNodeParams) -> CUresult;
        }
        cuGraphNodeSetParams(hNode, nodeParams)
    }
}
pub unsafe fn cuGraphReleaseUserObject(graph: CUgraph, object: CUuserObject, count: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, CUuserObject, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphReleaseUserObject") });
        _f(graph, object, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphReleaseUserObject(graph: CUgraph, object: CUuserObject, count: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphReleaseUserObject(graph, object, count)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuGraphRemoveDependencies(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, numDependencies: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *const CUgraphNode, *const CUgraphNode, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphRemoveDependencies") });
        _f(hGraph, from, to, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphRemoveDependencies(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, numDependencies: usize) -> CUresult;
        }
        cuGraphRemoveDependencies(hGraph, from, to, numDependencies)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGraphRemoveDependencies_v2(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, edgeData: *const CUgraphEdgeData, numDependencies: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, *const CUgraphNode, *const CUgraphNode, *const CUgraphEdgeData, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphRemoveDependencies_v2") });
        _f(hGraph, from, to, edgeData, numDependencies)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphRemoveDependencies_v2(hGraph: CUgraph, from: *const CUgraphNode, to: *const CUgraphNode, edgeData: *const CUgraphEdgeData, numDependencies: usize) -> CUresult;
        }
        cuGraphRemoveDependencies_v2(hGraph, from, to, edgeData, numDependencies)
    }
}
pub unsafe fn cuGraphRetainUserObject(graph: CUgraph, object: CUuserObject, count: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraph, CUuserObject, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphRetainUserObject") });
        _f(graph, object, count, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphRetainUserObject(graph: CUgraph, object: CUuserObject, count: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphRetainUserObject(graph, object, count, flags)
    }
}
pub unsafe fn cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphUpload") });
        _f(hGraphExec, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;
        }
        cuGraphUpload(hGraphExec, hStream)
    }
}
pub unsafe fn cuGraphicsMapResources(count: ::core::ffi::c_uint, resources: *mut CUgraphicsResource, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint, *mut CUgraphicsResource, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsMapResources") });
        _f(count, resources, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsMapResources(count: ::core::ffi::c_uint, resources: *mut CUgraphicsResource, hStream: CUstream) -> CUresult;
        }
        cuGraphicsMapResources(count, resources, hStream)
    }
}
pub unsafe fn cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray: *mut CUmipmappedArray, resource: CUgraphicsResource) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmipmappedArray, CUgraphicsResource) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsResourceGetMappedMipmappedArray") });
        _f(pMipmappedArray, resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray: *mut CUmipmappedArray, resource: CUgraphicsResource) -> CUresult;
        }
        cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource)
    }
}
pub unsafe fn cuGraphicsResourceGetMappedPointer_v2(pDevPtr: *mut CUdeviceptr, pSize: *mut usize, resource: CUgraphicsResource) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUgraphicsResource) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsResourceGetMappedPointer_v2") });
        _f(pDevPtr, pSize, resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsResourceGetMappedPointer_v2(pDevPtr: *mut CUdeviceptr, pSize: *mut usize, resource: CUgraphicsResource) -> CUresult;
        }
        cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)
    }
}
pub unsafe fn cuGraphicsResourceSetMapFlags_v2(resource: CUgraphicsResource, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphicsResource, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsResourceSetMapFlags_v2") });
        _f(resource, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsResourceSetMapFlags_v2(resource: CUgraphicsResource, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphicsResourceSetMapFlags_v2(resource, flags)
    }
}
pub unsafe fn cuGraphicsSubResourceGetMappedArray(pArray: *mut CUarray, resource: CUgraphicsResource, arrayIndex: ::core::ffi::c_uint, mipLevel: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, CUgraphicsResource, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsSubResourceGetMappedArray") });
        _f(pArray, resource, arrayIndex, mipLevel)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsSubResourceGetMappedArray(pArray: *mut CUarray, resource: CUgraphicsResource, arrayIndex: ::core::ffi::c_uint, mipLevel: ::core::ffi::c_uint) -> CUresult;
        }
        cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)
    }
}
pub unsafe fn cuGraphicsUnmapResources(count: ::core::ffi::c_uint, resources: *mut CUgraphicsResource, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint, *mut CUgraphicsResource, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsUnmapResources") });
        _f(count, resources, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsUnmapResources(count: ::core::ffi::c_uint, resources: *mut CUgraphicsResource, hStream: CUstream) -> CUresult;
        }
        cuGraphicsUnmapResources(count, resources, hStream)
    }
}
pub unsafe fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgraphicsResource) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGraphicsUnregisterResource") });
        _f(resource)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;
        }
        cuGraphicsUnregisterResource(resource)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxCreate(phCtx: *mut CUgreenCtx, desc: CUdevResourceDesc, dev: CUdevice, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUgreenCtx, CUdevResourceDesc, CUdevice, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxCreate") });
        _f(phCtx, desc, dev, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxCreate(phCtx: *mut CUgreenCtx, desc: CUdevResourceDesc, dev: CUdevice, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuGreenCtxCreate(phCtx, desc, dev, flags)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxDestroy(hCtx: CUgreenCtx) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgreenCtx) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxDestroy") });
        _f(hCtx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxDestroy(hCtx: CUgreenCtx) -> CUresult;
        }
        cuGreenCtxDestroy(hCtx)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxGetDevResource(hCtx: CUgreenCtx, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgreenCtx, *mut CUdevResource, CUdevResourceType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxGetDevResource") });
        _f(hCtx, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxGetDevResource(hCtx: CUgreenCtx, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult;
        }
        cuGreenCtxGetDevResource(hCtx, resource, type_)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxGetId(greenCtx: CUgreenCtx, greenCtxId: *mut ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgreenCtx, *mut ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxGetId") });
        _f(greenCtx, greenCtxId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxGetId(greenCtx: CUgreenCtx, greenCtxId: *mut ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuGreenCtxGetId(greenCtx, greenCtxId)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxRecordEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgreenCtx, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxRecordEvent") });
        _f(hCtx, hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxRecordEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult;
        }
        cuGreenCtxRecordEvent(hCtx, hEvent)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxStreamCreate(phStream: *mut CUstream, greenCtx: CUgreenCtx, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUstream, CUgreenCtx, ::core::ffi::c_uint, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxStreamCreate") });
        _f(phStream, greenCtx, flags, priority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxStreamCreate(phStream: *mut CUstream, greenCtx: CUgreenCtx, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> CUresult;
        }
        cuGreenCtxStreamCreate(phStream, greenCtx, flags, priority)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuGreenCtxWaitEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUgreenCtx, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuGreenCtxWaitEvent") });
        _f(hCtx, hEvent)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuGreenCtxWaitEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult;
        }
        cuGreenCtxWaitEvent(hCtx, hEvent)
    }
}
pub unsafe fn cuImportExternalMemory(extMem_out: *mut CUexternalMemory, memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUexternalMemory, *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuImportExternalMemory") });
        _f(extMem_out, memHandleDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuImportExternalMemory(extMem_out: *mut CUexternalMemory, memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC) -> CUresult;
        }
        cuImportExternalMemory(extMem_out, memHandleDesc)
    }
}
pub unsafe fn cuImportExternalSemaphore(extSem_out: *mut CUexternalSemaphore, semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUexternalSemaphore, *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuImportExternalSemaphore") });
        _f(extSem_out, semHandleDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuImportExternalSemaphore(extSem_out: *mut CUexternalSemaphore, semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC) -> CUresult;
        }
        cuImportExternalSemaphore(extSem_out, semHandleDesc)
    }
}
pub unsafe fn cuInit(Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuInit") });
        _f(Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuInit(Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuInit(Flags)
    }
}
pub unsafe fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuIpcCloseMemHandle") });
        _f(dptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult;
        }
        cuIpcCloseMemHandle(dptr)
    }
}
pub unsafe fn cuIpcGetEventHandle(pHandle: *mut CUipcEventHandle, event: CUevent) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUipcEventHandle, CUevent) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuIpcGetEventHandle") });
        _f(pHandle, event)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuIpcGetEventHandle(pHandle: *mut CUipcEventHandle, event: CUevent) -> CUresult;
        }
        cuIpcGetEventHandle(pHandle, event)
    }
}
pub unsafe fn cuIpcGetMemHandle(pHandle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUipcMemHandle, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuIpcGetMemHandle") });
        _f(pHandle, dptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuIpcGetMemHandle(pHandle: *mut CUipcMemHandle, dptr: CUdeviceptr) -> CUresult;
        }
        cuIpcGetMemHandle(pHandle, dptr)
    }
}
pub unsafe fn cuIpcOpenEventHandle(phEvent: *mut CUevent, handle: CUipcEventHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUevent, CUipcEventHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuIpcOpenEventHandle") });
        _f(phEvent, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuIpcOpenEventHandle(phEvent: *mut CUevent, handle: CUipcEventHandle) -> CUresult;
        }
        cuIpcOpenEventHandle(phEvent, handle)
    }
}
pub unsafe fn cuIpcOpenMemHandle_v2(pdptr: *mut CUdeviceptr, handle: CUipcMemHandle, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, CUipcMemHandle, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuIpcOpenMemHandle_v2") });
        _f(pdptr, handle, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuIpcOpenMemHandle_v2(pdptr: *mut CUdeviceptr, handle: CUipcMemHandle, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuIpcOpenMemHandle_v2(pdptr, handle, Flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelGetAttribute(pi: *mut ::core::ffi::c_int, attrib: CUfunction_attribute, kernel: CUkernel, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUfunction_attribute, CUkernel, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelGetAttribute") });
        _f(pi, attrib, kernel, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelGetAttribute(pi: *mut ::core::ffi::c_int, attrib: CUfunction_attribute, kernel: CUkernel, dev: CUdevice) -> CUresult;
        }
        cuKernelGetAttribute(pi, attrib, kernel, dev)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelGetFunction(pFunc: *mut CUfunction, kernel: CUkernel) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfunction, CUkernel) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelGetFunction") });
        _f(pFunc, kernel)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelGetFunction(pFunc: *mut CUfunction, kernel: CUkernel) -> CUresult;
        }
        cuKernelGetFunction(pFunc, kernel)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelGetLibrary(pLib: *mut CUlibrary, kernel: CUkernel) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlibrary, CUkernel) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelGetLibrary") });
        _f(pLib, kernel)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelGetLibrary(pLib: *mut CUlibrary, kernel: CUkernel) -> CUresult;
        }
        cuKernelGetLibrary(pLib, kernel)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelGetName(name: *mut *const ::core::ffi::c_char, hfunc: CUkernel) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *const ::core::ffi::c_char, CUkernel) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelGetName") });
        _f(name, hfunc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelGetName(name: *mut *const ::core::ffi::c_char, hfunc: CUkernel) -> CUresult;
        }
        cuKernelGetName(name, hfunc)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelGetParamCount(kernel: CUkernel, paramCount: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUkernel, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelGetParamCount") });
        _f(kernel, paramCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelGetParamCount(kernel: CUkernel, paramCount: *mut usize) -> CUresult;
        }
        cuKernelGetParamCount(kernel, paramCount)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelGetParamInfo(kernel: CUkernel, paramIndex: usize, paramOffset: *mut usize, paramSize: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUkernel, usize, *mut usize, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelGetParamInfo") });
        _f(kernel, paramIndex, paramOffset, paramSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelGetParamInfo(kernel: CUkernel, paramIndex: usize, paramOffset: *mut usize, paramSize: *mut usize) -> CUresult;
        }
        cuKernelGetParamInfo(kernel, paramIndex, paramOffset, paramSize)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelSetAttribute(attrib: CUfunction_attribute, val: ::core::ffi::c_int, kernel: CUkernel, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction_attribute, ::core::ffi::c_int, CUkernel, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelSetAttribute") });
        _f(attrib, val, kernel, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelSetAttribute(attrib: CUfunction_attribute, val: ::core::ffi::c_int, kernel: CUkernel, dev: CUdevice) -> CUresult;
        }
        cuKernelSetAttribute(attrib, val, kernel, dev)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuKernelSetCacheConfig(kernel: CUkernel, config: CUfunc_cache, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUkernel, CUfunc_cache, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuKernelSetCacheConfig") });
        _f(kernel, config, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuKernelSetCacheConfig(kernel: CUkernel, config: CUfunc_cache, dev: CUdevice) -> CUresult;
        }
        cuKernelSetCacheConfig(kernel, config, dev)
    }
}
pub unsafe fn cuLaunch(f: CUfunction) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunch") });
        _f(f)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunch(f: CUfunction) -> CUresult;
        }
        cuLaunch(f)
    }
}
pub unsafe fn cuLaunchCooperativeKernel(f: CUfunction, gridDimX: ::core::ffi::c_uint, gridDimY: ::core::ffi::c_uint, gridDimZ: ::core::ffi::c_uint, blockDimX: ::core::ffi::c_uint, blockDimY: ::core::ffi::c_uint, blockDimZ: ::core::ffi::c_uint, sharedMemBytes: ::core::ffi::c_uint, hStream: CUstream, kernelParams: *mut *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, CUstream, *mut *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchCooperativeKernel") });
        _f(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchCooperativeKernel(f: CUfunction, gridDimX: ::core::ffi::c_uint, gridDimY: ::core::ffi::c_uint, gridDimZ: ::core::ffi::c_uint, blockDimX: ::core::ffi::c_uint, blockDimY: ::core::ffi::c_uint, blockDimZ: ::core::ffi::c_uint, sharedMemBytes: ::core::ffi::c_uint, hStream: CUstream, kernelParams: *mut *mut ::core::ffi::c_void) -> CUresult;
        }
        cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams)
    }
}
pub unsafe fn cuLaunchCooperativeKernelMultiDevice(launchParamsList: *mut CUDA_LAUNCH_PARAMS, numDevices: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_LAUNCH_PARAMS, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchCooperativeKernelMultiDevice") });
        _f(launchParamsList, numDevices, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchCooperativeKernelMultiDevice(launchParamsList: *mut CUDA_LAUNCH_PARAMS, numDevices: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
    }
}
pub unsafe fn cuLaunchGrid(f: CUfunction, grid_width: ::core::ffi::c_int, grid_height: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchGrid") });
        _f(f, grid_width, grid_height)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchGrid(f: CUfunction, grid_width: ::core::ffi::c_int, grid_height: ::core::ffi::c_int) -> CUresult;
        }
        cuLaunchGrid(f, grid_width, grid_height)
    }
}
pub unsafe fn cuLaunchGridAsync(f: CUfunction, grid_width: ::core::ffi::c_int, grid_height: ::core::ffi::c_int, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, ::core::ffi::c_int, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchGridAsync") });
        _f(f, grid_width, grid_height, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchGridAsync(f: CUfunction, grid_width: ::core::ffi::c_int, grid_height: ::core::ffi::c_int, hStream: CUstream) -> CUresult;
        }
        cuLaunchGridAsync(f, grid_width, grid_height, hStream)
    }
}
pub unsafe fn cuLaunchHostFunc(hStream: CUstream, fn_: CUhostFn, userData: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUhostFn, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchHostFunc") });
        _f(hStream, fn_, userData)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchHostFunc(hStream: CUstream, fn_: CUhostFn, userData: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuLaunchHostFunc(hStream, fn_, userData)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLaunchHostFunc_v2(hStream: CUstream, fn_: CUhostFn, userData: *mut ::core::ffi::c_void, syncMode: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUhostFn, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchHostFunc_v2") });
        _f(hStream, fn_, userData, syncMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchHostFunc_v2(hStream: CUstream, fn_: CUhostFn, userData: *mut ::core::ffi::c_void, syncMode: ::core::ffi::c_uint) -> CUresult;
        }
        cuLaunchHostFunc_v2(hStream, fn_, userData, syncMode)
    }
}
pub unsafe fn cuLaunchKernel(f: CUfunction, gridDimX: ::core::ffi::c_uint, gridDimY: ::core::ffi::c_uint, gridDimZ: ::core::ffi::c_uint, blockDimX: ::core::ffi::c_uint, blockDimY: ::core::ffi::c_uint, blockDimZ: ::core::ffi::c_uint, sharedMemBytes: ::core::ffi::c_uint, hStream: CUstream, kernelParams: *mut *mut ::core::ffi::c_void, extra: *mut *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, ::core::ffi::c_uint, CUstream, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchKernel") });
        _f(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchKernel(f: CUfunction, gridDimX: ::core::ffi::c_uint, gridDimY: ::core::ffi::c_uint, gridDimZ: ::core::ffi::c_uint, blockDimX: ::core::ffi::c_uint, blockDimY: ::core::ffi::c_uint, blockDimZ: ::core::ffi::c_uint, sharedMemBytes: ::core::ffi::c_uint, hStream: CUstream, kernelParams: *mut *mut ::core::ffi::c_void, extra: *mut *mut ::core::ffi::c_void) -> CUresult;
        }
        cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLaunchKernelEx(config: *const CUlaunchConfig, f: CUfunction, kernelParams: *mut *mut ::core::ffi::c_void, extra: *mut *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUlaunchConfig, CUfunction, *mut *mut ::core::ffi::c_void, *mut *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLaunchKernelEx") });
        _f(config, f, kernelParams, extra)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLaunchKernelEx(config: *const CUlaunchConfig, f: CUfunction, kernelParams: *mut *mut ::core::ffi::c_void, extra: *mut *mut ::core::ffi::c_void) -> CUresult;
        }
        cuLaunchKernelEx(config, f, kernelParams, extra)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryEnumerateKernels(kernels: *mut CUkernel, numKernels: ::core::ffi::c_uint, lib: CUlibrary) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUkernel, ::core::ffi::c_uint, CUlibrary) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryEnumerateKernels") });
        _f(kernels, numKernels, lib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryEnumerateKernels(kernels: *mut CUkernel, numKernels: ::core::ffi::c_uint, lib: CUlibrary) -> CUresult;
        }
        cuLibraryEnumerateKernels(kernels, numKernels, lib)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryGetGlobal(dptr: *mut CUdeviceptr, bytes: *mut usize, library: CUlibrary, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUlibrary, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryGetGlobal") });
        _f(dptr, bytes, library, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryGetGlobal(dptr: *mut CUdeviceptr, bytes: *mut usize, library: CUlibrary, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuLibraryGetGlobal(dptr, bytes, library, name)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryGetKernel(pKernel: *mut CUkernel, library: CUlibrary, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUkernel, CUlibrary, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryGetKernel") });
        _f(pKernel, library, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryGetKernel(pKernel: *mut CUkernel, library: CUlibrary, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuLibraryGetKernel(pKernel, library, name)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryGetKernelCount(count: *mut ::core::ffi::c_uint, lib: CUlibrary) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, CUlibrary) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryGetKernelCount") });
        _f(count, lib)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryGetKernelCount(count: *mut ::core::ffi::c_uint, lib: CUlibrary) -> CUresult;
        }
        cuLibraryGetKernelCount(count, lib)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryGetManaged(dptr: *mut CUdeviceptr, bytes: *mut usize, library: CUlibrary, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUlibrary, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryGetManaged") });
        _f(dptr, bytes, library, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryGetManaged(dptr: *mut CUdeviceptr, bytes: *mut usize, library: CUlibrary, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuLibraryGetManaged(dptr, bytes, library, name)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryGetModule(pMod: *mut CUmodule, library: CUlibrary) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmodule, CUlibrary) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryGetModule") });
        _f(pMod, library)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryGetModule(pMod: *mut CUmodule, library: CUlibrary) -> CUresult;
        }
        cuLibraryGetModule(pMod, library)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryGetUnifiedFunction(fptr: *mut *mut ::core::ffi::c_void, library: CUlibrary, symbol: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, CUlibrary, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryGetUnifiedFunction") });
        _f(fptr, library, symbol)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryGetUnifiedFunction(fptr: *mut *mut ::core::ffi::c_void, library: CUlibrary, symbol: *const ::core::ffi::c_char) -> CUresult;
        }
        cuLibraryGetUnifiedFunction(fptr, library, symbol)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryLoadData(library: *mut CUlibrary, code: *const ::core::ffi::c_void, jitOptions: *mut CUjit_option, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut CUlibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlibrary, *const ::core::ffi::c_void, *mut CUjit_option, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint, *mut CUlibraryOption, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryLoadData") });
        _f(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryLoadData(library: *mut CUlibrary, code: *const ::core::ffi::c_void, jitOptions: *mut CUjit_option, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut CUlibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> CUresult;
        }
        cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryLoadFromFile(library: *mut CUlibrary, fileName: *const ::core::ffi::c_char, jitOptions: *mut CUjit_option, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut CUlibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlibrary, *const ::core::ffi::c_char, *mut CUjit_option, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint, *mut CUlibraryOption, *mut *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryLoadFromFile") });
        _f(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryLoadFromFile(library: *mut CUlibrary, fileName: *const ::core::ffi::c_char, jitOptions: *mut CUjit_option, jitOptionsValues: *mut *mut ::core::ffi::c_void, numJitOptions: ::core::ffi::c_uint, libraryOptions: *mut CUlibraryOption, libraryOptionValues: *mut *mut ::core::ffi::c_void, numLibraryOptions: ::core::ffi::c_uint) -> CUresult;
        }
        cuLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLibraryUnload(library: CUlibrary) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlibrary) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLibraryUnload") });
        _f(library)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLibraryUnload(library: CUlibrary) -> CUresult;
        }
        cuLibraryUnload(library)
    }
}
pub unsafe fn cuLinkAddData_v2(state: CUlinkState, type_: CUjitInputType, data: *mut ::core::ffi::c_void, size: usize, name: *const ::core::ffi::c_char, numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlinkState, CUjitInputType, *mut ::core::ffi::c_void, usize, *const ::core::ffi::c_char, ::core::ffi::c_uint, *mut CUjit_option, *mut *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLinkAddData_v2") });
        _f(state, type_, data, size, name, numOptions, options, optionValues)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLinkAddData_v2(state: CUlinkState, type_: CUjitInputType, data: *mut ::core::ffi::c_void, size: usize, name: *const ::core::ffi::c_char, numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void) -> CUresult;
        }
        cuLinkAddData_v2(state, type_, data, size, name, numOptions, options, optionValues)
    }
}
pub unsafe fn cuLinkAddFile_v2(state: CUlinkState, type_: CUjitInputType, path: *const ::core::ffi::c_char, numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlinkState, CUjitInputType, *const ::core::ffi::c_char, ::core::ffi::c_uint, *mut CUjit_option, *mut *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLinkAddFile_v2") });
        _f(state, type_, path, numOptions, options, optionValues)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLinkAddFile_v2(state: CUlinkState, type_: CUjitInputType, path: *const ::core::ffi::c_char, numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void) -> CUresult;
        }
        cuLinkAddFile_v2(state, type_, path, numOptions, options, optionValues)
    }
}
pub unsafe fn cuLinkComplete(state: CUlinkState, cubinOut: *mut *mut ::core::ffi::c_void, sizeOut: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlinkState, *mut *mut ::core::ffi::c_void, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLinkComplete") });
        _f(state, cubinOut, sizeOut)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLinkComplete(state: CUlinkState, cubinOut: *mut *mut ::core::ffi::c_void, sizeOut: *mut usize) -> CUresult;
        }
        cuLinkComplete(state, cubinOut, sizeOut)
    }
}
pub unsafe fn cuLinkCreate_v2(numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void, stateOut: *mut CUlinkState) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint, *mut CUjit_option, *mut *mut ::core::ffi::c_void, *mut CUlinkState) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLinkCreate_v2") });
        _f(numOptions, options, optionValues, stateOut)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLinkCreate_v2(numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void, stateOut: *mut CUlinkState) -> CUresult;
        }
        cuLinkCreate_v2(numOptions, options, optionValues, stateOut)
    }
}
pub unsafe fn cuLinkDestroy(state: CUlinkState) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlinkState) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLinkDestroy") });
        _f(state)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLinkDestroy(state: CUlinkState) -> CUresult;
        }
        cuLinkDestroy(state)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointAddDevice(leId: CUlogicalEndpointId, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointAddDevice") });
        _f(leId, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointAddDevice(leId: CUlogicalEndpointId, dev: CUdevice) -> CUresult;
        }
        cuLogicalEndpointAddDevice(leId, dev)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointBindAddr(leId: CUlogicalEndpointId, dev: CUdevice, offset: cuuint64_t, ptr: *mut ::core::ffi::c_void, size: cuuint64_t, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, CUdevice, cuuint64_t, *mut ::core::ffi::c_void, cuuint64_t, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointBindAddr") });
        _f(leId, dev, offset, ptr, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointBindAddr(leId: CUlogicalEndpointId, dev: CUdevice, offset: cuuint64_t, ptr: *mut ::core::ffi::c_void, size: cuuint64_t, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuLogicalEndpointBindAddr(leId, dev, offset, ptr, size, flags)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointBindMem(leId: CUlogicalEndpointId, dev: CUdevice, offset: cuuint64_t, memHandle: CUmemGenericAllocationHandle, memOffset: cuuint64_t, size: cuuint64_t, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, CUdevice, cuuint64_t, CUmemGenericAllocationHandle, cuuint64_t, cuuint64_t, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointBindMem") });
        _f(leId, dev, offset, memHandle, memOffset, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointBindMem(leId: CUlogicalEndpointId, dev: CUdevice, offset: cuuint64_t, memHandle: CUmemGenericAllocationHandle, memOffset: cuuint64_t, size: cuuint64_t, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuLogicalEndpointBindMem(leId, dev, offset, memHandle, memOffset, size, flags)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointCreate(leId: CUlogicalEndpointId, prop: *const CUlogicalEndpointProp) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, *const CUlogicalEndpointProp) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointCreate") });
        _f(leId, prop)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointCreate(leId: CUlogicalEndpointId, prop: *const CUlogicalEndpointProp) -> CUresult;
        }
        cuLogicalEndpointCreate(leId, prop)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointDestroy(leId: CUlogicalEndpointId) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointDestroy") });
        _f(leId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointDestroy(leId: CUlogicalEndpointId) -> CUresult;
        }
        cuLogicalEndpointDestroy(leId)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointExport(handle: *mut ::core::ffi::c_void, leId: CUlogicalEndpointId, handleType: CUlogicalEndpointIpcHandleType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUlogicalEndpointId, CUlogicalEndpointIpcHandleType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointExport") });
        _f(handle, leId, handleType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointExport(handle: *mut ::core::ffi::c_void, leId: CUlogicalEndpointId, handleType: CUlogicalEndpointIpcHandleType) -> CUresult;
        }
        cuLogicalEndpointExport(handle, leId, handleType)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointGetLimits(bindAlignment: *mut cuuint64_t, maxSize: *mut cuuint64_t, prop: *const CUlogicalEndpointProp) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut cuuint64_t, *mut cuuint64_t, *const CUlogicalEndpointProp) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointGetLimits") });
        _f(bindAlignment, maxSize, prop)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointGetLimits(bindAlignment: *mut cuuint64_t, maxSize: *mut cuuint64_t, prop: *const CUlogicalEndpointProp) -> CUresult;
        }
        cuLogicalEndpointGetLimits(bindAlignment, maxSize, prop)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointIdRelease(baseLeId: CUlogicalEndpointId, count: cuuint32_t) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, cuuint32_t) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointIdRelease") });
        _f(baseLeId, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointIdRelease(baseLeId: CUlogicalEndpointId, count: cuuint32_t) -> CUresult;
        }
        cuLogicalEndpointIdRelease(baseLeId, count)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointIdReserve(baseLeId: *mut CUlogicalEndpointId, count: cuuint32_t) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlogicalEndpointId, cuuint32_t) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointIdReserve") });
        _f(baseLeId, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointIdReserve(baseLeId: *mut CUlogicalEndpointId, count: cuuint32_t) -> CUresult;
        }
        cuLogicalEndpointIdReserve(baseLeId, count)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointImport(leId: CUlogicalEndpointId, handle: *const ::core::ffi::c_void, handleType: CUlogicalEndpointIpcHandleType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, *const ::core::ffi::c_void, CUlogicalEndpointIpcHandleType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointImport") });
        _f(leId, handle, handleType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointImport(leId: CUlogicalEndpointId, handle: *const ::core::ffi::c_void, handleType: CUlogicalEndpointIpcHandleType) -> CUresult;
        }
        cuLogicalEndpointImport(leId, handle, handleType)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointQuery(leId: CUlogicalEndpointId, count: cuuint32_t, queryStatus: *mut ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, cuuint32_t, *mut ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointQuery") });
        _f(leId, count, queryStatus)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointQuery(leId: CUlogicalEndpointId, count: cuuint32_t, queryStatus: *mut ::core::ffi::c_int) -> CUresult;
        }
        cuLogicalEndpointQuery(leId, count, queryStatus)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuLogicalEndpointUnbind(leId: CUlogicalEndpointId, dev: CUdevice, offset: cuuint64_t, size: cuuint64_t) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogicalEndpointId, CUdevice, cuuint64_t, cuuint64_t) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogicalEndpointUnbind") });
        _f(leId, dev, offset, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogicalEndpointUnbind(leId: CUlogicalEndpointId, dev: CUdevice, offset: cuuint64_t, size: cuuint64_t) -> CUresult;
        }
        cuLogicalEndpointUnbind(leId, dev, offset, size)
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLogsCurrent(iterator_out: *mut CUlogIterator, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlogIterator, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogsCurrent") });
        _f(iterator_out, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogsCurrent(iterator_out: *mut CUlogIterator, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuLogsCurrent(iterator_out, flags)
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLogsDumpToFile(iterator: *mut CUlogIterator, pathToFile: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlogIterator, *const ::core::ffi::c_char, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogsDumpToFile") });
        _f(iterator, pathToFile, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogsDumpToFile(iterator: *mut CUlogIterator, pathToFile: *const ::core::ffi::c_char, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuLogsDumpToFile(iterator, pathToFile, flags)
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLogsDumpToMemory(iterator: *mut CUlogIterator, buffer: *mut ::core::ffi::c_char, size: *mut usize, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUlogIterator, *mut ::core::ffi::c_char, *mut usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogsDumpToMemory") });
        _f(iterator, buffer, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogsDumpToMemory(iterator: *mut CUlogIterator, buffer: *mut ::core::ffi::c_char, size: *mut usize, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuLogsDumpToMemory(iterator, buffer, size, flags)
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLogsRegisterCallback(callbackFunc: CUlogsCallback, userData: *mut ::core::ffi::c_void, callback_out: *mut CUlogsCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogsCallback, *mut ::core::ffi::c_void, *mut CUlogsCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogsRegisterCallback") });
        _f(callbackFunc, userData, callback_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogsRegisterCallback(callbackFunc: CUlogsCallback, userData: *mut ::core::ffi::c_void, callback_out: *mut CUlogsCallbackHandle) -> CUresult;
        }
        cuLogsRegisterCallback(callbackFunc, userData, callback_out)
    }
}
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuLogsUnregisterCallback(callback: CUlogsCallbackHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUlogsCallbackHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuLogsUnregisterCallback") });
        _f(callback)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuLogsUnregisterCallback(callback: CUlogsCallbackHandle) -> CUresult;
        }
        cuLogsUnregisterCallback(callback)
    }
}
pub unsafe fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAddressFree") });
        _f(ptr, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult;
        }
        cuMemAddressFree(ptr, size)
    }
}
pub unsafe fn cuMemAddressReserve(ptr: *mut CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAddressReserve") });
        _f(ptr, size, alignment, addr, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAddressReserve(ptr: *mut CUdeviceptr, size: usize, alignment: usize, addr: CUdeviceptr, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemAddressReserve(ptr, size, alignment, addr, flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuMemAdvise(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, device: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, CUmem_advise, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAdvise") });
        _f(devPtr, count, advice, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAdvise(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, device: CUdevice) -> CUresult;
        }
        cuMemAdvise(devPtr, count, advice, device)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemAdvise_v2(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, location: CUmemLocation) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, CUmem_advise, CUmemLocation) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAdvise_v2") });
        _f(devPtr, count, advice, location)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAdvise_v2(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, location: CUmemLocation) -> CUresult;
        }
        cuMemAdvise_v2(devPtr, count, advice, location)
    }
}
pub unsafe fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAllocAsync") });
        _f(dptr, bytesize, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) -> CUresult;
        }
        cuMemAllocAsync(dptr, bytesize, hStream)
    }
}
pub unsafe fn cuMemAllocFromPoolAsync(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, usize, CUmemoryPool, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAllocFromPoolAsync") });
        _f(dptr, bytesize, pool, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAllocFromPoolAsync(dptr: *mut CUdeviceptr, bytesize: usize, pool: CUmemoryPool, hStream: CUstream) -> CUresult;
        }
        cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream)
    }
}
pub unsafe fn cuMemAllocHost_v2(pp: *mut *mut ::core::ffi::c_void, bytesize: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAllocHost_v2") });
        _f(pp, bytesize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAllocHost_v2(pp: *mut *mut ::core::ffi::c_void, bytesize: usize) -> CUresult;
        }
        cuMemAllocHost_v2(pp, bytesize)
    }
}
pub unsafe fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAllocManaged") });
        _f(dptr, bytesize, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuMemAllocManaged(dptr, bytesize, flags)
    }
}
pub unsafe fn cuMemAllocPitch_v2(dptr: *mut CUdeviceptr, pPitch: *mut usize, WidthInBytes: usize, Height: usize, ElementSizeBytes: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, usize, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAllocPitch_v2") });
        _f(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAllocPitch_v2(dptr: *mut CUdeviceptr, pPitch: *mut usize, WidthInBytes: usize, Height: usize, ElementSizeBytes: ::core::ffi::c_uint) -> CUresult;
        }
        cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    }
}
pub unsafe fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemAlloc_v2") });
        _f(dptr, bytesize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
        }
        cuMemAlloc_v2(dptr, bytesize)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemBatchDecompressAsync(paramsArray: *mut CUmemDecompressParams, count: usize, flags: ::core::ffi::c_uint, errorIndex: *mut usize, stream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemDecompressParams, usize, ::core::ffi::c_uint, *mut usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemBatchDecompressAsync") });
        _f(paramsArray, count, flags, errorIndex, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemBatchDecompressAsync(paramsArray: *mut CUmemDecompressParams, count: usize, flags: ::core::ffi::c_uint, errorIndex: *mut usize, stream: CUstream) -> CUresult;
        }
        cuMemBatchDecompressAsync(paramsArray, count, flags, errorIndex, stream)
    }
}
pub unsafe fn cuMemCreate(handle: *mut CUmemGenericAllocationHandle, size: usize, prop: *const CUmemAllocationProp, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemGenericAllocationHandle, usize, *const CUmemAllocationProp, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemCreate") });
        _f(handle, size, prop, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemCreate(handle: *mut CUmemGenericAllocationHandle, size: usize, prop: *const CUmemAllocationProp, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemCreate(handle, size, prop, flags)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemDiscardAndPrefetchBatchAsync(dptrs: *mut CUdeviceptr, sizes: *mut usize, count: usize, prefetchLocs: *mut CUmemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, usize, *mut CUmemLocation, *mut usize, usize, ::core::ffi::c_ulonglong, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemDiscardAndPrefetchBatchAsync") });
        _f(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemDiscardAndPrefetchBatchAsync(dptrs: *mut CUdeviceptr, sizes: *mut usize, count: usize, prefetchLocs: *mut CUmemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult;
        }
        cuMemDiscardAndPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemDiscardBatchAsync(dptrs: *mut CUdeviceptr, sizes: *mut usize, count: usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, usize, ::core::ffi::c_ulonglong, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemDiscardBatchAsync") });
        _f(dptrs, sizes, count, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemDiscardBatchAsync(dptrs: *mut CUdeviceptr, sizes: *mut usize, count: usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult;
        }
        cuMemDiscardBatchAsync(dptrs, sizes, count, flags, hStream)
    }
}
pub unsafe fn cuMemExportToShareableHandle(shareableHandle: *mut ::core::ffi::c_void, handle: CUmemGenericAllocationHandle, handleType: CUmemAllocationHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUmemGenericAllocationHandle, CUmemAllocationHandleType, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemExportToShareableHandle") });
        _f(shareableHandle, handle, handleType, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemExportToShareableHandle(shareableHandle: *mut ::core::ffi::c_void, handle: CUmemGenericAllocationHandle, handleType: CUmemAllocationHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags)
    }
}
pub unsafe fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemFreeAsync") });
        _f(dptr, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult;
        }
        cuMemFreeAsync(dptr, hStream)
    }
}
pub unsafe fn cuMemFreeHost(p: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemFreeHost") });
        _f(p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemFreeHost(p: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuMemFreeHost(p)
    }
}
pub unsafe fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemFree_v2") });
        _f(dptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
        }
        cuMemFree_v2(dptr)
    }
}
pub unsafe fn cuMemGetAccess(flags: *mut ::core::ffi::c_ulonglong, location: *const CUmemLocation, ptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_ulonglong, *const CUmemLocation, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetAccess") });
        _f(flags, location, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetAccess(flags: *mut ::core::ffi::c_ulonglong, location: *const CUmemLocation, ptr: CUdeviceptr) -> CUresult;
        }
        cuMemGetAccess(flags, location, ptr)
    }
}
pub unsafe fn cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetAddressRange_v2") });
        _f(pbase, psize, dptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetAddressRange_v2(pbase: *mut CUdeviceptr, psize: *mut usize, dptr: CUdeviceptr) -> CUresult;
        }
        cuMemGetAddressRange_v2(pbase, psize, dptr)
    }
}
pub unsafe fn cuMemGetAllocationGranularity(granularity: *mut usize, prop: *const CUmemAllocationProp, option: CUmemAllocationGranularity_flags) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, CUmemAllocationGranularity_flags) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetAllocationGranularity") });
        _f(granularity, prop, option)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetAllocationGranularity(granularity: *mut usize, prop: *const CUmemAllocationProp, option: CUmemAllocationGranularity_flags) -> CUresult;
        }
        cuMemGetAllocationGranularity(granularity, prop, option)
    }
}
pub unsafe fn cuMemGetAllocationPropertiesFromHandle(prop: *mut CUmemAllocationProp, handle: CUmemGenericAllocationHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemAllocationProp, CUmemGenericAllocationHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetAllocationPropertiesFromHandle") });
        _f(prop, handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetAllocationPropertiesFromHandle(prop: *mut CUmemAllocationProp, handle: CUmemGenericAllocationHandle) -> CUresult;
        }
        cuMemGetAllocationPropertiesFromHandle(prop, handle)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemGetDefaultMemPool(pool_out: *mut CUmemoryPool, location: *mut CUmemLocation, type_: CUmemAllocationType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemoryPool, *mut CUmemLocation, CUmemAllocationType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetDefaultMemPool") });
        _f(pool_out, location, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetDefaultMemPool(pool_out: *mut CUmemoryPool, location: *mut CUmemLocation, type_: CUmemAllocationType) -> CUresult;
        }
        cuMemGetDefaultMemPool(pool_out, location, type_)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemGetHandleForAddressRange(handle: *mut ::core::ffi::c_void, dptr: CUdeviceptr, size: usize, handleType: CUmemRangeHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUdeviceptr, usize, CUmemRangeHandleType, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetHandleForAddressRange") });
        _f(handle, dptr, size, handleType, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetHandleForAddressRange(handle: *mut ::core::ffi::c_void, dptr: CUdeviceptr, size: usize, handleType: CUmemRangeHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemGetHandleForAddressRange(handle, dptr, size, handleType, flags)
    }
}
pub unsafe fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetInfo_v2") });
        _f(free, total)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
        }
        cuMemGetInfo_v2(free, total)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemGetMemPool(pool: *mut CUmemoryPool, location: *mut CUmemLocation, type_: CUmemAllocationType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemoryPool, *mut CUmemLocation, CUmemAllocationType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemGetMemPool") });
        _f(pool, location, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemGetMemPool(pool: *mut CUmemoryPool, location: *mut CUmemLocation, type_: CUmemAllocationType) -> CUresult;
        }
        cuMemGetMemPool(pool, location, type_)
    }
}
pub unsafe fn cuMemHostAlloc(pp: *mut *mut ::core::ffi::c_void, bytesize: usize, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemHostAlloc") });
        _f(pp, bytesize, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemHostAlloc(pp: *mut *mut ::core::ffi::c_void, bytesize: usize, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuMemHostAlloc(pp, bytesize, Flags)
    }
}
pub unsafe fn cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut ::core::ffi::c_void, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemHostGetDevicePointer_v2") });
        _f(pdptr, p, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut ::core::ffi::c_void, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuMemHostGetDevicePointer_v2(pdptr, p, Flags)
    }
}
pub unsafe fn cuMemHostGetFlags(pFlags: *mut ::core::ffi::c_uint, p: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemHostGetFlags") });
        _f(pFlags, p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemHostGetFlags(pFlags: *mut ::core::ffi::c_uint, p: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuMemHostGetFlags(pFlags, p)
    }
}
pub unsafe fn cuMemHostRegister_v2(p: *mut ::core::ffi::c_void, bytesize: usize, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemHostRegister_v2") });
        _f(p, bytesize, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemHostRegister_v2(p: *mut ::core::ffi::c_void, bytesize: usize, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuMemHostRegister_v2(p, bytesize, Flags)
    }
}
pub unsafe fn cuMemHostUnregister(p: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemHostUnregister") });
        _f(p)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemHostUnregister(p: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuMemHostUnregister(p)
    }
}
pub unsafe fn cuMemImportFromShareableHandle(handle: *mut CUmemGenericAllocationHandle, osHandle: *mut ::core::ffi::c_void, shHandleType: CUmemAllocationHandleType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemGenericAllocationHandle, *mut ::core::ffi::c_void, CUmemAllocationHandleType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemImportFromShareableHandle") });
        _f(handle, osHandle, shHandleType)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemImportFromShareableHandle(handle: *mut CUmemGenericAllocationHandle, osHandle: *mut ::core::ffi::c_void, shHandleType: CUmemAllocationHandleType) -> CUresult;
        }
        cuMemImportFromShareableHandle(handle, osHandle, shHandleType)
    }
}
pub unsafe fn cuMemMap(ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, usize, CUmemGenericAllocationHandle, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemMap") });
        _f(ptr, size, offset, handle, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemMap(ptr: CUdeviceptr, size: usize, offset: usize, handle: CUmemGenericAllocationHandle, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemMap(ptr, size, offset, handle, flags)
    }
}
pub unsafe fn cuMemMapArrayAsync(mapInfoList: *mut CUarrayMapInfo, count: ::core::ffi::c_uint, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarrayMapInfo, ::core::ffi::c_uint, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemMapArrayAsync") });
        _f(mapInfoList, count, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemMapArrayAsync(mapInfoList: *mut CUarrayMapInfo, count: ::core::ffi::c_uint, hStream: CUstream) -> CUresult;
        }
        cuMemMapArrayAsync(mapInfoList, count, hStream)
    }
}
pub unsafe fn cuMemPoolCreate(pool: *mut CUmemoryPool, poolProps: *const CUmemPoolProps) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemoryPool, *const CUmemPoolProps) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolCreate") });
        _f(pool, poolProps)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolCreate(pool: *mut CUmemoryPool, poolProps: *const CUmemPoolProps) -> CUresult;
        }
        cuMemPoolCreate(pool, poolProps)
    }
}
pub unsafe fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemoryPool) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolDestroy") });
        _f(pool)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult;
        }
        cuMemPoolDestroy(pool)
    }
}
pub unsafe fn cuMemPoolExportPointer(shareData_out: *mut CUmemPoolPtrExportData, ptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemPoolPtrExportData, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolExportPointer") });
        _f(shareData_out, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolExportPointer(shareData_out: *mut CUmemPoolPtrExportData, ptr: CUdeviceptr) -> CUresult;
        }
        cuMemPoolExportPointer(shareData_out, ptr)
    }
}
pub unsafe fn cuMemPoolExportToShareableHandle(handle_out: *mut ::core::ffi::c_void, pool: CUmemoryPool, handleType: CUmemAllocationHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUmemoryPool, CUmemAllocationHandleType, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolExportToShareableHandle") });
        _f(handle_out, pool, handleType, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolExportToShareableHandle(handle_out: *mut ::core::ffi::c_void, pool: CUmemoryPool, handleType: CUmemAllocationHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemPoolExportToShareableHandle(handle_out, pool, handleType, flags)
    }
}
pub unsafe fn cuMemPoolGetAccess(flags: *mut CUmemAccess_flags, memPool: CUmemoryPool, location: *mut CUmemLocation) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemAccess_flags, CUmemoryPool, *mut CUmemLocation) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolGetAccess") });
        _f(flags, memPool, location)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolGetAccess(flags: *mut CUmemAccess_flags, memPool: CUmemoryPool, location: *mut CUmemLocation) -> CUresult;
        }
        cuMemPoolGetAccess(flags, memPool, location)
    }
}
pub unsafe fn cuMemPoolGetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemoryPool, CUmemPool_attribute, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolGetAttribute") });
        _f(pool, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolGetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuMemPoolGetAttribute(pool, attr, value)
    }
}
pub unsafe fn cuMemPoolImportFromShareableHandle(pool_out: *mut CUmemoryPool, handle: *mut ::core::ffi::c_void, handleType: CUmemAllocationHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemoryPool, *mut ::core::ffi::c_void, CUmemAllocationHandleType, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolImportFromShareableHandle") });
        _f(pool_out, handle, handleType, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolImportFromShareableHandle(pool_out: *mut CUmemoryPool, handle: *mut ::core::ffi::c_void, handleType: CUmemAllocationHandleType, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMemPoolImportFromShareableHandle(pool_out, handle, handleType, flags)
    }
}
pub unsafe fn cuMemPoolImportPointer(ptr_out: *mut CUdeviceptr, pool: CUmemoryPool, shareData: *mut CUmemPoolPtrExportData) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, CUmemoryPool, *mut CUmemPoolPtrExportData) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolImportPointer") });
        _f(ptr_out, pool, shareData)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolImportPointer(ptr_out: *mut CUdeviceptr, pool: CUmemoryPool, shareData: *mut CUmemPoolPtrExportData) -> CUresult;
        }
        cuMemPoolImportPointer(ptr_out, pool, shareData)
    }
}
pub unsafe fn cuMemPoolSetAccess(pool: CUmemoryPool, map: *const CUmemAccessDesc, count: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemoryPool, *const CUmemAccessDesc, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolSetAccess") });
        _f(pool, map, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolSetAccess(pool: CUmemoryPool, map: *const CUmemAccessDesc, count: usize) -> CUresult;
        }
        cuMemPoolSetAccess(pool, map, count)
    }
}
pub unsafe fn cuMemPoolSetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemoryPool, CUmemPool_attribute, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolSetAttribute") });
        _f(pool, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolSetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuMemPoolSetAttribute(pool, attr, value)
    }
}
pub unsafe fn cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemoryPool, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPoolTrimTo") });
        _f(pool, minBytesToKeep)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) -> CUresult;
        }
        cuMemPoolTrimTo(pool, minBytesToKeep)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuMemPrefetchAsync(devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, CUdevice, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPrefetchAsync") });
        _f(devPtr, count, dstDevice, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPrefetchAsync(devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: CUstream) -> CUresult;
        }
        cuMemPrefetchAsync(devPtr, count, dstDevice, hStream)
    }
}
#[cfg(any(feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemPrefetchAsync_v2(devPtr: CUdeviceptr, count: usize, location: CUmemLocation, flags: ::core::ffi::c_uint, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, CUmemLocation, ::core::ffi::c_uint, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPrefetchAsync_v2") });
        _f(devPtr, count, location, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPrefetchAsync_v2(devPtr: CUdeviceptr, count: usize, location: CUmemLocation, flags: ::core::ffi::c_uint, hStream: CUstream) -> CUresult;
        }
        cuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemPrefetchBatchAsync(dptrs: *mut CUdeviceptr, sizes: *mut usize, count: usize, prefetchLocs: *mut CUmemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, usize, *mut CUmemLocation, *mut usize, usize, ::core::ffi::c_ulonglong, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemPrefetchBatchAsync") });
        _f(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemPrefetchBatchAsync(dptrs: *mut CUdeviceptr, sizes: *mut usize, count: usize, prefetchLocs: *mut CUmemLocation, prefetchLocIdxs: *mut usize, numPrefetchLocs: usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult;
        }
        cuMemPrefetchBatchAsync(dptrs, sizes, count, prefetchLocs, prefetchLocIdxs, numPrefetchLocs, flags, hStream)
    }
}
pub unsafe fn cuMemRangeGetAttribute(data: *mut ::core::ffi::c_void, dataSize: usize, attribute: CUmem_range_attribute, devPtr: CUdeviceptr, count: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, usize, CUmem_range_attribute, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemRangeGetAttribute") });
        _f(data, dataSize, attribute, devPtr, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemRangeGetAttribute(data: *mut ::core::ffi::c_void, dataSize: usize, attribute: CUmem_range_attribute, devPtr: CUdeviceptr, count: usize) -> CUresult;
        }
        cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
    }
}
pub unsafe fn cuMemRangeGetAttributes(data: *mut *mut ::core::ffi::c_void, dataSizes: *mut usize, attributes: *mut CUmem_range_attribute, numAttributes: usize, devPtr: CUdeviceptr, count: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_void, *mut usize, *mut CUmem_range_attribute, usize, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemRangeGetAttributes") });
        _f(data, dataSizes, attributes, numAttributes, devPtr, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemRangeGetAttributes(data: *mut *mut ::core::ffi::c_void, dataSizes: *mut usize, attributes: *mut CUmem_range_attribute, numAttributes: usize, devPtr: CUdeviceptr, count: usize) -> CUresult;
        }
        cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)
    }
}
pub unsafe fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemRelease") });
        _f(handle)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult;
        }
        cuMemRelease(handle)
    }
}
pub unsafe fn cuMemRetainAllocationHandle(handle: *mut CUmemGenericAllocationHandle, addr: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemGenericAllocationHandle, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemRetainAllocationHandle") });
        _f(handle, addr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemRetainAllocationHandle(handle: *mut CUmemGenericAllocationHandle, addr: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuMemRetainAllocationHandle(handle, addr)
    }
}
pub unsafe fn cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemSetAccess") });
        _f(ptr, size, desc, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemSetAccess(ptr: CUdeviceptr, size: usize, desc: *const CUmemAccessDesc, count: usize) -> CUresult;
        }
        cuMemSetAccess(ptr, size, desc, count)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemSetMemPool(location: *mut CUmemLocation, type_: CUmemAllocationType, pool: CUmemoryPool) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemLocation, CUmemAllocationType, CUmemoryPool) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemSetMemPool") });
        _f(location, type_, pool)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemSetMemPool(location: *mut CUmemLocation, type_: CUmemAllocationType, pool: CUmemoryPool) -> CUresult;
        }
        cuMemSetMemPool(location, type_, pool)
    }
}
pub unsafe fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemUnmap") });
        _f(ptr, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult;
        }
        cuMemUnmap(ptr, size)
    }
}
pub unsafe fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy") });
        _f(dst, src, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) -> CUresult;
        }
        cuMemcpy(dst, src, ByteCount)
    }
}
pub unsafe fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY2D, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy2DAsync_v2") });
        _f(pCopy, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) -> CUresult;
        }
        cuMemcpy2DAsync_v2(pCopy, hStream)
    }
}
pub unsafe fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy2DUnaligned_v2") });
        _f(pCopy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;
        }
        cuMemcpy2DUnaligned_v2(pCopy)
    }
}
pub unsafe fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY2D) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy2D_v2") });
        _f(pCopy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;
        }
        cuMemcpy2D_v2(pCopy)
    }
}
pub unsafe fn cuMemcpy3DAsync_v2(pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY3D, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3DAsync_v2") });
        _f(pCopy, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3DAsync_v2(pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) -> CUresult;
        }
        cuMemcpy3DAsync_v2(pCopy, hStream)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuMemcpy3DBatchAsync(numOps: usize, opList: *mut CUDA_MEMCPY3D_BATCH_OP, failIdx: *mut usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(usize, *mut CUDA_MEMCPY3D_BATCH_OP, *mut usize, ::core::ffi::c_ulonglong, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3DBatchAsync") });
        _f(numOps, opList, failIdx, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3DBatchAsync(numOps: usize, opList: *mut CUDA_MEMCPY3D_BATCH_OP, failIdx: *mut usize, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult;
        }
        cuMemcpy3DBatchAsync(numOps, opList, failIdx, flags, hStream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemcpy3DBatchAsync_v2(numOps: usize, opList: *mut CUDA_MEMCPY3D_BATCH_OP, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(usize, *mut CUDA_MEMCPY3D_BATCH_OP, ::core::ffi::c_ulonglong, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3DBatchAsync_v2") });
        _f(numOps, opList, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3DBatchAsync_v2(numOps: usize, opList: *mut CUDA_MEMCPY3D_BATCH_OP, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult;
        }
        cuMemcpy3DBatchAsync_v2(numOps, opList, flags, hStream)
    }
}
pub unsafe fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY3D_PEER) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3DPeer") });
        _f(pCopy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult;
        }
        cuMemcpy3DPeer(pCopy)
    }
}
pub unsafe fn cuMemcpy3DPeerAsync(pCopy: *const CUDA_MEMCPY3D_PEER, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY3D_PEER, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3DPeerAsync") });
        _f(pCopy, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3DPeerAsync(pCopy: *const CUDA_MEMCPY3D_PEER, hStream: CUstream) -> CUresult;
        }
        cuMemcpy3DPeerAsync(pCopy, hStream)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemcpy3DWithAttributesAsync(op: *mut CUDA_MEMCPY3D_BATCH_OP, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_MEMCPY3D_BATCH_OP, ::core::ffi::c_ulonglong, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3DWithAttributesAsync") });
        _f(op, flags, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3DWithAttributesAsync(op: *mut CUDA_MEMCPY3D_BATCH_OP, flags: ::core::ffi::c_ulonglong, hStream: CUstream) -> CUresult;
        }
        cuMemcpy3DWithAttributesAsync(op, flags, hStream)
    }
}
pub unsafe fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUDA_MEMCPY3D) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpy3D_v2") });
        _f(pCopy)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult;
        }
        cuMemcpy3D_v2(pCopy)
    }
}
pub unsafe fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyAsync") });
        _f(dst, src, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyAsync(dst, src, ByteCount, hStream)
    }
}
pub unsafe fn cuMemcpyAtoA_v2(dstArray: CUarray, dstOffset: usize, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUarray, usize, CUarray, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyAtoA_v2") });
        _f(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyAtoA_v2(dstArray: CUarray, dstOffset: usize, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult;
        }
        cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    }
}
pub unsafe fn cuMemcpyAtoD_v2(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUarray, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyAtoD_v2") });
        _f(dstDevice, srcArray, srcOffset, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyAtoD_v2(dstDevice: CUdeviceptr, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult;
        }
        cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount)
    }
}
pub unsafe fn cuMemcpyAtoHAsync_v2(dstHost: *mut ::core::ffi::c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUarray, usize, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyAtoHAsync_v2") });
        _f(dstHost, srcArray, srcOffset, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyAtoHAsync_v2(dstHost: *mut ::core::ffi::c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream)
    }
}
pub unsafe fn cuMemcpyAtoH_v2(dstHost: *mut ::core::ffi::c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUarray, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyAtoH_v2") });
        _f(dstHost, srcArray, srcOffset, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyAtoH_v2(dstHost: *mut ::core::ffi::c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize) -> CUresult;
        }
        cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuMemcpyBatchAsync(dsts: *mut CUdeviceptr, srcs: *mut CUdeviceptr, sizes: *mut usize, count: usize, attrs: *mut CUmemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, failIdx: *mut usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut CUdeviceptr, *mut usize, usize, *mut CUmemcpyAttributes, *mut usize, usize, *mut usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyBatchAsync") });
        _f(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyBatchAsync(dsts: *mut CUdeviceptr, srcs: *mut CUdeviceptr, sizes: *mut usize, count: usize, attrs: *mut CUmemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, failIdx: *mut usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, hStream)
    }
}
#[cfg(any(feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemcpyBatchAsync_v2(dsts: *mut CUdeviceptr, srcs: *mut CUdeviceptr, sizes: *mut usize, count: usize, attrs: *mut CUmemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut CUdeviceptr, *mut usize, usize, *mut CUmemcpyAttributes, *mut usize, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyBatchAsync_v2") });
        _f(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyBatchAsync_v2(dsts: *mut CUdeviceptr, srcs: *mut CUdeviceptr, sizes: *mut usize, count: usize, attrs: *mut CUmemcpyAttributes, attrsIdxs: *mut usize, numAttrs: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyBatchAsync_v2(dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, hStream)
    }
}
pub unsafe fn cuMemcpyDtoA_v2(dstArray: CUarray, dstOffset: usize, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUarray, usize, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyDtoA_v2") });
        _f(dstArray, dstOffset, srcDevice, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyDtoA_v2(dstArray: CUarray, dstOffset: usize, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;
        }
        cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount)
    }
}
pub unsafe fn cuMemcpyDtoDAsync_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyDtoDAsync_v2") });
        _f(dstDevice, srcDevice, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyDtoDAsync_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)
    }
}
pub unsafe fn cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyDtoD_v2") });
        _f(dstDevice, srcDevice, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;
        }
        cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount)
    }
}
pub unsafe fn cuMemcpyDtoHAsync_v2(dstHost: *mut ::core::ffi::c_void, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUdeviceptr, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyDtoHAsync_v2") });
        _f(dstHost, srcDevice, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyDtoHAsync_v2(dstHost: *mut ::core::ffi::c_void, srcDevice: CUdeviceptr, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)
    }
}
pub unsafe fn cuMemcpyDtoH_v2(dstHost: *mut ::core::ffi::c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyDtoH_v2") });
        _f(dstHost, srcDevice, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyDtoH_v2(dstHost: *mut ::core::ffi::c_void, srcDevice: CUdeviceptr, ByteCount: usize) -> CUresult;
        }
        cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)
    }
}
pub unsafe fn cuMemcpyHtoAAsync_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const ::core::ffi::c_void, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUarray, usize, *const ::core::ffi::c_void, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyHtoAAsync_v2") });
        _f(dstArray, dstOffset, srcHost, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyHtoAAsync_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const ::core::ffi::c_void, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream)
    }
}
pub unsafe fn cuMemcpyHtoA_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const ::core::ffi::c_void, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUarray, usize, *const ::core::ffi::c_void, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyHtoA_v2") });
        _f(dstArray, dstOffset, srcHost, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyHtoA_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const ::core::ffi::c_void, ByteCount: usize) -> CUresult;
        }
        cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount)
    }
}
pub unsafe fn cuMemcpyHtoDAsync_v2(dstDevice: CUdeviceptr, srcHost: *const ::core::ffi::c_void, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, *const ::core::ffi::c_void, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyHtoDAsync_v2") });
        _f(dstDevice, srcHost, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyHtoDAsync_v2(dstDevice: CUdeviceptr, srcHost: *const ::core::ffi::c_void, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)
    }
}
pub unsafe fn cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: *const ::core::ffi::c_void, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, *const ::core::ffi::c_void, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyHtoD_v2") });
        _f(dstDevice, srcHost, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyHtoD_v2(dstDevice: CUdeviceptr, srcHost: *const ::core::ffi::c_void, ByteCount: usize) -> CUresult;
        }
        cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
    }
}
pub unsafe fn cuMemcpyPeer(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyPeer") });
        _f(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyPeer(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize) -> CUresult;
        }
        cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    }
}
pub unsafe fn cuMemcpyPeerAsync(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyPeerAsync") });
        _f(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyPeerAsync(dstDevice: CUdeviceptr, dstContext: CUcontext, srcDevice: CUdeviceptr, srcContext: CUcontext, ByteCount: usize, hStream: CUstream) -> CUresult;
        }
        cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMemcpyWithAttributesAsync(dst: CUdeviceptr, src: CUdeviceptr, size: usize, attr: *mut CUmemcpyAttributes, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, *mut CUmemcpyAttributes, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemcpyWithAttributesAsync") });
        _f(dst, src, size, attr, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemcpyWithAttributesAsync(dst: CUdeviceptr, src: CUdeviceptr, size: usize, attr: *mut CUmemcpyAttributes, hStream: CUstream) -> CUresult;
        }
        cuMemcpyWithAttributesAsync(dst, src, size, attr, hStream)
    }
}
pub unsafe fn cuMemsetD16Async(dstDevice: CUdeviceptr, us: ::core::ffi::c_ushort, N: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_ushort, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD16Async") });
        _f(dstDevice, us, N, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD16Async(dstDevice: CUdeviceptr, us: ::core::ffi::c_ushort, N: usize, hStream: CUstream) -> CUresult;
        }
        cuMemsetD16Async(dstDevice, us, N, hStream)
    }
}
pub unsafe fn cuMemsetD16_v2(dstDevice: CUdeviceptr, us: ::core::ffi::c_ushort, N: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_ushort, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD16_v2") });
        _f(dstDevice, us, N)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD16_v2(dstDevice: CUdeviceptr, us: ::core::ffi::c_ushort, N: usize) -> CUresult;
        }
        cuMemsetD16_v2(dstDevice, us, N)
    }
}
pub unsafe fn cuMemsetD2D16Async(dstDevice: CUdeviceptr, dstPitch: usize, us: ::core::ffi::c_ushort, Width: usize, Height: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, ::core::ffi::c_ushort, usize, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD2D16Async") });
        _f(dstDevice, dstPitch, us, Width, Height, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD2D16Async(dstDevice: CUdeviceptr, dstPitch: usize, us: ::core::ffi::c_ushort, Width: usize, Height: usize, hStream: CUstream) -> CUresult;
        }
        cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream)
    }
}
pub unsafe fn cuMemsetD2D16_v2(dstDevice: CUdeviceptr, dstPitch: usize, us: ::core::ffi::c_ushort, Width: usize, Height: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, ::core::ffi::c_ushort, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD2D16_v2") });
        _f(dstDevice, dstPitch, us, Width, Height)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD2D16_v2(dstDevice: CUdeviceptr, dstPitch: usize, us: ::core::ffi::c_ushort, Width: usize, Height: usize) -> CUresult;
        }
        cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height)
    }
}
pub unsafe fn cuMemsetD2D32Async(dstDevice: CUdeviceptr, dstPitch: usize, ui: ::core::ffi::c_uint, Width: usize, Height: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, ::core::ffi::c_uint, usize, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD2D32Async") });
        _f(dstDevice, dstPitch, ui, Width, Height, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD2D32Async(dstDevice: CUdeviceptr, dstPitch: usize, ui: ::core::ffi::c_uint, Width: usize, Height: usize, hStream: CUstream) -> CUresult;
        }
        cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream)
    }
}
pub unsafe fn cuMemsetD2D32_v2(dstDevice: CUdeviceptr, dstPitch: usize, ui: ::core::ffi::c_uint, Width: usize, Height: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, ::core::ffi::c_uint, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD2D32_v2") });
        _f(dstDevice, dstPitch, ui, Width, Height)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD2D32_v2(dstDevice: CUdeviceptr, dstPitch: usize, ui: ::core::ffi::c_uint, Width: usize, Height: usize) -> CUresult;
        }
        cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height)
    }
}
pub unsafe fn cuMemsetD2D8Async(dstDevice: CUdeviceptr, dstPitch: usize, uc: ::core::ffi::c_uchar, Width: usize, Height: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, ::core::ffi::c_uchar, usize, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD2D8Async") });
        _f(dstDevice, dstPitch, uc, Width, Height, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD2D8Async(dstDevice: CUdeviceptr, dstPitch: usize, uc: ::core::ffi::c_uchar, Width: usize, Height: usize, hStream: CUstream) -> CUresult;
        }
        cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream)
    }
}
pub unsafe fn cuMemsetD2D8_v2(dstDevice: CUdeviceptr, dstPitch: usize, uc: ::core::ffi::c_uchar, Width: usize, Height: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, usize, ::core::ffi::c_uchar, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD2D8_v2") });
        _f(dstDevice, dstPitch, uc, Width, Height)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD2D8_v2(dstDevice: CUdeviceptr, dstPitch: usize, uc: ::core::ffi::c_uchar, Width: usize, Height: usize) -> CUresult;
        }
        cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height)
    }
}
pub unsafe fn cuMemsetD32Async(dstDevice: CUdeviceptr, ui: ::core::ffi::c_uint, N: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_uint, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD32Async") });
        _f(dstDevice, ui, N, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD32Async(dstDevice: CUdeviceptr, ui: ::core::ffi::c_uint, N: usize, hStream: CUstream) -> CUresult;
        }
        cuMemsetD32Async(dstDevice, ui, N, hStream)
    }
}
pub unsafe fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: ::core::ffi::c_uint, N: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_uint, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD32_v2") });
        _f(dstDevice, ui, N)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD32_v2(dstDevice: CUdeviceptr, ui: ::core::ffi::c_uint, N: usize) -> CUresult;
        }
        cuMemsetD32_v2(dstDevice, ui, N)
    }
}
pub unsafe fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: ::core::ffi::c_uchar, N: usize, hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_uchar, usize, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD8Async") });
        _f(dstDevice, uc, N, hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: ::core::ffi::c_uchar, N: usize, hStream: CUstream) -> CUresult;
        }
        cuMemsetD8Async(dstDevice, uc, N, hStream)
    }
}
pub unsafe fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: ::core::ffi::c_uchar, N: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_uchar, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMemsetD8_v2") });
        _f(dstDevice, uc, N)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMemsetD8_v2(dstDevice: CUdeviceptr, uc: ::core::ffi::c_uchar, N: usize) -> CUresult;
        }
        cuMemsetD8_v2(dstDevice, uc, N)
    }
}
pub unsafe fn cuMipmappedArrayCreate(pHandle: *mut CUmipmappedArray, pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmipmappedArray, *const CUDA_ARRAY3D_DESCRIPTOR, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMipmappedArrayCreate") });
        _f(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMipmappedArrayCreate(pHandle: *mut CUmipmappedArray, pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR, numMipmapLevels: ::core::ffi::c_uint) -> CUresult;
        }
        cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    }
}
pub unsafe fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmipmappedArray) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMipmappedArrayDestroy") });
        _f(hMipmappedArray)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult;
        }
        cuMipmappedArrayDestroy(hMipmappedArray)
    }
}
pub unsafe fn cuMipmappedArrayGetLevel(pLevelArray: *mut CUarray, hMipmappedArray: CUmipmappedArray, level: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, CUmipmappedArray, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMipmappedArrayGetLevel") });
        _f(pLevelArray, hMipmappedArray, level)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMipmappedArrayGetLevel(pLevelArray: *mut CUarray, hMipmappedArray: CUmipmappedArray, level: ::core::ffi::c_uint) -> CUresult;
        }
        cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)
    }
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMipmappedArrayGetMemoryRequirements(memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS, mipmap: CUmipmappedArray, device: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_ARRAY_MEMORY_REQUIREMENTS, CUmipmappedArray, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMipmappedArrayGetMemoryRequirements") });
        _f(memoryRequirements, mipmap, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMipmappedArrayGetMemoryRequirements(memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS, mipmap: CUmipmappedArray, device: CUdevice) -> CUresult;
        }
        cuMipmappedArrayGetMemoryRequirements(memoryRequirements, mipmap, device)
    }
}
pub unsafe fn cuMipmappedArrayGetSparseProperties(sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES, mipmap: CUmipmappedArray) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_ARRAY_SPARSE_PROPERTIES, CUmipmappedArray) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMipmappedArrayGetSparseProperties") });
        _f(sparseProperties, mipmap)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMipmappedArrayGetSparseProperties(sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES, mipmap: CUmipmappedArray) -> CUresult;
        }
        cuMipmappedArrayGetSparseProperties(sparseProperties, mipmap)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuModuleEnumerateFunctions(functions: *mut CUfunction, numFunctions: ::core::ffi::c_uint, mod_: CUmodule) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfunction, ::core::ffi::c_uint, CUmodule) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleEnumerateFunctions") });
        _f(functions, numFunctions, mod_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleEnumerateFunctions(functions: *mut CUfunction, numFunctions: ::core::ffi::c_uint, mod_: CUmodule) -> CUresult;
        }
        cuModuleEnumerateFunctions(functions, numFunctions, mod_)
    }
}
pub unsafe fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleGetFunction") });
        _f(hfunc, hmod, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleGetFunction(hfunc: *mut CUfunction, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuModuleGetFunction(hfunc, hmod, name)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuModuleGetFunctionCount(count: *mut ::core::ffi::c_uint, mod_: CUmodule) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, CUmodule) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleGetFunctionCount") });
        _f(count, mod_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleGetFunctionCount(count: *mut ::core::ffi::c_uint, mod_: CUmodule) -> CUresult;
        }
        cuModuleGetFunctionCount(count, mod_)
    }
}
pub unsafe fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUmodule, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleGetGlobal_v2") });
        _f(dptr, bytes, hmod, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleGetGlobal_v2(dptr: *mut CUdeviceptr, bytes: *mut usize, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuModuleGetGlobal_v2(dptr, bytes, hmod, name)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuModuleGetLoadingMode(mode: *mut CUmoduleLoadingMode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmoduleLoadingMode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleGetLoadingMode") });
        _f(mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleGetLoadingMode(mode: *mut CUmoduleLoadingMode) -> CUresult;
        }
        cuModuleGetLoadingMode(mode)
    }
}
pub unsafe fn cuModuleGetSurfRef(pSurfRef: *mut CUsurfref, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUsurfref, CUmodule, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleGetSurfRef") });
        _f(pSurfRef, hmod, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleGetSurfRef(pSurfRef: *mut CUsurfref, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuModuleGetSurfRef(pSurfRef, hmod, name)
    }
}
pub unsafe fn cuModuleGetTexRef(pTexRef: *mut CUtexref, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtexref, CUmodule, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleGetTexRef") });
        _f(pTexRef, hmod, name)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleGetTexRef(pTexRef: *mut CUtexref, hmod: CUmodule, name: *const ::core::ffi::c_char) -> CUresult;
        }
        cuModuleGetTexRef(pTexRef, hmod, name)
    }
}
pub unsafe fn cuModuleLoad(module: *mut CUmodule, fname: *const ::core::ffi::c_char) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmodule, *const ::core::ffi::c_char) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleLoad") });
        _f(module, fname)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleLoad(module: *mut CUmodule, fname: *const ::core::ffi::c_char) -> CUresult;
        }
        cuModuleLoad(module, fname)
    }
}
pub unsafe fn cuModuleLoadData(module: *mut CUmodule, image: *const ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmodule, *const ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleLoadData") });
        _f(module, image)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleLoadData(module: *mut CUmodule, image: *const ::core::ffi::c_void) -> CUresult;
        }
        cuModuleLoadData(module, image)
    }
}
pub unsafe fn cuModuleLoadDataEx(module: *mut CUmodule, image: *const ::core::ffi::c_void, numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmodule, *const ::core::ffi::c_void, ::core::ffi::c_uint, *mut CUjit_option, *mut *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleLoadDataEx") });
        _f(module, image, numOptions, options, optionValues)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleLoadDataEx(module: *mut CUmodule, image: *const ::core::ffi::c_void, numOptions: ::core::ffi::c_uint, options: *mut CUjit_option, optionValues: *mut *mut ::core::ffi::c_void) -> CUresult;
        }
        cuModuleLoadDataEx(module, image, numOptions, options, optionValues)
    }
}
pub unsafe fn cuModuleLoadFatBinary(module: *mut CUmodule, fatCubin: *const ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmodule, *const ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleLoadFatBinary") });
        _f(module, fatCubin)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleLoadFatBinary(module: *mut CUmodule, fatCubin: *const ::core::ffi::c_void) -> CUresult;
        }
        cuModuleLoadFatBinary(module, fatCubin)
    }
}
pub unsafe fn cuModuleUnload(hmod: CUmodule) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmodule) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuModuleUnload") });
        _f(hmod)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuModuleUnload(hmod: CUmodule) -> CUresult;
        }
        cuModuleUnload(hmod)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastAddDevice(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle, CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastAddDevice") });
        _f(mcHandle, dev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastAddDevice(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice) -> CUresult;
        }
        cuMulticastAddDevice(mcHandle, dev)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastBindAddr(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memptr: CUdeviceptr, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle, usize, CUdeviceptr, usize, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastBindAddr") });
        _f(mcHandle, mcOffset, memptr, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastBindAddr(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memptr: CUdeviceptr, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMulticastBindAddr(mcHandle, mcOffset, memptr, size, flags)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastBindAddr_v2(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, memptr: CUdeviceptr, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle, CUdevice, usize, CUdeviceptr, usize, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastBindAddr_v2") });
        _f(mcHandle, dev, mcOffset, memptr, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastBindAddr_v2(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, memptr: CUdeviceptr, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMulticastBindAddr_v2(mcHandle, dev, mcOffset, memptr, size, flags)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastBindMem(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memHandle: CUmemGenericAllocationHandle, memOffset: usize, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle, usize, CUmemGenericAllocationHandle, usize, usize, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastBindMem") });
        _f(mcHandle, mcOffset, memHandle, memOffset, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastBindMem(mcHandle: CUmemGenericAllocationHandle, mcOffset: usize, memHandle: CUmemGenericAllocationHandle, memOffset: usize, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, flags)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastBindMem_v2(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, memHandle: CUmemGenericAllocationHandle, memOffset: usize, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle, CUdevice, usize, CUmemGenericAllocationHandle, usize, usize, ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastBindMem_v2") });
        _f(mcHandle, dev, mcOffset, memHandle, memOffset, size, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastBindMem_v2(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, memHandle: CUmemGenericAllocationHandle, memOffset: usize, size: usize, flags: ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuMulticastBindMem_v2(mcHandle, dev, mcOffset, memHandle, memOffset, size, flags)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastCreate(mcHandle: *mut CUmemGenericAllocationHandle, prop: *const CUmulticastObjectProp) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmemGenericAllocationHandle, *const CUmulticastObjectProp) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastCreate") });
        _f(mcHandle, prop)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastCreate(mcHandle: *mut CUmemGenericAllocationHandle, prop: *const CUmulticastObjectProp) -> CUresult;
        }
        cuMulticastCreate(mcHandle, prop)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastGetGranularity(granularity: *mut usize, prop: *const CUmulticastObjectProp, option: CUmulticastGranularity_flags) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, *const CUmulticastObjectProp, CUmulticastGranularity_flags) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastGetGranularity") });
        _f(granularity, prop, option)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastGetGranularity(granularity: *mut usize, prop: *const CUmulticastObjectProp, option: CUmulticastGranularity_flags) -> CUresult;
        }
        cuMulticastGetGranularity(granularity, prop, option)
    }
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuMulticastUnbind(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, size: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUmemGenericAllocationHandle, CUdevice, usize, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuMulticastUnbind") });
        _f(mcHandle, dev, mcOffset, size)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuMulticastUnbind(mcHandle: CUmemGenericAllocationHandle, dev: CUdevice, mcOffset: usize, size: usize) -> CUresult;
        }
        cuMulticastUnbind(mcHandle, dev, mcOffset, size)
    }
}
pub unsafe fn cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: *mut usize, func: CUfunction, numBlocks: ::core::ffi::c_int, blockSize: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, CUfunction, ::core::ffi::c_int, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyAvailableDynamicSMemPerBlock") });
        _f(dynamicSmemSize, func, numBlocks, blockSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize: *mut usize, func: CUfunction, numBlocks: ::core::ffi::c_int, blockSize: ::core::ffi::c_int) -> CUresult;
        }
        cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks, blockSize)
    }
}
pub unsafe fn cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *mut ::core::ffi::c_int, func: CUfunction, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUfunction, ::core::ffi::c_int, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyMaxActiveBlocksPerMultiprocessor") });
        _f(numBlocks, func, blockSize, dynamicSMemSize)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *mut ::core::ffi::c_int, func: CUfunction, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize) -> CUresult;
        }
        cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize)
    }
}
pub unsafe fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut ::core::ffi::c_int, func: CUfunction, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUfunction, ::core::ffi::c_int, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") });
        _f(numBlocks, func, blockSize, dynamicSMemSize, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks: *mut ::core::ffi::c_int, func: CUfunction, blockSize: ::core::ffi::c_int, dynamicSMemSize: usize, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuOccupancyMaxActiveClusters(numClusters: *mut ::core::ffi::c_int, func: CUfunction, config: *const CUlaunchConfig) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUfunction, *const CUlaunchConfig) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyMaxActiveClusters") });
        _f(numClusters, func, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyMaxActiveClusters(numClusters: *mut ::core::ffi::c_int, func: CUfunction, config: *const CUlaunchConfig) -> CUresult;
        }
        cuOccupancyMaxActiveClusters(numClusters, func, config)
    }
}
pub unsafe fn cuOccupancyMaxPotentialBlockSize(minGridSize: *mut ::core::ffi::c_int, blockSize: *mut ::core::ffi::c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *mut ::core::ffi::c_int, CUfunction, CUoccupancyB2DSize, usize, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyMaxPotentialBlockSize") });
        _f(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyMaxPotentialBlockSize(minGridSize: *mut ::core::ffi::c_int, blockSize: *mut ::core::ffi::c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: ::core::ffi::c_int) -> CUresult;
        }
        cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit)
    }
}
pub unsafe fn cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize: *mut ::core::ffi::c_int, blockSize: *mut ::core::ffi::c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, *mut ::core::ffi::c_int, CUfunction, CUoccupancyB2DSize, usize, ::core::ffi::c_int, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyMaxPotentialBlockSizeWithFlags") });
        _f(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize: *mut ::core::ffi::c_int, blockSize: *mut ::core::ffi::c_int, func: CUfunction, blockSizeToDynamicSMemSize: CUoccupancyB2DSize, dynamicSMemSize: usize, blockSizeLimit: ::core::ffi::c_int, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags)
    }
}
#[cfg(any(feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuOccupancyMaxPotentialClusterSize(clusterSize: *mut ::core::ffi::c_int, func: CUfunction, config: *const CUlaunchConfig) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUfunction, *const CUlaunchConfig) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuOccupancyMaxPotentialClusterSize") });
        _f(clusterSize, func, config)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuOccupancyMaxPotentialClusterSize(clusterSize: *mut ::core::ffi::c_int, func: CUfunction, config: *const CUlaunchConfig) -> CUresult;
        }
        cuOccupancyMaxPotentialClusterSize(clusterSize, func, config)
    }
}
pub unsafe fn cuParamSetSize(hfunc: CUfunction, numbytes: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuParamSetSize") });
        _f(hfunc, numbytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuParamSetSize(hfunc: CUfunction, numbytes: ::core::ffi::c_uint) -> CUresult;
        }
        cuParamSetSize(hfunc, numbytes)
    }
}
pub unsafe fn cuParamSetTexRef(hfunc: CUfunction, texunit: ::core::ffi::c_int, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuParamSetTexRef") });
        _f(hfunc, texunit, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuParamSetTexRef(hfunc: CUfunction, texunit: ::core::ffi::c_int, hTexRef: CUtexref) -> CUresult;
        }
        cuParamSetTexRef(hfunc, texunit, hTexRef)
    }
}
pub unsafe fn cuParamSetf(hfunc: CUfunction, offset: ::core::ffi::c_int, value: f32) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, f32) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuParamSetf") });
        _f(hfunc, offset, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuParamSetf(hfunc: CUfunction, offset: ::core::ffi::c_int, value: f32) -> CUresult;
        }
        cuParamSetf(hfunc, offset, value)
    }
}
pub unsafe fn cuParamSeti(hfunc: CUfunction, offset: ::core::ffi::c_int, value: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuParamSeti") });
        _f(hfunc, offset, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuParamSeti(hfunc: CUfunction, offset: ::core::ffi::c_int, value: ::core::ffi::c_uint) -> CUresult;
        }
        cuParamSeti(hfunc, offset, value)
    }
}
pub unsafe fn cuParamSetv(hfunc: CUfunction, offset: ::core::ffi::c_int, ptr: *mut ::core::ffi::c_void, numbytes: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUfunction, ::core::ffi::c_int, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuParamSetv") });
        _f(hfunc, offset, ptr, numbytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuParamSetv(hfunc: CUfunction, offset: ::core::ffi::c_int, ptr: *mut ::core::ffi::c_void, numbytes: ::core::ffi::c_uint) -> CUresult;
        }
        cuParamSetv(hfunc, offset, ptr, numbytes)
    }
}
pub unsafe fn cuPointerGetAttribute(data: *mut ::core::ffi::c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_void, CUpointer_attribute, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuPointerGetAttribute") });
        _f(data, attribute, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuPointerGetAttribute(data: *mut ::core::ffi::c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult;
        }
        cuPointerGetAttribute(data, attribute, ptr)
    }
}
pub unsafe fn cuPointerGetAttributes(numAttributes: ::core::ffi::c_uint, attributes: *mut CUpointer_attribute, data: *mut *mut ::core::ffi::c_void, ptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(::core::ffi::c_uint, *mut CUpointer_attribute, *mut *mut ::core::ffi::c_void, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuPointerGetAttributes") });
        _f(numAttributes, attributes, data, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuPointerGetAttributes(numAttributes: ::core::ffi::c_uint, attributes: *mut CUpointer_attribute, data: *mut *mut ::core::ffi::c_void, ptr: CUdeviceptr) -> CUresult;
        }
        cuPointerGetAttributes(numAttributes, attributes, data, ptr)
    }
}
pub unsafe fn cuPointerSetAttribute(value: *const ::core::ffi::c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_void, CUpointer_attribute, CUdeviceptr) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuPointerSetAttribute") });
        _f(value, attribute, ptr)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuPointerSetAttribute(value: *const ::core::ffi::c_void, attribute: CUpointer_attribute, ptr: CUdeviceptr) -> CUresult;
        }
        cuPointerSetAttribute(value, attribute, ptr)
    }
}
pub unsafe fn cuProfilerInitialize(configFile: *const ::core::ffi::c_char, outputFile: *const ::core::ffi::c_char, outputMode: CUoutput_mode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const ::core::ffi::c_char, *const ::core::ffi::c_char, CUoutput_mode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuProfilerInitialize") });
        _f(configFile, outputFile, outputMode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuProfilerInitialize(configFile: *const ::core::ffi::c_char, outputFile: *const ::core::ffi::c_char, outputMode: CUoutput_mode) -> CUresult;
        }
        cuProfilerInitialize(configFile, outputFile, outputMode)
    }
}
pub unsafe fn cuProfilerStart() -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuProfilerStart") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuProfilerStart() -> CUresult;
        }
        cuProfilerStart()
    }
}
pub unsafe fn cuProfilerStop() -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn() -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuProfilerStop") });
        _f()
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuProfilerStop() -> CUresult;
        }
        cuProfilerStop()
    }
}
pub unsafe fn cuSignalExternalSemaphoresAsync(extSemArray: *const CUexternalSemaphore, paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, numExtSems: ::core::ffi::c_uint, stream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUexternalSemaphore, *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, ::core::ffi::c_uint, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuSignalExternalSemaphoresAsync") });
        _f(extSemArray, paramsArray, numExtSems, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuSignalExternalSemaphoresAsync(extSemArray: *const CUexternalSemaphore, paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, numExtSems: ::core::ffi::c_uint, stream: CUstream) -> CUresult;
        }
        cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    }
}
pub unsafe fn cuStreamAddCallback(hStream: CUstream, callback: CUstreamCallback, userData: *mut ::core::ffi::c_void, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUstreamCallback, *mut ::core::ffi::c_void, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamAddCallback") });
        _f(hStream, callback, userData, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamAddCallback(hStream: CUstream, callback: CUstreamCallback, userData: *mut ::core::ffi::c_void, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamAddCallback(hStream, callback, userData, flags)
    }
}
pub unsafe fn cuStreamAttachMemAsync(hStream: CUstream, dptr: CUdeviceptr, length: usize, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamAttachMemAsync") });
        _f(hStream, dptr, length, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamAttachMemAsync(hStream: CUstream, dptr: CUdeviceptr, length: usize, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamAttachMemAsync(hStream, dptr, length, flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuStreamBatchMemOp(stream: CUstream, count: ::core::ffi::c_uint, paramArray: *mut CUstreamBatchMemOpParams, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, ::core::ffi::c_uint, *mut CUstreamBatchMemOpParams, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamBatchMemOp") });
        _f(stream, count, paramArray, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamBatchMemOp(stream: CUstream, count: ::core::ffi::c_uint, paramArray: *mut CUstreamBatchMemOpParams, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamBatchMemOp(stream, count, paramArray, flags)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamBatchMemOp_v2(stream: CUstream, count: ::core::ffi::c_uint, paramArray: *mut CUstreamBatchMemOpParams, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, ::core::ffi::c_uint, *mut CUstreamBatchMemOpParams, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamBatchMemOp_v2") });
        _f(stream, count, paramArray, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamBatchMemOp_v2(stream: CUstream, count: ::core::ffi::c_uint, paramArray: *mut CUstreamBatchMemOpParams, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamBatchMemOp_v2(stream, count, paramArray, flags)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamBeginCaptureToCig(hStream: CUstream, streamCigCaptureParams: *mut CUstreamCigCaptureParams) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUstreamCigCaptureParams) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamBeginCaptureToCig") });
        _f(hStream, streamCigCaptureParams)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamBeginCaptureToCig(hStream: CUstream, streamCigCaptureParams: *mut CUstreamCigCaptureParams) -> CUresult;
        }
        cuStreamBeginCaptureToCig(hStream, streamCigCaptureParams)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamBeginCaptureToGraph(hStream: CUstream, hGraph: CUgraph, dependencies: *const CUgraphNode, dependencyData: *const CUgraphEdgeData, numDependencies: usize, mode: CUstreamCaptureMode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUgraph, *const CUgraphNode, *const CUgraphEdgeData, usize, CUstreamCaptureMode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamBeginCaptureToGraph") });
        _f(hStream, hGraph, dependencies, dependencyData, numDependencies, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamBeginCaptureToGraph(hStream: CUstream, hGraph: CUgraph, dependencies: *const CUgraphNode, dependencyData: *const CUgraphEdgeData, numDependencies: usize, mode: CUstreamCaptureMode) -> CUresult;
        }
        cuStreamBeginCaptureToGraph(hStream, hGraph, dependencies, dependencyData, numDependencies, mode)
    }
}
pub unsafe fn cuStreamBeginCapture_v2(hStream: CUstream, mode: CUstreamCaptureMode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUstreamCaptureMode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamBeginCapture_v2") });
        _f(hStream, mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamBeginCapture_v2(hStream: CUstream, mode: CUstreamCaptureMode) -> CUresult;
        }
        cuStreamBeginCapture_v2(hStream, mode)
    }
}
#[cfg(any(feature = "cuda-13030"))]
pub unsafe fn cuStreamBeginRecaptureToGraph(hStream: CUstream, mode: CUstreamCaptureMode, hGraph: CUgraph, callbackFunc: CUgraphRecaptureCallback, userData: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUstreamCaptureMode, CUgraph, CUgraphRecaptureCallback, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamBeginRecaptureToGraph") });
        _f(hStream, mode, hGraph, callbackFunc, userData)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamBeginRecaptureToGraph(hStream: CUstream, mode: CUstreamCaptureMode, hGraph: CUgraph, callbackFunc: CUgraphRecaptureCallback, userData: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuStreamBeginRecaptureToGraph(hStream, mode, hGraph, callbackFunc, userData)
    }
}
pub unsafe fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamCopyAttributes") });
        _f(dst, src)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult;
        }
        cuStreamCopyAttributes(dst, src)
    }
}
pub unsafe fn cuStreamCreate(phStream: *mut CUstream, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUstream, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamCreate") });
        _f(phStream, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamCreate(phStream: *mut CUstream, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamCreate(phStream, Flags)
    }
}
pub unsafe fn cuStreamCreateWithPriority(phStream: *mut CUstream, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUstream, ::core::ffi::c_uint, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamCreateWithPriority") });
        _f(phStream, flags, priority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamCreateWithPriority(phStream: *mut CUstream, flags: ::core::ffi::c_uint, priority: ::core::ffi::c_int) -> CUresult;
        }
        cuStreamCreateWithPriority(phStream, flags, priority)
    }
}
pub unsafe fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamDestroy_v2") });
        _f(hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;
        }
        cuStreamDestroy_v2(hStream)
    }
}
pub unsafe fn cuStreamEndCapture(hStream: CUstream, phGraph: *mut CUgraph) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamEndCapture") });
        _f(hStream, phGraph)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamEndCapture(hStream: CUstream, phGraph: *mut CUgraph) -> CUresult;
        }
        cuStreamEndCapture(hStream, phGraph)
    }
}
#[cfg(any(feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamEndCaptureToCig(hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamEndCaptureToCig") });
        _f(hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamEndCaptureToCig(hStream: CUstream) -> CUresult;
        }
        cuStreamEndCaptureToCig(hStream)
    }
}
pub unsafe fn cuStreamGetAttribute(hStream: CUstream, attr: CUstreamAttrID, value_out: *mut CUstreamAttrValue) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUstreamAttrID, *mut CUstreamAttrValue) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetAttribute") });
        _f(hStream, attr, value_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetAttribute(hStream: CUstream, attr: CUstreamAttrID, value_out: *mut CUstreamAttrValue) -> CUresult;
        }
        cuStreamGetAttribute(hStream, attr, value_out)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuStreamGetCaptureInfo(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUstreamCaptureStatus, *mut cuuint64_t) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetCaptureInfo") });
        _f(hStream, captureStatus_out, id_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetCaptureInfo(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t) -> CUresult;
        }
        cuStreamGetCaptureInfo(hStream, captureStatus_out, id_out)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuStreamGetCaptureInfo_v2(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t, graph_out: *mut CUgraph, dependencies_out: *mut *const CUgraphNode, numDependencies_out: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUstreamCaptureStatus, *mut cuuint64_t, *mut CUgraph, *mut *const CUgraphNode, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetCaptureInfo_v2") });
        _f(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetCaptureInfo_v2(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t, graph_out: *mut CUgraph, dependencies_out: *mut *const CUgraphNode, numDependencies_out: *mut usize) -> CUresult;
        }
        cuStreamGetCaptureInfo_v2(hStream, captureStatus_out, id_out, graph_out, dependencies_out, numDependencies_out)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamGetCaptureInfo_v3(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t, graph_out: *mut CUgraph, dependencies_out: *mut *const CUgraphNode, edgeData_out: *mut *const CUgraphEdgeData, numDependencies_out: *mut usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUstreamCaptureStatus, *mut cuuint64_t, *mut CUgraph, *mut *const CUgraphNode, *mut *const CUgraphEdgeData, *mut usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetCaptureInfo_v3") });
        _f(hStream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetCaptureInfo_v3(hStream: CUstream, captureStatus_out: *mut CUstreamCaptureStatus, id_out: *mut cuuint64_t, graph_out: *mut CUgraph, dependencies_out: *mut *const CUgraphNode, edgeData_out: *mut *const CUgraphEdgeData, numDependencies_out: *mut usize) -> CUresult;
        }
        cuStreamGetCaptureInfo_v3(hStream, captureStatus_out, id_out, graph_out, dependencies_out, edgeData_out, numDependencies_out)
    }
}
pub unsafe fn cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUcontext) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetCtx") });
        _f(hStream, pctx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult;
        }
        cuStreamGetCtx(hStream, pctx)
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamGetCtx_v2(hStream: CUstream, pCtx: *mut CUcontext, pGreenCtx: *mut CUgreenCtx) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUcontext, *mut CUgreenCtx) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetCtx_v2") });
        _f(hStream, pCtx, pGreenCtx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetCtx_v2(hStream: CUstream, pCtx: *mut CUcontext, pGreenCtx: *mut CUgreenCtx) -> CUresult;
        }
        cuStreamGetCtx_v2(hStream, pCtx, pGreenCtx)
    }
}
#[cfg(any(feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamGetDevResource(hStream: CUstream, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUdevResource, CUdevResourceType) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetDevResource") });
        _f(hStream, resource, type_)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetDevResource(hStream: CUstream, resource: *mut CUdevResource, type_: CUdevResourceType) -> CUresult;
        }
        cuStreamGetDevResource(hStream, resource, type_)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamGetDevice(hStream: CUstream, device: *mut CUdevice) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUdevice) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetDevice") });
        _f(hStream, device)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetDevice(hStream: CUstream, device: *mut CUdevice) -> CUresult;
        }
        cuStreamGetDevice(hStream, device)
    }
}
pub unsafe fn cuStreamGetFlags(hStream: CUstream, flags: *mut ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetFlags") });
        _f(hStream, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetFlags(hStream: CUstream, flags: *mut ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamGetFlags(hStream, flags)
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamGetGreenCtx(hStream: CUstream, phCtx: *mut CUgreenCtx) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUgreenCtx) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetGreenCtx") });
        _f(hStream, phCtx)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetGreenCtx(hStream: CUstream, phCtx: *mut CUgreenCtx) -> CUresult;
        }
        cuStreamGetGreenCtx(hStream, phCtx)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamGetId(hStream: CUstream, streamId: *mut ::core::ffi::c_ulonglong) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut ::core::ffi::c_ulonglong) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetId") });
        _f(hStream, streamId)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetId(hStream: CUstream, streamId: *mut ::core::ffi::c_ulonglong) -> CUresult;
        }
        cuStreamGetId(hStream, streamId)
    }
}
pub unsafe fn cuStreamGetPriority(hStream: CUstream, priority: *mut ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamGetPriority") });
        _f(hStream, priority)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamGetPriority(hStream: CUstream, priority: *mut ::core::ffi::c_int) -> CUresult;
        }
        cuStreamGetPriority(hStream, priority)
    }
}
pub unsafe fn cuStreamIsCapturing(hStream: CUstream, captureStatus: *mut CUstreamCaptureStatus) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUstreamCaptureStatus) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamIsCapturing") });
        _f(hStream, captureStatus)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamIsCapturing(hStream: CUstream, captureStatus: *mut CUstreamCaptureStatus) -> CUresult;
        }
        cuStreamIsCapturing(hStream, captureStatus)
    }
}
pub unsafe fn cuStreamQuery(hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamQuery") });
        _f(hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamQuery(hStream: CUstream) -> CUresult;
        }
        cuStreamQuery(hStream)
    }
}
pub unsafe fn cuStreamSetAttribute(hStream: CUstream, attr: CUstreamAttrID, value: *const CUstreamAttrValue) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUstreamAttrID, *const CUstreamAttrValue) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamSetAttribute") });
        _f(hStream, attr, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamSetAttribute(hStream: CUstream, attr: CUstreamAttrID, value: *const CUstreamAttrValue) -> CUresult;
        }
        cuStreamSetAttribute(hStream, attr, value)
    }
}
pub unsafe fn cuStreamSynchronize(hStream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamSynchronize") });
        _f(hStream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
        }
        cuStreamSynchronize(hStream)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
pub unsafe fn cuStreamUpdateCaptureDependencies(hStream: CUstream, dependencies: *mut CUgraphNode, numDependencies: usize, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUgraphNode, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamUpdateCaptureDependencies") });
        _f(hStream, dependencies, numDependencies, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamUpdateCaptureDependencies(hStream: CUstream, dependencies: *mut CUgraphNode, numDependencies: usize, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamUpdateCaptureDependencies(hStream, dependencies, numDependencies, flags)
    }
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamUpdateCaptureDependencies_v2(hStream: CUstream, dependencies: *mut CUgraphNode, dependencyData: *const CUgraphEdgeData, numDependencies: usize, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, *mut CUgraphNode, *const CUgraphEdgeData, usize, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamUpdateCaptureDependencies_v2") });
        _f(hStream, dependencies, dependencyData, numDependencies, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamUpdateCaptureDependencies_v2(hStream: CUstream, dependencies: *mut CUgraphNode, dependencyData: *const CUgraphEdgeData, numDependencies: usize, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamUpdateCaptureDependencies_v2(hStream, dependencies, dependencyData, numDependencies, flags)
    }
}
pub unsafe fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUevent, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWaitEvent") });
        _f(hStream, hEvent, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWaitEvent(hStream, hEvent, Flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuStreamWaitValue32(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint32_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWaitValue32") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWaitValue32(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWaitValue32(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamWaitValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint32_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWaitValue32_v2") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWaitValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWaitValue32_v2(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuStreamWaitValue64(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint64_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWaitValue64") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWaitValue64(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWaitValue64(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamWaitValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint64_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWaitValue64_v2") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWaitValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWaitValue64_v2(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuStreamWriteValue32(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint32_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWriteValue32") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWriteValue32(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWriteValue32(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamWriteValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint32_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWriteValue32_v2") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWriteValue32_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint32_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWriteValue32_v2(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080"))]
pub unsafe fn cuStreamWriteValue64(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint64_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWriteValue64") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWriteValue64(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWriteValue64(stream, addr, value, flags)
    }
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuStreamWriteValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUstream, CUdeviceptr, cuuint64_t, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuStreamWriteValue64_v2") });
        _f(stream, addr, value, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuStreamWriteValue64_v2(stream: CUstream, addr: CUdeviceptr, value: cuuint64_t, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuStreamWriteValue64_v2(stream, addr, value, flags)
    }
}
pub unsafe fn cuSurfObjectCreate(pSurfObject: *mut CUsurfObject, pResDesc: *const CUDA_RESOURCE_DESC) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUsurfObject, *const CUDA_RESOURCE_DESC) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuSurfObjectCreate") });
        _f(pSurfObject, pResDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuSurfObjectCreate(pSurfObject: *mut CUsurfObject, pResDesc: *const CUDA_RESOURCE_DESC) -> CUresult;
        }
        cuSurfObjectCreate(pSurfObject, pResDesc)
    }
}
pub unsafe fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUsurfObject) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuSurfObjectDestroy") });
        _f(surfObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult;
        }
        cuSurfObjectDestroy(surfObject)
    }
}
pub unsafe fn cuSurfObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, surfObject: CUsurfObject) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_RESOURCE_DESC, CUsurfObject) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuSurfObjectGetResourceDesc") });
        _f(pResDesc, surfObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuSurfObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, surfObject: CUsurfObject) -> CUresult;
        }
        cuSurfObjectGetResourceDesc(pResDesc, surfObject)
    }
}
pub unsafe fn cuSurfRefGetArray(phArray: *mut CUarray, hSurfRef: CUsurfref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, CUsurfref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuSurfRefGetArray") });
        _f(phArray, hSurfRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuSurfRefGetArray(phArray: *mut CUarray, hSurfRef: CUsurfref) -> CUresult;
        }
        cuSurfRefGetArray(phArray, hSurfRef)
    }
}
pub unsafe fn cuSurfRefSetArray(hSurfRef: CUsurfref, hArray: CUarray, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUsurfref, CUarray, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuSurfRefSetArray") });
        _f(hSurfRef, hArray, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuSurfRefSetArray(hSurfRef: CUsurfref, hArray: CUarray, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuSurfRefSetArray(hSurfRef, hArray, Flags)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuTensorMapEncodeIm2col(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut ::core::ffi::c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, pixelBoxLowerCorner: *const ::core::ffi::c_int, pixelBoxUpperCorner: *const ::core::ffi::c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtensorMap, CUtensorMapDataType, cuuint32_t, *mut ::core::ffi::c_void, *const cuuint64_t, *const cuuint64_t, *const ::core::ffi::c_int, *const ::core::ffi::c_int, cuuint32_t, cuuint32_t, *const cuuint32_t, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTensorMapEncodeIm2col") });
        _f(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTensorMapEncodeIm2col(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut ::core::ffi::c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, pixelBoxLowerCorner: *const ::core::ffi::c_int, pixelBoxUpperCorner: *const ::core::ffi::c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult;
        }
        cuTensorMapEncodeIm2col(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill)
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuTensorMapEncodeIm2colWide(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut ::core::ffi::c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, pixelBoxLowerCornerWidth: ::core::ffi::c_int, pixelBoxUpperCornerWidth: ::core::ffi::c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, mode: CUtensorMapIm2ColWideMode, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtensorMap, CUtensorMapDataType, cuuint32_t, *mut ::core::ffi::c_void, *const cuuint64_t, *const cuuint64_t, ::core::ffi::c_int, ::core::ffi::c_int, cuuint32_t, cuuint32_t, *const cuuint32_t, CUtensorMapInterleave, CUtensorMapIm2ColWideMode, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTensorMapEncodeIm2colWide") });
        _f(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTensorMapEncodeIm2colWide(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut ::core::ffi::c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, pixelBoxLowerCornerWidth: ::core::ffi::c_int, pixelBoxUpperCornerWidth: ::core::ffi::c_int, channelsPerPixel: cuuint32_t, pixelsPerColumn: cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, mode: CUtensorMapIm2ColWideMode, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult;
        }
        cuTensorMapEncodeIm2colWide(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuTensorMapEncodeTiled(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut ::core::ffi::c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, boxDim: *const cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtensorMap, CUtensorMapDataType, cuuint32_t, *mut ::core::ffi::c_void, *const cuuint64_t, *const cuuint64_t, *const cuuint32_t, *const cuuint32_t, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTensorMapEncodeTiled") });
        _f(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTensorMapEncodeTiled(tensorMap: *mut CUtensorMap, tensorDataType: CUtensorMapDataType, tensorRank: cuuint32_t, globalAddress: *mut ::core::ffi::c_void, globalDim: *const cuuint64_t, globalStrides: *const cuuint64_t, boxDim: *const cuuint32_t, elementStrides: *const cuuint32_t, interleave: CUtensorMapInterleave, swizzle: CUtensorMapSwizzle, l2Promotion: CUtensorMapL2promotion, oobFill: CUtensorMapFloatOOBfill) -> CUresult;
        }
        cuTensorMapEncodeTiled(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill)
    }
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
pub unsafe fn cuTensorMapReplaceAddress(tensorMap: *mut CUtensorMap, globalAddress: *mut ::core::ffi::c_void) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtensorMap, *mut ::core::ffi::c_void) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTensorMapReplaceAddress") });
        _f(tensorMap, globalAddress)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTensorMapReplaceAddress(tensorMap: *mut CUtensorMap, globalAddress: *mut ::core::ffi::c_void) -> CUresult;
        }
        cuTensorMapReplaceAddress(tensorMap, globalAddress)
    }
}
pub unsafe fn cuTexObjectCreate(pTexObject: *mut CUtexObject, pResDesc: *const CUDA_RESOURCE_DESC, pTexDesc: *const CUDA_TEXTURE_DESC, pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtexObject, *const CUDA_RESOURCE_DESC, *const CUDA_TEXTURE_DESC, *const CUDA_RESOURCE_VIEW_DESC) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexObjectCreate") });
        _f(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexObjectCreate(pTexObject: *mut CUtexObject, pResDesc: *const CUDA_RESOURCE_DESC, pTexDesc: *const CUDA_TEXTURE_DESC, pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC) -> CUresult;
        }
        cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
}
pub unsafe fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexObject) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexObjectDestroy") });
        _f(texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult;
        }
        cuTexObjectDestroy(texObject)
    }
}
pub unsafe fn cuTexObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, texObject: CUtexObject) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_RESOURCE_DESC, CUtexObject) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexObjectGetResourceDesc") });
        _f(pResDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexObjectGetResourceDesc(pResDesc: *mut CUDA_RESOURCE_DESC, texObject: CUtexObject) -> CUresult;
        }
        cuTexObjectGetResourceDesc(pResDesc, texObject)
    }
}
pub unsafe fn cuTexObjectGetResourceViewDesc(pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC, texObject: CUtexObject) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_RESOURCE_VIEW_DESC, CUtexObject) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexObjectGetResourceViewDesc") });
        _f(pResViewDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexObjectGetResourceViewDesc(pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC, texObject: CUtexObject) -> CUresult;
        }
        cuTexObjectGetResourceViewDesc(pResViewDesc, texObject)
    }
}
pub unsafe fn cuTexObjectGetTextureDesc(pTexDesc: *mut CUDA_TEXTURE_DESC, texObject: CUtexObject) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUDA_TEXTURE_DESC, CUtexObject) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexObjectGetTextureDesc") });
        _f(pTexDesc, texObject)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexObjectGetTextureDesc(pTexDesc: *mut CUDA_TEXTURE_DESC, texObject: CUtexObject) -> CUresult;
        }
        cuTexObjectGetTextureDesc(pTexDesc, texObject)
    }
}
pub unsafe fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefCreate") });
        _f(pTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult;
        }
        cuTexRefCreate(pTexRef)
    }
}
pub unsafe fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefDestroy") });
        _f(hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefDestroy(hTexRef)
    }
}
pub unsafe fn cuTexRefGetAddressMode(pam: *mut CUaddress_mode, hTexRef: CUtexref, dim: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUaddress_mode, CUtexref, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetAddressMode") });
        _f(pam, hTexRef, dim)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetAddressMode(pam: *mut CUaddress_mode, hTexRef: CUtexref, dim: ::core::ffi::c_int) -> CUresult;
        }
        cuTexRefGetAddressMode(pam, hTexRef, dim)
    }
}
pub unsafe fn cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUdeviceptr, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetAddress_v2") });
        _f(pdptr, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetAddress_v2(pdptr, hTexRef)
    }
}
pub unsafe fn cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetArray") });
        _f(phArray, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetArray(phArray, hTexRef)
    }
}
pub unsafe fn cuTexRefGetBorderColor(pBorderColor: *mut f32, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetBorderColor") });
        _f(pBorderColor, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetBorderColor(pBorderColor: *mut f32, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetBorderColor(pBorderColor, hTexRef)
    }
}
pub unsafe fn cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfilter_mode, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetFilterMode") });
        _f(pfm, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetFilterMode(pfm, hTexRef)
    }
}
pub unsafe fn cuTexRefGetFlags(pFlags: *mut ::core::ffi::c_uint, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_uint, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetFlags") });
        _f(pFlags, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetFlags(pFlags: *mut ::core::ffi::c_uint, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetFlags(pFlags, hTexRef)
    }
}
pub unsafe fn cuTexRefGetFormat(pFormat: *mut CUarray_format, pNumChannels: *mut ::core::ffi::c_int, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUarray_format, *mut ::core::ffi::c_int, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetFormat") });
        _f(pFormat, pNumChannels, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetFormat(pFormat: *mut CUarray_format, pNumChannels: *mut ::core::ffi::c_int, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetFormat(pFormat, pNumChannels, hTexRef)
    }
}
pub unsafe fn cuTexRefGetMaxAnisotropy(pmaxAniso: *mut ::core::ffi::c_int, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetMaxAnisotropy") });
        _f(pmaxAniso, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetMaxAnisotropy(pmaxAniso: *mut ::core::ffi::c_int, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef)
    }
}
pub unsafe fn cuTexRefGetMipmapFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUfilter_mode, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetMipmapFilterMode") });
        _f(pfm, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetMipmapFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetMipmapFilterMode(pfm, hTexRef)
    }
}
pub unsafe fn cuTexRefGetMipmapLevelBias(pbias: *mut f32, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetMipmapLevelBias") });
        _f(pbias, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetMipmapLevelBias(pbias: *mut f32, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetMipmapLevelBias(pbias, hTexRef)
    }
}
pub unsafe fn cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp: *mut f32, pmaxMipmapLevelClamp: *mut f32, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut f32, *mut f32, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetMipmapLevelClamp") });
        _f(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp: *mut f32, pmaxMipmapLevelClamp: *mut f32, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
    }
}
pub unsafe fn cuTexRefGetMipmappedArray(phMipmappedArray: *mut CUmipmappedArray, hTexRef: CUtexref) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUmipmappedArray, CUtexref) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefGetMipmappedArray") });
        _f(phMipmappedArray, hTexRef)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefGetMipmappedArray(phMipmappedArray: *mut CUmipmappedArray, hTexRef: CUtexref) -> CUresult;
        }
        cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef)
    }
}
pub unsafe fn cuTexRefSetAddress2D_v3(hTexRef: CUtexref, desc: *const CUDA_ARRAY_DESCRIPTOR, dptr: CUdeviceptr, Pitch: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, *const CUDA_ARRAY_DESCRIPTOR, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetAddress2D_v3") });
        _f(hTexRef, desc, dptr, Pitch)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetAddress2D_v3(hTexRef: CUtexref, desc: *const CUDA_ARRAY_DESCRIPTOR, dptr: CUdeviceptr, Pitch: usize) -> CUresult;
        }
        cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch)
    }
}
pub unsafe fn cuTexRefSetAddressMode(hTexRef: CUtexref, dim: ::core::ffi::c_int, am: CUaddress_mode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, ::core::ffi::c_int, CUaddress_mode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetAddressMode") });
        _f(hTexRef, dim, am)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetAddressMode(hTexRef: CUtexref, dim: ::core::ffi::c_int, am: CUaddress_mode) -> CUresult;
        }
        cuTexRefSetAddressMode(hTexRef, dim, am)
    }
}
pub unsafe fn cuTexRefSetAddress_v2(ByteOffset: *mut usize, hTexRef: CUtexref, dptr: CUdeviceptr, bytes: usize) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut usize, CUtexref, CUdeviceptr, usize) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetAddress_v2") });
        _f(ByteOffset, hTexRef, dptr, bytes)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetAddress_v2(ByteOffset: *mut usize, hTexRef: CUtexref, dptr: CUdeviceptr, bytes: usize) -> CUresult;
        }
        cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes)
    }
}
pub unsafe fn cuTexRefSetArray(hTexRef: CUtexref, hArray: CUarray, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, CUarray, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetArray") });
        _f(hTexRef, hArray, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetArray(hTexRef: CUtexref, hArray: CUarray, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuTexRefSetArray(hTexRef, hArray, Flags)
    }
}
pub unsafe fn cuTexRefSetBorderColor(hTexRef: CUtexref, pBorderColor: *mut f32) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, *mut f32) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetBorderColor") });
        _f(hTexRef, pBorderColor)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetBorderColor(hTexRef: CUtexref, pBorderColor: *mut f32) -> CUresult;
        }
        cuTexRefSetBorderColor(hTexRef, pBorderColor)
    }
}
pub unsafe fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, CUfilter_mode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetFilterMode") });
        _f(hTexRef, fm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;
        }
        cuTexRefSetFilterMode(hTexRef, fm)
    }
}
pub unsafe fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetFlags") });
        _f(hTexRef, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuTexRefSetFlags(hTexRef, Flags)
    }
}
pub unsafe fn cuTexRefSetFormat(hTexRef: CUtexref, fmt: CUarray_format, NumPackedComponents: ::core::ffi::c_int) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, CUarray_format, ::core::ffi::c_int) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetFormat") });
        _f(hTexRef, fmt, NumPackedComponents)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetFormat(hTexRef: CUtexref, fmt: CUarray_format, NumPackedComponents: ::core::ffi::c_int) -> CUresult;
        }
        cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents)
    }
}
pub unsafe fn cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetMaxAnisotropy") });
        _f(hTexRef, maxAniso)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetMaxAnisotropy(hTexRef: CUtexref, maxAniso: ::core::ffi::c_uint) -> CUresult;
        }
        cuTexRefSetMaxAnisotropy(hTexRef, maxAniso)
    }
}
pub unsafe fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, CUfilter_mode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetMipmapFilterMode") });
        _f(hTexRef, fm)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;
        }
        cuTexRefSetMipmapFilterMode(hTexRef, fm)
    }
}
pub unsafe fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, f32) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetMipmapLevelBias") });
        _f(hTexRef, bias)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult;
        }
        cuTexRefSetMipmapLevelBias(hTexRef, bias)
    }
}
pub unsafe fn cuTexRefSetMipmapLevelClamp(hTexRef: CUtexref, minMipmapLevelClamp: f32, maxMipmapLevelClamp: f32) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, f32, f32) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetMipmapLevelClamp") });
        _f(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetMipmapLevelClamp(hTexRef: CUtexref, minMipmapLevelClamp: f32, maxMipmapLevelClamp: f32) -> CUresult;
        }
        cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
    }
}
pub unsafe fn cuTexRefSetMipmappedArray(hTexRef: CUtexref, hMipmappedArray: CUmipmappedArray, Flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUtexref, CUmipmappedArray, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuTexRefSetMipmappedArray") });
        _f(hTexRef, hMipmappedArray, Flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuTexRefSetMipmappedArray(hTexRef: CUtexref, hMipmappedArray: CUmipmappedArray, Flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags)
    }
}
pub unsafe fn cuThreadExchangeStreamCaptureMode(mode: *mut CUstreamCaptureMode) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUstreamCaptureMode) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuThreadExchangeStreamCaptureMode") });
        _f(mode)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuThreadExchangeStreamCaptureMode(mode: *mut CUstreamCaptureMode) -> CUresult;
        }
        cuThreadExchangeStreamCaptureMode(mode)
    }
}
pub unsafe fn cuUserObjectCreate(object_out: *mut CUuserObject, ptr: *mut ::core::ffi::c_void, destroy: CUhostFn, initialRefcount: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut CUuserObject, *mut ::core::ffi::c_void, CUhostFn, ::core::ffi::c_uint, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuUserObjectCreate") });
        _f(object_out, ptr, destroy, initialRefcount, flags)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuUserObjectCreate(object_out: *mut CUuserObject, ptr: *mut ::core::ffi::c_void, destroy: CUhostFn, initialRefcount: ::core::ffi::c_uint, flags: ::core::ffi::c_uint) -> CUresult;
        }
        cuUserObjectCreate(object_out, ptr, destroy, initialRefcount, flags)
    }
}
pub unsafe fn cuUserObjectRelease(object: CUuserObject, count: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUuserObject, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuUserObjectRelease") });
        _f(object, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuUserObjectRelease(object: CUuserObject, count: ::core::ffi::c_uint) -> CUresult;
        }
        cuUserObjectRelease(object, count)
    }
}
pub unsafe fn cuUserObjectRetain(object: CUuserObject, count: ::core::ffi::c_uint) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(CUuserObject, ::core::ffi::c_uint) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuUserObjectRetain") });
        _f(object, count)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuUserObjectRetain(object: CUuserObject, count: ::core::ffi::c_uint) -> CUresult;
        }
        cuUserObjectRetain(object, count)
    }
}
pub unsafe fn cuWaitExternalSemaphoresAsync(extSemArray: *const CUexternalSemaphore, paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, numExtSems: ::core::ffi::c_uint, stream: CUstream) -> CUresult {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*const CUexternalSemaphore, *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, ::core::ffi::c_uint, CUstream) -> CUresult;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("cuWaitExternalSemaphoresAsync") });
        _f(extSemArray, paramsArray, numExtSems, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn cuWaitExternalSemaphoresAsync(extSemArray: *const CUexternalSemaphore, paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS, numExtSems: ::core::ffi::c_uint, stream: CUstream) -> CUresult;
        }
        cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["cuda", "nvcuda"];
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
        let lib_names = std::vec!["cuda", "nvcuda"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
