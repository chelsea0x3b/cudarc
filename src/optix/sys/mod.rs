//! Raw FFI bindings and dynamic function table loader for the NVIDIA OptiX Ray Tracing Engine
//! (`libnvoptix.so.1` on Linux, `nvoptix.dll` on Windows).
//!
//! Because `libnvoptix` exports a single entry point ([`optixQueryFunctionTable`]) rather than
//! individual C symbols, this module dynamically loads `libnvoptix` via `libloading` and queries
//! the ABI 41 [`OptixFunctionTable`] on first use.

#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;

pub use crate::driver::sys::{CUcontext, CUdeviceptr, CUstream};
use crate::driver::{DeviceRepr, ValidAsZeroBits};

/// OptiX 7.2 Function Table ABI version (38 function pointers, 304 bytes).
/// Supported across NVIDIA display drivers from R455 through R595+.
pub const OPTIX_ABI_VERSION: c_int = 41;

/// Size in bytes of an OptiX Shader Binding Table (SBT) record header.
pub const OPTIX_SBT_RECORD_HEADER_SIZE: usize = 32;
/// Required byte alignment for OptiX Shader Binding Table (SBT) records.
pub const OPTIX_SBT_RECORD_ALIGNMENT: usize = 16;
/// Required byte alignment for OptiX acceleration structure output and temp buffers.
pub const OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT: usize = 128;
/// Required byte alignment for `OptixInstance` buffers.
pub const OPTIX_INSTANCE_BYTE_ALIGNMENT: usize = 16;
/// Required byte alignment for `OptixAabb` buffers.
pub const OPTIX_AABB_BUFFER_BYTE_ALIGNMENT: usize = 8;
/// Required byte alignment for geometry coordinate buffers.
pub const OPTIX_COORD_BUFFER_BYTE_ALIGNMENT: usize = 4;
/// Required byte alignment for transform buffers.
pub const OPTIX_TRANSFORM_BYTE_ALIGNMENT: usize = 64;

/// Default maximum register count (`0` means no explicit register limit).
pub const OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT: c_int = 0;
/// Default maximum number of payload types.
pub const OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT: c_uint = 8;
/// Default maximum number of 32-bit payload values.
pub const OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT: c_uint = 32;

/// Opaque struct representing an OptiX device context target.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixDeviceContext_t {
    _unused: [u8; 0],
}
/// Opaque handle to an OptiX device context.
pub type OptixDeviceContext = *mut OptixDeviceContext_t;

/// Opaque struct representing a compiled OptiX module target.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixModule_t {
    _unused: [u8; 0],
}
/// Opaque handle to a compiled OptiX PTX/OptiX-IR module.
pub type OptixModule = *mut OptixModule_t;

/// Opaque struct representing an OptiX program group target.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixProgramGroup_t {
    _unused: [u8; 0],
}
/// Opaque handle to an OptiX program group (raygen, miss, hitgroup, exception, or callables).
pub type OptixProgramGroup = *mut OptixProgramGroup_t;

/// Opaque struct representing an OptiX pipeline target.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixPipeline_t {
    _unused: [u8; 0],
}
/// Opaque handle to a linked OptiX ray-tracing pipeline.
pub type OptixPipeline = *mut OptixPipeline_t;

/// Opaque struct representing an OptiX denoiser target.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixDenoiser_t {
    _unused: [u8; 0],
}
/// Opaque handle to an OptiX AI denoiser instance.
pub type OptixDenoiser = *mut OptixDenoiser_t;

/// 64-bit hardware traversable handle representing an acceleration structure (GAS/BLAS or IAS/TLAS)
/// or transform node on the device.
pub type OptixTraversableHandle = u64;
/// Visibility mask bitfield tested against `OptixInstance::visibilityMask` during traversal.
pub type OptixVisibilityMask = c_uint;

/// Optional logging callback invoked by OptiX during context, module, pipeline, or acceleration structure operations.
pub type OptixLogCallback = Option<
    unsafe extern "C" fn(
        level: c_uint,
        tag: *const c_char,
        message: *const c_char,
        cbdata: *mut c_void,
    ),
>;

/// Reserved option key for [`optixQueryFunctionTable`].
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct OptixQueryFunctionTableOptions {
    pub placeholder: c_int,
}

/// Result status codes returned by OptiX API functions.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OptixResult {
    OPTIX_SUCCESS = 0,
    OPTIX_ERROR_INVALID_VALUE = 7001,
    OPTIX_ERROR_HOST_OUT_OF_MEMORY = 7002,
    OPTIX_ERROR_INVALID_OPERATION = 7003,
    OPTIX_ERROR_FILE_IO_ERROR = 7004,
    OPTIX_ERROR_INVALID_FILE_FORMAT = 7005,
    OPTIX_ERROR_DISK_CACHE_INVALID_PATH = 7010,
    OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR = 7011,
    OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR = 7012,
    OPTIX_ERROR_DISK_CACHE_INVALID_DATA = 7013,
    OPTIX_ERROR_LAUNCH_FAILURE = 7050,
    OPTIX_ERROR_INVALID_DEVICE_CONTEXT = 7051,
    OPTIX_ERROR_CUDA_NOT_INITIALIZED = 7052,
    OPTIX_ERROR_VALIDATION_FAILURE = 7053,
    OPTIX_ERROR_INVALID_PTX = 7200,
    OPTIX_ERROR_INVALID_LAUNCH_PARAMETER = 7201,
    OPTIX_ERROR_INVALID_PAYLOAD_ACCESS = 7202,
    OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS = 7203,
    OPTIX_ERROR_INVALID_FUNCTION_USE = 7204,
    OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS = 7205,
    OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY = 7250,
    OPTIX_ERROR_PIPELINE_LINK_ERROR = 7251,
    OPTIX_ERROR_ILLEGAL_INSTRUMENTATION_PASS = 7252,
    OPTIX_ERROR_INTERNAL_COMPILER_ERROR = 7299,
    OPTIX_ERROR_DENOISER_MODEL_NOT_SET = 7300,
    OPTIX_ERROR_DENOISER_NOT_INITIALIZED = 7301,
    OPTIX_ERROR_ACCEL_NOT_COMPATIBLE = 7400,
    OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH = 7500,
    OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED = 7501,
    OPTIX_ERROR_PAYLOAD_TYPE_ID_INVALID = 7502,
    OPTIX_ERROR_NOT_SUPPORTED = 7800,
    OPTIX_ERROR_UNSUPPORTED_ABI_VERSION = 7801,
    OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH = 7802,
    OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS = 7803,
    OPTIX_ERROR_LIBRARY_NOT_FOUND = 7804,
    OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND = 7805,
    OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE = 7806,
    OPTIX_ERROR_DEVICE_OUT_OF_MEMORY = 7807,
    OPTIX_ERROR_CUDA_ERROR = 7900,
    OPTIX_ERROR_INTERNAL_ERROR = 7990,
    OPTIX_ERROR_UNKNOWN = 7999,
}

impl OptixResult {
    pub const OPTIX_ERROR_INVALID_INPUT: Self = Self::OPTIX_ERROR_INVALID_PTX;
    pub const OPTIX_ERROR_NOT_COMPATIBLE: Self = Self::OPTIX_ERROR_ACCEL_NOT_COMPATIBLE;
}

/// Context validation mode for [`OptixDeviceContextOptions`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixDeviceContextValidationMode {
    #[default]
    OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF = 0,
    OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL = 0xFFFFFFFF,
}

/// Parameters used to configure an [`OptixDeviceContext`] via [`optixDeviceContextCreate`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixDeviceContextOptions {
    pub logCallbackFunction: OptixLogCallback,
    pub logCallbackData: *mut c_void,
    pub logCallbackLevel: c_int,
    pub validationMode: OptixDeviceContextValidationMode,
}

impl Default for OptixDeviceContextOptions {
    fn default() -> Self {
        Self {
            logCallbackFunction: None,
            logCallbackData: core::ptr::null_mut(),
            logCallbackLevel: 0,
            validationMode: OptixDeviceContextValidationMode::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF,
        }
    }
}

/// Device properties queryable via [`optixDeviceContextGetProperty`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixDeviceProperty {
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH = 0x2001,
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH = 0x2002,
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS = 0x2003,
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS = 0x2004,
    OPTIX_DEVICE_PROPERTY_RTCORE_VERSION = 0x2005,
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID = 0x2006,
    OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK = 0x2007,
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS = 0x2008,
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET = 0x2009,
}

/// Discriminant for [`OptixBuildInput`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixBuildInputType {
    OPTIX_BUILD_INPUT_TYPE_TRIANGLES = 0x2141,
    OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES = 0x2142,
    OPTIX_BUILD_INPUT_TYPE_INSTANCES = 0x2143,
    OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS = 0x2144,
    OPTIX_BUILD_INPUT_TYPE_CURVES = 0x2145,
}

pub use self::OptixBuildInputType::*;

/// Vertex position binary format for [`OptixBuildInputTriangleArray`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixVertexFormat {
    #[default]
    OPTIX_VERTEX_FORMAT_NONE = 0,
    OPTIX_VERTEX_FORMAT_FLOAT3 = 0x2121,
    OPTIX_VERTEX_FORMAT_FLOAT2 = 0x2122,
    OPTIX_VERTEX_FORMAT_HALF3 = 0x2123,
    OPTIX_VERTEX_FORMAT_HALF2 = 0x2124,
    OPTIX_VERTEX_FORMAT_SNORM16_3 = 0x2125,
    OPTIX_VERTEX_FORMAT_SNORM16_2 = 0x2126,
}

pub use self::OptixVertexFormat::*;

/// Pre-transform matrix format for [`OptixBuildInputTriangleArray`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixTransformFormat {
    #[default]
    OPTIX_TRANSFORM_FORMAT_NONE = 0,
    OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 = 0x21E1,
}

pub use self::OptixTransformFormat::*;

/// Triangle index triplet format for [`OptixBuildInputTriangleArray`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixIndicesFormat {
    #[default]
    OPTIX_INDICES_FORMAT_NONE = 0,
    OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 = 0x2102,
    OPTIX_INDICES_FORMAT_UNSIGNED_INT3 = 0x2103,
}

pub use self::OptixIndicesFormat::*;

/// Per-geometry flags controlling any-hit shader invocation and triangle face culling.
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixGeometryFlags {
    #[default]
    OPTIX_GEOMETRY_FLAG_NONE = 0,
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT = 1 << 0,
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1 << 1,
    OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1 << 2,
}

pub use self::OptixGeometryFlags::*;

/// Triangle mesh build input descriptor for bottom-level acceleration structures (GAS / BLAS).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixBuildInputTriangleArray {
    pub vertexBuffers: *const CUdeviceptr,
    pub numVertices: c_uint,
    pub vertexFormat: OptixVertexFormat,
    pub vertexStrideInBytes: c_uint,
    pub indexBuffer: CUdeviceptr,
    pub numIndexTriplets: c_uint,
    pub indexFormat: OptixIndicesFormat,
    pub indexStrideInBytes: c_uint,
    pub preTransform: CUdeviceptr,
    pub flags: *const c_uint,
    pub numSbtRecords: c_uint,
    pub sbtIndexOffsetBuffer: CUdeviceptr,
    pub sbtIndexOffsetSizeInBytes: c_uint,
    pub sbtIndexOffsetStrideInBytes: c_uint,
    pub primitiveIndexOffset: c_uint,
    pub transformFormat: OptixTransformFormat,
}

impl Default for OptixBuildInputTriangleArray {
    fn default() -> Self {
        Self {
            vertexBuffers: core::ptr::null(),
            numVertices: 0,
            vertexFormat: OptixVertexFormat::OPTIX_VERTEX_FORMAT_NONE,
            vertexStrideInBytes: 0,
            indexBuffer: 0,
            numIndexTriplets: 0,
            indexFormat: OptixIndicesFormat::OPTIX_INDICES_FORMAT_NONE,
            indexStrideInBytes: 0,
            preTransform: 0,
            flags: core::ptr::null(),
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
            transformFormat: OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_NONE,
        }
    }
}

/// Custom primitive axis-aligned bounding box (AABB) build input descriptor.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixBuildInputCustomPrimitiveArray {
    pub aabbBuffers: *const CUdeviceptr,
    pub numPrimitives: c_uint,
    pub strideInBytes: c_uint,
    pub flags: *const c_uint,
    pub numSbtRecords: c_uint,
    pub sbtIndexOffsetBuffer: CUdeviceptr,
    pub sbtIndexOffsetSizeInBytes: c_uint,
    pub sbtIndexOffsetStrideInBytes: c_uint,
    pub primitiveIndexOffset: c_uint,
}

impl Default for OptixBuildInputCustomPrimitiveArray {
    fn default() -> Self {
        Self {
            aabbBuffers: core::ptr::null(),
            numPrimitives: 0,
            strideInBytes: 0,
            flags: core::ptr::null(),
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
        }
    }
}

/// Instance array build input descriptor for top-level acceleration structures (IAS / TLAS).
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct OptixBuildInputInstanceArray {
    pub instances: CUdeviceptr,
    pub numInstances: c_uint,
}

/// Union payload inside [`OptixBuildInput`].
#[repr(C)]
#[derive(Copy, Clone)]
pub union OptixBuildInputUnion {
    pub triangleArray: OptixBuildInputTriangleArray,
    pub customPrimitiveArray: OptixBuildInputCustomPrimitiveArray,
    pub instanceArray: OptixBuildInputInstanceArray,
    pub pad: [c_char; 1024],
}

/// Tagged union describing a single geometry or instance input to [`optixAccelComputeMemoryUsage`] and [`optixAccelBuild`].
#[repr(C)]
#[derive(Copy, Clone)]
pub struct OptixBuildInput {
    pub type_: OptixBuildInputType,
    pub input: OptixBuildInputUnion,
}

impl OptixBuildInput {
    /// Creates an [`OptixBuildInput`] for a triangle array.
    pub fn triangles(triangle_array: OptixBuildInputTriangleArray) -> Self {
        let mut input = OptixBuildInputUnion { pad: [0; 1024] };
        input.triangleArray = triangle_array;
        Self {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            input,
        }
    }

    /// Creates an [`OptixBuildInput`] for a custom primitive AABB array.
    pub fn custom_primitives(custom_primitive_array: OptixBuildInputCustomPrimitiveArray) -> Self {
        let mut input = OptixBuildInputUnion { pad: [0; 1024] };
        input.customPrimitiveArray = custom_primitive_array;
        Self {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
            input,
        }
    }

    /// Creates an [`OptixBuildInput`] for an instance array.
    pub fn instances(instance_array: OptixBuildInputInstanceArray) -> Self {
        let mut input = OptixBuildInputUnion { pad: [0; 1024] };
        input.instanceArray = instance_array;
        Self {
            type_: OptixBuildInputType::OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            input,
        }
    }
}

/// 8-byte aligned 3D axis-aligned bounding box (`[minX, minY, minZ, maxX, maxY, maxZ]`) for custom primitives.
#[repr(C, align(8))]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct OptixAabb {
    pub minX: f32,
    pub minY: f32,
    pub minZ: f32,
    pub maxX: f32,
    pub maxY: f32,
    pub maxZ: f32,
}

impl OptixAabb {
    /// Constructs a new [`OptixAabb`] from `[minX, minY, minZ]` and `[maxX, maxY, maxZ]`.
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self {
            minX: min[0],
            minY: min[1],
            minZ: min[2],
            maxX: max[0],
            maxY: max[1],
            maxZ: max[2],
        }
    }
}

unsafe impl DeviceRepr for OptixAabb {}
unsafe impl ValidAsZeroBits for OptixAabb {}

/// Per-instance traversal flags for [`OptixInstance`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixInstanceFlags {
    #[default]
    OPTIX_INSTANCE_FLAG_NONE = 0,
    OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1 << 0,
    OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1 << 1,
    OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT = 1 << 2,
    OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT = 1 << 3,
    OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM = 1 << 6,
}

pub use self::OptixInstanceFlags::*;

/// 16-byte aligned, 80-byte instance descriptor for top-level acceleration structures (IAS / TLAS).
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct OptixInstance {
    pub transform: [f32; 12],
    pub instanceId: c_uint,
    pub sbtOffset: c_uint,
    pub visibilityMask: c_uint,
    pub flags: c_uint,
    pub traversableHandle: OptixTraversableHandle,
    pub pad: [c_uint; 2],
}

impl Default for OptixInstance {
    fn default() -> Self {
        Self {
            transform: [
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
                0.0, 0.0, 1.0, 0.0, // row 2
            ],
            instanceId: 0,
            sbtOffset: 0,
            visibilityMask: 255,
            flags: OptixInstanceFlags::OPTIX_INSTANCE_FLAG_NONE as c_uint,
            traversableHandle: 0,
            pad: [0, 0],
        }
    }
}

impl OptixInstance {
    /// Creates an [`OptixInstance`] with an identity 3x4 affine transform for the given BLAS handle.
    pub fn with_identity(traversable_handle: OptixTraversableHandle, instance_id: u32) -> Self {
        Self {
            instanceId: instance_id,
            traversableHandle: traversable_handle,
            ..Default::default()
        }
    }
}

unsafe impl DeviceRepr for OptixInstance {}
unsafe impl ValidAsZeroBits for OptixInstance {}

/// Acceleration structure build flags for [`OptixAccelBuildOptions`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixBuildFlags {
    #[default]
    OPTIX_BUILD_FLAG_NONE = 0,
    OPTIX_BUILD_FLAG_ALLOW_UPDATE = 1 << 0,
    OPTIX_BUILD_FLAG_ALLOW_COMPACTION = 1 << 1,
    OPTIX_BUILD_FLAG_PREFER_FAST_TRACE = 1 << 2,
    OPTIX_BUILD_FLAG_PREFER_FAST_BUILD = 1 << 3,
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS = 1 << 4,
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS = 1 << 5,
}

pub use self::OptixBuildFlags::*;

/// Acceleration structure operation mode (full build vs. in-place update).
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixBuildOperation {
    #[default]
    OPTIX_BUILD_OPERATION_BUILD = 0x2161,
    OPTIX_BUILD_OPERATION_UPDATE = 0x2162,
}

pub use self::OptixBuildOperation::*;

/// Motion blur flags for [`OptixMotionOptions`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixMotionFlags {
    #[default]
    OPTIX_MOTION_FLAG_NONE = 0,
    OPTIX_MOTION_FLAG_START_VANISH = 1 << 0,
    OPTIX_MOTION_FLAG_END_VANISH = 1 << 1,
}

/// Motion blur configuration for acceleration structures and transforms.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct OptixMotionOptions {
    pub numKeys: u16,
    pub flags: u16,
    pub timeBegin: f32,
    pub timeEnd: f32,
}

/// Build options passed to [`optixAccelComputeMemoryUsage`] and [`optixAccelBuild`].
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct OptixAccelBuildOptions {
    pub buildFlags: c_uint,
    pub operation: OptixBuildOperation,
    pub motionOptions: OptixMotionOptions,
}

impl Default for OptixAccelBuildOptions {
    fn default() -> Self {
        Self {
            buildFlags: OptixBuildFlags::OPTIX_BUILD_FLAG_PREFER_FAST_TRACE as c_uint,
            operation: OptixBuildOperation::OPTIX_BUILD_OPERATION_BUILD,
            motionOptions: OptixMotionOptions::default(),
        }
    }
}

/// Device memory buffer size requirements returned by [`optixAccelComputeMemoryUsage`].
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct OptixAccelBufferSizes {
    pub outputSizeInBytes: usize,
    pub tempSizeInBytes: usize,
    pub tempUpdateSizeInBytes: usize,
}

/// Post-build property type emitted during [`optixAccelBuild`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixAccelPropertyType {
    OPTIX_PROPERTY_TYPE_COMPACTED_SIZE = 0x2181,
    OPTIX_PROPERTY_TYPE_AABBS = 0x2182,
}

pub use self::OptixAccelPropertyType::*;

/// Descriptor for a post-build property emitted to device memory by [`optixAccelBuild`].
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct OptixAccelEmitDesc {
    pub result: CUdeviceptr,
    pub type_: OptixAccelPropertyType,
}

/// Opaque relocation metadata populated by [`optixAccelGetRelocationInfo`].
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct OptixAccelRelocationInfo {
    pub info: [u64; 4],
}

/// Traversable handle type for [`optixConvertPointerToTraversableHandle`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixTraversableType {
    OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM = 0x21C1,
    OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM = 0x21C2,
    OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM = 0x21C3,
}

/// Optimization level for PTX module compilation ([`OptixModuleCompileOptions`]).
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixCompileOptimizationLevel {
    #[default]
    OPTIX_COMPILE_OPTIMIZATION_DEFAULT = 0,
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 = 0x2340,
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 = 0x2341,
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 = 0x2342,
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 = 0x2343,
}

pub use self::OptixCompileOptimizationLevel::*;

/// Debug information level for PTX module compilation and pipeline linking.
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixCompileDebugLevel {
    #[default]
    OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT = 0,
    OPTIX_COMPILE_DEBUG_LEVEL_NONE = 0x2350,
    OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO = 0x2351,
    OPTIX_COMPILE_DEBUG_LEVEL_MODERATE = 0x2352,
    OPTIX_COMPILE_DEBUG_LEVEL_FULL = 0x2353,
}

impl OptixCompileDebugLevel {
    pub const OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL: Self = Self::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
}

pub use self::OptixCompileDebugLevel::*;

/// Specialization constant bound value entry for [`OptixModuleCompileOptions`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixModuleCompileBoundValueEntry {
    pub pipelineParamOffsetInBytes: usize,
    pub sizeInBytes: usize,
    pub boundValuePtr: *const c_void,
    pub annotation: *const c_char,
}

/// Compilation options passed to [`optixModuleCreateFromPTX`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixModuleCompileOptions {
    pub maxRegisterCount: c_int,
    pub optLevel: OptixCompileOptimizationLevel,
    pub debugLevel: OptixCompileDebugLevel,
    pub boundValues: *const OptixModuleCompileBoundValueEntry,
    pub numBoundValues: c_uint,
}

impl Default for OptixModuleCompileOptions {
    fn default() -> Self {
        Self {
            maxRegisterCount: OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel: OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel: OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
            boundValues: core::ptr::null(),
            numBoundValues: 0,
        }
    }
}

/// Traversable graph topology flags for [`OptixPipelineCompileOptions`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixTraversableGraphFlags {
    #[default]
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0,
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1 << 0,
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1 << 1,
}

pub use self::OptixTraversableGraphFlags::*;

/// Hardware/software exception flags for [`OptixPipelineCompileOptions`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixExceptionFlags {
    #[default]
    OPTIX_EXCEPTION_FLAG_NONE = 0,
    OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW = 1 << 0,
    OPTIX_EXCEPTION_FLAG_TRACE_DEPTH = 1 << 1,
    OPTIX_EXCEPTION_FLAG_USER = 1 << 2,
    OPTIX_EXCEPTION_FLAG_DEBUG = 1 << 3,
}

pub use self::OptixExceptionFlags::*;

/// Built-in primitive geometry type.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixPrimitiveType {
    OPTIX_PRIMITIVE_TYPE_CUSTOM = 0x2500,
    OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE = 0x2501,
    OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE = 0x2502,
    OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR = 0x2503,
    OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM = 0x2504,
    OPTIX_PRIMITIVE_TYPE_TRIANGLE = 0x2531,
}

pub use self::OptixPrimitiveType::*;

/// Bitflags corresponding to [`OptixPrimitiveType`] for [`OptixPipelineCompileOptions::usesPrimitiveTypeFlags`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixPrimitiveTypeFlags {
    OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM = 1 << 0,
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE = 1 << 1,
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE = 1 << 2,
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR = 1 << 3,
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM = 1 << 4,
    OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE = 1 << 31,
}

pub use self::OptixPrimitiveTypeFlags::*;

/// Per-ray flags passed to `optixTrace`.
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixRayFlags {
    #[default]
    OPTIX_RAY_FLAG_NONE = 0,
    OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1 << 0,
    OPTIX_RAY_FLAG_ENFORCE_ANYHIT = 1 << 1,
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1 << 2,
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT = 1 << 3,
    OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1 << 4,
    OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1 << 5,
    OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT = 1 << 6,
    OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1 << 7,
}

pub use self::OptixRayFlags::*;

/// Pipeline-wide compile options shared across [`optixModuleCreateFromPTX`] and [`optixPipelineCreate`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixPipelineCompileOptions {
    pub usesMotionBlur: c_int,
    pub traversableGraphFlags: c_uint,
    pub numPayloadValues: c_int,
    pub numAttributeValues: c_int,
    pub exceptionFlags: c_uint,
    pub pipelineLaunchParamsVariableName: *const c_char,
    pub usesPrimitiveTypeFlags: c_uint,
}

impl Default for OptixPipelineCompileOptions {
    fn default() -> Self {
        Self {
            usesMotionBlur: 0,
            traversableGraphFlags: OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY
                as c_uint,
            numPayloadValues: 2,
            numAttributeValues: 2,
            exceptionFlags: OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE as c_uint,
            pipelineLaunchParamsVariableName: c"params".as_ptr(),
            usesPrimitiveTypeFlags: 0,
        }
    }
}

/// Pipeline linking options passed to [`optixPipelineCreate`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixPipelineLinkOptions {
    pub maxTraceDepth: c_uint,
    pub debugLevel: OptixCompileDebugLevel,
}

impl Default for OptixPipelineLinkOptions {
    fn default() -> Self {
        Self {
            maxTraceDepth: 1,
            debugLevel: OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT,
        }
    }
}

/// Options for retrieving a built-in intersection module via [`optixBuiltinISModuleGet`].
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixBuiltinISOptions {
    pub builtinISModuleType: OptixPrimitiveType,
    pub usesMotionBlur: c_int,
}

/// Kind of an [`OptixProgramGroup`].
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixProgramGroupKind {
    OPTIX_PROGRAM_GROUP_KIND_RAYGEN = 0x2421,
    OPTIX_PROGRAM_GROUP_KIND_MISS = 0x2422,
    OPTIX_PROGRAM_GROUP_KIND_EXCEPTION = 0x2423,
    OPTIX_PROGRAM_GROUP_KIND_HITGROUP = 0x2424,
    OPTIX_PROGRAM_GROUP_KIND_CALLABLES = 0x2425,
}

pub use self::OptixProgramGroupKind::*;

/// Flags for [`OptixProgramGroupDesc`].
#[repr(u32)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub enum OptixProgramGroupFlags {
    #[default]
    OPTIX_PROGRAM_GROUP_FLAGS_NONE = 0,
}

pub use self::OptixProgramGroupFlags::*;

/// Single-shader program group entry (`raygen`, `miss`, or `exception`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixProgramGroupSingleModule {
    pub module: OptixModule,
    pub entryFunctionName: *const c_char,
}

impl Default for OptixProgramGroupSingleModule {
    fn default() -> Self {
        Self {
            module: core::ptr::null_mut(),
            entryFunctionName: core::ptr::null(),
        }
    }
}

/// Hitgroup shader program group entry combining closest-hit (`CH`), any-hit (`AH`), and intersection (`IS`) modules.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixProgramGroupHitgroup {
    pub moduleCH: OptixModule,
    pub entryFunctionNameCH: *const c_char,
    pub moduleAH: OptixModule,
    pub entryFunctionNameAH: *const c_char,
    pub moduleIS: OptixModule,
    pub entryFunctionNameIS: *const c_char,
}

impl Default for OptixProgramGroupHitgroup {
    fn default() -> Self {
        Self {
            moduleCH: core::ptr::null_mut(),
            entryFunctionNameCH: core::ptr::null(),
            moduleAH: core::ptr::null_mut(),
            entryFunctionNameAH: core::ptr::null(),
            moduleIS: core::ptr::null_mut(),
            entryFunctionNameIS: core::ptr::null(),
        }
    }
}

/// Direct (`DC`) and continuation (`CC`) callable shader program group entry.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptixProgramGroupCallables {
    pub moduleDC: OptixModule,
    pub entryFunctionNameDC: *const c_char,
    pub moduleCC: OptixModule,
    pub entryFunctionNameCC: *const c_char,
}

impl Default for OptixProgramGroupCallables {
    fn default() -> Self {
        Self {
            moduleDC: core::ptr::null_mut(),
            entryFunctionNameDC: core::ptr::null(),
            moduleCC: core::ptr::null_mut(),
            entryFunctionNameCC: core::ptr::null(),
        }
    }
}

/// Union payload inside [`OptixProgramGroupDesc`].
#[repr(C)]
#[derive(Copy, Clone)]
pub union OptixProgramGroupDescUnion {
    pub raygen: OptixProgramGroupSingleModule,
    pub miss: OptixProgramGroupSingleModule,
    pub exception: OptixProgramGroupSingleModule,
    pub callables: OptixProgramGroupCallables,
    pub hitgroup: OptixProgramGroupHitgroup,
}

/// Program group creation descriptor passed to [`optixProgramGroupCreate`].
#[repr(C)]
#[derive(Copy, Clone)]
pub struct OptixProgramGroupDesc {
    pub kind: OptixProgramGroupKind,
    pub flags: c_uint,
    pub desc: OptixProgramGroupDescUnion,
}

impl OptixProgramGroupDesc {
    /// Creates an [`OptixProgramGroupDesc`] for a raygen program.
    pub fn raygen(module: OptixModule, entry_function_name: *const c_char) -> Self {
        let mut desc: OptixProgramGroupDescUnion = unsafe { core::mem::zeroed() };
        desc.raygen = OptixProgramGroupSingleModule {
            module,
            entryFunctionName: entry_function_name,
        };
        Self {
            kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            flags: OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE as c_uint,
            desc,
        }
    }

    /// Creates an [`OptixProgramGroupDesc`] for a miss program.
    pub fn miss(module: OptixModule, entry_function_name: *const c_char) -> Self {
        let mut desc: OptixProgramGroupDescUnion = unsafe { core::mem::zeroed() };
        desc.miss = OptixProgramGroupSingleModule {
            module,
            entryFunctionName: entry_function_name,
        };
        Self {
            kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
            flags: OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE as c_uint,
            desc,
        }
    }

    /// Creates an [`OptixProgramGroupDesc`] for a hitgroup program.
    pub fn hitgroup(hitgroup: OptixProgramGroupHitgroup) -> Self {
        let mut desc: OptixProgramGroupDescUnion = unsafe { core::mem::zeroed() };
        desc.hitgroup = hitgroup;
        Self {
            kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            flags: OptixProgramGroupFlags::OPTIX_PROGRAM_GROUP_FLAGS_NONE as c_uint,
            desc,
        }
    }
}

/// Reserved option struct for [`optixProgramGroupCreate`].
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct OptixProgramGroupOptions {
    pub placeholder: c_int,
}

/// Per-program-group stack size requirements returned by [`optixProgramGroupGetStackSize`].
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct OptixStackSizes {
    pub cssRG: c_uint,
    pub cssMS: c_uint,
    pub cssCH: c_uint,
    pub cssAH: c_uint,
    pub cssIS: c_uint,
    pub cssCC: c_uint,
    pub dssDC: c_uint,
}

/// Shader Binding Table (SBT) device pointer and stride layout passed to [`optixLaunch`].
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct OptixShaderBindingTable {
    pub raygenRecord: CUdeviceptr,
    pub exceptionRecord: CUdeviceptr,
    pub missRecordBase: CUdeviceptr,
    pub missRecordStrideInBytes: c_uint,
    pub missRecordCount: c_uint,
    pub hitgroupRecordBase: CUdeviceptr,
    pub hitgroupRecordStrideInBytes: c_uint,
    pub hitgroupRecordCount: c_uint,
    pub callablesRecordBase: CUdeviceptr,
    pub callablesRecordStrideInBytes: c_uint,
    pub callablesRecordCount: c_uint,
}

/// OptiX ABI 41 function dispatch table (38 function pointers = 304 bytes on 64-bit).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct OptixFunctionTable {
    // Error handling
    pub optixGetErrorName: unsafe extern "C" fn(result: OptixResult) -> *const c_char,
    pub optixGetErrorString: unsafe extern "C" fn(result: OptixResult) -> *const c_char,

    // Device context
    pub optixDeviceContextCreate: unsafe extern "C" fn(
        fromContext: CUcontext,
        options: *const OptixDeviceContextOptions,
        context: *mut OptixDeviceContext,
    ) -> OptixResult,
    pub optixDeviceContextDestroy:
        unsafe extern "C" fn(context: OptixDeviceContext) -> OptixResult,
    pub optixDeviceContextGetProperty: unsafe extern "C" fn(
        context: OptixDeviceContext,
        property: OptixDeviceProperty,
        value: *mut c_void,
        sizeInBytes: usize,
    ) -> OptixResult,
    pub optixDeviceContextSetLogCallback: unsafe extern "C" fn(
        context: OptixDeviceContext,
        callbackFunction: OptixLogCallback,
        callbackData: *mut c_void,
        callbackLevel: c_uint,
    ) -> OptixResult,
    pub optixDeviceContextSetCacheEnabled:
        unsafe extern "C" fn(context: OptixDeviceContext, enabled: c_int) -> OptixResult,
    pub optixDeviceContextSetCacheLocation:
        unsafe extern "C" fn(context: OptixDeviceContext, location: *const c_char) -> OptixResult,
    pub optixDeviceContextSetCacheDatabaseSizes: unsafe extern "C" fn(
        context: OptixDeviceContext,
        lowWaterMark: usize,
        highWaterMark: usize,
    ) -> OptixResult,
    pub optixDeviceContextGetCacheEnabled:
        unsafe extern "C" fn(context: OptixDeviceContext, enabled: *mut c_int) -> OptixResult,
    pub optixDeviceContextGetCacheLocation: unsafe extern "C" fn(
        context: OptixDeviceContext,
        location: *mut c_char,
        locationSize: usize,
    ) -> OptixResult,
    pub optixDeviceContextGetCacheDatabaseSizes: unsafe extern "C" fn(
        context: OptixDeviceContext,
        lowWaterMark: *mut usize,
        highWaterMark: *mut usize,
    ) -> OptixResult,

    // Modules
    pub optixModuleCreateFromPTX: unsafe extern "C" fn(
        context: OptixDeviceContext,
        moduleCompileOptions: *const OptixModuleCompileOptions,
        pipelineCompileOptions: *const OptixPipelineCompileOptions,
        PTX: *const c_char,
        PTXsize: usize,
        logString: *mut c_char,
        logStringSize: *mut usize,
        module: *mut OptixModule,
    ) -> OptixResult,
    pub optixModuleDestroy: unsafe extern "C" fn(module: OptixModule) -> OptixResult,
    pub optixBuiltinISModuleGet: unsafe extern "C" fn(
        context: OptixDeviceContext,
        moduleCompileOptions: *const OptixModuleCompileOptions,
        pipelineCompileOptions: *const OptixPipelineCompileOptions,
        builtinISOptions: *const OptixBuiltinISOptions,
        builtinModule: *mut OptixModule,
    ) -> OptixResult,

    // Program groups
    pub optixProgramGroupCreate: unsafe extern "C" fn(
        context: OptixDeviceContext,
        programDescriptions: *const OptixProgramGroupDesc,
        numProgramGroups: c_uint,
        options: *const OptixProgramGroupOptions,
        logString: *mut c_char,
        logStringSize: *mut usize,
        programGroups: *mut OptixProgramGroup,
    ) -> OptixResult,
    pub optixProgramGroupDestroy:
        unsafe extern "C" fn(group: OptixProgramGroup) -> OptixResult,
    pub optixProgramGroupGetStackSize: unsafe extern "C" fn(
        group: OptixProgramGroup,
        stackSizes: *mut OptixStackSizes,
    ) -> OptixResult,

    // Pipelines
    pub optixPipelineCreate: unsafe extern "C" fn(
        context: OptixDeviceContext,
        pipelineCompileOptions: *const OptixPipelineCompileOptions,
        pipelineLinkOptions: *const OptixPipelineLinkOptions,
        programGroups: *const OptixProgramGroup,
        numProgramGroups: c_uint,
        logString: *mut c_char,
        logStringSize: *mut usize,
        pipeline: *mut OptixPipeline,
    ) -> OptixResult,
    pub optixPipelineDestroy: unsafe extern "C" fn(pipeline: OptixPipeline) -> OptixResult,
    pub optixPipelineSetStackSize: unsafe extern "C" fn(
        pipeline: OptixPipeline,
        directCallableStackSizeFromTraversal: c_uint,
        directCallableStackSizeFromState: c_uint,
        continuationStackSize: c_uint,
        maxTraversableGraphDepth: c_uint,
    ) -> OptixResult,

    // Acceleration structures
    pub optixAccelComputeMemoryUsage: unsafe extern "C" fn(
        context: OptixDeviceContext,
        accelOptions: *const OptixAccelBuildOptions,
        buildInputs: *const OptixBuildInput,
        numBuildInputs: c_uint,
        bufferSizes: *mut OptixAccelBufferSizes,
    ) -> OptixResult,
    pub optixAccelBuild: unsafe extern "C" fn(
        context: OptixDeviceContext,
        stream: CUstream,
        accelOptions: *const OptixAccelBuildOptions,
        buildInputs: *const OptixBuildInput,
        numBuildInputs: c_uint,
        tempBuffer: CUdeviceptr,
        tempBufferSizeInBytes: usize,
        outputBuffer: CUdeviceptr,
        outputBufferSizeInBytes: usize,
        outputHandle: *mut OptixTraversableHandle,
        emittedProperties: *const OptixAccelEmitDesc,
        numEmittedProperties: c_uint,
    ) -> OptixResult,
    pub optixAccelGetRelocationInfo: unsafe extern "C" fn(
        context: OptixDeviceContext,
        handle: OptixTraversableHandle,
        info: *mut OptixAccelRelocationInfo,
    ) -> OptixResult,
    pub optixAccelCheckRelocationCompatibility: unsafe extern "C" fn(
        context: OptixDeviceContext,
        info: *const OptixAccelRelocationInfo,
        compatible: *mut c_int,
    ) -> OptixResult,
    pub optixAccelRelocate: unsafe extern "C" fn(
        context: OptixDeviceContext,
        stream: CUstream,
        info: *const OptixAccelRelocationInfo,
        instanceTraversableHandles: CUdeviceptr,
        numInstanceTraversableHandles: usize,
        targetAccel: CUdeviceptr,
        targetAccelSizeInBytes: usize,
        targetHandle: *mut OptixTraversableHandle,
    ) -> OptixResult,
    pub optixAccelCompact: unsafe extern "C" fn(
        context: OptixDeviceContext,
        stream: CUstream,
        inputHandle: OptixTraversableHandle,
        outputBuffer: CUdeviceptr,
        outputBufferSizeInBytes: usize,
        outputHandle: *mut OptixTraversableHandle,
    ) -> OptixResult,
    pub optixConvertPointerToTraversableHandle: unsafe extern "C" fn(
        onDevice: OptixDeviceContext,
        pointer: CUdeviceptr,
        traversableType: OptixTraversableType,
        traversableHandle: *mut OptixTraversableHandle,
    ) -> OptixResult,

    // Launch & SBT
    pub optixSbtRecordPackHeader: unsafe extern "C" fn(
        programGroup: OptixProgramGroup,
        sbtRecordHeaderHostPointer: *mut c_void,
    ) -> OptixResult,
    pub optixLaunch: unsafe extern "C" fn(
        pipeline: OptixPipeline,
        stream: CUstream,
        pipelineParams: CUdeviceptr,
        pipelineParamsSize: usize,
        sbt: *const OptixShaderBindingTable,
        width: c_uint,
        height: c_uint,
        depth: c_uint,
    ) -> OptixResult,

    // Denoiser slots (part of ABI 41 table layout)
    pub optixDenoiserCreate: *const c_void,
    pub optixDenoiserDestroy: *const c_void,
    pub optixDenoiserComputeMemoryResources: *const c_void,
    pub optixDenoiserSetup: *const c_void,
    pub optixDenoiserInvoke: *const c_void,
    pub optixDenoiserSetModel: *const c_void,
    pub optixDenoiserComputeIntensity: *const c_void,
    pub optixDenoiserComputeAverageColor: *const c_void,
}

unsafe impl Send for OptixFunctionTable {}
unsafe impl Sync for OptixFunctionTable {}

fn optix_lib_candidates() -> std::vec::Vec<std::string::String> {
    #[cfg(feature = "dynamic-loading")]
    let mut choices = crate::get_lib_name_candidates("nvoptix");
    #[cfg(not(feature = "dynamic-loading"))]
    let mut choices = std::vec::Vec::new();

    #[cfg(unix)]
    let extras = ["libnvoptix.so.1", "libnvoptix.so"];
    #[cfg(windows)]
    let extras = ["nvoptix.dll"];
    #[cfg(not(any(unix, windows)))]
    let extras: [&str; 0] = [];

    for extra in extras {
        let s = std::string::String::from(extra);
        if !choices.contains(&s) {
            choices.push(s);
        }
    }
    choices
}

/// Checks whether the NVIDIA OptiX driver library (`libnvoptix.so.1` / `nvoptix.dll`)
/// is present on the host system without panicking.
pub unsafe fn is_optixlib_present() -> bool {
    for choice in optix_lib_candidates() {
        if ::libloading::Library::new(choice).is_ok() {
            return true;
        }
    }
    false
}

/// Dynamically loads and caches the `libnvoptix` shared library.
pub unsafe fn optixlib() -> &'static ::libloading::Library {
    static LIB: OnceLock<::libloading::Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let choices = optix_lib_candidates();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        #[cfg(feature = "dynamic-loading")]
        crate::panic_no_lib_found("nvoptix", &choices);
        #[cfg(not(feature = "dynamic-loading"))]
        panic!(
            "Unable to dynamically load the \"nvoptix\" shared library - searched for library names: {choices:?}. Ensure that the NVIDIA display driver is installed."
        );
    })
}

/// Queries the OptiX driver function table for a specific ABI version.
pub unsafe fn optixQueryFunctionTable(
    abiId: c_int,
    numOptions: c_uint,
    optionKeys: *mut OptixQueryFunctionTableOptions,
    optionValues: *mut *const c_void,
    functionTable: *mut c_void,
    sizeOfTable: usize,
) -> OptixResult {
    type QueryFn = unsafe extern "C" fn(
        c_int,
        c_uint,
        *mut OptixQueryFunctionTableOptions,
        *mut *const c_void,
        *mut c_void,
        usize,
    ) -> OptixResult;
    static SYM: OnceLock<QueryFn> = OnceLock::new();
    let f = SYM.get_or_init(|| {
        *optixlib()
            .get::<QueryFn>(b"optixQueryFunctionTable\0")
            .unwrap_or_else(|e| panic!("Missing symbol optixQueryFunctionTable: {e}"))
    });
    f(
        abiId,
        numOptions,
        optionKeys,
        optionValues,
        functionTable,
        sizeOfTable,
    )
}

/// Queries and caches the [`OptixFunctionTable`] (ABI 41, 38 function pointers / 304 bytes).
pub unsafe fn optix_function_table() -> Result<&'static OptixFunctionTable, OptixResult> {
    static TABLE: OnceLock<Result<OptixFunctionTable, OptixResult>> = OnceLock::new();
    let res = TABLE.get_or_init(|| {
        if !is_optixlib_present() {
            return Err(OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND);
        }
        let mut table = core::mem::MaybeUninit::<OptixFunctionTable>::uninit();
        let status = optixQueryFunctionTable(
            OPTIX_ABI_VERSION,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            table.as_mut_ptr().cast::<c_void>(),
            core::mem::size_of::<OptixFunctionTable>(),
        );
        if status == OptixResult::OPTIX_SUCCESS {
            Ok(table.assume_init())
        } else {
            Err(status)
        }
    });
    match res {
        Ok(t) => Ok(t),
        Err(e) => Err(*e),
    }
}

/// Returns the symbolic string name of an [`OptixResult`] status code.
pub unsafe fn optixGetErrorName(result: OptixResult) -> *const c_char {
    match optix_function_table() {
        Ok(t) => (t.optixGetErrorName)(result),
        Err(_) => c"OPTIX_ERROR_LIBRARY_NOT_FOUND".as_ptr(),
    }
}

/// Returns a human-readable description string for an [`OptixResult`] status code.
pub unsafe fn optixGetErrorString(result: OptixResult) -> *const c_char {
    match optix_function_table() {
        Ok(t) => (t.optixGetErrorString)(result),
        Err(_) => c"OptiX library not found or function table query failed".as_ptr(),
    }
}

/// Creates a new [`OptixDeviceContext`] associated with the specified CUDA context `fromContext`.
pub unsafe fn optixDeviceContextCreate(
    fromContext: CUcontext,
    options: *const OptixDeviceContextOptions,
    context: *mut OptixDeviceContext,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextCreate)(fromContext, options, context),
        Err(e) => e,
    }
}

/// Destroys an [`OptixDeviceContext`] and cleans up all associated OptiX state.
pub unsafe fn optixDeviceContextDestroy(context: OptixDeviceContext) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextDestroy)(context),
        Err(e) => e,
    }
}

/// Queries a hardware or driver property of an [`OptixDeviceContext`].
pub unsafe fn optixDeviceContextGetProperty(
    context: OptixDeviceContext,
    property: OptixDeviceProperty,
    value: *mut c_void,
    sizeInBytes: usize,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextGetProperty)(context, property, value, sizeInBytes),
        Err(e) => e,
    }
}

/// Sets or updates the logging callback function for an [`OptixDeviceContext`].
pub unsafe fn optixDeviceContextSetLogCallback(
    context: OptixDeviceContext,
    callbackFunction: OptixLogCallback,
    callbackData: *mut c_void,
    callbackLevel: c_uint,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => {
            (t.optixDeviceContextSetLogCallback)(context, callbackFunction, callbackData, callbackLevel)
        }
        Err(e) => e,
    }
}

/// Enables or disables the on-disk compilation cache for an [`OptixDeviceContext`].
pub unsafe fn optixDeviceContextSetCacheEnabled(
    context: OptixDeviceContext,
    enabled: c_int,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextSetCacheEnabled)(context, enabled),
        Err(e) => e,
    }
}

/// Sets the directory path of the on-disk compilation cache for an [`OptixDeviceContext`].
pub unsafe fn optixDeviceContextSetCacheLocation(
    context: OptixDeviceContext,
    location: *const c_char,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextSetCacheLocation)(context, location),
        Err(e) => e,
    }
}

/// Sets the low and high water-mark database sizes (in bytes) for the on-disk compilation cache.
pub unsafe fn optixDeviceContextSetCacheDatabaseSizes(
    context: OptixDeviceContext,
    lowWaterMark: usize,
    highWaterMark: usize,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextSetCacheDatabaseSizes)(context, lowWaterMark, highWaterMark),
        Err(e) => e,
    }
}

/// Queries whether the on-disk compilation cache is enabled (`1`) or disabled (`0`).
pub unsafe fn optixDeviceContextGetCacheEnabled(
    context: OptixDeviceContext,
    enabled: *mut c_int,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextGetCacheEnabled)(context, enabled),
        Err(e) => e,
    }
}

/// Queries the directory path of the on-disk compilation cache.
pub unsafe fn optixDeviceContextGetCacheLocation(
    context: OptixDeviceContext,
    location: *mut c_char,
    locationSize: usize,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextGetCacheLocation)(context, location, locationSize),
        Err(e) => e,
    }
}

/// Queries the low and high water-mark database sizes (in bytes) of the on-disk compilation cache.
pub unsafe fn optixDeviceContextGetCacheDatabaseSizes(
    context: OptixDeviceContext,
    lowWaterMark: *mut usize,
    highWaterMark: *mut usize,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixDeviceContextGetCacheDatabaseSizes)(context, lowWaterMark, highWaterMark),
        Err(e) => e,
    }
}

/// Compiles an [`OptixModule`] from PTX source code.
pub unsafe fn optixModuleCreateFromPTX(
    context: OptixDeviceContext,
    moduleCompileOptions: *const OptixModuleCompileOptions,
    pipelineCompileOptions: *const OptixPipelineCompileOptions,
    PTX: *const c_char,
    PTXsize: usize,
    logString: *mut c_char,
    logStringSize: *mut usize,
    module: *mut OptixModule,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixModuleCreateFromPTX)(
            context,
            moduleCompileOptions,
            pipelineCompileOptions,
            PTX,
            PTXsize,
            logString,
            logStringSize,
            module,
        ),
        Err(e) => e,
    }
}

/// Destroys a compiled [`OptixModule`].
pub unsafe fn optixModuleDestroy(module: OptixModule) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixModuleDestroy)(module),
        Err(e) => e,
    }
}

/// Retrieves a built-in intersection [`OptixModule`] (such as triangles or curve primitives).
pub unsafe fn optixBuiltinISModuleGet(
    context: OptixDeviceContext,
    moduleCompileOptions: *const OptixModuleCompileOptions,
    pipelineCompileOptions: *const OptixPipelineCompileOptions,
    builtinISOptions: *const OptixBuiltinISOptions,
    builtinModule: *mut OptixModule,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixBuiltinISModuleGet)(
            context,
            moduleCompileOptions,
            pipelineCompileOptions,
            builtinISOptions,
            builtinModule,
        ),
        Err(e) => e,
    }
}

/// Creates one or more [`OptixProgramGroup`] objects from an array of [`OptixProgramGroupDesc`] descriptors.
pub unsafe fn optixProgramGroupCreate(
    context: OptixDeviceContext,
    programDescriptions: *const OptixProgramGroupDesc,
    numProgramGroups: c_uint,
    options: *const OptixProgramGroupOptions,
    logString: *mut c_char,
    logStringSize: *mut usize,
    programGroups: *mut OptixProgramGroup,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixProgramGroupCreate)(
            context,
            programDescriptions,
            numProgramGroups,
            options,
            logString,
            logStringSize,
            programGroups,
        ),
        Err(e) => e,
    }
}

/// Destroys an [`OptixProgramGroup`].
pub unsafe fn optixProgramGroupDestroy(group: OptixProgramGroup) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixProgramGroupDestroy)(group),
        Err(e) => e,
    }
}

/// Queries the continuation and direct-callable stack sizes required by an [`OptixProgramGroup`].
pub unsafe fn optixProgramGroupGetStackSize(
    group: OptixProgramGroup,
    stackSizes: *mut OptixStackSizes,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixProgramGroupGetStackSize)(group, stackSizes),
        Err(e) => e,
    }
}

/// Links one or more [`OptixProgramGroup`]s into an executable [`OptixPipeline`].
pub unsafe fn optixPipelineCreate(
    context: OptixDeviceContext,
    pipelineCompileOptions: *const OptixPipelineCompileOptions,
    pipelineLinkOptions: *const OptixPipelineLinkOptions,
    programGroups: *const OptixProgramGroup,
    numProgramGroups: c_uint,
    logString: *mut c_char,
    logStringSize: *mut usize,
    pipeline: *mut OptixPipeline,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixPipelineCreate)(
            context,
            pipelineCompileOptions,
            pipelineLinkOptions,
            programGroups,
            numProgramGroups,
            logString,
            logStringSize,
            pipeline,
        ),
        Err(e) => e,
    }
}

/// Destroys an [`OptixPipeline`].
pub unsafe fn optixPipelineDestroy(pipeline: OptixPipeline) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixPipelineDestroy)(pipeline),
        Err(e) => e,
    }
}

/// Configures direct-callable and continuation stack sizes for an [`OptixPipeline`].
pub unsafe fn optixPipelineSetStackSize(
    pipeline: OptixPipeline,
    directCallableStackSizeFromTraversal: c_uint,
    directCallableStackSizeFromState: c_uint,
    continuationStackSize: c_uint,
    maxTraversableGraphDepth: c_uint,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixPipelineSetStackSize)(
            pipeline,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize,
            maxTraversableGraphDepth,
        ),
        Err(e) => e,
    }
}

/// Computes the device memory buffer sizes (`outputSizeInBytes`, `tempSizeInBytes`, `tempUpdateSizeInBytes`)
/// required to build or update an acceleration structure.
pub unsafe fn optixAccelComputeMemoryUsage(
    context: OptixDeviceContext,
    accelOptions: *const OptixAccelBuildOptions,
    buildInputs: *const OptixBuildInput,
    numBuildInputs: c_uint,
    bufferSizes: *mut OptixAccelBufferSizes,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixAccelComputeMemoryUsage)(
            context,
            accelOptions,
            buildInputs,
            numBuildInputs,
            bufferSizes,
        ),
        Err(e) => e,
    }
}

/// Builds or updates a bottom-level (GAS/BLAS) or top-level (IAS/TLAS) acceleration structure on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn optixAccelBuild(
    context: OptixDeviceContext,
    stream: CUstream,
    accelOptions: *const OptixAccelBuildOptions,
    buildInputs: *const OptixBuildInput,
    numBuildInputs: c_uint,
    tempBuffer: CUdeviceptr,
    tempBufferSizeInBytes: usize,
    outputBuffer: CUdeviceptr,
    outputBufferSizeInBytes: usize,
    outputHandle: *mut OptixTraversableHandle,
    emittedProperties: *const OptixAccelEmitDesc,
    numEmittedProperties: c_uint,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixAccelBuild)(
            context,
            stream,
            accelOptions,
            buildInputs,
            numBuildInputs,
            tempBuffer,
            tempBufferSizeInBytes,
            outputBuffer,
            outputBufferSizeInBytes,
            outputHandle,
            emittedProperties,
            numEmittedProperties,
        ),
        Err(e) => e,
    }
}

/// Retrieves relocation metadata for an acceleration structure so it can be relocated to another buffer or device.
pub unsafe fn optixAccelGetRelocationInfo(
    context: OptixDeviceContext,
    handle: OptixTraversableHandle,
    info: *mut OptixAccelRelocationInfo,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixAccelGetRelocationInfo)(context, handle, info),
        Err(e) => e,
    }
}

/// Checks whether an acceleration structure described by `info` is compatible with `context`.
pub unsafe fn optixAccelCheckRelocationCompatibility(
    context: OptixDeviceContext,
    info: *const OptixAccelRelocationInfo,
    compatible: *mut c_int,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixAccelCheckRelocationCompatibility)(context, info, compatible),
        Err(e) => e,
    }
}

/// Relocates an acceleration structure copied to `targetAccel` and returns its new [`OptixTraversableHandle`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn optixAccelRelocate(
    context: OptixDeviceContext,
    stream: CUstream,
    info: *const OptixAccelRelocationInfo,
    instanceTraversableHandles: CUdeviceptr,
    numInstanceTraversableHandles: usize,
    targetAccel: CUdeviceptr,
    targetAccelSizeInBytes: usize,
    targetHandle: *mut OptixTraversableHandle,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixAccelRelocate)(
            context,
            stream,
            info,
            instanceTraversableHandles,
            numInstanceTraversableHandles,
            targetAccel,
            targetAccelSizeInBytes,
            targetHandle,
        ),
        Err(e) => e,
    }
}

/// Compacts a previously built acceleration structure (`OPTIX_BUILD_FLAG_ALLOW_COMPACTION`) into `outputBuffer`.
pub unsafe fn optixAccelCompact(
    context: OptixDeviceContext,
    stream: CUstream,
    inputHandle: OptixTraversableHandle,
    outputBuffer: CUdeviceptr,
    outputBufferSizeInBytes: usize,
    outputHandle: *mut OptixTraversableHandle,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixAccelCompact)(
            context,
            stream,
            inputHandle,
            outputBuffer,
            outputBufferSizeInBytes,
            outputHandle,
        ),
        Err(e) => e,
    }
}

/// Converts a device pointer to a transform descriptor into an [`OptixTraversableHandle`].
pub unsafe fn optixConvertPointerToTraversableHandle(
    onDevice: OptixDeviceContext,
    pointer: CUdeviceptr,
    traversableType: OptixTraversableType,
    traversableHandle: *mut OptixTraversableHandle,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixConvertPointerToTraversableHandle)(
            onDevice,
            pointer,
            traversableType,
            traversableHandle,
        ),
        Err(e) => e,
    }
}

/// Writes the 32-byte opaque SBT record header for `programGroup` into `sbtRecordHeaderHostPointer`.
pub unsafe fn optixSbtRecordPackHeader(
    programGroup: OptixProgramGroup,
    sbtRecordHeaderHostPointer: *mut c_void,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixSbtRecordPackHeader)(programGroup, sbtRecordHeaderHostPointer),
        Err(e) => e,
    }
}

/// Launches an [`OptixPipeline`] ray-generation grid of `(width, height, depth)` rays asynchronously on `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn optixLaunch(
    pipeline: OptixPipeline,
    stream: CUstream,
    pipelineParams: CUdeviceptr,
    pipelineParamsSize: usize,
    sbt: *const OptixShaderBindingTable,
    width: c_uint,
    height: c_uint,
    depth: c_uint,
) -> OptixResult {
    match optix_function_table() {
        Ok(t) => (t.optixLaunch)(
            pipeline,
            stream,
            pipelineParams,
            pipelineParamsSize,
            sbt,
            width,
            height,
            depth,
        ),
        Err(e) => e,
    }
}
