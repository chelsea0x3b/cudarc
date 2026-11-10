//! A thin result wrapper around [`sys`].
//!
//! While all the functions here return [`Result`], they are mostly still `unsafe`
//! because raw pointer validity and CUDA context/stream ordering must be upheld by the caller.

use super::sys;
use core::ffi::{c_char, c_uint, c_void, CStr};
use core::mem::MaybeUninit;
use std::string::String;
use std::vec;
use std::vec::Vec;

/// Wrapper around an error [`sys::OptixResult`] code returned by the NVIDIA OptiX API.
///
/// Formats both the symbolic error name ([`OptixError::error_name`]) and the human-readable
/// error description ([`OptixError::error_string`]) queried from the OptiX function table.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OptixError(
    /// The underlying raw [`sys::OptixResult`] status code.
    pub sys::OptixResult,
);

impl sys::OptixResult {
    /// Converts this [`sys::OptixResult`] into a `Result<(), OptixError>`.
    #[inline]
    pub fn result(self) -> Result<(), OptixError> {
        match self {
            sys::OptixResult::OPTIX_SUCCESS => Ok(()),
            _ => Err(OptixError(self)),
        }
    }
}

impl OptixError {
    /// Returns the symbolic name of this OptiX error (via `optixGetErrorName`).
    pub fn error_name(&self) -> &CStr {
        unsafe {
            let ptr = sys::optixGetErrorName(self.0);
            if ptr.is_null() {
                c"OPTIX_ERROR_UNKNOWN"
            } else {
                CStr::from_ptr(ptr)
            }
        }
    }

    /// Returns a human-readable description of this OptiX error (via `optixGetErrorString`).
    pub fn error_string(&self) -> &CStr {
        unsafe {
            let ptr = sys::optixGetErrorString(self.0);
            if ptr.is_null() {
                c"Unknown OptiX error"
            } else {
                CStr::from_ptr(ptr)
            }
        }
    }
}

impl core::fmt::Debug for OptixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("OptixError")
            .field(&self.0)
            .field(&self.error_name())
            .field(&self.error_string())
            .finish()
    }
}

impl core::fmt::Display for OptixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?} ({}): {}",
            self.0,
            self.error_name().to_string_lossy(),
            self.error_string().to_string_lossy()
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OptixError {}

impl From<crate::driver::DriverError> for OptixError {
    fn from(_: crate::driver::DriverError) -> Self {
        OptixError(sys::OptixResult::OPTIX_ERROR_CUDA_ERROR)
    }
}

/// Creates a new [`sys::OptixDeviceContext`] bound to the given CUDA context (`from_context`).
///
/// # Safety
/// `from_context` must be a valid [`sys::CUcontext`] (or null to use the current CUDA context).
pub unsafe fn create_device_context(
    from_context: sys::CUcontext,
    options: Option<&sys::OptixDeviceContextOptions>,
) -> Result<sys::OptixDeviceContext, OptixError> {
    let mut ctx = MaybeUninit::uninit();
    let opts_ptr = match options {
        Some(o) => o as *const sys::OptixDeviceContextOptions,
        None => core::ptr::null(),
    };
    sys::optixDeviceContextCreate(from_context, opts_ptr, ctx.as_mut_ptr()).result()?;
    Ok(ctx.assume_init())
}

/// Destroys an [`sys::OptixDeviceContext`] and all associated OptiX state.
///
/// # Safety
/// `context` must be a valid [`sys::OptixDeviceContext`] not used after this call.
pub unsafe fn destroy_device_context(context: sys::OptixDeviceContext) -> Result<(), OptixError> {
    sys::optixDeviceContextDestroy(context).result()
}

/// Queries a device property from an [`sys::OptixDeviceContext`].
///
/// # Safety
/// `context` must be valid and `value` must point to at least `size_in_bytes` of writable memory.
pub unsafe fn device_context_get_property(
    context: sys::OptixDeviceContext,
    property: sys::OptixDeviceProperty,
    value: *mut c_void,
    size_in_bytes: usize,
) -> Result<(), OptixError> {
    sys::optixDeviceContextGetProperty(context, property, value, size_in_bytes).result()
}

/// Computes the memory usage (output and temporary scratch buffer sizes) required to build
/// an acceleration structure.
///
/// # Safety
/// `context` and all device pointers referenced inside `build_inputs` must be valid.
pub unsafe fn accel_compute_memory_usage(
    context: sys::OptixDeviceContext,
    accel_options: &sys::OptixAccelBuildOptions,
    build_inputs: &[sys::OptixBuildInput],
) -> Result<sys::OptixAccelBufferSizes, OptixError> {
    let mut sizes = MaybeUninit::<sys::OptixAccelBufferSizes>::zeroed();
    sys::optixAccelComputeMemoryUsage(
        context,
        accel_options,
        build_inputs.as_ptr(),
        build_inputs.len() as c_uint,
        sizes.as_mut_ptr(),
    )
    .result()?;
    Ok(sizes.assume_init())
}

/// Builds a bottom-level (BLAS / GAS) or top-level (TLAS / IAS) acceleration structure on `stream`.
///
/// # Safety
/// `context`, `stream`, `build_inputs`, `temp_buffer`, and `output_buffer` must be valid and properly aligned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn accel_build(
    context: sys::OptixDeviceContext,
    stream: sys::CUstream,
    accel_options: &sys::OptixAccelBuildOptions,
    build_inputs: &[sys::OptixBuildInput],
    temp_buffer: sys::CUdeviceptr,
    temp_buffer_size_in_bytes: usize,
    output_buffer: sys::CUdeviceptr,
    output_buffer_size_in_bytes: usize,
    emitted_properties: &[sys::OptixAccelEmitDesc],
) -> Result<sys::OptixTraversableHandle, OptixError> {
    let mut handle: sys::OptixTraversableHandle = 0;
    let emitted_ptr = if emitted_properties.is_empty() {
        core::ptr::null()
    } else {
        emitted_properties.as_ptr()
    };
    sys::optixAccelBuild(
        context,
        stream,
        accel_options,
        build_inputs.as_ptr(),
        build_inputs.len() as c_uint,
        temp_buffer,
        temp_buffer_size_in_bytes,
        output_buffer,
        output_buffer_size_in_bytes,
        &mut handle,
        emitted_ptr,
        emitted_properties.len() as c_uint,
    )
    .result()?;
    Ok(handle)
}

/// Compacts an acceleration structure into a smaller output buffer.
///
/// # Safety
/// `context`, `stream`, `input_handle`, and `output_buffer` must be valid.
pub unsafe fn accel_compact(
    context: sys::OptixDeviceContext,
    stream: sys::CUstream,
    input_handle: sys::OptixTraversableHandle,
    output_buffer: sys::CUdeviceptr,
    output_buffer_size_in_bytes: usize,
) -> Result<sys::OptixTraversableHandle, OptixError> {
    let mut handle: sys::OptixTraversableHandle = 0;
    sys::optixAccelCompact(
        context,
        stream,
        input_handle,
        output_buffer,
        output_buffer_size_in_bytes,
        &mut handle,
    )
    .result()?;
    Ok(handle)
}

/// Compiles an [`sys::OptixModule`] from PTX source bytes and captures the OptiX compiler diagnostic log.
///
/// # Safety
/// `context`, `module_compile_options`, and `pipeline_compile_options` must be valid.
pub unsafe fn module_create_with_log(
    context: sys::OptixDeviceContext,
    module_compile_options: &sys::OptixModuleCompileOptions,
    pipeline_compile_options: &sys::OptixPipelineCompileOptions,
    ptx: &[u8],
) -> Result<(sys::OptixModule, String), (OptixError, String)> {
    let mut module = MaybeUninit::uninit();
    // Strip trailing NUL byte if the caller passed a C string slice.
    let ptx_len = match ptx.last() {
        Some(&0) => ptx.len() - 1,
        _ => ptx.len(),
    };
    let mut log_buf = vec![0u8; 4096];
    let mut log_size = log_buf.len();
    let status = sys::optixModuleCreateFromPTX(
        context,
        module_compile_options,
        pipeline_compile_options,
        ptx.as_ptr().cast::<c_char>(),
        ptx_len,
        log_buf.as_mut_ptr().cast::<c_char>(),
        &mut log_size,
        module.as_mut_ptr(),
    );
    let actual_len = log_size.min(log_buf.len()).saturating_sub(1);
    let log = String::from_utf8_lossy(&log_buf[..actual_len]).into_owned();
    match status.result() {
        Ok(()) => Ok((module.assume_init(), log)),
        Err(e) => Err((e, log)),
    }
}

/// Compiles an [`sys::OptixModule`] from PTX source bytes.
///
/// # Safety
/// `context`, `module_compile_options`, and `pipeline_compile_options` must be valid.
pub unsafe fn module_create(
    context: sys::OptixDeviceContext,
    module_compile_options: &sys::OptixModuleCompileOptions,
    pipeline_compile_options: &sys::OptixPipelineCompileOptions,
    ptx: &[u8],
) -> Result<sys::OptixModule, OptixError> {
    let mut module = MaybeUninit::uninit();
    // Strip trailing NUL byte if the caller passed a C string slice.
    let ptx_len = match ptx.last() {
        Some(&0) => ptx.len() - 1,
        _ => ptx.len(),
    };
    sys::optixModuleCreateFromPTX(
        context,
        module_compile_options,
        pipeline_compile_options,
        ptx.as_ptr().cast::<c_char>(),
        ptx_len,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        module.as_mut_ptr(),
    )
    .result()?;
    Ok(module.assume_init())
}

/// Destroys an [`sys::OptixModule`].
///
/// # Safety
/// `module` must be a valid [`sys::OptixModule`] not used after this call.
pub unsafe fn module_destroy(module: sys::OptixModule) -> Result<(), OptixError> {
    sys::optixModuleDestroy(module).result()
}

/// Creates one or more [`sys::OptixProgramGroup`]s from their descriptions.
///
/// # Safety
/// `context`, `program_descriptions`, and `options` must be valid.
pub unsafe fn program_group_create(
    context: sys::OptixDeviceContext,
    program_descriptions: &[sys::OptixProgramGroupDesc],
    options: &sys::OptixProgramGroupOptions,
) -> Result<Vec<sys::OptixProgramGroup>, OptixError> {
    let mut groups = vec![core::ptr::null_mut(); program_descriptions.len()];
    sys::optixProgramGroupCreate(
        context,
        program_descriptions.as_ptr(),
        program_descriptions.len() as c_uint,
        options,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        groups.as_mut_ptr(),
    )
    .result()?;
    Ok(groups)
}

/// Destroys an [`sys::OptixProgramGroup`].
///
/// # Safety
/// `group` must be a valid [`sys::OptixProgramGroup`] not used after this call.
pub unsafe fn program_group_destroy(group: sys::OptixProgramGroup) -> Result<(), OptixError> {
    sys::optixProgramGroupDestroy(group).result()
}

/// Queries the stack size requirements of an [`sys::OptixProgramGroup`].
///
/// # Safety
/// `group` must be a valid [`sys::OptixProgramGroup`].
pub unsafe fn program_group_get_stack_size(
    group: sys::OptixProgramGroup,
) -> Result<sys::OptixStackSizes, OptixError> {
    let mut stack_sizes = MaybeUninit::<sys::OptixStackSizes>::zeroed();
    sys::optixProgramGroupGetStackSize(group, stack_sizes.as_mut_ptr()).result()?;
    Ok(stack_sizes.assume_init())
}

/// Creates an [`sys::OptixPipeline`] by linking the specified program groups.
///
/// # Safety
/// `context`, `pipeline_compile_options`, `pipeline_link_options`, and `program_groups` must be valid.
pub unsafe fn pipeline_create(
    context: sys::OptixDeviceContext,
    pipeline_compile_options: &sys::OptixPipelineCompileOptions,
    pipeline_link_options: &sys::OptixPipelineLinkOptions,
    program_groups: &[sys::OptixProgramGroup],
) -> Result<sys::OptixPipeline, OptixError> {
    let mut pipeline = MaybeUninit::uninit();
    sys::optixPipelineCreate(
        context,
        pipeline_compile_options,
        pipeline_link_options,
        program_groups.as_ptr(),
        program_groups.len() as c_uint,
        core::ptr::null_mut(),
        core::ptr::null_mut(),
        pipeline.as_mut_ptr(),
    )
    .result()?;
    Ok(pipeline.assume_init())
}

/// Destroys an [`sys::OptixPipeline`].
///
/// # Safety
/// `pipeline` must be a valid [`sys::OptixPipeline`] not used after this call.
pub unsafe fn pipeline_destroy(pipeline: sys::OptixPipeline) -> Result<(), OptixError> {
    sys::optixPipelineDestroy(pipeline).result()
}

/// Configures the stack sizes for an [`sys::OptixPipeline`].
///
/// # Safety
/// `pipeline` must be a valid [`sys::OptixPipeline`].
pub unsafe fn pipeline_set_stack_size(
    pipeline: sys::OptixPipeline,
    direct_callable_stack_size_from_traversal: u32,
    direct_callable_stack_size_from_state: u32,
    continuation_stack_size: u32,
    max_traversable_graph_depth: u32,
) -> Result<(), OptixError> {
    sys::optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversable_graph_depth,
    )
    .result()
}

/// Packs the 32-byte shader binding table (SBT) header for a program group into `header`.
///
/// # Safety
/// `program_group` must be a valid [`sys::OptixProgramGroup`].
pub unsafe fn sbt_record_pack_header(
    program_group: sys::OptixProgramGroup,
    header: &mut [u8; sys::OPTIX_SBT_RECORD_HEADER_SIZE],
) -> Result<(), OptixError> {
    sys::optixSbtRecordPackHeader(program_group, header.as_mut_ptr().cast::<c_void>()).result()
}

/// Launches an [`sys::OptixPipeline`] asynchronously on `stream`.
///
/// # Safety
/// `pipeline`, `stream`, `pipeline_params`, and `sbt` device pointers must be valid.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch(
    pipeline: sys::OptixPipeline,
    stream: sys::CUstream,
    pipeline_params: sys::CUdeviceptr,
    pipeline_params_size: usize,
    sbt: &sys::OptixShaderBindingTable,
    width: u32,
    height: u32,
    depth: u32,
) -> Result<(), OptixError> {
    sys::optixLaunch(
        pipeline,
        stream,
        pipeline_params,
        pipeline_params_size,
        sbt,
        width,
        height,
        depth,
    )
    .result()
}
