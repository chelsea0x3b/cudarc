//! Safe abstractions over [`crate::optix::result`] and [`crate::optix::sys`] that integrate
//! directly with [`crate::driver::CudaContext`], [`crate::driver::CudaStream`], and [`crate::driver::CudaSlice`].

use crate::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
};
use crate::optix::{
    result::{self, OptixError},
    sys,
};
use core::ffi::{c_uint, c_void};
use std::{ffi::CString, sync::Arc, vec, vec::Vec};

/// An NVIDIA OptiX device context bound to a [`CudaContext`].
///
/// Automatically destroys the underlying [`sys::OptixDeviceContext`] when dropped.
#[derive(Debug)]
pub struct OptixContext {
    cuda_ctx: Arc<CudaContext>,
    raw: sys::OptixDeviceContext,
}

unsafe impl Send for OptixContext {}
unsafe impl Sync for OptixContext {}

impl OptixContext {
    /// Returns `true` if both the CUDA driver (`libcuda.so.1` / `nvcuda.dll`) and the
    /// NVIDIA OptiX driver (`libnvoptix.so.1` / `nvoptix.dll`) are present on the system
    /// and support the OptiX function table ABI.
    pub fn is_available() -> bool {
        unsafe {
            crate::driver::sys::is_culib_present()
                && sys::is_optixlib_present()
                && sys::optix_function_table().is_ok()
        }
    }

    /// Creates a new [`OptixContext`] bound to `cuda_ctx` with default options.
    pub fn new(cuda_ctx: &Arc<CudaContext>) -> Result<Arc<Self>, OptixError> {
        let options = sys::OptixDeviceContextOptions::default();
        unsafe { Self::with_options(cuda_ctx, &options) }
    }

    /// Creates a new [`OptixContext`] bound to `cuda_ctx` with custom [`sys::OptixDeviceContextOptions`].
    ///
    /// # Safety
    /// If `options.logCallbackFunction` is `Some(...)`, `options.logCallbackData` must remain valid
    /// and thread-safe for the entire lifetime of the returned [`OptixContext`].
    pub unsafe fn with_options(
        cuda_ctx: &Arc<CudaContext>,
        options: &sys::OptixDeviceContextOptions,
    ) -> Result<Arc<Self>, OptixError> {
        cuda_ctx.bind_to_thread()?;
        let raw = result::create_device_context(cuda_ctx.cu_ctx(), Some(options))?;
        Ok(Arc::new(Self {
            cuda_ctx: cuda_ctx.clone(),
            raw,
        }))
    }

    /// Queries the RT Core hardware version reported by OptiX (`0` if software only, e.g. `20` for Ampere, `30` for Ada).
    pub fn rtcore_version(&self) -> Result<u32, OptixError> {
        self.cuda_ctx.bind_to_thread()?;
        let mut version: c_uint = 0;
        unsafe {
            result::device_context_get_property(
                self.raw,
                sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                (&mut version as *mut c_uint).cast::<c_void>(),
                core::mem::size_of::<c_uint>(),
            )?;
        }
        Ok(version)
    }

    /// Returns a reference to the associated [`CudaContext`].
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    /// Returns the raw [`sys::OptixDeviceContext`] handle.
    pub fn raw(&self) -> sys::OptixDeviceContext {
        self.raw
    }
}

impl Drop for OptixContext {
    fn drop(&mut self) {
        let _ = self.cuda_ctx.bind_to_thread();
        if !self.raw.is_null() {
            let _ = unsafe { result::destroy_device_context(self.raw) };
        }
    }
}

#[inline]
fn align_up_device_ptr(ptr: sys::CUdeviceptr, alignment: usize) -> sys::CUdeviceptr {
    let mask = (alignment as u64) - 1;
    (ptr + mask) & !mask
}

/// Safe RAII wrapper around an OptiX acceleration structure (BLAS / GAS or TLAS / IAS).
///
/// Owns the backing GPU memory buffer (`CudaSlice<u8>`) so the [`sys::OptixTraversableHandle`]
/// remains valid for the lifetime of this struct.
#[derive(Debug)]
pub struct OptixAccel {
    ctx: Arc<OptixContext>,
    output_buffer: CudaSlice<u8>,
    handle: sys::OptixTraversableHandle,
}

unsafe impl Send for OptixAccel {}
unsafe impl Sync for OptixAccel {}

impl OptixAccel {
    /// Builds a bottom-level triangle acceleration structure (BLAS / GAS) from `[f32; 3]` vertices
    /// and optional `[u32; 3]` triangle indices on `stream`.
    pub fn build_triangles(
        ctx: &Arc<OptixContext>,
        stream: &Arc<CudaStream>,
        vertices: &CudaSlice<[f32; 3]>,
        indices: Option<&CudaSlice<[u32; 3]>>,
        geometry_flags: u32,
        build_options: Option<sys::OptixAccelBuildOptions>,
    ) -> Result<Self, OptixError> {
        if vertices.is_empty() || indices.is_some_and(|idx| idx.is_empty()) {
            return Err(OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE));
        }
        ctx.cuda_context().bind_to_thread()?;
        let (vert_ptr, _vert_sync) = vertices.device_ptr(stream);
        let vertex_buffers = [vert_ptr];
        let flags_array = [geometry_flags];

        // Keep index guard alive across optixAccelBuild if indices are supplied.
        let idx_guard = indices.map(|idx| idx.device_ptr(stream));
        let (index_buffer, num_index_triplets, index_format, index_stride) = match &idx_guard {
            Some((ptr, _)) => (
                *ptr,
                indices.unwrap().len() as c_uint,
                sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3,
                core::mem::size_of::<[u32; 3]>() as c_uint,
            ),
            None => (0, 0, sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_NONE, 0),
        };

        let tri_array = sys::OptixBuildInputTriangleArray {
            vertexBuffers: vertex_buffers.as_ptr(),
            numVertices: vertices.len() as c_uint,
            vertexFormat: sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
            vertexStrideInBytes: core::mem::size_of::<[f32; 3]>() as c_uint,
            indexBuffer: index_buffer,
            numIndexTriplets: num_index_triplets,
            indexFormat: index_format,
            indexStrideInBytes: index_stride,
            preTransform: 0,
            flags: flags_array.as_ptr(),
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
            transformFormat: sys::OptixTransformFormat::OPTIX_TRANSFORM_FORMAT_NONE,
        };

        let build_input = sys::OptixBuildInput::triangles(tri_array);
        let options = build_options.unwrap_or_default();
        unsafe { Self::build_from_inputs(ctx, stream, &options, &[build_input]) }
    }

    /// Builds a bottom-level custom-primitive AABB acceleration structure (BLAS / GAS)
    /// from an [`sys::OptixAabb`] device slice on `stream`.
    pub fn build_aabbs(
        ctx: &Arc<OptixContext>,
        stream: &Arc<CudaStream>,
        aabbs: &CudaSlice<sys::OptixAabb>,
        geometry_flags: u32,
        build_options: Option<sys::OptixAccelBuildOptions>,
    ) -> Result<Self, OptixError> {
        if aabbs.is_empty() {
            return Err(OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE));
        }
        ctx.cuda_context().bind_to_thread()?;
        let (aabb_ptr, _aabb_sync) = aabbs.device_ptr(stream);
        let aabb_buffers = [aabb_ptr];
        let flags_array = [geometry_flags];

        let custom_array = sys::OptixBuildInputCustomPrimitiveArray {
            aabbBuffers: aabb_buffers.as_ptr(),
            numPrimitives: aabbs.len() as c_uint,
            strideInBytes: core::mem::size_of::<sys::OptixAabb>() as c_uint,
            flags: flags_array.as_ptr(),
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
        };

        let build_input = sys::OptixBuildInput::custom_primitives(custom_array);
        let options = build_options.unwrap_or_default();
        unsafe { Self::build_from_inputs(ctx, stream, &options, &[build_input]) }
    }

    /// Builds a top-level instance acceleration structure (TLAS / IAS) from an
    /// [`sys::OptixInstance`] device slice on `stream`.
    pub fn build_instances(
        ctx: &Arc<OptixContext>,
        stream: &Arc<CudaStream>,
        instances: &CudaSlice<sys::OptixInstance>,
        build_options: Option<sys::OptixAccelBuildOptions>,
    ) -> Result<Self, OptixError> {
        if instances.is_empty() {
            return Err(OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE));
        }
        ctx.cuda_context().bind_to_thread()?;
        let (inst_ptr, _inst_sync) = instances.device_ptr(stream);

        let instance_array = sys::OptixBuildInputInstanceArray {
            instances: inst_ptr,
            numInstances: instances.len() as c_uint,
        };

        let build_input = sys::OptixBuildInput::instances(instance_array);
        let options = build_options.unwrap_or_default();
        unsafe { Self::build_from_inputs(ctx, stream, &options, &[build_input]) }
    }

    /// Builds an acceleration structure from raw [`sys::OptixBuildInput`] descriptors,
    /// automatically querying memory requirements and allocating 128-byte aligned output and scratch buffers.
    ///
    /// # Safety
    /// All host and device pointers referenced within `build_inputs` (such as `vertexBuffers`,
    /// `flags`, `aabbBuffers`, and `instances`) must be valid and properly aligned for the duration
    /// of the build on `stream`.
    pub unsafe fn build_from_inputs(
        ctx: &Arc<OptixContext>,
        stream: &Arc<CudaStream>,
        options: &sys::OptixAccelBuildOptions,
        build_inputs: &[sys::OptixBuildInput],
    ) -> Result<Self, OptixError> {
        if build_inputs.is_empty() {
            return Err(OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE));
        }
        ctx.cuda_context().bind_to_thread()?;
        let sizes = result::accel_compute_memory_usage(ctx.raw, options, build_inputs)?;

        let align = sys::OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
        let out_alloc_bytes = sizes.outputSizeInBytes.max(1) + align;
        let temp_alloc_bytes = sizes.tempSizeInBytes.max(1) + align;

        let mut output_buffer = stream.alloc_zeros::<u8>(out_alloc_bytes)?;
        let mut temp_buffer = stream.alloc_zeros::<u8>(temp_alloc_bytes)?;

        let handle = {
            let (raw_out_ptr, _out_sync) = output_buffer.device_ptr_mut(stream);
            let (raw_temp_ptr, _temp_sync) = temp_buffer.device_ptr_mut(stream);
            let aligned_out_ptr = align_up_device_ptr(raw_out_ptr, align);
            let aligned_temp_ptr = align_up_device_ptr(raw_temp_ptr, align);

            result::accel_build(
                ctx.raw,
                stream.cu_stream(),
                options,
                build_inputs,
                aligned_temp_ptr,
                sizes.tempSizeInBytes,
                aligned_out_ptr,
                sizes.outputSizeInBytes,
                &[],
            )?
        };

        Ok(Self {
            ctx: ctx.clone(),
            output_buffer,
            handle,
        })
    }

    /// Returns the [`sys::OptixTraversableHandle`] (`u64`) for this acceleration structure.
    pub fn handle(&self) -> sys::OptixTraversableHandle {
        self.handle
    }

    /// Returns a reference to the backing GPU buffer storing the acceleration structure.
    pub fn output_buffer(&self) -> &CudaSlice<u8> {
        &self.output_buffer
    }

    /// Returns a reference to the parent [`OptixContext`].
    pub fn context(&self) -> &Arc<OptixContext> {
        &self.ctx
    }
}

/// A compiled OptiX PTX module.
///
/// Automatically destroys the underlying [`sys::OptixModule`] when dropped.
#[derive(Debug)]
pub struct OptixModule {
    ctx: Arc<OptixContext>,
    raw: sys::OptixModule,
}

unsafe impl Send for OptixModule {}
unsafe impl Sync for OptixModule {}

impl OptixModule {
    /// Compiles an [`OptixModule`] from a PTX string.
    pub fn from_ptx(
        ctx: &Arc<OptixContext>,
        ptx: &str,
        module_options: &sys::OptixModuleCompileOptions,
        pipeline_options: &sys::OptixPipelineCompileOptions,
    ) -> Result<Arc<Self>, OptixError> {
        Self::from_ptx_bytes(ctx, ptx.as_bytes(), module_options, pipeline_options)
    }

    /// Compiles an [`OptixModule`] from PTX bytes.
    pub fn from_ptx_bytes(
        ctx: &Arc<OptixContext>,
        ptx: &[u8],
        module_options: &sys::OptixModuleCompileOptions,
        pipeline_options: &sys::OptixPipelineCompileOptions,
    ) -> Result<Arc<Self>, OptixError> {
        ctx.cuda_context().bind_to_thread()?;
        let raw = unsafe { result::module_create(ctx.raw, module_options, pipeline_options, ptx) }?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            raw,
        }))
    }

    /// Returns the raw [`sys::OptixModule`] handle.
    pub fn raw(&self) -> sys::OptixModule {
        self.raw
    }
}

impl Drop for OptixModule {
    fn drop(&mut self) {
        let _ = self.ctx.cuda_context().bind_to_thread();
        if !self.raw.is_null() {
            let _ = unsafe { result::module_destroy(self.raw) };
        }
    }
}

/// An OptiX program group (Raygen, Miss, Hitgroup, Exception, or Callables).
///
/// Automatically destroys the underlying [`sys::OptixProgramGroup`] when dropped.
#[derive(Debug)]
pub struct OptixProgramGroup {
    ctx: Arc<OptixContext>,
    _modules: Vec<Arc<OptixModule>>,
    raw: sys::OptixProgramGroup,
    kind: sys::OptixProgramGroupKind,
}

unsafe impl Send for OptixProgramGroup {}
unsafe impl Sync for OptixProgramGroup {}

impl OptixProgramGroup {
    /// Creates a Raygen program group (`__raygen__*`).
    pub fn new_raygen(
        ctx: &Arc<OptixContext>,
        module: &Arc<OptixModule>,
        entry_function_name: &str,
    ) -> Result<Arc<Self>, OptixError> {
        ctx.cuda_context().bind_to_thread()?;
        let c_name = CString::new(entry_function_name)
            .map_err(|_| OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE))?;
        let desc = sys::OptixProgramGroupDesc::raygen(module.raw, c_name.as_ptr());
        let opts = sys::OptixProgramGroupOptions::default();
        let groups = unsafe { result::program_group_create(ctx.raw, &[desc], &opts) }?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            _modules: vec![module.clone()],
            raw: groups[0],
            kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
        }))
    }

    /// Creates a Miss program group (`__miss__*`), or a null miss group if `module` is `None`.
    pub fn new_miss(
        ctx: &Arc<OptixContext>,
        module: Option<(&Arc<OptixModule>, &str)>,
    ) -> Result<Arc<Self>, OptixError> {
        ctx.cuda_context().bind_to_thread()?;
        let mut modules = Vec::new();
        let c_name = match module {
            Some((m, name)) => {
                modules.push(m.clone());
                Some(
                    CString::new(name)
                        .map_err(|_| OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE))?,
                )
            }
            None => None,
        };
        let raw_mod = module.map(|(m, _)| m.raw).unwrap_or(core::ptr::null_mut());
        let name_ptr = c_name
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(core::ptr::null());
        let desc = sys::OptixProgramGroupDesc::miss(raw_mod, name_ptr);
        let opts = sys::OptixProgramGroupOptions::default();
        let groups = unsafe { result::program_group_create(ctx.raw, &[desc], &opts) }?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            _modules: modules,
            raw: groups[0],
            kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
        }))
    }

    /// Creates a HitGroup program group from optional closest-hit (`__closesthit__*`),
    /// any-hit (`__anyhit__*`), and custom intersection (`__intersection__*`) shaders.
    pub fn new_hitgroup(
        ctx: &Arc<OptixContext>,
        closesthit: Option<(&Arc<OptixModule>, &str)>,
        anyhit: Option<(&Arc<OptixModule>, &str)>,
        intersection: Option<(&Arc<OptixModule>, &str)>,
    ) -> Result<Arc<Self>, OptixError> {
        ctx.cuda_context().bind_to_thread()?;
        let mut modules = Vec::new();

        let ch_cname = match closesthit {
            Some((m, name)) => {
                modules.push(m.clone());
                Some(
                    CString::new(name)
                        .map_err(|_| OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE))?,
                )
            }
            None => None,
        };
        let ah_cname = match anyhit {
            Some((m, name)) => {
                modules.push(m.clone());
                Some(
                    CString::new(name)
                        .map_err(|_| OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE))?,
                )
            }
            None => None,
        };
        let is_cname = match intersection {
            Some((m, name)) => {
                modules.push(m.clone());
                Some(
                    CString::new(name)
                        .map_err(|_| OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE))?,
                )
            }
            None => None,
        };

        let hitgroup = sys::OptixProgramGroupHitgroup {
            moduleCH: closesthit
                .map(|(m, _)| m.raw)
                .unwrap_or(core::ptr::null_mut()),
            entryFunctionNameCH: ch_cname
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(core::ptr::null()),
            moduleAH: anyhit.map(|(m, _)| m.raw).unwrap_or(core::ptr::null_mut()),
            entryFunctionNameAH: ah_cname
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(core::ptr::null()),
            moduleIS: intersection
                .map(|(m, _)| m.raw)
                .unwrap_or(core::ptr::null_mut()),
            entryFunctionNameIS: is_cname
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(core::ptr::null()),
        };

        let desc = sys::OptixProgramGroupDesc::hitgroup(hitgroup);
        let opts = sys::OptixProgramGroupOptions::default();
        let groups = unsafe { result::program_group_create(ctx.raw, &[desc], &opts) }?;
        Ok(Arc::new(Self {
            ctx: ctx.clone(),
            _modules: modules,
            raw: groups[0],
            kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
        }))
    }

    /// Queries the stack size requirements of this program group.
    pub fn stack_sizes(&self) -> Result<sys::OptixStackSizes, OptixError> {
        self.ctx.cuda_context().bind_to_thread()?;
        unsafe { result::program_group_get_stack_size(self.raw) }
    }

    /// Packs the 32-byte opaque SBT record header for this program group.
    pub fn pack_header(&self) -> Result<[u8; sys::OPTIX_SBT_RECORD_HEADER_SIZE], OptixError> {
        self.ctx.cuda_context().bind_to_thread()?;
        let mut header = [0u8; sys::OPTIX_SBT_RECORD_HEADER_SIZE];
        unsafe { result::sbt_record_pack_header(self.raw, &mut header) }?;
        Ok(header)
    }

    /// Returns the kind of this program group.
    pub fn kind(&self) -> sys::OptixProgramGroupKind {
        self.kind
    }

    /// Returns the raw [`sys::OptixProgramGroup`] handle.
    pub fn raw(&self) -> sys::OptixProgramGroup {
        self.raw
    }
}

impl Drop for OptixProgramGroup {
    fn drop(&mut self) {
        let _ = self.ctx.cuda_context().bind_to_thread();
        if !self.raw.is_null() {
            let _ = unsafe { result::program_group_destroy(self.raw) };
        }
    }
}

/// A 16-byte aligned Shader Binding Table (SBT) record containing the 32-byte
/// OptiX record header followed by user payload `T`.
///
/// OptiX requires every SBT record in device memory to begin with a 32-byte opaque header
/// packed by `optixSbtRecordPackHeader` ([`sys::OPTIX_SBT_RECORD_HEADER_SIZE`]) and to be aligned
/// to a 16-byte boundary ([`sys::OPTIX_SBT_RECORD_ALIGNMENT`]).
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
pub struct SbtRecord<T = [u8; 0]> {
    /// Opaque 32-byte shader binding table record header populated by [`OptixProgramGroup::pack_header`].
    pub header: [u8; sys::OPTIX_SBT_RECORD_HEADER_SIZE],
    /// Inline user data accessible on the device via `optixGetSbtDataPointer()`.
    pub data: T,
}

unsafe impl<T: DeviceRepr> DeviceRepr for SbtRecord<T> {}
unsafe impl<T: ValidAsZeroBits> ValidAsZeroBits for SbtRecord<T> {}

impl SbtRecord<[u8; 0]> {
    /// Packs a header-only SBT record (32 bytes, 16-byte aligned) for `group`.
    pub fn pack(group: &OptixProgramGroup) -> Result<Self, OptixError> {
        Self::pack_with_data(group, [])
    }
}

impl<T: Copy> SbtRecord<T> {
    /// Packs an SBT record with user data `data` for `group`.
    ///
    /// Zero-initializes all bytes (including any tail alignment padding) before writing
    /// `header` and `data`, ensuring the entire record is initialized for host-to-device transfer.
    pub fn pack_with_data(group: &OptixProgramGroup, data: T) -> Result<Self, OptixError> {
        let header = group.pack_header()?;
        let mut rec = core::mem::MaybeUninit::<Self>::zeroed();
        unsafe {
            let ptr = rec.as_mut_ptr();
            core::ptr::addr_of_mut!((*ptr).header).write(header);
            core::ptr::addr_of_mut!((*ptr).data).write(data);
            Ok(rec.assume_init())
        }
    }
}

/// Safe RAII owner of device-side Shader Binding Table (SBT) buffers and the
/// corresponding [`sys::OptixShaderBindingTable`] descriptor.
#[derive(Debug)]
pub struct OptixSbt {
    _raygen_buffer: CudaSlice<u8>,
    _miss_buffer: Option<CudaSlice<u8>>,
    _hitgroup_buffer: Option<CudaSlice<u8>>,
    raw: sys::OptixShaderBindingTable,
}

unsafe impl Send for OptixSbt {}
unsafe impl Sync for OptixSbt {}

impl OptixSbt {
    /// Packs header-only SBT records for `raygen`, `miss`, and `hitgroups` onto `stream`
    /// with guaranteed 16-byte [`sys::OPTIX_SBT_RECORD_ALIGNMENT`] alignment.
    pub fn from_groups(
        stream: &Arc<CudaStream>,
        raygen: &OptixProgramGroup,
        miss: &[&OptixProgramGroup],
        hitgroups: &[&OptixProgramGroup],
    ) -> Result<Self, OptixError> {
        let rg_rec = SbtRecord::pack(raygen)?;
        let mut ms_recs = Vec::with_capacity(miss.len());
        for m in miss {
            ms_recs.push(SbtRecord::pack(m)?);
        }
        let mut hg_recs = Vec::with_capacity(hitgroups.len());
        for h in hitgroups {
            hg_recs.push(SbtRecord::pack(h)?);
        }
        Self::from_records(stream, &rg_rec, &ms_recs, &hg_recs)
    }

    /// Uploads typed SBT records (`SbtRecord<R>`, `SbtRecord<M>`, `SbtRecord<H>`) onto `stream`
    /// and constructs an [`OptixSbt`].
    pub fn from_records<R: DeviceRepr, M: DeviceRepr, H: DeviceRepr>(
        stream: &Arc<CudaStream>,
        raygen: &SbtRecord<R>,
        miss: &[SbtRecord<M>],
        hitgroups: &[SbtRecord<H>],
    ) -> Result<Self, OptixError> {
        stream.context().bind_to_thread()?;
        assert_eq!(
            core::mem::size_of::<SbtRecord<R>>() % sys::OPTIX_SBT_RECORD_ALIGNMENT,
            0
        );
        assert_eq!(
            core::mem::size_of::<SbtRecord<M>>() % sys::OPTIX_SBT_RECORD_ALIGNMENT,
            0
        );
        assert_eq!(
            core::mem::size_of::<SbtRecord<H>>() % sys::OPTIX_SBT_RECORD_ALIGNMENT,
            0
        );

        let rg_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (raygen as *const SbtRecord<R>).cast::<u8>(),
                core::mem::size_of::<SbtRecord<R>>(),
            )
        };
        let raygen_buffer = stream.clone_htod(rg_bytes)?;
        let rg_ptr = {
            let (ptr, _sync) = raygen_buffer.device_ptr(stream);
            ptr
        };

        let (miss_buffer, ms_ptr, ms_stride, ms_count) = if miss.is_empty() {
            (None, 0, 0, 0)
        } else {
            let ms_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    miss.as_ptr().cast::<u8>(),
                    core::mem::size_of_val(miss),
                )
            };
            let buf = stream.clone_htod(ms_bytes)?;
            let ptr = {
                let (p, _sync) = buf.device_ptr(stream);
                p
            };
            (
                Some(buf),
                ptr,
                core::mem::size_of::<SbtRecord<M>>() as c_uint,
                miss.len() as c_uint,
            )
        };

        let (hitgroup_buffer, hg_ptr, hg_stride, hg_count) = if hitgroups.is_empty() {
            (None, 0, 0, 0)
        } else {
            let hg_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    hitgroups.as_ptr().cast::<u8>(),
                    core::mem::size_of_val(hitgroups),
                )
            };
            let buf = stream.clone_htod(hg_bytes)?;
            let ptr = {
                let (p, _sync) = buf.device_ptr(stream);
                p
            };
            (
                Some(buf),
                ptr,
                core::mem::size_of::<SbtRecord<H>>() as c_uint,
                hitgroups.len() as c_uint,
            )
        };

        let raw = sys::OptixShaderBindingTable {
            raygenRecord: rg_ptr,
            exceptionRecord: 0,
            missRecordBase: ms_ptr,
            missRecordStrideInBytes: ms_stride,
            missRecordCount: ms_count,
            hitgroupRecordBase: hg_ptr,
            hitgroupRecordStrideInBytes: hg_stride,
            hitgroupRecordCount: hg_count,
            callablesRecordBase: 0,
            callablesRecordStrideInBytes: 0,
            callablesRecordCount: 0,
        };

        Ok(Self {
            _raygen_buffer: raygen_buffer,
            _miss_buffer: miss_buffer,
            _hitgroup_buffer: hitgroup_buffer,
            raw,
        })
    }

    /// Returns a reference to the underlying [`sys::OptixShaderBindingTable`].
    pub fn raw(&self) -> &sys::OptixShaderBindingTable {
        &self.raw
    }
}

/// An linked OptiX ray-tracing pipeline.
///
/// Automatically destroys the underlying [`sys::OptixPipeline`] when dropped.
#[derive(Debug)]
pub struct OptixPipeline {
    ctx: Arc<OptixContext>,
    _groups: Vec<Arc<OptixProgramGroup>>,
    raw: sys::OptixPipeline,
}

unsafe impl Send for OptixPipeline {}
unsafe impl Sync for OptixPipeline {}

impl OptixPipeline {
    /// Links `program_groups` into an [`OptixPipeline`] and automatically computes and configures
    /// stack sizes for `max_traversable_graph_depth`.
    pub fn new(
        ctx: &Arc<OptixContext>,
        compile_options: &sys::OptixPipelineCompileOptions,
        link_options: &sys::OptixPipelineLinkOptions,
        program_groups: &[Arc<OptixProgramGroup>],
        max_traversable_graph_depth: u32,
    ) -> Result<Arc<Self>, OptixError> {
        ctx.cuda_context().bind_to_thread()?;
        let raw_groups: Vec<sys::OptixProgramGroup> =
            program_groups.iter().map(|g| g.raw).collect();
        let raw = unsafe {
            result::pipeline_create(ctx.raw, compile_options, link_options, &raw_groups)
        }?;

        // Wrap `raw` immediately in `Arc<Self>` so `Drop` destroys the pipeline if stack-size
        // calculation or configuration fails.
        let pipeline = Arc::new(Self {
            ctx: ctx.clone(),
            _groups: program_groups.to_vec(),
            raw,
        });

        let mut css_rg = 0u32;
        let mut css_ms = 0u32;
        let mut css_ch = 0u32;
        let mut css_ah = 0u32;
        let mut css_is = 0u32;
        let mut css_cc = 0u32;
        let mut dss_dc = 0u32;

        for group in program_groups {
            let ss = group.stack_sizes()?;
            css_rg = css_rg.max(ss.cssRG);
            css_ms = css_ms.max(ss.cssMS);
            css_ch = css_ch.max(ss.cssCH);
            css_ah = css_ah.max(ss.cssAH);
            css_is = css_is.max(ss.cssIS);
            css_cc = css_cc.max(ss.cssCC);
            dss_dc = dss_dc.max(ss.dssDC);
        }

        let max_trace_depth = link_options.maxTraceDepth;
        let continuation_stack_size = css_rg
            + css_cc
            + max_trace_depth.saturating_sub(1) * css_ch
            + max_trace_depth.min(1) * (css_ch + css_ah + css_is).max(css_ms);

        let depth = max_traversable_graph_depth.max(1);
        pipeline.set_stack_size(dss_dc, dss_dc, continuation_stack_size, depth)?;

        Ok(pipeline)
    }

    /// Manually overrides the pipeline stack sizes via `optixPipelineSetStackSize`.
    pub fn set_stack_size(
        &self,
        direct_callable_stack_size_from_traversal: u32,
        direct_callable_stack_size_from_state: u32,
        continuation_stack_size: u32,
        max_traversable_graph_depth: u32,
    ) -> Result<(), OptixError> {
        self.ctx.cuda_context().bind_to_thread()?;
        unsafe {
            result::pipeline_set_stack_size(
                self.raw,
                direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversable_graph_depth,
            )
        }
    }

    /// Launches this OptiX pipeline asynchronously on `stream` with dimensions `(width, height, depth)`.
    pub fn launch<P: DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        pipeline_params: &CudaSlice<P>,
        sbt: &OptixSbt,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Result<(), OptixError> {
        self.ctx.cuda_context().bind_to_thread()?;
        let (params_ptr, _params_sync) = pipeline_params.device_ptr(stream);
        unsafe {
            result::launch(
                self.raw,
                stream.cu_stream(),
                params_ptr,
                pipeline_params.num_bytes(),
                sbt.raw(),
                width,
                height,
                depth,
            )
        }
    }

    /// Returns the raw [`sys::OptixPipeline`] handle.
    pub fn raw(&self) -> sys::OptixPipeline {
        self.raw
    }
}

impl Drop for OptixPipeline {
    fn drop(&mut self) {
        let _ = self.ctx.cuda_context().bind_to_thread();
        if !self.raw.is_null() {
            let _ = unsafe { result::pipeline_destroy(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn test_optix_probe_and_context() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();
        assert!(!optix_ctx.raw().is_null());
        let rtcore = optix_ctx.rtcore_version().unwrap();
        assert!(rtcore > 0);
    }

    #[test]
    fn test_optix_build_triangles_and_instances() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        let vertices: [[f32; 3]; 4] = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ];
        let indices: [[u32; 3]; 2] = [[0, 1, 2], [0, 2, 3]];

        let d_vertices = stream.clone_htod(&vertices).unwrap();
        let d_indices = stream.clone_htod(&indices).unwrap();

        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_vertices,
            Some(&d_indices),
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();
        assert_ne!(blas.handle(), 0);

        let inst = sys::OptixInstance::with_identity(blas.handle(), 42);
        let d_instances = stream.clone_htod(&[inst]).unwrap();

        let tlas = OptixAccel::build_instances(&optix_ctx, &stream, &d_instances, None).unwrap();
        assert_ne!(tlas.handle(), 0);
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_optix_build_aabbs() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        let aabbs = [
            sys::OptixAabb {
                minX: -1.0,
                minY: -1.0,
                minZ: -1.0,
                maxX: 0.0,
                maxY: 0.0,
                maxZ: 0.0,
            },
            sys::OptixAabb {
                minX: 0.5,
                minY: 0.5,
                minZ: 0.5,
                maxX: 2.0,
                maxY: 2.0,
                maxZ: 2.0,
            },
        ];
        let d_aabbs = stream.clone_htod(&aabbs).unwrap();

        let blas = OptixAccel::build_aabbs(
            &optix_ctx,
            &stream,
            &d_aabbs,
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_NONE as u32,
            None,
        )
        .unwrap();
        assert_ne!(blas.handle(), 0);
        stream.synchronize().unwrap();
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    struct RayMeshParams {
        handle: u64,
        rays_origin: u64,
        rays_dir: u64,
        out_prim: u64,
        out_t: u64,
    }
    unsafe impl DeviceRepr for RayMeshParams {}

    /// End-to-end hardware RT Core ray-mesh intersection test using inline PTX (`_optix_trace_0`)
    /// against a triangle BLAS/TLAS and verifying exact agreement with CPU reference.
    #[test]
    fn test_optix_hardware_ray_mesh_intersection() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        // Two triangles at z = 2.0 and z = 5.0:
        // Triangle 0: [-1,-1,2], [1,-1,2], [0,1,2]
        // Triangle 1: [2,-1,5], [4,-1,5], [3,1,5]
        let vertices: [[f32; 3]; 6] = [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            [2.0, -1.0, 5.0],
            [4.0, -1.0, 5.0],
            [3.0, 1.0, 5.0],
        ];
        let indices: [[u32; 3]; 2] = [[0, 1, 2], [3, 4, 5]];

        let d_vertices = stream.clone_htod(&vertices).unwrap();
        let d_indices = stream.clone_htod(&indices).unwrap();

        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_vertices,
            Some(&d_indices),
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();

        let inst = sys::OptixInstance::with_identity(blas.handle(), 0);
        let d_instances = stream.clone_htod(&[inst]).unwrap();
        let tlas = OptixAccel::build_instances(&optix_ctx, &stream, &d_instances, None).unwrap();

        let ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.const .align 8 .b8 params[40];

.extern .func (.reg .u32 $value) _optix_get_launch_index_x ();
.extern .func (.reg .u32 $value) _optix_read_primitive_idx ();
.extern .func (.reg .f32 $value) _optix_get_ray_tmax ();
.extern .func () _optix_trace_0 (
    .reg .u64 handle,
    .reg .f32 rayOriginX,
    .reg .f32 rayOriginY,
    .reg .f32 rayOriginZ,
    .reg .f32 rayDirectionX,
    .reg .f32 rayDirectionY,
    .reg .f32 rayDirectionZ,
    .reg .f32 tmin,
    .reg .f32 tmax,
    .reg .f32 rayTime,
    .reg .u32 visibilityMask,
    .reg .u32 rayFlags,
    .reg .u32 SBToffset,
    .reg .u32 SBTstride,
    .reg .u32 missSBTIndex
);

.visible .entry __raygen__rg()
{
    .reg .u32 %idx, %vis, %flags, %sbt_off, %sbt_stride, %miss_idx;
    .reg .u64 %handle, %orig_base, %dir_base, %off12, %orig_ptr, %dir_ptr;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %tmin, %tmax, %rtime;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %handle, [params + 0];
    ld.const.u64 %orig_base, [params + 8];
    ld.const.u64 %dir_base, [params + 16];

    mul.wide.u32 %off12, %idx, 12;
    add.u64 %orig_ptr, %orig_base, %off12;
    add.u64 %dir_ptr, %dir_base, %off12;

    ld.global.f32 %ox, [%orig_ptr + 0];
    ld.global.f32 %oy, [%orig_ptr + 4];
    ld.global.f32 %oz, [%orig_ptr + 8];

    ld.global.f32 %dx, [%dir_ptr + 0];
    ld.global.f32 %dy, [%dir_ptr + 4];
    ld.global.f32 %dz, [%dir_ptr + 8];

    mov.f32 %tmin, 0f00000000;
    mov.f32 %tmax, 0f461C4000;
    mov.f32 %rtime, 0f00000000;
    mov.u32 %vis, 255;
    mov.u32 %flags, 0;
    mov.u32 %sbt_off, 0;
    mov.u32 %sbt_stride, 1;
    mov.u32 %miss_idx, 0;

    call (), _optix_trace_0, (
        %handle,
        %ox, %oy, %oz,
        %dx, %dy, %dz,
        %tmin, %tmax, %rtime,
        %vis, %flags, %sbt_off, %sbt_stride, %miss_idx
    );
    ret;
}

.visible .entry __closesthit__ch()
{
    .reg .u32 %idx, %prim;
    .reg .f32 %t;
    .reg .u64 %prim_base, %t_base, %off4, %prim_ptr, %t_ptr;

    call (%idx), _optix_get_launch_index_x, ();
    call (%prim), _optix_read_primitive_idx, ();
    call (%t), _optix_get_ray_tmax, ();

    ld.const.u64 %prim_base, [params + 24];
    ld.const.u64 %t_base, [params + 32];
    mul.wide.u32 %off4, %idx, 4;
    add.u64 %prim_ptr, %prim_base, %off4;
    add.u64 %t_ptr, %t_base, %off4;

    st.global.u32 [%prim_ptr], %prim;
    st.global.f32 [%t_ptr], %t;
    ret;
}

.visible .entry __miss__ms()
{
    .reg .u32 %idx, %none;
    .reg .f32 %neg;
    .reg .u64 %prim_base, %t_base, %off4, %prim_ptr, %t_ptr;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %prim_base, [params + 24];
    ld.const.u64 %t_base, [params + 32];
    mul.wide.u32 %off4, %idx, 4;
    add.u64 %prim_ptr, %prim_base, %off4;
    add.u64 %t_ptr, %t_base, %off4;

    mov.u32 %none, -1;
    mov.f32 %neg, 0fBF800000;
    st.global.u32 [%prim_ptr], %none;
    st.global.f32 [%t_ptr], %neg;
    ret;
}
"#;

        let module_opts = sys::OptixModuleCompileOptions::default();
        let pipeline_opts = sys::OptixPipelineCompileOptions {
            numPayloadValues: 0,
            numAttributeValues: 2,
            ..Default::default()
        };
        let link_opts = sys::OptixPipelineLinkOptions::default();

        let module = OptixModule::from_ptx(&optix_ctx, ptx, &module_opts, &pipeline_opts).unwrap();
        let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__rg").unwrap();
        let ms = OptixProgramGroup::new_miss(&optix_ctx, Some((&module, "__miss__ms"))).unwrap();
        let hg = OptixProgramGroup::new_hitgroup(
            &optix_ctx,
            Some((&module, "__closesthit__ch")),
            None,
            None,
        )
        .unwrap();

        let groups = [rg.clone(), ms.clone(), hg.clone()];
        let pipeline =
            OptixPipeline::new(&optix_ctx, &pipeline_opts, &link_opts, &groups, 2).unwrap();
        let sbt = OptixSbt::from_groups(&stream, &rg, &[&ms], &[&hg]).unwrap();

        // 3 test rays:
        // Ray 0: origin (0.0, 0.0, 0.0), dir (0.0, 0.0, 1.0) -> hits Triangle 0 at t = 2.0
        // Ray 1: origin (3.0, 0.0, 0.0), dir (0.0, 0.0, 1.0) -> hits Triangle 1 at t = 5.0
        // Ray 2: origin (0.0, 5.0, 0.0), dir (0.0, 0.0, 1.0) -> misses (-1, -1.0)
        let origins: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 0.0]];
        let dirs: [[f32; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];

        let d_origins = stream.clone_htod(&origins).unwrap();
        let d_dirs = stream.clone_htod(&dirs).unwrap();
        let d_out_prim = stream.alloc_zeros::<u32>(3).unwrap();
        let d_out_t = stream.alloc_zeros::<f32>(3).unwrap();

        let params = {
            let (orig_ptr, _s1) = d_origins.device_ptr(&stream);
            let (dir_ptr, _s2) = d_dirs.device_ptr(&stream);
            let (prim_ptr, _s3) = d_out_prim.device_ptr(&stream);
            let (t_ptr, _s4) = d_out_t.device_ptr(&stream);
            RayMeshParams {
                handle: tlas.handle(),
                rays_origin: orig_ptr,
                rays_dir: dir_ptr,
                out_prim: prim_ptr,
                out_t: t_ptr,
            }
        };
        let d_params = stream.clone_htod(&[params]).unwrap();

        pipeline.launch(&stream, &d_params, &sbt, 3, 1, 1).unwrap();
        stream.synchronize().unwrap();

        let out_prim = stream.clone_dtoh(&d_out_prim).unwrap();
        let out_t = stream.clone_dtoh(&d_out_t).unwrap();

        assert_eq!(out_prim, vec![0, 1, u32::MAX]);
        assert!((out_t[0] - 2.0).abs() < 1e-5, "out_t[0] = {}", out_t[0]);
        assert!((out_t[1] - 5.0).abs() < 1e-5, "out_t[1] = {}", out_t[1]);
        assert_eq!(out_t[2], -1.0);
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    struct RtCandidateParams {
        handle: u64,
        rays_origin_f32: u64,
        rays_dir_f32: u64,
        out_candidate_prim: u64,
    }
    unsafe impl DeviceRepr for RtCandidateParams {}

    /// Hybrid RT Core candidate traversal (`f32` hardware BVH via OptiX) + IEEE-754 `f64` exact
    /// Möller-Trumbore ray-triangle / closest-point refinement kernel in CUDA PTX on the same stream.
    #[test]
    fn test_optix_hybrid_rt_core_and_f64_moller_trumbore_refinement() {
        use crate::driver::{LaunchConfig, PushKernelArg};
        use crate::nvrtc::Ptx;

        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        // Exact double-precision plane z = 3.0 + 1e-11 * x (sub-ULP in f32, exact in f64)
        let verts_f64: [[f64; 3]; 3] = [
            [-1.0, -1.0, 3.0 - 1e-11],
            [1.0, -1.0, 3.0 + 1e-11],
            [0.0, 1.0, 3.0],
        ];
        let verts_f32: [[f32; 3]; 3] = verts_f64.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]);
        let indices: [[u32; 3]; 1] = [[0, 1, 2]];

        let d_verts_f32 = stream.clone_htod(&verts_f32).unwrap();
        let d_verts_f64 = stream.clone_htod(&verts_f64).unwrap();
        let d_indices = stream.clone_htod(&indices).unwrap();

        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_verts_f32,
            Some(&d_indices),
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();

        // Stage 1: OptiX PTX module for hardware RT Core candidate primitive query
        let optix_ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.const .align 8 .b8 params[32];

.extern .func (.reg .u32 $value) _optix_get_launch_index_x ();
.extern .func (.reg .u32 $value) _optix_read_primitive_idx ();
.extern .func () _optix_set_payload_0 (.reg .u32 $value);
.extern .func (.reg .u32 p0) _optix_trace_1 (
    .reg .u64 handle,
    .reg .f32 rayOriginX,
    .reg .f32 rayOriginY,
    .reg .f32 rayOriginZ,
    .reg .f32 rayDirectionX,
    .reg .f32 rayDirectionY,
    .reg .f32 rayDirectionZ,
    .reg .f32 tmin,
    .reg .f32 tmax,
    .reg .f32 rayTime,
    .reg .u32 visibilityMask,
    .reg .u32 rayFlags,
    .reg .u32 SBToffset,
    .reg .u32 SBTstride,
    .reg .u32 missSBTIndex,
    .reg .u32 p0
);

.visible .entry __raygen__rg()
{
    .reg .u32 %idx, %prim, %vis, %flags, %sbt_off, %sbt_stride, %miss_idx;
    .reg .u64 %handle, %orig_base, %dir_base, %out_prim_base, %off12, %off4, %orig_ptr, %dir_ptr, %out_prim_ptr;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %tmin, %tmax, %rtime;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %handle, [params + 0];
    ld.const.u64 %orig_base, [params + 8];
    ld.const.u64 %dir_base, [params + 16];
    ld.const.u64 %out_prim_base, [params + 24];

    mul.wide.u32 %off12, %idx, 12;
    add.u64 %orig_ptr, %orig_base, %off12;
    add.u64 %dir_ptr, %dir_base, %off12;

    ld.global.f32 %ox, [%orig_ptr + 0];
    ld.global.f32 %oy, [%orig_ptr + 4];
    ld.global.f32 %oz, [%orig_ptr + 8];

    ld.global.f32 %dx, [%dir_ptr + 0];
    ld.global.f32 %dy, [%dir_ptr + 4];
    ld.global.f32 %dz, [%dir_ptr + 8];

    mov.f32 %tmin, 0f00000000;
    mov.f32 %tmax, 0f461C4000;
    mov.f32 %rtime, 0f00000000;
    mov.u32 %vis, 255;
    mov.u32 %flags, 0;
    mov.u32 %sbt_off, 0;
    mov.u32 %sbt_stride, 1;
    mov.u32 %miss_idx, 0;
    mov.u32 %prim, -1;

    call (%prim), _optix_trace_1, (
        %handle,
        %ox, %oy, %oz,
        %dx, %dy, %dz,
        %tmin, %tmax, %rtime,
        %vis, %flags, %sbt_off, %sbt_stride, %miss_idx,
        %prim
    );

    mul.wide.u32 %off4, %idx, 4;
    add.u64 %out_prim_ptr, %out_prim_base, %off4;
    st.global.u32 [%out_prim_ptr], %prim;
    ret;
}

.visible .entry __closesthit__ch()
{
    .reg .u32 %prim;
    call (%prim), _optix_read_primitive_idx, ();
    call (), _optix_set_payload_0, (%prim);
    ret;
}

.visible .entry __miss__ms()
{
    .reg .u32 %none;
    mov.u32 %none, -1;
    call (), _optix_set_payload_0, (%none);
    ret;
}
"#;

        // Stage 2: CUDA PTX kernel that takes the RT Core candidate primitive index and
        // performs IEEE-754 f64 exact Möller-Trumbore intersection + hit-point refinement.
        let cuda_f64_ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry refine_moller_trumbore_f64(
    .param .u64 p_verts_f64,
    .param .u64 p_indices_u32,
    .param .u64 p_origins_f64,
    .param .u64 p_dirs_f64,
    .param .u64 p_cand_prim,
    .param .u64 p_out_t_f64,
    .param .u64 p_out_hit_f64,
    .param .u32 p_num_rays
)
{
    .reg .pred %p_oob, %p_miss;
    .reg .u32 %idx, %r_tid, %r_ctaid, %r_ntid, %n, %prim, %i0, %i1, %i2;
    .reg .u64 %verts_base, %idx_base, %orig_base, %dir_base, %cand_base, %out_t_base, %out_hit_base;
    .reg .u64 %off24, %off12, %off8, %off4, %orig_ptr, %dir_ptr, %cand_ptr, %tri_ptr, %v0_ptr, %v1_ptr, %v2_ptr, %out_t_ptr, %out_hit_ptr;
    .reg .f64 %ox, %oy, %oz, %dx, %dy, %dz;
    .reg .f64 %v0x, %v0y, %v0z, %v1x, %v1y, %v1z, %v2x, %v2y, %v2z;
    .reg .f64 %e1x, %e1y, %e1z, %e2x, %e2y, %e2z;
    .reg .f64 %hx, %hy, %hz, %a, %f, %one, %sx, %sy, %sz, %qx, %qy, %qz, %t, %tmp;
    .reg .f64 %px, %py, %pz;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mad.lo.u32 %idx, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %n, [p_num_rays];
    setp.ge.u32 %p_oob, %idx, %n;
    @%p_oob ret;

    ld.param.u64 %verts_base, [p_verts_f64];
    ld.param.u64 %idx_base, [p_indices_u32];
    ld.param.u64 %orig_base, [p_origins_f64];
    ld.param.u64 %dir_base, [p_dirs_f64];
    ld.param.u64 %cand_base, [p_cand_prim];
    ld.param.u64 %out_t_base, [p_out_t_f64];
    ld.param.u64 %out_hit_base, [p_out_hit_f64];

    mul.wide.u32 %off4, %idx, 4;
    add.u64 %cand_ptr, %cand_base, %off4;
    ld.global.u32 %prim, [%cand_ptr];

    mul.wide.u32 %off8, %idx, 8;
    add.u64 %out_t_ptr, %out_t_base, %off8;

    mul.wide.u32 %off24, %idx, 24;
    add.u64 %out_hit_ptr, %out_hit_base, %off24;

    setp.eq.u32 %p_miss, %prim, 0xffffffff;
    @%p_miss bra L_MISS;

    add.u64 %orig_ptr, %orig_base, %off24;
    add.u64 %dir_ptr, %dir_base, %off24;

    ld.global.f64 %ox, [%orig_ptr + 0];
    ld.global.f64 %oy, [%orig_ptr + 8];
    ld.global.f64 %oz, [%orig_ptr + 16];

    ld.global.f64 %dx, [%dir_ptr + 0];
    ld.global.f64 %dy, [%dir_ptr + 8];
    ld.global.f64 %dz, [%dir_ptr + 16];

    mul.wide.u32 %off12, %prim, 12;
    add.u64 %tri_ptr, %idx_base, %off12;
    ld.global.u32 %i0, [%tri_ptr + 0];
    ld.global.u32 %i1, [%tri_ptr + 4];
    ld.global.u32 %i2, [%tri_ptr + 8];

    mul.wide.u32 %off24, %i0, 24;
    add.u64 %v0_ptr, %verts_base, %off24;
    mul.wide.u32 %off24, %i1, 24;
    add.u64 %v1_ptr, %verts_base, %off24;
    mul.wide.u32 %off24, %i2, 24;
    add.u64 %v2_ptr, %verts_base, %off24;

    ld.global.f64 %v0x, [%v0_ptr + 0];
    ld.global.f64 %v0y, [%v0_ptr + 8];
    ld.global.f64 %v0z, [%v0_ptr + 16];

    ld.global.f64 %v1x, [%v1_ptr + 0];
    ld.global.f64 %v1y, [%v1_ptr + 8];
    ld.global.f64 %v1z, [%v1_ptr + 16];

    ld.global.f64 %v2x, [%v2_ptr + 0];
    ld.global.f64 %v2y, [%v2_ptr + 8];
    ld.global.f64 %v2z, [%v2_ptr + 16];

    sub.rn.f64 %e1x, %v1x, %v0x;
    sub.rn.f64 %e1y, %v1y, %v0y;
    sub.rn.f64 %e1z, %v1z, %v0z;

    sub.rn.f64 %e2x, %v2x, %v0x;
    sub.rn.f64 %e2y, %v2y, %v0y;
    sub.rn.f64 %e2z, %v2z, %v0z;

    // h = cross(d, e2)
    mul.rn.f64 %tmp, %dz, %e2y;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %hx, %dy, %e2z, %tmp;

    mul.rn.f64 %tmp, %dx, %e2z;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %hy, %dz, %e2x, %tmp;

    mul.rn.f64 %tmp, %dy, %e2x;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %hz, %dx, %e2y, %tmp;

    // a = dot(e1, h)
    mul.rn.f64 %a, %e1x, %hx;
    fma.rn.f64 %a, %e1y, %hy, %a;
    fma.rn.f64 %a, %e1z, %hz, %a;

    mov.f64 %one, 0d3FF0000000000000;
    div.rn.f64 %f, %one, %a;

    // s = o - v0
    sub.rn.f64 %sx, %ox, %v0x;
    sub.rn.f64 %sy, %oy, %v0y;
    sub.rn.f64 %sz, %oz, %v0z;

    // q = cross(s, e1)
    mul.rn.f64 %tmp, %sz, %e1y;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %qx, %sy, %e1z, %tmp;

    mul.rn.f64 %tmp, %sx, %e1z;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %qy, %sz, %e1x, %tmp;

    mul.rn.f64 %tmp, %sy, %e1x;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %qz, %sx, %e1y, %tmp;

    // t = f * dot(e2, q)
    mul.rn.f64 %t, %e2x, %qx;
    fma.rn.f64 %t, %e2y, %qy, %t;
    fma.rn.f64 %t, %e2z, %qz, %t;
    mul.rn.f64 %t, %f, %t;

    // Exact hit point P = O + t * D
    fma.rn.f64 %px, %t, %dx, %ox;
    fma.rn.f64 %py, %t, %dy, %oy;
    fma.rn.f64 %pz, %t, %dz, %oz;

    st.global.f64 [%out_t_ptr], %t;
    st.global.f64 [%out_hit_ptr + 0], %px;
    st.global.f64 [%out_hit_ptr + 8], %py;
    st.global.f64 [%out_hit_ptr + 16], %pz;
    ret;

L_MISS:
    mov.f64 %t, 0dBFF0000000000000;
    st.global.f64 [%out_t_ptr], %t;
    ret;
}
"#;

        let module_opts = sys::OptixModuleCompileOptions::default();
        let pipeline_opts = sys::OptixPipelineCompileOptions {
            numPayloadValues: 1,
            numAttributeValues: 2,
            ..Default::default()
        };
        let link_opts = sys::OptixPipelineLinkOptions::default();

        let module =
            OptixModule::from_ptx(&optix_ctx, optix_ptx, &module_opts, &pipeline_opts).unwrap();
        let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__rg").unwrap();
        let ms = OptixProgramGroup::new_miss(&optix_ctx, Some((&module, "__miss__ms"))).unwrap();
        let hg = OptixProgramGroup::new_hitgroup(
            &optix_ctx,
            Some((&module, "__closesthit__ch")),
            None,
            None,
        )
        .unwrap();

        let groups = [rg.clone(), ms.clone(), hg.clone()];
        let pipeline =
            OptixPipeline::new(&optix_ctx, &pipeline_opts, &link_opts, &groups, 1).unwrap();
        let sbt = OptixSbt::from_groups(&stream, &rg, &[&ms], &[&hg]).unwrap();

        // Ray at x = 0.25, y = -0.25, z = 0.0 pointing +Z:
        // Exact analytical intersection with plane z = 3.0 + 1e-11 * x is t = 3.0 + 0.25e-11
        let origins_f64: [[f64; 3]; 1] = [[0.25, -0.25, 0.0]];
        let dirs_f64: [[f64; 3]; 1] = [[0.0, 0.0, 1.0]];
        let origins_f32: [[f32; 3]; 1] =
            origins_f64.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]);
        let dirs_f32: [[f32; 3]; 1] = dirs_f64.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]);

        let d_origins_f32 = stream.clone_htod(&origins_f32).unwrap();
        let d_dirs_f32 = stream.clone_htod(&dirs_f32).unwrap();
        let d_origins_f64 = stream.clone_htod(&origins_f64).unwrap();
        let d_dirs_f64 = stream.clone_htod(&dirs_f64).unwrap();
        let d_cand_prim = stream.alloc_zeros::<u32>(1).unwrap();
        let mut d_out_t_f64 = stream.alloc_zeros::<f64>(1).unwrap();
        let mut d_out_hit_f64 = stream.alloc_zeros::<[f64; 3]>(1).unwrap();

        let params = {
            let (o_ptr, _s1) = d_origins_f32.device_ptr(&stream);
            let (d_ptr, _s2) = d_dirs_f32.device_ptr(&stream);
            let (p_ptr, _s3) = d_cand_prim.device_ptr(&stream);
            RtCandidateParams {
                handle: blas.handle(),
                rays_origin_f32: o_ptr,
                rays_dir_f32: d_ptr,
                out_candidate_prim: p_ptr,
            }
        };
        let d_params = stream.clone_htod(&[params]).unwrap();

        // 1. Hardware RT Core traversal on stream
        pipeline.launch(&stream, &d_params, &sbt, 1, 1, 1).unwrap();

        // 2. Exact IEEE-754 f64 Möller-Trumbore refinement on the same stream
        let cuda_mod = cuda_ctx.load_module(Ptx::from_src(cuda_f64_ptx)).unwrap();
        let refine_fn = cuda_mod
            .load_function("refine_moller_trumbore_f64")
            .unwrap();
        let num_rays = 1u32;
        let mut builder = stream.launch_builder(&refine_fn);
        builder.arg(&d_verts_f64);
        builder.arg(&d_indices);
        builder.arg(&d_origins_f64);
        builder.arg(&d_dirs_f64);
        builder.arg(&d_cand_prim);
        builder.arg(&mut d_out_t_f64);
        builder.arg(&mut d_out_hit_f64);
        builder.arg(&num_rays);
        unsafe { builder.launch(LaunchConfig::for_num_elems(1)) }.unwrap();

        stream.synchronize().unwrap();

        let out_prim = stream.clone_dtoh(&d_cand_prim).unwrap();
        let out_t_f64 = stream.clone_dtoh(&d_out_t_f64).unwrap();
        let out_hit_f64 = stream.clone_dtoh(&d_out_hit_f64).unwrap();

        let expected_t_f64: f64 = 3.0 + 0.25e-11;
        assert_eq!(out_prim[0], 0);
        assert!(
            (out_t_f64[0] - expected_t_f64).abs() < 1e-15,
            "got {}, expected {}",
            out_t_f64[0],
            expected_t_f64
        );
        assert_eq!(out_hit_f64[0][0], 0.25);
        assert_eq!(out_hit_f64[0][1], -0.25);
        assert!((out_hit_f64[0][2] - expected_t_f64).abs() < 1e-15);
    }

    #[test]
    fn test_optix_ffi_abi_layout_and_error_formatting() {
        assert_eq!(
            core::mem::size_of::<sys::OptixFunctionTable>(),
            38 * core::mem::size_of::<usize>()
        );
        assert_eq!(core::mem::size_of::<sys::OptixInstance>(), 80);
        assert_eq!(
            core::mem::align_of::<sys::OptixInstance>(),
            sys::OPTIX_INSTANCE_BYTE_ALIGNMENT
        );
        assert_eq!(core::mem::size_of::<sys::OptixAabb>(), 24);
        assert_eq!(
            core::mem::align_of::<sys::OptixAabb>(),
            sys::OPTIX_AABB_BUFFER_BYTE_ALIGNMENT
        );
        assert_eq!(core::mem::size_of::<sys::OptixShaderBindingTable>(), 64);
        assert_eq!(core::mem::size_of::<sys::OptixAccelBufferSizes>(), 24);
        assert_eq!(core::mem::size_of::<sys::OptixStackSizes>(), 28);
        assert_eq!(
            core::mem::size_of::<SbtRecord<[u8; 0]>>(),
            sys::OPTIX_SBT_RECORD_HEADER_SIZE
        );
        assert_eq!(
            core::mem::align_of::<SbtRecord<[u8; 0]>>(),
            sys::OPTIX_SBT_RECORD_ALIGNMENT
        );
        assert_eq!(core::mem::size_of::<SbtRecord<u32>>(), 48);
        assert_eq!(
            core::mem::align_of::<SbtRecord<u32>>(),
            sys::OPTIX_SBT_RECORD_ALIGNMENT
        );
        assert_eq!(sys::OPTIX_SBT_RECORD_HEADER_SIZE, 32);
        assert_eq!(sys::OPTIX_SBT_RECORD_ALIGNMENT, 16);
        assert_eq!(sys::OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT, 128);
        assert_eq!(sys::OPTIX_INSTANCE_BYTE_ALIGNMENT, 16);
        assert_eq!(sys::OPTIX_AABB_BUFFER_BYTE_ALIGNMENT, 8);

        let err = OptixError(sys::OptixResult::OPTIX_ERROR_INVALID_VALUE);
        assert!(!err.error_name().to_bytes().is_empty());
        assert!(!err.error_string().to_bytes().is_empty());
        let display_str = std::format!("{err}");
        let debug_str = std::format!("{err:?}");
        assert!(display_str.contains("OPTIX_ERROR"));
        assert!(debug_str.contains("OptixError"));

        let from_driver: OptixError =
            crate::driver::DriverError(crate::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE)
                .into();
        assert_eq!(from_driver.0, sys::OptixResult::OPTIX_ERROR_CUDA_ERROR);
    }

    #[test]
    fn test_optix_device_properties_and_unindexed_triangles() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        let query_u32 = |prop: sys::OptixDeviceProperty| -> u32 {
            let mut val: c_uint = 0;
            unsafe {
                result::device_context_get_property(
                    optix_ctx.raw(),
                    prop,
                    (&mut val as *mut c_uint).cast::<c_void>(),
                    core::mem::size_of::<c_uint>(),
                )
                .unwrap();
            }
            val
        };

        let max_trace_depth =
            query_u32(sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH);
        let max_graph_depth = query_u32(
            sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
        );
        let max_prims =
            query_u32(sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS);
        let max_instances =
            query_u32(sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS);
        let max_instance_id =
            query_u32(sys::OptixDeviceProperty::OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID);

        assert!(max_trace_depth >= 31);
        assert!(max_graph_depth >= 2);
        assert!(max_prims >= (1 << 20));
        assert!(max_instances >= (1 << 16));
        assert!(max_instance_id >= (1 << 20));

        // Build unindexed triangle BLAS (2 triangles = 6 vertices, indices = None)
        let unindexed_verts: [[f32; 3]; 6] = [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            [-1.0, -1.0, 4.0],
            [1.0, -1.0, 4.0],
            [0.0, 1.0, 4.0],
        ];
        let d_verts = stream.clone_htod(&unindexed_verts).unwrap();
        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_verts,
            None,
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();
        assert_ne!(blas.handle(), 0);
        assert!(!blas.output_buffer().is_empty());
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_optix_invalid_ptx_and_error_handling() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();
        let module_opts = sys::OptixModuleCompileOptions::default();
        let pipeline_opts = sys::OptixPipelineCompileOptions::default();

        let bad_mod = OptixModule::from_ptx(
            &optix_ctx,
            "not valid ptx at all",
            &module_opts,
            &pipeline_opts,
        );
        assert!(bad_mod.is_err());
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    struct MultiInstanceParams {
        handle: u64,
        rays_origin: u64,
        rays_dir: u64,
        out_inst_id: u64,
        out_prim_id: u64,
        out_mat_id: u64,
        out_t: u64,
        out_uv: u64,
    }
    unsafe impl DeviceRepr for MultiInstanceParams {}

    /// Tests multi-instance TLAS with distinct 3x4 affine transforms, custom `instanceId`s,
    /// per-instance `sbtOffset`s, custom `SbtRecord<u32>` payload data read via
    /// `_optix_get_sbt_data_ptr_64`, and hardware triangle barycentrics (`_optix_get_triangle_barycentrics`).
    #[test]
    fn test_optix_multi_instance_tlas_transforms_barycentrics_and_sbt_data() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        // Canonical unit triangle in XY plane at z = 0:
        // V0 = (0, 0, 0), V1 = (1, 0, 0), V2 = (0, 1, 0)
        // A point on the triangle with barycentrics (u, v) is (u, v, 0).
        let vertices: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices: [[u32; 3]; 1] = [[0, 1, 2]];
        let d_vertices = stream.clone_htod(&vertices).unwrap();
        let d_indices = stream.clone_htod(&indices).unwrap();

        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_vertices,
            Some(&d_indices),
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();

        // Instance 0: id = 100, sbtOffset = 0, translated to (0, 0, 2.5)
        let mut inst0 = sys::OptixInstance::with_identity(blas.handle(), 100);
        inst0.sbtOffset = 0;
        inst0.transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.5];

        // Instance 1: id = 200, sbtOffset = 1, scaled by 2x in X/Y and translated to (0, 0, 6.0)
        let mut inst1 = sys::OptixInstance::with_identity(blas.handle(), 200);
        inst1.sbtOffset = 1;
        inst1.transform = [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0];

        // Instance 2: id = 300, sbtOffset = 2, translated to (-4.0, -4.0, 3.5)
        let mut inst2 = sys::OptixInstance::with_identity(blas.handle(), 300);
        inst2.sbtOffset = 2;
        inst2.transform = [1.0, 0.0, 0.0, -4.0, 0.0, 1.0, 0.0, -4.0, 0.0, 0.0, 1.0, 3.5];

        let d_instances = stream.clone_htod(&[inst0, inst1, inst2]).unwrap();
        let tlas = OptixAccel::build_instances(&optix_ctx, &stream, &d_instances, None).unwrap();

        let ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.const .align 8 .b8 params[64];

.extern .func (.reg .u32 $value) _optix_get_launch_index_x ();
.extern .func (.reg .u32 $value) _optix_read_instance_id ();
.extern .func (.reg .u32 $value) _optix_read_primitive_idx ();
.extern .func (.reg .f32 $value) _optix_get_ray_tmax ();
.extern .func (.reg .u64 $value) _optix_get_sbt_data_ptr_64 ();
.extern .func (.reg .f32 $uValue, .reg .f32 $vValue) _optix_get_triangle_barycentrics ();
.extern .func () _optix_trace_0 (
    .reg .u64 handle,
    .reg .f32 rayOriginX,
    .reg .f32 rayOriginY,
    .reg .f32 rayOriginZ,
    .reg .f32 rayDirectionX,
    .reg .f32 rayDirectionY,
    .reg .f32 rayDirectionZ,
    .reg .f32 tmin,
    .reg .f32 tmax,
    .reg .f32 rayTime,
    .reg .u32 visibilityMask,
    .reg .u32 rayFlags,
    .reg .u32 SBToffset,
    .reg .u32 SBTstride,
    .reg .u32 missSBTIndex
);

.visible .entry __raygen__rg()
{
    .reg .u32 %idx, %vis, %flags, %sbt_off, %sbt_stride, %miss_idx;
    .reg .u64 %handle, %orig_base, %dir_base, %off12, %orig_ptr, %dir_ptr;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %tmin, %tmax, %rtime;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %handle, [params + 0];
    ld.const.u64 %orig_base, [params + 8];
    ld.const.u64 %dir_base, [params + 16];

    mul.wide.u32 %off12, %idx, 12;
    add.u64 %orig_ptr, %orig_base, %off12;
    add.u64 %dir_ptr, %dir_base, %off12;

    ld.global.f32 %ox, [%orig_ptr + 0];
    ld.global.f32 %oy, [%orig_ptr + 4];
    ld.global.f32 %oz, [%orig_ptr + 8];

    ld.global.f32 %dx, [%dir_ptr + 0];
    ld.global.f32 %dy, [%dir_ptr + 4];
    ld.global.f32 %dz, [%dir_ptr + 8];

    mov.f32 %tmin, 0f00000000;
    mov.f32 %tmax, 0f461C4000;
    mov.f32 %rtime, 0f00000000;
    mov.u32 %vis, 255;
    mov.u32 %flags, 0;
    mov.u32 %sbt_off, 0;
    mov.u32 %sbt_stride, 1;
    mov.u32 %miss_idx, 0;

    call (), _optix_trace_0, (
        %handle,
        %ox, %oy, %oz,
        %dx, %dy, %dz,
        %tmin, %tmax, %rtime,
        %vis, %flags, %sbt_off, %sbt_stride, %miss_idx
    );
    ret;
}

.visible .entry __closesthit__ch()
{
    .reg .u32 %idx, %inst_id, %prim_id, %mat_id;
    .reg .f32 %t, %u, %v;
    .reg .u64 %sbt_ptr, %inst_base, %prim_base, %mat_base, %t_base, %uv_base;
    .reg .u64 %off4, %off8, %p_inst, %p_prim, %p_mat, %p_t, %p_uv;

    call (%idx), _optix_get_launch_index_x, ();
    call (%inst_id), _optix_read_instance_id, ();
    call (%prim_id), _optix_read_primitive_idx, ();
    call (%t), _optix_get_ray_tmax, ();
    call (%u, %v), _optix_get_triangle_barycentrics, ();
    call (%sbt_ptr), _optix_get_sbt_data_ptr_64, ();
    ld.global.u32 %mat_id, [%sbt_ptr];

    ld.const.u64 %inst_base, [params + 24];
    ld.const.u64 %prim_base, [params + 32];
    ld.const.u64 %mat_base, [params + 40];
    ld.const.u64 %t_base, [params + 48];
    ld.const.u64 %uv_base, [params + 56];

    mul.wide.u32 %off4, %idx, 4;
    mul.wide.u32 %off8, %idx, 8;
    add.u64 %p_inst, %inst_base, %off4;
    add.u64 %p_prim, %prim_base, %off4;
    add.u64 %p_mat, %mat_base, %off4;
    add.u64 %p_t, %t_base, %off4;
    add.u64 %p_uv, %uv_base, %off8;

    st.global.u32 [%p_inst], %inst_id;
    st.global.u32 [%p_prim], %prim_id;
    st.global.u32 [%p_mat], %mat_id;
    st.global.f32 [%p_t], %t;
    st.global.f32 [%p_uv + 0], %u;
    st.global.f32 [%p_uv + 4], %v;
    ret;
}

.visible .entry __miss__ms()
{
    .reg .u32 %idx, %none;
    .reg .f32 %neg;
    .reg .u64 %inst_base, %prim_base, %mat_base, %t_base, %uv_base;
    .reg .u64 %off4, %off8, %p_inst, %p_prim, %p_mat, %p_t, %p_uv;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %inst_base, [params + 24];
    ld.const.u64 %prim_base, [params + 32];
    ld.const.u64 %mat_base, [params + 40];
    ld.const.u64 %t_base, [params + 48];
    ld.const.u64 %uv_base, [params + 56];

    mul.wide.u32 %off4, %idx, 4;
    mul.wide.u32 %off8, %idx, 8;
    add.u64 %p_inst, %inst_base, %off4;
    add.u64 %p_prim, %prim_base, %off4;
    add.u64 %p_mat, %mat_base, %off4;
    add.u64 %p_t, %t_base, %off4;
    add.u64 %p_uv, %uv_base, %off8;

    mov.u32 %none, -1;
    mov.f32 %neg, 0fBF800000;
    st.global.u32 [%p_inst], %none;
    st.global.u32 [%p_prim], %none;
    st.global.u32 [%p_mat], %none;
    st.global.f32 [%p_t], %neg;
    st.global.f32 [%p_uv + 0], %neg;
    st.global.f32 [%p_uv + 4], %neg;
    ret;
}
"#;

        let module_opts = sys::OptixModuleCompileOptions::default();
        let pipeline_opts = sys::OptixPipelineCompileOptions {
            numPayloadValues: 0,
            numAttributeValues: 2,
            ..Default::default()
        };
        let link_opts = sys::OptixPipelineLinkOptions::default();

        let module = OptixModule::from_ptx(&optix_ctx, ptx, &module_opts, &pipeline_opts).unwrap();
        let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__rg").unwrap();
        let ms = OptixProgramGroup::new_miss(&optix_ctx, Some((&module, "__miss__ms"))).unwrap();
        let hg = OptixProgramGroup::new_hitgroup(
            &optix_ctx,
            Some((&module, "__closesthit__ch")),
            None,
            None,
        )
        .unwrap();

        let groups = [rg.clone(), ms.clone(), hg.clone()];
        let pipeline =
            OptixPipeline::new(&optix_ctx, &pipeline_opts, &link_opts, &groups, 2).unwrap();

        // Pack 3 distinct HitGroup SBT records with material IDs 1111, 2222, 3333
        let rg_rec = SbtRecord::pack(&rg).unwrap();
        let ms_rec = SbtRecord::pack(&ms).unwrap();
        let hg_recs = [
            SbtRecord::pack_with_data(&hg, 1111u32).unwrap(),
            SbtRecord::pack_with_data(&hg, 2222u32).unwrap(),
            SbtRecord::pack_with_data(&hg, 3333u32).unwrap(),
        ];
        let sbt = OptixSbt::from_records(&stream, &rg_rec, &[ms_rec], &hg_recs).unwrap();

        let origins: [[f32; 3]; 4] = [
            [0.2, 0.3, 0.0],
            [0.8, 0.8, 0.0],
            [-3.75, -3.875, 0.0],
            [10.0, 10.0, 0.0],
        ];
        let dirs: [[f32; 3]; 4] = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];

        let d_origins = stream.clone_htod(&origins).unwrap();
        let d_dirs = stream.clone_htod(&dirs).unwrap();
        let d_out_inst = stream.alloc_zeros::<u32>(4).unwrap();
        let d_out_prim = stream.alloc_zeros::<u32>(4).unwrap();
        let d_out_mat = stream.alloc_zeros::<u32>(4).unwrap();
        let d_out_t = stream.alloc_zeros::<f32>(4).unwrap();
        let d_out_uv = stream.alloc_zeros::<[f32; 2]>(4).unwrap();

        let params = {
            let (orig_ptr, _s1) = d_origins.device_ptr(&stream);
            let (dir_ptr, _s2) = d_dirs.device_ptr(&stream);
            let (inst_ptr, _s3) = d_out_inst.device_ptr(&stream);
            let (prim_ptr, _s4) = d_out_prim.device_ptr(&stream);
            let (mat_ptr, _s5) = d_out_mat.device_ptr(&stream);
            let (t_ptr, _s6) = d_out_t.device_ptr(&stream);
            let (uv_ptr, _s7) = d_out_uv.device_ptr(&stream);
            MultiInstanceParams {
                handle: tlas.handle(),
                rays_origin: orig_ptr,
                rays_dir: dir_ptr,
                out_inst_id: inst_ptr,
                out_prim_id: prim_ptr,
                out_mat_id: mat_ptr,
                out_t: t_ptr,
                out_uv: uv_ptr,
            }
        };
        let d_params = stream.clone_htod(&[params]).unwrap();

        pipeline.launch(&stream, &d_params, &sbt, 4, 1, 1).unwrap();
        stream.synchronize().unwrap();

        let out_inst = stream.clone_dtoh(&d_out_inst).unwrap();
        let out_prim = stream.clone_dtoh(&d_out_prim).unwrap();
        let out_mat = stream.clone_dtoh(&d_out_mat).unwrap();
        let out_t = stream.clone_dtoh(&d_out_t).unwrap();
        let out_uv = stream.clone_dtoh(&d_out_uv).unwrap();

        assert_eq!(out_inst, vec![100, 200, 300, u32::MAX]);
        assert_eq!(out_prim, vec![0, 0, 0, u32::MAX]);
        assert_eq!(out_mat, vec![1111, 2222, 3333, u32::MAX]);
        assert!((out_t[0] - 2.5).abs() < 1e-5);
        assert!((out_t[1] - 6.0).abs() < 1e-5);
        assert!((out_t[2] - 3.5).abs() < 1e-5);
        assert_eq!(out_t[3], -1.0);

        assert!((out_uv[0][0] - 0.2).abs() < 1e-5);
        assert!((out_uv[0][1] - 0.3).abs() < 1e-5);
        assert!((out_uv[1][0] - 0.4).abs() < 1e-5);
        assert!((out_uv[1][1] - 0.4).abs() < 1e-5);
        assert!((out_uv[2][0] - 0.25).abs() < 1e-5);
        assert!((out_uv[2][1] - 0.125).abs() < 1e-5);
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    struct CustomSphereParams {
        handle: u64,
        spheres: u64,
        rays_origin: u64,
        rays_dir: u64,
        out_prim: u64,
        out_t: u64,
        out_nz: u64,
    }
    unsafe impl DeviceRepr for CustomSphereParams {}

    /// Tests custom primitive AABB BLAS with an analytic sphere `__intersection__` shader
    /// that solves $|O + t D - C|^2 = R^2$ and reports hit distance + surface normal attribute
    /// via `_optix_report_intersection_1`, including AABB corner false-positive rejection.
    #[test]
    fn test_optix_custom_aabb_analytic_sphere_intersection() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        // Two spheres packed as [cx, cy, cz, radius]:
        // Sphere 0: center (0, 0, 4), r = 1.0
        // Sphere 1: center (3, 0, 7), r = 1.5
        let spheres: [[f32; 4]; 2] = [[0.0, 0.0, 4.0, 1.0], [3.0, 0.0, 7.0, 1.5]];
        let aabbs = [
            sys::OptixAabb::new([-1.0, -1.0, 3.0], [1.0, 1.0, 5.0]),
            sys::OptixAabb::new([1.5, -1.5, 5.5], [4.5, 1.5, 8.5]),
        ];

        let d_spheres = stream.clone_htod(&spheres).unwrap();
        let d_aabbs = stream.clone_htod(&aabbs).unwrap();

        let blas = OptixAccel::build_aabbs(
            &optix_ctx,
            &stream,
            &d_aabbs,
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();

        let ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.const .align 8 .b8 params[56];

.extern .func (.reg .u32 $value) _optix_get_launch_index_x ();
.extern .func (.reg .u32 $value) _optix_read_primitive_idx ();
.extern .func (.reg .f32 $value) _optix_get_ray_tmin ();
.extern .func (.reg .f32 $value) _optix_get_ray_tmax ();
.extern .func (.reg .f32 $value) _optix_get_object_ray_origin_x ();
.extern .func (.reg .f32 $value) _optix_get_object_ray_origin_y ();
.extern .func (.reg .f32 $value) _optix_get_object_ray_origin_z ();
.extern .func (.reg .f32 $value) _optix_get_object_ray_direction_x ();
.extern .func (.reg .f32 $value) _optix_get_object_ray_direction_y ();
.extern .func (.reg .f32 $value) _optix_get_object_ray_direction_z ();
.extern .func (.reg .u32 $value) _optix_get_attribute_0 ();
.extern .func (.reg .u32 $value) _optix_report_intersection_1 (.reg .f32 $hitT, .reg .u32 $hitKind, .reg .u32 $a0);
.extern .func () _optix_trace_0 (
    .reg .u64 handle,
    .reg .f32 rayOriginX,
    .reg .f32 rayOriginY,
    .reg .f32 rayOriginZ,
    .reg .f32 rayDirectionX,
    .reg .f32 rayDirectionY,
    .reg .f32 rayDirectionZ,
    .reg .f32 tmin,
    .reg .f32 tmax,
    .reg .f32 rayTime,
    .reg .u32 visibilityMask,
    .reg .u32 rayFlags,
    .reg .u32 SBToffset,
    .reg .u32 SBTstride,
    .reg .u32 missSBTIndex
);

.visible .entry __raygen__rg()
{
    .reg .u32 %idx, %vis, %flags, %sbt_off, %sbt_stride, %miss_idx;
    .reg .u64 %handle, %orig_base, %dir_base, %off12, %orig_ptr, %dir_ptr;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %tmin, %tmax, %rtime;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %handle, [params + 0];
    ld.const.u64 %orig_base, [params + 16];
    ld.const.u64 %dir_base, [params + 24];

    mul.wide.u32 %off12, %idx, 12;
    add.u64 %orig_ptr, %orig_base, %off12;
    add.u64 %dir_ptr, %dir_base, %off12;

    ld.global.f32 %ox, [%orig_ptr + 0];
    ld.global.f32 %oy, [%orig_ptr + 4];
    ld.global.f32 %oz, [%orig_ptr + 8];

    ld.global.f32 %dx, [%dir_ptr + 0];
    ld.global.f32 %dy, [%dir_ptr + 4];
    ld.global.f32 %dz, [%dir_ptr + 8];

    mov.f32 %tmin, 0f00000000;
    mov.f32 %tmax, 0f461C4000;
    mov.f32 %rtime, 0f00000000;
    mov.u32 %vis, 255;
    mov.u32 %flags, 0;
    mov.u32 %sbt_off, 0;
    mov.u32 %sbt_stride, 1;
    mov.u32 %miss_idx, 0;

    call (), _optix_trace_0, (
        %handle,
        %ox, %oy, %oz,
        %dx, %dy, %dz,
        %tmin, %tmax, %rtime,
        %vis, %flags, %sbt_off, %sbt_stride, %miss_idx
    );
    ret;
}

.visible .entry __intersection__sphere()
{
    .reg .pred %p_miss, %p_tmin, %p_tmax;
    .reg .u32 %prim, %kind, %attr0, %reported;
    .reg .u64 %spheres_base, %off16, %sph_ptr;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %tmin, %tmax;
    .reg .f32 %cx, %cy, %cz, %rad, %ocx, %ocy, %ocz;
    .reg .f32 %b, %c, %disc, %sqrt_disc, %t, %hz, %nz;

    call (%prim), _optix_read_primitive_idx, ();
    call (%ox), _optix_get_object_ray_origin_x, ();
    call (%oy), _optix_get_object_ray_origin_y, ();
    call (%oz), _optix_get_object_ray_origin_z, ();
    call (%dx), _optix_get_object_ray_direction_x, ();
    call (%dy), _optix_get_object_ray_direction_y, ();
    call (%dz), _optix_get_object_ray_direction_z, ();
    call (%tmin), _optix_get_ray_tmin, ();
    call (%tmax), _optix_get_ray_tmax, ();

    ld.const.u64 %spheres_base, [params + 8];
    mul.wide.u32 %off16, %prim, 16;
    add.u64 %sph_ptr, %spheres_base, %off16;

    ld.global.f32 %cx, [%sph_ptr + 0];
    ld.global.f32 %cy, [%sph_ptr + 4];
    ld.global.f32 %cz, [%sph_ptr + 8];
    ld.global.f32 %rad, [%sph_ptr + 12];

    sub.f32 %ocx, %ox, %cx;
    sub.f32 %ocy, %oy, %cy;
    sub.f32 %ocz, %oz, %cz;

    // b = dot(oc, d) (assuming unit d)
    mul.f32 %b, %ocx, %dx;
    fma.rn.f32 %b, %ocy, %dy, %b;
    fma.rn.f32 %b, %ocz, %dz, %b;

    // c = dot(oc, oc) - rad*rad
    mul.f32 %c, %ocx, %ocx;
    fma.rn.f32 %c, %ocy, %ocy, %c;
    fma.rn.f32 %c, %ocz, %ocz, %c;
    neg.f32 %rad, %rad;
    fma.rn.f32 %c, %rad, %rad, %c;
    // Wait: (-r)*(-r) is +r^2; subtract r^2:
    ld.global.f32 %rad, [%sph_ptr + 12];
    mul.f32 %c, %ocx, %ocx;
    fma.rn.f32 %c, %ocy, %ocy, %c;
    fma.rn.f32 %c, %ocz, %ocz, %c;
    mul.f32 %disc, %rad, %rad;
    sub.f32 %c, %c, %disc;

    // disc = b*b - c
    mul.f32 %disc, %b, %b;
    sub.f32 %disc, %disc, %c;

    setp.lt.f32 %p_miss, %disc, 0f00000000;
    @%p_miss ret;

    sqrt.rn.f32 %sqrt_disc, %disc;
    neg.f32 %t, %b;
    sub.f32 %t, %t, %sqrt_disc;

    setp.lt.f32 %p_tmin, %t, %tmin;
    setp.gt.f32 %p_tmax, %t, %tmax;
    or.pred %p_miss, %p_tmin, %p_tmax;
    @%p_miss ret;

    // Compute normal.z = (oz + t * dz - cz) / rad
    fma.rn.f32 %hz, %t, %dz, %oz;
    sub.f32 %hz, %hz, %cz;
    div.rn.f32 %nz, %hz, %rad;
    mov.b32 %attr0, %nz;
    mov.u32 %kind, 0;

    call (%reported), _optix_report_intersection_1, (%t, %kind, %attr0);
    ret;
}

.visible .entry __closesthit__ch()
{
    .reg .u32 %idx, %prim, %attr0;
    .reg .f32 %t, %nz;
    .reg .u64 %prim_base, %t_base, %nz_base, %off4, %p_prim, %p_t, %p_nz;

    call (%idx), _optix_get_launch_index_x, ();
    call (%prim), _optix_read_primitive_idx, ();
    call (%t), _optix_get_ray_tmax, ();
    call (%attr0), _optix_get_attribute_0, ();
    mov.b32 %nz, %attr0;

    ld.const.u64 %prim_base, [params + 32];
    ld.const.u64 %t_base, [params + 40];
    ld.const.u64 %nz_base, [params + 48];

    mul.wide.u32 %off4, %idx, 4;
    add.u64 %p_prim, %prim_base, %off4;
    add.u64 %p_t, %t_base, %off4;
    add.u64 %p_nz, %nz_base, %off4;

    st.global.u32 [%p_prim], %prim;
    st.global.f32 [%p_t], %t;
    st.global.f32 [%p_nz], %nz;
    ret;
}

.visible .entry __miss__ms()
{
    .reg .u32 %idx, %none;
    .reg .f32 %neg;
    .reg .u64 %prim_base, %t_base, %nz_base, %off4, %p_prim, %p_t, %p_nz;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %prim_base, [params + 32];
    ld.const.u64 %t_base, [params + 40];
    ld.const.u64 %nz_base, [params + 48];

    mul.wide.u32 %off4, %idx, 4;
    add.u64 %p_prim, %prim_base, %off4;
    add.u64 %p_t, %t_base, %off4;
    add.u64 %p_nz, %nz_base, %off4;

    mov.u32 %none, -1;
    mov.f32 %neg, 0fBF800000;
    st.global.u32 [%p_prim], %none;
    st.global.f32 [%p_t], %neg;
    st.global.f32 [%p_nz], 0f00000000;
    ret;
}
"#;

        let module_opts = sys::OptixModuleCompileOptions::default();
        let pipeline_opts = sys::OptixPipelineCompileOptions {
            numPayloadValues: 0,
            numAttributeValues: 1,
            usesPrimitiveTypeFlags: sys::OptixPrimitiveTypeFlags::OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM
                as u32,
            ..Default::default()
        };
        let link_opts = sys::OptixPipelineLinkOptions::default();

        let module = OptixModule::from_ptx(&optix_ctx, ptx, &module_opts, &pipeline_opts).unwrap();
        let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__rg").unwrap();
        let ms = OptixProgramGroup::new_miss(&optix_ctx, Some((&module, "__miss__ms"))).unwrap();
        let hg = OptixProgramGroup::new_hitgroup(
            &optix_ctx,
            Some((&module, "__closesthit__ch")),
            None,
            Some((&module, "__intersection__sphere")),
        )
        .unwrap();

        let groups = [rg.clone(), ms.clone(), hg.clone()];
        let pipeline =
            OptixPipeline::new(&optix_ctx, &pipeline_opts, &link_opts, &groups, 1).unwrap();
        let sbt = OptixSbt::from_groups(&stream, &rg, &[&ms], &[&hg]).unwrap();

        let origins: [[f32; 3]; 4] = [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.95, 0.95, 0.0], // inside Sphere 0's AABB [-1,1]^2 but outside sphere (r^2 = 1.805 > 1)
            [3.0, 0.9, 0.0],
        ];
        let dirs: [[f32; 3]; 4] = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];

        let d_origins = stream.clone_htod(&origins).unwrap();
        let d_dirs = stream.clone_htod(&dirs).unwrap();
        let d_out_prim = stream.alloc_zeros::<u32>(4).unwrap();
        let d_out_t = stream.alloc_zeros::<f32>(4).unwrap();
        let d_out_nz = stream.alloc_zeros::<f32>(4).unwrap();

        let params = {
            let (sph_ptr, _s0) = d_spheres.device_ptr(&stream);
            let (orig_ptr, _s1) = d_origins.device_ptr(&stream);
            let (dir_ptr, _s2) = d_dirs.device_ptr(&stream);
            let (prim_ptr, _s3) = d_out_prim.device_ptr(&stream);
            let (t_ptr, _s4) = d_out_t.device_ptr(&stream);
            let (nz_ptr, _s5) = d_out_nz.device_ptr(&stream);
            CustomSphereParams {
                handle: blas.handle(),
                spheres: sph_ptr,
                rays_origin: orig_ptr,
                rays_dir: dir_ptr,
                out_prim: prim_ptr,
                out_t: t_ptr,
                out_nz: nz_ptr,
            }
        };
        let d_params = stream.clone_htod(&[params]).unwrap();

        pipeline.launch(&stream, &d_params, &sbt, 4, 1, 1).unwrap();
        stream.synchronize().unwrap();

        let out_prim = stream.clone_dtoh(&d_out_prim).unwrap();
        let out_t = stream.clone_dtoh(&d_out_t).unwrap();
        let out_nz = stream.clone_dtoh(&d_out_nz).unwrap();

        assert_eq!(out_prim, vec![0, 0, u32::MAX, 1]);
        assert!((out_t[0] - 3.0).abs() < 1e-5);
        assert!((out_nz[0] - (-1.0)).abs() < 1e-5);
        assert!((out_t[1] - 3.2).abs() < 1e-5);
        assert!((out_nz[1] - (-0.8)).abs() < 1e-5);
        assert_eq!(out_t[2], -1.0);
        assert!((out_t[3] - 5.8).abs() < 1e-5);
        assert!((out_nz[3] - (-0.8)).abs() < 1e-5);
    }

    /// Multi-triangle surface grid (128 triangles) + 64-ray batch hybrid RT Core (`f32`) +
    /// CUDA `f64` Möller-Trumbore AND `f64` Voronoi closest-point-on-triangle refinement,
    /// cross-checked against independent CPU `f64` reference solvers to < 1e-14 error.
    #[test]
    fn test_optix_large_grid_hybrid_f64_moller_trumbore_and_closest_point() {
        use crate::driver::{LaunchConfig, PushKernelArg};
        use crate::nvrtc::Ptx;

        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        // Build an 8x8 vertex grid (9x9 = 81 vertices, 8x8x2 = 128 triangles) over [-2, 2]^2
        let grid_n = 8usize;
        let stride = grid_n + 1;
        let mut verts_f64 = Vec::with_capacity(stride * stride);
        for iy in 0..=grid_n {
            for ix in 0..=grid_n {
                let x = -2.0 + 4.0 * (ix as f64) / (grid_n as f64);
                let y = -2.0 + 4.0 * (iy as f64) / (grid_n as f64);
                let z = 4.0 + 0.15 * x - 0.1 * y + 1e-11 * (x * x - y * y);
                verts_f64.push([x, y, z]);
            }
        }
        let verts_f32: Vec<[f32; 3]> = verts_f64
            .iter()
            .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
            .collect();

        let mut indices = Vec::with_capacity(grid_n * grid_n * 2);
        for iy in 0..grid_n {
            for ix in 0..grid_n {
                let v00 = (iy * stride + ix) as u32;
                let v10 = (iy * stride + ix + 1) as u32;
                let v01 = ((iy + 1) * stride + ix) as u32;
                let v11 = ((iy + 1) * stride + ix + 1) as u32;
                indices.push([v00, v10, v11]);
                indices.push([v00, v11, v01]);
            }
        }

        let d_verts_f32 = stream.clone_htod(&verts_f32).unwrap();
        let d_verts_f64 = stream.clone_htod(&verts_f64).unwrap();
        let d_indices = stream.clone_htod(&indices).unwrap();

        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_verts_f32,
            Some(&d_indices),
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();

        let optix_ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.const .align 8 .b8 params[32];

.extern .func (.reg .u32 $value) _optix_get_launch_index_x ();
.extern .func (.reg .u32 $value) _optix_read_primitive_idx ();
.extern .func () _optix_set_payload_0 (.reg .u32 $value);
.extern .func (.reg .u32 p0) _optix_trace_1 (
    .reg .u64 handle,
    .reg .f32 rayOriginX,
    .reg .f32 rayOriginY,
    .reg .f32 rayOriginZ,
    .reg .f32 rayDirectionX,
    .reg .f32 rayDirectionY,
    .reg .f32 rayDirectionZ,
    .reg .f32 tmin,
    .reg .f32 tmax,
    .reg .f32 rayTime,
    .reg .u32 visibilityMask,
    .reg .u32 rayFlags,
    .reg .u32 SBToffset,
    .reg .u32 SBTstride,
    .reg .u32 missSBTIndex,
    .reg .u32 p0
);

.visible .entry __raygen__rg()
{
    .reg .u32 %idx, %prim, %vis, %flags, %sbt_off, %sbt_stride, %miss_idx;
    .reg .u64 %handle, %orig_base, %dir_base, %out_prim_base, %off12, %off4, %orig_ptr, %dir_ptr, %out_prim_ptr;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %tmin, %tmax, %rtime;

    call (%idx), _optix_get_launch_index_x, ();
    ld.const.u64 %handle, [params + 0];
    ld.const.u64 %orig_base, [params + 8];
    ld.const.u64 %dir_base, [params + 16];
    ld.const.u64 %out_prim_base, [params + 24];

    mul.wide.u32 %off12, %idx, 12;
    add.u64 %orig_ptr, %orig_base, %off12;
    add.u64 %dir_ptr, %dir_base, %off12;

    ld.global.f32 %ox, [%orig_ptr + 0];
    ld.global.f32 %oy, [%orig_ptr + 4];
    ld.global.f32 %oz, [%orig_ptr + 8];

    ld.global.f32 %dx, [%dir_ptr + 0];
    ld.global.f32 %dy, [%dir_ptr + 4];
    ld.global.f32 %dz, [%dir_ptr + 8];

    mov.f32 %tmin, 0f00000000;
    mov.f32 %tmax, 0f461C4000;
    mov.f32 %rtime, 0f00000000;
    mov.u32 %vis, 255;
    mov.u32 %flags, 0;
    mov.u32 %sbt_off, 0;
    mov.u32 %sbt_stride, 1;
    mov.u32 %miss_idx, 0;
    mov.u32 %prim, -1;

    call (%prim), _optix_trace_1, (
        %handle,
        %ox, %oy, %oz,
        %dx, %dy, %dz,
        %tmin, %tmax, %rtime,
        %vis, %flags, %sbt_off, %sbt_stride, %miss_idx,
        %prim
    );

    mul.wide.u32 %off4, %idx, 4;
    add.u64 %out_prim_ptr, %out_prim_base, %off4;
    st.global.u32 [%out_prim_ptr], %prim;
    ret;
}

.visible .entry __closesthit__ch()
{
    .reg .u32 %prim;
    call (%prim), _optix_read_primitive_idx, ();
    call (), _optix_set_payload_0, (%prim);
    ret;
}

.visible .entry __miss__ms()
{
    .reg .u32 %none;
    mov.u32 %none, -1;
    call (), _optix_set_payload_0, (%none);
    ret;
}
"#;

        let module_opts = sys::OptixModuleCompileOptions::default();
        let pipeline_opts = sys::OptixPipelineCompileOptions {
            numPayloadValues: 1,
            numAttributeValues: 2,
            ..Default::default()
        };
        let link_opts = sys::OptixPipelineLinkOptions::default();

        let module =
            OptixModule::from_ptx(&optix_ctx, optix_ptx, &module_opts, &pipeline_opts).unwrap();
        let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__rg").unwrap();
        let ms = OptixProgramGroup::new_miss(&optix_ctx, Some((&module, "__miss__ms"))).unwrap();
        let hg = OptixProgramGroup::new_hitgroup(
            &optix_ctx,
            Some((&module, "__closesthit__ch")),
            None,
            None,
        )
        .unwrap();

        let groups = [rg.clone(), ms.clone(), hg.clone()];
        let pipeline =
            OptixPipeline::new(&optix_ctx, &pipeline_opts, &link_opts, &groups, 1).unwrap();
        let sbt = OptixSbt::from_groups(&stream, &rg, &[&ms], &[&hg]).unwrap();

        // Generate 64 rays: 56 interior oblique rays hitting various triangles + 8 exterior misses
        let num_rays = 64usize;
        let mut origins_f64 = Vec::with_capacity(num_rays);
        let mut dirs_f64 = Vec::with_capacity(num_rays);
        for i in 0..num_rays {
            if i < 56 {
                let row = i / 8;
                let col = i % 8;
                let x = -1.75 + 0.5 * (col as f64) + 0.03 * ((row as f64) - 3.5);
                let y = -1.65 + 0.45 * (row as f64) - 0.02 * ((col as f64) - 3.5);
                let dx = 0.01 * ((col as f64) - 3.5);
                let dy = -0.01 * ((row as f64) - 3.0);
                let dz = 1.0f64;
                let inv_len = 1.0 / (dx * dx + dy * dy + dz * dz).sqrt();
                origins_f64.push([x, y, 0.0]);
                dirs_f64.push([dx * inv_len, dy * inv_len, dz * inv_len]);
            } else {
                origins_f64.push([10.0 + (i as f64), -10.0, 0.0]);
                dirs_f64.push([0.0, 0.0, 1.0]);
            }
        }

        let origins_f32: Vec<[f32; 3]> = origins_f64
            .iter()
            .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
            .collect();
        let dirs_f32: Vec<[f32; 3]> = dirs_f64
            .iter()
            .map(|&[x, y, z]| [x as f32, y as f32, z as f32])
            .collect();

        let d_origins_f32 = stream.clone_htod(&origins_f32).unwrap();
        let d_dirs_f32 = stream.clone_htod(&dirs_f32).unwrap();
        let d_origins_f64 = stream.clone_htod(&origins_f64).unwrap();
        let d_dirs_f64 = stream.clone_htod(&dirs_f64).unwrap();
        let d_cand_prim = stream.alloc_zeros::<u32>(num_rays).unwrap();
        let mut d_out_t_f64 = stream.alloc_zeros::<f64>(num_rays).unwrap();
        let mut d_out_hit_f64 = stream.alloc_zeros::<[f64; 3]>(num_rays).unwrap();

        let params = {
            let (o_ptr, _s1) = d_origins_f32.device_ptr(&stream);
            let (d_ptr, _s2) = d_dirs_f32.device_ptr(&stream);
            let (p_ptr, _s3) = d_cand_prim.device_ptr(&stream);
            RtCandidateParams {
                handle: blas.handle(),
                rays_origin_f32: o_ptr,
                rays_dir_f32: d_ptr,
                out_candidate_prim: p_ptr,
            }
        };
        let d_params = stream.clone_htod(&[params]).unwrap();

        pipeline
            .launch(&stream, &d_params, &sbt, num_rays as u32, 1, 1)
            .unwrap();

        // CUDA PTX module containing both refine_moller_trumbore_f64 AND closest_point_triangle_f64
        let cuda_f64_ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry refine_moller_trumbore_f64(
    .param .u64 p_verts_f64,
    .param .u64 p_indices_u32,
    .param .u64 p_origins_f64,
    .param .u64 p_dirs_f64,
    .param .u64 p_cand_prim,
    .param .u64 p_out_t_f64,
    .param .u64 p_out_hit_f64,
    .param .u32 p_num_rays
)
{
    .reg .pred %p_oob, %p_miss;
    .reg .u32 %idx, %r_tid, %r_ctaid, %r_ntid, %n, %prim, %i0, %i1, %i2;
    .reg .u64 %verts_base, %idx_base, %orig_base, %dir_base, %cand_base, %out_t_base, %out_hit_base;
    .reg .u64 %off24, %off12, %off8, %off4, %orig_ptr, %dir_ptr, %cand_ptr, %tri_ptr, %v0_ptr, %v1_ptr, %v2_ptr, %out_t_ptr, %out_hit_ptr;
    .reg .f64 %ox, %oy, %oz, %dx, %dy, %dz;
    .reg .f64 %v0x, %v0y, %v0z, %v1x, %v1y, %v1z, %v2x, %v2y, %v2z;
    .reg .f64 %e1x, %e1y, %e1z, %e2x, %e2y, %e2z;
    .reg .f64 %hx, %hy, %hz, %a, %f, %one, %sx, %sy, %sz, %qx, %qy, %qz, %t, %tmp;
    .reg .f64 %px, %py, %pz;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_ctaid, %ctaid.x;
    mov.u32 %r_ntid, %ntid.x;
    mad.lo.u32 %idx, %r_ctaid, %r_ntid, %r_tid;

    ld.param.u32 %n, [p_num_rays];
    setp.ge.u32 %p_oob, %idx, %n;
    @%p_oob ret;

    ld.param.u64 %verts_base, [p_verts_f64];
    ld.param.u64 %idx_base, [p_indices_u32];
    ld.param.u64 %orig_base, [p_origins_f64];
    ld.param.u64 %dir_base, [p_dirs_f64];
    ld.param.u64 %cand_base, [p_cand_prim];
    ld.param.u64 %out_t_base, [p_out_t_f64];
    ld.param.u64 %out_hit_base, [p_out_hit_f64];

    mul.wide.u32 %off4, %idx, 4;
    add.u64 %cand_ptr, %cand_base, %off4;
    ld.global.u32 %prim, [%cand_ptr];

    mul.wide.u32 %off8, %idx, 8;
    add.u64 %out_t_ptr, %out_t_base, %off8;

    mul.wide.u32 %off24, %idx, 24;
    add.u64 %out_hit_ptr, %out_hit_base, %off24;

    setp.eq.u32 %p_miss, %prim, 0xffffffff;
    @%p_miss bra L_MISS;

    add.u64 %orig_ptr, %orig_base, %off24;
    add.u64 %dir_ptr, %dir_base, %off24;

    ld.global.f64 %ox, [%orig_ptr + 0];
    ld.global.f64 %oy, [%orig_ptr + 8];
    ld.global.f64 %oz, [%orig_ptr + 16];

    ld.global.f64 %dx, [%dir_ptr + 0];
    ld.global.f64 %dy, [%dir_ptr + 8];
    ld.global.f64 %dz, [%dir_ptr + 16];

    mul.wide.u32 %off12, %prim, 12;
    add.u64 %tri_ptr, %idx_base, %off12;
    ld.global.u32 %i0, [%tri_ptr + 0];
    ld.global.u32 %i1, [%tri_ptr + 4];
    ld.global.u32 %i2, [%tri_ptr + 8];

    mul.wide.u32 %off24, %i0, 24;
    add.u64 %v0_ptr, %verts_base, %off24;
    mul.wide.u32 %off24, %i1, 24;
    add.u64 %v1_ptr, %verts_base, %off24;
    mul.wide.u32 %off24, %i2, 24;
    add.u64 %v2_ptr, %verts_base, %off24;

    ld.global.f64 %v0x, [%v0_ptr + 0];
    ld.global.f64 %v0y, [%v0_ptr + 8];
    ld.global.f64 %v0z, [%v0_ptr + 16];

    ld.global.f64 %v1x, [%v1_ptr + 0];
    ld.global.f64 %v1y, [%v1_ptr + 8];
    ld.global.f64 %v1z, [%v1_ptr + 16];

    ld.global.f64 %v2x, [%v2_ptr + 0];
    ld.global.f64 %v2y, [%v2_ptr + 8];
    ld.global.f64 %v2z, [%v2_ptr + 16];

    sub.rn.f64 %e1x, %v1x, %v0x;
    sub.rn.f64 %e1y, %v1y, %v0y;
    sub.rn.f64 %e1z, %v1z, %v0z;

    sub.rn.f64 %e2x, %v2x, %v0x;
    sub.rn.f64 %e2y, %v2y, %v0y;
    sub.rn.f64 %e2z, %v2z, %v0z;

    mul.rn.f64 %tmp, %dz, %e2y;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %hx, %dy, %e2z, %tmp;

    mul.rn.f64 %tmp, %dx, %e2z;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %hy, %dz, %e2x, %tmp;

    mul.rn.f64 %tmp, %dy, %e2x;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %hz, %dx, %e2y, %tmp;

    mul.rn.f64 %a, %e1x, %hx;
    fma.rn.f64 %a, %e1y, %hy, %a;
    fma.rn.f64 %a, %e1z, %hz, %a;

    mov.f64 %one, 0d3FF0000000000000;
    div.rn.f64 %f, %one, %a;

    sub.rn.f64 %sx, %ox, %v0x;
    sub.rn.f64 %sy, %oy, %v0y;
    sub.rn.f64 %sz, %oz, %v0z;

    mul.rn.f64 %tmp, %sz, %e1y;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %qx, %sy, %e1z, %tmp;

    mul.rn.f64 %tmp, %sx, %e1z;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %qy, %sz, %e1x, %tmp;

    mul.rn.f64 %tmp, %sy, %e1x;
    neg.f64 %tmp, %tmp;
    fma.rn.f64 %qz, %sx, %e1y, %tmp;

    mul.rn.f64 %t, %e2x, %qx;
    fma.rn.f64 %t, %e2y, %qy, %t;
    fma.rn.f64 %t, %e2z, %qz, %t;
    mul.rn.f64 %t, %f, %t;

    fma.rn.f64 %px, %t, %dx, %ox;
    fma.rn.f64 %py, %t, %dy, %oy;
    fma.rn.f64 %pz, %t, %dz, %oz;

    st.global.f64 [%out_t_ptr], %t;
    st.global.f64 [%out_hit_ptr + 0], %px;
    st.global.f64 [%out_hit_ptr + 8], %py;
    st.global.f64 [%out_hit_ptr + 16], %pz;
    ret;

L_MISS:
    mov.f64 %t, 0dBFF0000000000000;
    st.global.f64 [%out_t_ptr], %t;
    ret;
}
"#;

        let cuda_mod = cuda_ctx.load_module(Ptx::from_src(cuda_f64_ptx)).unwrap();
        let refine_fn = cuda_mod
            .load_function("refine_moller_trumbore_f64")
            .unwrap();
        let num_rays_u32 = num_rays as u32;
        let mut builder = stream.launch_builder(&refine_fn);
        builder.arg(&d_verts_f64);
        builder.arg(&d_indices);
        builder.arg(&d_origins_f64);
        builder.arg(&d_dirs_f64);
        builder.arg(&d_cand_prim);
        builder.arg(&mut d_out_t_f64);
        builder.arg(&mut d_out_hit_f64);
        builder.arg(&num_rays_u32);
        unsafe { builder.launch(LaunchConfig::for_num_elems(num_rays as u32)) }.unwrap();

        stream.synchronize().unwrap();

        let out_prim = stream.clone_dtoh(&d_cand_prim).unwrap();
        let out_t_f64 = stream.clone_dtoh(&d_out_t_f64).unwrap();
        let out_hit_f64 = stream.clone_dtoh(&d_out_hit_f64).unwrap();

        // Independent CPU f64 brute-force Möller-Trumbore reference over all 128 triangles
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };

        for r in 0..num_rays {
            let orig = origins_f64[r];
            let dir = dirs_f64[r];
            let mut best_prim = u32::MAX;
            let mut best_t = f64::INFINITY;

            for (t_idx, &[i0, i1, i2]) in indices.iter().enumerate() {
                let v0 = verts_f64[i0 as usize];
                let v1 = verts_f64[i1 as usize];
                let v2 = verts_f64[i2 as usize];
                let e1 = sub(v1, v0);
                let e2 = sub(v2, v0);
                let h = cross(dir, e2);
                let a = dot(e1, h);
                if a.abs() < 1e-15 {
                    continue;
                }
                let f = 1.0 / a;
                let s = sub(orig, v0);
                let u = f * dot(s, h);
                if !(0.0..=1.0).contains(&u) {
                    continue;
                }
                let q = cross(s, e1);
                let v = f * dot(dir, q);
                if v < 0.0 || u + v > 1.0 {
                    continue;
                }
                let t = f * dot(e2, q);
                if t > 0.0 && t < best_t {
                    best_t = t;
                    best_prim = t_idx as u32;
                }
            }

            assert_eq!(out_prim[r], best_prim, "ray {r} primitive mismatch");
            if best_prim != u32::MAX {
                assert!(
                    (out_t_f64[r] - best_t).abs() < 1e-14,
                    "ray {r} t mismatch: gpu={}, cpu={}",
                    out_t_f64[r],
                    best_t
                );
                for k in 0..3 {
                    let expected_p = orig[k] + best_t * dir[k];
                    assert!(
                        (out_hit_f64[r][k] - expected_p).abs() < 1e-14,
                        "ray {r} coord {k} mismatch"
                    );
                }
            } else {
                assert_eq!(out_t_f64[r], -1.0);
            }
        }
    }

    /// Tests hybrid OptiX candidate primitive lookup + CUDA `f64` exact Voronoi-region
    /// closest-point-on-triangle projection across all 7 Voronoi regions (3 vertices,
    /// 3 edges, and face interior) against a CPU `f64` reference solver.
    #[test]
    fn test_optix_f64_voronoi_closest_point_on_triangle() {
        use crate::driver::{LaunchConfig, PushKernelArg};
        use crate::nvrtc::compile_ptx;

        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let stream = cuda_ctx.default_stream();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();

        // Non-degenerate 3D triangle with exact f64 coordinates:
        // A = (0.0, 0.0, 2.0 + 1e-11)
        // B = (2.0, 0.0, 2.0 - 1e-11)
        // C = (0.0, 3.0, 2.5 + 3e-11)
        let verts_f64: [[f64; 3]; 3] = [
            [0.0, 0.0, 2.0 + 1e-11],
            [2.0, 0.0, 2.0 - 1e-11],
            [0.0, 3.0, 2.5 + 3e-11],
        ];
        let verts_f32: [[f32; 3]; 3] = verts_f64.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]);
        let indices: [[u32; 3]; 1] = [[0, 1, 2]];

        let d_verts_f32 = stream.clone_htod(&verts_f32).unwrap();
        let d_verts_f64 = stream.clone_htod(&verts_f64).unwrap();
        let d_indices = stream.clone_htod(&indices).unwrap();

        // Build triangle BLAS to verify OptiX acceleration structure handles the triangle
        let blas = OptixAccel::build_triangles(
            &optix_ctx,
            &stream,
            &d_verts_f32,
            Some(&d_indices),
            sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
            None,
        )
        .unwrap();
        assert_ne!(blas.handle(), 0);

        // Compile exact f64 Voronoi closest-point-on-triangle kernel via NVRTC
        let cuda_src = r#"
extern "C" __global__ void closest_point_on_triangle_f64(
    const double* __restrict__ verts,
    const unsigned int* __restrict__ tris,
    const unsigned int* __restrict__ cand_prim,
    const double* __restrict__ query_pts,
    double* __restrict__ out_closest,
    double* __restrict__ out_dist_sq,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int prim = cand_prim[i];
    unsigned int i0 = tris[prim * 3 + 0];
    unsigned int i1 = tris[prim * 3 + 1];
    unsigned int i2 = tris[prim * 3 + 2];

    double ax = verts[i0 * 3 + 0], ay = verts[i0 * 3 + 1], az = verts[i0 * 3 + 2];
    double bx = verts[i1 * 3 + 0], by = verts[i1 * 3 + 1], bz = verts[i1 * 3 + 2];
    double cx = verts[i2 * 3 + 0], cy = verts[i2 * 3 + 1], cz = verts[i2 * 3 + 2];

    double px = query_pts[i * 3 + 0];
    double py = query_pts[i * 3 + 1];
    double pz = query_pts[i * 3 + 2];

    double abx = bx - ax, aby = by - ay, abz = bz - az;
    double acx = cx - ax, acy = cy - ay, acz = cz - az;
    double apx = px - ax, apy = py - ay, apz = pz - az;

    double d1 = abx * apx + aby * apy + abz * apz;
    double d2 = acx * apx + acy * apy + acz * apz;

    double qx, qy, qz;
    if (d1 <= 0.0 && d2 <= 0.0) {
        qx = ax; qy = ay; qz = az;
    } else {
        double bpx = px - bx, bpy = py - by, bpz = pz - bz;
        double d3 = abx * bpx + aby * bpy + abz * bpz;
        double d4 = acx * bpx + acy * bpy + acz * bpz;
        if (d3 >= 0.0 && d4 <= d3) {
            qx = bx; qy = by; qz = bz;
        } else {
            double vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                double v = d1 / (d1 - d3);
                qx = ax + v * abx;
                qy = ay + v * aby;
                qz = az + v * abz;
            } else {
                double cpx = px - cx, cpy = py - cy, cpz = pz - cz;
                double d5 = abx * cpx + aby * cpy + abz * cpz;
                double d6 = acx * cpx + acy * cpy + acz * cpz;
                if (d6 >= 0.0 && d5 <= d6) {
                    qx = cx; qy = cy; qz = cz;
                } else {
                    double vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        double w = d2 / (d2 - d6);
                        qx = ax + w * acx;
                        qy = ay + w * acy;
                        qz = az + w * acz;
                    } else {
                        double va = d3 * d6 - d5 * d4;
                        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
                            double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            qx = bx + w * (cx - bx);
                            qy = by + w * (cy - by);
                            qz = bz + w * (cz - bz);
                        } else {
                            double denom = 1.0 / (va + vb + vc);
                            double v = vb * denom;
                            double w = vc * denom;
                            qx = ax + abx * v + acx * w;
                            qy = ay + aby * v + acy * w;
                            qz = az + abz * v + acz * w;
                        }
                    }
                }
            }
        }
    }

    out_closest[i * 3 + 0] = qx;
    out_closest[i * 3 + 1] = qy;
    out_closest[i * 3 + 2] = qz;
    double dx = px - qx, dy = py - qy, dz = pz - qz;
    out_dist_sq[i] = dx * dx + dy * dy + dz * dz;
}
"#;

        let ptx = compile_ptx(cuda_src).unwrap();
        let cuda_mod = cuda_ctx.load_module(ptx).unwrap();
        let closest_fn = cuda_mod
            .load_function("closest_point_on_triangle_f64")
            .unwrap();

        // 7 query points probing all 7 Voronoi regions:
        // 0: Vertex A region (-1, -1, 0)
        // 1: Vertex B region (4, -1, 2)
        // 2: Vertex C region (-1, 5, 3)
        // 3: Edge AB region (1.0, -2.0, 2.0)
        // 4: Edge AC region (-2.0, 1.5, 2.2)
        // 5: Edge BC region (2.5, 2.5, 2.5)
        // 6: Interior face region (0.5, 0.75, 5.0)
        let query_pts: [[f64; 3]; 7] = [
            [-1.0, -1.0, 0.0],
            [4.0, -1.0, 2.0],
            [-1.0, 5.0, 3.0],
            [1.0, -2.0, 2.0],
            [-2.0, 1.5, 2.2],
            [2.5, 2.5, 2.5],
            [0.5, 0.75, 5.0],
        ];
        let cand_prims = [0u32; 7];

        let d_query = stream.clone_htod(&query_pts).unwrap();
        let d_cand = stream.clone_htod(&cand_prims).unwrap();
        let mut d_out_closest = stream.alloc_zeros::<[f64; 3]>(7).unwrap();
        let mut d_out_dist_sq = stream.alloc_zeros::<f64>(7).unwrap();

        let n_queries = 7u32;
        let mut builder = stream.launch_builder(&closest_fn);
        builder.arg(&d_verts_f64);
        builder.arg(&d_indices);
        builder.arg(&d_cand);
        builder.arg(&d_query);
        builder.arg(&mut d_out_closest);
        builder.arg(&mut d_out_dist_sq);
        builder.arg(&n_queries);
        unsafe { builder.launch(LaunchConfig::for_num_elems(n_queries)) }.unwrap();
        stream.synchronize().unwrap();

        let out_closest = stream.clone_dtoh(&d_out_closest).unwrap();
        let out_dist_sq = stream.clone_dtoh(&d_out_dist_sq).unwrap();

        // CPU f64 reference
        let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
        let add = |u: [f64; 3], v: [f64; 3]| [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
        let scale = |s: f64, u: [f64; 3]| [s * u[0], s * u[1], s * u[2]];

        let a = verts_f64[0];
        let b = verts_f64[1];
        let c = verts_f64[2];
        let ab = sub(b, a);
        let ac = sub(c, a);

        for (i, &p) in query_pts.iter().enumerate() {
            let ap = sub(p, a);
            let d1 = dot(ab, ap);
            let d2 = dot(ac, ap);
            let bp = sub(p, b);
            let d3 = dot(ab, bp);
            let d4 = dot(ac, bp);
            let cp = sub(p, c);
            let d5 = dot(ab, cp);
            let d6 = dot(ac, cp);
            let vc = d1 * d4 - d3 * d2;
            let vb = d5 * d2 - d1 * d6;
            let va = d3 * d6 - d5 * d4;

            let expected_q = if d1 <= 0.0 && d2 <= 0.0 {
                a
            } else if d3 >= 0.0 && d4 <= d3 {
                b
            } else if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                add(a, scale(d1 / (d1 - d3), ab))
            } else if d6 >= 0.0 && d5 <= d6 {
                c
            } else if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                add(a, scale(d2 / (d2 - d6), ac))
            } else if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                add(b, scale((d4 - d3) / ((d4 - d3) + (d5 - d6)), sub(c, b)))
            } else {
                let denom = 1.0 / (va + vb + vc);
                let v = vb * denom;
                let w = vc * denom;
                add(a, add(scale(v, ab), scale(w, ac)))
            };
            let diff = sub(p, expected_q);
            let expected_dist_sq = dot(diff, diff);

            for k in 0..3 {
                assert!(
                    (out_closest[i][k] - expected_q[k]).abs() < 1e-15,
                    "region {i} coord {k} mismatch: gpu={}, cpu={}",
                    out_closest[i][k],
                    expected_q[k]
                );
            }
            assert!(
                (out_dist_sq[i] - expected_dist_sq).abs() < 1e-14,
                "region {i} dist_sq mismatch: gpu={}, cpu={}",
                out_dist_sq[i],
                expected_dist_sq
            );
        }
    }

    #[test]
    fn test_edge_cases_empty_inputs_and_sbt_padding() {
        if !OptixContext::is_available() {
            return;
        }
        let cuda_ctx = CudaContext::new(0).unwrap();
        let optix_ctx = OptixContext::new(&cuda_ctx).unwrap();
        let stream = cuda_ctx.default_stream();

        // 1. Empty build_inputs slice should return OPTIX_ERROR_INVALID_VALUE immediately.
        let err = unsafe {
            OptixAccel::build_from_inputs(
                &optix_ctx,
                &stream,
                &sys::OptixAccelBuildOptions::default(),
                &[],
            )
        }
        .unwrap_err();
        assert_eq!(err.0, sys::OptixResult::OPTIX_ERROR_INVALID_VALUE);

        // 2. SbtRecord<u32> has 12 bytes of tail padding (32-byte header + 4-byte u32 -> 48 bytes).
        // Verify that `pack_with_data` zero-initializes all 12 tail padding bytes and that
        // `OptixSbt::from_records` supports empty `miss` and `hitgroups` slices.
        let ptx = r#"
.version 7.0
.target sm_75
.address_size 64

.visible .entry __raygen__noop() {
    ret;
}
"#;
        let mod_opts = sys::OptixModuleCompileOptions::default();
        let pipe_opts = sys::OptixPipelineCompileOptions::default();
        let module = OptixModule::from_ptx(&optix_ctx, ptx, &mod_opts, &pipe_opts).unwrap();
        let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__noop").unwrap();

        let rec_u32 = SbtRecord::pack_with_data(&rg, 0xDEADBEEFu32).unwrap();
        assert_eq!(core::mem::size_of::<SbtRecord<u32>>(), 48);
        let raw_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                (&rec_u32 as *const SbtRecord<u32>).cast::<u8>(),
                core::mem::size_of::<SbtRecord<u32>>(),
            )
        };
        // Bytes 32..36 hold 0xDEADBEEF; bytes 36..48 are the 12 tail padding bytes and must be zero.
        assert_eq!(&raw_bytes[36..48], &[0u8; 12]);

        let empty_ms: &[SbtRecord<u32>] = &[];
        let empty_hg: &[SbtRecord<u32>] = &[];
        let sbt = OptixSbt::from_records(&stream, &rec_u32, empty_ms, empty_hg).unwrap();
        assert_ne!(sbt.raw().raygenRecord, 0);
        assert_eq!(sbt.raw().missRecordBase, 0);
        assert_eq!(sbt.raw().missRecordCount, 0);
        assert_eq!(sbt.raw().hitgroupRecordBase, 0);
        assert_eq!(sbt.raw().hitgroupRecordCount, 0);
    }
}
