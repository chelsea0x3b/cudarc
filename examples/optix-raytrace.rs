//! End-to-end NVIDIA OptiX hardware ray-tracing example using `cudarc::optix`.
//!
//! Demonstrates:
//! 1. Probing runtime OptiX driver availability (`OptixContext::is_available`) and binding
//!    an [`OptixContext`] to a [`CudaContext`].
//! 2. Building a hardware Bottom-Level Acceleration Structure (BLAS / GAS) from a triangle
//!    mesh stored in [`CudaSlice`] buffers on a [`CudaStream`].
//! 3. Compiling an [`OptixModule`] from PTX containing `__raygen__rg`, `__closesthit__ch`,
//!    and `__miss__ms` entry points that invoke the hardware RT Core `_optix_trace_0` intrinsic.
//! 4. Creating [`OptixProgramGroup`]s, packing 16-byte aligned Shader Binding Table records
//!    into an [`OptixSbt`], and linking an [`OptixPipeline`].
//! 5. Launching rays asynchronously on the [`CudaStream`] via [`OptixPipeline::launch`] and
//!    reading back the hit primitive IDs and intersection distances.
//!
//! Run with:
//! ```bash
//! cargo run --features optix --example optix-raytrace
//! ```

use cudarc::driver::{CudaContext, DevicePtr, DeviceRepr};
use cudarc::optix::{
    sys, OptixAccel, OptixContext, OptixError, OptixModule, OptixPipeline, OptixProgramGroup,
    OptixSbt,
};

/// Constant-memory launch parameters (`params` in PTX) uploaded to the device before `optixLaunch`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct LaunchParams {
    traversable: u64,
    rays_origin: u64,
    rays_dir: u64,
    out_prim: u64,
    out_t: u64,
}
unsafe impl DeviceRepr for LaunchParams {}

const PTX_SRC: &str = r#"
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

fn main() -> Result<(), OptixError> {
    if !OptixContext::is_available() {
        println!("OptiX driver is not available on this machine; skipping.");
        return Ok(());
    }

    let cuda_ctx = CudaContext::new(0)?;
    let stream = cuda_ctx.default_stream();
    let optix_ctx = OptixContext::new(&cuda_ctx)?;
    println!(
        "Initialized OptiX context (RT Core version: {})",
        optix_ctx.rtcore_version()?
    );

    // Build a triangle mesh BLAS
    let vertices: [[f32; 3]; 3] = [[-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0]];
    let indices: [[u32; 3]; 1] = [[0, 1, 2]];
    let d_vertices = stream.clone_htod(&vertices)?;
    let d_indices = stream.clone_htod(&indices)?;

    let blas = OptixAccel::build_triangles(
        &optix_ctx,
        &stream,
        &d_vertices,
        Some(&d_indices),
        sys::OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as u32,
        None,
    )?;
    println!("Built triangle BLAS handle: {:#x}", blas.handle());

    // Compile PTX module, program groups, SBT, and pipeline
    let module_opts = sys::OptixModuleCompileOptions::default();
    let pipeline_opts = sys::OptixPipelineCompileOptions {
        numPayloadValues: 0,
        numAttributeValues: 2,
        ..Default::default()
    };
    let link_opts = sys::OptixPipelineLinkOptions::default();

    let module = OptixModule::from_ptx(&optix_ctx, PTX_SRC, &module_opts, &pipeline_opts)?;
    let rg = OptixProgramGroup::new_raygen(&optix_ctx, &module, "__raygen__rg")?;
    let ms = OptixProgramGroup::new_miss(&optix_ctx, Some((&module, "__miss__ms")))?;
    let hg = OptixProgramGroup::new_hitgroup(
        &optix_ctx,
        Some((&module, "__closesthit__ch")),
        None,
        None,
    )?;

    let groups = [rg.clone(), ms.clone(), hg.clone()];
    let pipeline = OptixPipeline::new(&optix_ctx, &pipeline_opts, &link_opts, &groups, 1)?;
    let sbt = OptixSbt::from_groups(&stream, &rg, &[&ms], &[&hg])?;

    let origins: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]];
    let dirs: [[f32; 3]; 2] = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    let d_origins = stream.clone_htod(&origins)?;
    let d_dirs = stream.clone_htod(&dirs)?;
    let d_out_prim = stream.alloc_zeros::<u32>(2)?;
    let d_out_t = stream.alloc_zeros::<f32>(2)?;

    let params = {
        let (orig_ptr, _s1) = d_origins.device_ptr(&stream);
        let (dir_ptr, _s2) = d_dirs.device_ptr(&stream);
        let (prim_ptr, _s3) = d_out_prim.device_ptr(&stream);
        let (t_ptr, _s4) = d_out_t.device_ptr(&stream);
        LaunchParams {
            traversable: blas.handle(),
            rays_origin: orig_ptr,
            rays_dir: dir_ptr,
            out_prim: prim_ptr,
            out_t: t_ptr,
        }
    };
    let d_params = stream.clone_htod(&[params])?;

    pipeline.launch(&stream, &d_params, &sbt, 2, 1, 1)?;
    stream.synchronize()?;

    let out_prim = stream.clone_dtoh(&d_out_prim)?;
    let out_t = stream.clone_dtoh(&d_out_t)?;
    println!("Ray 0 -> prim={}, t={}", out_prim[0] as i32, out_t[0]);
    println!("Ray 1 -> prim={}, t={}", out_prim[1] as i32, out_t[1]);
    assert_eq!(out_prim, vec![0, u32::MAX]);
    assert!((out_t[0] - 4.0).abs() < 1e-5);

    Ok(())
}
