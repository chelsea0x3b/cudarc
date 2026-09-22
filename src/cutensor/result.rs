use super::sys;
use core::mem::MaybeUninit;
use std::sync::OnceLock;
use std::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CutensorError(pub sys::cutensorStatus_t);

impl sys::cutensorStatus_t {
    fn result(self) -> Result<(), CutensorError> {
        match self {
            sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS => Ok(()),
            _ => Err(CutensorError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CutensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CutensorError {}

/// Creates a cuTENSOR handle. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
pub fn create_handle() -> Result<sys::cutensorHandle_t, CutensorError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cutensorCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// Destroys a handle previously created with [create_handle()]. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// `handle` must not have been freed already.
pub unsafe fn destroy_handle(handle: sys::cutensorHandle_t) -> Result<(), CutensorError> {
    sys::cutensorDestroy(handle).result()
}

type CreateComputeDescFn = unsafe extern "C" fn(
    sys::cutensorHandle_t,
    *mut sys::cutensorComputeDescriptor_t,
    sys::cudaDataType_t,
    u32,
    *const core::ffi::c_void,
    usize,
) -> sys::cutensorStatus_t;

type DestroyComputeDescFn =
    unsafe extern "C" fn(sys::cutensorComputeDescriptor_t) -> sys::cutensorStatus_t;

fn compute_desc_fns() -> (CreateComputeDescFn, DestroyComputeDescFn) {
    static FNS: OnceLock<(CreateComputeDescFn, DestroyComputeDescFn)> = OnceLock::new();
    *FNS.get_or_init(|| unsafe {
        let lib_names: Vec<_> = crate::get_lib_name_candidates("cutensor");
        let lib = lib_names
            .iter()
            .find_map(|n| libloading::Library::new(n).ok())
            .expect("failed to load cutensor library");
        let create: libloading::Symbol<CreateComputeDescFn> =
            lib.get(b"cutensorCreateComputeDescriptor\0").unwrap();
        let destroy: libloading::Symbol<DestroyComputeDescFn> =
            lib.get(b"cutensorDestroyComputeDescriptor\0").unwrap();
        let fns = (*create, *destroy);
        std::mem::forget(lib);
        fns
    })
}

/// Creates a compute descriptor from a data type.
///
/// # Safety
///
/// `handle` must be valid.
pub unsafe fn create_compute_descriptor(
    handle: sys::cutensorHandle_t,
    data_type: sys::cudaDataType_t,
) -> Result<sys::cutensorComputeDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    let (create_fn, _) = compute_desc_fns();
    create_fn(
        handle,
        desc.as_mut_ptr(),
        data_type,
        0,
        core::ptr::null(),
        0,
    )
    .result()?;
    Ok(desc.assume_init())
}

/// Destroys a compute descriptor.
///
/// # Safety
///
/// `desc` must not have been freed already.
pub unsafe fn destroy_compute_descriptor(
    desc: sys::cutensorComputeDescriptor_t,
) -> Result<(), CutensorError> {
    let (_, destroy_fn) = compute_desc_fns();
    destroy_fn(desc).result()
}

/// Returns the cuTENSOR library version as (major, minor, patch).
///
/// # Safety
///
/// This function is safe to call.
pub fn get_version() -> (usize, usize, usize) {
    let version = unsafe { sys::cutensorGetVersion() };
    let major = version / 10000;
    let minor = (version % 10000) / 100;
    let patch = version % 100;
    (major, minor, patch)
}

/// Creates a tensor descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// - `handle` must be valid
/// - `num_modes` must match the length of `extent` and `stride` arrays
/// - `extent` and `stride` must point to valid memory
pub unsafe fn create_tensor_descriptor(
    handle: sys::cutensorHandle_t,
    num_modes: u32,
    extent: *const i64,
    stride: *const i64,
    data_type: sys::cudaDataType_t,
    alignment_requirement: u32,
) -> Result<sys::cutensorTensorDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    sys::cutensorCreateTensorDescriptor(
        handle,
        desc.as_mut_ptr(),
        num_modes,
        extent,
        stride,
        data_type,
        alignment_requirement,
    )
    .result()?;
    Ok(desc.assume_init())
}

/// Destroys a tensor descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// `desc` must not have been freed already.
pub unsafe fn destroy_tensor_descriptor(
    desc: sys::cutensorTensorDescriptor_t,
) -> Result<(), CutensorError> {
    sys::cutensorDestroyTensorDescriptor(desc).result()
}

/// Creates a contraction operation descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// - All handles and descriptors must be valid
/// - Mode arrays must have correct lengths matching the tensor descriptors
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_contraction(
    handle: sys::cutensorHandle_t,
    desc_a: sys::cutensorTensorDescriptor_t,
    mode_a: *const i32,
    op_a: sys::cutensorOperator_t,
    desc_b: sys::cutensorTensorDescriptor_t,
    mode_b: *const i32,
    op_b: sys::cutensorOperator_t,
    desc_c: sys::cutensorTensorDescriptor_t,
    mode_c: *const i32,
    op_c: sys::cutensorOperator_t,
    desc_d: sys::cutensorTensorDescriptor_t,
    mode_d: *const i32,
    compute_desc: sys::cutensorComputeDescriptor_t,
) -> Result<sys::cutensorOperationDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    sys::cutensorCreateContraction(
        handle,
        desc.as_mut_ptr(),
        desc_a,
        mode_a,
        op_a,
        desc_b,
        mode_b,
        op_b,
        desc_c,
        mode_c,
        op_c,
        desc_d,
        mode_d,
        compute_desc,
    )
    .result()?;
    Ok(desc.assume_init())
}

/// Creates a reduction operation descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// - All handles and descriptors must be valid
/// - Mode arrays must have correct lengths
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_reduction(
    handle: sys::cutensorHandle_t,
    desc_a: sys::cutensorTensorDescriptor_t,
    mode_a: *const i32,
    op_a: sys::cutensorOperator_t,
    desc_c: sys::cutensorTensorDescriptor_t,
    mode_c: *const i32,
    op_c: sys::cutensorOperator_t,
    desc_d: sys::cutensorTensorDescriptor_t,
    mode_d: *const i32,
    op_reduce: sys::cutensorOperator_t,
    compute_desc: sys::cutensorComputeDescriptor_t,
) -> Result<sys::cutensorOperationDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    sys::cutensorCreateReduction(
        handle,
        desc.as_mut_ptr(),
        desc_a,
        mode_a,
        op_a,
        desc_c,
        mode_c,
        op_c,
        desc_d,
        mode_d,
        op_reduce,
        compute_desc,
    )
    .result()?;
    Ok(desc.assume_init())
}

/// Destroys an operation descriptor. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// `desc` must not have been freed already.
pub unsafe fn destroy_operation_descriptor(
    desc: sys::cutensorOperationDescriptor_t,
) -> Result<(), CutensorError> {
    sys::cutensorDestroyOperationDescriptor(desc).result()
}

/// Creates a plan preference object. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// `handle` must be valid.
pub unsafe fn create_plan_preference(
    handle: sys::cutensorHandle_t,
    algo: sys::cutensorAlgo_t,
    jit_mode: sys::cutensorJitMode_t,
) -> Result<sys::cutensorPlanPreference_t, CutensorError> {
    let mut pref = MaybeUninit::uninit();
    sys::cutensorCreatePlanPreference(handle, pref.as_mut_ptr(), algo, jit_mode).result()?;
    Ok(pref.assume_init())
}

/// Destroys a plan preference object. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// `pref` must not have been freed already.
pub unsafe fn destroy_plan_preference(
    pref: sys::cutensorPlanPreference_t,
) -> Result<(), CutensorError> {
    sys::cutensorDestroyPlanPreference(pref).result()
}

/// Estimates the workspace size required for an operation. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// All handles and descriptors must be valid.
pub unsafe fn estimate_workspace_size(
    handle: sys::cutensorHandle_t,
    desc: sys::cutensorOperationDescriptor_t,
    pref: sys::cutensorPlanPreference_t,
    workspace_pref: sys::cutensorWorksizePreference_t,
) -> Result<u64, CutensorError> {
    let mut workspace_size = MaybeUninit::uninit();
    sys::cutensorEstimateWorkspaceSize(
        handle,
        desc,
        pref,
        workspace_pref,
        workspace_size.as_mut_ptr(),
    )
    .result()?;
    Ok(workspace_size.assume_init())
}

/// Creates an execution plan. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// All handles and descriptors must be valid.
pub unsafe fn create_plan(
    handle: sys::cutensorHandle_t,
    desc: sys::cutensorOperationDescriptor_t,
    pref: sys::cutensorPlanPreference_t,
    workspace_size: u64,
) -> Result<sys::cutensorPlan_t, CutensorError> {
    let mut plan = MaybeUninit::uninit();
    sys::cutensorCreatePlan(handle, plan.as_mut_ptr(), desc, pref, workspace_size).result()?;
    Ok(plan.assume_init())
}

/// Destroys an execution plan. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// `plan` must not have been freed already.
pub unsafe fn destroy_plan(plan: sys::cutensorPlan_t) -> Result<(), CutensorError> {
    sys::cutensorDestroyPlan(plan).result()
}

/// Executes a tensor contraction. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// - All handles, plan, and device pointers must be valid
/// - Workspace must be properly allocated with the required size
/// - All tensor data must be accessible from the specified stream
#[allow(clippy::too_many_arguments)]
pub unsafe fn contract(
    handle: sys::cutensorHandle_t,
    plan: sys::cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_size: u64,
    stream: sys::cudaStream_t,
) -> Result<(), CutensorError> {
    sys::cutensorContract(
        handle,
        plan,
        alpha,
        a,
        b,
        beta,
        c,
        d,
        workspace,
        workspace_size,
        stream,
    )
    .result()
}

/// Executes a tensor reduction. See
/// [nvidia docs](https://docs.nvidia.com/cuda/cutensor/latest/index.html)
///
/// # Safety
///
/// - All handles, plan, and device pointers must be valid
/// - Workspace must be properly allocated with the required size
/// - All tensor data must be accessible from the specified stream
#[allow(clippy::too_many_arguments)]
pub unsafe fn reduce(
    handle: sys::cutensorHandle_t,
    plan: sys::cutensorPlan_t,
    alpha: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    c: *const core::ffi::c_void,
    d: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_size: u64,
    stream: sys::cudaStream_t,
) -> Result<(), CutensorError> {
    sys::cutensorReduce(
        handle,
        plan,
        alpha,
        a,
        beta,
        c,
        d,
        workspace,
        workspace_size,
        stream,
    )
    .result()
}
