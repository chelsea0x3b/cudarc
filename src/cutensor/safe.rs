//! Safe abstractions around [crate::cutensor::result] with [CuTensor].

use super::{result, result::CutensorError, sys};
use crate::driver::{self, CudaStream};
use core::ffi::c_void;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CuTensorDataType - maps Rust scalar types to cudaDataType_t
// ---------------------------------------------------------------------------

/// Marker trait that associates a Rust scalar type with the corresponding
/// `cudaDataType_t` constant used by cuTENSOR APIs, along with the matching
/// compute descriptor.
pub trait CuTensorDataType: Copy {
    fn cuda_data_type() -> sys::cudaDataType_t;
}

impl CuTensorDataType for f32 {
    fn cuda_data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_32F
    }
}

impl CuTensorDataType for f64 {
    fn cuda_data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_64F
    }
}

#[cfg(feature = "f16")]
impl CuTensorDataType for half::f16 {
    fn cuda_data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16F
    }
}

#[cfg(feature = "f16")]
impl CuTensorDataType for half::bf16 {
    fn cuda_data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16BF
    }
}

// ---------------------------------------------------------------------------
// Convenience enums
// ---------------------------------------------------------------------------

/// Algorithm selection for plan creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Default,
    DefaultPatient,
    Gett,
    Tgett,
    Ttgt,
}

impl From<Algorithm> for sys::cutensorAlgo_t {
    fn from(algo: Algorithm) -> Self {
        match algo {
            Algorithm::Default => sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            Algorithm::DefaultPatient => sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT_PATIENT,
            Algorithm::Gett => sys::cutensorAlgo_t::CUTENSOR_ALGO_GETT,
            Algorithm::Tgett => sys::cutensorAlgo_t::CUTENSOR_ALGO_TGETT,
            Algorithm::Ttgt => sys::cutensorAlgo_t::CUTENSOR_ALGO_TTGT,
        }
    }
}

/// JIT compilation mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitMode {
    None,
    Default,
}

impl From<JitMode> for sys::cutensorJitMode_t {
    fn from(mode: JitMode) -> Self {
        match mode {
            JitMode::None => sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
            JitMode::Default => sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_DEFAULT,
        }
    }
}

/// Element-wise unary/binary operator applied to tensor elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operator {
    Identity,
    Sqrt,
    Relu,
    Conj,
    Rcp,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Abs,
    Neg,
    Sin,
    Cos,
    Tan,
    Add,
    Mul,
    Max,
    Min,
}

impl From<Operator> for sys::cutensorOperator_t {
    fn from(op: Operator) -> Self {
        match op {
            Operator::Identity => sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY,
            Operator::Sqrt => sys::cutensorOperator_t::CUTENSOR_OP_SQRT,
            Operator::Relu => sys::cutensorOperator_t::CUTENSOR_OP_RELU,
            Operator::Conj => sys::cutensorOperator_t::CUTENSOR_OP_CONJ,
            Operator::Rcp => sys::cutensorOperator_t::CUTENSOR_OP_RCP,
            Operator::Sigmoid => sys::cutensorOperator_t::CUTENSOR_OP_SIGMOID,
            Operator::Tanh => sys::cutensorOperator_t::CUTENSOR_OP_TANH,
            Operator::Exp => sys::cutensorOperator_t::CUTENSOR_OP_EXP,
            Operator::Log => sys::cutensorOperator_t::CUTENSOR_OP_LOG,
            Operator::Abs => sys::cutensorOperator_t::CUTENSOR_OP_ABS,
            Operator::Neg => sys::cutensorOperator_t::CUTENSOR_OP_NEG,
            Operator::Sin => sys::cutensorOperator_t::CUTENSOR_OP_SIN,
            Operator::Cos => sys::cutensorOperator_t::CUTENSOR_OP_COS,
            Operator::Tan => sys::cutensorOperator_t::CUTENSOR_OP_TAN,
            Operator::Add => sys::cutensorOperator_t::CUTENSOR_OP_ADD,
            Operator::Mul => sys::cutensorOperator_t::CUTENSOR_OP_MUL,
            Operator::Max => sys::cutensorOperator_t::CUTENSOR_OP_MAX,
            Operator::Min => sys::cutensorOperator_t::CUTENSOR_OP_MIN,
        }
    }
}

/// Workspace size preference for plan creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspacePreference {
    Min,
    Default,
    Max,
}

impl From<WorkspacePreference> for sys::cutensorWorksizePreference_t {
    fn from(pref: WorkspacePreference) -> Self {
        match pref {
            WorkspacePreference::Min => sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_MIN,
            WorkspacePreference::Default => {
                sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT
            }
            WorkspacePreference::Max => sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_MAX,
        }
    }
}

// ---------------------------------------------------------------------------
// Error type that can hold either a cutensor or driver error
// ---------------------------------------------------------------------------

/// Combined error type for safe cuTENSOR operations that may also invoke
/// driver allocation/free calls.
#[derive(Debug)]
pub enum CuTensorError {
    CuTensor(CutensorError),
    Driver(driver::DriverError),
}

impl From<CutensorError> for CuTensorError {
    fn from(e: CutensorError) -> Self {
        CuTensorError::CuTensor(e)
    }
}

impl From<driver::DriverError> for CuTensorError {
    fn from(e: driver::DriverError) -> Self {
        CuTensorError::Driver(e)
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CuTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CuTensorError::CuTensor(e) => write!(f, "{e}"),
            CuTensorError::Driver(e) => write!(f, "{e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CuTensorError {}

// ---------------------------------------------------------------------------
// CuTensor - main handle wrapper
// ---------------------------------------------------------------------------

/// Safe wrapper around a cuTENSOR handle ([sys::cutensorHandle_t]).
///
/// # Creating a handle
///
/// ```rust,ignore
/// use cudarc::{driver::*, cutensor::*};
/// let ctx = CudaContext::new(0).unwrap();
/// let stream = ctx.default_stream();
/// let cutensor = CuTensor::new(stream).unwrap();
/// ```
pub struct CuTensor {
    pub(crate) handle: sys::cutensorHandle_t,
    pub(crate) stream: Arc<CudaStream>,
}

unsafe impl Send for CuTensor {}
unsafe impl Sync for CuTensor {}

impl CuTensor {
    /// Creates a new cuTENSOR handle bound to `stream`.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CutensorError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::create_handle()?;
        Ok(Self { handle, stream })
    }

    /// Returns the underlying cuTENSOR handle.
    pub fn handle(&self) -> sys::cutensorHandle_t {
        self.handle
    }

    /// Returns the stream this handle is bound to.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Returns the cuTENSOR library version as `(major, minor, patch)`.
    pub fn version(&self) -> (usize, usize, usize) {
        result::get_version()
    }

    /// Performs a tensor contraction: `D = alpha * op_a(A) * op_b(B) + beta * C`.
    ///
    /// This is a convenience method that manages the full cuTENSOR contraction
    /// pipeline internally: creates descriptors, operation, plan preference,
    /// estimates workspace, creates plan, allocates workspace, executes, and
    /// cleans up.
    ///
    /// # Arguments
    ///
    /// * `alpha`, `beta` - Scaling factors
    /// * `a`, `extent_a`, `stride_a`, `mode_a` - Tensor A data, shape, strides, mode labels
    /// * `op_a` - Element-wise operator applied to A
    /// * `b`, `extent_b`, `stride_b`, `mode_b` - Tensor B data, shape, strides, mode labels
    /// * `op_b` - Element-wise operator applied to B
    /// * `c` - Tensor C data (input for beta scaling)
    /// * `d` - Tensor D data (output, may alias C)
    /// * `extent_c`, `stride_c`, `mode_c` - Shape, strides, mode labels for C (and D)
    /// * `op_c` - Element-wise operator applied to C
    ///
    /// # Safety
    ///
    /// All device pointers (`a`, `b`, `c`, `d`) must be valid and point to
    /// GPU memory accessible from this handle's stream. The extents and strides
    /// must accurately describe the tensor layouts.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn contract<T: CuTensorDataType>(
        &self,
        alpha: &T,
        a: *const c_void,
        extent_a: &[i64],
        stride_a: &[i64],
        mode_a: &[i32],
        op_a: Operator,
        b: *const c_void,
        extent_b: &[i64],
        stride_b: &[i64],
        mode_b: &[i32],
        op_b: Operator,
        beta: &T,
        c: *const c_void,
        d: *mut c_void,
        extent_c: &[i64],
        stride_c: &[i64],
        mode_c: &[i32],
        op_c: Operator,
    ) -> Result<(), CuTensorError> {
        let data_type = T::cuda_data_type();
        let compute_desc = result::create_compute_descriptor(self.handle, data_type)?;
        let alignment = 128u32;

        // Create tensor descriptors
        let desc_a = result::create_tensor_descriptor(
            self.handle,
            mode_a.len() as u32,
            extent_a.as_ptr(),
            stride_a.as_ptr(),
            data_type,
            alignment,
        )?;
        let desc_b = result::create_tensor_descriptor(
            self.handle,
            mode_b.len() as u32,
            extent_b.as_ptr(),
            stride_b.as_ptr(),
            data_type,
            alignment,
        )?;
        let desc_c = result::create_tensor_descriptor(
            self.handle,
            mode_c.len() as u32,
            extent_c.as_ptr(),
            stride_c.as_ptr(),
            data_type,
            alignment,
        )?;
        // D shares the same descriptor layout as C
        let desc_d = result::create_tensor_descriptor(
            self.handle,
            mode_c.len() as u32,
            extent_c.as_ptr(),
            stride_c.as_ptr(),
            data_type,
            alignment,
        )?;

        let contract_result = self.contract_with_descriptors(
            alpha,
            a,
            desc_a,
            mode_a,
            op_a,
            b,
            desc_b,
            mode_b,
            op_b,
            beta,
            c,
            d,
            desc_c,
            desc_d,
            mode_c,
            op_c,
            compute_desc,
        );

        let _ = result::destroy_tensor_descriptor(desc_d);
        let _ = result::destroy_tensor_descriptor(desc_c);
        let _ = result::destroy_tensor_descriptor(desc_b);
        let _ = result::destroy_tensor_descriptor(desc_a);
        let _ = result::destroy_compute_descriptor(compute_desc);

        contract_result
    }

    /// Inner helper for contraction that takes pre-created tensor descriptors.
    #[allow(clippy::too_many_arguments)]
    unsafe fn contract_with_descriptors<T: CuTensorDataType>(
        &self,
        alpha: &T,
        a: *const c_void,
        desc_a: sys::cutensorTensorDescriptor_t,
        mode_a: &[i32],
        op_a: Operator,
        b: *const c_void,
        desc_b: sys::cutensorTensorDescriptor_t,
        mode_b: &[i32],
        op_b: Operator,
        beta: &T,
        c: *const c_void,
        d: *mut c_void,
        desc_c: sys::cutensorTensorDescriptor_t,
        desc_d: sys::cutensorTensorDescriptor_t,
        mode_c: &[i32],
        op_c: Operator,
        compute_desc: sys::cutensorComputeDescriptor_t,
    ) -> Result<(), CuTensorError> {
        // Create contraction operation descriptor
        let op_desc = result::create_contraction(
            self.handle,
            desc_a,
            mode_a.as_ptr(),
            op_a.into(),
            desc_b,
            mode_b.as_ptr(),
            op_b.into(),
            desc_c,
            mode_c.as_ptr(),
            op_c.into(),
            desc_d,
            mode_c.as_ptr(),
            compute_desc,
        )?;

        let exec_result = self.execute_plan(
            op_desc,
            alpha as *const T as *const c_void,
            beta as *const T as *const c_void,
            &[a, b],
            c,
            d,
        );

        let _ = result::destroy_operation_descriptor(op_desc);
        exec_result
    }

    /// Performs a tensor reduction: `D = alpha * reduce(op_a(A)) + beta * C`.
    ///
    /// This is a convenience method that manages the full cuTENSOR reduction
    /// pipeline internally.
    ///
    /// # Safety
    ///
    /// All device pointers must be valid and point to GPU memory accessible
    /// from this handle's stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn reduce<T: CuTensorDataType>(
        &self,
        alpha: &T,
        a: *const c_void,
        extent_a: &[i64],
        stride_a: &[i64],
        mode_a: &[i32],
        op_a: Operator,
        beta: &T,
        c: *const c_void,
        d: *mut c_void,
        extent_c: &[i64],
        stride_c: &[i64],
        mode_c: &[i32],
        op_c: Operator,
        op_reduce: Operator,
    ) -> Result<(), CuTensorError> {
        let data_type = T::cuda_data_type();
        let compute_desc = result::create_compute_descriptor(self.handle, data_type)?;
        let alignment = 128u32;

        let desc_a = result::create_tensor_descriptor(
            self.handle,
            mode_a.len() as u32,
            extent_a.as_ptr(),
            stride_a.as_ptr(),
            data_type,
            alignment,
        )?;
        let desc_c = result::create_tensor_descriptor(
            self.handle,
            mode_c.len() as u32,
            extent_c.as_ptr(),
            stride_c.as_ptr(),
            data_type,
            alignment,
        )?;
        let desc_d = result::create_tensor_descriptor(
            self.handle,
            mode_c.len() as u32,
            extent_c.as_ptr(),
            stride_c.as_ptr(),
            data_type,
            alignment,
        )?;

        let op_desc = result::create_reduction(
            self.handle,
            desc_a,
            mode_a.as_ptr(),
            op_a.into(),
            desc_c,
            mode_c.as_ptr(),
            op_c.into(),
            desc_d,
            mode_c.as_ptr(),
            op_reduce.into(),
            compute_desc,
        )?;

        let exec_result = self.execute_reduction_plan(op_desc, alpha, beta, a, c, d);

        let _ = result::destroy_operation_descriptor(op_desc);
        let _ = result::destroy_tensor_descriptor(desc_d);
        let _ = result::destroy_tensor_descriptor(desc_c);
        let _ = result::destroy_tensor_descriptor(desc_a);
        let _ = result::destroy_compute_descriptor(compute_desc);

        exec_result
    }

    /// Shared helper: create plan preference, estimate workspace, create plan,
    /// allocate workspace, execute contraction, free workspace, destroy
    /// intermediates.
    #[allow(clippy::too_many_arguments)]
    unsafe fn execute_plan(
        &self,
        op_desc: sys::cutensorOperationDescriptor_t,
        alpha: *const c_void,
        beta: *const c_void,
        inputs: &[*const c_void],
        c: *const c_void,
        d: *mut c_void,
    ) -> Result<(), CuTensorError> {
        // Create plan preference (default algo, no JIT)
        let pref = result::create_plan_preference(
            self.handle,
            sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
        )?;

        let plan_result = self.execute_with_preference(op_desc, pref, alpha, beta, inputs, c, d);

        let _ = result::destroy_plan_preference(pref);
        plan_result
    }

    /// Continue plan creation and execution with a given preference.
    #[allow(clippy::too_many_arguments)]
    unsafe fn execute_with_preference(
        &self,
        op_desc: sys::cutensorOperationDescriptor_t,
        pref: sys::cutensorPlanPreference_t,
        alpha: *const c_void,
        beta: *const c_void,
        inputs: &[*const c_void],
        c: *const c_void,
        d: *mut c_void,
    ) -> Result<(), CuTensorError> {
        // Estimate workspace
        let workspace_size = result::estimate_workspace_size(
            self.handle,
            op_desc,
            pref,
            sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
        )?;

        // Create plan
        let plan = result::create_plan(self.handle, op_desc, pref, workspace_size)?;

        // Allocate workspace
        let workspace = if workspace_size > 0 {
            driver::result::malloc_async(self.stream.cu_stream(), workspace_size as usize)?
        } else {
            0
        };

        // Execute contraction: inputs[0]=A, inputs[1]=B
        let exec_result = result::contract(
            self.handle,
            plan,
            alpha,
            inputs[0],
            inputs[1],
            beta,
            c,
            d,
            workspace as *mut c_void,
            workspace_size,
            self.stream.cu_stream() as _,
        );

        // Free workspace (always, even on error)
        if workspace_size > 0 {
            let _ = driver::result::free_async(workspace, self.stream.cu_stream());
        }
        let _ = result::destroy_plan(plan);

        exec_result?;
        Ok(())
    }

    /// Shared helper for reduction: plan preference, workspace, plan, execute.
    #[allow(clippy::too_many_arguments)]
    unsafe fn execute_reduction_plan<T: CuTensorDataType>(
        &self,
        op_desc: sys::cutensorOperationDescriptor_t,
        alpha: &T,
        beta: &T,
        a: *const c_void,
        c: *const c_void,
        d: *mut c_void,
    ) -> Result<(), CuTensorError> {
        let pref = result::create_plan_preference(
            self.handle,
            sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
        )?;

        let workspace_size = result::estimate_workspace_size(
            self.handle,
            op_desc,
            pref,
            sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
        )?;

        let plan = result::create_plan(self.handle, op_desc, pref, workspace_size)?;

        let workspace = if workspace_size > 0 {
            driver::result::malloc_async(self.stream.cu_stream(), workspace_size as usize)?
        } else {
            0
        };

        let alpha_ptr = alpha as *const T as *const c_void;
        let beta_ptr = beta as *const T as *const c_void;

        let exec_result = result::reduce(
            self.handle,
            plan,
            alpha_ptr,
            a,
            beta_ptr,
            c,
            d,
            workspace as *mut c_void,
            workspace_size,
            self.stream.cu_stream() as _,
        );

        if workspace_size > 0 {
            let _ = driver::result::free_async(workspace, self.stream.cu_stream());
        }
        let _ = result::destroy_plan(plan);
        let _ = result::destroy_plan_preference(pref);

        exec_result?;
        Ok(())
    }
}

impl Drop for CuTensor {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor RAII wrappers (for advanced/manual usage)
// ---------------------------------------------------------------------------

/// RAII wrapper around a cuTENSOR tensor descriptor
/// ([sys::cutensorTensorDescriptor_t]).
///
/// Automatically destroyed on drop.
pub struct TensorDescriptor {
    pub(crate) desc: sys::cutensorTensorDescriptor_t,
}

impl TensorDescriptor {
    /// Creates a new tensor descriptor.
    ///
    /// # Arguments
    ///
    /// * `handle` - The cuTENSOR handle
    /// * `extent` - Size of each dimension
    /// * `stride` - Stride (in elements) of each dimension
    /// * `data_type` - Element data type
    /// * `alignment` - Alignment requirement in bytes (typically 128)
    pub fn new(
        handle: &CuTensor,
        extent: &[i64],
        stride: &[i64],
        data_type: sys::cudaDataType_t,
        alignment: u32,
    ) -> Result<Self, CutensorError> {
        assert_eq!(
            extent.len(),
            stride.len(),
            "extent and stride must have the same length"
        );
        let desc = unsafe {
            result::create_tensor_descriptor(
                handle.handle,
                extent.len() as u32,
                extent.as_ptr(),
                stride.as_ptr(),
                data_type,
                alignment,
            )?
        };
        Ok(Self { desc })
    }

    /// Returns the underlying descriptor pointer.
    pub fn desc(&self) -> sys::cutensorTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_tensor_descriptor(desc) }.unwrap();
        }
    }
}

/// RAII wrapper around a cuTENSOR operation descriptor
/// ([sys::cutensorOperationDescriptor_t]).
///
/// Created by [`OperationDescriptor::new_contraction`] or
/// [`OperationDescriptor::new_reduction`]. Automatically destroyed on drop.
pub struct OperationDescriptor {
    pub(crate) desc: sys::cutensorOperationDescriptor_t,
}

impl OperationDescriptor {
    /// Creates a contraction operation descriptor.
    ///
    /// # Safety
    ///
    /// All tensor descriptors must be valid and the mode arrays must have
    /// correct lengths matching their respective tensor descriptors.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_contraction(
        handle: &CuTensor,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: Operator,
        desc_b: &TensorDescriptor,
        mode_b: &[i32],
        op_b: Operator,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        op_c: Operator,
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        compute_desc: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self, CutensorError> {
        let desc = result::create_contraction(
            handle.handle,
            desc_a.desc,
            mode_a.as_ptr(),
            op_a.into(),
            desc_b.desc,
            mode_b.as_ptr(),
            op_b.into(),
            desc_c.desc,
            mode_c.as_ptr(),
            op_c.into(),
            desc_d.desc,
            mode_d.as_ptr(),
            compute_desc,
        )?;
        Ok(Self { desc })
    }

    /// Creates a reduction operation descriptor.
    ///
    /// # Safety
    ///
    /// All tensor descriptors must be valid and the mode arrays must have
    /// correct lengths matching their respective tensor descriptors.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_reduction(
        handle: &CuTensor,
        desc_a: &TensorDescriptor,
        mode_a: &[i32],
        op_a: Operator,
        desc_c: &TensorDescriptor,
        mode_c: &[i32],
        op_c: Operator,
        desc_d: &TensorDescriptor,
        mode_d: &[i32],
        op_reduce: Operator,
        compute_desc: sys::cutensorComputeDescriptor_t,
    ) -> Result<Self, CutensorError> {
        let desc = result::create_reduction(
            handle.handle,
            desc_a.desc,
            mode_a.as_ptr(),
            op_a.into(),
            desc_c.desc,
            mode_c.as_ptr(),
            op_c.into(),
            desc_d.desc,
            mode_d.as_ptr(),
            op_reduce.into(),
            compute_desc,
        )?;
        Ok(Self { desc })
    }

    /// Returns the underlying descriptor pointer.
    pub fn desc(&self) -> sys::cutensorOperationDescriptor_t {
        self.desc
    }
}

impl Drop for OperationDescriptor {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::destroy_operation_descriptor(desc) }.unwrap();
        }
    }
}

/// RAII wrapper around a cuTENSOR plan preference
/// ([sys::cutensorPlanPreference_t]).
pub struct PlanPreference {
    pub(crate) pref: sys::cutensorPlanPreference_t,
}

impl PlanPreference {
    /// Creates a new plan preference.
    pub fn new(
        handle: &CuTensor,
        algo: Algorithm,
        jit_mode: JitMode,
    ) -> Result<Self, CutensorError> {
        let pref =
            unsafe { result::create_plan_preference(handle.handle, algo.into(), jit_mode.into())? };
        Ok(Self { pref })
    }

    /// Estimates the workspace size for a given operation.
    pub fn estimate_workspace_size(
        &self,
        handle: &CuTensor,
        op_desc: &OperationDescriptor,
        workspace_pref: WorkspacePreference,
    ) -> Result<u64, CutensorError> {
        unsafe {
            result::estimate_workspace_size(
                handle.handle,
                op_desc.desc,
                self.pref,
                workspace_pref.into(),
            )
        }
    }

    /// Returns the underlying preference pointer.
    pub fn pref(&self) -> sys::cutensorPlanPreference_t {
        self.pref
    }
}

impl Drop for PlanPreference {
    fn drop(&mut self) {
        let pref = std::mem::replace(&mut self.pref, std::ptr::null_mut());
        if !pref.is_null() {
            unsafe { result::destroy_plan_preference(pref) }.unwrap();
        }
    }
}

/// RAII wrapper around a cuTENSOR execution plan ([sys::cutensorPlan_t]).
pub struct ContractionPlan {
    pub(crate) plan: sys::cutensorPlan_t,
}

impl ContractionPlan {
    /// Creates a new execution plan.
    pub fn new(
        handle: &CuTensor,
        op_desc: &OperationDescriptor,
        pref: &PlanPreference,
        workspace_size: u64,
    ) -> Result<Self, CutensorError> {
        let plan =
            unsafe { result::create_plan(handle.handle, op_desc.desc, pref.pref, workspace_size)? };
        Ok(Self { plan })
    }

    /// Returns the underlying plan pointer.
    pub fn plan(&self) -> sys::cutensorPlan_t {
        self.plan
    }
}

impl Drop for ContractionPlan {
    fn drop(&mut self) {
        let plan = std::mem::replace(&mut self.plan, std::ptr::null_mut());
        if !plan.is_null() {
            unsafe { result::destroy_plan(plan) }.unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CudaContext;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn test_create_and_drop() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let _cutensor = CuTensor::new(stream).unwrap();
    }

    #[test]
    fn test_version() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cutensor = CuTensor::new(stream).unwrap();
        let (major, minor, patch) = cutensor.version();
        // cuTENSOR 2.x
        assert!(major >= 2, "expected major >= 2, got {major}");
        assert!(
            major > 0 || minor > 0 || patch > 0,
            "version should be non-zero"
        );
    }

    #[test]
    fn test_contraction_f32() {
        // Matrix multiply via tensor contraction:
        //   A_{ij} * B_{jk} = C_{ik}
        //
        // A is 2x3, B is 3x4, C is 2x4
        //
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        //
        // B = [[1, 2, 3, 4],
        //      [5, 6, 7, 8],
        //      [9, 10, 11, 12]]
        //
        // C = A * B = [[38,  44,  50,  56],
        //              [83,  98, 113, 128]]

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cutensor = CuTensor::new(stream.clone()).unwrap();

        let m: i64 = 2;
        let n: i64 = 4;
        let k: i64 = 3;

        // Row-major A (2x3) stored as column-major for cuTENSOR:
        // mode_a = [i, j], extent = [m, k], stride = [1, m]
        // Column-major storage: columns of [1,4], [2,5], [3,6]
        let a_data: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let a_dev = stream.clone_htod(&a_data).unwrap();

        // B (3x4) in column-major:
        // mode_b = [j, k], extent = [k, n], stride = [1, k]
        // Columns: [1,5,9], [2,6,10], [3,7,11], [4,8,12]
        let b_data: Vec<f32> = vec![
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ];
        let b_dev = stream.clone_htod(&b_data).unwrap();

        // C/D (2x4) in column-major, initialized to zero
        let c_data: Vec<f32> = vec![0.0; (m * n) as usize];
        let c_dev = stream.clone_htod(&c_data).unwrap();
        let d_dev = stream.clone_htod(&c_data).unwrap();

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // Mode labels: i=0, j=1, k=2
        let mode_a: Vec<i32> = vec![0, 1]; // i, j
        let mode_b: Vec<i32> = vec![1, 2]; // j, k
        let mode_c: Vec<i32> = vec![0, 2]; // i, k

        let extent_a: Vec<i64> = vec![m, k];
        let stride_a: Vec<i64> = vec![1, m];

        let extent_b: Vec<i64> = vec![k, n];
        let stride_b: Vec<i64> = vec![1, k];

        let extent_c: Vec<i64> = vec![m, n];
        let stride_c: Vec<i64> = vec![1, m];

        unsafe {
            cutensor
                .contract::<f32>(
                    &alpha,
                    a_dev.cu_device_ptr as *const c_void,
                    &extent_a,
                    &stride_a,
                    &mode_a,
                    Operator::Identity,
                    b_dev.cu_device_ptr as *const c_void,
                    &extent_b,
                    &stride_b,
                    &mode_b,
                    Operator::Identity,
                    &beta,
                    c_dev.cu_device_ptr as *const c_void,
                    d_dev.cu_device_ptr as *mut c_void,
                    &extent_c,
                    &stride_c,
                    &mode_c,
                    Operator::Identity,
                )
                .unwrap();
        }
        // Read back result (column-major 2x4)
        let result: Vec<f32> = stream.clone_dtoh(&d_dev).unwrap();

        // Expected C (column-major): columns of [38,83], [44,98], [50,113], [56,128]
        let expected: Vec<f32> = vec![38.0, 83.0, 44.0, 98.0, 50.0, 113.0, 56.0, 128.0];
        for i in 0..expected.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-3,
                "mismatch at index {i}: got {} expected {}",
                result[i],
                expected[i]
            );
        }
    }
}
