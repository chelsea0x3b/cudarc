//! Safe abstractions around [crate::cusparse::result] with [CudaSparse].

use super::{result, result::CusparseError, sys};
use crate::driver::{self, CudaStream};
use core::ffi::c_void;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// CudaSparseDataType - maps Rust scalar types to cudaDataType
// ---------------------------------------------------------------------------

/// Marker trait that associates a Rust scalar type with the corresponding
/// `cudaDataType` constant used by cuSPARSE generic APIs.
pub trait CudaSparseDataType: Copy {
    fn cuda_data_type() -> sys::cudaDataType;
}

impl CudaSparseDataType for f32 {
    fn cuda_data_type() -> sys::cudaDataType {
        sys::cudaDataType::CUDA_R_32F
    }
}

impl CudaSparseDataType for f64 {
    fn cuda_data_type() -> sys::cudaDataType {
        sys::cudaDataType::CUDA_R_64F
    }
}

// ---------------------------------------------------------------------------
// Convenience enums
// ---------------------------------------------------------------------------

/// Sparse/dense operation transpose mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    NonTranspose,
    Transpose,
    ConjugateTranspose,
}

impl From<Operation> for sys::cusparseOperation_t {
    fn from(op: Operation) -> Self {
        match op {
            Operation::NonTranspose => sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            Operation::Transpose => sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE,
            Operation::ConjugateTranspose => {
                sys::cusparseOperation_t::CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
            }
        }
    }
}

/// Index data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexType {
    U16,
    I32,
    I64,
}

impl From<IndexType> for sys::cusparseIndexType_t {
    fn from(t: IndexType) -> Self {
        match t {
            IndexType::U16 => sys::cusparseIndexType_t::CUSPARSE_INDEX_16U,
            IndexType::I32 => sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            IndexType::I64 => sys::cusparseIndexType_t::CUSPARSE_INDEX_64I,
        }
    }
}

/// Index base (zero or one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexBase {
    Zero,
    One,
}

impl From<IndexBase> for sys::cusparseIndexBase_t {
    fn from(b: IndexBase) -> Self {
        match b {
            IndexBase::Zero => sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            IndexBase::One => sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ONE,
        }
    }
}

/// Dense matrix storage order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Column,
    Row,
}

impl From<Order> for sys::cusparseOrder_t {
    fn from(o: Order) -> Self {
        match o {
            Order::Column => sys::cusparseOrder_t::CUSPARSE_ORDER_COL,
            Order::Row => sys::cusparseOrder_t::CUSPARSE_ORDER_ROW,
        }
    }
}

// ---------------------------------------------------------------------------
// Error type that can hold either a cusparse or driver error
// ---------------------------------------------------------------------------

/// Combined error type for safe cusparse operations that may also invoke
/// driver allocation/free calls.
#[derive(Debug)]
pub enum CudaSparseError {
    Cusparse(CusparseError),
    Driver(driver::DriverError),
}

impl From<CusparseError> for CudaSparseError {
    fn from(e: CusparseError) -> Self {
        CudaSparseError::Cusparse(e)
    }
}

impl From<driver::DriverError> for CudaSparseError {
    fn from(e: driver::DriverError) -> Self {
        CudaSparseError::Driver(e)
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CudaSparseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaSparseError::Cusparse(e) => write!(f, "{e}"),
            CudaSparseError::Driver(e) => write!(f, "{e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CudaSparseError {}

// ---------------------------------------------------------------------------
// CudaSparse - main handle wrapper
// ---------------------------------------------------------------------------

/// Safe wrapper around a cuSPARSE handle ([sys::cusparseHandle_t]).
///
/// # Creating a handle
///
/// ```rust,ignore
/// use cudarc::{driver::*, cusparse::*};
/// let ctx = CudaContext::new(0).unwrap();
/// let stream = ctx.default_stream();
/// let sparse = CudaSparse::new(stream).unwrap();
/// ```
pub struct CudaSparse {
    pub(crate) handle: sys::cusparseHandle_t,
    pub(crate) stream: Arc<CudaStream>,
}

unsafe impl Send for CudaSparse {}
unsafe impl Sync for CudaSparse {}

impl CudaSparse {
    /// Creates a new cuSPARSE handle bound to `stream`.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CusparseError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::create()?;
        unsafe { result::set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    /// Returns the underlying cuSPARSE handle.
    pub fn handle(&self) -> sys::cusparseHandle_t {
        self.handle
    }

    /// Sets the stream used by this handle.
    ///
    /// # Safety
    /// The caller must ensure the stream is properly synchronised and
    /// belongs to the same CUDA context.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CusparseError> {
        self.stream = stream;
        result::set_stream(self.handle, self.stream.cu_stream() as _)
    }

    /// Sparse matrix-dense vector multiply: `y = alpha * op(A) * x + beta * y`.
    ///
    /// Workspace allocation and deallocation is handled internally.
    #[allow(clippy::too_many_arguments)]
    pub fn spmv<T: CudaSparseDataType>(
        &self,
        op_a: Operation,
        alpha: &T,
        mat_a: &SpMatDescr,
        vec_x: &DnVecDescr,
        beta: &T,
        vec_y: &mut DnVecDescr,
        alg: sys::cusparseSpMVAlg_t,
    ) -> Result<(), CudaSparseError> {
        let compute_type = T::cuda_data_type();
        let op_a_sys: sys::cusparseOperation_t = op_a.into();

        let alpha_ptr = alpha as *const T as *const c_void;
        let beta_ptr = beta as *const T as *const c_void;

        // 1. Query required workspace size.
        let buffer_size = unsafe {
            result::spmv_buffer_size(
                self.handle,
                op_a_sys,
                alpha_ptr,
                mat_a.descr,
                vec_x.descr,
                beta_ptr,
                vec_y.descr,
                compute_type,
                alg,
            )?
        };

        // 2. Allocate workspace on GPU.
        let workspace = if buffer_size > 0 {
            unsafe { driver::result::malloc_async(self.stream.cu_stream(), buffer_size)? }
        } else {
            0
        };

        // 3. Execute SpMV.
        let exec_result = unsafe {
            result::spmv(
                self.handle,
                op_a_sys,
                alpha_ptr,
                mat_a.descr,
                vec_x.descr,
                beta_ptr,
                vec_y.descr,
                compute_type,
                alg,
                workspace as *mut c_void,
            )
        };

        // 4. Free workspace (always, even on error).
        if buffer_size > 0 {
            unsafe { driver::result::free_async(workspace, self.stream.cu_stream())? };
        }

        exec_result?;
        Ok(())
    }

    /// Sparse matrix-dense matrix multiply: `C = alpha * op(A) * op(B) + beta * C`.
    ///
    /// Workspace allocation and deallocation is handled internally.
    #[allow(clippy::too_many_arguments)]
    pub fn spmm<T: CudaSparseDataType>(
        &self,
        op_a: Operation,
        op_b: Operation,
        alpha: &T,
        mat_a: &SpMatDescr,
        mat_b: &DnMatDescr,
        beta: &T,
        mat_c: &mut DnMatDescr,
        alg: sys::cusparseSpMMAlg_t,
    ) -> Result<(), CudaSparseError> {
        let compute_type = T::cuda_data_type();
        let op_a_sys: sys::cusparseOperation_t = op_a.into();
        let op_b_sys: sys::cusparseOperation_t = op_b.into();

        let alpha_ptr = alpha as *const T as *const c_void;
        let beta_ptr = beta as *const T as *const c_void;

        // 1. Query required workspace size.
        let buffer_size = unsafe {
            result::spmm_buffer_size(
                self.handle,
                op_a_sys,
                op_b_sys,
                alpha_ptr,
                mat_a.descr,
                mat_b.descr,
                beta_ptr,
                mat_c.descr,
                compute_type,
                alg,
            )?
        };

        // 2. Allocate workspace on GPU.
        let workspace = if buffer_size > 0 {
            unsafe { driver::result::malloc_async(self.stream.cu_stream(), buffer_size)? }
        } else {
            0
        };

        // 3. Execute SpMM.
        let exec_result = unsafe {
            result::spmm(
                self.handle,
                op_a_sys,
                op_b_sys,
                alpha_ptr,
                mat_a.descr,
                mat_b.descr,
                beta_ptr,
                mat_c.descr,
                compute_type,
                alg,
                workspace as *mut c_void,
            )
        };

        // 4. Free workspace (always, even on error).
        if buffer_size > 0 {
            unsafe { driver::result::free_async(workspace, self.stream.cu_stream())? };
        }

        exec_result?;
        Ok(())
    }
}

impl Drop for CudaSparse {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy(handle) }.unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor RAII wrappers
// ---------------------------------------------------------------------------

/// RAII wrapper around a sparse matrix descriptor (`cusparseSpMatDescr_t`).
///
/// Created via [`SpMatDescr::new_csr`], [`SpMatDescr::new_csc`], or
/// [`SpMatDescr::new_coo`].  Automatically destroyed on drop.
pub struct SpMatDescr {
    pub(crate) descr: sys::cusparseSpMatDescr_t,
}

impl SpMatDescr {
    /// Creates a CSR sparse matrix descriptor.
    ///
    /// # Safety
    /// `csr_row_offsets`, `csr_col_ind`, and `csr_values` must be valid
    /// device pointers that remain live for the lifetime of this descriptor.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_csr(
        rows: i64,
        cols: i64,
        nnz: i64,
        csr_row_offsets: *mut c_void,
        csr_col_ind: *mut c_void,
        csr_values: *mut c_void,
        row_offsets_type: IndexType,
        col_ind_type: IndexType,
        idx_base: IndexBase,
        value_type: sys::cudaDataType,
    ) -> Result<Self, CusparseError> {
        let descr = result::create_csr(
            rows,
            cols,
            nnz,
            csr_row_offsets,
            csr_col_ind,
            csr_values,
            row_offsets_type.into(),
            col_ind_type.into(),
            idx_base.into(),
            value_type,
        )?;
        Ok(Self { descr })
    }

    /// Creates a CSC sparse matrix descriptor.
    ///
    /// # Safety
    /// `csc_col_offsets`, `csc_row_ind`, and `csc_values` must be valid
    /// device pointers that remain live for the lifetime of this descriptor.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_csc(
        rows: i64,
        cols: i64,
        nnz: i64,
        csc_col_offsets: *mut c_void,
        csc_row_ind: *mut c_void,
        csc_values: *mut c_void,
        col_offsets_type: IndexType,
        row_ind_type: IndexType,
        idx_base: IndexBase,
        value_type: sys::cudaDataType,
    ) -> Result<Self, CusparseError> {
        let descr = result::create_csc(
            rows,
            cols,
            nnz,
            csc_col_offsets,
            csc_row_ind,
            csc_values,
            col_offsets_type.into(),
            row_ind_type.into(),
            idx_base.into(),
            value_type,
        )?;
        Ok(Self { descr })
    }

    /// Creates a COO sparse matrix descriptor.
    ///
    /// # Safety
    /// `coo_row_ind`, `coo_col_ind`, and `coo_values` must be valid
    /// device pointers that remain live for the lifetime of this descriptor.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_coo(
        rows: i64,
        cols: i64,
        nnz: i64,
        coo_row_ind: *mut c_void,
        coo_col_ind: *mut c_void,
        coo_values: *mut c_void,
        coo_idx_type: IndexType,
        idx_base: IndexBase,
        value_type: sys::cudaDataType,
    ) -> Result<Self, CusparseError> {
        let descr = result::create_coo(
            rows,
            cols,
            nnz,
            coo_row_ind,
            coo_col_ind,
            coo_values,
            coo_idx_type.into(),
            idx_base.into(),
            value_type,
        )?;
        Ok(Self { descr })
    }

    /// Returns the dimensions `(rows, cols, nnz)` of this sparse matrix.
    pub fn get_size(&self) -> Result<(i64, i64, i64), CusparseError> {
        unsafe { result::sp_mat_get_size(self.descr) }
    }

    /// Returns the storage format of this sparse matrix.
    pub fn get_format(&self) -> Result<sys::cusparseFormat_t, CusparseError> {
        unsafe { result::sp_mat_get_format(self.descr) }
    }
}

impl Drop for SpMatDescr {
    fn drop(&mut self) {
        let descr = std::mem::replace(&mut self.descr, std::ptr::null_mut());
        if !descr.is_null() {
            unsafe { result::destroy_sp_mat(descr) }.unwrap();
        }
    }
}

/// RAII wrapper around a sparse vector descriptor (`cusparseSpVecDescr_t`).
pub struct SpVecDescr {
    pub(crate) descr: sys::cusparseSpVecDescr_t,
}

impl SpVecDescr {
    /// Creates a sparse vector descriptor.
    ///
    /// # Safety
    /// `indices` and `values` must be valid device pointers that remain
    /// live for the lifetime of this descriptor.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        size: i64,
        nnz: i64,
        indices: *mut c_void,
        values: *mut c_void,
        idx_type: IndexType,
        idx_base: IndexBase,
        value_type: sys::cudaDataType,
    ) -> Result<Self, CusparseError> {
        let descr = result::create_sp_vec(
            size,
            nnz,
            indices,
            values,
            idx_type.into(),
            idx_base.into(),
            value_type,
        )?;
        Ok(Self { descr })
    }
}

impl Drop for SpVecDescr {
    fn drop(&mut self) {
        let descr = std::mem::replace(&mut self.descr, std::ptr::null_mut());
        if !descr.is_null() {
            unsafe { result::destroy_sp_vec(descr) }.unwrap();
        }
    }
}

/// RAII wrapper around a dense vector descriptor (`cusparseDnVecDescr_t`).
pub struct DnVecDescr {
    pub(crate) descr: sys::cusparseDnVecDescr_t,
}

impl DnVecDescr {
    /// Creates a dense vector descriptor.
    ///
    /// # Safety
    /// `values` must be a valid device pointer that remains live for the
    /// lifetime of this descriptor.
    pub unsafe fn new(
        size: i64,
        values: *mut c_void,
        value_type: sys::cudaDataType,
    ) -> Result<Self, CusparseError> {
        let descr = result::create_dn_vec(size, values, value_type)?;
        Ok(Self { descr })
    }
}

impl Drop for DnVecDescr {
    fn drop(&mut self) {
        let descr = std::mem::replace(&mut self.descr, std::ptr::null_mut());
        if !descr.is_null() {
            unsafe { result::destroy_dn_vec(descr) }.unwrap();
        }
    }
}

/// RAII wrapper around a dense matrix descriptor (`cusparseDnMatDescr_t`).
pub struct DnMatDescr {
    pub(crate) descr: sys::cusparseDnMatDescr_t,
}

impl DnMatDescr {
    /// Creates a dense matrix descriptor.
    ///
    /// # Safety
    /// `values` must be a valid device pointer that remains live for the
    /// lifetime of this descriptor.
    pub unsafe fn new(
        rows: i64,
        cols: i64,
        ld: i64,
        values: *mut c_void,
        value_type: sys::cudaDataType,
        order: Order,
    ) -> Result<Self, CusparseError> {
        let descr = result::create_dn_mat(rows, cols, ld, values, value_type, order.into())?;
        Ok(Self { descr })
    }
}

impl Drop for DnMatDescr {
    fn drop(&mut self) {
        let descr = std::mem::replace(&mut self.descr, std::ptr::null_mut());
        if !descr.is_null() {
            unsafe { result::destroy_dn_mat(descr) }.unwrap();
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
        let _sparse = CudaSparse::new(stream).unwrap();
    }

    #[test]
    fn test_spmv_csr_f32() {
        // 2x3 CSR matrix:
        //  [1.0, 0.0, 2.0]
        //  [0.0, 3.0, 0.0]
        //
        // x = [1.0, 2.0, 3.0]
        // y = A*x = [1*1+2*3, 3*2] = [7.0, 6.0]

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let sparse = CudaSparse::new(stream.clone()).unwrap();

        let rows: i64 = 2;
        let cols: i64 = 3;
        let nnz: i64 = 3;

        // CSR arrays
        let row_offsets: Vec<i32> = vec![0, 2, 3]; // rows+1
        let col_indices: Vec<i32> = vec![0, 2, 1]; // nnz
        let values: Vec<f32> = vec![1.0, 2.0, 3.0]; // nnz

        let row_offsets_dev = stream.clone_htod(&row_offsets).unwrap();
        let col_indices_dev = stream.clone_htod(&col_indices).unwrap();
        let values_dev = stream.clone_htod(&values).unwrap();

        let mat_a = unsafe {
            SpMatDescr::new_csr(
                rows,
                cols,
                nnz,
                row_offsets_dev.cu_device_ptr as *mut c_void,
                col_indices_dev.cu_device_ptr as *mut c_void,
                values_dev.cu_device_ptr as *mut c_void,
                IndexType::I32,
                IndexType::I32,
                IndexBase::Zero,
                f32::cuda_data_type(),
            )
            .unwrap()
        };

        let x_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let x_dev = stream.clone_htod(&x_data).unwrap();
        let vec_x = unsafe {
            DnVecDescr::new(
                cols,
                x_dev.cu_device_ptr as *mut c_void,
                f32::cuda_data_type(),
            )
            .unwrap()
        };

        let y_data: Vec<f32> = vec![0.0, 0.0];
        let y_dev = stream.clone_htod(&y_data).unwrap();
        let mut vec_y = unsafe {
            DnVecDescr::new(
                rows,
                y_dev.cu_device_ptr as *mut c_void,
                f32::cuda_data_type(),
            )
            .unwrap()
        };

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        sparse
            .spmv(
                Operation::NonTranspose,
                &alpha,
                &mat_a,
                &vec_x,
                &beta,
                &mut vec_y,
                sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
            )
            .unwrap();

        let y_host: Vec<f32> = stream.clone_dtoh(&y_dev).unwrap();
        assert!(
            (y_host[0] - 7.0).abs() < 1e-5,
            "expected 7.0, got {}",
            y_host[0]
        );
        assert!(
            (y_host[1] - 6.0).abs() < 1e-5,
            "expected 6.0, got {}",
            y_host[1]
        );
    }

    #[test]
    fn test_spmm_csr_f32() {
        // A (2x3 CSR):
        //  [1.0, 0.0, 2.0]
        //  [0.0, 3.0, 0.0]
        //
        // B (3x2 dense, column-major):
        //  [1.0, 4.0]
        //  [2.0, 5.0]
        //  [3.0, 6.0]
        //
        // C = A * B =
        //  [1*1+2*3, 1*4+2*6] = [7.0, 16.0]
        //  [3*2,     3*5    ] = [6.0, 15.0]

        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let sparse = CudaSparse::new(stream.clone()).unwrap();

        let a_rows: i64 = 2;
        let a_cols: i64 = 3;
        let a_nnz: i64 = 3;

        let row_offsets: Vec<i32> = vec![0, 2, 3];
        let col_indices: Vec<i32> = vec![0, 2, 1];
        let a_values: Vec<f32> = vec![1.0, 2.0, 3.0];

        let row_offsets_dev = stream.clone_htod(&row_offsets).unwrap();
        let col_indices_dev = stream.clone_htod(&col_indices).unwrap();
        let a_values_dev = stream.clone_htod(&a_values).unwrap();

        let mat_a = unsafe {
            SpMatDescr::new_csr(
                a_rows,
                a_cols,
                a_nnz,
                row_offsets_dev.cu_device_ptr as *mut c_void,
                col_indices_dev.cu_device_ptr as *mut c_void,
                a_values_dev.cu_device_ptr as *mut c_void,
                IndexType::I32,
                IndexType::I32,
                IndexBase::Zero,
                f32::cuda_data_type(),
            )
            .unwrap()
        };

        let b_rows: i64 = 3;
        let b_cols: i64 = 2;
        // Column-major: col0=[1,2,3], col1=[4,5,6]
        let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_dev = stream.clone_htod(&b_data).unwrap();
        let mat_b = unsafe {
            DnMatDescr::new(
                b_rows,
                b_cols,
                b_rows, // ld = rows for column-major
                b_dev.cu_device_ptr as *mut c_void,
                f32::cuda_data_type(),
                Order::Column,
            )
            .unwrap()
        };

        let c_rows: i64 = 2;
        let c_cols: i64 = 2;
        let c_data: Vec<f32> = vec![0.0; 4];
        let c_dev = stream.clone_htod(&c_data).unwrap();
        let mut mat_c = unsafe {
            DnMatDescr::new(
                c_rows,
                c_cols,
                c_rows,
                c_dev.cu_device_ptr as *mut c_void,
                f32::cuda_data_type(),
                Order::Column,
            )
            .unwrap()
        };

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        sparse
            .spmm(
                Operation::NonTranspose,
                Operation::NonTranspose,
                &alpha,
                &mat_a,
                &mat_b,
                &beta,
                &mut mat_c,
                sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
            )
            .unwrap();

        let c_host: Vec<f32> = stream.clone_dtoh(&c_dev).unwrap();
        // Column-major: [c00, c10, c01, c11] = [7, 6, 16, 15]
        assert!(
            (c_host[0] - 7.0).abs() < 1e-5,
            "C[0,0]: expected 7.0, got {}",
            c_host[0]
        );
        assert!(
            (c_host[1] - 6.0).abs() < 1e-5,
            "C[1,0]: expected 6.0, got {}",
            c_host[1]
        );
        assert!(
            (c_host[2] - 16.0).abs() < 1e-5,
            "C[0,1]: expected 16.0, got {}",
            c_host[2]
        );
        assert!(
            (c_host[3] - 15.0).abs() < 1e-5,
            "C[1,1]: expected 15.0, got {}",
            c_host[3]
        );
    }
}
