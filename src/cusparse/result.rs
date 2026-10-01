use core::ffi::c_void;
use core::mem::MaybeUninit;

use super::sys;

/// Wrapper around [sys::cusparseStatus_t]. See
/// nvidia's [cusparseStatus_t docs](https://docs.nvidia.com/cuda/cusparse/#cusparseStatus_t)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CusparseError(pub sys::cusparseStatus_t);

impl sys::cusparseStatus_t {
    #[inline]
    pub fn result(self) -> Result<(), CusparseError> {
        match self {
            sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS => Ok(()),
            _ => Err(CusparseError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CusparseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CusparseError {}

/// Features covering CUDA 11.040 through 11.080 (the "legacy generic API" range).
macro_rules! cfg_cuda_11 {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            $item
        )*
    };
}

/// Features covering CUDA 12.000+.
macro_rules! cfg_cuda_12 {
    ($($item:item)*) => {
        $(
            #[cfg(any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000",
                feature = "cuda-13010",
                feature = "cuda-13020"
            ))]
            $item
        )*
    };
}

// ---------------------------------------------------------------------------
// Handle management
// ---------------------------------------------------------------------------

/// See [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreate)
pub fn create() -> Result<sys::cusparseHandle_t, CusparseError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cusparseCreate(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroy)
///
/// # Safety
/// `handle` must not have been freed already.
pub unsafe fn destroy(handle: sys::cusparseHandle_t) -> Result<(), CusparseError> {
    sys::cusparseDestroy(handle).result()
}

/// Sets the stream cuSPARSE will use. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsesetstream)
///
/// # Safety
/// `handle` and `stream` must be valid.
pub unsafe fn set_stream(
    handle: sys::cusparseHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CusparseError> {
    sys::cusparseSetStream(handle, stream).result()
}

/// Gets the stream cuSPARSE is using. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsegetstream)
///
/// # Safety
/// `handle` must be valid.
pub unsafe fn get_stream(
    handle: sys::cusparseHandle_t,
) -> Result<sys::cudaStream_t, CusparseError> {
    let mut stream = MaybeUninit::uninit();
    sys::cusparseGetStream(handle, stream.as_mut_ptr()).result()?;
    Ok(stream.assume_init())
}

/// Gets the pointer mode. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsegetpointermode)
///
/// # Safety
/// `handle` must be valid.
pub unsafe fn get_pointer_mode(
    handle: sys::cusparseHandle_t,
) -> Result<sys::cusparsePointerMode_t, CusparseError> {
    let mut mode = MaybeUninit::uninit();
    sys::cusparseGetPointerMode(handle, mode.as_mut_ptr()).result()?;
    Ok(mode.assume_init())
}

/// Sets the pointer mode. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsesetpointermode)
///
/// # Safety
/// `handle` must be valid.
pub unsafe fn set_pointer_mode(
    handle: sys::cusparseHandle_t,
    mode: sys::cusparsePointerMode_t,
) -> Result<(), CusparseError> {
    sys::cusparseSetPointerMode(handle, mode).result()
}

// ---------------------------------------------------------------------------
// Sparse matrix descriptors (CSR, CSC, COO)
// ---------------------------------------------------------------------------

/// Creates a CSR sparse matrix descriptor. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreatecsr)
///
/// # Safety
/// `csr_row_offsets`, `csr_col_ind`, and `csr_values` must be valid device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_csr(
    rows: i64,
    cols: i64,
    nnz: i64,
    csr_row_offsets: *mut c_void,
    csr_col_ind: *mut c_void,
    csr_values: *mut c_void,
    csr_row_offsets_type: sys::cusparseIndexType_t,
    csr_col_ind_type: sys::cusparseIndexType_t,
    idx_base: sys::cusparseIndexBase_t,
    value_type: sys::cudaDataType,
) -> Result<sys::cusparseSpMatDescr_t, CusparseError> {
    let mut descr = MaybeUninit::uninit();
    sys::cusparseCreateCsr(
        descr.as_mut_ptr(),
        rows,
        cols,
        nnz,
        csr_row_offsets,
        csr_col_ind,
        csr_values,
        csr_row_offsets_type,
        csr_col_ind_type,
        idx_base,
        value_type,
    )
    .result()?;
    Ok(descr.assume_init())
}

/// Creates a CSC sparse matrix descriptor. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreatecsc)
///
/// # Safety
/// `csc_col_offsets`, `csc_row_ind`, and `csc_values` must be valid device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_csc(
    rows: i64,
    cols: i64,
    nnz: i64,
    csc_col_offsets: *mut c_void,
    csc_row_ind: *mut c_void,
    csc_values: *mut c_void,
    csc_col_offsets_type: sys::cusparseIndexType_t,
    csc_row_ind_type: sys::cusparseIndexType_t,
    idx_base: sys::cusparseIndexBase_t,
    value_type: sys::cudaDataType,
) -> Result<sys::cusparseSpMatDescr_t, CusparseError> {
    let mut descr = MaybeUninit::uninit();
    sys::cusparseCreateCsc(
        descr.as_mut_ptr(),
        rows,
        cols,
        nnz,
        csc_col_offsets,
        csc_row_ind,
        csc_values,
        csc_col_offsets_type,
        csc_row_ind_type,
        idx_base,
        value_type,
    )
    .result()?;
    Ok(descr.assume_init())
}

/// Creates a COO sparse matrix descriptor. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreatecoo)
///
/// # Safety
/// `coo_row_ind`, `coo_col_ind`, and `coo_values` must be valid device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_coo(
    rows: i64,
    cols: i64,
    nnz: i64,
    coo_row_ind: *mut c_void,
    coo_col_ind: *mut c_void,
    coo_values: *mut c_void,
    coo_idx_type: sys::cusparseIndexType_t,
    idx_base: sys::cusparseIndexBase_t,
    value_type: sys::cudaDataType,
) -> Result<sys::cusparseSpMatDescr_t, CusparseError> {
    let mut descr = MaybeUninit::uninit();
    sys::cusparseCreateCoo(
        descr.as_mut_ptr(),
        rows,
        cols,
        nnz,
        coo_row_ind,
        coo_col_ind,
        coo_values,
        coo_idx_type,
        idx_base,
        value_type,
    )
    .result()?;
    Ok(descr.assume_init())
}

cfg_cuda_11! {
    /// Destroys a sparse matrix descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroyspmat)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_sp_mat(descr: sys::cusparseSpMatDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroySpMat(descr).result()
    }
}
cfg_cuda_12! {
    /// Destroys a sparse matrix descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroyspmat)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_sp_mat(descr: sys::cusparseSpMatDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroySpMat(descr as sys::cusparseConstSpMatDescr_t).result()
    }
}

cfg_cuda_11! {
    /// Gets the size (rows, cols, nnz) of a sparse matrix. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmatgetsize)
    ///
    /// # Safety
    /// `descr` must be a valid sparse matrix descriptor.
    pub unsafe fn sp_mat_get_size(
        descr: sys::cusparseSpMatDescr_t,
    ) -> Result<(i64, i64, i64), CusparseError> {
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut nnz = MaybeUninit::uninit();
        sys::cusparseSpMatGetSize(descr, rows.as_mut_ptr(), cols.as_mut_ptr(), nnz.as_mut_ptr())
            .result()?;
        Ok((rows.assume_init(), cols.assume_init(), nnz.assume_init()))
    }
}
cfg_cuda_12! {
    /// Gets the size (rows, cols, nnz) of a sparse matrix. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmatgetsize)
    ///
    /// # Safety
    /// `descr` must be a valid sparse matrix descriptor.
    pub unsafe fn sp_mat_get_size(
        descr: sys::cusparseSpMatDescr_t,
    ) -> Result<(i64, i64, i64), CusparseError> {
        let mut rows = MaybeUninit::uninit();
        let mut cols = MaybeUninit::uninit();
        let mut nnz = MaybeUninit::uninit();
        sys::cusparseSpMatGetSize(
            descr as sys::cusparseConstSpMatDescr_t,
            rows.as_mut_ptr(),
            cols.as_mut_ptr(),
            nnz.as_mut_ptr(),
        )
        .result()?;
        Ok((rows.assume_init(), cols.assume_init(), nnz.assume_init()))
    }
}

cfg_cuda_11! {
    /// Gets the format of a sparse matrix. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmatgetformat)
    ///
    /// # Safety
    /// `descr` must be a valid sparse matrix descriptor.
    pub unsafe fn sp_mat_get_format(
        descr: sys::cusparseSpMatDescr_t,
    ) -> Result<sys::cusparseFormat_t, CusparseError> {
        let mut format = MaybeUninit::uninit();
        sys::cusparseSpMatGetFormat(descr, format.as_mut_ptr()).result()?;
        Ok(format.assume_init())
    }
}
cfg_cuda_12! {
    /// Gets the format of a sparse matrix. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmatgetformat)
    ///
    /// # Safety
    /// `descr` must be a valid sparse matrix descriptor.
    pub unsafe fn sp_mat_get_format(
        descr: sys::cusparseSpMatDescr_t,
    ) -> Result<sys::cusparseFormat_t, CusparseError> {
        let mut format = MaybeUninit::uninit();
        sys::cusparseSpMatGetFormat(
            descr as sys::cusparseConstSpMatDescr_t,
            format.as_mut_ptr(),
        )
        .result()?;
        Ok(format.assume_init())
    }
}

// ---------------------------------------------------------------------------
// Sparse vector descriptors
// ---------------------------------------------------------------------------

/// Creates a sparse vector descriptor. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreatespvec)
///
/// # Safety
/// `indices` and `values` must be valid device pointers.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_sp_vec(
    size: i64,
    nnz: i64,
    indices: *mut c_void,
    values: *mut c_void,
    idx_type: sys::cusparseIndexType_t,
    idx_base: sys::cusparseIndexBase_t,
    value_type: sys::cudaDataType,
) -> Result<sys::cusparseSpVecDescr_t, CusparseError> {
    let mut descr = MaybeUninit::uninit();
    sys::cusparseCreateSpVec(
        descr.as_mut_ptr(),
        size,
        nnz,
        indices,
        values,
        idx_type,
        idx_base,
        value_type,
    )
    .result()?;
    Ok(descr.assume_init())
}

cfg_cuda_11! {
    /// Destroys a sparse vector descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroyspvec)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_sp_vec(descr: sys::cusparseSpVecDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroySpVec(descr).result()
    }
}
cfg_cuda_12! {
    /// Destroys a sparse vector descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroyspvec)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_sp_vec(descr: sys::cusparseSpVecDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroySpVec(descr as sys::cusparseConstSpVecDescr_t).result()
    }
}

// ---------------------------------------------------------------------------
// Dense descriptors
// ---------------------------------------------------------------------------

/// Creates a dense vector descriptor. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreatednvec)
///
/// # Safety
/// `values` must be a valid device pointer.
pub unsafe fn create_dn_vec(
    size: i64,
    values: *mut c_void,
    value_type: sys::cudaDataType,
) -> Result<sys::cusparseDnVecDescr_t, CusparseError> {
    let mut descr = MaybeUninit::uninit();
    sys::cusparseCreateDnVec(descr.as_mut_ptr(), size, values, value_type).result()?;
    Ok(descr.assume_init())
}

cfg_cuda_11! {
    /// Destroys a dense vector descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroydnvec)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_dn_vec(descr: sys::cusparseDnVecDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroyDnVec(descr).result()
    }
}
cfg_cuda_12! {
    /// Destroys a dense vector descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroydnvec)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_dn_vec(descr: sys::cusparseDnVecDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroyDnVec(descr as sys::cusparseConstDnVecDescr_t).result()
    }
}

/// Creates a dense matrix descriptor. See
/// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsecreatedmnat)
///
/// # Safety
/// `values` must be a valid device pointer.
pub unsafe fn create_dn_mat(
    rows: i64,
    cols: i64,
    ld: i64,
    values: *mut c_void,
    value_type: sys::cudaDataType,
    order: sys::cusparseOrder_t,
) -> Result<sys::cusparseDnMatDescr_t, CusparseError> {
    let mut descr = MaybeUninit::uninit();
    sys::cusparseCreateDnMat(
        descr.as_mut_ptr(),
        rows,
        cols,
        ld,
        values,
        value_type,
        order,
    )
    .result()?;
    Ok(descr.assume_init())
}

cfg_cuda_11! {
    /// Destroys a dense matrix descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroydnmat)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_dn_mat(descr: sys::cusparseDnMatDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroyDnMat(descr).result()
    }
}
cfg_cuda_12! {
    /// Destroys a dense matrix descriptor. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsedestroydnmat)
    ///
    /// # Safety
    /// `descr` must not have been freed already.
    pub unsafe fn destroy_dn_mat(descr: sys::cusparseDnMatDescr_t) -> Result<(), CusparseError> {
        sys::cusparseDestroyDnMat(descr as sys::cusparseConstDnMatDescr_t).result()
    }
}

// ---------------------------------------------------------------------------
// Core operations: SpMV
// ---------------------------------------------------------------------------

cfg_cuda_11! {
    /// Computes the required buffer size for [spmv]. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmv_buffer_size(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        vec_x: sys::cusparseDnVecDescr_t,
        beta: *const c_void,
        vec_y: sys::cusparseDnVecDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMVAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseSpMV_bufferSize(
            handle,
            op_a,
            alpha,
            mat_a,
            vec_x,
            beta,
            vec_y,
            compute_type,
            alg,
            buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}
cfg_cuda_12! {
    /// Computes the required buffer size for [spmv]. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmv_buffer_size(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        vec_x: sys::cusparseDnVecDescr_t,
        beta: *const c_void,
        vec_y: sys::cusparseDnVecDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMVAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseSpMV_bufferSize(
            handle,
            op_a,
            alpha,
            mat_a as sys::cusparseConstSpMatDescr_t,
            vec_x as sys::cusparseConstDnVecDescr_t,
            beta,
            vec_y,
            compute_type,
            alg,
            buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}

cfg_cuda_11! {
    /// Sparse matrix - dense vector multiplication: y = alpha * op(A) * x + beta * y. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [spmv_buffer_size].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmv(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        vec_x: sys::cusparseDnVecDescr_t,
        beta: *const c_void,
        vec_y: sys::cusparseDnVecDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMVAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseSpMV(
            handle,
            op_a,
            alpha,
            mat_a,
            vec_x,
            beta,
            vec_y,
            compute_type,
            alg,
            external_buffer,
        )
        .result()
    }
}
cfg_cuda_12! {
    /// Sparse matrix - dense vector multiplication: y = alpha * op(A) * x + beta * y. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [spmv_buffer_size].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmv(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        vec_x: sys::cusparseDnVecDescr_t,
        beta: *const c_void,
        vec_y: sys::cusparseDnVecDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMVAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseSpMV(
            handle,
            op_a,
            alpha,
            mat_a as sys::cusparseConstSpMatDescr_t,
            vec_x as sys::cusparseConstDnVecDescr_t,
            beta,
            vec_y,
            compute_type,
            alg,
            external_buffer,
        )
        .result()
    }
}

// ---------------------------------------------------------------------------
// Core operations: SpMM
// ---------------------------------------------------------------------------

cfg_cuda_11! {
    /// Computes the required buffer size for [spmm]. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmm)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmm_buffer_size(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        op_b: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        beta: *const c_void,
        mat_c: sys::cusparseDnMatDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMMAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseSpMM_bufferSize(
            handle,
            op_a,
            op_b,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            compute_type,
            alg,
            buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}
cfg_cuda_12! {
    /// Computes the required buffer size for [spmm]. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmm)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmm_buffer_size(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        op_b: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        beta: *const c_void,
        mat_c: sys::cusparseDnMatDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMMAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseSpMM_bufferSize(
            handle,
            op_a,
            op_b,
            alpha,
            mat_a as sys::cusparseConstSpMatDescr_t,
            mat_b as sys::cusparseConstDnMatDescr_t,
            beta,
            mat_c,
            compute_type,
            alg,
            buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}

cfg_cuda_11! {
    /// Sparse matrix - dense matrix multiplication: C = alpha * op(A) * op(B) + beta * C. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmm)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [spmm_buffer_size].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmm(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        op_b: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        beta: *const c_void,
        mat_c: sys::cusparseDnMatDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMMAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseSpMM(
            handle,
            op_a,
            op_b,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            compute_type,
            alg,
            external_buffer,
        )
        .result()
    }
}
cfg_cuda_12! {
    /// Sparse matrix - dense matrix multiplication: C = alpha * op(A) * op(B) + beta * C. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparsespmm)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [spmm_buffer_size].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn spmm(
        handle: sys::cusparseHandle_t,
        op_a: sys::cusparseOperation_t,
        op_b: sys::cusparseOperation_t,
        alpha: *const c_void,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        beta: *const c_void,
        mat_c: sys::cusparseDnMatDescr_t,
        compute_type: sys::cudaDataType,
        alg: sys::cusparseSpMMAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseSpMM(
            handle,
            op_a,
            op_b,
            alpha,
            mat_a as sys::cusparseConstSpMatDescr_t,
            mat_b as sys::cusparseConstDnMatDescr_t,
            beta,
            mat_c,
            compute_type,
            alg,
            external_buffer,
        )
        .result()
    }
}

// ---------------------------------------------------------------------------
// Format conversion: SparseToDense
// ---------------------------------------------------------------------------

cfg_cuda_11! {
    /// Computes the required buffer size for [sparse_to_dense]. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseSparseToDense)
    ///
    /// # Safety
    /// All handles and descriptors must be valid.
    pub unsafe fn sparse_to_dense_buffer_size(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        alg: sys::cusparseSparseToDenseAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseSparseToDense_bufferSize(
            handle, mat_a, mat_b, alg, buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}
cfg_cuda_12! {
    /// Computes the required buffer size for [sparse_to_dense]. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseSparseToDense)
    ///
    /// # Safety
    /// All handles and descriptors must be valid.
    pub unsafe fn sparse_to_dense_buffer_size(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        alg: sys::cusparseSparseToDenseAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseSparseToDense_bufferSize(
            handle,
            mat_a as sys::cusparseConstSpMatDescr_t,
            mat_b,
            alg,
            buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}

cfg_cuda_11! {
    /// Converts a sparse matrix to dense format. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseSparseToDense)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [sparse_to_dense_buffer_size].
    pub unsafe fn sparse_to_dense(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        alg: sys::cusparseSparseToDenseAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseSparseToDense(handle, mat_a, mat_b, alg, external_buffer).result()
    }
}
cfg_cuda_12! {
    /// Converts a sparse matrix to dense format. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseSparseToDense)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [sparse_to_dense_buffer_size].
    pub unsafe fn sparse_to_dense(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseSpMatDescr_t,
        mat_b: sys::cusparseDnMatDescr_t,
        alg: sys::cusparseSparseToDenseAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseSparseToDense(
            handle,
            mat_a as sys::cusparseConstSpMatDescr_t,
            mat_b,
            alg,
            external_buffer,
        )
        .result()
    }
}

// ---------------------------------------------------------------------------
// Format conversion: DenseToSparse
// ---------------------------------------------------------------------------

cfg_cuda_11! {
    /// Computes the required buffer size for dense-to-sparse conversion. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseDenseToSparse)
    ///
    /// # Safety
    /// All handles and descriptors must be valid.
    pub unsafe fn dense_to_sparse_buffer_size(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseDnMatDescr_t,
        mat_b: sys::cusparseSpMatDescr_t,
        alg: sys::cusparseDenseToSparseAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseDenseToSparse_bufferSize(
            handle, mat_a, mat_b, alg, buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}
cfg_cuda_12! {
    /// Computes the required buffer size for dense-to-sparse conversion. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseDenseToSparse)
    ///
    /// # Safety
    /// All handles and descriptors must be valid.
    pub unsafe fn dense_to_sparse_buffer_size(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseDnMatDescr_t,
        mat_b: sys::cusparseSpMatDescr_t,
        alg: sys::cusparseDenseToSparseAlg_t,
    ) -> Result<usize, CusparseError> {
        let mut buffer_size = MaybeUninit::uninit();
        sys::cusparseDenseToSparse_bufferSize(
            handle,
            mat_a as sys::cusparseConstDnMatDescr_t,
            mat_b,
            alg,
            buffer_size.as_mut_ptr(),
        )
        .result()?;
        Ok(buffer_size.assume_init())
    }
}

cfg_cuda_11! {
    /// Performs analysis for dense-to-sparse conversion. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseDenseToSparse)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [dense_to_sparse_buffer_size].
    pub unsafe fn dense_to_sparse_analysis(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseDnMatDescr_t,
        mat_b: sys::cusparseSpMatDescr_t,
        alg: sys::cusparseDenseToSparseAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseDenseToSparse_analysis(handle, mat_a, mat_b, alg, external_buffer).result()
    }
}
cfg_cuda_12! {
    /// Performs analysis for dense-to-sparse conversion. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseDenseToSparse)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. `external_buffer` must
    /// be at least as large as reported by [dense_to_sparse_buffer_size].
    pub unsafe fn dense_to_sparse_analysis(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseDnMatDescr_t,
        mat_b: sys::cusparseSpMatDescr_t,
        alg: sys::cusparseDenseToSparseAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseDenseToSparse_analysis(
            handle,
            mat_a as sys::cusparseConstDnMatDescr_t,
            mat_b,
            alg,
            external_buffer,
        )
        .result()
    }
}

cfg_cuda_11! {
    /// Executes the dense-to-sparse conversion. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseDenseToSparse)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. [dense_to_sparse_analysis]
    /// must have been called first. `external_buffer` must be at least as large as
    /// reported by [dense_to_sparse_buffer_size].
    pub unsafe fn dense_to_sparse_convert(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseDnMatDescr_t,
        mat_b: sys::cusparseSpMatDescr_t,
        alg: sys::cusparseDenseToSparseAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseDenseToSparse_convert(handle, mat_a, mat_b, alg, external_buffer).result()
    }
}
cfg_cuda_12! {
    /// Executes the dense-to-sparse conversion. See
    /// [cuda docs](https://docs.nvidia.com/cuda/cusparse/#cusparseDenseToSparse)
    ///
    /// # Safety
    /// All handles, descriptors, and pointers must be valid. [dense_to_sparse_analysis]
    /// must have been called first. `external_buffer` must be at least as large as
    /// reported by [dense_to_sparse_buffer_size].
    pub unsafe fn dense_to_sparse_convert(
        handle: sys::cusparseHandle_t,
        mat_a: sys::cusparseDnMatDescr_t,
        mat_b: sys::cusparseSpMatDescr_t,
        alg: sys::cusparseDenseToSparseAlg_t,
        external_buffer: *mut c_void,
    ) -> Result<(), CusparseError> {
        sys::cusparseDenseToSparse_convert(
            handle,
            mat_a as sys::cusparseConstDnMatDescr_t,
            mat_b,
            alg,
            external_buffer,
        )
        .result()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_destroy() -> Result<(), CusparseError> {
        let handle = create()?;
        unsafe { destroy(handle) }?;
        Ok(())
    }
}
