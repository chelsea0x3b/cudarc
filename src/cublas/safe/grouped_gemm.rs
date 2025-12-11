//! Grouped GEMM matrix multiplication operations.
//! Safe version of cublasGemmGroupedBatchedEx.
//! ref: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex
use std::vec::Vec;

use crate::{
    cublas::{result::CublasError, sys, CudaBlas},
    driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr},
};

pub trait GroupedGemmDtype: DeviceRepr {
    type ComputeType: DeviceRepr + Copy;

    fn data_type() -> sys::cudaDataType_t;
    fn compute_type() -> sys::cublasComputeType_t;
}

#[cfg(feature = "f16")]
impl GroupedGemmDtype for half::f16 {
    type ComputeType = f32;

    fn data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16F
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

#[cfg(feature = "f16")]
impl GroupedGemmDtype for half::bf16 {
    type ComputeType = f32;

    fn data_type() -> sys::cudaDataType_t {
        sys::cudaDataType_t::CUDA_R_16BF
    }

    fn compute_type() -> sys::cublasComputeType_t {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

pub struct GroupedGemmConfig<T: GroupedGemmDtype> {
    /// transb for each group (len = group count)
    pub transbs: Vec<sys::cublasOperation_t>,
    /// transa for each group (len = group count)
    pub transas: Vec<sys::cublasOperation_t>,

    /// m for each group (len = group count)
    pub ms: Vec<usize>,
    /// n for each group (len = group count)
    pub ns: Vec<usize>,
    /// k for each group (len = group count)
    pub ks: Vec<usize>,

    /// alpha for each group, must be same as compute type
    /// for data type (len = group count)
    pub alphas: Vec<T::ComputeType>,

    /// beta for each group, must be same as compute type
    /// for data type (len = group count)
    pub betas: Vec<T::ComputeType>,

    /// A leading dim for each group (len = group count)
    pub ldas: Vec<usize>,
    /// B leading dim for each group (len = group count)
    pub ldbs: Vec<usize>,
    /// C leading dim for each group (len = group count)
    pub ldcs: Vec<usize>,

    /// number of problems in each group (len = group count)
    pub problem_sizes: Vec<usize>,
}

impl<T: GroupedGemmDtype> GroupedGemmConfig<T> {
    pub fn problem_count(&self) -> usize {
        self.problem_sizes.iter().sum()
    }

    pub fn group_count(&self) -> usize {
        self.problem_sizes.len()
    }

    #[inline]
    fn validate(&self) {
        let group_count = self.group_count();
        assert_eq!(self.transbs.len(), group_count);
        assert_eq!(self.transas.len(), group_count);
        assert_eq!(self.ms.len(), group_count);
        assert_eq!(self.ns.len(), group_count);
        assert_eq!(self.ks.len(), group_count);
        assert_eq!(self.alphas.len(), group_count);
        assert_eq!(self.betas.len(), group_count);
        assert_eq!(self.ldas.len(), group_count);
        assert_eq!(self.ldbs.len(), group_count);
        assert_eq!(self.ldcs.len(), group_count);
    }
}

pub trait GroupedGemm<T: GroupedGemmDtype> {
    /// Grouped matrix multiplication using device slices.
    ///
    /// This will incur a htod copy because the pointer to matrix pointers
    /// must be on-device.
    ///
    /// * `config` – sizes, leading dimensions, scalars, and counts per group.
    /// * `a_slices` – device slices for matrices A for every problem (len = problem count).
    /// * `b_slices` – device slices for matrices B for every problem (len = problem count).
    /// * `c_slices` – device slices for output matrices C for every problem (len = problem count).
    fn grouped_gemm(
        &self,
        config: GroupedGemmConfig<T>,
        a_slices: &[&CudaSlice<T>],
        b_slices: &[&CudaSlice<T>],
        c_slices: &mut [&mut CudaSlice<T>],
    ) -> Result<(), CublasError>;
}

impl<T: GroupedGemmDtype> GroupedGemm<T> for CudaBlas {
    fn grouped_gemm(
        &self,
        config: GroupedGemmConfig<T>,
        a_slices: &[&CudaSlice<T>],
        b_slices: &[&CudaSlice<T>],
        c_slices: &mut [&mut CudaSlice<T>],
    ) -> Result<(), CublasError> {
        config.validate();
        assert_eq!(a_slices.len(), config.problem_count());
        assert_eq!(b_slices.len(), config.problem_count());
        assert_eq!(c_slices.len(), config.problem_count());

        let (a_ptrs, _a_guard_vec): (Vec<u64>, Vec<_>) =
            a_slices.iter().map(|s| s.device_ptr(&self.stream)).unzip();

        let (b_ptrs, _b_guard_vec): (Vec<u64>, Vec<_>) =
            b_slices.iter().map(|s| s.device_ptr(&self.stream)).unzip();

        let (mut c_ptrs, _c_guard_vec): (Vec<u64>, Vec<_>) = c_slices
            .iter_mut()
            .map(|s| s.device_ptr_mut(&self.stream))
            .unzip();

        // // TODO coalesce these allocations
        // let a_ptrs_dev = htod_copy(&self.stream, &a_ptrs_host);
        // let b_ptrs_dev = htod_copy(&self.stream, &b_ptrs_host);
        // let c_ptrs_dev = htod_copy(&self.stream, &c_ptrs_host);

        // let (a_ptrs, _a_guard) = a_ptrs_dev.device_ptr(&self.stream);
        // let (b_ptrs, _b_guard) = b_ptrs_dev.device_ptr(&self.stream);
        // let (c_ptrs, _c_guard) = c_ptrs_dev.device_ptr(&self.stream);

        let cuda_dtype = T::data_type();
        let group_count = config.group_count();

        // For CUBLAS_COMPUTE_32F, alpha and beta must be f32
        let alpha_f32: Vec<T::ComputeType> =
            config.alphas.iter().map(|x| *x as T::ComputeType).collect();
        let beta_f32: Vec<T::ComputeType> =
            config.betas.iter().map(|x| *x as T::ComputeType).collect();

        let m_array: Vec<i32> = config.ms.iter().map(|&x| x as i32).collect();
        let n_array: Vec<i32> = config.ns.iter().map(|&x| x as i32).collect();
        let k_array: Vec<i32> = config.ks.iter().map(|&x| x as i32).collect();
        let lda_array: Vec<i32> = config.ldas.iter().map(|&x| x as i32).collect();
        let ldb_array: Vec<i32> = config.ldbs.iter().map(|&x| x as i32).collect();
        let ldc_array: Vec<i32> = config.ldcs.iter().map(|&x| x as i32).collect();
        let group_size: Vec<i32> = config.problem_sizes.iter().map(|&x| x as i32).collect();

        unsafe {
            sys::cublasGemmGroupedBatchedEx(
                self.handle,
                config.transas.as_ptr(),
                config.transbs.as_ptr(),
                m_array.as_ptr(),
                n_array.as_ptr(),
                k_array.as_ptr(),
                alpha_f32.as_ptr() as _,
                a_ptrs.as_ptr() as _,
                cuda_dtype,
                lda_array.as_ptr(),
                b_ptrs.as_ptr() as _,
                cuda_dtype,
                ldb_array.as_ptr(),
                beta_f32.as_ptr() as _,
                c_ptrs.as_mut_ptr() as _,
                cuda_dtype,
                ldc_array.as_ptr(),
                group_count as i32,
                group_size.as_ptr(),
                T::compute_type(),
            )
            .result()?;
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused)]

    use super::*;
    use crate::driver::CudaContext;
    use std::vec;

    #[test]
    #[cfg(feature = "f16")]
    fn test_grouped_gemm_f16() {
        use half::f16;
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let handle = CudaBlas::new(stream.clone()).unwrap();

        // Group 0: 2 problems, 2x2
        // Problem 0
        // A = | 1 2 |  B = | 5 6 |
        //     | 3 4 |      | 7 8 |
        // C = | 19 22 |
        //     | 43 50 |
        // (Column-Major storage)
        let a0_host = [1.0, 3.0, 2.0, 4.0].map(f16::from_f32);
        let b0_host = [5.0, 7.0, 6.0, 8.0].map(f16::from_f32);
        // Problem 1
        // A = | 5 6 |  B = | 9 10 |
        //     | 7 8 |      | 11 12 |
        // C = | 111 122 |
        //     | 151 166 |
        let a1_host = [5.0, 7.0, 6.0, 8.0].map(f16::from_f32);
        let b1_host = [9.0, 11.0, 10.0, 12.0].map(f16::from_f32);

        // Group 1: 1 problem, 3x3
        // Problem 2
        let a2_host = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0].map(f16::from_f32);
        let b2_host = [4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0].map(f16::from_f32);

        let a0 = stream.clone_htod(&a0_host).unwrap();
        let b0 = stream.clone_htod(&b0_host).unwrap();
        let a1 = stream.clone_htod(&a1_host).unwrap();
        let b1 = stream.clone_htod(&b1_host).unwrap();
        let a2 = stream.clone_htod(&a2_host).unwrap();
        let b2 = stream.clone_htod(&b2_host).unwrap();
        let mut c0 = stream.alloc_zeros::<f16>(4).unwrap();
        let mut c1 = stream.alloc_zeros::<f16>(4).unwrap();
        let mut c2 = stream.alloc_zeros::<f16>(9).unwrap();

        let config = GroupedGemmConfig {
            transbs: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            transas: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            ms: vec![2, 3],
            ns: vec![2, 3],
            ks: vec![2, 3],
            alphas: vec![1.0; 2],
            betas: vec![0.0; 2],
            ldas: vec![2, 3],
            ldbs: vec![2, 3],
            ldcs: vec![2, 3],
            problem_sizes: vec![2, 1],
        };

        handle
            .grouped_gemm(
                config,
                &[&a0, &a1, &a2],
                &[&b0, &b1, &b2],
                &mut [&mut c0, &mut c1, &mut c2],
            )
            .unwrap();

        let c0_host = stream.clone_dtoh(&c0).unwrap();
        let c1_host = stream.clone_dtoh(&c1).unwrap();
        let c2_host = stream.clone_dtoh(&c2).unwrap();

        let expected_c0 = [19.0, 43.0, 22.0, 50.0].map(f16::from_f32);
        let expected_c1 = [111.0, 151.0, 122.0, 166.0].map(f16::from_f32);
        let expected_c2 =
            [48.0, 111.0, 174.0, 54.0, 126.0, 198.0, 60.0, 141.0, 222.0].map(f16::from_f32);

        assert_eq!(c0_host, expected_c0);
        assert_eq!(c1_host, expected_c1);
        assert_eq!(c2_host, expected_c2);
    }

    #[test]
    #[cfg(feature = "f16")]
    fn test_grouped_gemm_raw_bf16() {
        use half::bf16;
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let handle = CudaBlas::new(stream.clone()).unwrap();

        // Group 0: 2 problems, 2x2
        // Problem 0
        // A = | 1 2 |  B = | 5 6 |
        //     | 3 4 |      | 7 8 |
        // C = | 19 22 |
        //     | 43 50 |
        // (Column-Major storage)
        let a0_host = [1.0, 3.0, 2.0, 4.0].map(bf16::from_f32);
        let b0_host = [5.0, 7.0, 6.0, 8.0].map(bf16::from_f32);
        // Problem 1
        // A = | 5 6 |  B = | 9 10 |
        //     | 7 8 |      | 11 12 |
        // C = | 111 122 |
        //     | 151 166 |
        let a1_host = [5.0, 7.0, 6.0, 8.0].map(bf16::from_f32);
        let b1_host = [9.0, 11.0, 10.0, 12.0].map(bf16::from_f32);

        // Group 1: 1 problem, 3x3
        // Problem 2
        let a2_host = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0].map(bf16::from_f32);
        let b2_host = [4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0].map(bf16::from_f32);

        let a0 = stream.clone_htod(&a0_host).unwrap();
        let b0 = stream.clone_htod(&b0_host).unwrap();
        let a1 = stream.clone_htod(&a1_host).unwrap();
        let b1 = stream.clone_htod(&b1_host).unwrap();
        let a2 = stream.clone_htod(&a2_host).unwrap();
        let b2 = stream.clone_htod(&b2_host).unwrap();

        let mut c0 = stream.alloc_zeros::<bf16>(4).unwrap();
        let mut c1 = stream.alloc_zeros::<bf16>(4).unwrap();
        let mut c2 = stream.alloc_zeros::<bf16>(9).unwrap();

        let config = GroupedGemmConfig {
            transbs: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            transas: vec![sys::cublasOperation_t::CUBLAS_OP_N; 2],
            ms: vec![2, 3],
            ns: vec![2, 3],
            ks: vec![2, 3],
            alphas: vec![1.0; 2],
            betas: vec![0.0; 2],
            ldas: vec![2, 3],
            ldbs: vec![2, 3],
            ldcs: vec![2, 3],
            problem_sizes: vec![2, 1],
        };

        handle
            .grouped_gemm(
                config,
                &[&a0, &a1, &a2],
                &[&b0, &b1, &b2],
                &mut [&mut c0, &mut c1, &mut c2],
            )
            .unwrap();

        let c0_host = stream.clone_dtoh(&c0).unwrap();
        let c1_host = stream.clone_dtoh(&c1).unwrap();
        let c2_host = stream.clone_dtoh(&c2).unwrap();

        let expected_c0 = [19.0, 43.0, 22.0, 50.0].map(bf16::from_f32);
        let expected_c1 = [111.0, 151.0, 122.0, 166.0].map(bf16::from_f32);
        let expected_c2 =
            [48.0, 111.0, 174.0, 54.0, 126.0, 198.0, 60.0, 141.0, 222.0].map(bf16::from_f32);

        assert_eq!(c0_host, expected_c0);
        assert_eq!(c1_host, expected_c1);
        assert_eq!(c2_host, expected_c2);
    }
}
