//! Factor column-major f64 matrices without introducing a new safe cuSOLVER API.
//!
//! Run with a CUDA installation available to the dynamic loader:
//! cargo run --example cusolver-qr --no-default-features \
//!   --features std,driver,cusolver,dynamic-loading,cuda-13000
//!
//! API contract: https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-geqrf

use cudarc::{
    cusolver::{sys, DnHandle},
    driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut},
};
use std::{error::Error, mem::size_of, sync::Arc};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn dimensions(rows: usize, cols: usize, len: usize) -> Result<(i32, i32)> {
    if rows == 0 || cols == 0 {
        return Err("this example requires nonempty matrices".into());
    }
    let m = i32::try_from(rows)?;
    let n = i32::try_from(cols)?;
    let elements = rows.checked_mul(cols).ok_or("matrix size overflow")?;
    elements
        .checked_mul(size_of::<f64>())
        .ok_or("matrix byte size overflow")?;
    if len != elements {
        return Err("expected rows * cols column-major elements".into());
    }
    Ok((m, n))
}

fn factorize(
    handle: &DnHandle,
    transfer: &Arc<CudaStream>,
    rows: usize,
    cols: usize,
    input: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (m, n) = dimensions(rows, cols, input.len())?;
    let stream = handle.stream();
    if stream.context() != transfer.context() {
        return Err("the cuSOLVER handle and transfer stream must share a context".into());
    }
    // Copy into a separate factor buffer: geqrf overwrites A. Allocation/upload
    // uses another stream so the pointer guards also demonstrate event tracking.
    let original = transfer.clone_htod(input)?;
    let mut a = transfer.clone_dtod(&original)?;
    let mut tau = transfer.alloc_zeros::<f64>(rows.min(cols))?;
    let mut info = transfer.alloc_zeros::<i32>(1)?;
    let mut lwork = 0;
    {
        let (a_ptr, _record_a) = a.device_ptr(stream);
        // SAFETY: handle and stream share a context with a; lda = m and a has
        // m*n elements. The workspace query does not modify A despite its *mut
        // signature. Keep the read guard alive through this call.
        unsafe {
            sys::cusolverDnDgeqrf_bufferSize(handle.cu(), m, n, a_ptr as *mut f64, m, &mut lwork)
        }
        .result()?;
    }
    let work_len = usize::try_from(lwork)?;
    if work_len == 0 {
        return Err("cuSOLVER returned an empty workspace for a nonempty matrix".into());
    }
    // This API returns a count of f64 elements, not a byte count.
    work_len
        .checked_mul(size_of::<f64>())
        .ok_or("workspace byte size overflow")?;
    let mut workspace = stream.alloc_zeros::<f64>(work_len)?;
    {
        let (a_ptr, _record_a) = a.device_ptr_mut(stream);
        let (tau_ptr, _record_tau) = tau.device_ptr_mut(stream);
        let (work_ptr, _record_work) = workspace.device_ptr_mut(stream);
        let (info_ptr, _record_info) = info.device_ptr_mut(stream);
        // SAFETY: all buffers are distinct, sized as required, and in the
        // handle's context. These guards live until AFTER the call is queued;
        // their drops record events needed by subsequent accesses on transfer.
        unsafe {
            sys::cusolverDnDgeqrf(
                handle.cu(),
                m,
                n,
                a_ptr as *mut f64,
                m,
                tau_ptr as *mut f64,
                work_ptr as *mut f64,
                lwork,
                info_ptr as *mut i32,
            )
        }
        .result()?;
    }
    // Host readback waits for the recorded writes; no global synchronization
    // is needed between allocation, the workspace query and factorization.
    let info = transfer.clone_dtoh(&info)?[0];
    if info != 0 {
        return Err(format!("geqrf devInfo = {info} (negative means an invalid argument)").into());
    }
    let packed = transfer.clone_dtoh(&a)?;
    let tau = transfer.clone_dtoh(&tau)?;
    assert_eq!(transfer.clone_dtoh(&original)?, input);
    Ok((packed, tau))
}

// A contains R on/above the diagonal and Householder vectors below it. Each
// vector has an implicit 1 on its diagonal; TAU stores its scale. Q is H0*H1*...
// (not their reverse), so apply reflectors to the identity in reverse order.
fn unpack(rows: usize, cols: usize, packed: &[f64], tau: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0; rows * rows];
    for i in 0..rows {
        q[i + i * rows] = 1.0;
    }
    for j in (0..rows.min(cols)).rev() {
        for col in 0..rows {
            let dot = q[j + col * rows]
                + ((j + 1)..rows)
                    .map(|row| packed[row + j * rows] * q[row + col * rows])
                    .sum::<f64>();
            q[j + col * rows] -= tau[j] * dot;
            for row in (j + 1)..rows {
                q[row + col * rows] -= tau[j] * packed[row + j * rows] * dot;
            }
        }
    }
    let mut r = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows.min(col + 1) {
            r[row + col * rows] = packed[row + col * rows];
        }
    }
    (q, r)
}

fn residuals(rows: usize, cols: usize, input: &[f64], q: &[f64], r: &[f64]) -> (f64, f64) {
    let mut reconstruction = 0.0;
    for col in 0..cols {
        for row in 0..rows {
            let qr = (0..rows)
                .map(|i| q[row + i * rows] * r[i + col * rows])
                .sum::<f64>();
            reconstruction += (input[row + col * rows] - qr).powi(2);
        }
    }
    let mut orthogonality = 0.0;
    for col in 0..rows {
        for row in 0..rows {
            let qtq = (0..rows)
                .map(|i| q[i + row * rows] * q[i + col * rows])
                .sum::<f64>();
            orthogonality += (qtq - if row == col { 1.0 } else { 0.0 }).powi(2);
        }
    }
    let norm = input.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
    (reconstruction.sqrt() / norm, orthogonality.sqrt())
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let transfer = ctx.default_stream();
    let stream = ctx.new_stream()?;
    let handle = DnHandle::new(stream)?;
    for (name, rows, cols, input) in [
        (
            "square",
            3,
            3,
            vec![12., 6., -4., -51., 167., 24., 4., -68., -41.],
        ),
        ("tall", 4, 2, vec![1., 2., 3., 4., 4., 1., 3., 2.]),
        ("wide", 2, 4, vec![1., 2., 3., 4., -2., 1., 0., 5.]),
        ("rank deficient", 3, 2, vec![1., 2., 3., 2., 4., 6.]),
    ] {
        let (packed, tau) = factorize(&handle, &transfer, rows, cols, &input)?;
        let (q, r) = unpack(rows, cols, &packed, &tau);
        let (reconstruction, orthogonality) = residuals(rows, cols, &input, &q, &r);
        println!("{name}: normalized |A-QR| = {reconstruction:e}, |Q^TQ-I| = {orthogonality:e}");
        assert!(reconstruction < 1e-12);
        assert!(orthogonality < 1e-12);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_dimensions_before_device_access() {
        for (rows, cols, len) in [(0, 1, 0), (1, 0, 0), (2, 3, 5), (usize::MAX, 2, 0)] {
            assert!(dimensions(rows, cols, len).is_err());
        }
        assert_eq!(dimensions(3, 2, 6).unwrap(), (3, 2));
    }

    #[test]
    fn householder_order_is_not_transposed() {
        // H0 exchanges coordinates 0/1 with minus signs; H1 exchanges 1/2.
        // Their product is not symmetric, distinguishing Q from Q^T.
        let (q, r) = unpack(3, 2, &[2., 1., 0., 3., 4., 1.], &[1., 1.]);
        assert_eq!(q, [0., -1., 0., 0., 0., -1., 1., 0., 0.]);
        assert_eq!(r, [2., 0., 0., 3., 4., 0.]);
        assert_eq!(
            residuals(3, 2, &[0., -2., 0., 0., -3., -4.], &q, &r),
            (0., 0.)
        );
    }

    #[test]
    fn wide_upper_trapezoid_with_zero_reflectors() {
        let input = [2., 0., 3., 4., 5., 6.];
        let (q, r) = unpack(2, 3, &input, &[0., 0.]);
        assert_eq!(q, [1., 0., 0., 1.]);
        assert_eq!(r, input);
        assert_eq!(residuals(2, 3, &input, &q, &r), (0., 0.));
    }
}
