//! NVFP4 scaled matmul example using cuBLASLt.
//!
//! Requires: Blackwell GPU (compute capability 10.x) + CUDA 12.8+.
//!
//! Run with:
//! ```sh
//! cargo run --example nvfp4-matmul --features f4,f8,f16,cublaslt,cuda-12080
//! ```

#[cfg(all(
    any(
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010",
    ),
    feature = "f4",
    feature = "f8",
    feature = "f16",
))]
fn main() {
    use cudarc::cublaslt::{CudaBlasLT, ScaledMatmul, ScaledMatmulConfig};
    use cudarc::driver::CudaContext;

    // NVFP4 block-scaled matmul: D (bf16) = alpha * (A_fp4 * B_fp4) + beta * C (bf16)
    // Block scales are F8E4M3, one per 16-element block along K.

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let blas = CudaBlasLT::new(stream.clone()).unwrap();

    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 128;

    // Packed FP4 inputs: K/2 bytes per row (two values per byte)
    let a_packed = vec![float4::F4E2M1x2::from_bits(0x11); M * K / 2];
    let b_packed = vec![float4::F4E2M1x2::from_bits(0x11); K * N / 2];

    // Block scales: one F8E4M3 per 16-element block along K
    let num_a_scales = M * (K / 16);
    let num_b_scales = N * (K / 16);
    let a_scale_data = vec![float8::F8E4M3::from(1.0_f32); num_a_scales];
    let b_scale_data = vec![float8::F8E4M3::from(1.0_f32); num_b_scales];

    // D scale (scalar f32 on device) and D output scale buffer
    let d_scale_data = vec![1.0_f32];
    let d_out_scale_data = vec![float8::F8E4M3::from(1.0_f32); M * (N / 16)];

    // Accumulator input C (bf16, zeros for pure A*B)
    let c_data = vec![half::bf16::from_f32(0.0); M * N];

    // Upload to device
    let a_dev = stream.clone_htod(&a_packed).unwrap();
    let b_dev = stream.clone_htod(&b_packed).unwrap();
    let c_dev = stream.clone_htod(&c_data).unwrap();
    let mut d_dev = stream.alloc_zeros::<half::bf16>(M * N).unwrap();
    let a_scale_dev = stream.clone_htod(&a_scale_data).unwrap();
    let b_scale_dev = stream.clone_htod(&b_scale_data).unwrap();
    let d_scale_dev = stream.clone_htod(&d_scale_data).unwrap();
    let mut d_out_scale_dev = stream.clone_htod(&d_out_scale_data).unwrap();

    // cuBLASLt uses column-major layout. For row-major A(M,K) * B(K,N) = D(M,N),
    // compute D_col(N,M) = B_col(N,K) * A_col(K,M) — swap m/n and A/B.
    let cfg = ScaledMatmulConfig {
        transa: false,
        transb: false,
        m: N as u64,
        n: M as u64,
        k: K as u64,
        alpha: 1.0,
        beta: 0.0,
        lda: N as i64,
        ldb: K as i64,
        ldc: N as i64,
        ldd: N as i64,
    };

    unsafe {
        blas.scaled_matmul(
            cfg,
            &b_dev, // B is "A" in column-major GEMM
            &a_dev, // A is "B" in column-major GEMM
            &c_dev,
            &mut d_dev,
            &b_scale_dev, // scales follow their matrix
            &a_scale_dev,
            &d_scale_dev,
            &mut d_out_scale_dev,
        )
        .unwrap();
    }

    let result = stream.clone_dtoh(&d_dev).unwrap();
    println!("NVFP4 scaled matmul complete. First 8 output values (bf16):");
    for val in result.iter().take(8) {
        print!("{:.4} ", val.to_f32());
    }
    println!();
}

#[cfg(not(all(
    any(
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000",
        feature = "cuda-13010",
    ),
    feature = "f4",
    feature = "f8",
    feature = "f16",
)))]
fn main() {
    println!("This example requires features: f4, f8, f16, cublaslt, and cuda-12080+");
}
