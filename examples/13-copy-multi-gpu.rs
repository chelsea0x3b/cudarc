use cudarc::driver::{CudaContext, CudaSlice, DriverError};

fn main() -> Result<(), DriverError> {
    let size = 10;

    let ctx1 = CudaContext::new(0)?;
    let stream1 = ctx1.default_stream();
    let a: CudaSlice<f64> = stream1.alloc_zeros::<f64>(size)?;

    let ctx2 = CudaContext::new(1)?;
    let stream2 = ctx2.default_stream();

    let b = a.clone_peer(&stream2)?;

    stream2.clone_dtoh(&b)?;
    stream1.clone_dtoh(&a)?;

    Ok(())
}
