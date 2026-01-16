use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Load the module containing the kernel with constant memory
    let module = ctx.load_module(compile_ptx(include_str!("./constant_memory.cu"))?)?;

    // Get the constant memory symbol as a CudaSlice<u8>
    let mut coefficients_symbol = module.get_global("coefficients", &stream)?;
    println!(
        "Constant memory symbol 'coefficients' has {} bytes",
        coefficients_symbol.len()
    );

    // Set up polynomial coefficients: 1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3
    let coefficients = [1.0f32, 2.0, 3.0, 4.0];

    // Transmute the symbol to f32 and copy coefficients to constant memory
    let mut symbol_f32 = unsafe { coefficients_symbol.transmute_mut::<f32>(4).unwrap() };
    stream.memcpy_htod(&coefficients, &mut symbol_f32)?;

    // Load the kernel function
    let polynomial_kernel = module.load_function("polynomial_kernel")?;

    // Prepare input data
    let input = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let n = input.len();

    // Copy input to device
    let input_dev = stream.clone_htod(&input)?;
    let mut output_dev = stream.alloc_zeros::<f32>(n)?;

    // Launch kernel
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        stream
            .launch_builder(&polynomial_kernel)
            .arg(&mut output_dev)
            .arg(&input_dev)
            .arg(&(n as i32))
            .launch(cfg)
    }?;

    // Copy results back
    let output = stream.clone_dtoh(&output_dev)?;

    // Verify results
    println!("\nPolynomial evaluation (1.0 + 2.0*x + 3.0*x^2 + 4.0*x^3):");
    for (i, (&x, &y)) in input.iter().zip(output.iter()).enumerate() {
        let expected = coefficients[0]
            + coefficients[1] * x
            + coefficients[2] * x * x
            + coefficients[3] * x * x * x;
        println!("  f({:.1}) = {:.1} (expected {:.1})", x, y, expected);
        assert!(
            (y - expected).abs() < 1e-4,
            "Mismatch at index {}: got {}, expected {}",
            i,
            y,
            expected
        );
    }

    println!("\nAll results match expected values!");

    Ok(())
}
