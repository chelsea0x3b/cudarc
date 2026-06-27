#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
use std::sync::OnceLock;
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
#[cfg(feature = "dynamic-loading")]
fn load<F: Copy>(name: &str) -> F {
    unsafe { *culib().get::<F>(name.as_bytes()).unwrap_or_else(|e| panic!("Missing symbol {name}: {e}")) }
}
pub use self::curandDirectionVectorSet as curandDirectionVectorSet_t;
pub use self::curandMethod as curandMethod_t;
pub use self::curandOrdering as curandOrdering_t;
pub use self::curandRngType as curandRngType_t;
pub use self::curandStatus as curandStatus_t;
pub use self::libraryPropertyType_t as libraryPropertyType;
pub type cudaStream_t = *mut CUstream_st;
pub type curandDirectionVectors32_t = [::core::ffi::c_uint; 32usize];
pub type curandDirectionVectors64_t = [::core::ffi::c_ulonglong; 64usize];
pub type curandDiscreteDistribution_t = *mut curandDiscreteDistribution_st;
pub type curandDistributionM2Shift_t = *mut curandDistributionM2Shift_st;
pub type curandDistributionShift_t = *mut curandDistributionShift_st;
pub type curandDistribution_st = f64;
pub type curandDistribution_t = *mut curandDistribution_st;
pub type curandGenerator_t = *mut curandGenerator_st;
pub type curandHistogramM2K_st = ::core::ffi::c_uint;
pub type curandHistogramM2K_t = *mut curandHistogramM2K_st;
pub type curandHistogramM2V_st = curandDistribution_st;
pub type curandHistogramM2V_t = *mut curandHistogramM2V_st;
pub type curandHistogramM2_t = *mut curandHistogramM2_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandDirectionVectorSet {
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandMethod {
    CURAND_CHOOSE_BEST = 0,
    CURAND_ITR = 1,
    CURAND_KNUTH = 2,
    CURAND_HITR = 3,
    CURAND_M1 = 4,
    CURAND_M2 = 5,
    CURAND_BINARY_SEARCH = 6,
    CURAND_DISCRETE_GAUSS = 7,
    CURAND_REJECTION = 8,
    CURAND_DEVICE_API = 9,
    CURAND_FAST_REJECTION = 10,
    CURAND_3RD = 11,
    CURAND_DEFINITION = 12,
    CURAND_POISSON = 13,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_QUASI_DEFAULT = 201,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070", feature = "cuda-11080", feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020", feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000", feature = "cuda-13010", feature = "cuda-13020", feature = "cuda-13030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104,
    CURAND_ORDERING_QUASI_DEFAULT = 201,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100,
    CURAND_RNG_PSEUDO_XORWOW = 101,
    CURAND_RNG_PSEUDO_MRG32K3A = 121,
    CURAND_RNG_PSEUDO_MTGP32 = 141,
    CURAND_RNG_PSEUDO_MT19937 = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
    CURAND_RNG_QUASI_DEFAULT = 200,
    CURAND_RNG_QUASI_SOBOL32 = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64 = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDiscreteDistribution_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDistributionM2Shift_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandDistributionShift_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandGenerator_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandHistogramM2_st {
    _unused: [u8; 0],
}
pub unsafe fn curandCreateGenerator(generator: *mut curandGenerator_t, rng_type: curandRngType_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut curandGenerator_t, curandRngType_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandCreateGenerator") });
        _f(generator, rng_type)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandCreateGenerator(generator: *mut curandGenerator_t, rng_type: curandRngType_t) -> curandStatus_t;
        }
        curandCreateGenerator(generator, rng_type)
    }
}
pub unsafe fn curandCreateGeneratorHost(generator: *mut curandGenerator_t, rng_type: curandRngType_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut curandGenerator_t, curandRngType_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandCreateGeneratorHost") });
        _f(generator, rng_type)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandCreateGeneratorHost(generator: *mut curandGenerator_t, rng_type: curandRngType_t) -> curandStatus_t;
        }
        curandCreateGeneratorHost(generator, rng_type)
    }
}
pub unsafe fn curandCreatePoissonDistribution(lambda: f64, discrete_distribution: *mut curandDiscreteDistribution_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(f64, *mut curandDiscreteDistribution_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandCreatePoissonDistribution") });
        _f(lambda, discrete_distribution)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandCreatePoissonDistribution(lambda: f64, discrete_distribution: *mut curandDiscreteDistribution_t) -> curandStatus_t;
        }
        curandCreatePoissonDistribution(lambda, discrete_distribution)
    }
}
pub unsafe fn curandDestroyDistribution(discrete_distribution: curandDiscreteDistribution_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandDiscreteDistribution_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandDestroyDistribution") });
        _f(discrete_distribution)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandDestroyDistribution(discrete_distribution: curandDiscreteDistribution_t) -> curandStatus_t;
        }
        curandDestroyDistribution(discrete_distribution)
    }
}
pub unsafe fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandDestroyGenerator") });
        _f(generator)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;
        }
        curandDestroyGenerator(generator)
    }
}
pub unsafe fn curandGenerate(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_uint, num: usize) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut ::core::ffi::c_uint, usize) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerate") });
        _f(generator, outputPtr, num)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerate(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_uint, num: usize) -> curandStatus_t;
        }
        curandGenerate(generator, outputPtr, num)
    }
}
pub unsafe fn curandGenerateLogNormal(generator: curandGenerator_t, outputPtr: *mut f32, n: usize, mean: f32, stddev: f32) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut f32, usize, f32, f32) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateLogNormal") });
        _f(generator, outputPtr, n, mean, stddev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateLogNormal(generator: curandGenerator_t, outputPtr: *mut f32, n: usize, mean: f32, stddev: f32) -> curandStatus_t;
        }
        curandGenerateLogNormal(generator, outputPtr, n, mean, stddev)
    }
}
pub unsafe fn curandGenerateLogNormalDouble(generator: curandGenerator_t, outputPtr: *mut f64, n: usize, mean: f64, stddev: f64) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut f64, usize, f64, f64) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateLogNormalDouble") });
        _f(generator, outputPtr, n, mean, stddev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateLogNormalDouble(generator: curandGenerator_t, outputPtr: *mut f64, n: usize, mean: f64, stddev: f64) -> curandStatus_t;
        }
        curandGenerateLogNormalDouble(generator, outputPtr, n, mean, stddev)
    }
}
pub unsafe fn curandGenerateLongLong(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_ulonglong, num: usize) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut ::core::ffi::c_ulonglong, usize) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateLongLong") });
        _f(generator, outputPtr, num)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateLongLong(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_ulonglong, num: usize) -> curandStatus_t;
        }
        curandGenerateLongLong(generator, outputPtr, num)
    }
}
pub unsafe fn curandGenerateNormal(generator: curandGenerator_t, outputPtr: *mut f32, n: usize, mean: f32, stddev: f32) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut f32, usize, f32, f32) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateNormal") });
        _f(generator, outputPtr, n, mean, stddev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateNormal(generator: curandGenerator_t, outputPtr: *mut f32, n: usize, mean: f32, stddev: f32) -> curandStatus_t;
        }
        curandGenerateNormal(generator, outputPtr, n, mean, stddev)
    }
}
pub unsafe fn curandGenerateNormalDouble(generator: curandGenerator_t, outputPtr: *mut f64, n: usize, mean: f64, stddev: f64) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut f64, usize, f64, f64) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateNormalDouble") });
        _f(generator, outputPtr, n, mean, stddev)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateNormalDouble(generator: curandGenerator_t, outputPtr: *mut f64, n: usize, mean: f64, stddev: f64) -> curandStatus_t;
        }
        curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev)
    }
}
pub unsafe fn curandGeneratePoisson(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_uint, n: usize, lambda: f64) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut ::core::ffi::c_uint, usize, f64) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGeneratePoisson") });
        _f(generator, outputPtr, n, lambda)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGeneratePoisson(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_uint, n: usize, lambda: f64) -> curandStatus_t;
        }
        curandGeneratePoisson(generator, outputPtr, n, lambda)
    }
}
pub unsafe fn curandGeneratePoissonMethod(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_uint, n: usize, lambda: f64, method: curandMethod_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut ::core::ffi::c_uint, usize, f64, curandMethod_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGeneratePoissonMethod") });
        _f(generator, outputPtr, n, lambda, method)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGeneratePoissonMethod(generator: curandGenerator_t, outputPtr: *mut ::core::ffi::c_uint, n: usize, lambda: f64, method: curandMethod_t) -> curandStatus_t;
        }
        curandGeneratePoissonMethod(generator, outputPtr, n, lambda, method)
    }
}
pub unsafe fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateSeeds") });
        _f(generator)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateSeeds(generator: curandGenerator_t) -> curandStatus_t;
        }
        curandGenerateSeeds(generator)
    }
}
pub unsafe fn curandGenerateUniform(generator: curandGenerator_t, outputPtr: *mut f32, num: usize) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut f32, usize) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateUniform") });
        _f(generator, outputPtr, num)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateUniform(generator: curandGenerator_t, outputPtr: *mut f32, num: usize) -> curandStatus_t;
        }
        curandGenerateUniform(generator, outputPtr, num)
    }
}
pub unsafe fn curandGenerateUniformDouble(generator: curandGenerator_t, outputPtr: *mut f64, num: usize) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, *mut f64, usize) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGenerateUniformDouble") });
        _f(generator, outputPtr, num)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGenerateUniformDouble(generator: curandGenerator_t, outputPtr: *mut f64, num: usize) -> curandStatus_t;
        }
        curandGenerateUniformDouble(generator, outputPtr, num)
    }
}
pub unsafe fn curandGetDirectionVectors32(vectors: *mut *mut curandDirectionVectors32_t, set: curandDirectionVectorSet_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut curandDirectionVectors32_t, curandDirectionVectorSet_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGetDirectionVectors32") });
        _f(vectors, set)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGetDirectionVectors32(vectors: *mut *mut curandDirectionVectors32_t, set: curandDirectionVectorSet_t) -> curandStatus_t;
        }
        curandGetDirectionVectors32(vectors, set)
    }
}
pub unsafe fn curandGetDirectionVectors64(vectors: *mut *mut curandDirectionVectors64_t, set: curandDirectionVectorSet_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut curandDirectionVectors64_t, curandDirectionVectorSet_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGetDirectionVectors64") });
        _f(vectors, set)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGetDirectionVectors64(vectors: *mut *mut curandDirectionVectors64_t, set: curandDirectionVectorSet_t) -> curandStatus_t;
        }
        curandGetDirectionVectors64(vectors, set)
    }
}
pub unsafe fn curandGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(libraryPropertyType, *mut ::core::ffi::c_int) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGetProperty") });
        _f(type_, value)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGetProperty(type_: libraryPropertyType, value: *mut ::core::ffi::c_int) -> curandStatus_t;
        }
        curandGetProperty(type_, value)
    }
}
pub unsafe fn curandGetScrambleConstants32(constants: *mut *mut ::core::ffi::c_uint) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_uint) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGetScrambleConstants32") });
        _f(constants)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGetScrambleConstants32(constants: *mut *mut ::core::ffi::c_uint) -> curandStatus_t;
        }
        curandGetScrambleConstants32(constants)
    }
}
pub unsafe fn curandGetScrambleConstants64(constants: *mut *mut ::core::ffi::c_ulonglong) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut *mut ::core::ffi::c_ulonglong) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGetScrambleConstants64") });
        _f(constants)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGetScrambleConstants64(constants: *mut *mut ::core::ffi::c_ulonglong) -> curandStatus_t;
        }
        curandGetScrambleConstants64(constants)
    }
}
pub unsafe fn curandGetVersion(version: *mut ::core::ffi::c_int) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(*mut ::core::ffi::c_int) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandGetVersion") });
        _f(version)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandGetVersion(version: *mut ::core::ffi::c_int) -> curandStatus_t;
        }
        curandGetVersion(version)
    }
}
pub unsafe fn curandSetGeneratorOffset(generator: curandGenerator_t, offset: ::core::ffi::c_ulonglong) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, ::core::ffi::c_ulonglong) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandSetGeneratorOffset") });
        _f(generator, offset)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandSetGeneratorOffset(generator: curandGenerator_t, offset: ::core::ffi::c_ulonglong) -> curandStatus_t;
        }
        curandSetGeneratorOffset(generator, offset)
    }
}
pub unsafe fn curandSetGeneratorOrdering(generator: curandGenerator_t, order: curandOrdering_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, curandOrdering_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandSetGeneratorOrdering") });
        _f(generator, order)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandSetGeneratorOrdering(generator: curandGenerator_t, order: curandOrdering_t) -> curandStatus_t;
        }
        curandSetGeneratorOrdering(generator, order)
    }
}
pub unsafe fn curandSetPseudoRandomGeneratorSeed(generator: curandGenerator_t, seed: ::core::ffi::c_ulonglong) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, ::core::ffi::c_ulonglong) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandSetPseudoRandomGeneratorSeed") });
        _f(generator, seed)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandSetPseudoRandomGeneratorSeed(generator: curandGenerator_t, seed: ::core::ffi::c_ulonglong) -> curandStatus_t;
        }
        curandSetPseudoRandomGeneratorSeed(generator, seed)
    }
}
pub unsafe fn curandSetQuasiRandomGeneratorDimensions(generator: curandGenerator_t, num_dimensions: ::core::ffi::c_uint) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, ::core::ffi::c_uint) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandSetQuasiRandomGeneratorDimensions") });
        _f(generator, num_dimensions)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandSetQuasiRandomGeneratorDimensions(generator: curandGenerator_t, num_dimensions: ::core::ffi::c_uint) -> curandStatus_t;
        }
        curandSetQuasiRandomGeneratorDimensions(generator, num_dimensions)
    }
}
pub unsafe fn curandSetStream(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t {
    #[cfg(feature = "dynamic-loading")]
    {
        type _F = unsafe extern "C" fn(curandGenerator_t, cudaStream_t) -> curandStatus_t;
        static _S: OnceLock<_F> = OnceLock::new();
        let _f = _S.get_or_init(|| unsafe { load::<_F>("curandSetStream") });
        _f(generator, stream)
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        extern "C" {
            fn curandSetStream(generator: curandGenerator_t, stream: cudaStream_t) -> curandStatus_t;
        }
        curandSetStream(generator, stream)
    }
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn is_culib_present() -> bool {
    let lib_names = ["curand"];
    let choices = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten();
    for choice in choices {
        if ::libloading::Library::new(choice).is_ok() {
            return true;
        }
    }
    false
}
#[cfg(feature = "dynamic-loading")]
pub unsafe fn culib() -> &'static ::libloading::Library {
    static LIB: OnceLock<::libloading::Library> = OnceLock::new();
    LIB.get_or_init(|| {
        let lib_names = std::vec!["curand"];
        let choices: std::vec::Vec<_> = lib_names.iter().map(|l| crate::get_lib_name_candidates(l)).flatten().collect();
        for choice in choices.iter() {
            if let Ok(lib) = ::libloading::Library::new(choice) {
                return lib;
            }
        }
        crate::panic_no_lib_found(lib_names[0], &choices);
    })
}
