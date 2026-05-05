use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use bindgen::Builder;

mod download;
mod extract;
mod merge;

/// Cuda is split in various modules in cudarc.
/// Those configs decide how to download and
/// export bindings with bindgen. See [`ModuleConfig`].
fn create_modules() -> Vec<ModuleConfig> {
    vec![
        ModuleConfig {
            cudarc_name: "runtime",
            redist_name: "cuda_cudart",
            allowlist: Filters {
                types: vec!["^[Cc][Uu][Dd][Aa].*"],
                functions: vec!["^[Cc][Uu][Dd][Aa].*"],
                vars: vec!["^[Cc][Uu][Dd][Aa].*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                // NOTE: See https://github.com/chelsea0x3b/cudarc/issues/397
                types: vec![],
                functions: vec!["cudaDeviceGetNvSciSyncAttributes"],
                vars: vec![],
            },
            libs: vec!["cudart"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "driver",
            redist_name: "cuda_cudart",
            allowlist: Filters {
                types: vec![
                    "^CU.*",
                    "^cuuint(32|64)_t",
                    "^cudaError_enum",
                    "^cu.*Complex$",
                    "^cuda.*",
                    "^libraryPropertyType.*",
                ],
                functions: vec!["^cu.*"],
                vars: vec!["^CU.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                // NOTE: See https://github.com/chelsea0x3b/cudarc/issues/385
                types: vec!["^cuCheckpoint.*"],
                functions: vec!["^cuCheckpoint.*", "cuDeviceGetNvSciSyncAttributes"],
                vars: vec![],
            },
            libs: vec!["cuda", "nvcuda"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cublas",
            redist_name: "libcublas",
            allowlist: Filters {
                types: vec!["^cublas.*"],
                functions: vec!["^cublas.*"],
                vars: vec!["^cublas.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec![],
                functions: vec![
                    // NOTE: see https://github.com/chelsea0x3b/cudarc/issues/489
                    "cublasGetEmulationSpecialValuesSupport",
                    "cublasGetFixedPointEmulationMantissaBitCountPointer",
                    "cublasGetFixedPointEmulationMantissaBitOffset",
                    "cublasGetFixedPointEmulationMantissaControl",
                    "cublasGetFixedPointEmulationMaxMantissaBitCount",
                    "cublasSetEmulationSpecialValuesSupport",
                    "cublasSetFixedPointEmulationMantissaBitCountPointer",
                    "cublasSetFixedPointEmulationMantissaBitOffset",
                    "cublasSetFixedPointEmulationMantissaControl",
                    "cublasSetFixedPointEmulationMaxMantissaBitCount",
                ],
                vars: vec![],
            },
            libs: vec!["cublas"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cublaslt",
            redist_name: "libcublas",
            allowlist: Filters {
                types: vec!["^cublasLt.*"],
                functions: vec!["^cublasLt.*"],
                vars: vec!["^cublasLt.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec![],
                functions: vec!["cublasLtDisableCpuInstructionsSetMask"],
                vars: vec![],
            },
            libs: vec!["cublasLt"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "curand",
            redist_name: "libcurand",
            allowlist: Filters {
                types: vec!["^curand.*"],
                functions: vec!["^curand.*"],
                vars: vec!["^curand.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec![],
                functions: vec!["curandGenerateBinomial", "curandGenerateBinomialMethod"],
                vars: vec![],
            },
            libs: vec!["curand"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "nvrtc",
            redist_name: "cuda_nvrtc",
            allowlist: Filters {
                types: vec!["^nvrtc.*"],
                functions: vec!["^nvrtc.*"],
                vars: vec!["^nvrtc.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec![],
                functions: vec![
                    // NOTE: see https://github.com/chelsea0x3b/cudarc/pull/431
                    "nvrtcGetPCHCreateStatus",
                    "nvrtcGetPCHHeapSize",
                    "nvrtcGetPCHHeapSizeRequired",
                    "nvrtcSetFlowCallback",
                    "nvrtcSetPCHHeapSize",
                    // NOTE: see https://github.com/chelsea0x3b/cudarc/issues/490
                    "nvrtcGetNVVM",
                    "nvrtcGetNVVMSize",
                ],
                vars: vec![],
            },
            libs: vec!["nvrtc"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cudnn",
            redist_name: "cudnn",
            allowlist: Filters {
                types: vec!["^cudnn.*"],
                functions: vec!["^cudnn.*"],
                vars: vec!["^cudnn.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters::none(),
            libs: vec!["cudnn"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cudnn",
            lib_versions: vec![(8, 9, 7), (9, 10, 2), (9, 21, 1)],
        },
        ModuleConfig {
            cudarc_name: "nccl",
            redist_name: "libnccl",
            allowlist: Filters {
                types: vec!["^nccl.*"],
                functions: vec!["^nccl.*"],
                vars: vec!["^nccl.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters::none(),
            libs: vec!["nccl"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "nccl",
            lib_versions: vec![
                (2, 18, 5),
                (2, 19, 3),
                (2, 20, 5),
                (2, 21, 5),
                (2, 22, 3),
                (2, 24, 3),
                (2, 25, 1),
                (2, 26, 5),
                (2, 27, 6),
                (2, 28, 9),
                (2, 29, 7),
                (2, 30, 4),
            ],
        },
        ModuleConfig {
            cudarc_name: "cusparse",
            redist_name: "libcusparse",
            allowlist: Filters {
                types: vec!["^cusparse.*"],
                functions: vec!["^cusparse.*"],
                vars: vec!["^cusparse.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec![],
                functions: vec![
                    "cusparseCbsric02_bufferSizeExt",
                    "cusparseCbsrilu02_bufferSizeExt",
                    "cusparseCbsrsm2_bufferSizeExt",
                    "cusparseCbsrsv2_bufferSizeExt",
                    "cusparseCcsr2gebsr_bufferSizeExt",
                    "cusparseCcsric02_bufferSizeExt",
                    "cusparseCcsrilu02_bufferSizeExt",
                    "cusparseCgebsr2gebsc_bufferSizeExt",
                    "cusparseCgebsr2gebsr_bufferSizeExt",
                    "cusparseDbsric02_bufferSizeExt",
                    "cusparseDbsrilu02_bufferSizeExt",
                    "cusparseDbsrsm2_bufferSizeExt",
                    "cusparseDbsrsv2_bufferSizeExt",
                    "cusparseDcsr2gebsr_bufferSizeExt",
                    "cusparseDcsric02_bufferSizeExt",
                    "cusparseDcsrilu02_bufferSizeExt",
                    "cusparseDgebsr2gebsc_bufferSizeExt",
                    "cusparseDgebsr2gebsr_bufferSizeExt",
                    "cusparseSbsric02_bufferSizeExt",
                    "cusparseSbsrilu02_bufferSizeExt",
                    "cusparseSbsrsm2_bufferSizeExt",
                    "cusparseSbsrsv2_bufferSizeExt",
                    "cusparseScsr2gebsr_bufferSizeExt",
                    "cusparseScsric02_bufferSizeExt",
                    "cusparseScsrilu02_bufferSizeExt",
                    "cusparseSgebsr2gebsc_bufferSizeExt",
                    "cusparseSgebsr2gebsr_bufferSizeExt",
                    "cusparseXgebsr2csr",
                    "cusparseZbsric02_bufferSizeExt",
                    "cusparseZbsrilu02_bufferSizeExt",
                    "cusparseZbsrsm2_bufferSizeExt",
                    "cusparseZbsrsv2_bufferSizeExt",
                    "cusparseZcsr2gebsr_bufferSizeExt",
                    "cusparseZcsric02_bufferSizeExt",
                    "cusparseZcsrilu02_bufferSizeExt",
                    "cusparseZgebsr2gebsc_bufferSizeExt",
                    "cusparseZgebsr2gebsr_bufferSizeExt",
                ],
                vars: vec![],
            },
            libs: vec!["cusparse"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cusolver",
            redist_name: "libcusolver",
            allowlist: Filters {
                types: vec!["^cusolver.*"],
                functions: vec!["^cusolver.*"],
                vars: vec!["^cusolver.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec!["^cusolverMg.*"],
                functions: vec!["^cusolverMg.*", "^cusolverDnLogger.*"],
                vars: vec!["^cusolverMg.*"],
            },
            libs: vec!["cusolver"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            // cusolverDn.h transitively includes cublas_v2.h
            module_dependencies: vec!["cublas", "cusparse"],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cusolvermg",
            redist_name: "libcusolver",
            allowlist: Filters {
                types: vec!["^cusolverMg.*"],
                functions: vec!["^cusolverMg.*"],
                vars: vec!["^cusolverMg.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters::none(),
            libs: vec!["cusolverMg"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            // cusolverMg.h transitively includes cublas_v2.h
            module_dependencies: vec!["cublas", "cusparse"],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cufile",
            redist_name: "libcufile",
            allowlist: Filters {
                types: vec!["^[Cc][Uu][Ff][Ii][Ll][Ee].*"],
                functions: vec!["^cuFile.*"],
                vars: vec![],
            },
            allowlist_recursively: true,
            blocklist: Filters::none(),
            libs: vec!["cufile"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "nvtx",
            redist_name: "cuda_nvtx",
            allowlist: Filters {
                types: vec!["^nvtx.*"],
                functions: vec!["^nvtx.*"],
                vars: vec!["^nvtx.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters {
                types: vec![],
                functions: vec!["nvtxInitialize"],
                vars: vec![],
            },
            libs: vec!["nvToolsExt"],
            clang_args: vec!["-DNVTX_NO_IMPL=0", "-DNVTX_DECLSPEC="],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cupti",
            redist_name: "cuda_cupti",
            allowlist: Filters {
                types: vec![
                    // CUPTI types:
                    "^[Cc][Uu][Pp][Tt][Ii].*",
                    // Types from the generated_cuda(_meta / runtime_api_meta).h
                    // headers. These help dissect data representing function arguments
                    // of CUDA functions in the CUPTI Callback API.
                    "^[Cc][Uu][Dd][Aa].*_params.*",
                    "^[Cc][Uu].*_params.*",
                    // Types that are obsolete but still used in CUPTI.
                    "CUDA_ARRAY_DESCRIPTOR_v1_st",
                    "CUDA_ARRAY_DESCRIPTOR_v1",
                    "CUDA_ARRAY3D_DESCRIPTOR_v1_st",
                    "CUDA_ARRAY3D_DESCRIPTOR_v1",
                    "CUDA_MEMCPY2D_v1_st",
                    "CUDA_MEMCPY2D_v1",
                    "CUDA_MEMCPY3D_v1_st",
                    "CUDA_MEMCPY3D_v1",
                    "CUdeviceptr_v1",
                ],
                functions: vec!["^cupti.*"],
                vars: vec!["^[Cc][Uu][Pp][Tt][Ii].*"],
            },
            allowlist_recursively: false,
            blocklist: Filters {
                types: vec![
                    // For cuda-11040, the meta headers seem to include some osbolete
                    // types for which the definitions are missing because they are not
                    // included through any cupti headers, but only exist in a CUDA
                    // source, block these:
                    "cudaSignalExternalSemaphoresAsync_ptsz_v10000_params_st",
                    "cudaSignalExternalSemaphoresAsync_ptsz_v10000_params",
                    "cudaSignalExternalSemaphoresAsync_v10000_params_st",
                    "cudaSignalExternalSemaphoresAsync_v10000_params",
                    "cudaWaitExternalSemaphoresAsync_ptsz_v10000_params_st",
                    "cudaWaitExternalSemaphoresAsync_ptsz_v10000_params",
                    "cudaWaitExternalSemaphoresAsync_v10000_params_st",
                    "cudaWaitExternalSemaphoresAsync_v10000_params",
                ],
                functions: vec![],
                vars: vec![],
            },
            libs: vec!["cupti"],
            clang_args: vec![],
            raw_lines: vec!["use crate::driver::sys::*;", "use crate::runtime::sys::*;"],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
        ModuleConfig {
            cudarc_name: "cutensor",
            redist_name: "libcutensor",
            allowlist: Filters {
                types: vec!["^cutensor.*"],
                functions: vec!["^cutensor.*"],
                vars: vec!["^cutensor.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters::none(),
            libs: vec!["cutensor"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: None,
            module_dependencies: vec![],
            feature_prefix: "cutensor",
            lib_versions: vec![(2, 3, 1), (2, 4, 1), (2, 5, 0), (2, 6, 0)],
        },
        ModuleConfig {
            cudarc_name: "cufft",
            redist_name: "libcufft",
            allowlist: Filters {
                types: vec!["^cufft.*"],
                functions: vec!["^cufft.*"],
                vars: vec!["^cufft.*"],
            },
            allowlist_recursively: true,
            blocklist: Filters::none(),
            libs: vec!["cufft"],
            clang_args: vec![],
            raw_lines: vec![],
            min_cuda_version: Some("cuda-12000"),
            module_dependencies: vec![],
            feature_prefix: "cuda",
            lib_versions: vec![],
        },
    ]
}

#[derive(Debug)]
struct ModuleConfig {
    /// Name of corresponding module in cudarc
    cudarc_name: &'static str,
    /// The name of the library within cuda/redist
    redist_name: &'static str,
    /// The various filter used in bindgen to select the symbols we re-expose
    allowlist: Filters,
    blocklist: Filters,
    /// The various names used to look for symbols
    /// Those names are only used with the `dynamic-loading`
    /// feature.
    libs: Vec<&'static str>,
    /// Arguments passed directly to clang.
    clang_args: Vec<&'static str>,
    /// Whether to recursively add types from allowlist items. This can be set to false
    /// in order to prevent duplicate definitions for headers that include other headers
    /// for which bindings are also generated.
    allowlist_recursively: bool,
    /// Lines of code to add at the beginning of the generated bindings.
    raw_lines: Vec<&'static str>,
    /// Minimum CUDA version required for this module. If None, all versions are supported.
    min_cuda_version: Option<&'static str>,
    /// cudarc module names whose archive include dirs must be on the clang include path.
    /// Modules with dependencies are processed in a second wave, after all independent
    /// modules have been downloaded, extracted, and had bindings generated.
    module_dependencies: Vec<&'static str>,
    /// Cargo feature prefix for this module (e.g. "cuda", "nccl", "cudnn", "cutensor").
    feature_prefix: &'static str,
    /// Version triples for library-versioned modules. Empty means CUDA-version axis.
    lib_versions: Vec<(u32, u32, u32)>,
}

impl ModuleConfig {
    /// Returns true if this module supports the given CUDA version.
    fn supports_cuda_version(&self, cuda_version: &str) -> bool {
        match self.min_cuda_version {
            None => true,
            Some(min_version) => cuda_version >= min_version,
        }
    }
}

impl ModuleConfig {
    fn run_bindgen(
        &self,
        version_key: &str,
        archive_directory: &Path,
        primary_archives: &[PathBuf],
    ) -> Result<()> {
        let sysdir = Path::new(".")
            .join("out")
            .join(&self.cudarc_name)
            .join("sys");
        fs::create_dir_all(&sysdir)
            .context(format!("Failed to create directory {}", sysdir.display()))?;

        let linked_dir = sysdir.join("linked");
        fs::create_dir_all(&linked_dir).context(format!(
            "Failed to create directory {}",
            linked_dir.display()
        ))?;

        // cuda_version keys arrive as "cuda-12080"; library version keys arrive as "02276" etc.
        let key = version_key.strip_prefix("cuda-").unwrap_or(version_key);
        let outfilename = linked_dir.join(format!("sys_{key}.rs"));

        // Generate linked bindings using bindgen library
        let mut builder = Builder::default()
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_default(false)
            .derive_eq(true)
            .derive_hash(true)
            .derive_ord(true)
            .generate_comments(false)
            .layout_tests(false)
            .use_core();

        for &arg in self.clang_args.iter() {
            builder = builder.clang_arg(arg);
        }

        for filter_name in self.allowlist.types.iter() {
            builder = builder.allowlist_type(filter_name);
        }
        for filter_name in self.allowlist.vars.iter() {
            builder = builder.allowlist_var(filter_name);
        }
        for filter_name in self.allowlist.functions.iter() {
            builder = builder.allowlist_function(filter_name);
        }
        builder = builder.allowlist_recursively(self.allowlist_recursively);

        for filter_name in self.blocklist.types.iter() {
            builder = builder.blocklist_type(filter_name);
        }
        for filter_name in self.blocklist.vars.iter() {
            builder = builder.blocklist_var(filter_name);
        }
        for filter_name in self.blocklist.functions.iter() {
            builder = builder.blocklist_function(filter_name);
        }

        for &raw_line in self.raw_lines.iter() {
            builder = builder.raw_line(raw_line);
        }

        let parent_sysdir = Path::new("..")
            .join("src")
            .join(&self.cudarc_name)
            .join("sys");
        let wrapper_h = parent_sysdir.join("wrapper.h");
        let cuda_directory = archive_directory.join("include");
        let primary_includes: Vec<_> = primary_archives
            .into_iter()
            .map(|c| c.join("include"))
            .collect();
        log::debug!("Include directories {}", cuda_directory.display());
        log::debug!(
            "Include primary directories {:?}",
            primary_includes
                .iter()
                .map(|p| p.display())
                .collect::<Vec<_>>()
        );
        builder = builder
            .header(wrapper_h.to_string_lossy())
            .clang_arg(format!("-I{}", cuda_directory.display()))
            // For cuda profiler which has a very simple consistent API
            .clang_arg(format!(
                "-I{}",
                std::env::current_dir()
                    .expect("Current directory")
                    .join("include")
                    .display()
            ));
        for include in primary_includes {
            builder = builder.clang_arg(format!("-I{}", include.display()));
        }

        let bindings = builder.generate().context(format!(
            "Failed to generate bindings for {}",
            wrapper_h.display()
        ))?;

        bindings.write_to_file(&outfilename).context(format!(
            "Failed to write bindings to {}",
            outfilename.display()
        ))?;
        log::debug!("Wrote linked bindings to {}", outfilename.display());

        Ok(())
    }
}

#[derive(Debug)]
/// Bindgen filters
struct Filters {
    types: Vec<&'static str>,
    functions: Vec<&'static str>,
    vars: Vec<&'static str>,
}

impl Filters {
    fn none() -> Self {
        Self {
            types: vec![],
            functions: vec![],
            vars: vec![],
        }
    }
}

/// Downloads, unpacks and generate bindings for all modules.
fn create_bindings(modules: &[ModuleConfig], cuda_versions: &[&str]) -> Result<()> {
    let downloads_dir = std::env::temp_dir().join("cudarc").join("bindings");
    fs::create_dir_all(&downloads_dir).context("Failed to create downloads directory")?;

    let multi_progress = MultiProgress::new();

    // Phase A: download primary archives for all versions in parallel.
    // These are done upfront so module tasks don't race on the shared primary archive paths.
    let primary_pb = multi_progress.add(ProgressBar::new(cuda_versions.len() as u64));
    primary_pb.set_style(
        ProgressStyle::default_bar().template("primary archives {wide_bar} {pos}/{len}")?,
    );
    let primary_archives_map: HashMap<&str, Vec<PathBuf>> = cuda_versions
        .par_iter()
        .map(|&cuda_version| {
            // cuda_cudart provides cuda.h / cuda_runtime.h, which virtually every module
            // transitively includes. It must be a primary archive so all parallel module
            // tasks have those headers on their include path.
            let names = if cuda_version.starts_with("cuda-13") {
                vec!["cuda_nvcc", "cuda_cccl", "cuda_crt", "cuda_cudart"]
            } else if cuda_version.starts_with("cuda-12") {
                vec!["cuda_nvcc", "cuda_cccl", "cuda_cudart"]
            } else {
                vec!["cuda_nvcc", "cuda_cudart"]
            };
            let mut archives = vec![];
            for name in names {
                let archive = get_archive(
                    cuda_version,
                    name,
                    "primary",
                    &downloads_dir,
                    &multi_progress,
                )?;
                archives.push(archive);
            }
            primary_pb.inc(1);
            Ok((cuda_version, archives))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    primary_pb.finish_with_message("primary archives done");

    // Phase B: CUDA-versioned modules, processed in dependency order.
    let tasks: Vec<(&str, &ModuleConfig)> = cuda_versions
        .iter()
        .flat_map(|&v| modules.iter().map(move |m| (v, m)))
        .filter(|(v, m)| m.lib_versions.is_empty() && m.supports_cuda_version(v))
        .collect();

    let mut archive_dir_map: HashMap<(&str, &str), PathBuf> = HashMap::new();
    let mut remaining: Vec<(&str, &ModuleConfig)> = tasks;
    let mut wave = 0usize;

    while !remaining.is_empty() {
        let (ready, not_ready): (Vec<_>, Vec<_>) = remaining.into_iter().partition(|(v, m)| {
            m.module_dependencies
                .iter()
                .all(|dep| archive_dir_map.contains_key(&(*v, *dep)))
        });
        anyhow::ensure!(
            !ready.is_empty(),
            "dependency cycle detected: modules {:?} cannot be resolved",
            not_ready
                .iter()
                .map(|(_, m)| m.cudarc_name)
                .collect::<Vec<_>>()
        );

        let pb = multi_progress.add(ProgressBar::new(ready.len() as u64));
        pb.set_style(
            ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len} ({eta})")?,
        );
        let results = ready
            .par_iter()
            .map(|(cuda_version, module)| {
                let mut includes = primary_archives_map[*cuda_version].clone();
                for dep_name in &module.module_dependencies {
                    if let Some(dep_dir) = archive_dir_map.get(&(*cuda_version, *dep_name)) {
                        includes.push(dep_dir.clone());
                    }
                }
                let archive_dir = generate_sys(
                    cuda_version,
                    module,
                    &includes,
                    &downloads_dir,
                    &multi_progress,
                )
                .context(format!(
                    "Failed to generate {} for {cuda_version}",
                    module.cudarc_name
                ))?;
                pb.inc(1);
                Ok(((*cuda_version, module.cudarc_name), archive_dir))
            })
            .collect::<Result<Vec<_>>>()?;
        pb.finish_with_message(format!("wave {wave} done - {ready:?}"));
        archive_dir_map.extend(results);
        remaining = not_ready;
        wave += 1;
    }

    // Phase C: library-versioned modules (NCCL, cuDNN, cuTENSOR).
    let lib_tasks: Vec<(&ModuleConfig, (u32, u32, u32))> = modules
        .iter()
        .filter(|m| !m.lib_versions.is_empty())
        .flat_map(|m| m.lib_versions.iter().map(move |&v| (m, v)))
        .collect();

    let pb = multi_progress.add(ProgressBar::new(lib_tasks.len() as u64));
    pb.set_style(ProgressStyle::default_bar().template("{msg} {wide_bar} {pos}/{len} ({eta})")?);

    lib_tasks
        .into_par_iter()
        .map(|(module, lib_version)| {
            let result = if module.cudarc_name == "nccl" {
                get_nccl_archive(
                    lib_version,
                    module,
                    &primary_archives_map,
                    &downloads_dir,
                    &multi_progress,
                )
            } else {
                get_redist_lib_archive(
                    lib_version,
                    module,
                    &primary_archives_map,
                    &downloads_dir,
                    &multi_progress,
                )
            };
            pb.inc(1);
            result.context(format!(
                "Failed to generate {} {lib_version:?}",
                module.cudarc_name
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    pb.finish_with_message("lib-versioned modules done");

    Ok(())
}

fn get_version(cuda_version: &str) -> Result<(u32, u32, u32)> {
    let number = cuda_version
        .split('-')
        .last()
        .context(format!("Invalid CUDA version format: {}", cuda_version))?;

    let major = number[..2].parse().context(format!(
        "Failed to parse major version from {}",
        cuda_version
    ))?;
    let minor = number[2..4].parse().context(format!(
        "Failed to parse minor version from {}",
        cuda_version
    ))?;
    let patch = number[4..].parse().context(format!(
        "Failed to parse patch version from {}",
        cuda_version
    ))?;

    Ok((major, minor, patch))
}

fn get_archive(
    cuda_version: &str,
    cuda_name: &str,
    module_name: &str,
    downloads_dir: &Path,
    multi_progress: &MultiProgress,
) -> Result<PathBuf> {
    let (major, minor, patch) = get_version(cuda_version)?;
    let url = "https://developer.download.nvidia.com/compute/cuda/redist/";
    let data = download::cuda_redist(major, minor, patch, url, downloads_dir, multi_progress)?;

    let lib = &data[cuda_name]["linux-x86_64"];
    let path = lib["relative_path"].as_str().context(format!(
        "Missing relative_path in redistrib data for {}",
        cuda_name
    ))?;
    let checksum = lib["sha256"].as_str().context(format!(
        "Missing sha256 in redistrib data for {}",
        cuda_name
    ))?;

    let output_dir = downloads_dir.join(module_name);
    let parts: Vec<_> = Path::new(path)
        .file_name()
        .context(format!("Failed to get file name from {}", path))?
        .to_str()
        .expect("A valid filename")
        .split(".")
        .collect();
    let n = parts.len();
    let name = parts.into_iter().take(n - 2).collect::<Vec<_>>().join(".");
    let archive_dir = output_dir.join(name);
    log::debug!("Archive dir {archive_dir:?}");

    if !archive_dir.exists() {
        fs::create_dir_all(&output_dir).context(format!(
            "Failed to create directory {}",
            output_dir.display()
        ))?;
        let out_path = output_dir.join(
            Path::new(path)
                .file_name()
                .context(format!("Failed to get file name from {}", path))?,
        );
        log::debug!("Getting with checksum {url}/{path}");
        download::to_file_with_checksum(
            &format!("{}/{}", url, path),
            &out_path,
            checksum,
            multi_progress,
        )?;
        log::debug!("Got with checksum {url}/{path}");

        log::debug!("Extracting {}", out_path.display());
        extract::extract_archive(&out_path, &output_dir, multi_progress)?;
        log::debug!("Extracted {}", out_path.display());
    }
    Ok(archive_dir)
}

fn generate_sys(
    cuda_version: &str,
    module: &ModuleConfig,
    primary_archives: &[PathBuf],
    downloads_dir: &Path,
    multi_progress: &MultiProgress,
) -> Result<PathBuf> {
    let archive_dir = get_archive(
        cuda_version,
        &module.redist_name,
        &module.cudarc_name,
        downloads_dir,
        multi_progress,
    )?;
    module.run_bindgen(cuda_version, &archive_dir, primary_archives)?;
    Ok(archive_dir)
}

fn get_nccl_archive(
    (major, minor, patch): (u32, u32, u32),
    module: &ModuleConfig,
    primary_archives_map: &HashMap<&str, Vec<PathBuf>>,
    downloads_dir: &Path,
    multi_progress: &MultiProgress,
) -> Result<()> {
    let base_url = "https://developer.download.nvidia.com/compute/redist/nccl";
    let full_version = format!("{major}.{minor}.{patch}");

    let output_dir = downloads_dir.join(module.cudarc_name);
    fs::create_dir_all(&output_dir).context(format!(
        "Failed to create directory {}",
        output_dir.display()
    ))?;

    let cached_prefix = format!("nccl_{full_version}-1+cuda");
    let (archive_dir, cuda_major, cuda_minor) = if let Some(existing) =
        fs::read_dir(&output_dir)?.flatten().find_map(|e| {
            let path = e.path();
            let name = path.file_name()?.to_str()?;
            if path.is_dir() && name.starts_with(&cached_prefix) {
                // Parse cuda major/minor from directory name e.g. "nccl_2.30.4-1+cuda13.2_x86_64"
                let after_cuda = name.strip_prefix(&cached_prefix)?;
                let cuda_ver = after_cuda.split('_').next()?;
                let (maj_str, min_str) = cuda_ver.split_once('.')?;
                let cuda_major: u32 = maj_str.parse().ok()?;
                let cuda_minor: u32 = min_str.parse().ok()?;
                Some((path, cuda_major, cuda_minor))
            } else {
                None
            }
        }) {
        existing
    } else {
        let pairings = download::nccl_cuda_pairings(&full_version, base_url).context(format!(
            "Failed to discover CUDA pairings for NCCL {full_version}"
        ))?;
        let (cuda_major, cuda_minor) = pairings[0];

        let filename = format!("nccl_{full_version}-1+cuda{cuda_major}.{cuda_minor}_x86_64.txz");
        let archive_dir = output_dir.join(filename.trim_end_matches(".txz"));

        if !archive_dir.exists() {
            let full_url = format!("{base_url}/v{full_version}/{filename}");
            let out_path = output_dir.join(&filename);
            download::to_file(&full_url, &out_path, multi_progress)?;
            extract::extract_archive(&out_path, &output_dir, multi_progress)?;
        }
        (archive_dir, cuda_major, cuda_minor)
    };

    module.run_bindgen(
        &format!("{major:02}{minor:02}{patch}"),
        &archive_dir,
        &primary_archives_map[format!("cuda-{cuda_major:02}{cuda_minor:02}0").as_str()],
    )
}

fn get_redist_lib_archive(
    (major, minor, patch): (u32, u32, u32),
    module: &ModuleConfig,
    primary_archives_map: &HashMap<&str, Vec<PathBuf>>,
    downloads_dir: &Path,
    multi_progress: &MultiProgress,
) -> Result<()> {
    let url = match module.cudarc_name {
        "cudnn" => "https://developer.download.nvidia.com/compute/cudnn/redist/",
        "cutensor" => "https://developer.download.nvidia.com/compute/cutensor/redist/",
        other => panic!("Unknown lib-versioned redist module: {other}"),
    };

    let data = download::cuda_redist(major, minor, patch, url, downloads_dir, multi_progress)?;
    let variants = &data[module.redist_name]["linux-x86_64"];

    // Pick the newest CUDA variant available in the manifest.
    let (cuda_major, cuda_key) = variants
        .as_object()
        .context(format!(
            "Expected linux-x86_64 entry for {} {major}.{minor}.{patch}",
            module.redist_name
        ))?
        .keys()
        .filter_map(|k| Some((k.strip_prefix("cuda")?.parse::<u32>().ok()?, k.as_str())))
        .max_by_key(|&(n, _)| n)
        .context(format!(
            "No CUDA variants found for {} {major}.{minor}.{patch}",
            module.redist_name
        ))?;

    let lib = &variants[cuda_key];

    let path = lib["relative_path"].as_str().context(format!(
        "Missing relative_path for {} {major}.{minor}.{patch}",
        module.redist_name
    ))?;
    let checksum = lib["sha256"].as_str().context(format!(
        "Missing sha256 for {} {major}.{minor}.{patch}",
        module.redist_name
    ))?;
    let full_url = format!("{url}/{path}");

    let output_dir = downloads_dir.join(module.cudarc_name);
    let parts: Vec<_> = Path::new(path)
        .file_name()
        .context(format!("Failed to get file name from {path}"))?
        .to_str()
        .expect("A valid filename")
        .split('.')
        .collect();
    let n = parts.len();
    let name = parts.into_iter().take(n - 2).collect::<Vec<_>>().join(".");
    let archive_dir = output_dir.join(name);

    if !archive_dir.exists() {
        fs::create_dir_all(&output_dir).context(format!(
            "Failed to create directory {}",
            output_dir.display()
        ))?;
        let out_path = output_dir.join(
            Path::new(path)
                .file_name()
                .context(format!("Failed to get file name from {path}"))?,
        );
        download::to_file_with_checksum(&full_url, &out_path, checksum, multi_progress)?;
        extract::extract_archive(&out_path, &output_dir, multi_progress)
            .context("Extracting archive")?;
    }

    let prefix = format!("cuda-{:02}", cuda_major);
    let primary_archives = primary_archives_map
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .max()
        .and_then(|k| primary_archives_map.get(k))
        .unwrap();
    module.run_bindgen(
        &format!("{major:02}{minor:02}{patch}"),
        &archive_dir,
        primary_archives,
    )
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Generating the bindings from scratch takes a long
    /// time, but even if every archive is there too
    /// because we have to check Nvidia's website for updates
    /// Using this flag will skip that steps if you know you bindings
    /// exist and are up to date.
    #[arg(long, action)]
    skip_bindings: bool,

    #[arg(long, action)]
    cuda_version: Option<String>,

    /// Specify a single target to generate bindings for.
    #[arg(long, action)]
    target: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut modules = create_modules();
    if let Some(target) = args.target {
        modules.retain(|m| m.cudarc_name.contains(&target));
    }

    let mut cuda_versions = vec![
        "cuda-11040",
        "cuda-11050",
        "cuda-11060",
        "cuda-11070",
        "cuda-11080",
        "cuda-12000",
        "cuda-12010",
        "cuda-12020",
        "cuda-12030",
        "cuda-12040",
        "cuda-12050",
        "cuda-12060",
        "cuda-12080",
        "cuda-12090",
        "cuda-13000",
        "cuda-13010",
        "cuda-13020",
    ];
    if let Some(version) = args.cuda_version {
        cuda_versions.retain(|&v| v == version);
    }

    if !args.skip_bindings {
        create_bindings(&modules, &cuda_versions)?;
    }
    merge::merge_bindings(&modules)?;

    std::process::Command::new("cargo")
        .arg("fmt")
        .current_dir(std::fs::canonicalize("../")?)
        .status()?;
    Ok(())
}
