# Bindings Generator

This is a rust binary that does the following:
1. Reconstructs per-version bindings for already-supported cuda versions from the committed merged bindings in `../src` (no downloads needed)
2. Downloads cuda headers from <https://developer.download.nvidia.com/compute/cuda/redist/> for the remaining versions (typically just a newly added one)
3. Generate bindings for each of those versions separately
4. Merge the bindings together to:
    1. Unify static-linking/dynamic-linking/dynamic-loading
    2. Reduce code duplication across toolkit versions (they are generally additive)

Usage:
```bash
cargo run --release
```

## Adding a new cuda version

Add the version to `CUDA_VERSIONS` in `src/main.rs` and run the generator. Bindings for the existing versions are reconstructed from `../src/*/sys/mod.rs`, so only the new version's headers are downloaded and run through bindgen.

Note: the lib-versioned modules (cudnn, nccl, cutensor) generate against the newest cuda toolkit's headers. Reconstruction keeps their committed bindings as-is; pass `--no-reconstruct` (and optionally `--target <module>`) to regenerate them from scratch against the new toolkit.

## Flags

- `--no-reconstruct`: skip reconstruction and regenerate everything from the NVIDIA archives (the previous behavior)
- `--validate-reconstruction`: compare reconstructed bindings against the per-version files in `out/` from a real generation run, then exit
