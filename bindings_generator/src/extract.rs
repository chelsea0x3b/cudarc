use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use indicatif::{MultiProgress, ProgressBar};
use std::fs::File;
use std::path::{Path, PathBuf};
use tar::Archive;
use xz2::read::XzDecoder;

fn extract_tar_gz(tarball_path: &Path) -> Result<PathBuf> {
    let output_dir = match tarball_path.extension().and_then(|s| s.to_str()) {
        Some("tgz") => tarball_path.with_extension(""),
        Some("gz") => tarball_path.with_extension("").with_extension(""),
        _ => unreachable!(),
    };
    if output_dir.exists() && output_dir.read_dir().unwrap().next().is_some() {
        // no work to be done
        return Ok(output_dir);
    }

    let tarball = File::open(tarball_path)
        .context(format!("Failed to open tarball {}", tarball_path.display()))?;

    let decompressed = GzDecoder::new(tarball);
    let mut archive = Archive::new(decompressed);

    archive
        .unpack(tarball_path.parent().unwrap())
        .with_context(|| {
            format!(
                "Failed to unpack {} to {}",
                tarball_path.display(),
                output_dir.display()
            )
        })?;

    Ok(output_dir)
}

fn extract_tar_xz(tarball_path: &Path) -> Result<PathBuf> {
    let output_dir = match tarball_path.extension().and_then(|s| s.to_str()) {
        Some("txz") => tarball_path.with_extension(""),
        Some("xz") => tarball_path.with_extension("").with_extension(""),
        _ => unreachable!(),
    };
    if output_dir.exists() && output_dir.read_dir().unwrap().next().is_some() {
        // no work to be done
        return Ok(output_dir);
    }

    let tarball = File::open(tarball_path)
        .context(format!("Failed to open tarball {}", tarball_path.display()))?;

    let decompressed = XzDecoder::new(tarball);
    let mut archive = Archive::new(decompressed);

    archive
        .unpack(tarball_path.parent().unwrap())
        .with_context(|| {
            format!(
                "Failed to unpack {} to {}",
                tarball_path.display(),
                output_dir.display()
            )
        })?;

    Ok(output_dir)
}

pub(crate) fn extract_archive(
    archive_path: &Path,
    multi_progress: &MultiProgress,
) -> Result<PathBuf> {
    let pb = multi_progress.add(ProgressBar::new_spinner());
    pb.set_message(format!("Extracting {}", archive_path.display()));
    match archive_path.extension().and_then(|s| s.to_str()) {
        Some("gz") => extract_tar_gz(archive_path),
        Some("xz") => extract_tar_xz(archive_path),
        Some("tgz") => extract_tar_gz(archive_path),
        Some("txz") => extract_tar_xz(archive_path),
        _ => Err(anyhow::anyhow!(
            "Unsupported archive format: {}",
            archive_path.display()
        )),
    }
}
