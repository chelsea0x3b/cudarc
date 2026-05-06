use anyhow::{Context, Result, bail};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Cargo feature name for this version. cudarc uses two encodings:
    ///   - "cuda" prefix encodes major.minor.patch (e.g. cuda-12080)
    ///   - other prefixes encode major.minor only, dropping patch (e.g. nccl-02027)
    pub fn feature_name(&self, prefix: &str) -> String {
        if prefix == "cuda" {
            format!("{prefix}-{:02}{:02}{}", self.major, self.minor, self.patch)
        } else {
            format!("{prefix}-{:02}{:03}", self.major, self.minor)
        }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for Version {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            bail!("Version must be in 'major.minor.patch' form: {s}");
        }
        Ok(Self {
            major: parts[0]
                .parse()
                .with_context(|| format!("parsing major in {s}"))?,
            minor: parts[1]
                .parse()
                .with_context(|| format!("parsing minor in {s}"))?,
            patch: parts[2]
                .parse()
                .with_context(|| format!("parsing patch in {s}"))?,
        })
    }
}
