use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
pub struct Mind {
    pub mind: String,
    #[serde(flatten)]
    pub search: crate::search::Search,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Manifest {
    pub format: u32,
    pub name: String,
    pub sha: String,
    pub binary: String,
    pub weights: String,
    pub weights_sha: String,
    pub search: serde_json::Value,
    pub minutes: f64,
    pub note: String,
}

pub struct Packed {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Packed {
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = path.join("bot.json");
        let manifest: Manifest = serde_json::from_str(
            &fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?,
        )
        .map_err(|e| format!("{}: {e}", file.display()))?;
        if manifest.format != 1 {
            return Err(format!(
                "{} has bot format {}",
                path.display(),
                manifest.format
            ));
        }
        for (name, expected) in [
            ("bot", &manifest.binary),
            (&manifest.weights, &manifest.weights_sha),
        ] {
            let file = path.join(name);
            let bytes = fs::read(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            let actual = format!("{:x}", Sha256::digest(bytes));
            if &actual != expected {
                return Err(format!("{} does not match bot.json", file.display()));
            }
        }
        Ok(Self {
            dir: path.into(),
            manifest,
        })
    }
}
