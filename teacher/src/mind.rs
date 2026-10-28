use std::path::Path;
use std::sync::Arc;

use engine::bot::Brain;
use engine::packed::{Mind, Packed};
use engine::search::{Oracle, Search};

pub fn load(packed: &Packed, devices: &[usize]) -> Result<Brain, String> {
    let m: Mind = serde_json::from_value(packed.manifest.search.clone()).map_err(|e| e.to_string())?;
    match m.mind.as_str() {
        "greedy" => Ok(Brain::Greedy),
        "search" => Ok(Brain::Search { search: Search::Full(m.sims), oracle: gpu(&packed.dir.join(&packed.manifest.weights), devices)? }),
        other => Err(format!("unknown mind {other}")),
    }
}

#[cfg(feature = "gpu")]
fn gpu(weights: &Path, devices: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    let net = crate::net::Net::load_bin(weights.to_str().ok_or("invalid weights path")?).map_err(|e| format!("{}: {e}", weights.display()))?;
    Ok(Arc::new(crate::cuda::Gpu::new(devices, &net)?))
}

#[cfg(not(feature = "gpu"))]
fn gpu(weights: &Path, _: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    Err(format!("{} needs a bot built with the gpu feature", weights.display()))
}
