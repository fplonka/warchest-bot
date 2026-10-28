use std::path::Path;
use std::sync::Arc;

use engine::bot::Brain;
use engine::nnue::Nnue;
use engine::packed::{Mind, Packed};
use engine::search::{Answer, Oracle, Query, Search};

pub fn load(packed: &Packed, devices: &[usize]) -> Result<Brain, String> {
    let m: Mind = serde_json::from_value(packed.manifest.search.clone()).map_err(|e| e.to_string())?;
    match m.mind.as_str() {
        "greedy" => Ok(Brain::Greedy),
        _ => Ok(Brain::Search { search: Search::Full(m.sims), oracle: oracle(packed, devices)? }),
    }
}

fn oracle(packed: &Packed, devices: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    let search = &packed.manifest.search;
    let part = |key: &str| oracle(&Packed::load(Path::new(search[key].as_str().ok_or(format!("a mix needs a {key} bot"))?))?, devices);
    match search["mind"].as_str() {
        Some("nnue") => Ok(Arc::new(Nnue::load(&packed.dir.join(&packed.manifest.weights))?)),
        Some("search") => gpu(&packed.dir.join(&packed.manifest.weights), devices),
        Some("mix") => Ok(Arc::new(Mix { value: part("value")?, prior: part("prior")? })),
        other => Err(format!("unknown mind {other:?}")),
    }
}

struct Mix {
    value: Arc<dyn Oracle>,
    prior: Arc<dyn Oracle>,
}

impl Oracle for Mix {
    fn eval(&self, q: &Query) -> Answer {
        Answer { value: self.value.eval(q).value, prior: self.prior.eval(q).prior }
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
