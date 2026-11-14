use std::sync::Arc;

use engine::bot::{Brain, Opponent};
use engine::packed::{Mind, Packed};
use engine::search::Oracle;

use crate::net::Net;

pub fn load(packed: &Packed, devices: &[usize]) -> Result<(Brain, Opponent), String> {
    let m: Mind = serde_json::from_value(packed.manifest.search.clone()).map_err(|e| e.to_string())?;
    match m.mind.as_str() {
        "greedy" => Ok((Brain::Greedy, Opponent::Counted)),
        "search" => {
            let oracle = oracle(packed, devices)?;
            let opponent = m.human.map_or(Opponent::Counted, |h| Opponent::Human(h, oracle.clone()));
            Ok((Brain::Search { search: m.search, oracle }, opponent))
        }
        other => Err(format!("unknown mind {other:?}")),
    }
}

pub fn oracle(packed: &Packed, devices: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    let weights = packed.dir.join(&packed.manifest.weights);
    let net = Net::load_bin(weights.to_str().ok_or("invalid weights path")?).map_err(|e| format!("{}: {e}", weights.display()))?;
    accelerator(&net, devices)
}

#[cfg(feature = "metal")]
pub fn accelerator(net: &Net, _: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    Ok(Arc::new(crate::metal::Metal::new(net)?))
}

#[cfg(all(feature = "gpu", not(feature = "metal")))]
pub fn accelerator(net: &Net, devices: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    Ok(Arc::new(crate::cuda::Gpu::new(devices, net)?))
}

#[cfg(not(any(feature = "gpu", feature = "metal")))]
pub fn accelerator(_: &Net, _: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    Err("this bot was built without a gpu backend".into())
}
