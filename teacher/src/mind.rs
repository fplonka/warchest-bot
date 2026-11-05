use std::path::Path;
use std::sync::Arc;

use engine::bot::{Brain, Opponent};
use engine::nnue::Nnue;
use engine::packed::{Mind, Packed};
use engine::search::{Answer, Oracle, Query};

use crate::net::Net;

pub fn load(packed: &Packed, devices: &[usize]) -> Result<(Brain, Opponent), String> {
    let m: Mind = serde_json::from_value(packed.manifest.search.clone()).map_err(|e| e.to_string())?;
    if m.mind == "greedy" {
        return Ok((Brain::Greedy, Opponent::Counted));
    }
    let oracle = oracle(packed, devices)?;
    let opponent = m.human.map_or(Opponent::Counted, |h| Opponent::Human(h, oracle.clone()));
    Ok((Brain::Search { search: m.search, oracle }, opponent))
}

fn oracle(packed: &Packed, devices: &[usize]) -> Result<Arc<dyn Oracle>, String> {
    let search = &packed.manifest.search;
    let part = |key: &str| oracle(&Packed::load(Path::new(search[key].as_str().ok_or(format!("a mix needs a {key} bot"))?))?, devices);
    match search["mind"].as_str() {
        Some("nnue") => Ok(Arc::new(Nnue::load(&packed.dir.join(&packed.manifest.weights))?)),
        Some("search") => {
            let weights = packed.dir.join(&packed.manifest.weights);
            accelerator(&Net::load_bin(weights.to_str().ok_or("invalid weights path")?).map_err(|e| format!("{}: {e}", weights.display()))?, devices)
        }
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
