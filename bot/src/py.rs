use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

use warchest::actions::Action;
use warchest::board::board;
use crate::net::Net;
use engine::pbs::{expand_row, mirror_row, Ctx, PUBFEAT, ROW_BYTES};
use engine::search::{mirror_desc, Oracle, Query, ACT_BYTES};
use crate::selfplay::Data;
use warchest::units::{write_card_features, CARD_FEATS, N_UNITS};

static NET: LazyLock<RwLock<Option<Arc<Net>>>> = LazyLock::new(|| RwLock::new(None));

#[cfg(any(feature = "gpu", feature = "metal"))]
fn net() -> PyResult<Arc<Net>> {
    NET.read().clone().ok_or_else(|| PyValueError::new_err("no weights pushed"))
}

#[pyfunction]
fn set_weights(w: PyReadonlyArray1<f32>) -> PyResult<()> {
    let net = Net::new(w.as_slice()?.to_vec()).map_err(PyValueError::new_err)?;
    *NET.write() = Some(Arc::new(net));
    Ok(())
}

fn data_to_dict(py: Python<'_>, d: Data) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    const PLAYS: [&str; 7] = ["attack", "pass", "deploy", "bolster", "maneuver", "recruit", "claim_initiative"];
    for (name, v) in [
        ("games", d.games),
        ("decisions", d.decisions),
        ("white_wins", d.wins[0]),
        ("black_wins", d.wins[1]),
        ("draws", d.draws),
        ("timeouts", d.timeouts),
    ] {
        out.set_item(name, v)?;
    }
    for (name, v) in PLAYS.iter().zip(d.plays) {
        out.set_item(format!("plays_{name}"), v)?;
    }
    out.set_item("kinds", warchest::actions::KIND_NAMES.iter().zip(d.kinds.0).collect::<Vec<_>>().into_py_dict_bound(py))?;
    out.set_item("rows", d.q.rows.into_pyarray_bound(py))?;
    out.set_item("desc", d.q.desc.into_pyarray_bound(py))?;
    out.set_item("off", d.q.off.into_pyarray_bound(py))?;
    out.set_item("target", d.target.into_pyarray_bound(py))?;
    let shape = [d.z.len(), crate::selfplay::HORIZONS.len()];
    out.set_item("z", d.z.into_flattened().into_pyarray_bound(py).reshape(shape)?)?;
    Ok(out.into())
}

#[pyfunction]
fn warm(py: Python<'_>, games: usize, seed: u64) -> PyResult<PyObject> {
    let d = py.allow_threads(|| crate::selfplay::warm(games, seed));
    data_to_dict(py, d)
}

#[cfg(feature = "gpu")]
#[pyclass]
struct Farm {
    gpu: Arc<crate::cuda::Gpu>,
    farm: crate::selfplay::Farm,
    net: Arc<Net>,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl Farm {
    #[new]
    fn new(seed: u64, games: usize, devices: Vec<usize>, sims: u32, fast_sims: u32, p_full: f64, chunk: usize) -> PyResult<Farm> {
        use crate::selfplay::Playout;
        use engine::{bot::Brain, search::Search};
        let net = net()?;
        let gpu = Arc::new(crate::cuda::Gpu::new(&devices, &net).map_err(PyRuntimeError::new_err)?);
        let search = |search| Brain::Search { search, oracle: gpu.clone() };
        let playout = Playout { full: search(Search::full(sims)), fast: search(Search::fast(fast_sims)), p_full };
        let farm = crate::selfplay::Farm::new(seed, games, Arc::new(playout), chunk);
        Ok(Farm { gpu, farm, net })
    }

    fn collect(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let next = net()?;
        if !Arc::ptr_eq(&next, &self.net) {
            self.gpu.set_weights(&next).map_err(PyRuntimeError::new_err)?;
            self.net = next;
        }
        let farm = &self.farm;
        let d = py.allow_threads(|| farm.collect());
        data_to_dict(py, d.ok_or_else(|| PyRuntimeError::new_err("a self-play game died; the log says why"))?)
    }

    fn gpu_stats(&self) -> Vec<(u64, u64)> {
        let read = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        self.gpu.stats.iter().map(|s| (read(&s.calls), read(&s.rows))).collect()
    }
}

#[cfg(any(feature = "gpu", feature = "metal"))]
#[pyfunction]
fn evaluate(
    py: Python<'_>,
    rows: PyReadonlyArray1<u8>,
    desc: PyReadonlyArray1<u8>,
    off: PyReadonlyArray1<u32>,
    device: usize,
) -> PyResult<PyObject> {
    let gpu = crate::mind::accelerator(&*net()?, &[device]).map_err(PyRuntimeError::new_err)?;
    let q = Query { rows: rows.as_slice()?.to_vec(), desc: desc.as_slice()?.to_vec(), off: off.as_slice()?.to_vec() };
    let answer = py.allow_threads(|| gpu.eval(&q));
    Ok((answer.value.into_pyarray_bound(py), answer.prior.into_pyarray_bound(py)).into_py(py))
}

#[pyfunction]
fn card_features_table() -> Vec<f32> {
    let mut out = vec![0.0; N_UNITS * CARD_FEATS];
    for u in 0..N_UNITS {
        write_card_features(u as u8, &mut out[u * CARD_FEATS..(u + 1) * CARD_FEATS]);
    }
    out
}

#[pyfunction]
fn hex_location_flags() -> Vec<u8> {
    board().is_location.iter().map(|&x| x as u8).collect()
}

#[pyfunction]
fn sample_positions(games: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u32>) {
    let mut rng = warchest::rng::Rng::new(seed);
    let mut q = Query::default();
    for _ in 0..games {
        let mut s = warchest::arena::game(&mut rng);
        let ctx = Ctx::new(&s);
        for _ in 0..160 {
            if s.is_terminal() {
                break;
            }
            let acts = s.legal_actions();
            if !s.is_chance() {
                q.push(&s, &ctx, &acts);
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    (q.rows, q.desc, q.off)
}

#[pyfunction]
fn mirror_rows<'py>(py: Python<'py>, rows: PyReadonlyArray1<u8>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let rows = rows.as_slice()?;
    if rows.len() % ROW_BYTES != 0 {
        return Err(PyValueError::new_err("rows are not whole"));
    }
    let mut out = vec![0u8; rows.len()];
    for (src, dst) in rows.chunks_exact(ROW_BYTES).zip(out.chunks_exact_mut(ROW_BYTES)) {
        mirror_row(src, dst);
    }
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn mirror_descs<'py>(py: Python<'py>, desc: PyReadonlyArray1<u8>) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let mut out = desc.as_slice()?.to_vec();
    if out.len() % ACT_BYTES != 0 {
        return Err(PyValueError::new_err("descs are not whole"));
    }
    out.chunks_exact_mut(ACT_BYTES).for_each(mirror_desc);
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
fn rules_table_hash() -> u64 {
    warchest::arena::rules_table_hash()
}

#[pyfunction]
fn expand_rows(rows: PyReadonlyArray1<u8>) -> PyResult<Vec<f32>> {
    let rows = rows.as_slice()?;
    if rows.len() % ROW_BYTES != 0 {
        return Err(PyValueError::new_err("rows is not a multiple of ROW_BYTES"));
    }
    let mut out = vec![0.0f32; rows.len() / ROW_BYTES * PUBFEAT];
    for (row, x) in rows.chunks_exact(ROW_BYTES).zip(out.chunks_exact_mut(PUBFEAT)) {
        expand_row(row, x);
    }
    Ok(out)
}

#[pyfunction]
fn expand_rows_cuda(rows: u64, cards: u64, locations: u64, out: u64, n: usize, stream: usize, device: i32) -> PyResult<()> {
    #[cfg(feature = "gpu")]
    {
        crate::cuda::expand_rows_torch(rows, cards, locations, out, n, stream, device).map_err(PyRuntimeError::new_err)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (rows, cards, locations, out, n, stream, device);
        Err(PyRuntimeError::new_err("warchest was built without CUDA"))
    }
}

#[pyclass]
struct Player {
    oracle: Arc<dyn Oracle>,
}

#[pymethods]
impl Player {
    #[new]
    fn new(bot: PathBuf) -> PyResult<Player> {
        let packed = engine::packed::Packed::load(&bot).map_err(PyValueError::new_err)?;
        Ok(Player { oracle: crate::mind::oracle(&packed, &[0]).map_err(PyRuntimeError::new_err)? })
    }

    fn values(&self, py: Python<'_>, drafts: Vec<(Vec<u16>, Vec<u16>, u8)>) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            let (mut q, mut weight, mut spans) = (Query::default(), Vec::new(), vec![0]);
            for (white, black, first) in drafts.iter().cloned() {
                let start = warchest::arena::Draft { white, black, first }.state()?;
                let mut hands = Vec::new();
                opening_hands(&start.draw_pool(first), 0, 3, &mut Vec::new(), 1, &mut hands);
                let total = hands.iter().map(|(_, ways)| ways).sum::<u32>() as f64;
                for (hand, ways) in hands {
                    let (mut s, mut rng, mut hand) = (start, warchest::rng::Rng::new(0), hand.into_iter());
                    while s.is_chance() {
                        let p = s.to_act();
                        match p == first {
                            true => s.apply_inplace(Action::DrawCoin { unit: hand.next().expect("the first mover draws three coins") }),
                            false => _ = warchest::rules::resolve_chance(&mut s, p, &mut rng),
                        }
                    }
                    q.push(&s, &Ctx::new(&s), &s.legal_actions());
                    weight.push(ways as f64 / total * if s.to_act() == 0 { 1.0 } else { -1.0 });
                }
                spans.push(weight.len());
            }
            let value = self.oracle.eval(&q).value;
            Ok(spans.windows(2).map(|w| (w[0]..w[1]).map(|i| value[i] as f64 * weight[i]).sum()).collect())
        })
        .map_err(|e: String| PyValueError::new_err(e))
    }
}

fn opening_hands(pool: &[u8], u: usize, left: u8, hand: &mut Vec<u8>, ways: u32, out: &mut Vec<(Vec<u8>, u32)>) {
    if left == 0 {
        return out.push((hand.clone(), ways));
    }
    if u == pool.len() {
        return;
    }
    for k in 0..=left.min(pool[u]) {
        hand.extend(std::iter::repeat_n(u as u8, k as usize));
        let choose = (0..k as u32).fold(1, |c, i| c * (pool[u] as u32 - i) / (i + 1));
        opening_hands(pool, u + 1, left - k, hand, ways * choose, out);
        hand.truncate(hand.len() - k as usize);
    }
}

#[pymodule]
#[pyo3(name = "warchest")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(warm, m)?)?;
    #[cfg(feature = "gpu")]
    {
        m.add_class::<Farm>()?;
    }
    #[cfg(any(feature = "gpu", feature = "metal"))]
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(card_features_table, m)?)?;
    m.add_function(wrap_pyfunction!(hex_location_flags, m)?)?;
    m.add_function(wrap_pyfunction!(sample_positions, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_rows, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_descs, m)?)?;
    m.add_function(wrap_pyfunction!(rules_table_hash, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows_cuda, m)?)?;
    m.add_class::<Player>()?;
    m.add("PUBFEAT", PUBFEAT)?;
    m.add("HORIZONS", crate::selfplay::HORIZONS.to_vec())?;
    m.add("ROW_BYTES", ROW_BYTES)?;
    m.add("ACT_BYTES", ACT_BYTES)?;
    m.add("N_HEXES", warchest::board::N_HEXES)?;
    m.add("N_LOCATIONS", warchest::board::N_LOCATIONS)?;
    m.add("N_UNITS", N_UNITS)?;
    m.add("N_KINDS", warchest::actions::N_KINDS)?;
    m.add("KIND_NAMES", warchest::actions::KIND_NAMES.to_vec())?;
    m.add("CARD_FEATS", CARD_FEATS)?;
    m.add("NSLOT", engine::pbs::NSLOT)?;
    m.add("NTYPE", engine::pbs::NTYPE)?;
    m.add("HEX_CH", engine::pbs::HEX_CH)?;
    m.add("HEX_FACTS", engine::pbs::HEX_FACTS)?;
    m.add("PILE_COUNTS", engine::pbs::PILE_COUNTS)?;
    m.add("LOOSE", engine::pbs::LOOSE)?;
    m.add("OFF_PILES", engine::pbs::OFF_PILES)?;
    m.add("OFF_CARDS", engine::pbs::OFF_CARDS)?;
    m.add("OFF_LOOSE", engine::pbs::OFF_LOOSE)?;
    m.add("ROW_TO_ACT", engine::pbs::ROW_TO_ACT)?;
    m.add("ATTENTION_INDEX", crate::net::attention_index())?;
    Ok(())
}
