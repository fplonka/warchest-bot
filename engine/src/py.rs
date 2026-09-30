use std::sync::{Arc, LazyLock};

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::board::board;
use crate::net::Net;
use crate::online::{Game, Site};
use crate::pbs::{expand_row, mirror_row, pack_row, Ctx, PUBFEAT, ROW_BYTES};
use crate::selfplay::Data;
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

static NET: LazyLock<RwLock<Option<Arc<Net>>>> = LazyLock::new(|| RwLock::new(None));

#[cfg(feature = "gpu")]
fn net() -> PyResult<Arc<Net>> {
    NET.read().clone().ok_or_else(|| PyValueError::new_err("no weights pushed"))
}

#[pyfunction]
fn set_weights(w: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>, ln: PyReadonlyArray1<f32>) -> PyResult<()> {
    let net = Net::from_flat(w.as_slice()?, b.as_slice()?, ln.as_slice()?).map_err(PyValueError::new_err)?;
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
        use crate::{bot::Brain, search::Search, selfplay::Playout};
        let net = net()?;
        let gpu = Arc::new(crate::cuda::Gpu::new(&devices, &net).map_err(PyRuntimeError::new_err)?);
        let search = |search| Brain::Search { search, oracle: gpu.clone() };
        let playout = Playout { full: search(Search::Full(sims)), fast: search(Search::Fast(fast_sims)), p_full };
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

#[cfg(feature = "gpu")]
#[pyfunction]
fn evaluate(rows: Vec<u8>, desc: Vec<u8>, off: Vec<u32>, device: usize) -> PyResult<(Vec<f32>, Vec<f32>)> {
    use crate::search::Oracle;
    let net = net()?;
    let gpu = crate::cuda::Gpu::new(&[device], &net).map_err(PyRuntimeError::new_err)?;
    let answer = gpu.eval(crate::search::Query { rows, desc, off });
    Ok((answer.value, answer.prior))
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
fn hex_neighbours() -> Vec<u8> {
    crate::board::neighbour_gather()
}

#[pyfunction]
fn hex_mirror() -> Vec<u8> {
    (0..crate::board::N_HEXES).map(|h| crate::state::mirror_hex(h) as u8).collect()
}

#[pyfunction]
fn sample_positions(games: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u32>) {
    let mut rng = crate::rng::Rng::new(seed);
    let mut q = crate::search::Query::default();
    for _ in 0..games {
        let mut s = crate::selfplay::game(&mut rng);
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
fn rules_table_hash() -> u64 {
    crate::pbs::rules_table_hash()
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

#[pyfunction]
fn opening(white: Vec<u16>, black: Vec<u16>, first: u8) -> PyResult<Vec<u8>> {
    let mut s = crate::arena::Draft { white, black, first }.state().map_err(PyValueError::new_err)?;
    while s.is_chance() {
        s.apply_inplace(s.legal_actions()[0]);
    }
    let mut row = vec![0u8; ROW_BYTES];
    pack_row(&s, &Ctx::new(&s), &mut row);
    Ok(row)
}

#[pyclass(name = "Game")]
struct Online(Game);

fn site(json: &str) -> PyResult<Site> {
    serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl Online {
    #[new]
    fn new(journal: &str) -> PyResult<Online> {
        Game::new(journal).map(Online).map_err(PyValueError::new_err)
    }

    fn sync(&mut self, json: &str) -> PyResult<usize> {
        self.0.sync(&site(json)?).map_err(PyValueError::new_err)
    }

    fn rewind(&mut self, lines: usize) -> PyResult<()> {
        self.0.rewind(lines).map_err(PyValueError::new_err)
    }

    fn command(&self, json: &str) -> PyResult<Option<String>> {
        let cmd = self.0.command(&site(json)?).map_err(PyValueError::new_err)?;
        Ok(cmd.map(|c| serde_json::to_string(&c).expect("a command serializes")))
    }

    fn act(&mut self, code: u32) -> PyResult<()> {
        self.0.act(code).map_err(PyValueError::new_err)
    }

    fn ask(&self, told: usize) -> String {
        self.0.ask(told)
    }

    fn journal(&self) -> String {
        self.0.journal()
    }

    fn __len__(&self) -> usize {
        self.0.obs().len()
    }
}

#[pymodule]
fn warchest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(warm, m)?)?;
    #[cfg(feature = "gpu")]
    {
        m.add_class::<Farm>()?;
        m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    }
    m.add_function(wrap_pyfunction!(card_features_table, m)?)?;
    m.add_function(wrap_pyfunction!(hex_location_flags, m)?)?;
    m.add_function(wrap_pyfunction!(hex_neighbours, m)?)?;
    m.add_function(wrap_pyfunction!(hex_mirror, m)?)?;
    m.add_function(wrap_pyfunction!(sample_positions, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_rows, m)?)?;
    m.add_function(wrap_pyfunction!(rules_table_hash, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(opening, m)?)?;
    m.add_class::<Online>()?;
    m.add("PUBFEAT", PUBFEAT)?;
    m.add("HORIZONS", crate::selfplay::HORIZONS.to_vec())?;
    m.add("ROW_BYTES", ROW_BYTES)?;
    m.add("ACT_BYTES", crate::search::ACT_BYTES)?;
    m.add("N_HEXES", crate::board::N_HEXES)?;
    m.add("N_LOCATIONS", crate::board::N_LOCATIONS)?;
    m.add("N_UNITS", N_UNITS)?;
    m.add("N_KINDS", crate::actions::N_KINDS)?;
    m.add("KIND_NAMES", crate::actions::KIND_NAMES.to_vec())?;
    m.add("CARD_FEATS", CARD_FEATS)?;
    m.add("NSLOT", crate::pbs::NSLOT)?;
    m.add("NTYPE", crate::pbs::NTYPE)?;
    m.add("HEX_CH", crate::pbs::HEX_CH)?;
    m.add("HEX_FACTS", crate::pbs::HEX_FACTS)?;
    m.add("PILE_COUNTS", crate::pbs::PILE_COUNTS)?;
    m.add("LOOSE", crate::pbs::LOOSE)?;
    m.add("OFF_PILES", crate::pbs::OFF_PILES)?;
    m.add("OFF_CARDS", crate::pbs::OFF_CARDS)?;
    m.add("OFF_LOOSE", crate::pbs::OFF_LOOSE)?;
    m.add("ROW_HEX_OWNER", crate::pbs::ROW_HEX_OWNER)?;
    m.add("ROW_HEX_SLOT", crate::pbs::ROW_HEX_SLOT)?;
    m.add("ROW_HEX_MARKER", crate::pbs::ROW_HEX_MARKER)?;
    Ok(())
}
