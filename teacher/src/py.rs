use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

use warchest::board::board;
use crate::net::Net;
use engine::online::{Game, Site};
use engine::pbs::{expand_row, mirror_row, mover_row, pack_row, Ctx, PUBFEAT, ROW_BYTES};
use engine::nnue::{self, Nnue};
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

#[cfg(feature = "gpu")]
#[pyclass]
struct Generator {
    farm: crate::selfplay::Farm,
    recorder: Arc<crate::selfplay::Recorder>,
}

#[cfg(feature = "gpu")]
#[pymethods]
impl Generator {
    #[new]
    fn new(seed: u64, games: usize, devices: Vec<usize>, sims: u32, fast_sims: u32, p_full: f64, keep: f64, chunk: usize) -> PyResult<Generator> {
        use crate::selfplay::{Playout, Recorder};
        use engine::{bot::Brain, search::Search};
        let gpu = Arc::new(crate::cuda::Gpu::new(&devices, &*net()?).map_err(PyRuntimeError::new_err)?);
        let recorder = Arc::new(Recorder::new(gpu, keep));
        let search = |search| Brain::Search { search, oracle: recorder.clone() };
        let playout = Playout { full: search(Search::full(sims)), fast: search(Search::fast(fast_sims)), p_full };
        Ok(Generator { farm: crate::selfplay::Farm::new(seed, games, Arc::new(playout), chunk), recorder })
    }

    fn take(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let farm = &self.farm;
        py.allow_threads(|| farm.collect()).ok_or_else(|| PyRuntimeError::new_err("a self-play game died; the log says why"))?;
        labelled(py, self.recorder.take())
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
fn nnue_features(py: Python<'_>, rows: PyReadonlyArray1<u8>, desc: PyReadonlyArray1<u8>, off: PyReadonlyArray1<u32>) -> PyResult<PyObject> {
    let (rows, desc, off) = (rows.as_slice()?, desc.as_slice()?, off.as_slice()?);
    let (feats, counts, moves, sizes, phases) = py.allow_threads(|| {
        let (mut feats, mut counts, mut moves, mut sizes, mut phases) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (row, span) in rows.chunks_exact(ROW_BYTES).zip(off.windows(2)) {
            let row = row.try_into().unwrap();
            phases.push(nnue::phase(row));
            let f = nnue::features(row, &desc[span[0] as usize * ACT_BYTES..span[1] as usize * ACT_BYTES]);
            feats.extend_from_slice(&f);
            counts.push(f.len() as u8);
            for d in desc[span[0] as usize * ACT_BYTES..span[1] as usize * ACT_BYTES].chunks_exact(ACT_BYTES) {
                let m = nnue::move_features(row, d);
                moves.extend_from_slice(&m);
                sizes.push(m.len() as u8);
            }
        }
        (feats, counts, moves, sizes, phases)
    });
    Ok((feats.into_pyarray_bound(py), counts.into_pyarray_bound(py), moves.into_pyarray_bound(py), sizes.into_pyarray_bound(py),
        phases.into_pyarray_bound(py))
        .into_py(py))
}

#[pyfunction]
fn nnue_leaves(py: Python<'_>, weights: PathBuf, sims: u32, keep: f64, games: usize, seed: u64) -> PyResult<PyObject> {
    let nnue: Arc<dyn Oracle> = Arc::new(Nnue::load(&weights).map_err(PyValueError::new_err)?);
    let l = py.allow_threads(|| crate::selfplay::leaves(nnue, sims, keep, games, seed));
    labelled(py, l)
}

fn labelled(py: Python<'_>, l: crate::selfplay::Labelled) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("rows", l.q.rows.into_pyarray_bound(py))?;
    out.set_item("desc", l.q.desc.into_pyarray_bound(py))?;
    out.set_item("off", l.q.off.into_pyarray_bound(py))?;
    out.set_item("value", l.value.into_pyarray_bound(py))?;
    out.set_item("prior", l.prior.into_pyarray_bound(py))?;
    Ok(out.into())
}

#[pyfunction]
fn nnue_eval(py: Python<'_>, weights: PathBuf, rows: PyReadonlyArray1<u8>, desc: PyReadonlyArray1<u8>, off: PyReadonlyArray1<u32>) -> PyResult<PyObject> {
    let nnue = Nnue::load(&weights).map_err(PyValueError::new_err)?;
    let q = Query { rows: rows.as_slice()?.to_vec(), desc: desc.as_slice()?.to_vec(), off: off.as_slice()?.to_vec() };
    let answer = py.allow_threads(|| nnue.eval(&q));
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

#[pyfunction]
fn opening(white: Vec<u16>, black: Vec<u16>, first: u8) -> PyResult<Vec<u8>> {
    let mut s = warchest::arena::Draft { white, black, first }.state().map_err(PyValueError::new_err)?;
    while s.is_chance() {
        s.apply_inplace(s.legal_actions()[0]);
    }
    let (mut row, mut mover) = (vec![0u8; ROW_BYTES], vec![0u8; ROW_BYTES]);
    pack_row(&s, &Ctx::new(&s), &mut row);
    mover_row(&row, &mut mover);
    Ok(mover)
}

#[pyfunction]
fn replay_snapshots(py: Python<'_>, snapshots: &str, log: &str) -> PyResult<PyObject> {
    let (draft, decisions) = engine::online::decisions(snapshots, log).map_err(PyValueError::new_err)?;
    let start = draft.state().map_err(PyValueError::new_err)?;
    let (ctx, mut q, mut chosen, mut mover) = (Ctx::new(&start), Query::default(), Vec::new(), Vec::new());
    for (s, a) in decisions.into_iter().filter(|(s, _)| !s.is_chance()) {
        let acts = s.legal_actions();
        if acts.len() > 1 {
            q.push(&s, &ctx, &acts);
            chosen.push(acts.iter().position(|&b| b == a).expect("a replayed move is legal") as u32);
            mover.push(s.to_act());
        }
    }
    let out = PyDict::new_bound(py);
    out.set_item("rows", q.rows.into_pyarray_bound(py))?;
    out.set_item("desc", q.desc.into_pyarray_bound(py))?;
    out.set_item("off", q.off.into_pyarray_bound(py))?;
    out.set_item("chosen", chosen.into_pyarray_bound(py))?;
    out.set_item("mover", mover.into_pyarray_bound(py))?;
    Ok(out.into())
}

#[pyclass]
struct Player {
    search: engine::search::Search,
    oracle: Arc<dyn Oracle>,
    opponent: engine::bot::Opponent,
    sessions: std::sync::Mutex<std::collections::HashMap<u32, engine::bot::Session>>,
}

#[pymethods]
impl Player {
    #[new]
    fn new(bot: PathBuf) -> PyResult<Player> {
        let packed = engine::packed::Packed::load(&bot).map_err(PyValueError::new_err)?;
        match crate::mind::load(&packed, &[0]).map_err(PyRuntimeError::new_err)? {
            (engine::bot::Brain::Search { search, oracle }, opponent) => Ok(Player { search, oracle, opponent, sessions: Default::default() }),
            _ => Err(PyValueError::new_err("a player searches")),
        }
    }

    fn decide(&self, py: Python<'_>, request: &str, sims: u32) -> PyResult<u32> {
        let request: warchest::arena::Request = serde_json::from_str(request).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ask = request.go.into_iter().next().ok_or_else(|| PyValueError::new_err("the request asks nothing"))?;
        let brain = engine::bot::Brain::Search { search: engine::search::Search { sims, ..self.search }, oracle: self.oracle.clone() };
        py.allow_threads(|| engine::bot::answer(ask, &self.sessions, &brain, &self.opponent)).map(|d| d.action).map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (snapshots, log, seat, human=None))]
    fn trace(&self, py: Python<'_>, snapshots: &str, log: &str, seat: u8, human: Option<(f64, f64)>) -> PyResult<Vec<f64>> {
        use engine::bot::{Human, Opponent, Session};
        use warchest::arena::Obs;
        let opponent = human.map_or(Opponent::Counted, |(temperature, epsilon)| Opponent::Human(Human { temperature, epsilon }, self.oracle.clone()));
        py.allow_threads(|| {
            let (draft, steps) = engine::online::decisions(snapshots, log)?;
            let mut session = Session::new(&draft, seat, 0, opponent)?;
            let mut out = Vec::new();
            for (s, a) in steps {
                let player = s.to_act();
                if !s.is_chance() && player != seat {
                    out.push(session.belief(&s));
                }
                session.observe(&match (s.is_chance(), player == seat) {
                    (true, mine) => Obs::Draw { player, code: mine.then(|| a.encode()) },
                    (false, true) => Obs::Act { player, key: a.encode() },
                    (false, false) => Obs::Act { player, key: warchest::actions::obs_key(&a) },
                })?;
            }
            Ok(out)
        })
        .map_err(|e: String| PyValueError::new_err(e))
    }

    fn values(&self, py: Python<'_>, drafts: Vec<(Vec<u16>, Vec<u16>, u8)>, deals: u64) -> PyResult<Vec<f64>> {
        py.allow_threads(|| {
            let mut q = Query::default();
            let mut sign = Vec::new();
            for (white, black, first) in drafts.iter().cloned() {
                let start = warchest::arena::Draft { white, black, first }.state()?;
                for deal in 0..deals {
                    let (mut s, mut rng) = (start, warchest::rng::Rng::new(deal));
                    while s.is_chance() {
                        let p = s.to_act();
                        warchest::rules::resolve_chance(&mut s, p, &mut rng);
                    }
                    q.push(&s, &Ctx::new(&s), &s.legal_actions());
                    sign.push(if s.to_act() == 0 { 1.0 } else { -1.0 });
                }
            }
            let value = self.oracle.eval(&q).value;
            Ok(value.chunks(deals as usize).zip(sign.chunks(deals as usize)).map(|(v, s)| v.iter().zip(s).map(|(&v, s)| v as f64 * s).sum::<f64>() / deals as f64).collect())
        })
        .map_err(|e: String| PyValueError::new_err(e))
    }
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
#[pyo3(name = "warchest")]
fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(warm, m)?)?;
    #[cfg(feature = "gpu")]
    {
        m.add_class::<Farm>()?;
        m.add_class::<Generator>()?;
    }
    #[cfg(any(feature = "gpu", feature = "metal"))]
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_features, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_eval, m)?)?;
    m.add_function(wrap_pyfunction!(nnue_leaves, m)?)?;
    m.add_function(wrap_pyfunction!(card_features_table, m)?)?;
    m.add_function(wrap_pyfunction!(hex_location_flags, m)?)?;
    m.add_function(wrap_pyfunction!(sample_positions, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_rows, m)?)?;
    m.add_function(wrap_pyfunction!(mirror_descs, m)?)?;
    m.add_function(wrap_pyfunction!(rules_table_hash, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows, m)?)?;
    m.add_function(wrap_pyfunction!(expand_rows_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(opening, m)?)?;
    m.add_function(wrap_pyfunction!(replay_snapshots, m)?)?;
    m.add_class::<Online>()?;
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
    m.add("NNUE_STACKS", nnue::STACKS)?;
    m.add("NNUE_L2", nnue::L2)?;
    m.add("NNUE_FEATURES", nnue::FEATURES)?;
    m.add("NNUE_MOVES", nnue::MOVES)?;
    m.add("ATTENTION_INDEX", crate::net::attention_index())?;
    Ok(())
}
