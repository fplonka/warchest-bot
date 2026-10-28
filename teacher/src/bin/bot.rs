use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::Mutex;

use rayon::prelude::*;
use warchest::arena::{Ask, Done, Hello, Reply, Request, PROTOCOL};
use engine::bot::{Brain, Session};
use engine::packed::Packed;
use warchest::arena::rules_table_hash;

fn work(ask: Ask, sessions: &Mutex<HashMap<u32, Session>>, brain: &Brain) -> Result<Done, String> {
    let mut session = match ask.start {
        Some(start) => Session::new(&start.draft, start.seat, start.seed)?,
        None => sessions
            .lock()
            .unwrap()
            .remove(&ask.id)
            .ok_or_else(|| format!("game {} was never started", ask.id))?,
    };
    for obs in &ask.obs {
        session.observe(obs, brain.net())?;
    }
    let action = session.decide(brain)?;
    sessions.lock().unwrap().insert(ask.id, session);
    Ok(Done { id: ask.id, action })
}

const THINKERS: usize = 256;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().ok_or("usage: bot BOT_DIR --devices 0,1")?;
    let packed = Packed::load(path.as_ref())?;
    let flag = |name: &str| args.iter().position(|x| x == name).and_then(|i| args.get(i + 1));
    let device_text = flag("--devices").ok_or("packed bot needs --devices")?;
    let devices = device_text
        .split(',')
        .map(|x| x.parse().map_err(|_| format!("invalid device {x}")))
        .collect::<Result<Vec<_>, _>>()?;
    let brain = teacher::mind::load(&packed, &devices)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(THINKERS)
        .build()
        .map_err(|e| e.to_string())?;
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "{}",
        serde_json::to_string(&Hello {
            name: packed.manifest.name,
            protocol: PROTOCOL,
            rules: rules_table_hash(),
        })
        .unwrap()
    )
    .map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;
    let sessions = Mutex::new(HashMap::new());
    for line in std::io::stdin().lock().lines() {
        let request: Request =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        for id in request.drop {
            sessions.lock().unwrap().remove(&id);
        }
        if request.go.is_empty() {
            continue;
        }
        let done: Result<Vec<_>, _> = pool.install(|| {
            request.go.into_par_iter().map(|ask| work(ask, &sessions, &brain)).collect()
        });
        let reply = match done {
            Ok(done) => Reply { done, error: None },
            Err(error) => Reply {
                done: Vec::new(),
                error: Some(error),
            },
        };
        writeln!(out, "{}", serde_json::to_string(&reply).unwrap()).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        if reply.error.is_some() {
            return Err("session failed".into());
        }
    }
    Ok(())
}
