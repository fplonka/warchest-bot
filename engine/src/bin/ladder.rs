use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde::Serialize;
use warchest::arena::{Done, Draft, Hello, Reply, Request, Table, PROTOCOL};
use warchest::packed::Packed;
use warchest::pbs::rules_table_hash;
use warchest::rng::Rng;
use warchest::selfplay::deal;

struct BotProcess {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl BotProcess {
    fn launch(bot: &Packed, devices: &str) -> Result<BotProcess, String> {
        let m = &bot.manifest;
        let mut command = Command::new(bot.dir.join("bot"));
        command
            .arg(&bot.dir)
            .arg("--devices")
            .arg(devices)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command.spawn().map_err(|e| format!("{}: {e}", m.name))?;
        let input = child.stdin.take().ok_or("bot stdin")?;
        let mut output = BufReader::new(child.stdout.take().ok_or("bot stdout")?);
        let mut line = String::new();
        output.read_line(&mut line).map_err(|e| e.to_string())?;
        let hello: Hello = serde_json::from_str(&line)
            .map_err(|e| format!("{} sent an invalid greeting: {e}", m.name))?;
        if hello.name != m.name {
            return Err(format!("{} identified as {}", m.name, hello.name));
        }
        if hello.protocol != PROTOCOL {
            return Err(format!(
                "{} speaks protocol {}, not {}",
                m.name, hello.protocol, PROTOCOL
            ));
        }
        if hello.rules != rules_table_hash() {
            return Err(format!("{} was built against different rules", m.name));
        }
        Ok(BotProcess {
            child,
            input,
            output,
        })
    }

    fn send(&mut self, request: &Request) -> Result<(), String> {
        serde_json::to_writer(&mut self.input, request).map_err(|e| e.to_string())?;
        writeln!(self.input).map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())
    }

    fn receive(&mut self, expected: usize) -> Result<Vec<Done>, String> {
        let mut done = Vec::with_capacity(expected);
        while done.len() < expected {
            let mut line = String::new();
            self.output
                .read_line(&mut line)
                .map_err(|e| e.to_string())?;
            if line.is_empty() {
                return Err("bot exited before replying".into());
            }
            let reply: Reply =
                serde_json::from_str(&line).map_err(|e| format!("invalid bot reply: {e}"))?;
            if let Some(error) = reply.error {
                return Err(error);
            }
            done.extend(reply.done);
        }
        Ok(done)
    }
}

impl Drop for BotProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Serialize)]
struct Match {
    a: String,
    b: String,
    a_minutes: f64,
    b_minutes: f64,
    a_sha: String,
    b_sha: String,
    draft_seed: u64,
    protocol: u32,
    rules: u64,
    outcomes: Vec<[i8; 2]>,
}

fn drafts(seed: u64, count: usize) -> Vec<Draft> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let pool = deal(&mut rng);
            Draft {
                white: pool[..4].to_vec(),
                black: pool[4..8].to_vec(),
                first: (rng.next_u64() & 1) as u8,
            }
        })
        .collect()
}

const CONCURRENT: usize = 256;

fn play(a: &Packed, b: &Packed, games: usize, seed: u64) -> Result<Vec<[i8; 2]>, String> {
    let mut bots = [BotProcess::launch(a, "0")?, BotProcess::launch(b, "1")?];
    let mut table = Table::new();
    let mut outcomes = vec![[0; 2]; games / 2];
    for (pair, draft) in drafts(seed, games / 2).into_iter().enumerate() {
        for (color, seats) in [[0, 1], [1, 0]].into_iter().enumerate() {
            let id = (2 * pair + color) as u32;
            table.start(id, &draft, seats, seed ^ id as u64)?;
        }
        while table.live() >= CONCURRENT {
            step(&mut table, &mut bots, &mut outcomes)?;
        }
    }
    while table.live() > 0 {
        step(&mut table, &mut bots, &mut outcomes)?;
    }
    Ok(outcomes)
}

fn step(
    table: &mut Table,
    bots: &mut [BotProcess; 2],
    outcomes: &mut [[i8; 2]],
) -> Result<(), String> {
    table.settle();
    for (id, seats, utility) in table.reap() {
        outcomes[id as usize / 2][id as usize % 2] = if seats[0] == 0 { utility } else { -utility };
    }
    let requests = [table.request(0), table.request(1)];
    for (bot, request) in bots.iter_mut().zip(&requests) {
        bot.send(request)?;
    }
    for (index, (bot, request)) in bots.iter_mut().zip(&requests).enumerate() {
        table.accept(index, bot.receive(request.go.len() + request.watch.len())?)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [a, b, games, seed] = args.as_slice() else {
        return Err("usage: ladder <bot-dir> <bot-dir> <games> <draft-seed>".into());
    };
    let a = Packed::load(Path::new(a))?;
    let b = Packed::load(Path::new(b))?;
    let games: usize = games.parse().map_err(|e| format!("games: {e}"))?;
    let seed: u64 = seed.parse().map_err(|e| format!("draft-seed: {e}"))?;
    if games < 2 || games % 2 != 0 {
        return Err("games must be even and at least two".into());
    }
    let outcomes = play(&a, &b, games, seed)?;
    let record = Match {
        a: a.manifest.name,
        b: b.manifest.name,
        a_minutes: a.manifest.minutes,
        b_minutes: b.manifest.minutes,
        a_sha: a.manifest.weights_sha,
        b_sha: b.manifest.weights_sha,
        draft_seed: seed,
        protocol: PROTOCOL,
        rules: rules_table_hash(),
        outcomes,
    };
    println!("{}", serde_json::to_string(&record).map_err(|e| e.to_string())?);
    Ok(())
}
