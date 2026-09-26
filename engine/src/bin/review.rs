use std::path::Path;

use rayon::prelude::*;
use serde::Serialize;
use warchest::arena::{Obs, Start};
use warchest::bot::{Brain, Session, Verdict};
use warchest::packed::Packed;

#[derive(Serialize)]
struct Review<'a> {
    journal: &'a str,
    seat: u8,
    verdicts: Vec<Verdict>,
}

fn review(journal: &str, brain: &Brain) -> Result<(u8, Vec<Verdict>), String> {
    let text = std::fs::read_to_string(journal).map_err(|e| format!("{journal}: {e}"))?;
    let mut lines = text.lines();
    let start: Start = serde_json::from_str(lines.next().ok_or("the journal is empty")?).map_err(|e| e.to_string())?;
    let mut session = Session::new(&start.draft, start.seat, start.seed)?;
    let mut verdicts = Vec::new();
    for line in lines {
        let obs: Obs = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if let Obs::Act { player, key } = obs {
            session.watch(brain);
            verdicts.push(session.judge(player, key)?);
        }
        session.observe(&obs, brain)?;
    }
    Ok((start.seat, verdicts))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [bot, slots, journals @ ..] = args.as_slice() else {
        eprintln!("usage: review BOT_DIR SLOTS JOURNAL...");
        std::process::exit(2);
    };
    let brain = Packed::load(Path::new(bot))
        .and_then(|packed| Brain::load(&packed, &[0], slots.parse().map_err(|_| format!("invalid slots {slots}"))?))
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        });
    journals.par_iter().for_each(|journal| match review(journal, &brain) {
        Ok((seat, verdicts)) => println!("{}", serde_json::to_string(&Review { journal, seat, verdicts }).unwrap()),
        Err(e) => eprintln!("{journal}: {e}"),
    });
}
