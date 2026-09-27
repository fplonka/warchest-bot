use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::actions::Action;
use crate::rng::Rng;
use crate::selfplay::resolve_chance;
use crate::state::{Cont, State, BLACK, WHITE};
use crate::units::index_of_id;

pub const PROTOCOL: u32 = 7;

#[derive(Serialize, Deserialize, Debug)]
pub struct Hello {
    pub name: String,
    pub protocol: u32,
    pub rules: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Draft {
    pub white: Vec<u16>,
    pub black: Vec<u16>,
    pub first: u8,
}

impl Draft {
    pub fn state(&self) -> Result<State, String> {
        for (side, ids) in [("white", &self.white), ("black", &self.black)] {
            if ids.len() != 4 {
                return Err(format!("{} drafts 4 units, not {}", side, ids.len()));
            }
            if let Some(&id) = ids.iter().find(|&&id| index_of_id(id).is_none()) {
                return Err(format!("unitTypeId {} is out of scope", id));
            }
        }
        if self.first != WHITE && self.first != BLACK {
            return Err(format!("first must be {} or {}", WHITE, BLACK));
        }
        Ok(State::from_draft(&self.white, &self.black, self.first))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Obs {
    Draw {
        player: u8,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<u32>,
    },
    Act {
        player: u8,
        key: u32,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Start {
    pub draft: Draft,
    pub seat: u8,
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Ask {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<Start>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obs: Vec<Obs>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Request {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub go: Vec<Ask>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Done {
    pub id: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Reply {
    #[serde(default)]
    pub done: Vec<Done>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

struct Bout {
    s: State,
    rng: Rng,
    bots: [usize; 2],
    pending: [Vec<Obs>; 2],
    start: [Option<Start>; 2],
    asked: [bool; 2],
    over: bool,
    main_plays: u16,
}

#[derive(Default)]
pub struct Table {
    bouts: BTreeMap<u32, Bout>,
    done: Vec<(u32, [usize; 2], i8)>,
    dropped: BTreeMap<usize, Vec<u32>>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    pub fn live(&self) -> usize {
        self.bouts.values().filter(|b| !b.over).count()
    }

    pub fn start(
        &mut self,
        id: u32,
        draft: &Draft,
        bots: [usize; 2],
        seed: u64,
    ) -> Result<(), String> {
        let s = draft.state()?;
        let start = |seat: u8| {
            Some(Start {
                draft: draft.clone(),
                seat,
                seed: seed ^ (seat as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            })
        };
        self.bouts.insert(
            id,
            Bout {
                s,
                rng: Rng::new(seed),
                bots,
                pending: [Vec::new(), Vec::new()],
                start: [start(WHITE), start(BLACK)],
                asked: [false, false],
                over: false,
                main_plays: 0,
            },
        );
        Ok(())
    }

    pub fn settle(&mut self) {
        let mut ended = Vec::new();
        for (&id, b) in self.bouts.iter_mut() {
            if b.over {
                continue;
            }
            while !b.s.is_terminal() && b.s.is_chance() {
                let player = b.s.to_act();
                let drawn = resolve_chance(&mut b.s, player, &mut b.rng);
                b.pending[player as usize].push(Obs::Draw {
                    player,
                    code: Some(drawn.encode()),
                });
                b.pending[1 - player as usize].push(Obs::Draw { player, code: None });
            }
            if crate::finished(&b.s, b.main_plays) {
                b.over = true;
                let z = b.s.winner().map_or(0, |w| if w == WHITE { 1 } else { -1 });
                ended.push((id, b.bots, z));
            }
        }
        self.done.append(&mut ended);
        self.retire();
    }

    fn retire(&mut self) {
        let done: Vec<u32> = self
            .bouts
            .iter()
            .filter(|(_, b)| b.over && !b.asked[0] && !b.asked[1])
            .map(|(&id, _)| id)
            .collect();
        for id in done {
            let bout = self.bouts.remove(&id).expect("just listed");
            for bot in bout.bots {
                self.dropped.entry(bot).or_default().push(id);
            }
        }
    }

    pub fn play(&mut self, id: u32, code: u32) -> Result<(), String> {
        let b = self
            .bouts
            .get_mut(&id)
            .ok_or_else(|| format!("game {} is not live", id))?;
        let action =
            Action::decode(code).ok_or_else(|| format!("action {} does not decode", code))?;
        if !b.s.legal_actions().iter().any(|x| x.encode() == code) {
            return Err(format!("game {id}: illegal action {action:?}"));
        }
        let player = b.s.to_act();
        b.main_plays += matches!(b.s.pending(), Cont::MainPlay) as u16;
        b.s.apply_inplace(action);
        b.pending[1 - player as usize].push(Obs::Act {
            player,
            key: crate::pbs::obs_key(&action),
        });
        Ok(())
    }

    pub fn reap(&mut self) -> Vec<(u32, [usize; 2], i8)> {
        std::mem::take(&mut self.done)
    }

    pub fn request(&mut self, bot: usize) -> Request {
        let mut go = Vec::new();
        for (&id, b) in self.bouts.iter_mut() {
            if crate::finished(&b.s, b.main_plays) || b.s.is_chance() {
                continue;
            }
            let Some(seat) = b.bots.iter().position(|&x| x == bot) else {
                continue;
            };
            if b.asked[seat] || seat != b.s.to_act() as usize {
                continue;
            }
            b.asked[seat] = true;
            go.push(Ask {
                id,
                start: b.start[seat].take(),
                obs: std::mem::take(&mut b.pending[seat]),
            });
        }
        Request {
            go,
            drop: self.dropped.remove(&bot).unwrap_or_default(),
        }
    }

    pub fn accept(&mut self, bot: usize, done: Vec<Done>) -> Result<(), String> {
        for done in done {
            if let Some(b) = self.bouts.get_mut(&done.id) {
                if let Some(seat) = b.bots.iter().position(|&x| x == bot) {
                    b.asked[seat] = false;
                }
            }
            if let Some(action) = done.action {
                self.play(done.id, action)?;
            }
        }
        self.retire();
        Ok(())
    }
}
