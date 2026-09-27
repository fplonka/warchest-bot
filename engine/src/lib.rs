pub mod actions;
pub mod arena;
pub mod board;
pub mod bot;
#[cfg(feature = "gpu")]
pub mod cuda;
pub mod memo;
pub mod net;
pub mod online;
pub mod packed;
pub mod pbs;
pub mod policy;
pub mod rng;
pub mod rules;
pub mod search;
pub mod selfplay;
pub mod state;
pub mod units;

pub const PLAY_LIMIT: u16 = 256;

pub fn finished(s: &State) -> bool {
    s.is_terminal() || s.plays >= PLAY_LIMIT
}

pub fn score(s: &State, player: usize) -> f32 {
    if s.is_terminal() { s.utility(player) } else { 0.0 }
}

pub use actions::Action;
pub use state::{State, BLACK, WHITE};

#[cfg(feature = "python")]
mod py;
