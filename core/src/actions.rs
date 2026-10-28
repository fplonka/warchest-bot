use crate::board::NONE;
use crate::state::State;
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};

pub const KIND_BITS: u32 = 6;
pub const FIELD_BITS: u32 = 6;
pub const FIELD_NONE: u32 = 63;

#[inline]
fn hex_slot(field: &str) -> usize {
    match field {
        "from" => 0,
        "to" | "hex" => 1,
        "target" => 2,
        _ => 3,
    }
}

macro_rules! actions {
    ($($name:ident $({ $($field:ident),* })?),* $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum Action { $($name $({ $($field: u8),* })?),* }

        #[repr(u32)]
        enum Kind { $($name),* }

        pub const KIND_NAMES: [&str; N_KINDS] = [$(stringify!($name)),*];
        pub const N_KINDS: usize = [$(Kind::$name),*].len();

        impl Action {
            pub fn kind(&self) -> usize {
                match self {
                    $(Action::$name { .. } => Kind::$name as usize),*
                }
            }

            pub fn encode(&self) -> u32 {
                match *self {
                    $(
                        #[allow(unused_mut, unused_assignments, unused_variables)]
                        Action::$name $({ $($field),* })? => {
                        let mut shift = KIND_BITS;
                        let mut code = Kind::$name as u32;
                        $($(
                            code |= (if $field == NONE { FIELD_NONE } else { $field as u32 }) << shift;
                            shift += FIELD_BITS;
                        )*)?
                        code
                    })*
                }
            }

            pub fn decode(code: u32) -> Option<Action> {
                #[allow(unused_variables)]
                let field = |i: u32| -> u8 {
                    let v = (code >> (KIND_BITS + i * FIELD_BITS)) & ((1 << FIELD_BITS) - 1);
                    if v == FIELD_NONE { NONE } else { v as u8 }
                };
                #[allow(unused_mut, unused_variables)]
                let mut at = 0;
                $(
                    if code & ((1 << KIND_BITS) - 1) == Kind::$name as u32 {
                        return Some(Action::$name $({ $($field: { at += 1; field(at - 1) }),* })?);
                    }
                )*
                None
            }

            pub fn hexes(&self) -> [u8; 3] {
                #[allow(unused_mut)]
                let mut out = [NONE; 4];
                match *self {
                    $(Action::$name $({ $($field),* })? => {
                        $($( out[hex_slot(stringify!($field))] = $field; )*)?
                    })*
                }
                [out[0], out[1], out[2]]
            }
        }
    };
}

actions! {
    Deploy { unit, hex },
    Bolster { unit, hex },
    ClaimInitiative { coin },
    Recruit { coin, unit },
    Pass { coin },
    Move { from, to },
    Control { from },
    Attack { from, target },
    TacArcher { from, target },
    TacCavalryMove { from, to },
    TacCavalryAttack { from, target },
    TacCrossbow { from, target },
    TacEnsign { from, to },
    TacLancer { from, to, target },
    TacLightCav { from, to },
    TacMarshal { from, target },
    TacRoyalGuard { from, to },
    FootMove { from, to },
    FootControl { from },
    FootAttack { from, target },
    DrawCoin { unit },
    RGSoakSupply,
    RGSoakStack,
    SwordsmanMove { from, to },
    SwordsmanDecline,
    BerserkMove { from, to },
    BerserkControl { from },
    BerserkAttack { from, target },
    BerserkStop,
    MercMove { from, to },
    MercControl { from },
    MercAttack { from, target },
    MercDecline,
    FootmanInstantDeploy { hex },
    FootmanInstantDecline,
    FootDecline,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum Play {
    Attack,
    Pass,
    Deploy,
    Bolster,
    Maneuver,
    Recruit,
    ClaimInitiative,
    Other,
}

pub const N_PLAYS: usize = Play::Other as usize;

impl Action {
    #[inline]
    pub fn recruited(&self) -> u8 {
        match *self {
            Action::Recruit { unit, .. } => unit,
            _ => NONE,
        }
    }

    pub fn play(self) -> Play {
        use Action::*;
        match self {
            Deploy { .. } => Play::Deploy,
            Bolster { .. } => Play::Bolster,
            Pass { .. } => Play::Pass,
            Recruit { .. } => Play::Recruit,
            ClaimInitiative { .. } => Play::ClaimInitiative,
            Attack { .. }
            | TacArcher { .. }
            | TacCavalryAttack { .. }
            | TacCrossbow { .. }
            | TacLancer { .. }
            | TacMarshal { .. }
            | FootAttack { .. }
            | BerserkAttack { .. }
            | MercAttack { .. } => Play::Attack,
            Move { .. }
            | Control { .. }
            | TacCavalryMove { .. }
            | TacEnsign { .. }
            | TacLightCav { .. }
            | TacRoyalGuard { .. }
            | FootMove { .. }
            | FootControl { .. }
            | BerserkMove { .. }
            | BerserkControl { .. }
            | MercMove { .. }
            | MercControl { .. }
            | SwordsmanMove { .. } => Play::Maneuver,
            _ => Play::Other,
        }
    }
}

pub fn action_coin(a: &Action, s: &State) -> u8 {
    use Action::*;
    match *a {
        Deploy { unit, .. } | Bolster { unit, .. } => unit,
        ClaimInitiative { coin } | Recruit { coin, .. } | Pass { coin } => coin,
        TacRoyalGuard { .. } => ROYAL_COIN,
        TacEnsign { .. } => ENSIGN,
        TacMarshal { .. } => MARSHAL,
        Move { from, .. }
        | Control { from }
        | Attack { from, .. }
        | TacArcher { from, .. }
        | TacCavalryMove { from, .. }
        | TacCrossbow { from, .. }
        | TacLancer { from, .. }
        | TacLightCav { from, .. } => s.hex_type[from as usize],
        _ => NONE,
    }
}

pub fn obs_key(a: &Action) -> u32 {
    let code = a.encode();
    match a {
        Action::Pass { .. } | Action::ClaimInitiative { .. } => code & 63,
        Action::Recruit { unit, .. } => (code & 63) | ((*unit as u32) << 12),
        _ => code,
    }
}
