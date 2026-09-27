use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::search::{Answer, Keyed, Query};

const SHARDS: usize = 64;
const GENERATION: usize = 1 << 12;

type Seen = HashMap<u64, Arc<[f32]>, BuildHasherDefault<Keyed>>;

#[derive(Default)]
struct Shard {
    now: Seen,
    old: Seen,
}

pub struct Memo {
    shards: Vec<Mutex<Shard>>,
    epoch: AtomicU64,
}

impl Default for Memo {
    fn default() -> Memo {
        Memo { shards: (0..SHARDS).map(|_| Mutex::default()).collect(), epoch: AtomicU64::new(0) }
    }
}

impl Memo {
    fn shard(&self, key: u64) -> parking_lot::MutexGuard<'_, Shard> {
        self.shards[key as usize % SHARDS].lock()
    }

    pub fn forget(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        for s in &self.shards {
            *s.lock() = Shard::default();
        }
    }

    fn put(&self, key: u64, entry: Arc<[f32]>) {
        let mut s = self.shard(key);
        if s.now.len() >= GENERATION {
            s.old = std::mem::take(&mut s.now);
        }
        s.now.insert(key, entry);
    }

    pub fn eval(&self, q: Query, compute: impl FnOnce(Query) -> Answer) -> Answer {
        let epoch = self.epoch.load(Ordering::SeqCst);
        let seen: Vec<(u64, Option<Arc<[f32]>>)> = (0..q.len())
            .map(|i| {
                let mut h = DefaultHasher::new();
                q.row(i).hash(&mut h);
                let key = h.finish();
                let s = self.shard(key);
                (key, s.now.get(&key).or_else(|| s.old.get(&key)).cloned())
            })
            .collect();
        let mut miss = Query::default();
        for (i, _) in seen.iter().enumerate().filter(|(_, (_, hit))| hit.is_none()) {
            miss.push_row(q.row(i));
        }
        let fresh = if miss.is_empty() { Answer::default() } else { compute(miss) };
        let keep = self.epoch.load(Ordering::SeqCst) == epoch;
        let (mut out, mut r, mut a) = (Answer::default(), 0, 0);
        for (i, (key, hit)) in seen.into_iter().enumerate() {
            let entry = hit.unwrap_or_else(|| {
                let n = q.actions(i).len();
                let e: Arc<[f32]> = std::iter::once(fresh.value[r]).chain(fresh.prior[a..a + n].iter().copied()).collect();
                (r, a) = (r + 1, a + n);
                if keep {
                    self.put(key, e.clone());
                }
                e
            });
            out.value.push(entry[0]);
            out.prior.extend_from_slice(&entry[1..]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbs::Ctx;
    use crate::rng::Rng;
    use crate::search::uniform;

    fn positions() -> Query {
        let mut rng = Rng::new(9);
        let mut s = crate::selfplay::game(&mut rng);
        let ctx = Ctx::new(&s);
        let mut q = Query::default();
        while q.len() < 40 && !s.is_terminal() {
            let acts = s.legal_actions();
            if !s.is_chance() {
                q.push(&s, &ctx, &acts);
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
        q
    }

    #[test]
    fn a_memo_answers_seen_rows_itself_until_it_forgets() {
        let (memo, q, seen) = (Memo::default(), positions(), std::cell::Cell::new(0));
        let compute = |q: Query| {
            seen.set(seen.get() + q.len());
            Answer { value: (0..q.len()).map(|i| i as f32).collect(), prior: uniform(&q) }
        };
        let first = memo.eval(q.clone(), compute);
        let again = memo.eval(q.clone(), compute);
        assert_eq!(seen.get(), q.len());
        assert_eq!((first.value, first.prior), (again.value, again.prior));
        memo.forget();
        memo.eval(q.clone(), compute);
        assert_eq!(seen.get(), 2 * q.len());
    }
}
