//! Per-bin separator: runs the collision-separation loop on a single bin Layout.
//!
//! Mirrors `optimizer/separator.rs` but works directly with `jagua_rs::entities::Layout`
//! and `BPInstance` instead of `SPProblem`/`SPInstance`, so the existing strip-packing
//! code is completely untouched.

use crate::eval::sep_evaluator::SeparationEvaluator;
use crate::optimizer::separator::SeparatorConfig;
use crate::quantify::tracker::{CTSnapshot, CollisionTracker};
use crate::sample::search::{search_placement, SampleConfig};
use crate::util::terminator::Terminator;
use crate::FMT;
use itertools::Itertools;
use jagua_rs::entities::{Instance, Layout, LayoutSnapshot, PItemKey};
use jagua_rs::probs::bpp::entities::BPInstance;
use jagua_rs::Instant;
use log::{debug, log};
use ordered_float::OrderedFloat;
use rand::prelude::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand::rngs::Xoshiro256PlusPlus;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use rayon::ThreadPool;
use std::iter::Sum;
use std::ops::AddAssign;
use tap::Tap;

// ── Stats ─────────────────────────────────────────────────────────────────────

pub struct BpSepStats {
    pub total_moves: usize,
    pub total_evals: usize,
}

impl Sum for BpSepStats {
    fn sum<I: Iterator<Item = BpSepStats>>(iter: I) -> Self {
        let mut total_moves = 0;
        let mut total_evals = 0;
        for s in iter {
            total_moves += s.total_moves;
            total_evals += s.total_evals;
        }
        BpSepStats { total_moves, total_evals }
    }
}

impl AddAssign for BpSepStats {
    fn add_assign(&mut self, other: Self) {
        self.total_moves += other.total_moves;
        self.total_evals += other.total_evals;
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

/// A single worker that operates on a cloned Layout for parallel separation.
pub struct BinSepWorker {
    pub instance: BPInstance,
    pub layout: Layout,
    pub ct: CollisionTracker,
    pub rng: Xoshiro256PlusPlus,
    pub sample_config: SampleConfig,
}

impl BinSepWorker {
    pub fn load(&mut self, snap: &LayoutSnapshot, ct: &CollisionTracker) {
        self.layout.restore(snap);
        self.ct = ct.clone();
    }

    /// Move all colliding items to better positions (one pass).
    pub fn move_items(&mut self) -> BpSepStats {
        let candidates: Vec<PItemKey> = self
            .layout
            .placed_items
            .keys()
            .filter(|pk| self.ct.get_loss(*pk) > 0.0)
            .collect_vec()
            .tap_mut(|v| v.shuffle(&mut self.rng));

        let mut total_moves = 0;
        let mut total_evals = 0;

        for &pk in &candidates {
            if self.ct.get_loss(pk) == 0.0 {
                continue;
            }
            let item_id = self.layout.placed_items[pk].item_id;
            let item = self.instance.item(item_id);

            let evaluator = SeparationEvaluator::new(&self.layout, item, pk, &self.ct);
            let (best_sample, n_evals) = search_placement(
                &self.layout,
                item,
                Some(pk),
                evaluator,
                self.sample_config,
                &mut self.rng,
            );
            total_evals += n_evals;

            if let Some((new_dt, _)) = best_sample {
                // re-borrow item after evaluator+search_placement release the borrow
                let item = self.instance.item(item_id);
                self.layout.remove_item(pk);
                let new_pk = self.layout.place_item(item, new_dt);
                self.ct.register_item_move(&self.layout, pk, new_pk);
                total_moves += 1;
            }
        }

        BpSepStats { total_moves, total_evals }
    }
}

// ── Separator ─────────────────────────────────────────────────────────────────

/// Runs the full separation loop on a single bin's Layout.
pub struct BinSeparator {
    pub instance: BPInstance,
    pub layout: Layout,
    pub ct: CollisionTracker,
    pub workers: Vec<BinSepWorker>,
    pub config: SeparatorConfig,
    pub rng: Xoshiro256PlusPlus,
    pub thread_pool: Option<ThreadPool>,
}

impl BinSeparator {
    pub fn new(
        instance: BPInstance,
        layout: Layout,
        mut rng: Xoshiro256PlusPlus,
        config: SeparatorConfig,
    ) -> Self {
        let ct = CollisionTracker::new(&layout);
        let workers = (0..config.n_workers)
            .map(|_| BinSepWorker {
                instance: instance.clone(),
                layout: layout.clone(),
                ct: ct.clone(),
                rng: Xoshiro256PlusPlus::seed_from_u64(rng.random()),
                sample_config: config.sample_config,
            })
            .collect();

        let thread_pool = if cfg!(target_arch = "wasm32") {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.n_workers)
                    .build()
                    .unwrap(),
            )
        };

        Self {
            instance,
            layout,
            ct,
            workers,
            config,
            rng,
            thread_pool,
        }
    }

    /// Run the separation loop. Returns the best (LayoutSnapshot, CTSnapshot) found.
    pub fn separate(&mut self, term: &impl Terminator) -> (LayoutSnapshot, CTSnapshot) {
        let mut min_loss_sol = (self.layout.save(), self.ct.save());
        let mut min_loss = self.ct.get_total_loss();

        log!(
            self.config.log_level,
            "[BIN-SEP] separating bin, initial loss: {}",
            FMT().fmt2(min_loss)
        );

        let mut n_strikes = 0;
        let mut n_iter = 0;
        let start = Instant::now();

        'outer: while n_strikes < self.config.strike_limit && !term.kill() {
            let mut n_iter_no_improvement = 0;
            let initial_strike_loss = self.ct.get_total_loss();

            while n_iter_no_improvement < self.config.iter_no_imprv_limit && !term.kill() {
                let loss_before = self.ct.get_total_loss();
                self.move_items_multi();
                let loss = self.ct.get_total_loss();

                debug!(
                    "[BIN-SEP] [s:{n_strikes},i:{n_iter}] l: {} -> {}, (min l: {})",
                    FMT().fmt2(loss_before),
                    FMT().fmt2(loss),
                    FMT().fmt2(min_loss)
                );

                if loss == 0.0 {
                    log!(
                        self.config.log_level,
                        "[BIN-SEP] [s:{n_strikes},i:{n_iter}] (S) min_l: {}",
                        FMT().fmt2(loss)
                    );
                    min_loss_sol = (self.layout.save(), self.ct.save());
                    min_loss = 0.0;
                    break 'outer;
                } else if loss < min_loss {
                    log!(
                        self.config.log_level,
                        "[BIN-SEP] [s:{n_strikes},i:{n_iter}] (*) min_l: {}",
                        FMT().fmt2(loss)
                    );
                    if loss < min_loss * 0.98 {
                        n_iter_no_improvement = 0;
                    }
                    min_loss_sol = (self.layout.save(), self.ct.save());
                    min_loss = loss;
                } else {
                    n_iter_no_improvement += 1;
                }

                self.ct.update_weights();
                n_iter += 1;
            }

            if initial_strike_loss * 0.98 <= min_loss {
                n_strikes += 1;
            } else {
                n_strikes = 0;
            }
            self.rollback(&min_loss_sol.0, Some(&min_loss_sol.1));
        }

        log!(
            self.config.log_level,
            "[BIN-SEP] finished in {:.3}s, final loss: {}",
            start.elapsed().as_secs_f32(),
            FMT().fmt2(min_loss)
        );

        (min_loss_sol.0, min_loss_sol.1)
    }

    pub fn is_feasible(&self) -> bool {
        self.ct.get_total_loss() == 0.0
    }

    pub fn total_loss(&self) -> f32 {
        self.ct.get_total_loss()
    }

    fn move_items_multi(&mut self) {
        let master_snap = self.layout.save();

        let mut run_parallel = || {
            self.workers.par_iter_mut().map(|worker| {
                worker.load(&master_snap, &self.ct);
                worker.move_items()
            }).sum::<BpSepStats>()
        };

        match self.thread_pool.as_mut() {
            Some(pool) => { pool.install(&mut run_parallel); }
            None => { run_parallel(); }
        }

        // Pick the worker with lowest weighted loss
        let (best_snap, best_ct) = self
            .workers
            .iter_mut()
            .min_by_key(|w| OrderedFloat(w.ct.get_total_weighted_loss()))
            .map(|w| (w.layout.save(), w.ct.clone()))
            .unwrap();

        self.layout.restore(&best_snap);
        self.ct = best_ct;
    }

    pub fn rollback(&mut self, snap: &LayoutSnapshot, cts: Option<&CTSnapshot>) {
        self.layout.restore(snap);
        match cts {
            Some(cts) => self.ct.restore_but_keep_weights(cts, &self.layout),
            None => self.ct = CollisionTracker::new(&self.layout),
        }
    }
}
