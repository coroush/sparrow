//! Initial construction: First Fit Decreasing (FFD) + per-bin LBF/separator placement.
//!
//! For each item (largest first):
//!   1. Try a collision-free LBF placement in each open bin (fast path)
//!   2. If that fails, place in the most-available bin and run the separator
//!      (handles cases where the item only fits with rotation / tight packing)
//!   3. Only open a new bin if both approaches fail for all existing bins

use crate::bp_optimizer::bp_separator::BinSeparator;
use crate::eval::lbf_evaluator::LBFEvaluator;
use crate::eval::sample_eval::SampleEval;
use crate::optimizer::separator::SeparatorConfig;
use crate::quantify::tracker::CollisionTracker;
use crate::sample::search::{search_placement, SampleConfig};
use crate::util::terminator::AlwaysLiveTerminator;
use itertools::Itertools;
use jagua_rs::entities::{Instance, Layout};
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::bpp::entities::{BPInstance, BPLayoutType, BPPlacement, BPProblem, LayKey};
use jagua_rs::Instant;
use log::{debug, info};
use ordered_float::OrderedFloat;
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use std::cmp::Reverse;
use std::iter;

/// Samples for the fast LBF path (collision-free search in existing bins).
const LBF_EXISTING: SampleConfig = SampleConfig {
    n_container_samples: 800,
    n_focussed_samples: 0,
    n_coord_descents: 3,
};

/// Samples when opening a new empty bin (item is alone, almost always finds a spot quickly).
const LBF_NEW_BIN: SampleConfig = SampleConfig {
    n_container_samples: 200,
    n_focussed_samples: 0,
    n_coord_descents: 2,
};

/// Separator config used for the fallback pass — fewer strikes than the full config
/// so it stays fast during construction.
const FALLBACK_SEP_STRIKES: usize = 1;

pub struct BpLbfBuilder {
    pub instance: BPInstance,
    pub prob: BPProblem,
    pub rng: Xoshiro256PlusPlus,
    pub sep_config: SeparatorConfig,
}

impl BpLbfBuilder {
    pub fn new(
        instance: BPInstance,
        rng: Xoshiro256PlusPlus,
        sep_config: SeparatorConfig,
    ) -> Self {
        let prob = BPProblem::new(instance.clone());
        Self { instance, prob, rng, sep_config }
    }

    pub fn construct(mut self) -> BPProblem {
        let start = Instant::now();
        let n_items = self.instance.items.len();

        let sorted: Vec<usize> = (0..n_items)
            .sorted_by_cached_key(|&id| {
                let item = self.instance.item(id);
                let ch_area = item.shape_cd.as_ref().surrogate().convex_hull_area;
                let diameter = item.shape_cd.as_ref().diameter;
                Reverse(OrderedFloat(ch_area * diameter))
            })
            .flat_map(|id| iter::repeat_n(id, self.instance.item_qty(id)))
            .collect();

        info!("[BP-LBF] placing {} items (FFD order)", sorted.len());

        for item_id in sorted {
            self.place_item(item_id);
        }

        info!(
            "[BP-LBF] initial assignment done: {} bins in {:.2}s",
            self.prob.layouts.len(),
            start.elapsed().as_secs_f32()
        );
        self.prob
    }

    fn place_item(&mut self, item_id: usize) {
        let open_keys: Vec<LayKey> = self.prob.layouts.keys().collect();

        // ── Pass 1: collision-free LBF in each existing bin ───────────────────
        for &lkey in &open_keys {
            let layout_clone = self.prob.layouts[lkey].clone();
            if let Some(d_transf) = self.find_lbf_placement(item_id, &layout_clone, LBF_EXISTING) {
                self.prob.place_item(BPPlacement {
                    layout_id: BPLayoutType::Open(lkey),
                    item_id,
                    d_transf,
                });
                self.separate_bin(lkey);
                debug!("[BP-LBF] item {} → existing bin {:?} (LBF clear)", item_id, lkey);
                return;
            }
        }

        // ── Pass 2: separator fallback in the most-available bin ──────────────
        // Pick the bin with the most free area to give the separator the best chance
        if let Some(best_bin) = self.most_available_bin(&open_keys) {
            let snapshot = self.prob.save();
            // Place at origin as seed position for the separator
            self.prob.place_item(BPPlacement {
                layout_id: BPLayoutType::Open(best_bin),
                item_id,
                d_transf: DTransformation::empty(),
            });
            if self.separate_bin_feasible(best_bin) {
                debug!("[BP-LBF] item {} → existing bin {:?} (sep fallback)", item_id, best_bin);
                return;
            }
            // Separator couldn't fit it — restore and open a new bin
            self.prob.restore(&snapshot);
        }

        // ── Pass 3: open a new bin ────────────────────────────────────────────
        let bin_id = self.cheapest_available_bin_id();
        let (lkey, _) = self.prob.place_item(BPPlacement {
            layout_id: BPLayoutType::Closed { bin_id },
            item_id,
            d_transf: DTransformation::empty(),
        });
        debug!("[BP-LBF] item {} → new bin {:?}", item_id, lkey);
        self.separate_bin(lkey);
    }

    fn find_lbf_placement(
        &mut self,
        item_id: usize,
        layout: &Layout,
        config: SampleConfig,
    ) -> Option<DTransformation> {
        let item = self.instance.item(item_id);
        let evaluator = LBFEvaluator::new(layout, item);
        let (best, _) = search_placement(layout, item, None, evaluator, config, &mut self.rng);
        match best {
            Some((dt, SampleEval::Clear { .. })) => Some(dt),
            _ => None,
        }
    }

    /// Run the separator and return whether the bin is feasible afterward.
    fn separate_bin_feasible(&mut self, lkey: LayKey) -> bool {
        let layout = self.prob.layouts[lkey].clone();
        // Use reduced strikes for the fallback to keep construction fast
        let mut fast_cfg = self.sep_config;
        fast_cfg.strike_limit = FALLBACK_SEP_STRIKES;

        let mut sep = BinSeparator::new(
            self.instance.clone(),
            layout,
            Xoshiro256PlusPlus::seed_from_u64(self.rng.random()),
            fast_cfg,
        );
        let (best_snap, _) = sep.separate(&AlwaysLiveTerminator);
        self.prob.layouts[lkey].restore(&best_snap);
        // Re-check loss on the final layout
        CollisionTracker::new(&self.prob.layouts[lkey]).get_total_loss() == 0.0
    }

    /// Full separator pass (used when opening a new bin — item is alone so it's fast).
    fn separate_bin(&mut self, lkey: LayKey) {
        let layout = self.prob.layouts[lkey].clone();
        let mut sep = BinSeparator::new(
            self.instance.clone(),
            layout,
            Xoshiro256PlusPlus::seed_from_u64(self.rng.random()),
            self.sep_config,
        );
        let (best_snap, _) = sep.separate(&AlwaysLiveTerminator);
        self.prob.layouts[lkey].restore(&best_snap);
    }

    /// Return the open bin with the most available (free) area, if any.
    fn most_available_bin(&self, keys: &[LayKey]) -> Option<LayKey> {
        keys.iter().copied().max_by_key(|&lkey| {
            let layout = &self.prob.layouts[lkey];
            let bin_area = layout.container.area();
            let used_area: f32 = layout
                .placed_items
                .values()
                .map(|pi| self.instance.item(pi.item_id).shape_orig.area())
                .sum();
            OrderedFloat(bin_area - used_area)
        })
    }

    fn cheapest_available_bin_id(&self) -> usize {
        self.instance
            .bins
            .iter()
            .enumerate()
            .filter(|(id, _)| self.prob.bin_stock_qtys[*id] > 0)
            .min_by_key(|(_, bin)| bin.cost)
            .map(|(id, _)| id)
            .expect("no bins available")
    }
}
