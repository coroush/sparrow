//! Bin-reduction phase: repeatedly try to eliminate the least-loaded bin.

use crate::bp_optimizer::bp_moves::{resolve_by_transfers, separate_single_bin};
use crate::config::BinPackConfig;
use crate::eval::lbf_evaluator::LBFEvaluator;
use crate::eval::sample_eval::SampleEval;
use crate::optimizer::separator::SeparatorConfig;
use crate::sample::search::{search_placement, SampleConfig};
use crate::util::terminator::Terminator;
use jagua_rs::entities::{Instance, Layout, PItemKey};
use jagua_rs::geometry::DTransformation;
use jagua_rs::probs::bpp::entities::{BPInstance, BPLayoutType, BPPlacement, BPProblem, BPSolution, LayKey};
use log::info;
use ordered_float::OrderedFloat;
use rand::rngs::Xoshiro256PlusPlus;
use rand::seq::IteratorRandom;
use rand::{Rng, RngExt, SeedableRng};
use std::collections::HashSet;

/// LBF sample budget for redistributing displaced items and per-bin compaction.
const LBF_REDISTRIBUTE: SampleConfig = SampleConfig {
    n_container_samples: 400,
    n_focussed_samples: 0,
    n_coord_descents: 2,
};

/// Solution-pool / perturbation tuning (AC-3: sparrow exploration analog).
const POOL_MAX: usize = 5;
const PERTURB_AFTER_FAILURES: usize = 1;
const PERTURB_PROB: f32 = 0.7;
/// Max consecutive last-resort perturbations that fail to produce a reduction before we give up.
const MAX_LAST_RESORT_PERTURBS: usize = 8;
/// Give up entirely after this many consecutive failed reduction attempts (any kind).
/// Prevents burning the full time budget when the algorithm has clearly plateaued.
const MAX_CONSECUTIVE_FAILURES: usize = 15;

pub fn bin_reduction_phase(
    instance: BPInstance,
    mut prob: BPProblem,
    mut rng: Xoshiro256PlusPlus,
    term: &impl Terminator,
    config: &BinPackConfig,
) -> BPProblem {
    let bin_area = instance.bins[0].container.outer_cd.bbox.area();
    let total_item_area: f32 = instance
        .items
        .iter()
        .map(|(item, qty)| item.shape_orig.area() * *qty as f32)
        .sum();
    let lower_bound = (total_item_area / bin_area).ceil() as usize;

    let mut incumbent = prob.save();
    let mut n_bins = incumbent.layout_snapshots.len();

    // QW-3: track bins that have been tried and failed since the last successful reduction.
    // Reset on success so newly-rearranged bins get a fresh chance.
    let mut failed_bins: HashSet<LayKey> = HashSet::new();

    // AC-3: solution pool of alternative incumbents at the current best bin count.
    // Index 0 is always the primary incumbent. Extra entries come from large-item-swap
    // perturbations applied when the current incumbent has stalled.
    let mut pool: Vec<BPSolution> = vec![incumbent.clone()];
    let mut consecutive_failures: usize = 0;
    let mut last_resort_count: usize = 0;

    info!(
        "[BP-EXPL] starting bin reduction: {} bins, lower bound: {}",
        n_bins, lower_bound
    );

    while n_bins > lower_bound && !term.kill() {
        // Early termination: if perturbation + retries can't produce any reduction
        // after this many consecutive failures, stop — longer runs won't help.
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
            info!(
                "[BP-EXPL] early exit: {} consecutive failures, no further reduction found",
                consecutive_failures
            );
            break;
        }

        // Decide starting state for this attempt. After repeated failures, occasionally
        // restart from a perturbed pool member instead of the incumbent — this mirrors
        // sparrow's exploration-phase DSRP (Algorithm 3 of arXiv:2509.13329).
        let try_perturb = consecutive_failures >= PERTURB_AFTER_FAILURES
            && pool.len() > 0
            && rng.random_ratio(
                (PERTURB_PROB * 1000.0) as u32,
                1000,
            );

        if try_perturb {
            let idx = rng.random_range(0..pool.len());
            prob.restore(&pool[idx]);
            if perturb_swap_between_bins(&mut prob, &instance, &config.separator_config, &mut rng, term) {
                failed_bins.clear();
                let snap = prob.save();
                if snap.layout_snapshots.len() == n_bins {
                    if pool.len() < POOL_MAX {
                        pool.push(snap);
                    } else {
                        // Replace a non-incumbent pool entry.
                        let replace_at = 1 + rng.random_range(0..(pool.len() - 1));
                        pool[replace_at] = snap;
                    }
                }
                info!(
                    "[BP-EXPL] perturbed from pool[{}] (pool size {}, failures {})",
                    idx, pool.len(), consecutive_failures
                );
            } else {
                prob.restore(&incumbent);
            }
        } else {
            prob.restore(&incumbent);
        }

        // QW-3: skip previously-failed candidates until a success resets the set.
        let bin_to_remove = match select_candidate_bin(&prob, &instance, &failed_bins) {
            Some(b) => b,
            None => {
                // All candidates exhausted. Force a perturbation attempt from a pool member
                // before giving up — this is the whole point of the pool and uses the
                // remaining time budget productively.
                if last_resort_count >= MAX_LAST_RESORT_PERTURBS || term.kill() {
                    break;
                }
                last_resort_count += 1;
                let idx = rng.random_range(0..pool.len());
                prob.restore(&pool[idx]);
                if perturb_swap_between_bins(
                    &mut prob, &instance, &config.separator_config, &mut rng, term,
                ) {
                    info!(
                        "[BP-EXPL] last-resort perturbation #{} from pool[{}]",
                        last_resort_count, idx
                    );
                    failed_bins.clear();
                    let snap = prob.save();
                    if snap.layout_snapshots.len() == n_bins && pool.len() < POOL_MAX {
                        pool.push(snap);
                    }
                    consecutive_failures += 1;
                    continue;
                }
                prob.restore(&incumbent);
                continue;
            }
        };

        info!("[BP-EXPL] attempting to eliminate bin {:?}", bin_to_remove);

        // Collect displaced items sorted largest-first for better redistribution.
        let mut displaced: Vec<usize> = prob.layouts[bin_to_remove]
            .placed_items
            .values()
            .map(|pi| pi.item_id)
            .collect();
        displaced.sort_by(|&a, &b| {
            let a_area = instance.item(a).shape_orig.area();
            let b_area = instance.item(b).shape_orig.area();
            b_area.partial_cmp(&a_area).unwrap()
        });

        prob.remove_layout(bin_to_remove);
        let remaining_keys: Vec<LayKey> = prob.layouts.keys().collect();

        if remaining_keys.is_empty() {
            break;
        }

        // QW-1: place each displaced item via LBF rather than at its old coordinates.
        // QW-2: track which bins actually receive new items.
        let mut receiving: HashSet<LayKey> = HashSet::new();

        for &item_id in &displaced {
            if let Some(lkey) = try_lbf_into_any_bin(&mut prob, &instance, item_id, &remaining_keys, &mut rng) {
                receiving.insert(lkey);
            } else {
                // Fallback: most-available bin, origin seed for the separator.
                let target = most_available_bin(&prob, &remaining_keys, &instance)
                    .unwrap_or(remaining_keys[0]);
                prob.place_item(BPPlacement {
                    layout_id: BPLayoutType::Open(target),
                    item_id,
                    d_transf: DTransformation::empty(),
                });
                receiving.insert(target);
            }
        }

        // QW-2: only separate bins that received items; untouched bins are already feasible.
        let receiving_keys: Vec<LayKey> = receiving.into_iter().collect();
        let mut all_feasible = true;

        for &lkey in &receiving_keys {
            if term.kill() {
                break;
            }
            let feasible = separate_single_bin(
                &mut prob,
                lkey,
                &config.separator_config,
                &mut Xoshiro256PlusPlus::seed_from_u64(rng.random()),
                term,
            );
            if !feasible {
                all_feasible = false;
            }
        }

        if all_feasible {
            // QW-4: compact all bins via LBF reinsertion before saving the incumbent.
            // Snapshot first so we can revert if compaction leaves anything infeasible.
            let pre_compact = prob.save();
            let all_keys: Vec<LayKey> = prob.layouts.keys().collect();
            for &lkey in &all_keys {
                compact_bin(&mut prob, &instance, lkey, &mut rng, term);
            }
            // If compaction introduced any infeasibility, revert to the pre-compact state.
            let compact_ok = all_keys.iter().all(|&lkey| {
                !prob.layouts.contains_key(lkey)
                    || crate::quantify::tracker::CollisionTracker::new(&prob.layouts[lkey])
                        .get_total_loss()
                        == 0.0
            });
            if !compact_ok {
                prob.restore(&pre_compact);
            }

            incumbent = prob.save();
            n_bins -= 1;
            failed_bins.clear();
            consecutive_failures = 0;
            pool = vec![incumbent.clone()];
            info!(
                "[BP-EXPL] reduced to {} bins (density {:.3})",
                n_bins,
                incumbent.density(&instance)
            );
        } else if !term.kill() {
            info!("[BP-EXPL] direct redistribution infeasible, trying inter-bin moves...");

            let all_bins: Vec<LayKey> = prob.layouts.keys().collect();
            let infeasible_bins: Vec<LayKey> = receiving_keys
                .iter()
                .copied()
                .filter(|&lkey| {
                    if !prob.layouts.contains_key(lkey) {
                        return false;
                    }
                    crate::quantify::tracker::CollisionTracker::new(&prob.layouts[lkey])
                        .get_total_loss()
                        > 0.0
                })
                .collect();

            let resolved = resolve_by_transfers(
                &mut prob,
                &infeasible_bins,
                &all_bins,
                &config.separator_config,
                &mut rng,
                config.inter_bin_move_budget,
                term,
            );

            if resolved {
                let pre_compact = prob.save();
                for &lkey in &all_bins {
                    compact_bin(&mut prob, &instance, lkey, &mut rng, term);
                }
                let compact_ok = all_bins.iter().all(|&lkey| {
                    !prob.layouts.contains_key(lkey)
                        || crate::quantify::tracker::CollisionTracker::new(&prob.layouts[lkey])
                            .get_total_loss()
                            == 0.0
                });
                if !compact_ok {
                    prob.restore(&pre_compact);
                }

                incumbent = prob.save();
                n_bins -= 1;
                failed_bins.clear();
                consecutive_failures = 0;
                pool = vec![incumbent.clone()];
                info!(
                    "[BP-EXPL] reduced to {} bins via inter-bin moves (density {:.3})",
                    n_bins,
                    incumbent.density(&instance)
                );
            } else {
                failed_bins.insert(bin_to_remove);
                consecutive_failures += 1;
                info!(
                    "[BP-EXPL] bin {:?} failed; {} candidates remaining",
                    bin_to_remove,
                    prob.layouts.keys().filter(|k| !failed_bins.contains(k)).count()
                );
            }
        }
    }

    prob.restore(&incumbent);
    info!(
        "[BP-EXPL] bin reduction complete: {} bins (lower bound was {})",
        n_bins, lower_bound
    );
    prob
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// QW-1: Try collision-free LBF placement of `item_id` into any of `bins` (first-fit).
/// Returns the key of the bin where the item was placed, or `None` if all bins failed.
fn try_lbf_into_any_bin(
    prob: &mut BPProblem,
    instance: &BPInstance,
    item_id: usize,
    bins: &[LayKey],
    rng: &mut Xoshiro256PlusPlus,
) -> Option<LayKey> {
    let item = instance.item(item_id);
    for &lkey in bins {
        let layout = prob.layouts[lkey].clone();
        let evaluator = LBFEvaluator::new(&layout, item);
        let (best, _) = search_placement(&layout, item, None, evaluator, LBF_REDISTRIBUTE, rng);
        if let Some((dt, SampleEval::Clear { .. })) = best {
            prob.place_item(BPPlacement {
                layout_id: BPLayoutType::Open(lkey),
                item_id,
                d_transf: dt,
            });
            return Some(lkey);
        }
    }
    None
}

/// QW-4: Compact a single bin by reinserting items one-at-a-time in largest-first order,
/// moving each to its LBF-optimal position. Skips the last item to prevent the bin from
/// disappearing when emptied. Aborts early if the time budget expires.
fn compact_bin(
    prob: &mut BPProblem,
    instance: &BPInstance,
    lkey: LayKey,
    rng: &mut Xoshiro256PlusPlus,
    term: &impl Terminator,
) {
    if !prob.layouts.contains_key(lkey) {
        return;
    }

    let mut items: Vec<usize> = prob.layouts[lkey]
        .placed_items
        .values()
        .map(|pi| pi.item_id)
        .collect();
    // Largest-first so big items claim compact positions before small ones fill gaps.
    items.sort_by(|&a, &b| {
        instance.item(b).shape_orig.area()
            .partial_cmp(&instance.item(a).shape_orig.area())
            .unwrap()
    });

    for &item_id in &items {
        if term.kill() {
            break;
        }
        // Never remove the last item — it would destroy the bin's LayKey.
        if prob.layouts[lkey].placed_items.len() <= 1 {
            break;
        }

        let Some(pik) = prob.layouts[lkey]
            .placed_items
            .iter()
            .find(|(_, pi)| pi.item_id == item_id)
            .map(|(k, _)| k)
        else {
            continue;
        };

        let orig_d_transf = prob.layouts[lkey].placed_items[pik].d_transf;
        prob.remove_item(lkey, pik);

        let layout = prob.layouts[lkey].clone();
        let item = instance.item(item_id);
        let evaluator = LBFEvaluator::new(&layout, item);
        let (best, _) = search_placement(&layout, item, None, evaluator, LBF_REDISTRIBUTE, rng);

        let d_transf = match best {
            Some((dt, SampleEval::Clear { .. })) => dt,
            // No clear position found — keep original to avoid worsening things.
            _ => orig_d_transf,
        };
        prob.place_item(BPPlacement {
            layout_id: BPLayoutType::Open(lkey),
            item_id,
            d_transf,
        });
    }
}

/// Return the bin with the most free area (for fallback placement).
fn most_available_bin(
    prob: &BPProblem,
    keys: &[LayKey],
    instance: &BPInstance,
) -> Option<LayKey> {
    keys.iter().copied().max_by_key(|&lkey| {
        let layout = &prob.layouts[lkey];
        let used: f32 = layout
            .placed_items
            .values()
            .map(|pi| instance.item(pi.item_id).shape_orig.area())
            .sum();
        OrderedFloat(layout.container.area() - used)
    })
}

/// QW-3: Select the lowest-utilization bin, skipping any that have already failed
/// since the last successful reduction.
fn select_candidate_bin(
    prob: &BPProblem,
    instance: &BPInstance,
    failed: &HashSet<LayKey>,
) -> Option<LayKey> {
    prob.layouts
        .iter()
        .filter(|(k, _)| !failed.contains(k))
        .min_by(|(_, la), (_, lb)| {
            let ua = la.placed_item_area(instance) / la.container.area();
            let ub = lb.placed_item_area(instance) / lb.container.area();
            ua.partial_cmp(&ub).unwrap()
        })
        .map(|(k, _)| k)
}

/// AC-3: Perturb the current state by swapping a large item between two different bins,
/// then re-separating both bins. Returns `true` if both bins are feasible afterward;
/// otherwise restores the pre-perturbation state and returns `false`.
///
/// Only considers bins with ≥2 items (so `remove_item` never closes a bin mid-swap).
/// Items are chosen from the top half by area to make the perturbation meaningful —
/// swapping tiny items rarely changes the global structure.
fn perturb_swap_between_bins(
    prob: &mut BPProblem,
    instance: &BPInstance,
    sep_config: &SeparatorConfig,
    rng: &mut Xoshiro256PlusPlus,
    term: &impl Terminator,
) -> bool {
    let eligible: Vec<LayKey> = prob
        .layouts
        .iter()
        .filter(|(_, l)| l.placed_items.len() >= 2)
        .map(|(k, _)| k)
        .collect();
    if eligible.len() < 2 {
        return false;
    }

    // Pick 2 distinct bins.
    let mut pair_iter = eligible.iter().copied().sample(rng, 2);
    if pair_iter.len() < 2 {
        return false;
    }
    let bin_a = pair_iter.remove(0);
    let bin_b = pair_iter.remove(0);

    let Some(pik_a) = pick_large_item_pk(&prob.layouts[bin_a], instance, rng) else {
        return false;
    };
    let Some(pik_b) = pick_large_item_pk(&prob.layouts[bin_b], instance, rng) else {
        return false;
    };

    let snapshot = prob.save();

    let placement_a = prob.remove_item(bin_a, pik_a);
    let placement_b = prob.remove_item(bin_b, pik_b);

    // Cross-place each item into the opposite bin at the other item's former position.
    prob.place_item(BPPlacement {
        layout_id: BPLayoutType::Open(bin_b),
        item_id: placement_a.item_id,
        d_transf: placement_b.d_transf,
    });
    prob.place_item(BPPlacement {
        layout_id: BPLayoutType::Open(bin_a),
        item_id: placement_b.item_id,
        d_transf: placement_a.d_transf,
    });

    // Separate both touched bins. Share the outer terminator so this can't burn budget.
    let feasible_a = separate_single_bin(prob, bin_a, sep_config, rng, term);
    let feasible_b = separate_single_bin(prob, bin_b, sep_config, rng, term);

    if feasible_a && feasible_b {
        true
    } else {
        prob.restore(&snapshot);
        false
    }
}

/// Pick a `PItemKey` from `layout`, weighted toward larger items (top half by area).
fn pick_large_item_pk(
    layout: &Layout,
    instance: &BPInstance,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<PItemKey> {
    if layout.placed_items.is_empty() {
        return None;
    }
    let mut items: Vec<(PItemKey, f32)> = layout
        .placed_items
        .iter()
        .map(|(pk, pi)| (pk, instance.item(pi.item_id).shape_orig.area()))
        .collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_half = ((items.len() + 1) / 2).max(1);
    let idx = rng.random_range(0..top_half);
    Some(items[idx].0)
}
