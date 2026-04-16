//! Inter-bin move operators: transfer and swap.

use crate::bp_optimizer::bp_separator::BinSeparator;
use crate::optimizer::separator::SeparatorConfig;
use crate::util::terminator::Terminator;
use jagua_rs::probs::bpp::entities::{BPLayoutType, BPPlacement, BPProblem, LayKey};
use log::debug;
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

/// Run the separator on a single bin, respecting the outer time budget.
/// Returns `true` if the bin is feasible (zero loss) afterward.
pub fn separate_single_bin(
    prob: &mut BPProblem,
    lkey: LayKey,
    sep_config: &SeparatorConfig,
    rng: &mut Xoshiro256PlusPlus,
    term: &impl Terminator,
) -> bool {
    if !prob.layouts.contains_key(lkey) {
        return true;
    }
    let layout = prob.layouts[lkey].clone();
    let mut sep = BinSeparator::new(
        prob.instance.clone(),
        layout,
        Xoshiro256PlusPlus::seed_from_u64(rng.random()),
        *sep_config,
    );
    let (best_snap, _) = sep.separate(term);
    prob.layouts[lkey].restore(&best_snap);
    sep.is_feasible()
}

/// Transfer an item from `from_bin` to `to_bin`, then re-separate `to_bin`.
/// Returns `true` if `to_bin` is feasible afterward.
pub fn try_transfer(
    prob: &mut BPProblem,
    from_bin: LayKey,
    to_bin: LayKey,
    item_pik: jagua_rs::entities::PItemKey,
    sep_config: &SeparatorConfig,
    rng: &mut Xoshiro256PlusPlus,
    term: &impl Terminator,
) -> bool {
    let removed = prob.remove_item(from_bin, item_pik);
    prob.place_item(BPPlacement {
        layout_id: BPLayoutType::Open(to_bin),
        item_id: removed.item_id,
        d_transf: removed.d_transf,
    });
    let feasible = separate_single_bin(prob, to_bin, sep_config, rng, term);
    debug!(
        "[BP-MOVE] transfer item {} → bin {:?}: feasible={}",
        removed.item_id, to_bin, feasible
    );
    feasible
}

/// Swap items between two bins, re-separate both.
/// Returns `true` if both bins are feasible afterward.
pub fn try_swap(
    prob: &mut BPProblem,
    bin_a: LayKey,
    pik_a: jagua_rs::entities::PItemKey,
    bin_b: LayKey,
    pik_b: jagua_rs::entities::PItemKey,
    sep_config: &SeparatorConfig,
    rng: &mut Xoshiro256PlusPlus,
    term: &impl Terminator,
) -> bool {
    let removed_a = prob.remove_item(bin_a, pik_a);
    let removed_b = prob.remove_item(bin_b, pik_b);
    prob.place_item(BPPlacement {
        layout_id: BPLayoutType::Open(bin_b),
        item_id: removed_a.item_id,
        d_transf: removed_a.d_transf,
    });
    prob.place_item(BPPlacement {
        layout_id: BPLayoutType::Open(bin_a),
        item_id: removed_b.item_id,
        d_transf: removed_b.d_transf,
    });
    let feas_a = separate_single_bin(prob, bin_a, sep_config, rng, term);
    let feas_b = separate_single_bin(prob, bin_b, sep_config, rng, term);
    debug!(
        "[BP-MOVE] swap items {} ↔ {}: feas_a={}, feas_b={}",
        removed_a.item_id, removed_b.item_id, feas_a, feas_b
    );
    feas_a && feas_b
}

/// Try single-item transfers from infeasible bins to resolve overpacking.
/// Returns `true` if all bins end up feasible.
pub fn resolve_by_transfers(
    prob: &mut BPProblem,
    infeasible_bins: &[LayKey],
    all_bins: &[LayKey],
    sep_config: &SeparatorConfig,
    rng: &mut Xoshiro256PlusPlus,
    budget: usize,
    term: &impl Terminator,
) -> bool {
    let mut attempts = 0;

    'outer: for &src_bin in infeasible_bins {
        let items_in_bin: Vec<(jagua_rs::entities::PItemKey, usize)> = prob.layouts[src_bin]
            .placed_items
            .iter()
            .map(|(pk, pi)| (pk, pi.item_id))
            .collect();

        // Label the pik loop so we can break it when a transfer succeeds.
        // After a successful transfer, separate_single_bin() may move items inside
        // src_bin (remove_item + place_item), which reassigns PItemKeys. The
        // remaining entries in items_in_bin would then be stale and cause a panic
        // in try_transfer → remove_item. Breaking 'item avoids using those keys.
        'item: for (pik, _) in items_in_bin {
            for &dst_bin in all_bins {
                if dst_bin == src_bin || attempts >= budget || term.kill() {
                    if term.kill() { break 'outer; }
                    continue;
                }

                let snapshot = prob.save();
                let feasible = try_transfer(prob, src_bin, dst_bin, pik, sep_config, rng, term);
                attempts += 1;

                if feasible {
                    let src_empty    = !prob.layouts.contains_key(src_bin);
                    let src_feasible = !src_empty
                        && separate_single_bin(prob, src_bin, sep_config, rng, term);
                    if src_feasible || src_empty {
                        break 'item;
                    }
                }
                prob.restore(&snapshot);
            }
        }
    }

    all_bins.iter().all(|&lkey| separate_single_bin(prob, lkey, sep_config, rng, term))
}
