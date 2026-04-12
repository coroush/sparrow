//! Bin-reduction phase: repeatedly try to eliminate the least-loaded bin.

use crate::bp_optimizer::bp_moves::{resolve_by_transfers, separate_single_bin};
use crate::config::BinPackConfig;
use crate::util::terminator::Terminator;
use jagua_rs::entities::Instance;
use jagua_rs::probs::bpp::entities::{BPInstance, BPLayoutType, BPPlacement, BPProblem, LayKey};
use log::info;
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};

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
    let mut strikes = 0;

    info!(
        "[BP-EXPL] starting bin reduction: {} bins, lower bound: {}",
        n_bins, lower_bound
    );

    while n_bins > lower_bound && strikes < config.max_reduction_strikes && !term.kill() {
        prob.restore(&incumbent);

        let Some(bin_to_remove) = select_lowest_utilisation_bin(&prob, &instance) else {
            break;
        };

        info!("[BP-EXPL] attempting to eliminate bin {:?}", bin_to_remove);

        // Collect displaced items
        let mut displaced: Vec<(usize, jagua_rs::geometry::DTransformation)> = prob.layouts[bin_to_remove]
            .placed_items
            .values()
            .map(|pi| (pi.item_id, pi.d_transf))
            .collect();

        // Sort largest-first for better FFD redistribution
        displaced.sort_by(|(a_id, _), (b_id, _)| {
            let a = instance.item(*a_id).shape_orig.area();
            let b = instance.item(*b_id).shape_orig.area();
            b.partial_cmp(&a).unwrap()
        });

        prob.remove_layout(bin_to_remove);
        let remaining_keys: Vec<LayKey> = prob.layouts.keys().collect();

        if remaining_keys.is_empty() {
            strikes += 1;
            continue;
        }

        // Distribute into remaining bins round-robin style
        for (i, (item_id, d_transf)) in displaced.iter().enumerate() {
            let target = remaining_keys[i % remaining_keys.len()];
            prob.place_item(BPPlacement {
                layout_id: BPLayoutType::Open(target),
                item_id: *item_id,
                d_transf: *d_transf,
            });
        }

        // Separate all affected bins — using the outer terminator so we stop on timeout
        let affected_bins: Vec<LayKey> = prob.layouts.keys().collect();
        let mut all_feasible = true;

        for &lkey in &affected_bins {
            if term.kill() { break; }
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
            incumbent = prob.save();
            n_bins -= 1;
            strikes = 0;
            info!(
                "[BP-EXPL] reduced to {} bins (density {:.3})",
                n_bins,
                incumbent.density(&instance)
            );
        } else if !term.kill() {
            // Try inter-bin moves — only if we still have time
            info!("[BP-EXPL] direct redistribution infeasible, trying inter-bin moves...");

            let infeasible_bins: Vec<LayKey> = affected_bins
                .iter()
                .copied()
                .filter(|&lkey| {
                    if !prob.layouts.contains_key(lkey) { return false; }
                    use crate::quantify::tracker::CollisionTracker;
                    CollisionTracker::new(&prob.layouts[lkey]).get_total_loss() > 0.0
                })
                .collect();

            let resolved = resolve_by_transfers(
                &mut prob,
                &infeasible_bins,
                &affected_bins,
                &config.separator_config,
                &mut rng,
                config.inter_bin_move_budget,
                term,
            );

            if resolved {
                incumbent = prob.save();
                n_bins -= 1;
                strikes = 0;
                info!(
                    "[BP-EXPL] reduced to {} bins via inter-bin moves (density {:.3})",
                    n_bins,
                    incumbent.density(&instance)
                );
            } else {
                strikes += 1;
                info!(
                    "[BP-EXPL] bin reduction failed, strike {}/{}",
                    strikes, config.max_reduction_strikes
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

fn select_lowest_utilisation_bin(prob: &BPProblem, instance: &BPInstance) -> Option<LayKey> {
    prob.layouts
        .iter()
        .min_by(|(_, la), (_, lb)| {
            let ua = la.placed_item_area(instance) / la.container.area();
            let ub = lb.placed_item_area(instance) / lb.container.area();
            ua.partial_cmp(&ub).unwrap()
        })
        .map(|(k, _)| k)
}
