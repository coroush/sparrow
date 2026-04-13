pub mod bp_lbf;
pub mod bp_separator;
pub mod bp_explore;
pub mod bp_moves;

use crate::bp_optimizer::bp_explore::bin_reduction_phase;
use crate::bp_optimizer::bp_lbf::BpLbfBuilder;
use crate::config::BinPackConfig;
use crate::util::terminator::Terminator;
use anyhow::Result;
use jagua_rs::entities::Layout;
use jagua_rs::io::svg::{layout_to_svg, SvgDrawOptions, SvgLayoutTheme};
use jagua_rs::probs::bpp::entities::{BPInstance, BPProblem, BPSolution};
use log::info;
use rand::rngs::Xoshiro256PlusPlus;
use std::path::Path;
use svg::Document;

/// Entry point for bin packing optimization.
pub fn bp_optimize(
    instance: BPInstance,
    rng: Xoshiro256PlusPlus,
    terminator: &mut impl Terminator,
    config: &BinPackConfig,
) -> BPSolution {
    terminator.new_timeout(config.time_limit);

    // Pre-flight: warn about items whose bounding box exceeds the bin
    validate_items_fit_bins(&instance);

    // Phase 1: construct an initial feasible multi-bin solution via FFD + per-bin separator
    let prob = BpLbfBuilder::new(instance.clone(), rng.clone(), config.separator_config, config.sort_key).construct();
    let n_initial = prob.layouts.len();
    info!("[BP] initial assignment: {} bins used", n_initial);

    // Phase 2: try to reduce the bin count
    let final_prob = bin_reduction_phase(instance.clone(), prob, rng, terminator, config);

    let solution = final_prob.save();
    let n_final = solution.layout_snapshots.len();
    info!(
        "[BP] final solution: {} bins, density {:.3}",
        n_final,
        solution.density(&instance)
    );

    solution
}

/// Export a multi-bin SVG — each bin rendered side-by-side.
pub fn export_bp_svg(solution: &BPSolution, instance: &BPInstance, path: &Path) -> Result<()> {
    use jagua_rs::entities::LayoutSnapshot;
    use svg::node::element::Group;

    let gap = 50.0_f32;
    let draw_options = SvgDrawOptions {
        theme: SvgLayoutTheme::GRAY,
        quadtree: false,
        surrogate: false,
        highlight_collisions: false,
        draw_cd_shapes: false,
        highlight_cd_shapes: false,
    };

    // Reconstruct each bin Layout so we can render it
    let layouts: Vec<Layout> = solution
        .layout_snapshots
        .values()
        .map(|ls| Layout::from_snapshot(ls))
        .collect();

    if layouts.is_empty() {
        let doc = Document::new();
        crate::util::io::write_svg(&doc, path, log::Level::Info)?;
        return Ok(());
    }

    // Compute bin width from first bin
    let bin_width = layouts[0].container.outer_cd.bbox.width();
    let bin_height = layouts[0].container.outer_cd.bbox.height();

    // Build one SVG document with all bins side by side
    let total_width = layouts.len() as f32 * (bin_width + gap) - gap;
    let vbox = format!("0 0 {} {}", total_width * 1.05, bin_height * 1.1);

    let mut doc = Document::new().set("viewBox", vbox);

    for (idx, layout) in layouts.iter().enumerate() {
        let offset_x = idx as f32 * (bin_width + gap);
        // Render the layout to an SVG group using jagua-rs
        let (group, _) = jagua_rs::io::svg::layout_to_svg_group(
            layout,
            instance,
            draw_options,
            &format!("bin_{}", idx),
        );
        // Wrap in a translate group
        let translated = Group::new()
            .set("transform", format!("translate({}, 0)", offset_x))
            .add(group);
        doc = doc.add(translated);
    }

    crate::util::io::write_svg(&doc, path, log::Level::Info)?;
    Ok(())
}

/// Warn (or error) if any item's axis-aligned bounding box exceeds every bin's dimensions.
/// Logs a clear message so Grasshopper's Status panel shows it immediately.
fn validate_items_fit_bins(instance: &BPInstance) {
    use jagua_rs::entities::Instance;
    use log::warn;

    for bin in &instance.bins {
        let bw = bin.container.outer_cd.bbox.width();
        let bh = bin.container.outer_cd.bbox.height();

        for (item, qty) in &instance.items {
            if *qty == 0 { continue; }
            let iw = item.shape_cd.bbox.width();
            let ih = item.shape_cd.bbox.height();

            // Check both orientations (item may be rotatable 90°)
            let fits_normal  = iw <= bw && ih <= bh;
            let fits_rotated = ih <= bw && iw <= bh;

            if !fits_normal && !fits_rotated {
                warn!(
                    "[BP] ERROR: item {} ({:.2}×{:.2}) does not fit in bin ({:.2}×{:.2}) \
                     in any axis-aligned orientation — increase SheetWidth/SheetHeight",
                    item.id, iw, ih, bw, bh
                );
            } else if !fits_normal && fits_rotated {
                // Fine — will be rotated — but log it so the user knows
                info!(
                    "[BP] note: item {} ({:.2}×{:.2}) only fits rotated 90° in bin ({:.2}×{:.2})",
                    item.id, iw, ih, bw, bh
                );
            }
        }
    }
}
