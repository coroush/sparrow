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

/// Warn if any item cannot fit in any bin under *any* rotation (continuous sweep).
/// Sparrow supports continuous rotation, so we check the minimum AABB across rotations
/// instead of just 0°/90°. Logs a clear message so Grasshopper's Status panel shows it.
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

            let fits_axis_aligned = (iw <= bw && ih <= bh) || (ih <= bw && iw <= bh);

            if fits_axis_aligned {
                continue;
            }

            // Sweep rotations to find the smallest rotated AABB that fits.
            // Step in 0.5° over [0, 180) — for a polygon the rotated AABB period is 180°.
            let vertices = &item.shape_cd.vertices;
            let mut best: Option<(f32, f32, f32)> = None; // (angle_deg, w, h)
            let steps = 360; // 0.5° resolution
            for k in 0..steps {
                let theta = (k as f32) * std::f32::consts::PI / (steps as f32);
                let (s, c) = theta.sin_cos();
                let mut x_min = f32::INFINITY;
                let mut x_max = f32::NEG_INFINITY;
                let mut y_min = f32::INFINITY;
                let mut y_max = f32::NEG_INFINITY;
                for p in vertices {
                    let xr = p.0 * c - p.1 * s;
                    let yr = p.0 * s + p.1 * c;
                    if xr < x_min { x_min = xr; }
                    if xr > x_max { x_max = xr; }
                    if yr < y_min { y_min = yr; }
                    if yr > y_max { y_max = yr; }
                }
                let w = x_max - x_min;
                let h = y_max - y_min;
                let fits = (w <= bw && h <= bh) || (h <= bw && w <= bh);
                if fits {
                    let deg = theta.to_degrees();
                    match best {
                        None => best = Some((deg, w, h)),
                        Some((_, bw_old, bh_old)) if w * h < bw_old * bh_old => {
                            best = Some((deg, w, h));
                        }
                        _ => {}
                    }
                }
            }

            match best {
                Some((deg, w, h)) => {
                    info!(
                        "[BP] note: item {} ({:.2}×{:.2}) only fits in bin ({:.2}×{:.2}) when rotated ~{:.1}° (rotated AABB {:.2}×{:.2})",
                        item.id, iw, ih, bw, bh, deg, w, h
                    );
                }
                None => {
                    warn!(
                        "[BP] ERROR: item {} ({:.2}×{:.2}) does not fit in bin ({:.2}×{:.2}) \
                         under any rotation — increase SheetWidth/SheetHeight",
                        item.id, iw, ih, bw, bh
                    );
                }
            }
        }
    }
}
