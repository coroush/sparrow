extern crate core;

use clap::Parser as Clap;
use jagua_rs::io::import::Importer;
use log::{info, warn, Level};
use rand::SeedableRng;
use sparrow::config::*;
use sparrow::optimizer::optimize;
use sparrow::util::io;
use sparrow::util::io::{ExtBPOutput, ExtSPOutput, MainCli};
use sparrow::EPOCH;
use std::fs;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Result};
use rand::rngs::Xoshiro256PlusPlus;
use sparrow::config::ItemSortKey;
use sparrow::consts::{DEFAULT_COMPRESS_TIME_RATIO, DEFAULT_EXPLORE_TIME_RATIO, DEFAULT_FAIL_DECAY_RATIO_CMPR, DEFAULT_MAX_CONSEQ_FAILS_EXPL, LOG_LEVEL_FILTER_DEBUG, LOG_LEVEL_FILTER_RELEASE};
use sparrow::util::ctrlc_terminator::CtrlCTerminator;
use sparrow::util::svg_exporter::SvgExporter;

pub const OUTPUT_DIR: &str = "output";

pub const LIVE_DIR: &str = "data/live";

fn main() -> Result<()> {
    fs::create_dir_all(OUTPUT_DIR)?;
    let log_file_path = format!("{}/log.txt", OUTPUT_DIR);
    match cfg!(debug_assertions) {
        true => io::init_logger(LOG_LEVEL_FILTER_DEBUG, Path::new(&log_file_path))?,
        false => io::init_logger(LOG_LEVEL_FILTER_RELEASE, Path::new(&log_file_path))?,
    }

    let args = MainCli::parse();

    // Determine RNG seed
    let rng_seed = args.rng_seed;
    let rng = match rng_seed {
        Some(seed) => {
            info!("[MAIN] using seed: {}", seed);
            Xoshiro256PlusPlus::seed_from_u64(seed)
        }
        None => {
            let seed = rand::random();
            warn!("[MAIN] no seed provided, using: {}", seed);
            Xoshiro256PlusPlus::seed_from_u64(seed)
        }
    };

    info!("[MAIN] system time: {}", jiff::Timestamp::now());

    match args.mode.as_str() {
        "sp" => run_strip_packing(args, rng)?,
        "bp" => run_bin_packing(args, rng)?,
        _ => bail!("Unknown mode '{}'. Use 'sp' (strip packing) or 'bp' (bin packing).", args.mode),
    }

    Ok(())
}

fn run_strip_packing(args: MainCli, rng: Xoshiro256PlusPlus) -> Result<()> {
    let mut config = DEFAULT_SPARROW_CONFIG;

    let (explore_dur, compress_dur) = match (args.global_time, args.exploration, args.compression) {
        (Some(gt), None, None) => (
            Duration::from_secs(gt).mul_f32(DEFAULT_EXPLORE_TIME_RATIO),
            Duration::from_secs(gt).mul_f32(DEFAULT_COMPRESS_TIME_RATIO),
        ),
        (None, Some(et), Some(ct)) => (Duration::from_secs(et), Duration::from_secs(ct)),
        (None, None, None) => {
            warn!("[MAIN] no time limit specified");
            (
                Duration::from_secs(600).mul_f32(DEFAULT_EXPLORE_TIME_RATIO),
                Duration::from_secs(600).mul_f32(DEFAULT_COMPRESS_TIME_RATIO),
            )
        }
        _ => bail!("invalid cli pattern (clap should have caught this)"),
    };
    config.expl_cfg.time_limit = explore_dur;
    config.cmpr_cfg.time_limit = compress_dur;
    if args.early_termination {
        config.expl_cfg.max_conseq_failed_attempts = Some(DEFAULT_MAX_CONSEQ_FAILS_EXPL);
        config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
        warn!("[MAIN] early termination enabled!");
    }
    if let Some(arg_rng_seed) = args.rng_seed {
        config.rng_seed = Some(arg_rng_seed as usize);
    }
    if let Some(spacing) = args.spacing {
        config.min_item_separation = Some(spacing);
        info!("[MAIN] minimum item spacing: {}", spacing);
    }

    info!(
        "[MAIN] configured to explore for {}s and compress for {}s",
        explore_dur.as_secs(),
        compress_dur.as_secs()
    );

    let (ext_instance, ext_solution) = io::read_spp_input(Path::new(&args.input))?;

    let importer = Importer::new(
        config.cde_config,
        config.poly_simpl_tolerance,
        config.min_item_separation,
        config.narrow_concavity_cutoff_ratio,
    );
    let instance = jagua_rs::probs::spp::io::import_instance(&importer, &ext_instance)?;

    // Pre-flight: warn if any item can't fit within the strip height
    {
        use jagua_rs::entities::Instance;
        let strip_h = instance.base_strip.fixed_height;
        for item in instance.items() {
            let iw = item.shape_cd.bbox.width();
            let ih = item.shape_cd.bbox.height();
            if ih > strip_h && iw > strip_h {
                warn!(
                    "[MAIN] ERROR: item {} ({:.2}×{:.2}) does not fit in strip height {:.2} \
                     in any axis-aligned orientation — increase strip_h",
                    item.id, iw, ih, strip_h
                );
            }
        }
    }

    let initial_solution = ext_solution.map(|e| jagua_rs::probs::spp::io::import_solution(&instance, &e));

    info!("[MAIN] loaded instance {} with #{} items", ext_instance.name, instance.total_item_qty());

    let mut svg_exporter = {
        let final_svg_path = Some(format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name));
        let intermediate_svg_dir = match cfg!(feature = "only_final_svg") {
            true => None,
            false => Some(format!("{OUTPUT_DIR}/sols_{}", ext_instance.name)),
        };
        let live_svg_path = match cfg!(feature = "live_svg") {
            true => Some(format!("{LIVE_DIR}/.live_solution.svg")),
            false => None,
        };
        SvgExporter::new(final_svg_path, intermediate_svg_dir, live_svg_path)
    };

    let mut ctrlc_terminator = CtrlCTerminator::new();

    let solution = optimize(
        instance.clone(),
        rng,
        &mut svg_exporter,
        &mut ctrlc_terminator,
        &config.expl_cfg,
        &config.cmpr_cfg,
        initial_solution.as_ref(),
    );

    let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    let json_output = ExtSPOutput {
        instance: ext_instance,
        solution: jagua_rs::probs::spp::io::export(&instance, &solution, *EPOCH),
    };
    io::write_json(&json_output, Path::new(json_path.as_str()), Level::Info)?;

    Ok(())
}

fn run_bin_packing(args: MainCli, rng: Xoshiro256PlusPlus) -> Result<()> {
    let mut bp_config = DEFAULT_BINPACK_CONFIG;

    // Time limit: use global_time if provided, otherwise default
    if let Some(t) = args.global_time {
        bp_config.time_limit = Duration::from_secs(t);
    } else if args.global_time.is_none() && args.exploration.is_none() {
        warn!("[MAIN] no time limit specified for bin packing");
    }

    // Sort key for initial FFD construction
    bp_config.sort_key = match args.sort_key.as_str() {
        "ch-area"      => ItemSortKey::ChArea,
        "exact-area"   => ItemSortKey::ExactArea,
        _              => ItemSortKey::ChAreaTimesDiameter,
    };
    info!("[MAIN] BP sort key: {}", args.sort_key);

    // SPP-compatible CDE/importer config (reuse defaults)
    let spp_config = DEFAULT_SPARROW_CONFIG;
    let min_item_separation = args.spacing;
    if let Some(spacing) = min_item_separation {
        info!("[MAIN] minimum item spacing: {}", spacing);
    }

    let ext_instance = io::read_bpp_input(Path::new(&args.input))?;

    let importer = Importer::new(
        spp_config.cde_config,
        spp_config.poly_simpl_tolerance,
        min_item_separation,
        spp_config.narrow_concavity_cutoff_ratio,
    );
    let instance = jagua_rs::probs::bpp::io::import_instance(&importer, &ext_instance)?;

    info!("[MAIN] loaded BPP instance {} with #{} items", ext_instance.name, instance.total_item_qty());

    let mut ctrlc_terminator = CtrlCTerminator::new();

    let solution = sparrow::bp_optimizer::bp_optimize(
        instance.clone(),
        rng,
        &mut ctrlc_terminator,
        &bp_config,
    );

    // Export SVG
    let svg_path = format!("{OUTPUT_DIR}/final_{}.svg", ext_instance.name);
    sparrow::bp_optimizer::export_bp_svg(&solution, &instance, Path::new(&svg_path))?;

    // Export JSON
    let json_path = format!("{OUTPUT_DIR}/final_{}.json", ext_instance.name);
    let json_output = ExtBPOutput {
        instance: ext_instance,
        solution: jagua_rs::probs::bpp::io::export(&instance, &solution, *EPOCH),
    };
    io::write_json(&json_output, Path::new(json_path.as_str()), Level::Info)?;

    Ok(())
}
