use crate::eval::lbf_evaluator::{X_MULTIPLIER, Y_MULTIPLIER};
use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::eval::specialized_jaguars_pipeline::{collect_poly_collisions_in_detector_custom, SpecializedHazardCollector};
use crate::quantify::tracker::CollisionTracker;
use jagua_rs::collision_detection::hazards::collector::HazardCollector;
use jagua_rs::entities::Item;
use jagua_rs::entities::Layout;
use jagua_rs::entities::PItemKey;
use jagua_rs::geometry::primitives::SPolygon;
use jagua_rs::geometry::DTransformation;

/// Weight applied to the LBF gradient in the Clear branch.
/// Small enough that any collision always outranks any clear position,
/// but nonzero so the sampler gravitates toward compact bottom-left placements.
const SEP_LBF_EPSILON: f32 = 1e-4;

pub struct SeparationEvaluator<'a> {
    layout: &'a Layout,
    item: &'a Item,
    collector: SpecializedHazardCollector<'a>,
    shape_buff: SPolygon,
    n_evals: usize,
}

impl<'a> SeparationEvaluator<'a> {
    pub fn new(
        layout: &'a Layout,
        item: &'a Item,
        current_pk: PItemKey,
        ct: &'a CollisionTracker,
    ) -> Self {
        let collector = SpecializedHazardCollector::new(layout, ct, current_pk);

        Self {
            layout,
            item,
            collector,
            shape_buff: item.shape_cd.as_ref().clone(),
            n_evals: 0,
        }
    }
}

impl<'a> SampleEvaluator for SeparationEvaluator<'a> {
    /// Evaluates a transformation. An upper bound can be provided to early terminate the process.
    /// Algorithm 7 from https://doi.org/10.48550/arXiv.2509.13329
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        self.n_evals += 1;
        let cde = self.layout.cde();

        // Calculate an upper bound of quantification, above which samples are guaranteed to be rejected (because they are dominated by previous ones).
        let loss_bound = match upper_bound {
            Some(SampleEval::Collision { loss }) => loss,
            Some(SampleEval::Clear { .. }) => 0.0,
            _ => f32::INFINITY,
        };
        
        // Reload the hazard collector to prepare for a new query
        self.collector.reload(loss_bound);

        // Perform the collision detection and the collision quantification. The 'collector' will be populated with the results.
        collect_poly_collisions_in_detector_custom(cde, &dt, &mut self.shape_buff, self.item.shape_cd.as_ref(), &mut self.collector);
        
        if self.collector.early_terminate(&self.shape_buff) {
            // The total quantification of the collisions exceeded the upper bound and the process was terminated early.
            // Note that we might have exited before detecting/quantifying all collisions.
            // However, since we can asure that this sample will always be rejected, we don't need to spend any more time on it and just return `Invalid`.
            SampleEval::Invalid
        } else if self.collector.is_empty() {
            // No collisions detected; add a small LBF gradient so the sampler
            // gravitates toward compact bottom-left positions as a side effect.
            let poi = self.shape_buff.poi.center;
            let bbox_corner = self.shape_buff.bbox.corners()[0];
            let lbf = X_MULTIPLIER * (poi.0 + bbox_corner.0) + Y_MULTIPLIER * (poi.1 + bbox_corner.1);
            SampleEval::Clear { loss: SEP_LBF_EPSILON * lbf }
        } else {
            // Some collisions detected but withing the upper bound, return collision with total loss
            SampleEval::Collision {
                loss: self.collector.loss(&self.shape_buff),
            }
        }
    }

    fn n_evals(&self) -> usize {
        self.n_evals
    }
}

