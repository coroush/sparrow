use clipper2_rust::offset::{EndType, JoinType};
use clipper2_rust::{inflate_paths_d, make_path_d};
use jagua_rs::io::ext_repr::{ExtShape, ExtSPolygon};
use log::warn;

/// Inflates an [`ExtShape`] outward by `delta` using Clipper2.
/// Only `SimplePolygon` shapes are offset; other variants are returned unchanged.
/// Falls back to the original shape if Clipper2 produces no result.
pub fn offset_shape(shape: &ExtShape, delta: f64) -> ExtShape {
    match shape {
        ExtShape::SimplePolygon(esp) => ExtShape::SimplePolygon(offset_spolygon(esp, delta)),
        other => other.clone(),
    }
}

fn offset_spolygon(esp: &ExtSPolygon, delta: f64) -> ExtSPolygon {
    // Flatten points into [x0, y0, x1, y1, ...] as f64
    let flat: Vec<f64> = esp.0.iter().flat_map(|(x, y)| [*x as f64, *y as f64]).collect();
    let path = make_path_d(&flat);

    // EndType::Polygon = treat as closed shape, offset outward
    let result = inflate_paths_d(&vec![path], delta, JoinType::Round, EndType::Polygon, 2.0, 2, 0.0);

    match result.into_iter().next() {
        Some(path) => ExtSPolygon(path.iter().map(|p| (p.x as f32, p.y as f32)).collect()),
        None => {
            warn!("[SPACING] Clipper2 offset produced no result, using original shape");
            esp.clone()
        }
    }
}
