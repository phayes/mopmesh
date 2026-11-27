import mopmesh::geometry::triangle::{triangle_area, triangle_area_vector, triangle_normal, normalize_or_zero};

// This downstream shader chooses its own bindings and entry point.
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let a = vec3(0.0, 0.0, 0.0);
    let b = vec3(2.0, 0.0, 0.0);
    let c = vec3(0.0, 2.0, 0.0);
    output[0] = triangle_area_vector(a, b, c).z;
    output[1] = triangle_normal(a, b, c).z;
    output[2] = triangle_area(a, b, c);
    // Collinear corners give a zero normal, not NaN.
    output[3] = length(triangle_normal(a, b, vec3(4.0, 0.0, 0.0)));
    output[4] = length(normalize_or_zero(vec3(0.0)));
}
