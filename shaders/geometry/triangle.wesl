// Finite f32 positions are expected. Divide before adding to reduce overflow.
public fn triangle_centroid(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return a / 3.0 + b / 3.0 + c / 3.0;
}

// Returns the triangle's area vector: perpendicular to the triangle, pointing
// to the side from which a, b, c appear counter-clockwise, with length equal
// to twice the triangle's area. Summing these vectors over the faces around a
// vertex gives an area-weighted normal direction. Returns the zero vector for
// a triangle with zero area (for example, collinear corners).
// Finite, reasonably scaled f32 positions are expected.
public fn triangle_area_vector(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return cross(b - a, c - a);
}

// Returns `v` scaled to unit length, or the zero vector when `v` has zero
// length (including vectors so short that their squared length rounds to zero). Use it to turn a summed or area-weighted vector into a normal
// without producing NaN values for degenerate input.
public fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(v, v);
    if (length_squared == 0.0) { return vec3(0.0); }
    return v / sqrt(length_squared);
}

// Returns the triangle's unit normal, oriented as in `triangle_area_vector`,
// or the zero vector for a triangle with zero area.
public fn triangle_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return normalize_or_zero(triangle_area_vector(a, b, c));
}

// Returns the triangle's area: zero for collinear corners, never negative.
public fn triangle_area(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    return 0.5 * length(triangle_area_vector(a, b, c));
}
