// Exact checks on f32 coordinates, done on their bit patterns so the result
// never depends on a device's floating-point behaviour (fused operations,
// flushing tiny values to zero). Load positions with bitcast<vec3<u32>>.

// True if none of the three coordinates is infinite or NaN.
public fn is_finite_position(bits: vec3<u32>) -> bool {
    let exponents = (bits >> vec3(23u)) & vec3(0xffu);
    return all(exponents != vec3(0xffu));
}

// True if two coordinates are equal as f32 values: the same bits, or zeros of
// either sign. Inputs must be finite.
public fn same_coordinate(a: u32, b: u32) -> bool {
    return a == b || ((a | b) & 0x7fffffffu) == 0u;
}

// True if two finite positions are equal coordinate by coordinate.
public fn same_position(a: vec3<u32>, b: vec3<u32>) -> bool {
    return same_coordinate(a.x, b.x) && same_coordinate(a.y, b.y) && same_coordinate(a.z, b.z);
}

// True if two of a triangle's corner positions are equal, which makes the
// triangle degenerate under MOPMesh's rule. Collinear corners are not.
public fn has_coincident_corners(a: vec3<u32>, b: vec3<u32>, c: vec3<u32>) -> bool {
    return same_position(a, b) || same_position(b, c) || same_position(a, c);
}
