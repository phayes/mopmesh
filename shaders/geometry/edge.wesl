// Finite, reasonably scaled f32 positions are expected.
public fn edge_length(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return length(b - a);
}
