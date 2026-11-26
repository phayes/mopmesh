// Finite f32 positions are expected. Divide before adding to reduce overflow.
public fn triangle_centroid(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return a / 3.0 + b / 3.0 + c / 3.0;
}
