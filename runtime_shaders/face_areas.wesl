import package::shaders::geometry::triangle::triangle_area;

@group(0) @binding(0) var<storage, read> faces: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
// One area per face.
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&faces) / 3u) { return; }
    let base = id.x * 3u;
    let a = positions[faces[base]].xyz;
    let b = positions[faces[base + 1u]].xyz;
    let c = positions[faces[base + 2u]].xyz;
    output[id.x] = triangle_area(a, b, c);
}
