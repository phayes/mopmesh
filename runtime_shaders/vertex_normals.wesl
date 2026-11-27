import package::shaders::geometry::triangle::{normalize_or_zero, triangle_area_vector};

struct Params { vertices: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> faces: array<u32>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
// VF CSR: V + 1 offsets, then face IDs. Offsets are relative to the face-ID region.
@group(0) @binding(3) var<storage, read> vertex_faces: array<u32>;
// One [x, y, z, 0] unit normal per vertex; zero for isolated vertices or
// vertices whose incident area vectors cancel.
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

// Area-weighted vertex normals: the normalized sum of the unnormalized
// cross products of the incident faces.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.vertices) { return; }
    let entries = params.vertices + 1u;
    var sum = vec3(0.0);
    for (var i = vertex_faces[id.x]; i < vertex_faces[id.x + 1u]; i++) {
        let base = 3u * vertex_faces[entries + i];
        let a = positions[faces[base]].xyz;
        let b = positions[faces[base + 1u]].xyz;
        let c = positions[faces[base + 2u]].xyz;
        sum += triangle_area_vector(a, b, c);
    }
    output[id.x] = vec4(normalize_or_zero(sum), 0.0);
}
