// Unpartitioned twin of face_vertices.wesl, for benchmarks: the same decode
// from global FE and EV arrays instead of the patch arena.
import package::shaders::topology::face::face_vertices;

// Per face, three (global edge ID, reversed 0/1) pairs.
@group(0) @binding(0) var<storage, read> fe: array<u32>;
// Per edge, its two global vertex IDs in ascending order.
@group(0) @binding(1) var<storage, read> ev: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let face = id.x;
    if (face >= arrayLength(&fe) / 6u) { return; }
    var edges: array<vec2<u32>, 3>;
    var reversed: vec3<u32>;
    for (var i = 0u; i < 3u; i++) {
        edges[i] = ev[fe[6u * face + 2u * i]];
        reversed[i] = fe[6u * face + 2u * i + 1u];
    }
    let vertices = face_vertices(edges, reversed);
    output[3u * face] = vertices.x;
    output[3u * face + 1u] = vertices.y;
    output[3u * face + 2u] = vertices.z;
}
