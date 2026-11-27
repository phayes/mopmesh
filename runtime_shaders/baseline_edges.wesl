// Unpartitioned twin of edges.wesl, for benchmarks: edge records from global
// FE and EV arrays instead of the patch arena. Run `initialize` per edge,
// `scatter` per face, then `compact` per edge.
import package::shaders::topology::edge::INVALID_FACE;

// Per face, three (global edge ID, reversed 0/1) pairs.
@group(0) @binding(0) var<storage, read> fe: array<u32>;
// Per edge, its two global vertex IDs in ascending order.
@group(0) @binding(1) var<storage, read> ev: array<vec2<u32>>;
// Four words per edge, laid out as EdgeRecord: two vertices, two faces.
// Plain words, so faces writing different slots of one record never share a store.
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// Copies the endpoints and marks both faces missing.
@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&ev)) { return; }
    let base = 4u * id.x;
    output[base] = ev[id.x].x;
    output[base + 1u] = ev[id.x].y;
    output[base + 2u] = INVALID_FACE;
    output[base + 3u] = INVALID_FACE;
}

// Consistent winding uses each edge at most once in each direction, so the
// direction flag picks a slot no other face writes.
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let face = id.x;
    if (face >= arrayLength(&fe) / 6u) { return; }
    for (var i = 0u; i < 3u; i++) {
        let edge = fe[6u * face + 2u * i];
        let reversed = fe[6u * face + 2u * i + 1u];
        output[4u * edge + 2u + reversed] = face;
    }
}

// Moves a boundary edge's only face into the first slot.
@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&ev)) { return; }
    let base = 4u * id.x;
    if (output[base + 2u] == INVALID_FACE) {
        output[base + 2u] = output[base + 3u];
        output[base + 3u] = INVALID_FACE;
    }
}
