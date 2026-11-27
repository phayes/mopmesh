// GPU build, first stage: element checks and vertex -> corner rows.
// Corner `c` is slot `c % 3` of face `c / 3`.
import package::shaders::build::exact::{is_finite_position, has_coincident_corners};

import package::shaders::parallel::grid::flat_index;

struct Params { vertices: u32, faces: u32, }
@group(0) @binding(0) var<uniform> params: Params;
// Position bit patterns; the fourth lane is ignored.
@group(0) @binding(1) var<storage, read> positions: array<vec4<u32>>;
// Vertex IDs per face; the fourth lane is ignored.
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
// [first non-finite vertex, first failing face, failure kind, failing vertex].
// Unset words hold 0xffffffff. Kind 0: missing vertex; kind 1: degenerate face.
@group(0) @binding(3) var<storage, read_write> errors: array<atomic<u32>>;
// Per-vertex counts, scanned in place into row offsets, then used as fill cursors.
@group(0) @binding(4) var<storage, read_write> cursors: array<atomic<u32>>;
// [V + 1 row offsets, 3F corner IDs]. Offsets index the corner region.
@group(0) @binding(5) var<storage, read_write> rows: array<u32>;

const NONE: u32 = 0xffffffffu;

fn face_fails(face: u32) -> bool {
    let t = triangles[face];
    if (t.x >= params.vertices || t.y >= params.vertices || t.z >= params.vertices) {
        return true;
    }
    return has_coincident_corners(positions[t.x].xyz, positions[t.y].xyz, positions[t.z].xyz);
}

@compute @workgroup_size(64)
fn check_positions(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.vertices) { return; }
    if (!is_finite_position(positions[id.x].xyz)) {
        atomicMin(&errors[0], id.x);
    }
}

@compute @workgroup_size(64)
fn check_faces(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    if (face_fails(id.x)) {
        atomicMin(&errors[1], id.x);
    }
}

// One invocation: records why the first failing face failed.
@compute @workgroup_size(1)
fn describe_face_error() {
    let face = atomicLoad(&errors[1]);
    if (face == NONE) { return; }
    let t = triangles[face];
    for (var i = 0u; i < 3u; i++) {
        if (t[i] >= params.vertices) {
            atomicStore(&errors[2], 0u);
            atomicStore(&errors[3], t[i]);
            return;
        }
    }
    atomicStore(&errors[2], 1u);
}

// Corners with a missing vertex are skipped; the build reports them as errors.
@compute @workgroup_size(64)
fn count_corners(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    let t = triangles[id.x];
    for (var i = 0u; i < 3u; i++) {
        if (t[i] < params.vertices) {
            atomicAdd(&cursors[t[i]], 1u);
        }
    }
}

// Cursors now hold exclusive prefixes for V + 1 rows; the last is the total.
@compute @workgroup_size(64)
fn store_offsets(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x > params.vertices) { return; }
    rows[id.x] = atomicLoad(&cursors[id.x]);
}

// Places corners within each row in no particular order; the order may change
// from run to run, so later stages must not depend on it.
@compute @workgroup_size(64)
fn fill_corners(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    let t = triangles[id.x];
    for (var i = 0u; i < 3u; i++) {
        if (t[i] < params.vertices) {
            let slot = atomicAdd(&cursors[t[i]], 1u);
            rows[params.vertices + 1u + slot] = 3u * id.x + i;
        }
    }
}
