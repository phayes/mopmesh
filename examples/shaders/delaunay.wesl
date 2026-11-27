// Example application rule for a flat mesh: request a flip when the angles
// opposite an edge sum to more than 180 degrees, allowing a small tolerance.
// Repeating this rule aims for a Delaunay triangulation, where no interior
// edge exceeds that angle limit.
//
// Group 0 holds positions and results. `topology::flip_pass` supplies group 1
// and handles conflicts, storage, and mesh updates.
import mopmesh::topology::flip_pass::{
    flip_pass_count, flip_candidate, flip_claim, flip_reserve, flip_commit,
};
import mopmesh::topology::flip::{FLIP_DONE, FLIP_NOT_REQUESTED};
import mopmesh::parallel::grid::flat_index;

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
// One FLIP_* result per edge. An eligible edge that passes the angle test
// without needing a flip receives FLIP_NOT_REQUESTED.
@group(0) @binding(1) var<storage, read_write> status: array<u32>;

// Extra angle, in radians, required beyond 180 degrees. This avoids repeated
// flips when rounding makes the two diagonals nearly equally suitable.
const TOLERANCE: f32 = 1e-3;
const PI: f32 = 3.14159265;

fn angle_at(corner: vec3<f32>, p: vec3<f32>, q: vec3<f32>) -> f32 {
    let u = normalize(p - corner);
    let v = normalize(q - corner);
    return acos(clamp(dot(u, v), -1.0, 1.0));
}

// Test the angles at c and d, opposite the current edge a-b.
// The input is [a, b, c, d] in mesh-wide vertex IDs.
fn should_flip(corners: array<u32, 4>) -> bool {
    let a = positions[corners[0]].xyz;
    let b = positions[corners[1]].xyz;
    let c = positions[corners[2]].xyz;
    let d = positions[corners[3]].xyz;
    return angle_at(c, a, b) + angle_at(d, a, b) > PI + TOLERANCE;
}

// Use each mesh-wide edge ID as its request index; skip excess invocations.
fn request(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    let i = flat_index(id, groups);
    return select(0xffffffffu, i, i < flip_pass_count());
}

@compute @workgroup_size(64)
fn claim(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = request(id, groups);
    if (i == 0xffffffffu) { return; }
    let candidate = flip_candidate(i, i);
    var outcome = candidate.status;
    if (outcome == FLIP_DONE) {
        if (should_flip(candidate.corners)) {
            flip_claim(i, candidate);
        } else {
            outcome = FLIP_NOT_REQUESTED;
        }
    }
    status[i] = outcome;
}

@compute @workgroup_size(64)
fn reserve(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = request(id, groups);
    if (i == 0xffffffffu || status[i] != FLIP_DONE) { return; }
    status[i] = flip_reserve(i);
}

@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = request(id, groups);
    if (i == 0xffffffffu) { return; }
    flip_commit(i);
}
