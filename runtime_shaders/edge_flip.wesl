// Runs the edge requests supplied to `EdgeFlipQuery::encode`.
// Dispatch `claim`, `reserve`, and `commit` in that order to select compatible
// flips, obtain storage, and apply the changes. The final status for each
// request reports success or why it could not flip.
//
// `topology::flip_pass` supplies group 1. Application shaders can use the
// same module and add their own rule for choosing edges.

import package::shaders::topology::flip_pass::{
    flip_pass_count, flip_candidate, flip_claim, flip_reserve, flip_commit,
};
import package::shaders::topology::flip::FLIP_DONE;
import package::shaders::parallel::grid::flat_index;

const INVALID: u32 = 0xffffffffu;

// Mesh-wide edge IDs, one per request. Each ID must appear at most once.
@group(0) @binding(0) var<storage, read> edges: array<u32>;
// One FLIP_* result per request; read final outcomes after commit completes.
@group(0) @binding(1) var<storage, read_write> status: array<u32>;

fn request(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    let i = flat_index(id, groups);
    return select(INVALID, i, i < flip_pass_count());
}

// Stage 1: check each edge and request a flip if it is eligible.
@compute @workgroup_size(64)
fn claim(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = request(id, groups);
    if (i == INVALID) { return; }
    let candidate = flip_candidate(i, edges[i]);
    if (candidate.status == FLIP_DONE) { flip_claim(i, candidate); }
    status[i] = candidate.status;
}

// Stage 2: reserve storage for requests with the highest priority at every corner.
@compute @workgroup_size(64)
fn reserve(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = request(id, groups);
    if (i == INVALID || status[i] != FLIP_DONE) { return; }
    status[i] = flip_reserve(i);
}

// Stage 3: apply every successfully reserved flip.
@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = request(id, groups);
    if (i == INVALID) { return; }
    flip_commit(i);
}
