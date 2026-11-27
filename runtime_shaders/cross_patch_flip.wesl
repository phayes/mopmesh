// Entry points that connect a flip pass to the cavity rounds, for
// `EdgeFlipQuery::finish_pass`. One invocation per mesh-wide edge ID:
//
// - gather, after the flip pass: every requested edge that crosses patch
//   borders becomes a pending cavity request with the corners the pass saved.
// - merge, after each round's flips of the selected edges: their outcomes
//   go into the batch's statuses, the requests are marked done, and a flip
//   whose new edge already exists frees its corners.
// - finish, after the last round: edges still waiting report why.
import package::shaders::topology::flip::{
    FLIP_NOT_INTERIOR, FLIP_CONNECTED, FLIP_LOST_CLAIM, FLIP_OUT_OF_CAPACITY, FLIP_RETRY,
};
import package::shaders::topology::cavity_rounds::{
    CAVITY_REQUEST_WORDS, CAVITY_NONE, CAVITY_PENDING, CAVITY_SELECTED, CAVITY_LOST,
    CAVITY_FAILED, CAVITY_DONE, CAVITY_PLAN_PENDING,
};
import package::shaders::topology::flip_pass::{FLIP_STATE_WORDS, FLIP_STATE_CORNERS};
import package::shaders::parallel::grid::flat_index;

struct Params { edge_count: u32, _pad0: u32, _pad1: u32, _pad2: u32, }

// The batch's statuses, one per mesh-wide edge ID.
@group(0) @binding(0) var<storage, read_write> statuses: array<u32>;
// The flip pass's saved state: FLIP_STATE_WORDS words per edge, as
// `topology::flip_pass` writes it.
@group(0) @binding(1) var<storage, read> state: array<u32>;
// The statuses written by this round's flips of the selected edges.
@group(0) @binding(2) var<storage, read_write> round_statuses: array<u32>;
// Cavity requests, one per mesh-wide edge ID.
@group(0) @binding(3) var<storage, read_write> requests: array<u32>;
// Nonzero for each mesh vertex used by a selected flip.
@group(0) @binding(4) var<storage, read_write> taken: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
// The cavity rounds' plan buffer; gather counts the pending requests in it.
@group(0) @binding(6) var<storage, read_write> plan: array<atomic<u32>>;

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let e = flat_index(id, groups);
    if (e >= params.edge_count) { return; }
    let at = CAVITY_REQUEST_WORDS * e;
    if (statuses[e] != FLIP_NOT_INTERIOR) {
        requests[at] = CAVITY_NONE;
        return;
    }
    requests[at] = CAVITY_PENDING;
    atomicAdd(&plan[CAVITY_PLAN_PENDING], 1u);
    requests[at + 1u] = 0u;
    for (var k = 0u; k < 4u; k++) {
        requests[at + 2u + k] = state[FLIP_STATE_WORDS * e + FLIP_STATE_CORNERS + k];
    }
}

@compute @workgroup_size(64)
fn merge(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let e = flat_index(id, groups);
    if (e >= params.edge_count) { return; }
    let at = CAVITY_REQUEST_WORDS * e;
    if (requests[at] != CAVITY_SELECTED) { return; }
    requests[at] = CAVITY_DONE;

    // Every selected edge gets an outcome; one without would be a bug, and
    // reports FLIP_RETRY rather than staying FLIP_NOT_INTERIOR.
    let outcome = round_statuses[e];
    statuses[e] = select(outcome, FLIP_RETRY, outcome == 0u);
    if (outcome == FLIP_CONNECTED) {
        for (var k = 0u; k < 4u; k++) { taken[requests[at + 2u + k]] = 0u; }
    }
}

@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let e = flat_index(id, groups);
    if (e >= params.edge_count || statuses[e] != FLIP_NOT_INTERIOR) { return; }
    let stage = requests[CAVITY_REQUEST_WORDS * e];
    if (stage == CAVITY_LOST) {
        statuses[e] = FLIP_LOST_CLAIM;
    } else if (stage == CAVITY_FAILED) {
        statuses[e] = FLIP_OUT_OF_CAPACITY;
    } else {
        statuses[e] = FLIP_RETRY;
    }
}
