// Flips the edges marked in the request mask supplied by
// `EdgeFlipQuery::encode`. Dispatch `claim` with one workgroup per patch,
// then `reserve` and `commit` with one invocation per mesh edge. The status
// of each edge reports success or why it could not flip.
//
// `topology::flip_pass` supplies group 1. Application shaders can use the
// same module with their own rule for choosing edges.

import package::shaders::topology::flip_pass::{
    flip_patch_count, flip_patch, flip_edge_end, flip_stage, flip_candidate, flip_request, flip_select,
    flip_reserve, flip_commit,
};
import package::shaders::topology::patch_staging::STAGED_EDGES;
import package::shaders::parallel::grid::{flat_index, flat_group};

// One bit per mesh-wide edge ID: bit e % 32 of word e / 32 is set when edge
// e is requested.
@group(0) @binding(0) var<storage, read> requested: array<u32>;

fn is_requested(edge: u32) -> bool {
    return ((requested[edge / 32u] >> (edge % 32u)) & 1u) != 0u;
}

// Stage 1: check each requested edge, and claim it if it is eligible.
@compute @workgroup_size(64)
fn claim(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= flip_patch_count(lane)) { return; }
    let r = flip_patch(p);
    let edges = flip_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        flip_stage(lane, r, first);
        for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += 64u) {
            let c = flip_candidate(r, first, e);
            flip_request(c, c.edge != 0xffffffffu && is_requested(c.edge));
        }
    }
    flip_select(lane, r, edges);
}

// Stage 2: reserve storage for the selected flips.
@compute @workgroup_size(64)
fn reserve(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    flip_reserve(flat_index(id, groups));
}

// Stage 3: apply every flip that obtained storage.
@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    flip_commit(flat_index(id, groups));
}
