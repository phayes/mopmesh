// Flips the edges marked in the request mask supplied by
// `EdgeFlipQuery::encode`, in one dispatch with one workgroup per patch. The
// status of each edge reports success or why it could not flip.
//
// `topology::flip_pass` supplies group 1. Application shaders can use the
// same module with their own rule for choosing edges.

import package::shaders::topology::flip_pass::{
    flip_patch_count, flip_patch, flip_edge_end, flip_stage, flip_candidate, flip_request,
    flip_finish,
};
import package::shaders::topology::patch_staging::STAGED_EDGES;
import package::shaders::topology::patch::PATCH_LANES;
import package::shaders::parallel::grid::flat_group;

// One bit per mesh-wide edge ID: bit e % 32 of word e / 32 is set when edge
// e is requested.
@group(0) @binding(0) var<storage, read> requested: array<u32>;

fn is_requested(edge: u32) -> bool {
    return ((requested[edge / 32u] >> (edge % 32u)) & 1u) != 0u;
}

@compute @workgroup_size(PATCH_LANES)
fn flip(
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
        for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += PATCH_LANES) {
            let c = flip_candidate(r, first, e);
            flip_request(c, c.edge != 0xffffffffu && is_requested(c.edge));
        }
    }
    flip_finish(lane, r, edges);
}
