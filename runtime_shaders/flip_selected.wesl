// Flips the edges whose cavity requests were selected this round, for
// `EdgeFlipQuery::finish_pass`. It runs after the round's migration, so each
// selected edge's four corners are interior to the patch that owns it,
// which is the request's acting patch; other patches return at once.
//
// `topology::flip_pass` supplies group 1.

import package::shaders::topology::flip_pass::{
    flip_patch_count, flip_patch, flip_edge_end, flip_stage, flip_candidate, flip_request,
    flip_finish,
};
import package::shaders::topology::cavity_rounds::{
    CAVITY_REQUEST_WORDS, CAVITY_SELECTED, CAVITY_PATCH_WORDS, CAVITY_ACTING_WORD,
};
import package::shaders::topology::patch_staging::STAGED_EDGES;
import package::shaders::topology::patch::PATCH_LANES;
import package::shaders::parallel::grid::flat_group;

// Cavity requests, one per mesh-wide edge ID.
@group(0) @binding(0) var<storage, read> requests: array<u32>;
// The round's patch words from `topology::cavity_rounds`.
@group(0) @binding(1) var<storage, read> patch_words: array<u32>;

var<workgroup> acting_word: u32;

@compute @workgroup_size(PATCH_LANES)
fn flip(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= flip_patch_count(lane)) { return; }

    // Only acting patches own selected edges after migration.
    if (lane == 0u) { acting_word = patch_words[CAVITY_PATCH_WORDS * p + CAVITY_ACTING_WORD]; }
    if (workgroupUniformLoad(&acting_word) == 0u) { return; }
    let r = flip_patch(p);
    let edges = flip_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        flip_stage(lane, r, first);
        for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += PATCH_LANES) {
            let c = flip_candidate(r, first, e);
            let selected = c.edge != 0xffffffffu
                && requests[CAVITY_REQUEST_WORDS * c.edge] == CAVITY_SELECTED;
            flip_request(c, selected);
        }
    }
    flip_finish(lane, r, edges);
}
