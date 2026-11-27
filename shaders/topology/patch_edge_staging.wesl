// Finds the faces around each edge of a patch read through `patch_arena`,
// using the workgroup memory of `patch_staging`. Importing this module
// imports both, so the arena and masks are bound at group 1.
//
// Call these functions from a workgroup of 64 invocations that all work on
// the same patch, in uniform control flow:
//
//     let edges = patch_edge_end(lane, r);
//     for (var first = 0u; first < edges; first += STAGED_EDGES) {
//         stage_patch_edges(lane, r, first, edges);
//         // Read staged_edge_faces(first, e) for e in
//         // [first, min(edges, first + STAGED_EDGES)).
//         workgroupBarrier();
//     }

import super::patch::{PatchRecord, EDGE, FACE};
import super::patch_arena::{slot_end, slot_active, face_edge};
import super::patch_staging::{clear_staged_edges, stage_face_edge};

var<workgroup> shared_edge_end: u32;

// Returns slot_end(r, EDGE) as a value every invocation agrees on, so it can
// bound a loop that contains barriers. `lane` is the invocation's
// local_invocation_index.
public fn patch_edge_end(lane: u32, r: PatchRecord) -> u32 {
    if (lane == 0u) { shared_edge_end = slot_end(r, EDGE); }
    return workgroupUniformLoad(&shared_edge_end);
}

// Records the faces of every edge slot in the chunk starting at `first`,
// from all active faces of patch `r`. `edges` is the value returned by
// patch_edge_end. Ends with a barrier, so staged_edge_faces can be read
// right after it returns.
public fn stage_patch_edges(lane: u32, r: PatchRecord, first: u32, edges: u32) {
    clear_staged_edges(lane, edges - first);
    workgroupBarrier();
    let faces = slot_end(r, FACE);
    for (var f = lane; f < faces; f += 64u) {
        if (!slot_active(r, FACE, f)) { continue; }
        for (var i = 0u; i < 3u; i++) {
            let e = face_edge(r, f, i);
            stage_face_edge(first, e.x, e.y, f);
        }
    }
    workgroupBarrier();
}
