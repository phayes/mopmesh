// Ready-made loops for local mesh relations inside one patch: each edge's
// diamond, and each vertex's neighbors. They read the patch as it is now,
// so results stay correct after edits. Importing this module imports
// `patch_arena`, which binds the arena and masks at group 1 (read-only).
//
// Call these functions from a workgroup of 64 invocations that all work on
// the same patch, in uniform control flow. Example over vertex neighbors:
//
//     let vertices = patch_slot_end(lane, r, VERTEX);
//     var first = 0u;
//     while (first < vertices) {
//         let end = stage_vertex_neighbors(lane, r, first);
//         for (var v = first + lane; v < end; v += 64u) {
//             if (!slot_active(r, VERTEX, v) || !slot_owned(r, VERTEX, v)) { continue; }
//             for (var it = vv_begin(r, first, v); vv_valid(it); it = vv_next(r, it)) {
//                 let n = vv_vertex(r, it);   // a local vertex slot
//             }
//         }
//         first = end;
//     }
//
// The neighbors of a vertex the patch owns are complete, because the patch
// holds every face around it. Rows of vertices the patch doesn't own may
// miss neighbors held only by other patches.

import super::patch::{PatchRecord, VERTEX, EDGE};
import super::patch_arena::{slot_end, slot_active, face_edge, edge_vertices};
import super::patch_staging::staged_edge_faces;
import super::edge::{INVALID_FACE, INVALID_VERTEX};
import super::local_query::{
    face_edge_index, opposite_edge, opposite_vertex, vv_clear, vv_count_edge, vv_plan, vv_scatter_edge,
    vv_oversized, vv_row, vv_entry,
};

const LANES: u32 = 64u;

var<workgroup> shared_slot_end: u32;

// Returns slot_end(r, kind) as a value every invocation agrees on, so it can
// bound a loop that contains barriers. `lane` is the invocation's
// local_invocation_index.
public fn patch_slot_end(lane: u32, r: PatchRecord, kind: u32) -> u32 {
    if (lane == 0u) { shared_slot_end = slot_end(r, kind); }
    return workgroupUniformLoad(&shared_slot_end);
}

// Returns the diamond of active edge slot `e`: its two endpoints in stored
// order, then the vertex opposite the edge in each adjoining face, all as
// local vertex slots. The opposite vertices follow the face order of
// staged_edge_faces, and a missing boundary face gives INVALID_VERTEX.
// Stage the chunk containing `e` with `patch_edge_staging` first; `first`
// is that chunk's first edge slot.
public fn edge_diamond(r: PatchRecord, first: u32, e: u32) -> vec4<u32> {
    let faces = staged_edge_faces(first, e);
    let ends = edge_vertices(r, e);
    var opposite = vec2(INVALID_VERTEX);
    for (var side = 0u; side < 2u; side++) {
        let f = faces[side];
        if (f == INVALID_FACE) { continue; }
        let slots = vec3(face_edge(r, f, 0u).x, face_edge(r, f, 1u).x, face_edge(r, f, 2u).x);
        let next = face_edge(r, f, opposite_edge(face_edge_index(slots, e)));
        opposite[side] = opposite_vertex(edge_vertices(r, next.x), next.y);
    }
    return vec4(ends, opposite);
}

// Builds the neighbor rows for the chunk of vertex slots starting at
// `first`, and returns the chunk's end: rows for slots [first, end) can be
// read with vv_begin until the next call. Start the next chunk at `end`.
// The result is the same in every invocation. Ends with a barrier.
public fn stage_vertex_neighbors(lane: u32, r: PatchRecord, first: u32) -> u32 {
    // Count each active edge toward its endpoints, lay out the rows, then
    // fill them.
    workgroupBarrier();
    vv_clear(lane);
    workgroupBarrier();
    let edges = slot_end(r, EDGE);
    for (var e = lane; e < edges; e += LANES) {
        if (slot_active(r, EDGE, e)) { vv_count_edge(first, edge_vertices(r, e)); }
    }
    let end = vv_plan(lane, first, slot_end(r, VERTEX));
    for (var e = lane; e < edges; e += LANES) {
        if (slot_active(r, EDGE, e)) { vv_scatter_edge(first, end, edge_vertices(r, e)); }
    }
    workgroupBarrier();
    return end;
}

// A position in one vertex's neighbor row. Create it with vv_begin and
// don't read its fields.
public struct VvIter {
    at: u32,
    end: u32,
    vertex: u32,
    scan: bool,
}

// Moves a scanning iterator to the next active edge at or after `it.at`
// that touches its vertex.
fn vv_seek(r: PatchRecord, it: VvIter) -> VvIter {
    var out = it;
    while (out.at < out.end) {
        if (slot_active(r, EDGE, out.at) && any(edge_vertices(r, out.at) == vec2(out.vertex))) {
            break;
        }
        out.at++;
    }
    return out;
}

// Returns an iterator over the neighbors of vertex slot `v`, which must be
// in the chunk most recently built by stage_vertex_neighbors, starting at
// `first`. Neighbors come in no particular order, one per edge. Rows too
// large for shared memory are found by scanning the patch's edges, which
// takes longer but gives the same neighbors.
public fn vv_begin(r: PatchRecord, first: u32, v: u32) -> VvIter {
    if (vv_oversized(first, v)) {
        return vv_seek(r, VvIter(0u, slot_end(r, EDGE), v, true));
    }
    let row = vv_row(first, v);
    return VvIter(row.start, row.end, v, false);
}

// Returns whether `it` points at a neighbor.
public fn vv_valid(it: VvIter) -> bool {
    return it.at < it.end;
}

// Returns the neighboring local vertex slot at `it`. `it` must be valid.
public fn vv_vertex(r: PatchRecord, it: VvIter) -> u32 {
    if (it.scan) {
        let ends = edge_vertices(r, it.at);
        return select(ends.x, ends.y, ends.x == it.vertex);
    }
    return vv_entry(it.at);
}

// Returns the iterator after `it`, which must be valid.
public fn vv_next(r: PatchRecord, it: VvIter) -> VvIter {
    var out = it;
    out.at++;
    if (out.scan) { return vv_seek(r, out); }
    return out;
}
