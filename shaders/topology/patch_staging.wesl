// Finds the faces touching each edge of a patch using memory shared by a
// group of 64 shader invocations. Process up to STAGED_EDGES local edge slots
// at a time, starting at `first`. No buffer bindings are added; shared memory
// use is STAGED_EDGES * 8 bytes per workgroup.
//
// For each chunk, all invocations must follow the same sequence:
// 1. Call clear_staged_edges, then workgroupBarrier().
// 2. Divide the active faces among invocations and call stage_face_edge for
//    each face's three edges, then workgroupBarrier().
// 3. Read staged_edge_faces for the chunk, then workgroupBarrier() before
//    clearing shared memory for the next chunk.
//
// Each edge may have at most two faces. If it has two, they must traverse
// the edge in opposite directions.

import super::edge::INVALID_FACE;

public const STAGED_EDGES: u32 = 1024u;
const LANES: u32 = 64u;

var<workgroup> edge_faces: array<u32, 2048>;

// Resets the chunk so each edge starts with no faces. All 64 invocations
// must call this with their local_invocation_index as `lane` and the same
// `count`. Clears min(count, STAGED_EDGES) entries; read only those entries.
public fn clear_staged_edges(lane: u32, count: u32) {
    for (var i = lane; i < 2u * min(count, STAGED_EDGES); i += LANES) {
        edge_faces[i] = INVALID_FACE;
    }
}

// Records a face touching an edge, using patch-local slot numbers.
// Set `reversed` to 0 when the face follows the edge's vertex order, or 1
// when it reverses that order. Ignores edges outside the chunk starting at
// `first` and spanning STAGED_EDGES slots.
public fn stage_face_edge(first: u32, edge: u32, reversed: u32, face: u32) {
    if (edge < first || edge - first >= STAGED_EDGES) { return; }
    edge_faces[2u * (edge - first) + reversed] = face;
}

// Returns the local face slots touching `edge`, with the smaller slot first.
// A missing face is INVALID_FACE, as on the outer boundary of a mesh.
// `edge` must be in the cleared portion of the current chunk, and staging
// must be complete before reading.
public fn staged_edge_faces(first: u32, edge: u32) -> vec2<u32> {
    let a = edge_faces[2u * (edge - first)];
    let b = edge_faces[2u * (edge - first) + 1u];
    return vec2(min(a, b), max(a, b));
}
