// Read-only access to mesh patches and their vertices, edges and faces.
// Importing this module binds the patch arena at group 1, binding 0, and the
// slot masks at group 1, binding 1. Group 0 is available for your own buffers.
// To write to the arena or masks, use `patch` with your own bindings instead.

import super::patch::{
    PatchRecord, PATCH_COUNT_WORD, RECORDS_WORD, record_offset, decode_record, used_offset,
    mask_word, mask_test, face_edge_offset, edge_vertices_offset, global_vertex_offset,
    global_edge_offset, global_face_offset, VERTEX_COUNT_WORD, OWNERS_WORD, owner_offset,
};

@group(1) @binding(0) var<storage, read> arena: array<u32>;
@group(1) @binding(1) var<storage, read> masks: array<u32>;

// Returns the number of patches in the arena.
public fn patch_count() -> u32 {
    return arena[PATCH_COUNT_WORD];
}

// Returns the storage locations and slot capacities for patch `p`.
// `p` must be less than patch_count().
public fn patch_record(p: u32) -> PatchRecord {
    let at = record_offset(arena[RECORDS_WORD], p);
    var w: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        w[i] = arena[at + i];
    }
    return decode_record(w);
}

// Returns the exclusive upper bound for slots of `kind` to visit.
// Use slot_active to skip empty slots below this bound.
public fn slot_end(r: PatchRecord, kind: u32) -> u32 {
    return masks[used_offset(r, kind)];
}

// Returns whether this slot currently contains a vertex, edge or face.
public fn slot_active(r: PatchRecord, kind: u32, slot: u32) -> bool {
    return mask_test(masks[mask_word(r.active_masks[kind], slot)], slot);
}

// Returns whether this patch owns the element in this slot. Elements can
// appear in several patches; ownership lets you process each one only once.
public fn slot_owned(r: PatchRecord, kind: u32, slot: u32) -> bool {
    return mask_test(masks[mask_word(r.owned_masks[kind], slot)], slot);
}

// Returns edge `i` (0, 1 or 2) of an active triangular face. The first value
// is its local edge slot; the second is 0 if the face follows the edge's
// vertex order, or 1 if it reverses that order.
public fn face_edge(r: PatchRecord, face: u32, i: u32) -> vec2<u32> {
    let at = face_edge_offset(r, face, i);
    return vec2(arena[at], arena[at + 1u]);
}

// Returns the two vertex slots that form an active edge. Both slots are
// local to this patch; use global_vertex to get their mesh-wide IDs.
public fn edge_vertices(r: PatchRecord, edge: u32) -> vec2<u32> {
    let at = edge_vertices_offset(r, edge);
    return vec2(arena[at], arena[at + 1u]);
}

public fn global_vertex(r: PatchRecord, slot: u32) -> u32 {
    return arena[global_vertex_offset(r, slot)];
}
public fn global_edge(r: PatchRecord, slot: u32) -> u32 {
    return arena[global_edge_offset(r, slot)];
}
public fn global_face(r: PatchRecord, slot: u32) -> u32 {
    return arena[global_face_offset(r, slot)];
}

// Returns the total number of vertices, edges or faces in the mesh.
// Pass VERTEX, EDGE or FACE as `kind`; copies in neighboring patches are not counted.
public fn element_count(kind: u32) -> u32 {
    return arena[VERTEX_COUNT_WORD + kind];
}

// Finds the unique owning copy of an element identified by its mesh-wide ID.
// Returns (patch ID, slot within that patch). Pass VERTEX, EDGE or FACE as `kind`
// and an ID below element_count(kind). Use only with editable arenas; static
// arenas do not store the owner tables this lookup requires.
public fn owner(kind: u32, global: u32) -> vec2<u32> {
    let at = owner_offset(arena[OWNERS_WORD + kind], global);
    return vec2(arena[at], arena[at + 1u]);
}
