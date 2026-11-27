// Helpers for reading a patch arena, the buffer that stores mesh sections
// called patches. Each patch has local slots for its vertices, edges and faces;
// separate masks record which slots are active and which copies it owns.
// Use these helpers with your own buffer bindings, or import `patch_arena`
// for read-only access through group 1.
//
// Offsets count u32 words, not bytes. Slot numbers refer to one patch.

// Selects vertices in fields and functions indexed by element kind.
public const VERTEX: u32 = 0u;
// Selects edges in fields and functions indexed by element kind.
public const EDGE: u32 = 1u;
// Selects triangular faces in fields and functions indexed by element kind.
public const FACE: u32 = 2u;

// Index of the arena word that stores the mesh's vertex count. The edge and
// face counts follow at VERTEX_COUNT_WORD + EDGE and VERTEX_COUNT_WORD + FACE.
public const VERTEX_COUNT_WORD: u32 = 0u;
// Index of the arena word that stores the number of patches.
public const PATCH_COUNT_WORD: u32 = 3u;
// Index of the arena word that stores the first patch record's word offset.
public const RECORDS_WORD: u32 = 4u;
// Number of words in each patch record.
public const RECORD_WORDS: u32 = 16u;
// Index of the arena word that stores the word offset of the vertex owner
// table. The edge and face tables' offsets follow at OWNERS_WORD + EDGE and
// OWNERS_WORD + FACE. Only editable arenas have owner tables.
public const OWNERS_WORD: u32 = 5u;

// Returns the arena word offset of a mesh-wide element's owner entry. Pass
// the owner table's offset, read from OWNERS_WORD + kind. The entry is two
// words: the owning patch, then the element's local slot in that patch.
public fn owner_offset(table: u32, global: u32) -> u32 {
    return table + 2u * global;
}

// Describes the space reserved for one patch and where to find its data.
// Connectivity and global IDs use arena offsets; masks and used-slot bounds
// use mask-buffer offsets. Index vector fields with VERTEX, EDGE or FACE.
public struct PatchRecord {
    capacity: vec3<u32>,
    vertex_ids: u32,
    ev: u32,
    fe: u32,
    edge_ids: u32,
    face_ids: u32,
    active_masks: vec3<u32>,
    owned_masks: vec3<u32>,
    interior_mask: u32,
    used: u32,
}

// Returns the arena word offset for patch `p`. Pass the value stored at
// RECORDS_WORD as `records`.
public fn record_offset(records: u32, p: u32) -> u32 {
    return records + RECORD_WORDS * p;
}

// Reads a patch description from the RECORD_WORDS words at its record offset.
public fn decode_record(w: array<u32, 16>) -> PatchRecord {
    return PatchRecord(
        vec3(w[0], w[1], w[2]), w[3], w[4], w[5], w[6], w[7],
        vec3(w[8], w[9], w[10]), vec3(w[11], w[12], w[13]), w[14], w[15],
    );
}

// Returns the mask-buffer offset of the exclusive slot bound for `kind`.
// Active slots are below this bound; slots below it may still be inactive.
public fn used_offset(r: PatchRecord, kind: u32) -> u32 {
    return r.used + kind;
}

// Returns the mask-buffer word offset containing the bit for `slot`.
// `mask` is the word offset where the mask starts.
public fn mask_word(mask: u32, slot: u32) -> u32 {
    return mask + slot / 32u;
}

// Returns whether the bit for `slot` is set in the word read at mask_word.
public fn mask_test(word: u32, slot: u32) -> bool {
    return ((word >> (slot % 32u)) & 1u) != 0u;
}

// Returns the arena word offset for edge `i` of a triangular face (0, 1 or 2).
// The two words there hold the local edge slot and its direction in the face:
// 0 follows the edge's vertex order; 1 reverses it.
public fn face_edge_offset(r: PatchRecord, face: u32, i: u32) -> u32 {
    return r.fe + 6u * face + 2u * i;
}

// Returns the arena word offset of the two vertex slots that form an edge.
// Both vertex slots are local to this patch.
public fn edge_vertices_offset(r: PatchRecord, edge: u32) -> u32 {
    return r.ev + 2u * edge;
}

// Returns the arena word offset of a local vertex slot's mesh-wide ID.
public fn global_vertex_offset(r: PatchRecord, slot: u32) -> u32 {
    return r.vertex_ids + slot;
}
// Returns the arena word offset of a local edge slot's mesh-wide ID.
public fn global_edge_offset(r: PatchRecord, slot: u32) -> u32 {
    return r.edge_ids + slot;
}
// Returns the arena word offset of a local face slot's mesh-wide ID.
public fn global_face_offset(r: PatchRecord, slot: u32) -> u32 {
    return r.face_ids + slot;
}
