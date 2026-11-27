// Edge flips on triangle meshes, as binding-free helpers.
//
// Flipping the edge shared by two triangles replaces it with the other
// diagonal of the four-sided region they form. If the two triangles are
// a -> b -> c and b -> a -> d, the flip removes edge a-b and adds edge c-d,
// giving a -> d -> c and d -> b -> c. Every vertex keeps its position, and
// both new triangles keep the winding of the old ones.
//
// The caller loads each face's edges and each edge's endpoints from its own
// storage and passes the values in. Nothing here reads or writes buffers.
// To apply a flip to an editable patch arena, see `flip_writes` below.

import super::patch::{
    PatchRecord, EDGE, FACE, used_offset, mask_word, face_edge_offset,
    edge_vertices_offset, global_edge_offset, global_face_offset, owner_offset,
};

// Outcome codes for an edge flip. Callers may store these as status words.
public const FLIP_NOT_REQUESTED: u32 = 0u;
public const FLIP_DONE: u32 = 1u;
// The edge has only one face, so there is no second diagonal.
public const FLIP_BOUNDARY: u32 = 2u;
// Some face around one of the four vertices belongs to another patch.
public const FLIP_NOT_INTERIOR: u32 = 3u;
// Vertices c and d are already joined by an edge; flipping would duplicate it.
public const FLIP_CONNECTED: u32 = 4u;
// The two faces don't form a proper four-sided region: c equals d, or the
// faces run the shared edge in the same direction.
public const FLIP_DEGENERATE: u32 = 5u;
// Another flip claimed one of the same vertices first.
public const FLIP_LOST_CLAIM: u32 = 6u;
// The patch had no free slot for the new edge or faces.
public const FLIP_OUT_OF_CAPACITY: u32 = 7u;
// The requested edge ID is not an edge of the mesh.
public const FLIP_NOT_AN_EDGE: u32 = 8u;

// A face's three edges in winding order: edge IDs, and for each a flag that
// is 1 when the face runs the edge against its stored vertex order.
public struct FaceEdges {
    edges: vec3<u32>,
    reversed: vec3<u32>,
}

// The four-sided region around an edge, found by `flip_quad`.
// `first` runs a -> b -> c and `second` runs b -> a -> d. `first_slot` and
// `second_slot` are the positions (0, 1 or 2) of the shared edge in each
// face's FaceEdges. `status` is FLIP_DONE when the region is valid, or
// FLIP_DEGENERATE.
public struct FlipQuad {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    first_slot: u32,
    second_slot: u32,
    status: u32,
}

// Finds the region that flipping `edge` would change. `first` and `second`
// are the two faces on `edge`; `first_vertices` and `second_vertices` are the
// stored endpoints of each face's three edges, in FaceEdges order. Returns
// FLIP_DEGENERATE in `status` if either face doesn't contain `edge`, if both
// run it in the same direction, or if c equals d.
public fn flip_quad(
    edge: u32,
    first: FaceEdges,
    first_vertices: array<vec2<u32>, 3>,
    second: FaceEdges,
    second_vertices: array<vec2<u32>, 3>,
) -> FlipQuad {
    var quad = FlipQuad(0u, 0u, 0u, 0u, 3u, 3u, FLIP_DEGENERATE);

    // Find the shared edge in each face.
    for (var i = 0u; i < 3u; i++) {
        if (first.edges[i] == edge) { quad.first_slot = i; }
        if (second.edges[i] == edge) { quad.second_slot = i; }
    }
    if (quad.first_slot == 3u || quad.second_slot == 3u) { return quad; }
    let i = quad.first_slot;
    let j = quad.second_slot;
    if (first.reversed[i] == second.reversed[j]) { return quad; }

    // In `first` the shared edge runs a -> b and the edge after next starts at c.
    // In `second` it runs b -> a and the edge after next starts at d.
    let ab = first_vertices[i];
    let forward = first.reversed[i] == 0u;
    quad.a = select(ab.y, ab.x, forward);
    quad.b = select(ab.x, ab.y, forward);
    let ca = first_vertices[(i + 2u) % 3u];
    quad.c = select(ca.y, ca.x, first.reversed[(i + 2u) % 3u] == 0u);
    let db = second_vertices[(j + 2u) % 3u];
    quad.d = select(db.y, db.x, second.reversed[(j + 2u) % 3u] == 0u);
    if (quad.c != quad.d) { quad.status = FLIP_DONE; }
    return quad;
}

// The two faces that replace `first` and `second` after a flip, for a valid
// `quad`. `new_edge` is the ID of the new edge, whose stored vertex order
// must be (quad.c, quad.d). Returns a -> d -> c, then d -> b -> c.
public fn flipped_faces(
    quad: FlipQuad,
    first: FaceEdges,
    second: FaceEdges,
    new_edge: u32,
) -> array<FaceEdges, 2> {
    let i = quad.first_slot;
    let j = quad.second_slot;
    let bc = (i + 1u) % 3u;
    let ca = (i + 2u) % 3u;
    let ad = (j + 1u) % 3u;
    let db = (j + 2u) % 3u;
    // a -> d (from second), d -> c (new edge, against its stored order), c -> a (from first).
    let adc = FaceEdges(
        vec3(second.edges[ad], new_edge, first.edges[ca]),
        vec3(second.reversed[ad], 1u, first.reversed[ca]),
    );
    // d -> b (from second), b -> c (from first), c -> d (new edge, in its stored order).
    let dbc = FaceEdges(
        vec3(second.edges[db], first.edges[bc], new_edge),
        vec3(second.reversed[db], first.reversed[bc], 0u),
    );
    return array(adc, dbc);
}

// ---------------------------------------------------------------------------
// Committing a flip to an editable patch arena
//
// `flip_writes` turns a checked flip into a list of word writes for the
// arena and bit updates for the mask buffer. The library never touches your
// buffers: your kernel applies the list to its own bindings. That lets one
// kernel test its own condition, pick an edge and flip it in a single
// dispatch. A complete kernel is runtime_shaders/edge_flip.wesl. The steps:
//
//   1. Find the edge's owning patch through the edge owner table, and decode
//      that patch's record with `decode_record` (topology::patch).
//   2. Find the two active faces on the edge, then call `flip_quad`. Stop
//      unless its status is FLIP_DONE.
//   3. Check that all four corners are interior to the patch (interior mask),
//      and that no active edge already joins c and d. Otherwise stop with
//      FLIP_NOT_INTERIOR or FLIP_CONNECTED.
//   4. Pick one inactive edge slot and two inactive face slots, using
//      `free_bit` on active-mask words. Stop with FLIP_OUT_OF_CAPACITY if
//      there aren't enough.
//   5. Call `flip_writes` and apply every entry:
//        for (var i = 0u; i < FLIP_ARENA_WRITES; i++) {
//            let w = writes.arena[i];
//            arena[w.offset] = w.value;
//        }
//        for (var i = 0u; i < FLIP_MASK_UPDATES; i++) {
//            let m = writes.masks[i];
//            if (m.op == MASK_SET) { atomicOr(&masks[m.offset], m.bits); }
//            else if (m.op == MASK_CLEAR) { atomicAnd(&masks[m.offset], ~m.bits); }
//            else { atomicMax(&masks[m.offset], m.bits); }
//        }
//
// Apply the writes with these safeguards:
// - Complete all checks and slot selection before writing anything. Apply
//   the entire list exactly once; partial or repeated writes can corrupt
//   the patch, the part of the mesh that stores this edge and its faces.
// - The entries do not overlap, so they may be applied in any order.
// - Plain stores are sufficient for arena entries only when no other GPU
//   invocation accesses the same slots during the writes.
// - Use atomic operations for mask entries. Masks pack the state of several
//   slots into one word, which other flips may also update.
// - The steps above support one flip per patch per dispatch. For several
//   flips in the same patch, use the three stages below.

// ---------------------------------------------------------------------------
// Flipping many edges at once
//
// A batch can flip several edges safely by choosing flips that share no
// corner vertices and reserving separate storage for each. Run `claim`,
// `reserve`, and `commit` as three ordered GPU dispatches; there is no need
// to read results back to the CPU between them. Import `topology::flip_pass`
// to use these stages with bind group 1, or follow the steps below if your
// shader needs to manage its own bindings and buffers.
//
// Give every request a unique index. For each request, allocate a status
// word and five slot indices: two for the original faces, followed by one
// for the replacement edge and two for the replacement faces. Also allocate
// one atomic claim word per mesh vertex and clear those words to zero.
//
// 1. Claim: perform checks 1-3 above without changing the mesh. For each
//    valid request, save its face slots and use `atomicMax` to write
//    `flip_priority(index, seed)` to all four corners' claim words, indexed
//    by global vertex ID. Set its status to FLIP_DONE. At this stage that
//    status means eligible to continue, not that the flip has happened.
//    Store the reason for rejection for any invalid request.
// 2. Reserve: for each eligible request, rebuild its quad from the saved
//    face slots. It must have the highest priority at all four corners;
//    otherwise set FLIP_LOST_CLAIM. For each winner, reserve one inactive
//    edge slot and two inactive face slots. Use `free_bit` to find a slot,
//    check the patch capacity, and set its active bit with `atomicOr`.
//    The slot is yours only if the returned word had that bit clear. If
//    any reservation fails, release only this request's reservations and
//    set FLIP_OUT_OF_CAPACITY. Do not clear any other bits in this stage.
// 3. Commit: for each request still at FLIP_DONE, reload its saved faces,
//    call `flip_quad` and `flip_writes` with its reserved slots, and apply
//    every write. After this stage, FLIP_DONE means the edge was flipped.
//
// Keep these constraints when adapting the example:
// - Unique request indices give unique priorities for a fixed seed, so
//   each corner has at most one winner.
// - In reserve and commit, read only the request's own owner entry, saved
//   face rows, and their edge rows. Do not scan the patch for faces or
//   edges: reservations and commits change data elsewhere in the patch.
// - Reserve replacement storage before freeing the original slots. A flip
//   needs one spare edge slot and two spare face slots; slots freed in this
//   batch become available to later batches.
// - Requests rejected with FLIP_LOST_CLAIM or FLIP_OUT_OF_CAPACITY make no
//   mesh changes themselves and may be retried. Change the seed for later
//   batches to vary priorities. This does not guarantee a particular
//   request will win, or make space in a patch with no spare slots.
// - List each edge once. At most one duplicate request can flip it; retrying
//   another copy after that success can flip the edge back.

// Returns a repeatable, shuffled priority for a request. Higher values win
// shared corners. With a fixed `seed`, distinct request indices have
// distinct priorities; changing the seed can change which requests win.
public fn flip_priority(index: u32, seed: u32) -> u32 {
    var x = index ^ seed;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// One arena word to store.
public struct ArenaWrite {
    offset: u32,
    value: u32,
}

// How a MaskUpdate changes its word.
// MASK_SET: atomicOr with `bits`. MASK_CLEAR: atomicAnd with `~bits`.
// MASK_MAX: atomicMax with `bits`, for used-slot counts.
public const MASK_SET: u32 = 0u;
public const MASK_CLEAR: u32 = 1u;
public const MASK_MAX: u32 = 2u;

// One mask-buffer word to update atomically.
public struct MaskUpdate {
    offset: u32,
    bits: u32,
    op: u32,
}

public const FLIP_ARENA_WRITES: u32 = 37u;
public const FLIP_MASK_UPDATES: u32 = 14u;

// Everything one flip writes. Apply every entry; see the rules above.
public struct FlipWrites {
    arena: array<ArenaWrite, 37>,
    masks: array<MaskUpdate, 14>,
}

// The local slots a flip frees and fills, and the global IDs it carries over.
// `edge`, `first` and `second` are the old edge and its two faces, in the
// order passed to `flip_quad`. `new_edge`, `new_first` and `new_second` are
// inactive slots picked by the caller, all different. `global_*` are the
// global IDs stored at the old slots. The new edge and faces keep them.
public struct FlipSlots {
    edge: u32,
    first: u32,
    second: u32,
    new_edge: u32,
    new_first: u32,
    new_second: u32,
    global_edge: u32,
    global_first: u32,
    global_second: u32,
}

// Returns the lowest clear bit (0-31) of an active-mask word, or 32 if every
// bit is set. Bits past the patch's capacity may read as clear, so compare
// the slot with the record's capacity before using it.
public fn free_bit(word: u32) -> u32 {
    let bit = firstTrailingBit(~word);
    return select(bit, 32u, bit == 0xffffffffu);
}

// Returns the writes that commit a checked flip in patch `r`.
// `edge_owners` and `face_owners` are the arena offsets of the edge and face
// owner tables (the words at OWNERS_WORD + EDGE and OWNERS_WORD + FACE).
// `quad`, `first` and `second` are as passed to and returned by `flip_quad`.
public fn flip_writes(
    r: PatchRecord,
    edge_owners: u32,
    face_owners: u32,
    quad: FlipQuad,
    first: FaceEdges,
    second: FaceEdges,
    slots: FlipSlots,
) -> FlipWrites {
    const INVALID: u32 = 0xffffffffu;
    var out: FlipWrites;
    var n = 0u;
    let new_faces = flipped_faces(quad, first, second, slots.new_edge);
    let old_faces = array(slots.first, slots.second);
    let new_slots = array(slots.new_first, slots.new_second);
    let globals = array(slots.global_first, slots.global_second);

    // Clear the old edge's and faces' rows and IDs.
    let old_ev = edge_vertices_offset(r, slots.edge);
    out.arena[n] = ArenaWrite(old_ev, INVALID); n++;
    out.arena[n] = ArenaWrite(old_ev + 1u, INVALID); n++;
    out.arena[n] = ArenaWrite(global_edge_offset(r, slots.edge), INVALID); n++;
    for (var f = 0u; f < 2u; f++) {
        for (var i = 0u; i < 3u; i++) {
            let at = face_edge_offset(r, old_faces[f], i);
            out.arena[n] = ArenaWrite(at, INVALID); n++;
            out.arena[n] = ArenaWrite(at + 1u, INVALID); n++;
        }
        out.arena[n] = ArenaWrite(global_face_offset(r, old_faces[f]), INVALID); n++;
    }

    // Store the new edge c-d and point its owner entry at the new slot.
    let new_ev = edge_vertices_offset(r, slots.new_edge);
    out.arena[n] = ArenaWrite(new_ev, quad.c); n++;
    out.arena[n] = ArenaWrite(new_ev + 1u, quad.d); n++;
    out.arena[n] = ArenaWrite(global_edge_offset(r, slots.new_edge), slots.global_edge); n++;
    out.arena[n] = ArenaWrite(owner_offset(edge_owners, slots.global_edge) + 1u, slots.new_edge); n++;

    // Store the new faces under the old faces' global IDs.
    for (var f = 0u; f < 2u; f++) {
        for (var i = 0u; i < 3u; i++) {
            let at = face_edge_offset(r, new_slots[f], i);
            out.arena[n] = ArenaWrite(at, new_faces[f].edges[i]); n++;
            out.arena[n] = ArenaWrite(at + 1u, new_faces[f].reversed[i]); n++;
        }
        out.arena[n] = ArenaWrite(global_face_offset(r, new_slots[f]), globals[f]); n++;
        out.arena[n] = ArenaWrite(owner_offset(face_owners, globals[f]) + 1u, new_slots[f]); n++;
    }

    // Swap the active and owned bits from the old slots to the new ones, and
    // raise the used-slot counts to cover the new slots.
    var m = 0u;
    let kinds = array(EDGE, FACE, FACE);
    let freed = array(slots.edge, slots.first, slots.second);
    let filled = array(slots.new_edge, slots.new_first, slots.new_second);
    for (var k = 0u; k < 3u; k++) {
        let kind_masks = array(r.active_masks[kinds[k]], r.owned_masks[kinds[k]]);
        for (var j = 0u; j < 2u; j++) {
            out.masks[m] = MaskUpdate(mask_word(kind_masks[j], freed[k]), 1u << (freed[k] % 32u), MASK_CLEAR); m++;
            out.masks[m] = MaskUpdate(mask_word(kind_masks[j], filled[k]), 1u << (filled[k] % 32u), MASK_SET); m++;
        }
    }
    out.masks[m] = MaskUpdate(used_offset(r, EDGE), slots.new_edge + 1u, MASK_MAX); m++;
    out.masks[m] = MaskUpdate(used_offset(r, FACE), max(slots.new_first, slots.new_second) + 1u, MASK_MAX);
    return out;
}
