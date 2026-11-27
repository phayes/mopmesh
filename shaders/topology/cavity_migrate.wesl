// Carries out the edits that `topology::cavity_rounds` selected: each acting
// patch takes over the faces around its requests' corners, as RXMesh's
// migration does, and every patch involved then drops the copies it no
// longer needs.
//
// Run these stages as separate dispatches, in this order, right after a
// round's `cavity_resolve`, on the same vertex records and patch words:
//
//   1. `cavity_ring`: each corner's owner marks the vertices of the faces
//      around the corner, the cavity faces, and flags every patch owning a
//      cavity face, edge or vertex as involved. Only involved patches take
//      part in the stages below; the rest return at once.
//   2. `cavity_claim`: each marked vertex's owner adds the faces around the
//      vertex, with their edges and vertices, to the acting patch's maps,
//      counting the copies the acting patch lacks.
//   3. `cavity_preflight`: an acting patch without room for those copies
//      fails, and nothing below changes it.
//   4. `cavity_conclude` from `topology::cavity_rounds`, one invocation per
//      request: requests of failed acting patches become CAVITY_FAILED.
//   5. `cavity_fill_vertices`, `cavity_fill_edges`, `cavity_fill_faces`:
//      the same owners copy the claimed elements into free slots of the
//      acting patch.
//   6. `cavity_transfer`: owners hand the cavity faces, and their edges and
//      vertices, to the acting patch.
//   7. `cavity_cleanup`: every involved patch keeps exactly the copies its
//      owned faces need, rebuilds its maps, and recomputes its interior
//      vertices. Patches that are not involved are unchanged by the round,
//      so they keep what the previous cleanup left.
//
// Every stage except conclude runs one workgroup of PATCH_LANES
// invocations per patch, from uniform control flow.
//
// Importing this module binds, at group 1: the arena (binding 0), the masks
// (binding 1), the round's vertex records (binding 2) and patch words
// (binding 3) from `topology::cavity_rounds`, a scratch buffer as long as
// the masks (binding 4), and the maps (binding 7, from
// `topology::patch_map`). All are read-write. Clear the scratch buffer to
// zero before each round.

import super::patch::{
    PatchRecord, VERTEX, EDGE, FACE, PATCH_LANES, PATCH_COUNT_WORD, RECORDS_WORD, OWNERS_WORD,
    record_offset, decode_record, used_offset, mask_word, face_edge_offset, edge_vertices_offset,
    global_vertex_offset, global_edge_offset, global_face_offset, owner_offset,
};
import super::patch_map::{
    map_clear, map_set, map_find, map_claim, map_take, MAP_CLAIM_NEW, MAP_CLAIM_FULL,
};
import super::directed_edge::directed_start;
import super::flip::free_bit;
import super::cavity_rounds::{
    CAVITY_PATCH_WORDS, CAVITY_FAILED_WORD, CAVITY_NEW_WORDS, CAVITY_RING_WORD,
    CAVITY_ACTING_WORD, CAVITY_INVOLVED_WORD, CAVITY_VERTEX_WORDS, CAVITY_CORNER, CAVITY_RING,
};

const INVALID: u32 = 0xffffffffu;

@group(1) @binding(0) var<storage, read_write> arena: array<u32>;
@group(1) @binding(1) var<storage, read_write> masks: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> vertex_records: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> patch_words: array<atomic<u32>>;
// Per-slot bits for this round, at the same offsets as the masks: bits at a
// patch's active-mask offsets mark the copies cleanup keeps, and bits at its
// owned-edge mask offset the edges transfer has handed over.
@group(1) @binding(4) var<storage, read_write> scratch: array<atomic<u32>>;

var<workgroup> shared_word: u32;
// Bit k set if `cavity_cleanup` dropped a copy of kind k in this patch.
var<workgroup> dropped_kinds: atomic<u32>;

// Shares a value from the first invocation with the whole workgroup.
fn broadcast(lane: u32, value: u32) -> u32 {
    if (lane == 0u) { shared_word = value; }
    return workgroupUniformLoad(&shared_word);
}

fn record(p: u32) -> PatchRecord {
    let at = record_offset(arena[RECORDS_WORD], p);
    var words: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { words[i] = arena[at + i]; }
    return decode_record(words);
}

// The bit for `slot` within its mask word.
fn bit(slot: u32) -> u32 {
    return 1u << (slot % 32u);
}

fn mask_bit(mask: u32, slot: u32) -> bool {
    return (atomicLoad(&masks[mask_word(mask, slot)]) & bit(slot)) != 0u;
}

fn scratch_bit(mask: u32, slot: u32) -> bool {
    return (atomicLoad(&scratch[mask_word(mask, slot)]) & bit(slot)) != 0u;
}

fn used(r: PatchRecord, kind: u32) -> u32 {
    return atomicLoad(&masks[used_offset(r, kind)]);
}

fn edge_slot(r: PatchRecord, face: u32, i: u32) -> u32 {
    return arena[face_edge_offset(r, face, i)];
}

// The three vertex slots of active face `face`, in corner order.
fn face_vertex_slots(r: PatchRecord, face: u32) -> array<u32, 3> {
    var out: array<u32, 3>;
    for (var i = 0u; i < 3u; i++) {
        let at = face_edge_offset(r, face, i);
        let ev = edge_vertices_offset(r, arena[at]);
        out[i] = directed_start(vec2(arena[ev], arena[ev + 1u]), arena[at + 1u] != 0u);
    }
    return out;
}

fn global_id(r: PatchRecord, kind: u32, slot: u32) -> u32 {
    let ids = array(r.vertex_ids, r.edge_ids, r.face_ids);
    return arena[ids[kind] + slot];
}

// Returns the number of patches, the same in every invocation of the
// workgroup. Call it from every invocation, in uniform control flow.
public fn cavity_migrate_patch_count(lane: u32) -> u32 {
    return broadcast(lane, arena[PATCH_COUNT_WORD]);
}

fn patch_word(p: u32, field: u32) -> u32 {
    return CAVITY_PATCH_WORDS * p + field;
}

// Whether patch `p`'s word `field` (CAVITY_RING_WORD, CAVITY_ACTING_WORD or
// CAVITY_INVOLVED_WORD) is set this round.
fn flagged(p: u32, field: u32) -> bool {
    return atomicLoad(&patch_words[patch_word(p, field)]) != 0u;
}

// The patch that owns element `global` of `kind`.
fn owner_patch(kind: u32, global: u32) -> u32 {
    return arena[owner_offset(arena[OWNERS_WORD + kind], global)];
}

fn flag_involved(p: u32) {
    atomicStore(&patch_words[patch_word(p, CAVITY_INVOLVED_WORD)], 1u);
}

fn failed(p: u32) -> bool {
    return atomicLoad(&patch_words[patch_word(p, CAVITY_FAILED_WORD)]) != 0u;
}

// One more than an acting patch, or 0: see CAVITY_CORNER and CAVITY_RING.
fn vertex_tag(global: u32, field: u32) -> u32 {
    return atomicLoad(&vertex_records[CAVITY_VERTEX_WORDS * global + field]);
}

fn owned(r: PatchRecord, kind: u32, slot: u32) -> bool {
    return mask_bit(r.owned_masks[kind], slot);
}

// The face's three edge slots.
fn face_edge_slots(r: PatchRecord, face: u32) -> array<u32, 3> {
    return array(edge_slot(r, face, 0u), edge_slot(r, face, 1u), edge_slot(r, face, 2u));
}

// Stage 1 for patch `p`. Also flags the owners of every cavity face, its
// edges and its vertices as involved.
public fn cavity_ring(lane: u32, p: u32) {
    if (!flagged(p, CAVITY_RING_WORD)) { return; }
    let r = record(p);
    for (var f = lane; f < used(r, FACE); f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        var cavity = false;
        for (var i = 0u; i < 3u; i++) {
            if (!owned(r, VERTEX, vertices[i])) { continue; }
            let tag = vertex_tag(global_id(r, VERTEX, vertices[i]), CAVITY_CORNER);
            if (tag == 0u) { continue; }
            cavity = true;
            for (var j = 0u; j < 3u; j++) {
                let w = global_id(r, VERTEX, vertices[j]);
                atomicStore(&vertex_records[CAVITY_VERTEX_WORDS * w + CAVITY_RING], tag);
            }
        }
        if (!cavity) { continue; }

        // The face's owners: the patches the later stages change.
        flag_involved(owner_patch(FACE, global_id(r, FACE, f)));
        let edges = face_edge_slots(r, f);
        for (var i = 0u; i < 3u; i++) {
            flag_involved(owner_patch(VERTEX, global_id(r, VERTEX, vertices[i])));
            flag_involved(owner_patch(EDGE, global_id(r, EDGE, edges[i])));
        }
    }
}

// Returns the acting patch whose copies face `f` of patch `p` supplies: one
// whose cavity faces have a vertex of `f` that `p` owns, other than `p`
// itself, or INVALID.
fn supplied_actor(r: PatchRecord, p: u32, vertices: array<u32, 3>) -> u32 {
    for (var i = 0u; i < 3u; i++) {
        if (!owned(r, VERTEX, vertices[i])) { continue; }
        let tag = vertex_tag(global_id(r, VERTEX, vertices[i]), CAVITY_RING);
        if (tag != 0u && tag - 1u != p) { return tag - 1u; }
    }
    return INVALID;
}

fn claim(actor: u32, kind: u32, global: u32) {
    let result = map_claim(actor, kind, global);
    if (result == MAP_CLAIM_NEW) {
        atomicAdd(&patch_words[patch_word(actor, CAVITY_NEW_WORDS + kind)], 1u);
    } else if (result == MAP_CLAIM_FULL) {
        atomicStore(&patch_words[patch_word(actor, CAVITY_FAILED_WORD)], 1u);
    }
}

// Stage 2 for patch `p`.
public fn cavity_claim(lane: u32, p: u32) {
    if (!flagged(p, CAVITY_INVOLVED_WORD)) { return; }
    let r = record(p);
    for (var f = lane; f < used(r, FACE); f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        let actor = supplied_actor(r, p, vertices);
        if (actor == INVALID) { continue; }
        let edges = face_edge_slots(r, f);
        for (var i = 0u; i < 3u; i++) {
            claim(actor, VERTEX, global_id(r, VERTEX, vertices[i]));
            claim(actor, EDGE, global_id(r, EDGE, edges[i]));
        }
        claim(actor, FACE, global_id(r, FACE, f));
    }
}

// Stage 3 for patch `p`. Only the first invocation does any work.
public fn cavity_preflight(lane: u32, p: u32) {
    if (lane != 0u || !flagged(p, CAVITY_ACTING_WORD)) { return; }
    let r = record(p);
    for (var k = 0u; k < 3u; k++) {
        let needed = atomicLoad(&patch_words[patch_word(p, CAVITY_NEW_WORDS + k)]);
        if (needed == 0u) { continue; }
        var in_use = 0u;
        for (var w = 0u; w < (r.capacity[k] + 31u) / 32u; w++) {
            in_use += countOneBits(atomicLoad(&masks[r.active_masks[k] + w]));
        }
        if (needed > r.capacity[k] - in_use) {
            atomicStore(&patch_words[patch_word(p, CAVITY_FAILED_WORD)], 1u);
        }
    }
}

// Marks a free slot of `kind` active in patch `r` and returns it, or
// INVALID if there is none.
fn allocate(r: PatchRecord, kind: u32) -> u32 {
    for (var w = 0u; w < (r.capacity[kind] + 31u) / 32u; w++) {
        let word = r.active_masks[kind] + w;
        loop {
            let b = free_bit(atomicLoad(&masks[word]));
            let slot = 32u * w + b;
            if (b == 32u || slot >= r.capacity[kind]) { break; }
            // The slot is ours only if its bit was clear before our write.
            if ((atomicOr(&masks[word], 1u << b) & (1u << b)) == 0u) {
                atomicMax(&masks[used_offset(r, kind)], slot + 1u);
                return slot;
            }
        }
    }
    return INVALID;
}

// Copies the element of `kind` in `slot` of patch `r` into acting patch
// `actor`, if this invocation is the one to do it.
fn fill(r: PatchRecord, kind: u32, slot: u32, actor: u32) {
    let global = global_id(r, kind, slot);
    if (!map_take(actor, kind, global)) { return; }
    let ra = record(actor);
    let s = allocate(ra, kind);
    if (s == INVALID) { return; }

    // Write the row, with slots local to the acting patch.
    if (kind == VERTEX) {
        arena[global_vertex_offset(ra, s)] = global;
    } else if (kind == EDGE) {
        let src = edge_vertices_offset(r, slot);
        let dst = edge_vertices_offset(ra, s);
        for (var i = 0u; i < 2u; i++) {
            arena[dst + i] = map_find(actor, VERTEX, global_id(r, VERTEX, arena[src + i]));
        }
        arena[global_edge_offset(ra, s)] = global;
    } else {
        for (var i = 0u; i < 3u; i++) {
            let src = face_edge_offset(r, slot, i);
            let dst = face_edge_offset(ra, s, i);
            arena[dst] = map_find(actor, EDGE, global_id(r, EDGE, arena[src]));
            arena[dst + 1u] = arena[src + 1u];
        }
        arena[global_face_offset(ra, s)] = global;
    }
    map_set(actor, kind, global, s);
}

// Stage 5 for patch `p`, once per kind: VERTEX, then EDGE, then FACE, each
// in its own dispatch.
public fn cavity_fill(lane: u32, p: u32, kind: u32) {
    if (!flagged(p, CAVITY_INVOLVED_WORD)) { return; }
    let r = record(p);
    for (var f = lane; f < used(r, FACE); f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        let actor = supplied_actor(r, p, vertices);
        if (actor == INVALID || failed(actor)) { continue; }
        if (kind == FACE) {
            fill(r, FACE, f, actor);
            continue;
        }
        var slots = face_edge_slots(r, f);
        if (kind == VERTEX) { slots = vertices; }
        for (var i = 0u; i < 3u; i++) {
            fill(r, kind, slots[i], actor);
        }
    }
}

// Hands the element of `kind` in `slot` of patch `p` to `actor`, which must
// already hold a copy.
fn hand_over(r: PatchRecord, kind: u32, slot: u32, actor: u32) {
    let global = global_id(r, kind, slot);
    let ra = record(actor);
    let s = map_find(actor, kind, global);
    atomicAnd(&masks[mask_word(r.owned_masks[kind], slot)], ~bit(slot));
    atomicOr(&masks[mask_word(ra.owned_masks[kind], s)], bit(s));
    let at = owner_offset(arena[OWNERS_WORD + kind], global);
    arena[at] = actor;
    arena[at + 1u] = s;
}

// Stage 6 for patch `p`.
public fn cavity_transfer(lane: u32, p: u32) {
    if (!flagged(p, CAVITY_INVOLVED_WORD)) { return; }
    let r = record(p);

    // Cavity faces this patch owns, and the edges of every cavity face it
    // holds that it owns, each edge once.
    for (var f = lane; f < used(r, FACE); f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        var tag = 0u;
        for (var i = 0u; i < 3u; i++) {
            tag = max(tag, vertex_tag(global_id(r, VERTEX, vertices[i]), CAVITY_CORNER));
        }
        if (tag == 0u || tag - 1u == p || failed(tag - 1u)) { continue; }
        if (owned(r, FACE, f)) { hand_over(r, FACE, f, tag - 1u); }
        let edges = face_edge_slots(r, f);
        for (var i = 0u; i < 3u; i++) {
            let e = edges[i];
            if (!owned(r, EDGE, e)) { continue; }
            let seen = atomicOr(&scratch[mask_word(r.owned_masks[EDGE], e)], bit(e));
            if ((seen & bit(e)) == 0u) { hand_over(r, EDGE, e, tag - 1u); }
        }
    }

    // Vertices of cavity faces that this patch owns.
    for (var v = lane; v < used(r, VERTEX); v += PATCH_LANES) {
        if (!mask_bit(r.active_masks[VERTEX], v) || !owned(r, VERTEX, v)) { continue; }
        let tag = vertex_tag(global_id(r, VERTEX, v), CAVITY_RING);
        if (tag == 0u || tag - 1u == p || failed(tag - 1u)) { continue; }
        hand_over(r, VERTEX, v, tag - 1u);
    }
}

// Clears the row, mesh-wide ID, active bit and owned bit of `slot`.
fn deactivate(r: PatchRecord, kind: u32, slot: u32) {
    if (kind == VERTEX) {
        arena[global_vertex_offset(r, slot)] = INVALID;
    } else if (kind == EDGE) {
        let ev = edge_vertices_offset(r, slot);
        arena[ev] = INVALID;
        arena[ev + 1u] = INVALID;
        arena[global_edge_offset(r, slot)] = INVALID;
    } else {
        for (var i = 0u; i < 6u; i++) { arena[face_edge_offset(r, slot, 0u) + i] = INVALID; }
        arena[global_face_offset(r, slot)] = INVALID;
    }
    let clear = ~bit(slot);
    atomicAnd(&masks[mask_word(r.active_masks[kind], slot)], clear);
    atomicAnd(&masks[mask_word(r.owned_masks[kind], slot)], clear);
}

// Stage 7 for patch `p`. Does nothing unless the patch is involved this
// round (CAVITY_INVOLVED_WORD). Afterwards the patch holds exactly its
// owned elements and every face at a vertex of a face it owns, with their
// edges and vertices; its maps record exactly its active slots; and its
// interior bits mark the vertices whose faces it all owns.
public fn cavity_cleanup(lane: u32, p: u32) {
    let involved = atomicLoad(&patch_words[patch_word(p, CAVITY_INVOLVED_WORD)]);
    if (broadcast(lane, involved) == 0u) { return; }
    let r = record(p);
    let faces = used(r, FACE);

    // Clear the interior bits, this patch's keep bits and the dropped kinds.
    if (lane == 0u) { atomicStore(&dropped_kinds, 0u); }
    workgroupBarrier();
    for (var k = 0u; k < 3u; k++) {
        let words = (r.capacity[k] + 31u) / 32u;
        for (var w = lane; w < words; w += PATCH_LANES) {
            atomicStore(&scratch[r.active_masks[k] + w], 0u);
            if (k == VERTEX) { atomicStore(&masks[r.interior_mask + w], 0u); }
        }
    }
    storageBarrier();

    // Mark the vertices of owned faces, in the interior mask for now.
    for (var f = lane; f < faces; f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f) || !mask_bit(r.owned_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        for (var i = 0u; i < 3u; i++) {
            atomicOr(&masks[mask_word(r.interior_mask, vertices[i])], bit(vertices[i]));
        }
    }
    storageBarrier();

    // Keep owned elements, and every face at a marked vertex with its edges
    // and vertices.
    for (var k = 0u; k < 3u; k++) {
        for (var s = lane; s < used(r, k); s += PATCH_LANES) {
            if (mask_bit(r.active_masks[k], s) && mask_bit(r.owned_masks[k], s)) {
                atomicOr(&scratch[mask_word(r.active_masks[k], s)], bit(s));
            }
        }
    }
    for (var f = lane; f < faces; f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        var needed = false;
        for (var i = 0u; i < 3u; i++) {
            needed = needed || mask_bit(r.interior_mask, vertices[i]);
        }
        if (!needed) { continue; }
        atomicOr(&scratch[mask_word(r.active_masks[FACE], f)], bit(f));
        for (var i = 0u; i < 3u; i++) {
            let e = edge_slot(r, f, i);
            atomicOr(&scratch[mask_word(r.active_masks[EDGE], e)], bit(e));
            atomicOr(&scratch[mask_word(r.active_masks[VERTEX], vertices[i])], bit(vertices[i]));
        }
    }
    storageBarrier();

    // A marked vertex is interior unless a face around it is owned elsewhere.
    for (var f = lane; f < faces; f += PATCH_LANES) {
        if (!mask_bit(r.active_masks[FACE], f) || mask_bit(r.owned_masks[FACE], f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        for (var i = 0u; i < 3u; i++) {
            atomicAnd(&masks[mask_word(r.interior_mask, vertices[i])], ~bit(vertices[i]));
        }
    }
    storageBarrier();

    // Drop every active copy not kept, noting the kinds dropped.
    for (var k = 0u; k < 3u; k++) {
        for (var s = lane; s < used(r, k); s += PATCH_LANES) {
            if (mask_bit(r.active_masks[k], s) && !scratch_bit(r.active_masks[k], s)) {
                deactivate(r, k, s);
                atomicOr(&dropped_kinds, 1u << k);
            }
        }
    }
    workgroupBarrier();

    // A map still records exactly the active slots unless a copy of its
    // kind was dropped, or this is a failed acting patch, whose maps hold
    // claims that were never filled. Empty the others for a rebuild.
    let failed_actor = flagged(p, CAVITY_ACTING_WORD) && failed(p);
    let rebuild = broadcast(lane, atomicLoad(&dropped_kinds) | select(0u, 7u, failed_actor));
    for (var k = 0u; k < 3u; k++) {
        if ((rebuild & (1u << k)) != 0u) { map_clear(p, k, lane, PATCH_LANES); }
    }
    storageBarrier();

    // Rebuild those maps from the active slots.
    for (var k = 0u; k < 3u; k++) {
        if ((rebuild & (1u << k)) == 0u) { continue; }
        for (var s = lane; s < used(r, k); s += PATCH_LANES) {
            if (mask_bit(r.active_masks[k], s)) {
                map_set(p, k, global_id(r, k, s), s);
            }
        }
    }
}
