// Flip multiple edges safely from application compute shaders.
// An edge flip replaces the shared edge of two triangles with the other
// diagonal of their four-vertex region.
//
// Use `EdgeFlipQuery::pass_layout` for group 1 in the pipeline layout and
// `EdgeFlipQuery::begin_pass` to create its bindings. Group 0 is available
// for application data. Do not also import `topology::patch_arena`: it
// uses group 1 too.
//
// Number requests from zero to `flip_pass_count() - 1`, with each mesh-wide
// edge ID appearing at most once. Run these three dispatches in order:
//
// 1. Claim: call `flip_candidate(index, edge)`. When its status is FLIP_DONE,
//    use its corners to decide whether to call `flip_claim(index, candidate)`.
// 2. Reserve: call `flip_reserve(index)` once for each request.
// 3. Commit: call `flip_commit(index)` once for each request.
//
// No CPU readback is needed between stages. Flips that share any of their
// four vertices compete; at most one can succeed. A request that returns
// FLIP_LOST_CLAIM can be retried in a later batch.

import super::patch::{
    PatchRecord, EDGE, FACE, VERTEX_COUNT_WORD, RECORDS_WORD, OWNERS_WORD,
    record_offset, decode_record, owner_offset, used_offset, mask_word, mask_test,
    face_edge_offset, edge_vertices_offset, global_vertex_offset, global_face_offset,
};
import super::flip::{
    FaceEdges, FlipQuad, FlipSlots, flip_quad, flip_writes, free_bit, flip_priority,
    MASK_SET, MASK_CLEAR, FLIP_ARENA_WRITES, FLIP_MASK_UPDATES,
    FLIP_NOT_REQUESTED, FLIP_DONE, FLIP_BOUNDARY, FLIP_NOT_INTERIOR, FLIP_CONNECTED,
    FLIP_LOST_CLAIM, FLIP_OUT_OF_CAPACITY, FLIP_NOT_AN_EDGE,
};

const INVALID: u32 = 0xffffffffu;

struct PassParams { count: u32, seed: u32, _pad0: u32, _pad1: u32, }
@group(1) @binding(0) var<storage, read_write> arena: array<u32>;
@group(1) @binding(1) var<storage, read_write> masks: array<atomic<u32>>;
// One priority per mesh vertex; each batch starts with all priorities at zero.
@group(1) @binding(2) var<storage, read_write> claims: array<atomic<u32>>;
// Temporary request data, with SLOT_WORDS zero-initialized words per request.
@group(1) @binding(3) var<storage, read_write> slots: array<u32>;
@group(1) @binding(4) var<uniform> pass_params: PassParams;

// Words per request: one edge ID, two original face slots, three replacement
// slots, and one stage marker.
const SLOT_WORDS: u32 = 7u;
const SLOT_EDGE: u32 = 0u;
const SLOT_FACES: u32 = 1u;
const SLOT_NEW: u32 = 3u;
const SLOT_STAGE: u32 = 6u;
const STAGE_CLAIMED: u32 = 1u;
const STAGE_RESERVED: u32 = 2u;

// The result of checking an edge with `flip_candidate`.
// `status == FLIP_DONE` means eligible to claim, not already flipped.
// Other statuses explain why the edge was rejected; only read `corners`
// when the status is FLIP_DONE.
//
// `corners` contains mesh-wide vertex IDs [a, b, c, d]. The current edge
// joins a to b; the replacement would join c to d. The two triangles are
// oriented a -> b -> c and b -> a -> d. Pass the candidate unchanged to
// `flip_claim`; its remaining fields are not application inputs.
public struct FlipCandidate {
    status: u32,
    corners: array<u32, 4>,
    edge: u32,
    first_face: u32,
    second_face: u32,
}

// The number of requests in this batch. Guard every stage so that request
// indices are less than this value, including invocations in a partial workgroup.
public fn flip_pass_count() -> u32 {
    return pass_params.count;
}

fn record(patch_id: u32) -> PatchRecord {
    let at = record_offset(arena[RECORDS_WORD], patch_id);
    var words: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { words[i] = arena[at + i]; }
    return decode_record(words);
}
fn bit(mask: u32, slot: u32) -> bool {
    return mask_test(atomicLoad(&masks[mask_word(mask, slot)]), slot);
}
fn face_edges(r: PatchRecord, slot: u32) -> FaceEdges {
    var face: FaceEdges;
    for (var i = 0u; i < 3u; i++) {
        let at = face_edge_offset(r, slot, i);
        face.edges[i] = arena[at];
        face.reversed[i] = arena[at + 1u];
    }
    return face;
}
fn edge_vertices(r: PatchRecord, slot: u32) -> vec2<u32> {
    let at = edge_vertices_offset(r, slot);
    return vec2(arena[at], arena[at + 1u]);
}
fn face_edge_vertices(r: PatchRecord, face: FaceEdges) -> array<vec2<u32>, 3> {
    return array(
        edge_vertices(r, face.edges.x),
        edge_vertices(r, face.edges.y),
        edge_vertices(r, face.edges.z),
    );
}

// An edge location: its owning patch and its slot within that patch.
struct Located { r: PatchRecord, edge: u32, }
fn locate(global_edge: u32) -> Located {
    let owner = owner_offset(arena[OWNERS_WORD + EDGE], global_edge);
    return Located(record(arena[owner]), arena[owner + 1u]);
}

// Builds the four-vertex region formed by the two triangles sharing an edge.
fn quad_of(at: Located, first: FaceEdges, second: FaceEdges) -> FlipQuad {
    return flip_quad(
        at.edge, first, face_edge_vertices(at.r, first), second, face_edge_vertices(at.r, second),
    );
}

// Translates corner indices within a patch into mesh-wide vertex IDs.
fn corners(r: PatchRecord, quad: FlipQuad) -> array<u32, 4> {
    return array(
        arena[global_vertex_offset(r, quad.a)],
        arena[global_vertex_offset(r, quad.b)],
        arena[global_vertex_offset(r, quad.c)],
        arena[global_vertex_offset(r, quad.d)],
    );
}

// Checks `global_edge` for request `index` during the claim dispatch.
// The edge must join two valid triangles, all four vertices must be interior
// to one patch, and the replacement edge must not already exist. Returns
// a status and, on success, the four vertices for an application rule.
// This check does not change the mesh or claim the request. It does not
// check vertex positions, competing requests, or available storage.
public fn flip_candidate(index: u32, global_edge: u32) -> FlipCandidate {
    var out: FlipCandidate;
    out.status = FLIP_NOT_AN_EDGE;
    if (global_edge >= arena[VERTEX_COUNT_WORD + EDGE]) { return out; }
    let at = locate(global_edge);
    let r = at.r;

    // Collect the active triangles that contain this edge.
    var faces = array(INVALID, INVALID);
    var found = 0u;
    let used_faces = atomicLoad(&masks[used_offset(r, FACE)]);
    for (var f = 0u; f < used_faces; f++) {
        if (bit(r.active_masks[FACE], f) && any(face_edges(r, f).edges == vec3(at.edge))) {
            if (found < 2u) { faces[found] = f; }
            found++;
        }
    }
    out.status = FLIP_BOUNDARY;
    if (found < 2u) { return out; }
    let quad = quad_of(at, face_edges(r, faces[0]), face_edges(r, faces[1]));
    out.status = quad.status;
    if (quad.status != FLIP_DONE) { return out; }

    // All affected vertices must stay within the patch, and the new edge must be unique.
    let local = array(quad.a, quad.b, quad.c, quad.d);
    out.status = FLIP_NOT_INTERIOR;
    for (var k = 0u; k < 4u; k++) {
        if (!bit(r.interior_mask, local[k])) { return out; }
    }
    out.status = FLIP_CONNECTED;
    let used_edges = atomicLoad(&masks[used_offset(r, EDGE)]);
    for (var e = 0u; e < used_edges; e++) {
        let v = edge_vertices(r, e);
        let joins = (v.x == quad.c && v.y == quad.d) || (v.x == quad.d && v.y == quad.c);
        if (joins && bit(r.active_masks[EDGE], e)) { return out; }
    }

    out.status = FLIP_DONE;
    out.corners = corners(r, quad);
    out.edge = global_edge;
    out.first_face = faces[0];
    out.second_face = faces[1];
    return out;
}

// Requests a flip during the claim dispatch. Pass the unchanged result of
// `flip_candidate(index, ...)`, and only when its status is FLIP_DONE.
// Call at most once per request. A claim can still lose to another request
// or fail to obtain storage in the reserve stage.
public fn flip_claim(index: u32, candidate: FlipCandidate) {
    let base = index * SLOT_WORDS;
    slots[base + SLOT_EDGE] = candidate.edge;
    slots[base + SLOT_FACES] = candidate.first_face;
    slots[base + SLOT_FACES + 1u] = candidate.second_face;
    slots[base + SLOT_STAGE] = STAGE_CLAIMED;
    let priority = flip_priority(index, pass_params.seed);
    for (var k = 0u; k < 4u; k++) { atomicMax(&claims[candidate.corners[k]], priority); }
}

// Claims a free edge or face slot by atomically setting its active bit.
// Returns INVALID if no slot can be acquired within the patch capacity.
fn reserve_slot(r: PatchRecord, kind: u32) -> u32 {
    let capacity = r.capacity[kind];
    for (var base = 0u; base < capacity; base += 32u) {
        let at = mask_word(r.active_masks[kind], base);
        var word = atomicLoad(&masks[at]);
        // Bound the search when other requests compete for slots in this mask word.
        for (var attempt = 0u; attempt < 32u; attempt++) {
            let slot = base + free_bit(word);
            if (slot >= min(capacity, base + 32u)) { break; }
            let mask = 1u << (slot % 32u);
            let before = atomicOr(&masks[at], mask);
            if ((before & mask) == 0u) { return slot; }
            word = before | mask;
        }
    }
    return INVALID;
}
fn release_slot(r: PatchRecord, kind: u32, slot: u32) {
    atomicAnd(&masks[mask_word(r.active_masks[kind], slot)], ~(1u << (slot % 32u)));
}

// Prepares a claimed request for commit. Call once per request in a separate
// dispatch after all claims finish. Returns FLIP_DONE when the flip is ready,
// FLIP_LOST_CLAIM when another request won a shared vertex,
// FLIP_OUT_OF_CAPACITY when replacement storage could not be acquired, or
// FLIP_NOT_REQUESTED when the request was not claimed. The edge has not yet
// been flipped, even when this returns FLIP_DONE.
public fn flip_reserve(index: u32) -> u32 {
    let base = index * SLOT_WORDS;
    if (slots[base + SLOT_STAGE] != STAGE_CLAIMED) { return FLIP_NOT_REQUESTED; }
    let at = locate(slots[base + SLOT_EDGE]);
    let r = at.r;

    // A request can proceed only if it won every affected vertex.
    let first = face_edges(r, slots[base + SLOT_FACES]);
    let second = face_edges(r, slots[base + SLOT_FACES + 1u]);
    let priority = flip_priority(index, pass_params.seed);
    let global = corners(r, quad_of(at, first, second));
    for (var k = 0u; k < 4u; k++) {
        if (atomicLoad(&claims[global[k]]) != priority) { return FLIP_LOST_CLAIM; }
    }

    // Acquire all three replacement slots before committing. On failure,
    // release any slots acquired by this request.
    let new_edge = reserve_slot(r, EDGE);
    if (new_edge == INVALID) { return FLIP_OUT_OF_CAPACITY; }
    let new_first = reserve_slot(r, FACE);
    if (new_first == INVALID) {
        release_slot(r, EDGE, new_edge);
        return FLIP_OUT_OF_CAPACITY;
    }
    let new_second = reserve_slot(r, FACE);
    if (new_second == INVALID) {
        release_slot(r, EDGE, new_edge);
        release_slot(r, FACE, new_first);
        return FLIP_OUT_OF_CAPACITY;
    }
    slots[base + SLOT_NEW] = new_edge;
    slots[base + SLOT_NEW + 1u] = new_first;
    slots[base + SLOT_NEW + 2u] = new_second;
    slots[base + SLOT_STAGE] = STAGE_RESERVED;
    return FLIP_DONE;
}

// Applies a reserved flip and returns true, or returns false if the request
// was not reserved. Call once per request in a separate dispatch after all
// reservations finish. A successful flip preserves the mesh-wide edge ID
// and the two face IDs.
public fn flip_commit(index: u32) -> bool {
    let base = index * SLOT_WORDS;
    if (slots[base + SLOT_STAGE] != STAGE_RESERVED) { return false; }
    let global_edge = slots[base + SLOT_EDGE];
    let at = locate(global_edge);
    let r = at.r;

    // Gather the original triangles and their IDs for the replacement.
    let first_slot = slots[base + SLOT_FACES];
    let second_slot = slots[base + SLOT_FACES + 1u];
    let first = face_edges(r, first_slot);
    let second = face_edges(r, second_slot);
    let picked = FlipSlots(
        at.edge, first_slot, second_slot,
        slots[base + SLOT_NEW], slots[base + SLOT_NEW + 1u], slots[base + SLOT_NEW + 2u],
        global_edge, arena[global_face_offset(r, first_slot)], arena[global_face_offset(r, second_slot)],
    );
    let writes = flip_writes(
        r, arena[OWNERS_WORD + EDGE], arena[OWNERS_WORD + FACE],
        quad_of(at, first, second), first, second, picked,
    );

    // Write the replacement data. Update masks atomically because other flips
    // may change different bits in the same word.
    for (var k = 0u; k < FLIP_ARENA_WRITES; k++) {
        let w = writes.arena[k];
        arena[w.offset] = w.value;
    }
    for (var k = 0u; k < FLIP_MASK_UPDATES; k++) {
        let m = writes.masks[k];
        if (m.op == MASK_SET) {
            atomicOr(&masks[m.offset], m.bits);
        } else if (m.op == MASK_CLEAR) {
            atomicAnd(&masks[m.offset], ~m.bits);
        } else {
            atomicMax(&masks[m.offset], m.bits);
        }
    }
    return true;
}
