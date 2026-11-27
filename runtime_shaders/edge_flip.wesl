import package::shaders::topology::patch::{
    PatchRecord, EDGE, FACE, VERTEX_COUNT_WORD, RECORDS_WORD, OWNERS_WORD,
    record_offset, decode_record, owner_offset, used_offset, mask_word, mask_test,
    face_edge_offset, edge_vertices_offset, global_vertex_offset, global_face_offset,
};
import package::shaders::topology::flip::{
    FaceEdges, FlipQuad, FlipSlots, flip_quad, flip_writes, free_bit, flip_priority,
    MASK_SET, MASK_CLEAR, FLIP_ARENA_WRITES, FLIP_MASK_UPDATES,
    FLIP_DONE, FLIP_BOUNDARY, FLIP_NOT_INTERIOR, FLIP_CONNECTED, FLIP_LOST_CLAIM,
    FLIP_OUT_OF_CAPACITY, FLIP_NOT_AN_EDGE,
};
import package::shaders::parallel::grid::flat_index;

// Flips requested edges in three ordered GPU dispatches: `claim` selects
// non-overlapping flips, `reserve` allocates their replacement storage, and
// `commit` updates the mesh. Each request reports success or a reason it
// could not flip. Rejected requests make no mesh changes themselves.
// See "Flipping many edges at once" in topology::flip to adapt this kernel
// using the public shader helpers.

const INVALID: u32 = 0xffffffffu;

struct Params { count: u32, seed: u32, _pad0: u32, _pad1: u32, }
@group(0) @binding(0) var<uniform> params: Params;
// Mesh-wide edge IDs to flip, in request order. List each edge once.
@group(0) @binding(1) var<storage, read> edges: array<u32>;
@group(0) @binding(2) var<storage, read_write> arena: array<u32>;
@group(0) @binding(3) var<storage, read_write> masks: array<atomic<u32>>;
// Highest request priority at each mesh vertex. Clear to zero before `claim`.
@group(0) @binding(4) var<storage, read_write> claims: array<atomic<u32>>;
// Saved patch-local storage indices for each request; see SLOT_WORDS below.
@group(0) @binding(5) var<storage, read_write> slots: array<u32>;
@group(0) @binding(6) var<storage, read_write> status: array<u32>;

// Five words per request: two original face slots, one reserved edge slot,
// and two reserved face slots. Keep this layout in sync with the host buffer.
const SLOT_WORDS: u32 = 5u;

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

// The patch storing an edge and the edge's storage index within that patch.
struct Located { r: PatchRecord, edge: u32, }
fn locate(global_edge: u32) -> Located {
    let owner = owner_offset(arena[OWNERS_WORD + EDGE], global_edge);
    return Located(record(arena[owner]), arena[owner + 1u]);
}

// Finds the four corners around an edge using its two saved faces.
fn quad_of(at: Located, first: FaceEdges, second: FaceEdges) -> FlipQuad {
    return flip_quad(
        at.edge, first, face_edge_vertices(at.r, first), second, face_edge_vertices(at.r, second),
    );
}

// Converts the four patch-local corner indices to mesh-wide vertex IDs.
fn corners(r: PatchRecord, quad: FlipQuad) -> array<u32, 4> {
    return array(
        arena[global_vertex_offset(r, quad.a)],
        arena[global_vertex_offset(r, quad.b)],
        arena[global_vertex_offset(r, quad.c)],
        arena[global_vertex_offset(r, quad.d)],
    );
}

// Reserves an available edge or face storage slot by setting its active bit.
// Returns INVALID if the search cannot reserve a slot within patch capacity.
fn reserve_slot(r: PatchRecord, kind: u32) -> u32 {
    let capacity = r.capacity[kind];
    for (var base = 0u; base < capacity; base += 32u) {
        let at = mask_word(r.active_masks[kind], base);
        var word = atomicLoad(&masks[at]);
        // Limit retries to 32 per mask word when other requests take slots.
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

fn candidate(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    let i = flat_index(id, groups);
    return select(INVALID, i, i < params.count);
}

// Stage 1: validate requests and record their priorities at each corner.

@compute @workgroup_size(64)
fn claim(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = candidate(id, groups);
    if (i == INVALID) { return; }
    status[i] = check(i);
}

fn check(i: u32) -> u32 {
    let global_edge = edges[i];
    if (global_edge >= arena[VERTEX_COUNT_WORD + EDGE]) { return FLIP_NOT_AN_EDGE; }
    let at = locate(global_edge);
    let r = at.r;

    // Find the two current triangles sharing the requested edge.
    var faces = array(INVALID, INVALID);
    var found = 0u;
    let used_faces = atomicLoad(&masks[used_offset(r, FACE)]);
    for (var f = 0u; f < used_faces; f++) {
        if (bit(r.active_masks[FACE], f) && any(face_edges(r, f).edges == vec3(at.edge))) {
            if (found < 2u) { faces[found] = f; }
            found++;
        }
    }
    if (found < 2u) { return FLIP_BOUNDARY; }
    let quad = quad_of(at, face_edges(r, faces[0]), face_edges(r, faces[1]));
    if (quad.status != FLIP_DONE) { return quad.status; }

    // Reject flips that reach another patch or would duplicate an existing edge.
    let local = array(quad.a, quad.b, quad.c, quad.d);
    for (var k = 0u; k < 4u; k++) {
        if (!bit(r.interior_mask, local[k])) { return FLIP_NOT_INTERIOR; }
    }
    let used_edges = atomicLoad(&masks[used_offset(r, EDGE)]);
    for (var e = 0u; e < used_edges; e++) {
        let v = edge_vertices(r, e);
        let joins = (v.x == quad.c && v.y == quad.d) || (v.x == quad.d && v.y == quad.c);
        if (joins && bit(r.active_masks[EDGE], e)) { return FLIP_CONNECTED; }
    }

    // Save the face slots for later stages and claim each corner by priority.
    slots[i * SLOT_WORDS] = faces[0];
    slots[i * SLOT_WORDS + 1u] = faces[1];
    let priority = flip_priority(i, params.seed);
    let global = corners(r, quad);
    for (var k = 0u; k < 4u; k++) { atomicMax(&claims[global[k]], priority); }
    return FLIP_DONE;
}

// Stage 2: select requests that won every corner and reserve replacement storage.

@compute @workgroup_size(64)
fn reserve(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = candidate(id, groups);
    if (i == INVALID || status[i] != FLIP_DONE) { return; }
    status[i] = reserve_for(i);
}

fn reserve_for(i: u32) -> u32 {
    let at = locate(edges[i]);
    let r = at.r;
    let base = i * SLOT_WORDS;

    // Proceed only if this request has the highest priority at all four corners.
    let quad = quad_of(at, face_edges(r, slots[base]), face_edges(r, slots[base + 1u]));
    let priority = flip_priority(i, params.seed);
    let global = corners(r, quad);
    for (var k = 0u; k < 4u; k++) {
        if (atomicLoad(&claims[global[k]]) != priority) { return FLIP_LOST_CLAIM; }
    }

    // Reserve one edge slot and two face slots. Release any partial reservation
    // on failure so a rejected request does not consume storage.
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
    slots[base + 2u] = new_edge;
    slots[base + 3u] = new_first;
    slots[base + 4u] = new_second;
    return FLIP_DONE;
}

// Stage 3: update the mesh for requests that passed both earlier stages.

@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = candidate(id, groups);
    if (i == INVALID || status[i] != FLIP_DONE) { return; }
    let global_edge = edges[i];
    let at = locate(global_edge);
    let r = at.r;
    let base = i * SLOT_WORDS;

    // Read the original faces and IDs before replacing their storage.
    let first_slot = slots[base];
    let second_slot = slots[base + 1u];
    let first = face_edges(r, first_slot);
    let second = face_edges(r, second_slot);
    let quad = quad_of(at, first, second);
    let picked = FlipSlots(
        at.edge, first_slot, second_slot, slots[base + 2u], slots[base + 3u], slots[base + 4u],
        global_edge, arena[global_face_offset(r, first_slot)], arena[global_face_offset(r, second_slot)],
    );
    let writes = flip_writes(
        r, arena[OWNERS_WORD + EDGE], arena[OWNERS_WORD + FACE], quad, first, second, picked,
    );

    // Apply the complete update, using atomics for shared mask words.
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
}
