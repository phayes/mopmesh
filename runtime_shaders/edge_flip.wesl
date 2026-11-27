import package::shaders::topology::patch::{
    PatchRecord, VERTEX, EDGE, FACE, VERTEX_COUNT_WORD, RECORDS_WORD, OWNERS_WORD,
    record_offset, decode_record, owner_offset, used_offset, mask_word, mask_test,
    face_edge_offset, edge_vertices_offset, global_face_offset,
};
import package::shaders::topology::flip::{
    FaceEdges, FlipSlots, flip_quad, flip_writes, free_bit,
    MASK_SET, MASK_CLEAR, FLIP_ARENA_WRITES, FLIP_MASK_UPDATES,
    FLIP_DONE, FLIP_BOUNDARY, FLIP_NOT_INTERIOR, FLIP_CONNECTED,
    FLIP_OUT_OF_CAPACITY, FLIP_NOT_AN_EDGE,
};

// Flips one edge of an editable mesh, or leaves the mesh untouched and
// reports why not. This kernel only uses the public topology::patch and
// topology::flip modules, the same way a downstream kernel would.

const INVALID: u32 = 0xffffffffu;

struct Params { edge: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> arena: array<u32>;
@group(0) @binding(2) var<storage, read_write> masks: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

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
// Returns the lowest inactive slot of `kind` other than `skip`, or INVALID.
fn free_slot(r: PatchRecord, kind: u32, skip: u32) -> u32 {
    let capacity = r.capacity[kind];
    for (var base = 0u; base < capacity; base += 32u) {
        var word = atomicLoad(&masks[mask_word(r.active_masks[kind], base)]);
        if (skip != INVALID && skip / 32u == base / 32u) { word |= 1u << (skip % 32u); }
        let slot = base + free_bit(word);
        if (slot < min(capacity, base + 32u)) { return slot; }
    }
    return INVALID;
}

@compute @workgroup_size(1)
fn main() {
    status[0] = flip(params.edge);
}

fn flip(global_edge: u32) -> u32 {
    if (global_edge >= arena[VERTEX_COUNT_WORD + EDGE]) { return FLIP_NOT_AN_EDGE; }

    // 1. The edge's owning patch and local slot.
    let edge_owners = arena[OWNERS_WORD + EDGE];
    let face_owners = arena[OWNERS_WORD + FACE];
    let owner = owner_offset(edge_owners, global_edge);
    let r = record(arena[owner]);
    let edge = arena[owner + 1u];

    // 2. The two active faces on the edge, and the region they form.
    var faces = array(INVALID, INVALID);
    var found = 0u;
    let used_faces = atomicLoad(&masks[used_offset(r, FACE)]);
    for (var f = 0u; f < used_faces; f++) {
        if (bit(r.active_masks[FACE], f) && any(face_edges(r, f).edges == vec3(edge))) {
            if (found < 2u) { faces[found] = f; }
            found++;
        }
    }
    if (found < 2u) { return FLIP_BOUNDARY; }
    let first = face_edges(r, faces[0]);
    let second = face_edges(r, faces[1]);
    let quad = flip_quad(edge, first, face_edge_vertices(r, first), second, face_edge_vertices(r, second));
    if (quad.status != FLIP_DONE) { return quad.status; }

    // 3. All four corners interior to this patch, and c-d not already an edge.
    let corners = array(quad.a, quad.b, quad.c, quad.d);
    for (var i = 0u; i < 4u; i++) {
        if (!bit(r.interior_mask, corners[i])) { return FLIP_NOT_INTERIOR; }
    }
    let used_edges = atomicLoad(&masks[used_offset(r, EDGE)]);
    for (var e = 0u; e < used_edges; e++) {
        let v = edge_vertices(r, e);
        let joins = (v.x == quad.c && v.y == quad.d) || (v.x == quad.d && v.y == quad.c);
        if (joins && bit(r.active_masks[EDGE], e)) { return FLIP_CONNECTED; }
    }

    // 4. Free slots for the new edge and faces.
    let new_edge = free_slot(r, EDGE, INVALID);
    let new_first = free_slot(r, FACE, INVALID);
    let new_second = free_slot(r, FACE, new_first);
    if (new_edge == INVALID || new_first == INVALID || new_second == INVALID) {
        return FLIP_OUT_OF_CAPACITY;
    }

    // 5. Commit: apply every write. Nothing above wrote to the mesh.
    let slots = FlipSlots(
        edge, faces[0], faces[1], new_edge, new_first, new_second, global_edge,
        arena[global_face_offset(r, faces[0])], arena[global_face_offset(r, faces[1])],
    );
    let writes = flip_writes(r, edge_owners, face_owners, quad, first, second, slots);
    for (var i = 0u; i < FLIP_ARENA_WRITES; i++) {
        let w = writes.arena[i];
        arena[w.offset] = w.value;
    }
    for (var i = 0u; i < FLIP_MASK_UPDATES; i++) {
        let m = writes.masks[i];
        if (m.op == MASK_SET) {
            atomicOr(&masks[m.offset], m.bits);
        } else if (m.op == MASK_CLEAR) {
            atomicAnd(&masks[m.offset], ~m.bits);
        } else {
            atomicMax(&masks[m.offset], m.bits);
        }
    }
    return FLIP_DONE;
}
