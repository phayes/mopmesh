// A downstream kernel that flips an edge only when the flip shortens it,
// in one dispatch, using MOPMesh's public topology modules. It chooses its
// own bindings: the editable arena and masks from `GpuEditableMesh`, plus
// its own positions and request.
import mopmesh::topology::patch::{
    PatchRecord, EDGE, FACE, VERTEX_COUNT_WORD, RECORDS_WORD, OWNERS_WORD,
    record_offset, decode_record, owner_offset, used_offset, mask_word, mask_test,
    face_edge_offset, edge_vertices_offset, global_vertex_offset, global_face_offset,
};
import mopmesh::topology::flip::{
    FaceEdges, FlipSlots, flip_quad, flip_writes, free_bit, runs_stored_order,
    MASK_SET, MASK_CLEAR, FLIP_ARENA_WRITES, FLIP_MASK_UPDATES,
    FLIP_NOT_REQUESTED, FLIP_DONE, FLIP_BOUNDARY, FLIP_NOT_INTERIOR,
    FLIP_CONNECTED, FLIP_OUT_OF_CAPACITY, FLIP_NOT_AN_EDGE,
};

const INVALID: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<storage, read_write> masks: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
// Word 0, in: the edge to consider. Word 1, out: its flip status, or
// FLIP_NOT_REQUESTED if the flip wouldn't shorten it. Word 2, in: nonzero to
// flip whether or not it shortens the edge.
@group(0) @binding(3) var<storage, read_write> request: array<u32>;

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
fn position(r: PatchRecord, local: u32) -> vec3<f32> {
    return positions[arena[global_vertex_offset(r, local)]].xyz;
}
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
    request[1] = flip_if_shorter(request[0]);
}

fn flip_if_shorter(global_edge: u32) -> u32 {
    if (global_edge >= arena[VERTEX_COUNT_WORD + EDGE]) { return FLIP_NOT_AN_EDGE; }
    let edge_owners = arena[OWNERS_WORD + EDGE];
    let face_owners = arena[OWNERS_WORD + FACE];
    let owner = owner_offset(edge_owners, global_edge);
    let r = record(arena[owner]);
    let edge = arena[owner + 1u];

    // Find the two faces and the four corners.
    var faces = array(INVALID, INVALID);
    var found = 0u;
    for (var f = 0u; f < atomicLoad(&masks[used_offset(r, FACE)]); f++) {
        if (bit(r.active_masks[FACE], f) && any(face_edges(r, f).edges == vec3(edge))) {
            if (found < 2u) { faces[found] = f; }
            found++;
        }
    }
    if (found < 2u) { return FLIP_BOUNDARY; }
    if (!runs_stored_order(face_edges(r, faces[0]), edge)) {
        faces = array(faces[1], faces[0]);
    }
    let first = face_edges(r, faces[0]);
    let second = face_edges(r, faces[1]);
    let quad = flip_quad(
        edge,
        first, array(edge_vertices(r, first.edges.x), edge_vertices(r, first.edges.y), edge_vertices(r, first.edges.z)),
        second, array(edge_vertices(r, second.edges.x), edge_vertices(r, second.edges.y), edge_vertices(r, second.edges.z)),
    );
    if (quad.status != FLIP_DONE) { return quad.status; }

    // This kernel's own rule: only flip if the new edge is shorter.
    let old_length = distance(position(r, quad.a), position(r, quad.b));
    let new_length = distance(position(r, quad.c), position(r, quad.d));
    if (request[2] == 0u && new_length >= old_length) { return FLIP_NOT_REQUESTED; }

    // The library's preconditions: interior corners, no existing c-d edge,
    // and free slots.
    let corners = array(quad.a, quad.b, quad.c, quad.d);
    for (var i = 0u; i < 4u; i++) {
        if (!bit(r.interior_mask, corners[i])) { return FLIP_NOT_INTERIOR; }
    }
    for (var e = 0u; e < atomicLoad(&masks[used_offset(r, EDGE)]); e++) {
        let v = edge_vertices(r, e);
        let joins = (v.x == quad.c && v.y == quad.d) || (v.x == quad.d && v.y == quad.c);
        if (joins && bit(r.active_masks[EDGE], e)) { return FLIP_CONNECTED; }
    }
    let new_edge = free_slot(r, EDGE, INVALID);
    let new_first = free_slot(r, FACE, INVALID);
    let new_second = free_slot(r, FACE, new_first);
    if (new_edge == INVALID || new_first == INVALID || new_second == INVALID) {
        return FLIP_OUT_OF_CAPACITY;
    }

    // Apply every write the library returns.
    let slots = FlipSlots(
        edge, faces[0], faces[1], new_edge, new_first, new_second, global_edge,
        arena[global_face_offset(r, faces[0])], arena[global_face_offset(r, faces[1])],
        arena[global_vertex_offset(r, quad.c)], arena[global_vertex_offset(r, quad.d)],
    );
    let writes = flip_writes(r, edge_owners, face_owners, quad, first, second, slots);
    for (var i = 0u; i < FLIP_ARENA_WRITES; i++) {
        arena[writes.arena[i].offset] = writes.arena[i].value;
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
