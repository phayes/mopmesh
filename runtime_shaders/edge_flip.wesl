import package::shaders::topology::flip::{
    FaceEdges, FlipQuad, flip_quad, flipped_faces,
    FLIP_DONE, FLIP_BOUNDARY, FLIP_NOT_INTERIOR, FLIP_CONNECTED,
    FLIP_OUT_OF_CAPACITY, FLIP_NOT_AN_EDGE,
};

// Flips one edge of an editable mesh, or leaves the mesh untouched and
// reports why not. Must match the layout in src/gpu/editable.rs.

const INVALID: u32 = 0xffffffffu;
// Header word positions.
const EDGE_COUNT: u32 = 1u;
const RECORDS: u32 = 4u;
const EDGE_OWNERS: u32 = 6u;
const FACE_OWNERS: u32 = 7u;
const RECORD_WORDS: u32 = 16u;

struct Params { edge: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> arena: array<u32>;
@group(0) @binding(2) var<storage, read_write> masks: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

// One patch record, read word by word. Index 0, 1, 2 is vertex, edge, face.
struct Record {
    capacity: array<u32, 3>,
    vertex_ids: u32,
    ev: u32,
    fe: u32,
    edge_ids: u32,
    face_ids: u32,
    active_mask: array<u32, 3>,
    owned_mask: array<u32, 3>,
    interior: u32,
    used: u32,
}

fn record(patch_id: u32) -> Record {
    let at = arena[RECORDS] + RECORD_WORDS * patch_id;
    var r: Record;
    for (var k = 0u; k < 3u; k++) {
        r.capacity[k] = arena[at + k];
        r.active_mask[k] = arena[at + 8u + k];
        r.owned_mask[k] = arena[at + 11u + k];
    }
    r.vertex_ids = arena[at + 3u];
    r.ev = arena[at + 4u];
    r.fe = arena[at + 5u];
    r.edge_ids = arena[at + 6u];
    r.face_ids = arena[at + 7u];
    r.interior = arena[at + 14u];
    r.used = arena[at + 15u];
    return r;
}

fn bit(offset: u32, slot: u32) -> bool {
    return (atomicLoad(&masks[offset + slot / 32u]) & (1u << (slot % 32u))) != 0u;
}
fn set_bit(offset: u32, slot: u32) {
    atomicOr(&masks[offset + slot / 32u], 1u << (slot % 32u));
}
fn clear_bit(offset: u32, slot: u32) {
    atomicAnd(&masks[offset + slot / 32u], ~(1u << (slot % 32u)));
}
fn used(r: Record, kind: u32) -> u32 {
    return atomicLoad(&masks[r.used + kind]);
}

fn face_edges(r: Record, slot: u32) -> FaceEdges {
    let at = r.fe + 6u * slot;
    return FaceEdges(
        vec3(arena[at], arena[at + 2u], arena[at + 4u]),
        vec3(arena[at + 1u], arena[at + 3u], arena[at + 5u]),
    );
}
fn edge_vertices(r: Record, slot: u32) -> vec2<u32> {
    let at = r.ev + 2u * slot;
    return vec2(arena[at], arena[at + 1u]);
}
fn face_edge_vertices(r: Record, face: FaceEdges) -> array<vec2<u32>, 3> {
    return array(
        edge_vertices(r, face.edges.x),
        edge_vertices(r, face.edges.y),
        edge_vertices(r, face.edges.z),
    );
}
fn write_face(r: Record, slot: u32, face: FaceEdges) {
    let at = r.fe + 6u * slot;
    for (var i = 0u; i < 3u; i++) {
        arena[at + 2u * i] = face.edges[i];
        arena[at + 2u * i + 1u] = face.reversed[i];
    }
}

// Returns the first inactive slot of `kind` that isn't `skip`, or INVALID.
fn free_slot(r: Record, kind: u32, skip: u32) -> u32 {
    for (var slot = 0u; slot < r.capacity[kind]; slot++) {
        if (slot != skip && !bit(r.active_mask[kind], slot)) { return slot; }
    }
    return INVALID;
}

@compute @workgroup_size(1)
fn main() {
    status[0] = flip(params.edge);
}

fn flip(global_edge: u32) -> u32 {
    if (global_edge >= arena[EDGE_COUNT]) { return FLIP_NOT_AN_EDGE; }

    // Find the edge's owning patch and the two active faces on it there.
    let owner = arena[EDGE_OWNERS] + 2u * global_edge;
    let patch_id = arena[owner];
    let edge = arena[owner + 1u];
    let r = record(patch_id);
    var faces = array(INVALID, INVALID);
    var found = 0u;
    for (var f = 0u; f < used(r, 2u); f++) {
        if (bit(r.active_mask[2], f) && any(face_edges(r, f).edges == vec3(edge))) {
            if (found < 2u) { faces[found] = f; }
            found++;
        }
    }
    if (found < 2u) { return FLIP_BOUNDARY; }

    // Find the four corners, then check that the flip stays inside this patch
    // and wouldn't duplicate an edge.
    let first = face_edges(r, faces[0]);
    let second = face_edges(r, faces[1]);
    let quad = flip_quad(edge, first, face_edge_vertices(r, first), second, face_edge_vertices(r, second));
    if (quad.status != FLIP_DONE) { return quad.status; }
    let corners = array(quad.a, quad.b, quad.c, quad.d);
    for (var i = 0u; i < 4u; i++) {
        if (!bit(r.interior, corners[i])) { return FLIP_NOT_INTERIOR; }
    }
    for (var e = 0u; e < used(r, 1u); e++) {
        let v = edge_vertices(r, e);
        if (bit(r.active_mask[1], e) && ((v.x == quad.c && v.y == quad.d) || (v.x == quad.d && v.y == quad.c))) {
            return FLIP_CONNECTED;
        }
    }

    // Pick free slots for the new edge and faces. Slots freed by this flip
    // aren't reused until a later flip.
    let new_edge = free_slot(r, 1u, INVALID);
    let new_first = free_slot(r, 2u, INVALID);
    let new_second = free_slot(r, 2u, new_first);
    if (new_edge == INVALID || new_first == INVALID || new_second == INVALID) {
        return FLIP_OUT_OF_CAPACITY;
    }

    // Commit. Nothing above wrote to the mesh.
    let global_first = arena[r.face_ids + faces[0]];
    let global_second = arena[r.face_ids + faces[1]];
    let new_faces = flipped_faces(quad, first, second, new_edge);

    // Remove the old edge and faces.
    arena[r.ev + 2u * edge] = INVALID;
    arena[r.ev + 2u * edge + 1u] = INVALID;
    arena[r.edge_ids + edge] = INVALID;
    clear_bit(r.active_mask[1], edge);
    clear_bit(r.owned_mask[1], edge);
    for (var i = 0u; i < 2u; i++) {
        write_face(r, faces[i], FaceEdges(vec3(INVALID), vec3(INVALID)));
        arena[r.face_ids + faces[i]] = INVALID;
        clear_bit(r.active_mask[2], faces[i]);
        clear_bit(r.owned_mask[2], faces[i]);
    }

    // Add the new edge c-d with the old edge's global ID.
    arena[r.ev + 2u * new_edge] = quad.c;
    arena[r.ev + 2u * new_edge + 1u] = quad.d;
    arena[r.edge_ids + new_edge] = global_edge;
    arena[owner + 1u] = new_edge;
    set_bit(r.active_mask[1], new_edge);
    set_bit(r.owned_mask[1], new_edge);
    atomicMax(&masks[r.used + 1u], new_edge + 1u);

    // Add the new faces with the old faces' global IDs.
    let slots = array(new_first, new_second);
    let globals = array(global_first, global_second);
    for (var i = 0u; i < 2u; i++) {
        write_face(r, slots[i], new_faces[i]);
        arena[r.face_ids + slots[i]] = globals[i];
        let face_owner = arena[FACE_OWNERS] + 2u * globals[i];
        arena[face_owner + 1u] = slots[i];
        set_bit(r.active_mask[2], slots[i]);
        set_bit(r.owned_mask[2], slots[i]);
        atomicMax(&masks[r.used + 2u], slots[i] + 1u);
    }
    return FLIP_DONE;
}
