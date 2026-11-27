// GPU build, patch membership. A patch holds its owned faces plus every face
// that shares a vertex with them (the ribbon), and the edges and vertices of
// those faces. With S(v) = the patches owning a face at vertex v:
//   face g is in patch p if p is in S(a), S(b) or S(c) for its vertices;
//   edge (a, b) is in p if p is in S(a), S(b), or S of either opposite vertex;
//   vertex w is in p if p is in S(w) or S of any neighbour of w.
// Faces with a missing vertex are skipped: earlier checks have reported them.
import package::shaders::parallel::grid::flat_index;

struct Params { vertices: u32, faces: u32, patches: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<vec4<u32>>;
// From the element stage: [V + 1 row offsets, corner IDs per vertex, unordered].
@group(0) @binding(2) var<storage, read> rows: array<u32>;
// The owning patch of each face.
@group(0) @binding(3) var<storage, read> face_patch: array<u32>;
// From the connectivity stage: [low, high, first face, second face or NONE].
@group(0) @binding(4) var<storage, read> edges: array<vec4<u32>>;
// From the connectivity stage; word V is the edge count.
@group(0) @binding(5) var<storage, read> edge_offsets: array<u32>;
// [V set sizes, then S(v) ascending in the slots of v's corner row].
@group(0) @binding(6) var<storage, read_write> vertex_patches: array<u32>;
// Member counts, scanned in place into first member indices:
// [F faces, 3F edges (only the first edge-count used), V vertices, 1 total].
@group(0) @binding(7) var<storage, read_write> counts: array<u32>;
// Per member: [row, element ID]. Rows 0..P hold faces, P..2P edges and
// 2P..3P vertices, each row for one patch.
@group(0) @binding(8) var<storage, read_write> members: array<vec2<u32>>;

const NONE: u32 = 0xffffffffu;

fn face_ok(face: u32) -> bool {
    let t = triangles[face];
    return t.x < params.vertices && t.y < params.vertices && t.z < params.vertices;
}

fn row_start(v: u32) -> u32 { return params.vertices + 1u + rows[v]; }
fn row_end(v: u32) -> u32 { return params.vertices + 1u + rows[v + 1u]; }
fn set_start(v: u32) -> u32 { return params.vertices + rows[v]; }

// Builds S(v): the distinct owning patches of the faces at `v`, ascending.
@compute @workgroup_size(64)
fn collect_vertex_patches(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let v = id.x;
    if (v >= params.vertices) { return; }
    let base = set_start(v);
    var size = 0u;
    for (var i = row_start(v); i < row_end(v); i++) {
        let face = rows[i] / 3u;
        if (!face_ok(face)) { continue; }
        let p = face_patch[face];
        // Insert `p` in order unless it is already present.
        var j = size;
        while (j > 0u && vertex_patches[base + j - 1u] > p) { j--; }
        if (j > 0u && vertex_patches[base + j - 1u] == p) { continue; }
        for (var k = size; k > j; k--) {
            vertex_patches[base + k] = vertex_patches[base + k - 1u];
        }
        vertex_patches[base + j] = p;
        size++;
    }
    vertex_patches[v] = size;
}

fn in_set(v: u32, p: u32) -> bool {
    var lo = set_start(v);
    var hi = lo + vertex_patches[v];
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let value = vertex_patches[mid];
        if (value == p) { return true; }
        if (value < p) { lo = mid + 1u; } else { hi = mid; }
    }
    return false;
}

// Member kinds and the vertices whose patch sets decide membership.
const FACE: u32 = 0u;
const EDGE: u32 = 1u;
const VERTEX: u32 = 2u;

fn opposite(face: u32, a: u32, b: u32) -> u32 {
    let t = triangles[face];
    return t.x + t.y + t.z - a - b;
}

fn list_count(kind: u32, element: u32) -> u32 {
    if (kind == FACE) { return 3u; }
    if (kind == EDGE) { return 4u; }
    return 1u + 2u * (row_end(element) - row_start(element));
}

// The `k`th vertex whose patch set decides membership of `element`, or NONE.
fn list_vertex(kind: u32, element: u32, k: u32) -> u32 {
    if (kind == FACE) { return triangles[element][k]; }
    if (kind == EDGE) {
        let record = edges[element];
        if (k == 0u) { return record.x; }
        if (k == 1u) { return record.y; }
        let face = select(record.w, record.z, k == 2u);
        if (face == NONE) { return NONE; }
        return opposite(face, record.x, record.y);
    }
    // A vertex, then for each corner leaving it, that corner's next and
    // previous vertices; together these are all its neighbours.
    if (k == 0u) { return element; }
    let corner = rows[row_start(element) + (k - 1u) / 2u];
    let slot = corner % 3u;
    let t = triangles[corner / 3u];
    return t[select((slot + 2u) % 3u, (slot + 1u) % 3u, k % 2u == 1u)];
}

// Counts the distinct patches of `element`, writing each as a member at
// `base` onward if `write` is set. The count and write passes call this with
// the same arguments, so they enumerate members in the same order.
fn visit_members(kind: u32, element: u32, write: bool, base: u32) -> u32 {
    var found = 0u;
    let lists = list_count(kind, element);
    for (var k = 0u; k < lists; k++) {
        let v = list_vertex(kind, element, k);
        if (v >= params.vertices) { continue; }
        for (var i = set_start(v); i < set_start(v) + vertex_patches[v]; i++) {
            let p = vertex_patches[i];
            var seen = false;
            for (var earlier = 0u; earlier < k && !seen; earlier++) {
                let u = list_vertex(kind, element, earlier);
                seen = u < params.vertices && in_set(u, p);
            }
            if (seen) { continue; }
            if (write) {
                members[base + found] = vec2<u32>(kind * params.patches + p, element);
            }
            found++;
        }
    }
    return found;
}

// Maps a count slot to its member kind and element, or kind NONE if unused.
fn decode_slot(slot: u32) -> vec2<u32> {
    let faces = params.faces;
    if (slot < faces) {
        if (!face_ok(slot)) { return vec2<u32>(NONE, 0u); }
        return vec2<u32>(FACE, slot);
    }
    if (slot < 4u * faces) {
        let edge = slot - faces;
        if (edge >= edge_offsets[params.vertices]) { return vec2<u32>(NONE, 0u); }
        return vec2<u32>(EDGE, edge);
    }
    if (slot < 4u * faces + params.vertices) {
        return vec2<u32>(VERTEX, slot - 4u * faces);
    }
    return vec2<u32>(NONE, 0u);
}

@compute @workgroup_size(64)
fn count_members(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let slot = decode_slot(id.x);
    if (slot.x == NONE) { return; }
    counts[id.x] = visit_members(slot.x, slot.y, false, 0u);
}

// Runs after `counts` has been scanned into first member indices.
@compute @workgroup_size(64)
fn write_members(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let slot = decode_slot(id.x);
    if (slot.x == NONE) { return; }
    visit_members(slot.x, slot.y, true, counts[id.x]);
}
