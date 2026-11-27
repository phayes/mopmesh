// GPU build, second stage: duplicate faces, edge IDs, edge faces, face edges,
// and the vertex fan checks.
// Corner `c` is slot `c % 3` of face `c / 3` and starts the directed edge to the
// face's next corner. Faces with a missing vertex are skipped: the element stage
// has already reported them.
import package::shaders::build::edge_group::{EdgeGroup, empty_edge_group, edge_group_add};
import package::shaders::build::edge_group::{edge_group_misoriented, edge_group_error_corner};
import package::shaders::topology::edge::other_edge_face;
import package::shaders::topology::face::{face_vertex_slot, INVALID_SLOT};

import package::shaders::parallel::grid::flat_index;

struct Params { vertices: u32, faces: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<vec4<u32>>;
// From the element stage: [V + 1 row offsets, corner IDs per vertex, unordered].
@group(0) @binding(2) var<storage, read> rows: array<u32>;
// Words 4..9: [first duplicate face, first edge-error corner, kind, low, high].
// Kind 0: inconsistent winding; kind 1: non-manifold edge.
// Words 9..11: [first isolated vertex, first non-manifold vertex].
@group(0) @binding(3) var<storage, read_write> errors: array<atomic<u32>>;
// Per corner: the smallest corner on the same undirected edge.
@group(0) @binding(4) var<storage, read_write> leaders: array<u32>;
// Edges led per smaller endpoint, scanned in place into the first edge ID per vertex.
@group(0) @binding(5) var<storage, read_write> edge_counts: array<u32>;
// Per edge: [low vertex, high vertex, first face, second face or NONE].
@group(0) @binding(6) var<storage, read_write> edges: array<vec4<u32>>;
// Per corner: [edge ID, reversed 0/1].
@group(0) @binding(7) var<storage, read_write> face_edges: array<vec2<u32>>;

const NONE: u32 = 0xffffffffu;

fn face_ok(face: u32) -> bool {
    let t = triangles[face];
    return t.x < params.vertices && t.y < params.vertices && t.z < params.vertices;
}

fn start(c: u32) -> u32 { return triangles[c / 3u][c % 3u]; }
fn end(c: u32) -> u32 { return triangles[c / 3u][(c % 3u + 1u) % 3u]; }
fn prev(c: u32) -> u32 { return 3u * (c / 3u) + (c % 3u + 2u) % 3u; }
fn row_start(v: u32) -> u32 { return params.vertices + 1u + rows[v]; }
fn row_end(v: u32) -> u32 { return params.vertices + 1u + rows[v + 1u]; }

// Collects the corners from `low` to `high` and from `high` to `low`.
fn edge_group(low: u32, high: u32) -> EdgeGroup {
    var group = empty_edge_group();
    for (var i = row_start(low); i < row_end(low); i++) {
        let c = rows[i];
        if (face_ok(c / 3u) && end(c) == high) { group = edge_group_add(group, c, false); }
    }
    for (var i = row_start(high); i < row_end(high); i++) {
        let c = rows[i];
        if (face_ok(c / 3u) && end(c) == low) { group = edge_group_add(group, c, true); }
    }
    return group;
}

// A face is a duplicate if an earlier face has the same three vertices; both
// contain the smallest one, so its corner row lists every candidate. Rows are
// in no particular order, so the whole row is searched.
@compute @workgroup_size(64)
fn find_duplicates(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let face = id.x;
    if (face >= params.faces || !face_ok(face)) { return; }
    let t = triangles[face].xyz;
    let low = min(t.x, min(t.y, t.z));
    let high = max(t.x, max(t.y, t.z));
    let middle = t.x + t.y + t.z - low - high;
    for (var i = row_start(low); i < row_end(low); i++) {
        let other = rows[i] / 3u;
        if (other >= face) { continue; }
        let u = triangles[other].xyz;
        let u_high = max(u.x, max(u.y, u.z));
        let u_low = min(u.x, min(u.y, u.z));
        if (u_low == low && u_high == high && u.x + u.y + u.z - u_low - u_high == middle) {
            atomicMin(&errors[4], face);
            return;
        }
    }
}

// Finds each corner's edge leader; leaders also record their edge's error.
@compute @workgroup_size(64)
fn group_corners(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let c = id.x;
    if (c >= 3u * params.faces || !face_ok(c / 3u)) { return; }
    let a = start(c);
    let b = end(c);
    let group = edge_group(min(a, b), max(a, b));
    leaders[c] = group.corners[0];
    if (group.corners[0] != c) { return; }
    let problem = edge_group_error_corner(group);
    if (problem != NONE) {
        atomicMin(&errors[5], problem);
    }
}

// Counts the edge leaders whose smaller endpoint is `v` and whose larger
// endpoint is below `limit`. Each such corner is seen exactly once: either it
// leaves `v`, or it arrives at `v` just before a corner that leaves `v`.
fn count_led_below(v: u32, limit: u32) -> u32 {
    var count = 0u;
    for (var i = row_start(v); i < row_end(v); i++) {
        let out = rows[i];
        let into = prev(out);
        if (!face_ok(out / 3u)) { continue; }
        let b = end(out);
        if (b > v && b < limit && leaders[out] == out) { count++; }
        let a = start(into);
        if (a > v && a < limit && leaders[into] == into) { count++; }
    }
    return count;
}

@compute @workgroup_size(64)
fn count_edges(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.vertices) { return; }
    edge_counts[id.x] = count_led_below(id.x, NONE);
}

// Leaders take the next ID in (low, high) order and link the edge with its faces.
@compute @workgroup_size(64)
fn number_edges(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let c = id.x;
    if (c >= 3u * params.faces || !face_ok(c / 3u) || leaders[c] != c) { return; }
    let a = start(c);
    let b = end(c);
    let low = min(a, b);
    let high = max(a, b);
    let group = edge_group(low, high);
    if (group.count > 2u) { return; }
    let edge = edge_counts[low] + count_led_below(low, high);
    var second = NONE;
    if (group.count == 2u) { second = group.corners[1] / 3u; }
    edges[edge] = vec4<u32>(low, high, group.corners[0] / 3u, second);
    for (var i = 0u; i < group.count; i++) {
        face_edges[group.corners[i]] = vec2<u32>(edge, u32(group.reversed[i]));
    }
}

// One invocation: records the kind and endpoints of the first edge error.
@compute @workgroup_size(1)
fn describe_edge_error() {
    let c = atomicLoad(&errors[5]);
    if (c == NONE) { return; }
    let a = start(c);
    let b = end(c);
    let group = edge_group(min(a, b), max(a, b));
    atomicStore(&errors[6], select(1u, 0u, edge_group_misoriented(group)));
    atomicStore(&errors[7], min(a, b));
    atomicStore(&errors[8], max(a, b));
}

// The face across the edge in `slot` of `face`, or NONE on a boundary.
fn across(face: u32, slot: u32) -> u32 {
    let record = edges[face_edges[3u * face + slot].x];
    return other_edge_face(record.zw, face);
}

// Turns around `v` from `first`, leaving each face through the edge after `v`
// (or before it, if `backward`), until reaching a boundary, returning to
// `first`, or passing `limit` faces. Returns the faces passed and whether the
// walk closed.
fn walk(v: u32, first: u32, limit: u32, backward: bool) -> vec2<u32> {
    var face = first;
    var passed = 0u;
    var closed = 0u;
    loop {
        // A face without `v` only arises from inputs already rejected.
        let p = face_vertex_slot(triangles[face].xyz, v);
        if (p == INVALID_SLOT) { break; }
        let next = across(face, select(p, (p + 2u) % 3u, backward));
        if (next == NONE) { break; }
        if (next == first || passed == limit) {
            closed = 1u;
            break;
        }
        face = next;
        passed++;
    }
    return vec2<u32>(passed, closed);
}

// Records vertices that no face uses, and vertices whose faces form more than
// one fan (a closed ring, or one strip between two boundary edges, is one fan).
@compute @workgroup_size(64)
fn check_fans(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let v = id.x;
    if (v >= params.vertices) { return; }
    let count = row_end(v) - row_start(v);
    if (count == 0u) {
        atomicMin(&errors[9], v);
        return;
    }
    let first = rows[row_start(v)] / 3u;
    let forward = walk(v, first, count, false);
    var passed = forward.x;
    if (forward.y == 0u) {
        passed += walk(v, first, count, true).x;
    }
    if (passed + 1u != count) {
        atomicMin(&errors[10], v);
    }
}
