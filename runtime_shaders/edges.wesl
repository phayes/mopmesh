import package::shaders::topology::edge::{EdgeRecord, INVALID_FACE, INVALID_VERTEX};
import package::shaders::topology::patch::{PatchRecord, EDGE};
import package::shaders::topology::patch_arena::{
    patch_count, patch_record, slot_active, slot_owned, face_edge, edge_vertices,
    global_vertex, global_edge, global_face,
};
import package::shaders::topology::patch_staging::{STAGED_EDGES, staged_edge_faces};
import package::shaders::topology::patch_edge_staging::{patch_edge_end, stage_patch_edges};
import package::shaders::topology::directed_edge::directed_start;
import package::shaders::parallel::grid::flat_group;

// One record per mesh edge, indexed by global edge ID.
@group(0) @binding(0) var<storage, read_write> records: array<EdgeRecord>;
// Three neighboring face IDs per face, indexed by global face ID: the faces
// across edges (a, b), (b, c) and (c, a) of triangle [a, b, c], or
// INVALID_FACE across a boundary edge. Written only by `adjacency`.
@group(0) @binding(1) var<storage, read_write> neighbors: array<u32>;
// One entry per edge: its two endpoints, then the vertex opposite the edge
// in each adjoining face, in the same order as the faces in its edge record.
// A missing face gives INVALID_VERTEX. Written only by `adjacency`.
@group(0) @binding(2) var<storage, read_write> diamonds: array<vec4<u32>>;

// Converts staged local face slots to an edge record with global IDs,
// keeping INVALID_FACE for a missing face at the mesh boundary.
fn edge_record(r: PatchRecord, e: u32, local: vec2<u32>) -> EdgeRecord {
    var faces = vec2(INVALID_FACE);
    if (local.x != INVALID_FACE) { faces.x = global_face(r, local.x); }
    if (local.y != INVALID_FACE) { faces.y = global_face(r, local.y); }
    let v = edge_vertices(r, e);
    return EdgeRecord(vec2(global_vertex(r, v.x), global_vertex(r, v.y)), faces);
}

// Returns which of the face's three edges is local edge `e` (0, 1 or 2).
// The face must contain the edge.
fn face_edge_index(r: PatchRecord, face: u32, e: u32) -> u32 {
    if (face_edge(r, face, 0u).x == e) { return 0u; }
    if (face_edge(r, face, 1u).x == e) { return 1u; }
    return 2u;
}

// Writes each mesh edge's endpoints and adjoining faces. One workgroup
// handles each patch, and only the patch that owns an edge writes it.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    let edges = patch_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        stage_patch_edges(lane, r, first, edges);
        let end = min(edges, first + STAGED_EDGES);
        for (var e = first + lane; e < end; e += 64u) {
            if (!slot_active(r, EDGE, e) || !slot_owned(r, EDGE, e)) { continue; }
            records[global_edge(r, e)] = edge_record(r, e, staged_edge_faces(first, e));
        }
        workgroupBarrier();
    }
}

// Writes the same edge records as `main`, together with each face's
// neighbors and each edge's diamond. The patch that owns an edge holds both
// of its faces, so it fills both faces' neighbor entries for that edge.
@compute @workgroup_size(64)
fn adjacency(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    let edges = patch_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        stage_patch_edges(lane, r, first, edges);
        let end = min(edges, first + STAGED_EDGES);
        for (var e = first + lane; e < end; e += 64u) {
            if (!slot_active(r, EDGE, e) || !slot_owned(r, EDGE, e)) { continue; }
            let local = staged_edge_faces(first, e);
            let record = edge_record(r, e, local);
            let global = global_edge(r, e);
            records[global] = record;

            // For each face, store the face across this edge and the vertex
            // opposite the edge, which starts the face's next-but-one edge.
            var opposite = vec2(INVALID_VERTEX);
            for (var side = 0u; side < 2u; side++) {
                let f = local[side];
                if (f == INVALID_FACE) { continue; }
                let i = face_edge_index(r, f, e);
                neighbors[3u * record.faces[side] + i] = record.faces[1u - side];
                let next = face_edge(r, f, (i + 2u) % 3u);
                let start = directed_start(edge_vertices(r, next.x), next.y != 0u);
                opposite[side] = global_vertex(r, start);
            }
            diamonds[global] = vec4(record.vertices, opposite);
        }
        workgroupBarrier();
    }
}
