import package::shaders::topology::edge::{EdgeRecord, INVALID_FACE};
import package::shaders::topology::patch::{EDGE, FACE};
import package::shaders::topology::patch_arena::{
    patch_count, patch_record, slot_end, slot_active, slot_owned, face_edge, edge_vertices,
    global_vertex, global_edge, global_face,
};
import package::shaders::topology::patch_staging::{
    STAGED_EDGES, clear_staged_edges, stage_face_edge, staged_edge_faces,
};
import package::shaders::parallel::grid::flat_group;

@group(0) @binding(0) var<storage, read_write> output: array<EdgeRecord>;

var<workgroup> edge_end: u32;

// Builds one record per mesh edge, containing its vertices and neighboring
// faces. Each workgroup handles one patch and writes only the edges it owns.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Give every invocation the same edge bound so the whole workgroup
    // reaches each barrier together.
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    if (lane == 0u) { edge_end = slot_end(r, EDGE); }
    let edges = workgroupUniformLoad(&edge_end);
    let faces = slot_end(r, FACE);

    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        // Collect the faces touching this chunk of edges in shared memory.
        clear_staged_edges(lane, edges - first);
        workgroupBarrier();
        for (var f = lane; f < faces; f += 64u) {
            if (!slot_active(r, FACE, f)) { continue; }
            for (var i = 0u; i < 3u; i++) {
                let e = face_edge(r, f, i);
                stage_face_edge(first, e.x, e.y, f);
            }
        }
        workgroupBarrier();

        // Convert local slots to mesh-wide IDs and write each owned edge once.
        let end = min(edges, first + STAGED_EDGES);
        for (var e = first + lane; e < end; e += 64u) {
            if (!slot_active(r, EDGE, e) || !slot_owned(r, EDGE, e)) { continue; }
            let local = staged_edge_faces(first, e);
            var faces_out = vec2(INVALID_FACE);
            if (local.x != INVALID_FACE) { faces_out.x = global_face(r, local.x); }
            if (local.y != INVALID_FACE) { faces_out.y = global_face(r, local.y); }
            let v = edge_vertices(r, e);
            let vertices = vec2(global_vertex(r, v.x), global_vertex(r, v.y));
            output[global_edge(r, e)] = EdgeRecord(vertices, faces_out);
        }
        workgroupBarrier();
    }
}
