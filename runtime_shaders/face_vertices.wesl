import package::shaders::topology::patch::FACE;
import package::shaders::topology::patch_arena::{
    patch_count, patch_record, slot_end, slot_active, slot_owned, face_edge, edge_vertices,
    global_vertex, global_face,
};
import package::shaders::topology::face::face_vertices;
import package::shaders::parallel::grid::flat_group;

// Stores three mesh-wide vertex IDs per face, preserving the triangle
// vertex order. Face f occupies output[3*f] through output[3*f + 2].
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

// Each workgroup processes one patch. Only the owning patch writes a face,
// so copies of that face in neighboring patches do not duplicate the output.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    let faces = slot_end(r, FACE);
    for (var f = lane; f < faces; f += 64u) {
        if (!slot_active(r, FACE, f) || !slot_owned(r, FACE, f)) { continue; }

        // Recover the triangle's vertex order from its edges and their directions.
        var edges: array<vec2<u32>, 3>;
        var reversed: vec3<u32>;
        for (var i = 0u; i < 3u; i++) {
            let e = face_edge(r, f, i);
            edges[i] = edge_vertices(r, e.x);
            reversed[i] = e.y;
        }
        let local = face_vertices(edges, reversed);
        let dst = 3u * global_face(r, f);
        output[dst] = global_vertex(r, local.x);
        output[dst + 1u] = global_vertex(r, local.y);
        output[dst + 2u] = global_vertex(r, local.z);
    }
}
