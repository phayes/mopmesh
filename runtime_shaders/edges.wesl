import package::shaders::topology::edge::{EdgeRecord, empty_edge_faces, append_edge_face};
import package::shaders::topology::face::face_contains_edge;

@group(0) @binding(0) var<storage, read> arena: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<EdgeRecord>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arena[2]) { return; }
    let task = arena[3] + 7u * id.x;
    let edge = arena[task];
    let count = arena[task + 1u];
    let fe_base = arena[task + 2u];
    let face_ids = arena[task + 3u];
    var faces = empty_edge_faces();
    // The owner patch's vertex-star ribbon includes every incident face.
    // One invocation owns the output record; no atomic slot allocation is needed.
    for (var f = 0u; f < count; f++) {
        let base = fe_base + 6u * f;
        let edges = vec3(arena[base], arena[base + 2u], arena[base + 4u]);
        if (face_contains_edge(edges, edge)) {
            faces = append_edge_face(faces, arena[face_ids + f]);
        }
    }
    let vertex_ids = arena[task + 5u];
    let ev = arena[task + 6u] + 2u * edge;
    let vertices = vec2(arena[vertex_ids + arena[ev]], arena[vertex_ids + arena[ev + 1u]]);
    output[arena[task + 4u]] = EdgeRecord(vertices, faces);
}
