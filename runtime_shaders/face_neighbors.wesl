import package::shaders::topology::edge::{EdgeRecord, INVALID_FACE, INVALID_VERTEX, other_edge_face};
import package::shaders::topology::face::{face_edge_slot, opposite_vertex};

@group(0) @binding(0) var<storage, read> edges: array<EdgeRecord>;
@group(0) @binding(1) var<storage, read> face_vertices: array<u32>;
@group(0) @binding(2) var<storage, read_write> neighbors: array<u32>;
// Endpoint pair followed by opposite vertices, aligned with EdgeRecord.faces.
@group(0) @binding(3) var<storage, read_write> diamonds: array<vec4<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&edges)) { return; }
    let edge = edges[id.x];
    var opposite = vec2(INVALID_VERTEX);
    for (var side = 0u; side < 2u; side++) {
        let face = edge.faces[side];
        if (face != INVALID_FACE) {
            let base = 3u * face;
            let vertices = vec3(face_vertices[base], face_vertices[base + 1u], face_vertices[base + 2u]);
            let slot = face_edge_slot(vertices, edge.vertices);
            // Valid manifold connectivity assigns exactly one writer per face slot.
            neighbors[base + slot] = other_edge_face(edge.faces, face);
            opposite[side] = opposite_vertex(vertices, edge.vertices);
        }
    }
    diamonds[id.x] = vec4(edge.vertices, opposite);
}
