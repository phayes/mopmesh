import mopmesh::topology::edge::{EdgeRecord, is_boundary_edge, edge_face_count};
import mopmesh::geometry::edge::edge_length;

// Application-owned bindings and output policy; only public package imports.
@group(0) @binding(0) var<storage, read> edges: array<EdgeRecord>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;

@compute @workgroup_size(64)
fn weighted_boundary(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&weights)) { return; }
    let edge = edges[id.x];
    var contribution = 0.0;
    if (is_boundary_edge(edge.faces)) {
        contribution = weights[id.x] * edge_length(
            positions[edge.vertices.x].xyz, positions[edge.vertices.y].xyz);
    }
    output[id.x] = vec2(contribution, f32(edge_face_count(edge.faces)));
}
