import mopmesh::topology::csr::{CsrRow, row_length};
import mopmesh::topology::edge::{EdgeRecord, is_boundary_edge};
import mopmesh::geometry::edge::edge_length;

@group(0) @binding(0) var<storage, read> adjacency: array<u32>;
@group(0) @binding(1) var<storage, read> edges: array<EdgeRecord>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(64)
fn neighborhood(@builtin(global_invocation_id) id: vec3<u32>) {
    let vertices = arrayLength(&weights);
    if (id.x >= vertices) { return; }
    let row = CsrRow(adjacency[id.x], adjacency[id.x + 1u]);
    var boundary = 0u;
    var total = 0.0;
    for (var entry = row.start; entry < row.end; entry++) {
        let base = vertices + 1u + 2u * entry;
        let edge = adjacency[base];
        let neighbor = adjacency[base + 1u];
        boundary += select(0u, 1u, is_boundary_edge(edges[edge].faces));
        total += edge_length(positions[id.x].xyz, positions[neighbor].xyz);
    }
    let degree = row_length(row);
    var mean = 0.0;
    if (degree != 0u) { mean = total / f32(degree); }
    // The last lane is an application-specific weighted metric.
    output[id.x] = vec4(f32(degree), f32(boundary), mean, weights[id.x] * mean);
}
