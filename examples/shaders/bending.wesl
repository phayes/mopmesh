import mopmesh::topology::csr::{CsrRow, row_length};
import mopmesh::geometry::triangle::normalize_or_zero;

// This downstream shader chooses its own bindings and entry point.
@group(0) @binding(0) var<storage, read> adjacency: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> bending: array<f32>;

// How far each vertex's edges leave its tangent plane: the mean of
// |cos(angle)| between the vertex normal and each edge direction. Zero on a
// flat patch, larger where the surface bends.
@compute @workgroup_size(64)
fn bending_metric(@builtin(global_invocation_id) id: vec3<u32>) {
    let vertices = arrayLength(&bending);
    if (id.x >= vertices) { return; }
    let row = CsrRow(adjacency[id.x], adjacency[id.x + 1u]);
    let p = positions[id.x].xyz;
    let n = normals[id.x].xyz;
    var total = 0.0;
    for (var entry = row.start; entry < row.end; entry++) {
        let neighbor = adjacency[vertices + 1u + 2u * entry + 1u];
        total += abs(dot(n, normalize_or_zero(positions[neighbor].xyz - p)));
    }
    let degree = row_length(row);
    bending[id.x] = select(0.0, total / f32(degree), degree != 0u);
}
