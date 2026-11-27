import package::shaders::topology::edge::{EdgeRecord, is_boundary_edge};

struct Params { edges: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edges: array<EdgeRecord>;
// One flag per edge: 1 on the boundary, 0 inside.
@group(0) @binding(2) var<storage, read_write> edge_mask: array<u32>;
// One flag per vertex, starting at zero: set to 1 by any boundary edge
// touching the vertex. Several edges may set the same flag.
@group(0) @binding(3) var<storage, read_write> vertex_mask: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.edges) { return; }
    let record = edges[id.x];
    let boundary = is_boundary_edge(record.faces);
    edge_mask[id.x] = select(0u, 1u, boundary);
    if (boundary) {
        atomicStore(&vertex_mask[record.vertices.x], 1u);
        atomicStore(&vertex_mask[record.vertices.y], 1u);
    }
}
