import package::shaders::topology::edge::{EdgeRecord, other_edge_vertex};

struct Params { vertices: u32, edges: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> edges: array<EdgeRecord>;
@group(0) @binding(2) var<storage, read_write> cursors: array<atomic<u32>>;
// V+1 offsets, then 2E pairs (edge ID, neighbor vertex ID). Offsets count pairs.
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.edges) { return; }
    let vertices = edges[id.x].vertices;
    atomicAdd(&cursors[vertices.x], 1u);
    atomicAdd(&cursors[vertices.y], 1u);
}

@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.vertices) { return; }
    output[id.x] = atomicLoad(&cursors[id.x]);
    if (id.x == 0u) { output[params.vertices] = 2u * params.edges; }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.edges) { return; }
    let vertices = edges[id.x].vertices;
    for (var i = 0u; i < 2u; i++) {
        let vertex = vertices[i];
        let slot = atomicAdd(&cursors[vertex], 1u);
        let base = params.vertices + 1u + 2u * slot;
        output[base] = id.x;
        output[base + 1u] = other_edge_vertex(vertices, vertex);
    }
}
