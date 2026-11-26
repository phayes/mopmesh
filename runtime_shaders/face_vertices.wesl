import package::shaders::topology::face::face_vertices;

@group(0) @binding(0) var<storage, read> arena: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arena[0]) { return; }
    let task = arena[1] + 4u * id.x;
    let ev_offset = arena[task];
    let face_offset = arena[task + 1u];
    let vertex_offset = arena[task + 2u];
    var edges: array<vec2<u32>, 3>;
    var reversed: vec3<u32>;
    for (var i = 0u; i < 3u; i++) {
        let edge_offset = ev_offset + 2u * arena[face_offset + 2u * i];
        edges[i] = vec2(arena[edge_offset], arena[edge_offset + 1u]);
        reversed[i] = arena[face_offset + 2u * i + 1u];
    }
    let vertices = face_vertices(edges, reversed);
    let dst = 3u * arena[task + 3u];
    output[dst] = arena[vertex_offset + vertices.x];
    output[dst + 1u] = arena[vertex_offset + vertices.y];
    output[dst + 2u] = arena[vertex_offset + vertices.z];
}
