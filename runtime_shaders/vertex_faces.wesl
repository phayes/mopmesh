struct Params { vertices: u32, faces: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> face_vertices: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursors: array<atomic<u32>>;
// [V + 1 offsets, 3F face IDs]. Offsets are relative to the face-ID region.
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.faces) { return; }
    for (var i = 0u; i < 3u; i++) {
        atomicAdd(&cursors[face_vertices[3u * id.x + i]], 1u);
    }
}

// Cursors now contain global exclusive prefixes from the hierarchical scan.
@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.vertices) { return; }
    output[id.x] = atomicLoad(&cursors[id.x]);
    if (id.x == 0u) { output[params.vertices] = 3u * params.faces; }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.faces) { return; }
    for (var i = 0u; i < 3u; i++) {
        let vertex = face_vertices[3u * id.x + i];
        let slot = atomicAdd(&cursors[vertex], 1u);
        output[params.vertices + 1u + slot] = id.x;
    }
}
