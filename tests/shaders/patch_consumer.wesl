import mopmesh::topology::patch::{
    FACE, RECORDS_WORD, record_offset, decode_record, used_offset, face_edge_offset,
};
import mopmesh::topology::patch_staging::{
    clear_staged_edges, stage_face_edge, staged_edge_faces,
};

// Example of a shader using the library with its own writable buffers in
// group 0. The patch helpers locate data without requiring fixed bindings.
@group(0) @binding(0) var<storage, read_write> topology: array<u32>;
@group(0) @binding(1) var<storage, read_write> slots: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64)
fn count_edge_faces(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    var w: array<u32, 16>;
    let at = record_offset(topology[RECORDS_WORD], group.x);
    for (var i = 0u; i < 16u; i++) { w[i] = topology[at + i]; }
    let r = decode_record(w);
    let faces = atomicLoad(&slots[used_offset(r, FACE)]);
    clear_staged_edges(lane, 64u);
    workgroupBarrier();
    for (var f = lane; f < faces; f += 64u) {
        for (var i = 0u; i < 3u; i++) {
            let e = face_edge_offset(r, f, i);
            stage_face_edge(0u, topology[e], topology[e + 1u], f);
        }
    }
    workgroupBarrier();
    output[lane] = staged_edge_faces(0u, lane);
}
