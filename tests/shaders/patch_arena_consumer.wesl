import mopmesh::topology::patch::PATCH_LANES;
import mopmesh::topology::patch_arena::{patch_count, patch_record};
import mopmesh::topology::patch_staging::{STAGED_EDGES, staged_edge_faces};
import mopmesh::topology::patch_edge_staging::{patch_edge_end, stage_patch_edges};
import mopmesh::topology::edge::edge_face_count;

// Example of a shader that reads patches through the library's group-1
// bindings and keeps group 0 for its own output.
@group(0) @binding(0) var<storage, read_write> counts: array<atomic<u32>>;

// Counts, per patch, the edge slots that have exactly one face in the patch.
@compute @workgroup_size(PATCH_LANES)
fn count_open_edges(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if (group.x >= patch_count()) { return; }
    let r = patch_record(group.x);
    let edges = patch_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        stage_patch_edges(lane, r, first, edges);
        for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += PATCH_LANES) {
            if (edge_face_count(staged_edge_faces(first, e)) == 1u) {
                atomicAdd(&counts[group.x], 1u);
            }
        }
        workgroupBarrier();
    }
}
