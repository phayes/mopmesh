import package::shaders::topology::edge::is_boundary_edge;
import package::shaders::topology::patch::EDGE;
import package::shaders::topology::patch_arena::{
    patch_count, patch_record, slot_active, slot_owned, edge_vertices, global_vertex, global_edge,
};
import package::shaders::topology::patch_staging::{STAGED_EDGES, staged_edge_faces};
import package::shaders::topology::patch_edge_staging::{patch_edge_end, stage_patch_edges};
import package::shaders::parallel::grid::flat_group;

// One flag per edge: 1 on the boundary, 0 inside.
@group(0) @binding(0) var<storage, read_write> edge_mask: array<u32>;
// One flag per vertex, starting at zero: set to 1 by any boundary edge
// touching the vertex. Several edges may set the same flag.
@group(0) @binding(1) var<storage, read_write> vertex_mask: array<atomic<u32>>;

// Flags boundary edges and their endpoints. One workgroup handles each
// patch, and only the patch that owns an edge writes its flag.
@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    let edges = patch_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        stage_patch_edges(lane, r, first, edges);
        let end = min(edges, first + STAGED_EDGES);
        for (var e = first + lane; e < end; e += 64u) {
            if (!slot_active(r, EDGE, e) || !slot_owned(r, EDGE, e)) { continue; }
            let boundary = is_boundary_edge(staged_edge_faces(first, e));
            edge_mask[global_edge(r, e)] = select(0u, 1u, boundary);
            if (boundary) {
                let v = edge_vertices(r, e);
                atomicStore(&vertex_mask[global_vertex(r, v.x)], 1u);
                atomicStore(&vertex_mask[global_vertex(r, v.y)], 1u);
            }
        }
        workgroupBarrier();
    }
}
