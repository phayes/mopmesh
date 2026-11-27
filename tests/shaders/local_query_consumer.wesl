// A downstream shader that works out edge diamonds and vertex neighbors
// inside its own kernels with `topology::patch_queries`, one workgroup per
// patch, and writes them in the layout of the library's global queries so a
// test can compare the two.
import mopmesh::topology::patch::{PatchRecord, VERTEX, EDGE, PATCH_LANES};
import mopmesh::topology::patch_arena::{
    patch_count, patch_record, slot_active, slot_owned, global_vertex, global_edge,
};
import mopmesh::topology::patch_staging::STAGED_EDGES;
import mopmesh::topology::patch_edge_staging::{patch_edge_end, stage_patch_edges};
import mopmesh::topology::patch_queries::{
    patch_slot_end, edge_diamond, stage_vertex_neighbors, vv_begin, vv_valid, vv_vertex, vv_next,
};
import mopmesh::topology::local_query::{vv_size, vv_oversized};
import mopmesh::topology::edge::INVALID_VERTEX;
import mopmesh::parallel::grid::flat_group;

// Row offsets per mesh vertex (V + 1 words), from the global VV query.
@group(0) @binding(0) var<storage, read> offsets: array<u32>;
// Neighbor vertex IDs, written into each vertex's row in any order.
@group(0) @binding(1) var<storage, read_write> neighbors: array<u32>;
// Per mesh vertex: [neighbors written, vv_size, 1 if the row was oversized].
@group(0) @binding(2) var<storage, read_write> rows: array<atomic<u32>>;
// Per mesh edge: [endpoint0, endpoint1, opposite0, opposite1] as mesh-wide IDs.
@group(0) @binding(3) var<storage, read_write> diamonds: array<u32>;

fn to_global(r: PatchRecord, slot: u32) -> u32 {
    if (slot == INVALID_VERTEX) { return INVALID_VERTEX; }
    return global_vertex(r, slot);
}

@compute @workgroup_size(PATCH_LANES)
fn edge_diamonds(
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
        for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += PATCH_LANES) {
            if (!slot_active(r, EDGE, e) || !slot_owned(r, EDGE, e)) { continue; }
            let d = edge_diamond(r, first, e);
            let at = 4u * global_edge(r, e);
            for (var i = 0u; i < 4u; i++) { diamonds[at + i] = to_global(r, d[i]); }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(PATCH_LANES)
fn vertex_neighbors(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    let vertices = patch_slot_end(lane, r, VERTEX);
    var first = 0u;
    while (first < vertices) {
        let end = stage_vertex_neighbors(lane, r, first);
        for (var v = first + lane; v < end; v += PATCH_LANES) {
            if (!slot_active(r, VERTEX, v) || !slot_owned(r, VERTEX, v)) { continue; }
            let g = global_vertex(r, v);
            atomicStore(&rows[3u * g + 1u], vv_size(first, v));
            atomicStore(&rows[3u * g + 2u], select(0u, 1u, vv_oversized(first, v)));
            for (var it = vv_begin(r, first, v); vv_valid(it); it = vv_next(r, it)) {
                let k = atomicAdd(&rows[3u * g], 1u);
                neighbors[offsets[g] + k] = global_vertex(r, vv_vertex(r, it));
            }
        }
        first = end;
    }
}
