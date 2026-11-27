// Mesh relations within one patch, worked out inside your own compute
// shader instead of read from precomputed query buffers. Because they are
// rebuilt from the patch as it is now, they stay correct after edits such as
// edge flips.
//
// This module declares no buffer bindings. You read the patch data from your
// own buffers and pass the values in. For ready-made loops over a read-only
// arena, import `topology::patch_queries` instead.
//
// Vertex neighbors (VV) are built in memory shared by a workgroup of 64
// invocations, for up to VV_ROWS vertex slots at a time. All 64 invocations
// must run these steps in order, each in uniform control flow, with a
// workgroupBarrier() after each one:
//
// 1. vv_clear(lane).
// 2. For every active edge of the patch, divided among the invocations:
//    vv_count_edge(first, endpoints).
// 3. let end = vv_plan(lane, first, vertex_end). This includes its own
//    barriers. Rows for vertex slots [first, end) are being built.
// 4. For every active edge again: vv_scatter_edge(first, end, endpoints).
// 5. Read rows with vv_size, vv_oversized, vv_row and vv_entry for vertex
//    slots in [first, end). Then start the next chunk at `end`.
//
// Shared memory use is about 7 KB per workgroup.

import super::directed_edge::directed_start;
import super::csr::CsrRow;

// Vertex slots handled per chunk.
public const VV_ROWS: u32 = 256u;
// Neighbor entries that fit in shared memory per chunk. A vertex with more
// neighbors than this is reported as oversized by vv_oversized.
public const VV_ENTRIES: u32 = 1024u;
const LANES: u32 = 64u;

var<workgroup> vv_cursor: array<atomic<u32>, 256>;
var<workgroup> vv_start: array<u32, 256>;
var<workgroup> vv_count: array<u32, 256>;
var<workgroup> vv_entries: array<u32, 1024>;
var<workgroup> vv_end: u32;

// Returns which of a face's three edges is `edge` (0, 1 or 2), or 3 if the
// face doesn't contain it. `edges` are the face's local edge slots in
// winding order.
public fn face_edge_index(edges: vec3<u32>, edge: u32) -> u32 {
    for (var i = 0u; i < 3u; i++) {
        if (edges[i] == edge) { return i; }
    }
    return 3u;
}

// Returns which of a face's three edges starts at the corner opposite its
// edge `i` (0, 1 or 2). Pass that edge to opposite_vertex.
public fn opposite_edge(i: u32) -> u32 {
    return (i + 2u) % 3u;
}

// Returns the face's corner opposite its edge `i`: the vertex of the
// triangle that is not on that edge. Pass the stored endpoints of the face's
// edge `opposite_edge(i)`, and its direction flag in the face: 0 when the
// face runs the edge in its stored vertex order, 1 when it reverses it.
public fn opposite_vertex(endpoints: vec2<u32>, reversed: u32) -> u32 {
    return directed_start(endpoints, reversed != 0u);
}

// Starts a VV chunk with no neighbors counted. All 64 invocations must call
// this with their local_invocation_index as `lane`.
public fn vv_clear(lane: u32) {
    for (var i = lane; i < VV_ROWS; i += LANES) {
        atomicStore(&vv_cursor[i], 0u);
    }
}

// Counts an active edge toward its endpoints' rows, if they fall in the
// chunk starting at vertex slot `first`. `endpoints` are the edge's two
// local vertex slots.
public fn vv_count_edge(first: u32, endpoints: vec2<u32>) {
    for (var i = 0u; i < 2u; i++) {
        let v = endpoints[i];
        if (v >= first && v - first < VV_ROWS) { atomicAdd(&vv_cursor[v - first], 1u); }
    }
}

// Lays out the rows counted for the chunk starting at `first` and returns
// the chunk's end. Rows for vertex slots [first, end) are built in this
// chunk; later slots wait for the next chunk, which starts at `end`. The
// result is the same in every invocation and is always greater than
// `first` when `first < vertex_end`. `vertex_end` is the patch's slot
// bound for vertices. All 64 invocations must call this.
public fn vv_plan(lane: u32, first: u32, vertex_end: u32) -> u32 {
    // One invocation assigns each row its place in the entry array, and
    // stops the chunk when the next row would not fit. A vertex whose row
    // alone is too large gets a chunk to itself and stores no entries.
    workgroupBarrier();
    if (lane == 0u) {
        let rows = min(VV_ROWS, vertex_end - min(first, vertex_end));
        var used = 0u;
        var end = first + rows;
        for (var i = 0u; i < rows; i++) {
            let n = atomicLoad(&vv_cursor[i]);
            vv_count[i] = n;
            if (i == 0u && n > VV_ENTRIES) {
                vv_start[0] = 0u;
                end = first + 1u;
                break;
            }
            if (used + n > VV_ENTRIES) {
                end = first + i;
                break;
            }
            vv_start[i] = used;
            used += n;
        }
        vv_end = end;
    }
    let end = workgroupUniformLoad(&vv_end);

    // Reset the cursors for the scatter step.
    for (var i = lane; i < VV_ROWS; i += LANES) {
        atomicStore(&vv_cursor[i], 0u);
    }
    workgroupBarrier();
    return end;
}

// Adds an active edge to its endpoints' rows, for endpoints in [first, end).
// Pass the same `first` and the `end` returned by vv_plan.
public fn vv_scatter_edge(first: u32, end: u32, endpoints: vec2<u32>) {
    for (var i = 0u; i < 2u; i++) {
        let v = endpoints[i];
        if (v < first || v >= end) { continue; }
        let row = v - first;
        if (vv_count[row] > VV_ENTRIES) { continue; }
        let k = vv_start[row] + atomicAdd(&vv_cursor[row], 1u);
        vv_entries[k] = endpoints[1u - i];
    }
}

// Returns the number of neighbors of vertex slot `v`, counting each edge
// once. `v` must be in the current chunk [first, end).
public fn vv_size(first: u32, v: u32) -> u32 {
    return vv_count[v - first];
}

// Returns whether the row of vertex slot `v` was too large to store. Its
// neighbors must then be found by scanning the patch's edges for `v`.
// `v` must be in the current chunk [first, end).
public fn vv_oversized(first: u32, v: u32) -> bool {
    return vv_count[v - first] > VV_ENTRIES;
}

// Returns the entry range of vertex slot `v` for vv_entry, in no
// particular order. Empty for an oversized row. `v` must be in the current
// chunk [first, end).
public fn vv_row(first: u32, v: u32) -> CsrRow {
    let row = v - first;
    if (vv_count[row] > VV_ENTRIES) { return CsrRow(0u, 0u); }
    return CsrRow(vv_start[row], vv_start[row] + vv_count[row]);
}

// Returns the neighboring vertex slot stored at entry `k` of a row.
public fn vv_entry(k: u32) -> u32 {
    return vv_entries[k];
}
