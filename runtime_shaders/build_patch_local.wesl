// GPU build, patch-local connectivity and owners, from the grouped patch rows.
// A member's local ID is its position in its patch row.
import package::shaders::parallel::grid::flat_index;

struct Params {
    vertices: u32,
    patches: u32,
    members: u32,
    face_members: u32,
    edge_members: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
// [3P + 1 row offsets, element IDs ascending per row]; faces, edges, vertices.
@group(0) @binding(1) var<storage, read> grouped: array<u32>;
@group(0) @binding(2) var<storage, read> face_patch: array<u32>;
// [V set sizes, then each vertex's owning patches ascending in its corner-row slots].
@group(0) @binding(3) var<storage, read> vertex_patches: array<u32>;
// From the element stage: [V + 1 row offsets, corner IDs].
@group(0) @binding(4) var<storage, read> rows: array<u32>;
// [low, high, first face, second face or NONE] per edge.
@group(0) @binding(5) var<storage, read> edges: array<vec4<u32>>;
// [edge ID, reversed 0/1] per corner.
@group(0) @binding(6) var<storage, read> face_edges: array<vec2<u32>>;
// [2 local vertex IDs per edge member, then 3 x (local edge, reversed) per
// face member], each in grouped order.
@group(0) @binding(7) var<storage, read_write> local_ids: array<u32>;
// [owner patch, owner-local ID] per member, in grouped order.
@group(0) @binding(8) var<storage, read_write> owners: array<vec2<u32>>;

const NONE: u32 = 0xffffffffu;

fn base() -> u32 { return 3u * params.patches + 1u; }

// The row containing member `j`: the last row whose offset is at most `j`.
fn row_of(j: u32) -> u32 {
    var lo = 0u;
    var hi = 3u * params.patches;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (grouped[mid] <= j) { lo = mid; } else { hi = mid; }
    }
    return lo;
}

// The position of `element` in `row`. The element must be present.
fn local_id(row: u32, element: u32) -> u32 {
    var lo = grouped[row];
    var hi = grouped[row + 1u];
    let start = lo;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (grouped[base() + mid] < element) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo - start;
}

@compute @workgroup_size(64)
fn derive_local(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let j = id.x;
    if (j >= params.members) { return; }
    let row = row_of(j);
    let kind = row / params.patches;
    let here = row % params.patches;
    let element = grouped[base() + j];
    let p = params.patches;

    if (kind == 0u) {
        // A face: its edges as local edges of this patch.
        for (var i = 0u; i < 3u; i++) {
            let d = face_edges[3u * element + i];
            let out = 2u * params.edge_members + 6u * j + 2u * i;
            local_ids[out] = local_id(p + here, d.x);
            local_ids[out + 1u] = d.y;
        }
        let owner = face_patch[element];
        owners[j] = vec2<u32>(owner, local_id(owner, element));
    } else if (kind == 1u) {
        // An edge: its endpoints as local vertices; the owner is the lower
        // patch of its faces.
        let record = edges[element];
        let out = 2u * (j - params.face_members);
        local_ids[out] = local_id(2u * p + here, record.x);
        local_ids[out + 1u] = local_id(2u * p + here, record.y);
        var owner = face_patch[record.z];
        if (record.w != NONE) { owner = min(owner, face_patch[record.w]); }
        owners[j] = vec2<u32>(owner, local_id(p + owner, element));
    } else {
        // A vertex: the owner is the lowest patch owning a face at it.
        let owner = vertex_patches[params.vertices + rows[element]];
        owners[j] = vec2<u32>(owner, local_id(2u * p + owner, element));
    }
}
