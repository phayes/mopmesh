// GPU build, arena assembly: writes the connectivity arena that queries read,
// word for word as GpuMesh::upload does.
// Per patch: vertex IDs, EV pairs, FE records (3 x [local edge, reversed]),
// face IDs. Then one face task per face and one edge task per edge, each from
// its owning patch, in patch order and then local order.
import package::shaders::parallel::grid::flat_index;

struct Params {
    faces: u32,
    patches: u32,
    members: u32,
    face_members: u32,
    edge_members: u32,
    face_tasks: u32,
    edge_tasks: u32,
    edges: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
// [3P + 1 row offsets, global IDs]: faces, edges, vertices of each patch.
@group(0) @binding(1) var<storage, read> grouped: array<u32>;
// [2 local vertex IDs per edge member, then 6 FE words per face member].
@group(0) @binding(2) var<storage, read> local_ids: array<u32>;
// [owner patch, owner-local ID] per member.
@group(0) @binding(3) var<storage, read> owners: array<vec2<u32>>;
// Region sizes per patch, scanned in place into each region's start.
@group(0) @binding(4) var<storage, read_write> regions: array<u32>;
// Owned flags for face then edge members, scanned in place into task indices.
@group(0) @binding(5) var<storage, read_write> tasks: array<u32>;
@group(0) @binding(6) var<storage, read_write> arena: array<u32>;

const HEADER: u32 = 4u;

fn base() -> u32 { return 3u * params.patches + 1u; }
fn row_size(row: u32) -> u32 { return grouped[row + 1u] - grouped[row]; }

fn row_of(j: u32) -> u32 {
    var lo = 0u;
    var hi = 3u * params.patches;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (grouped[mid] <= j) { lo = mid; } else { hi = mid; }
    }
    return lo;
}

// Arena word offsets of one patch's vertex IDs, EV pairs, FE records and face
// IDs, and the patch's face count.
struct Layout { start: u32, ev: u32, fe: u32, face_ids: u32, faces: u32, }

fn regions_of(patch_id: u32) -> Layout {
    let p = params.patches;
    let faces = row_size(patch_id);
    let start = HEADER + regions[patch_id];
    let ev = start + row_size(2u * p + patch_id);
    let fe = ev + 2u * row_size(p + patch_id);
    return Layout(start, ev, fe, fe + 6u * faces, faces);
}

// One invocation: [face count, face task offset, edge count, edge task offset].
@compute @workgroup_size(1)
fn write_header() {
    arena[0] = params.faces;
    arena[1] = params.face_tasks;
    arena[2] = params.edges;
    arena[3] = params.edge_tasks;
}

@compute @workgroup_size(64)
fn size_regions(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let q = id.x;
    if (q >= params.patches) { return; }
    let p = params.patches;
    regions[q] = row_size(2u * p + q) + 2u * row_size(p + q) + 7u * row_size(q);
}

@compute @workgroup_size(64)
fn flag_owned(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let j = id.x;
    if (j >= params.face_members + params.edge_members) { return; }
    let here = row_of(j) % params.patches;
    tasks[j] = u32(owners[j].x == here);
}

@compute @workgroup_size(64)
fn write_arena(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    let j = id.x;
    if (j >= params.members) { return; }
    let row = row_of(j);
    let kind = row / params.patches;
    let here = row % params.patches;
    let index = j - grouped[row];
    let element = grouped[base() + j];
    let owned = owners[j].x == here;
    let at = regions_of(here);

    if (kind == 0u) {
        // A face: its FE record and face ID, and its task if owned here.
        let fe = at.fe + 6u * index;
        for (var i = 0u; i < 6u; i++) {
            arena[fe + i] = local_ids[2u * params.edge_members + 6u * j + i];
        }
        arena[at.face_ids + index] = element;
        if (owned) {
            let task = params.face_tasks + 4u * tasks[j];
            arena[task] = at.ev;
            arena[task + 1u] = fe;
            arena[task + 2u] = at.start;
            arena[task + 3u] = element;
        }
    } else if (kind == 1u) {
        // An edge: its EV pair, and its task if owned here.
        let pair = 2u * (j - params.face_members);
        arena[at.ev + 2u * index] = local_ids[pair];
        arena[at.ev + 2u * index + 1u] = local_ids[pair + 1u];
        if (owned) {
            let task = params.edge_tasks + 7u * (tasks[j] - params.faces);
            arena[task] = index;
            arena[task + 1u] = at.faces;
            arena[task + 2u] = at.fe;
            arena[task + 3u] = at.face_ids;
            arena[task + 4u] = element;
            arena[task + 5u] = at.start;
            arena[task + 6u] = at.ev;
        }
    } else {
        // A vertex: its global ID.
        arena[at.start + index] = element;
    }
}
