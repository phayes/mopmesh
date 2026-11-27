// Assembles connectivity and slot-state buffers for GPU mesh queries.
// The result matches GpuMesh::upload, with possible extra zero mask words.
// Connectivity contains a header, a 16-word record per patch, and each
// patch's vertex IDs, edge endpoints, directed face edges, edge IDs and face IDs.
// Each patch's mask data contains slot bounds and bits marking active slots
// and the elements owned by that patch.
import package::shaders::parallel::grid::flat_index;
import package::shaders::topology::patch::{VERTEX, EDGE, FACE, RECORD_WORDS};

struct Params {
    vertices: u32,
    edges: u32,
    faces: u32,
    patches: u32,
    members: u32,
    face_members: u32,
    edge_members: u32,
    _pad: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
// Global IDs grouped into one row per patch and element kind: face rows,
// then edge rows, then vertex rows. Starts with 3*patches + 1 row offsets.
@group(0) @binding(1) var<storage, read> grouped: array<u32>;
// Patch-local connections: two vertex slots per edge, followed by three
// (edge slot, reversed direction) pairs per face.
@group(0) @binding(2) var<storage, read> local_ids: array<u32>;
// For each patch entry, identifies the owning patch and the slot within it.
@group(0) @binding(3) var<storage, read> owners: array<vec2<u32>>;
// Connectivity word counts per patch, later replaced by cumulative start offsets.
@group(0) @binding(4) var<storage, read_write> regions: array<u32>;
// Mask word counts per patch, later replaced by cumulative start offsets.
@group(0) @binding(5) var<storage, read_write> mask_regions: array<u32>;
@group(0) @binding(6) var<storage, read_write> arena: array<u32>;
// Must start at zero. Atomic updates let several invocations safely set
// different ownership bits in the same word.
@group(0) @binding(7) var<storage, read_write> masks: array<atomic<u32>>;

const HEADER: u32 = 8u;
const INVALID: u32 = 0xffffffffu;

fn row_size(row: u32) -> u32 { return grouped[row + 1u] - grouped[row]; }
fn mask_words(slots: u32) -> u32 { return (slots + 31u) / 32u; }

// Returns patch q's vertex, edge and face slot counts, in that order.
fn counts_of(q: u32) -> vec3<u32> {
    let p = params.patches;
    return vec3(row_size(2u * p + q), row_size(p + q), row_size(q));
}

fn row_of(j: u32) -> u32 {
    var lo = 0u;
    var hi = 3u * params.patches;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (grouped[mid] <= j) { lo = mid; } else { hi = mid; }
    }
    return lo;
}

// Word offsets for one patch, available once both size arrays have been
// converted to cumulative offsets.
struct Layout {
    vertex_ids: u32,
    ev: u32,
    fe: u32,
    edge_ids: u32,
    face_ids: u32,
    used: u32,
    active_masks: vec3<u32>,
    owned_masks: vec3<u32>,
}

fn layout_of(q: u32) -> Layout {
    let n = counts_of(q);
    let w = vec3(mask_words(n.x), mask_words(n.y), mask_words(n.z));
    let start = HEADER + RECORD_WORDS * params.patches + regions[q];
    let ev = start + n[VERTEX];
    let fe = ev + 2u * n[EDGE];
    let edge_ids = fe + 6u * n[FACE];
    let used = mask_regions[q];
    let active_masks = vec3(used + 3u, used + 3u + w.x, used + 3u + w.x + w.y);
    let o = active_masks.z + w.z;
    let owned_masks = vec3(o, o + w.x, o + w.x + w.y);
    return Layout(start, ev, fe, edge_ids, edge_ids + n[EDGE], used, active_masks, owned_masks);
}

// Writes mesh counts and the patch-record offset. Static meshes have no
// owner tables, so those offsets are marked INVALID. Runs once for the mesh.
@compute @workgroup_size(1)
fn write_header() {
    arena[0] = params.vertices;
    arena[1] = params.edges;
    arena[2] = params.faces;
    arena[3] = params.patches;
    arena[4] = HEADER;
    arena[5] = INVALID;
    arena[6] = INVALID;
    arena[7] = INVALID;
}

// Calculates each patch's connectivity and mask storage sizes in u32 words.
@compute @workgroup_size(64)
fn size_patches(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let q = flat_index(invocation, grid);
    if (q >= params.patches) { return; }
    let n = counts_of(q);
    regions[q] = n.x + 3u * n.y + 7u * n.z;
    mask_regions[q] = 3u + 2u * (mask_words(n.x) + mask_words(n.y) + mask_words(n.z));
}

// After offsets are ready, writes each patch's record and slot bounds.
// Every slot is active because a static mesh has no spare capacity.
@compute @workgroup_size(64)
fn write_records(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let q = flat_index(invocation, grid);
    if (q >= params.patches) { return; }
    let n = counts_of(q);
    let at = layout_of(q);
    let r = HEADER + RECORD_WORDS * q;
    let words = array<u32, 16>(
        n.x, n.y, n.z, at.vertex_ids, at.ev, at.fe, at.edge_ids, at.face_ids,
        at.active_masks.x, at.active_masks.y, at.active_masks.z,
        at.owned_masks.x, at.owned_masks.y, at.owned_masks.z,
        INVALID, at.used,
    );
    for (var i = 0u; i < RECORD_WORDS; i++) {
        arena[r + i] = words[i];
    }
    for (var k = 0u; k < 3u; k++) {
        atomicStore(&masks[at.used + k], n[k]);
        for (var w = 0u; w < mask_words(n[k]); w++) {
            let rest = n[k] - 32u * w;
            let bits = select((1u << rest) - 1u, INVALID, rest >= 32u);
            atomicStore(&masks[at.active_masks[k] + w], bits);
        }
    }
}

// After offsets are ready, writes each patch entry's connectivity and sets
// its ownership bit if this is the element's owning copy.
@compute @workgroup_size(64)
fn write_arena(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let j = flat_index(invocation, grid);
    if (j >= params.members) { return; }
    let row = row_of(j);
    let group = row / params.patches;
    let here = row % params.patches;
    let index = j - grouped[row];
    let element = grouped[3u * params.patches + 1u + j];
    let at = layout_of(here);

    // Select the data format for this row: faces first, then edges, then vertices.
    var kind = VERTEX;
    if (group == 0u) {
        kind = FACE;
        let fe = at.fe + 6u * index;
        for (var i = 0u; i < 6u; i++) {
            arena[fe + i] = local_ids[2u * params.edge_members + 6u * j + i];
        }
        arena[at.face_ids + index] = element;
    } else if (group == 1u) {
        kind = EDGE;
        let pair = 2u * (j - params.face_members);
        arena[at.ev + 2u * index] = local_ids[pair];
        arena[at.ev + 2u * index + 1u] = local_ids[pair + 1u];
        arena[at.edge_ids + index] = element;
    } else {
        arena[at.vertex_ids + index] = element;
    }
    if (owners[j].x == here) {
        atomicOr(&masks[at.owned_masks[kind] + index / 32u], 1u << (index % 32u));
    }
}
