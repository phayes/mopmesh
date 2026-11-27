// GPU build, patch rows: groups patch members by row, with element IDs
// ascending within each row. Rows 0..P hold each patch's faces, P..2P its edges
// and 2P..3P its vertices.
import package::shaders::parallel::grid::{flat_index, flat_group};

struct Params { rows: u32, members: u32, }
@group(0) @binding(0) var<uniform> params: Params;
// Per member: [row, element ID].
@group(0) @binding(1) var<storage, read> members: array<vec2<u32>>;
// Per-row counts, scanned in place into row offsets, then used as fill cursors.
@group(0) @binding(2) var<storage, read_write> cursors: array<atomic<u32>>;
// [rows + 1 offsets, element IDs]. Offsets index the element region.
@group(0) @binding(3) var<storage, read_write> grouped: array<u32>;
// Element IDs grouped by row in arbitrary order, before sort_rows.
@group(0) @binding(4) var<storage, read_write> unsorted: array<u32>;

@compute @workgroup_size(64)
fn count_rows(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.members) { return; }
    atomicAdd(&cursors[members[id.x].x], 1u);
}

// Cursors now hold exclusive prefixes for rows + 1 entries; the last is the total.
@compute @workgroup_size(64)
fn store_offsets(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x > params.rows) { return; }
    grouped[id.x] = atomicLoad(&cursors[id.x]);
}

// Places members in arbitrary order within each row; sort_rows orders them.
@compute @workgroup_size(64)
fn fill_rows(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.members) { return; }
    let member = members[id.x];
    let slot = atomicAdd(&cursors[member.x], 1u);
    unsorted[slot] = member.y;
}

var<workgroup> bounds: vec2<u32>;
var<workgroup> tile: array<u32, 64>;

// One workgroup per row: each element's position is the number of smaller
// elements in its row, counted through 64-element tiles in workgroup memory.
// A row holds each element at most once, so positions are distinct. Cost per
// row grows with the square of its length, shared by 64 invocations.
@compute @workgroup_size(64)
fn sort_rows(@builtin(local_invocation_index) lane: u32,
             @builtin(workgroup_id) group: vec3<u32>,
             @builtin(num_workgroups) grid: vec3<u32>) {
    let row = flat_group(group, grid);
    if (row >= params.rows) { return; }
    if (lane == 0u) { bounds = vec2(grouped[row], grouped[row + 1u]); }
    let range = workgroupUniformLoad(&bounds);
    let base = params.rows + 1u;

    // Each pass ranks up to 64 of the row's elements, one per invocation.
    for (var first = range.x; first < range.y; first += 64u) {
        let mine = first + lane;
        var value = 0xffffffffu;
        if (mine < range.y) { value = unsorted[mine]; }
        var rank = 0u;
        for (var start = range.x; start < range.y; start += 64u) {
            var other = 0xffffffffu;
            if (start + lane < range.y) { other = unsorted[start + lane]; }
            tile[lane] = other;
            workgroupBarrier();
            for (var k = 0u; k < 64u; k++) {
                rank += u32(tile[k] < value);
            }
            workgroupBarrier();
        }
        if (mine < range.y) { grouped[base + range.x + rank] = value; }
    }
}
