import package::shaders::parallel::scan::exclusive_scan_64;
import package::shaders::parallel::grid::flat_group;

struct Params { count: u32, offset: u32, next_offset: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> arena: array<atomic<u32>>;

// Each block writes exclusive prefixes in place and its total to the next level.
// Workgroups past the last block (in a two-dimensional grid's last row) only
// take part in the collective scan.
@compute @workgroup_size(64)
fn scan(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let block = flat_group(group, grid);
    let i = block * 64u + lane;
    var value = 0u;
    if (i < params.count) { value = atomicLoad(&arena[params.offset + i]); }
    let result = exclusive_scan_64(value, lane);
    if (i < params.count) { atomicStore(&arena[params.offset + i], result.x); }
    if (lane == 0u && block * 64u < params.count) {
        atomicStore(&arena[params.next_offset + block], result.y);
    }
}

// Called top-down after the next level contains global block prefixes.
@compute @workgroup_size(64)
fn add(@builtin(local_invocation_index) lane: u32,
       @builtin(workgroup_id) group: vec3<u32>,
       @builtin(num_workgroups) grid: vec3<u32>) {
    let block = flat_group(group, grid);
    let i = block * 64u + lane;
    if (i >= params.count) { return; }
    let base = atomicLoad(&arena[params.next_offset + block]);
    atomicAdd(&arena[params.offset + i], base);
}
