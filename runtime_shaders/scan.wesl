import package::shaders::parallel::scan::exclusive_scan_64;

struct Params { count: u32, offset: u32, next_offset: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> arena: array<atomic<u32>>;

// Each block writes exclusive prefixes in place and its total to the next level.
@compute @workgroup_size(64)
fn scan(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) block: vec3<u32>) {
    var value = 0u;
    if (id.x < params.count) { value = atomicLoad(&arena[params.offset + id.x]); }
    let result = exclusive_scan_64(value, lane);
    if (id.x < params.count) { atomicStore(&arena[params.offset + id.x], result.x); }
    if (lane == 0u) { atomicStore(&arena[params.next_offset + block.x], result.y); }
}

// Called top-down after the next level contains global block prefixes.
@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) id: vec3<u32>,
       @builtin(workgroup_id) block: vec3<u32>) {
    if (id.x >= params.count) { return; }
    let base = atomicLoad(&arena[params.next_offset + block.x]);
    atomicAdd(&arena[params.offset + id.x], base);
}
