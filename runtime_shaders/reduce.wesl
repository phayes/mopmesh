import package::shaders::parallel::grid::{flat_group, flat_index};

// Value type of the reduced attribute: 0 = f32, 1 = u32, 2 = i32.
override VALUE_TYPE: u32 = 0u;
// Operation: 0 = sum, 1 = min, 2 = max, 3 = dot product of input and
// input_b (f32 only; summed like operation 0).
override OP: u32 = 0u;

struct Params { count: u32, source: u32, destination: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> params: Params;
// The attribute as raw 32-bit words.
@group(0) @binding(1) var<storage, read> input: array<u32>;
// The second dot-product operand; bound to `input` for other operations.
@group(0) @binding(2) var<storage, read> input_b: array<u32>;
// Each level writes one partial result per workgroup at `destination`; later
// levels read the previous level's partials from `source`.
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;

var<workgroup> partial: array<u32, 64>;

// The value that leaves any other value unchanged under OP.
fn identity() -> u32 {
    // Minimum starts at +infinity, u32::MAX or i32::MAX.
    if (OP == 1u) {
        if (VALUE_TYPE == 0u) { return 0x7f800000u; }
        if (VALUE_TYPE == 1u) { return 0xffffffffu; }
        return 0x7fffffffu;
    }
    // Maximum starts at -infinity, 0 or i32::MIN.
    if (OP == 2u) {
        if (VALUE_TYPE == 0u) { return 0xff800000u; }
        if (VALUE_TYPE == 1u) { return 0u; }
        return 0x80000000u;
    }
    // Sum and dot start at zero, whose bits are zero for every type.
    return 0u;
}

// Combines two values, given and returned as raw bits.
fn combine(a: u32, b: u32) -> u32 {
    if (VALUE_TYPE == 0u) {
        let x = bitcast<f32>(a);
        let y = bitcast<f32>(b);
        if (OP == 1u) { return bitcast<u32>(min(x, y)); }
        if (OP == 2u) { return bitcast<u32>(max(x, y)); }
        return bitcast<u32>(x + y);
    }
    if (VALUE_TYPE == 1u) {
        if (OP == 1u) { return min(a, b); }
        if (OP == 2u) { return max(a, b); }
        return a + b;
    }
    let x = bitcast<i32>(a);
    let y = bitcast<i32>(b);
    if (OP == 1u) { return bitcast<u32>(min(x, y)); }
    if (OP == 2u) { return bitcast<u32>(max(x, y)); }
    return bitcast<u32>(x + y);
}

// Collective over the 64 lanes of a workgroup: lane 0 writes the combined
// value to scratch[params.destination + group]. All lanes must call it.
fn reduce_block(value: u32, lane: u32, group: u32) {
    partial[lane] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            partial[lane] = combine(partial[lane], partial[lane + stride]);
        }
        workgroupBarrier();
    }
    if (lane == 0u) { scratch[params.destination + group] = partial[0]; }
}

// First level: reduce the attribute itself.
@compute @workgroup_size(64)
fn first(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let i = flat_index(id, groups);
    var value = identity();
    if (i < params.count) {
        value = input[i];
        if (OP == 3u) {
            value = bitcast<u32>(bitcast<f32>(value) * bitcast<f32>(input_b[i]));
        }
    }
    reduce_block(value, lane, flat_group(group, groups));
}

// Later levels: reduce the previous level's partial results.
@compute @workgroup_size(64)
fn next(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let i = flat_index(id, groups);
    var value = identity();
    if (i < params.count) { value = scratch[params.source + i]; }
    reduce_block(value, lane, flat_group(group, groups));
}
