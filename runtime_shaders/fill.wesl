import package::shaders::parallel::grid::flat_index;

struct Params { value: u32, words: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

// Writes params.value to the first params.words words of output.
@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let i = flat_index(id, groups);
    if (i >= params.words) { return; }
    output[i] = params.value;
}
