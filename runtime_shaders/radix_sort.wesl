// One pass of a stable least-significant-digit radix sort of [key, value]
// pairs, on the 4-bit digit at `shift`. Each 64-pair block counts its digits;
// a scan over the counts, digit-major, gives every block its start per digit;
// each pair then lands after the earlier pairs of its block with the same digit.
// Repeating the pass for shifts 0, 4, ..., 28 sorts by the whole key, keeping
// equal keys in their input order.
import package::shaders::parallel::grid::flat_group;

struct Params { count: u32, shift: u32, blocks: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> sorted: array<vec2<u32>>;
// 16 x blocks digit counts, digit-major, scanned in place into starts.
@group(0) @binding(3) var<storage, read_write> histogram: array<u32>;

const RADIX: u32 = 16u;
const NO_DIGIT: u32 = 16u;

var<workgroup> counts: array<atomic<u32>, 16>;
var<workgroup> digits: array<u32, 64>;

fn digit_of(i: u32) -> u32 {
    if (i >= params.count) { return NO_DIGIT; }
    return (source[i].x >> params.shift) & (RADIX - 1u);
}

// Workgroups past the last block (in a two-dimensional grid's last row) do nothing.
@compute @workgroup_size(64)
fn count_digits(@builtin(local_invocation_index) lane: u32,
                @builtin(workgroup_id) group: vec3<u32>,
                @builtin(num_workgroups) grid: vec3<u32>) {
    let block = flat_group(group, grid);
    if (lane < RADIX) { atomicStore(&counts[lane], 0u); }
    workgroupBarrier();
    let digit = digit_of(block * 64u + lane);
    if (digit != NO_DIGIT) { atomicAdd(&counts[digit], 1u); }
    workgroupBarrier();
    if (lane < RADIX && block < params.blocks) {
        histogram[lane * params.blocks + block] = atomicLoad(&counts[lane]);
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(local_invocation_index) lane: u32,
           @builtin(workgroup_id) group: vec3<u32>,
           @builtin(num_workgroups) grid: vec3<u32>) {
    let block = flat_group(group, grid);
    let i = block * 64u + lane;
    let digit = digit_of(i);
    digits[lane] = digit;
    workgroupBarrier();
    if (digit == NO_DIGIT) { return; }
    var rank = 0u;
    for (var j = 0u; j < lane; j++) {
        rank += u32(digits[j] == digit);
    }
    sorted[histogram[digit * params.blocks + block] + rank] = source[i];
}
