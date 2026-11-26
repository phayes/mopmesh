// Private workgroup scratch; no resource bindings or entry points.
var<workgroup> scan_values: array<u32, 64>;

// Collective: all 64 invocations must call in uniform control flow with their
// unique local_invocation_index (0..63). Pad inactive lanes with zero; do not
// return early. Returns (exclusive prefix, block total). Addition wraps at u32.
// Calls may be repeated sequentially; each call finishes with a barrier.
public fn exclusive_scan_64(value: u32, lane: u32) -> vec2<u32> {
    scan_values[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride *= 2u) {
        var addend = 0u;
        if (lane >= stride) { addend = scan_values[lane - stride]; }
        // All reads must complete before any lane changes the shared array.
        workgroupBarrier();
        scan_values[lane] += addend;
        workgroupBarrier();
    }
    let result = vec2(scan_values[lane] - value, scan_values[63]);
    workgroupBarrier();
    return result;
}
