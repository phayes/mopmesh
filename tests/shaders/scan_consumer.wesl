import mopmesh::parallel::scan::exclusive_scan_64;

// A downstream collective uses its own binding and entry-point names.
@group(0) @binding(0) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64)
fn scan_twice(@builtin(local_invocation_index) lane: u32) {
    let first = exclusive_scan_64(lane + 1u, lane);
    let second = exclusive_scan_64(1u, lane);
    output[lane] = first;
    output[64u + lane] = second;
}
