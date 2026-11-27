struct Params { vertices: u32, step: f32, has_fixed: u32, _pad: u32, }
@group(0) @binding(0) var<uniform> params: Params;
// VV CSR: V + 1 offsets counting pairs, then (edge ID, neighbor ID) pairs.
@group(0) @binding(1) var<storage, read> neighbors: array<u32>;
// One flag per vertex when has_fixed is 1: non-zero keeps the vertex in place.
@group(0) @binding(2) var<storage, read> fixed: array<u32>;
@group(0) @binding(3) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> destination: array<vec4<f32>>;

// One uniform Laplacian step: move each free vertex `step` of the way from
// its position toward the average of its neighbors' positions. The w lane is
// carried over unchanged.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = id.x;
    if (v >= params.vertices) { return; }
    let p = source[v];
    let start = neighbors[v];
    let end = neighbors[v + 1u];
    if ((params.has_fixed != 0u && fixed[v] != 0u) || start == end) {
        destination[v] = p;
        return;
    }
    let entries = params.vertices + 1u;
    var sum = vec3(0.0);
    for (var i = start; i < end; i++) {
        sum += source[neighbors[entries + 2u * i + 1u]].xyz;
    }
    let average = sum / f32(end - start);
    destination[v] = vec4(p.xyz + params.step * (average - p.xyz), p.w);
}
