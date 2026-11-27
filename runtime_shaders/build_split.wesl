// GPU build, face split: assigns each face an owning patch of at most
// `patch_size` faces, either in input order or in Morton order of centroids.
import package::shaders::build::morton::{morton_cell, morton_code_30};
import package::shaders::geometry::triangle::triangle_centroid;

import package::shaders::parallel::grid::flat_index;

struct Params { faces: u32, patch_size: u32, }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
// Centroid bounds as order-preserving bits: [low x, y, z, high x, y, z].
// Starts as [0xffffffff x 3, 0 x 3].
@group(0) @binding(3) var<storage, read_write> bounds: array<atomic<u32>>;
// [Morton code, face] per face; sorted by code between the key and assign passes.
@group(0) @binding(4) var<storage, read_write> pairs: array<vec2<u32>>;
// The owning patch of each face.
@group(0) @binding(5) var<storage, read_write> face_patch: array<u32>;

// Maps f32 bits to u32 so that unsigned order matches float order.
fn order_bits(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn unorder_bits(bits: u32) -> f32 {
    return bitcast<f32>(select(~bits, bits & 0x7fffffffu, (bits & 0x80000000u) != 0u));
}

fn centroid(face: u32) -> vec3<f32> {
    let t = triangles[face];
    return triangle_centroid(positions[t.x].xyz, positions[t.y].xyz, positions[t.z].xyz);
}

@compute @workgroup_size(64)
fn split_input_order(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    face_patch[id.x] = id.x / params.patch_size;
}

@compute @workgroup_size(64)
fn centroid_bounds(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    let c = centroid(id.x);
    for (var i = 0u; i < 3u; i++) {
        atomicMin(&bounds[i], order_bits(c[i]));
        atomicMax(&bounds[3u + i], order_bits(c[i]));
    }
}

@compute @workgroup_size(64)
fn morton_keys(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    var low: vec3<f32>;
    var high: vec3<f32>;
    for (var i = 0u; i < 3u; i++) {
        low[i] = unorder_bits(atomicLoad(&bounds[i]));
        high[i] = unorder_bits(atomicLoad(&bounds[3u + i]));
    }
    pairs[id.x] = vec2<u32>(morton_code_30(morton_cell(centroid(id.x), low, high)), id.x);
}

// Runs on the sorted pairs: consecutive runs of `patch_size` faces share a patch.
@compute @workgroup_size(64)
fn assign_patches(@builtin(global_invocation_id) invocation: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    let id = vec3(flat_index(invocation, grid), 0u, 0u);
    if (id.x >= params.faces) { return; }
    face_patch[pairs[id.x].y] = id.x / params.patch_size;
}
