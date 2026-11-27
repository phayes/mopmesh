// Example application rule for a flat mesh: request a flip when the angles
// opposite an edge sum to more than 180 degrees, allowing a small tolerance.
// Repeating this rule aims for a Delaunay triangulation, where no interior
// edge exceeds that angle limit.
//
// Group 0 holds positions. `topology::flip_pass` supplies group 1 and handles
// conflicts, mesh updates and each edge's status.
import mopmesh::topology::flip_pass::{
    flip_patch_count, flip_patch, flip_edge_end, flip_stage, flip_candidate, flip_request,
    flip_finish, flip_has_corners,
};
import mopmesh::topology::patch_staging::STAGED_EDGES;
import mopmesh::topology::patch::PATCH_LANES;
import mopmesh::parallel::grid::flat_group;

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;

// Extra angle, in radians, required beyond 180 degrees. This avoids repeated
// flips when rounding makes the two diagonals nearly equally suitable.
const TOLERANCE: f32 = 1e-3;
const PI: f32 = 3.14159265;

fn angle_at(corner: vec3<f32>, p: vec3<f32>, q: vec3<f32>) -> f32 {
    let u = normalize(p - corner);
    let v = normalize(q - corner);
    return acos(clamp(dot(u, v), -1.0, 1.0));
}

// Test the angles at c and d, opposite the current edge a-b.
// The input is [a, b, c, d] in mesh-wide vertex IDs.
fn should_flip(corners: array<u32, 4>) -> bool {
    let a = positions[corners[0]].xyz;
    let b = positions[corners[1]].xyz;
    let c = positions[corners[2]].xyz;
    let d = positions[corners[3]].xyz;
    return angle_at(c, a, b) + angle_at(d, a, b) > PI + TOLERANCE;
}

// One workgroup per patch offers each of the patch's edges to the rule, then
// flips a non-conflicting set of the requested edges. Edges that can't flip
// report why; eligible edges that pass the angle test report
// FLIP_NOT_REQUESTED.
@compute @workgroup_size(PATCH_LANES)
fn flip(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= flip_patch_count(lane)) { return; }
    let r = flip_patch(p);
    let edges = flip_edge_end(lane, r);
    for (var first = 0u; first < edges; first += STAGED_EDGES) {
        flip_stage(lane, r, first);
        for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += PATCH_LANES) {
            let c = flip_candidate(r, first, e);
            flip_request(c, !flip_has_corners(c) || should_flip(c.corners));
        }
    }
    flip_finish(lane, r, edges);
}
