// Entry points for the stages that carry out a cavity round's selected
// edits, in dispatch order: ring, claim, preflight, then `conclude` from
// the cavity_rounds entry points, then fill_vertices, fill_edges,
// fill_faces, transfer and cleanup. Each runs one workgroup per patch.
import package::shaders::topology::patch::{PATCH_LANES, VERTEX, EDGE, FACE};
import package::shaders::topology::cavity_migrate::{
    cavity_migrate_patch_count, cavity_ring, cavity_claim, cavity_preflight, cavity_fill,
    cavity_transfer, cavity_cleanup,
};
import package::shaders::parallel::grid::flat_group;

@compute @workgroup_size(PATCH_LANES)
fn ring(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_ring(lane, p); }
}

@compute @workgroup_size(PATCH_LANES)
fn claim(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_claim(lane, p); }
}

@compute @workgroup_size(PATCH_LANES)
fn preflight(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_preflight(lane, p); }
}

@compute @workgroup_size(PATCH_LANES)
fn fill_vertices(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_fill(lane, p, VERTEX); }
}

@compute @workgroup_size(PATCH_LANES)
fn fill_edges(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_fill(lane, p, EDGE); }
}

@compute @workgroup_size(PATCH_LANES)
fn fill_faces(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_fill(lane, p, FACE); }
}

@compute @workgroup_size(PATCH_LANES)
fn transfer(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_transfer(lane, p); }
}

@compute @workgroup_size(PATCH_LANES)
fn cleanup(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < cavity_migrate_patch_count(lane)) { cavity_cleanup(lane, p); }
}
