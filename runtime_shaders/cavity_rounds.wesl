// Entry points for the stages of one cavity round, in dispatch order:
// lead, bid, relay_forward, relay_back, check, claim, resolve.
import package::shaders::topology::patch::PATCH_LANES;
import package::shaders::topology::patch_arena::{patch_count, patch_record};
import package::shaders::topology::cavity_rounds::{
    cavity_lead, cavity_bid, cavity_relay_forward, cavity_relay_back, cavity_check,
    cavity_claim, cavity_resolve, cavity_conclude, cavity_request_count, cavity_plan,
};
import package::shaders::parallel::grid::{flat_index, flat_group};

@compute @workgroup_size(PATCH_LANES)
fn lead(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < patch_count()) { cavity_lead(lane, patch_record(p)); }
}

@compute @workgroup_size(64)
fn bid(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(id, groups);
    if (i < cavity_request_count()) { cavity_bid(i); }
}

@compute @workgroup_size(PATCH_LANES)
fn relay_forward(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < patch_count()) { cavity_relay_forward(lane, patch_record(p)); }
}

@compute @workgroup_size(PATCH_LANES)
fn relay_back(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p < patch_count()) { cavity_relay_back(lane, patch_record(p)); }
}

@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(id, groups);
    if (i < cavity_request_count()) { cavity_check(i); }
}

@compute @workgroup_size(64)
fn claim(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(id, groups);
    if (i < cavity_request_count()) { cavity_claim(i); }
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(id, groups);
    if (i < cavity_request_count()) { cavity_resolve(i); }
}

// Runs once, before each round's first stage.
@compute @workgroup_size(1)
fn plan() {
    cavity_plan();
}

// Runs between the migration stages; see `topology::cavity_migrate`.
@compute @workgroup_size(64)
fn conclude(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(id, groups);
    if (i < cavity_request_count()) { cavity_conclude(i); }
}
