// Flat indices for 64-wide workgroups dispatched as a two-dimensional grid,
// which lets one dispatch cover more than 65,535 workgroups.

// The flat index of this invocation. Pass the global_invocation_id and
// num_workgroups builtins. The result can exceed the item count in the last
// row, so check it against the count as usual.
public fn flat_index(id: vec3<u32>, groups: vec3<u32>) -> u32 {
    return id.x + id.y * groups.x * 64u;
}

// The flat index of this invocation's workgroup. Pass the workgroup_id and
// num_workgroups builtins.
public fn flat_group(group: vec3<u32>, groups: vec3<u32>) -> u32 {
    return group.x + group.y * groups.x;
}
