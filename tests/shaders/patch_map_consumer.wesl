// A downstream kernel that runs lookups, sets and erases on an editable
// mesh's maps through `topology::patch_map`, which binds the maps at
// group 1, binding 7.
import mopmesh::topology::patch_map::{map_find, map_set, map_erase};

// Five words per operation: code (0 find, 1 set, 2 erase), patch, kind,
// mesh-wide ID and slot. For `concurrent`: patch, kind, then one ID per lane.
@group(0) @binding(0) var<storage, read> ops: array<u32>;
// One result per operation: the slot found, or 1/0 for a set or erase.
@group(0) @binding(1) var<storage, read_write> results: array<u32>;

// Runs every operation in order in one invocation.
@compute @workgroup_size(1)
fn sequential() {
    for (var i = 0u; i < arrayLength(&results); i++) {
        let o = 5u * i;
        let code = ops[o];
        let p = ops[o + 1u];
        let kind = ops[o + 2u];
        let global = ops[o + 3u];
        let slot = ops[o + 4u];
        if (code == 0u) {
            results[i] = map_find(p, kind, global);
        } else if (code == 1u) {
            results[i] = select(0u, 1u, map_set(p, kind, global, slot));
        } else {
            results[i] = select(0u, 1u, map_erase(p, kind, global));
        }
    }
}

// Each of 64 lanes sets its own ID at slot 1000 + lane, then, after a
// barrier, looks it up. Results: 64 set outcomes, then 64 slots found.
@compute @workgroup_size(64)
fn concurrent(@builtin(local_invocation_index) lane: u32) {
    let p = ops[0];
    let kind = ops[1];
    let global = ops[2u + lane];
    results[lane] = select(0u, 1u, map_set(p, kind, global, 1000u + lane));
    storageBarrier();
    results[64u + lane] = map_find(p, kind, global);
}
