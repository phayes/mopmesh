// Maps from mesh-wide element IDs to the local slots that hold them in each
// patch of an editable mesh (`GpuEditableMesh::maps`). Use them to find
// whether a patch already holds a copy of an element, and where.
//
// Importing this module binds the maps at group 1, binding 7, as read-write
// storage. That leaves bindings 0-6 for `flip_pass` or your own buffers.
//
// Each patch has one map per element kind. Lookups, sets and erases on one
// patch may run concurrently, with two rules:
// - At most one invocation may set or erase a given ID at a time.
// - A lookup that must see a set or erase made by another invocation in the
//   same dispatch must follow it across a `storageBarrier` (same workgroup)
//   or a dispatch boundary.
//
// To add IDs from many invocations at once without knowing whether another
// invocation adds the same ID, use `map_claim`, then `map_take` and
// `map_set` in a later dispatch; their own comments give their rules.
// `map_clear` empties a map so a workgroup can rebuild it.

@group(1) @binding(7) var<storage, read_write> maps: array<atomic<u32>>;

// Key of an entry that has never held an ID.
const MAP_EMPTY: u32 = 0xffffffffu;
// Key of an entry whose ID was erased. Lookups continue past it.
const MAP_TOMBSTONE: u32 = 0xfffffffeu;
// Words per patch at the start of the maps buffer.
const MAP_HEADER_WORDS: u32 = 6u;

// Returned by `map_find` for an ID the patch does not hold.
public const MAP_MISSING: u32 = 0xffffffffu;

// Returns a map's first probe position for `global`, before masking to its
// entry count.
public fn map_hash(global: u32) -> u32 {
    var x = global;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// One map's word offset and entry count (a power of two).
struct MapTable {
    offset: u32,
    entries: u32,
}

fn map_table(p: u32, kind: u32) -> MapTable {
    let at = MAP_HEADER_WORDS * p + 2u * kind;
    return MapTable(atomicLoad(&maps[at]), atomicLoad(&maps[at + 1u]));
}

// Returns the word offset of the entry holding `global`, or MAP_MISSING.
fn map_entry(t: MapTable, global: u32) -> u32 {
    var at = map_hash(global) & (t.entries - 1u);
    for (var probes = 0u; probes < t.entries; probes++) {
        let e = t.offset + 2u * at;
        let key = atomicLoad(&maps[e]);
        if (key == global) { return e; }
        if (key == MAP_EMPTY) { break; }
        at = (at + 1u) & (t.entries - 1u);
    }
    return MAP_MISSING;
}

// Returns the slot holding mesh-wide ID `global` of `kind` (VERTEX, EDGE or
// FACE from `topology::patch`) in patch `p`, or MAP_MISSING if the patch holds no copy of it.
// `p` must be below the patch count.
public fn map_find(p: u32, kind: u32, global: u32) -> u32 {
    let e = map_entry(map_table(p, kind), global);
    if (e == MAP_MISSING) { return MAP_MISSING; }
    return atomicLoad(&maps[e + 1u]);
}

// Records that patch `p` holds `global` of `kind` in `slot`, replacing any
// slot recorded for it before. Returns false, changing nothing, if the map
// has no free entry. `global` must be below 0xfffffffe.
public fn map_set(p: u32, kind: u32, global: u32, slot: u32) -> bool {
    // Replace the slot of an ID already present.
    let t = map_table(p, kind);
    let found = map_entry(t, global);
    if (found != MAP_MISSING) {
        atomicStore(&maps[found + 1u], slot);
        return true;
    }

    // Otherwise claim the first empty or erased entry along the probe
    // sequence. Another invocation may claim the same entry for another ID
    // first; then move on.
    var at = map_hash(global) & (t.entries - 1u);
    var probes = 0u;
    while (probes < t.entries) {
        let e = t.offset + 2u * at;
        let key = atomicLoad(&maps[e]);
        if (key == MAP_EMPTY || key == MAP_TOMBSTONE) {
            let claim = atomicCompareExchangeWeak(&maps[e], key, global);
            if (claim.exchanged) {
                atomicStore(&maps[e + 1u], slot);
                return true;
            }
            // A failed exchange that saw the expected key failed spuriously:
            // retry the same entry.
            if (claim.old_value == key) { continue; }
        }
        at = (at + 1u) & (t.entries - 1u);
        probes++;
    }
    return false;
}

// Results of `map_claim`.
// The ID was added, with no slot yet; `map_take` hands it to one caller.
public const MAP_CLAIM_NEW: u32 = 0u;
// The ID was already present, with or without a slot.
public const MAP_CLAIM_HELD: u32 = 1u;
// The ID was not present and the map has no never-used entry left.
public const MAP_CLAIM_FULL: u32 = 2u;

// Slot value of an entry handed out by `map_take`, until `map_set` stores
// the real slot.
const MAP_RESERVED: u32 = 0xfffffffeu;

// Adds mesh-wide ID `global` of `kind` to patch `p`'s map without a slot,
// unless it is already present. Any number of invocations may claim the
// same or different IDs at once, from any workgroups; each ID is added once.
// Erased entries are not reused, so a map can fill up before it holds as
// many IDs as it has entries. Don't call `map_set` or `map_erase` on this
// map in the same dispatch. `map_find` returns MAP_MISSING for an ID added
// here until `map_set` gives it a slot.
public fn map_claim(p: u32, kind: u32, global: u32) -> u32 {
    let t = map_table(p, kind);
    var at = map_hash(global) & (t.entries - 1u);
    var probes = 0u;
    while (probes < t.entries) {
        let e = t.offset + 2u * at;
        let key = atomicLoad(&maps[e]);
        if (key == global) { return MAP_CLAIM_HELD; }
        if (key == MAP_EMPTY) {
            let claim = atomicCompareExchangeWeak(&maps[e], MAP_EMPTY, global);
            if (claim.exchanged) { return MAP_CLAIM_NEW; }
            // Retry a spurious failure; otherwise another invocation took
            // the entry, perhaps for this same ID.
            if (claim.old_value == MAP_EMPTY) { continue; }
            if (claim.old_value == global) { return MAP_CLAIM_HELD; }
        }
        at = (at + 1u) & (t.entries - 1u);
        probes++;
    }
    return MAP_CLAIM_FULL;
}

// Returns true for exactly one caller per ID added by `map_claim` that has
// no slot yet; that caller must then store its slot with `map_set`. Returns
// false for every other caller, and for IDs that already have a slot or are
// not present. Use it in a later dispatch than the claims.
public fn map_take(p: u32, kind: u32, global: u32) -> bool {
    let e = map_entry(map_table(p, kind), global);
    if (e == MAP_MISSING) { return false; }
    var take = atomicCompareExchangeWeak(&maps[e + 1u], MAP_MISSING, MAP_RESERVED);
    // Retry spurious failures.
    while (!take.exchanged && take.old_value == MAP_MISSING) {
        take = atomicCompareExchangeWeak(&maps[e + 1u], MAP_MISSING, MAP_RESERVED);
    }
    return take.exchanged;
}

// Empties patch `p`'s map for `kind`, for rebuilding it with `map_set`.
// Spread the work over a workgroup: each invocation passes its own `first`
// (such as its local index) and the workgroup size as `step`. Nothing else
// may use this map until every invocation has finished and passed a
// `storageBarrier`.
public fn map_clear(p: u32, kind: u32, first: u32, step: u32) {
    let t = map_table(p, kind);
    for (var at = first; at < t.entries; at += step) {
        atomicStore(&maps[t.offset + 2u * at], MAP_EMPTY);
        atomicStore(&maps[t.offset + 2u * at + 1u], MAP_MISSING);
    }
}

// Removes `global` of `kind` from patch `p`'s map. Returns false if the
// patch held no copy of it.
public fn map_erase(p: u32, kind: u32, global: u32) -> bool {
    let e = map_entry(map_table(p, kind), global);
    if (e == MAP_MISSING) { return false; }
    atomicStore(&maps[e + 1u], MAP_MISSING);
    atomicStore(&maps[e], MAP_TOMBSTONE);
    return true;
}
