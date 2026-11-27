// Rounds that decide which pending edits may proceed together across patch
// borders, as RXMesh's patch locking does, but without waiting: no
// workgroup ever spins on another.
//
// An edit here is a cavity: the faces around up to four corner vertices,
// such as an edge flip's quad [a, b, c, d]. A cavity is made by one acting
// patch, which may take over faces, vertices and edges from other patches.
// Its footprint is every patch owning a face around one of its corners, or
// a vertex or edge of such a face. Each round:
//
//   1. `cavity_lead`, one workgroup per patch: each vertex's owner counts
//      how many faces around the vertex each patch owns.
//   2. `cavity_bid`, one invocation per request: pending requests with a
//      corner already used by an earlier selection become CAVITY_LOST; the
//      rest pick their acting patch (`cavity_actor`) and bid its round
//      priority on their four corners.
//   3. `cavity_relay_forward`, one workgroup per patch: each vertex's owner
//      raises the lock word of every patch in the footprint of the vertex's
//      star to the vertex's bid.
//   4. `cavity_relay_back`, one workgroup per patch: a vertex is lost if any
//      of those lock words holds a higher bid.
//   5. `cavity_check`, one invocation per request: a request becomes
//      CAVITY_WON if every corner carries its acting patch's bid and none is
//      lost.
//   6. `cavity_claim`, then `cavity_resolve`, one invocation per request
//      each: among one acting patch's won requests, those holding the highest
//      priority at all four corners become CAVITY_SELECTED and use up their
//      corners; the others become pending again. Resolve also flags the
//      acting patches and the owners of selected corners in the patch words
//      (CAVITY_ACTING_WORD, CAVITY_RING_WORD).
//
// To carry out the selected edits, run the stages of
// `topology::cavity_migrate` next; `cavity_conclude` runs among them and
// turns the requests of acting patches without room into CAVITY_FAILED.
//
// Selected requests of different acting patches have disjoint footprints,
// so their patches can edit independently. Every round selects at least one
// request, unless every pending one has lost to a used corner. Run the
// stages as separate dispatches in this order; each reads what the previous
// one wrote, after `cavity_plan`, which skips the whole round once no
// request is pending. Before each round, clear the vertex records and patch words to
// zero.
//
// Importing this module binds the mesh read-only through
// `topology::patch_arena` at group 1, and this module's state at group 2.

import super::patch::{PatchRecord, VERTEX, EDGE, FACE, PATCH_LANES};
import super::patch_arena::{
    slot_end, slot_active, slot_owned, face_edge, edge_vertices, global_vertex, global_edge,
    global_face, owner,
};
import super::directed_edge::directed_start;
import super::flip::flip_priority;

// Words per request: stage, acting patch, then four corner vertex IDs.
public const CAVITY_REQUEST_WORDS: u32 = 6u;
// Not requested.
public const CAVITY_NONE: u32 = 0u;
// Requested and not yet decided.
public const CAVITY_PENDING: u32 = 1u;
// Won its footprint this round; claim and resolve decide it.
public const CAVITY_WON: u32 = 2u;
// Selected: its acting patch may edit its footprint this round.
public const CAVITY_SELECTED: u32 = 3u;
// Shares a corner with a request selected in an earlier round.
public const CAVITY_LOST: u32 = 4u;
// Was selected, but its acting patch had no room for the copies it needed
// (`topology::cavity_migrate`). Its corners are free again.
public const CAVITY_FAILED: u32 = 5u;
// Was selected in an earlier round and carried out. Set it yourself after
// carrying out a selected request, so that CAVITY_SELECTED only marks this
// round's selections.
public const CAVITY_DONE: u32 = 6u;

// Words per mesh vertex in the round's vertex records.
public const CAVITY_VERTEX_WORDS: u32 = 14u;
// Offsets within a vertex record, written by `cavity_resolve` and
// `topology::cavity_migrate`: one more than the acting patch of the
// selected request using the vertex as a corner, and one more than the
// acting patch whose cavity faces have the vertex as a corner. 0 when unset.
public const CAVITY_CORNER: u32 = 12u;
public const CAVITY_RING: u32 = 13u;
// Words per patch in the round's patch words.
public const CAVITY_PATCH_WORDS: u32 = 8u;
// Offsets within a patch's words: the highest bid on the patch; nonzero if
// the patch's edits this round failed for lack of room; then how many new
// vertex, edge and face copies they need.
public const CAVITY_LOCK_WORD: u32 = 0u;
public const CAVITY_FAILED_WORD: u32 = 1u;
public const CAVITY_NEW_WORDS: u32 = 2u;
// Nonzero if the patch owns a corner of a request selected this round.
// Set by `cavity_resolve`.
public const CAVITY_RING_WORD: u32 = 5u;
// Nonzero if the patch is the acting patch of a request selected this
// round. Set by `cavity_resolve`.
public const CAVITY_ACTING_WORD: u32 = 6u;
// Nonzero if the patch is an acting patch, or owns a face, edge or vertex
// of a selected request's cavity faces: the only patches that
// `topology::cavity_migrate` changes this round. Set by `cavity_resolve`
// and `cavity_ring`.
public const CAVITY_INVOLVED_WORD: u32 = 7u;
// Owners counted around one vertex; a vertex with more is left out of
// choosing acting patches.
const LEADER_SLOTS: u32 = 4u;
// Offsets within a vertex record: the highest bid on the vertex; nonzero if
// that bid lost at some patch; the highest priority among won requests
// using the vertex; nonzero if more than LEADER_SLOTS patches own faces
// around it; then LEADER_SLOTS patch IDs plus one (0 when unused) and their
// face counts.
const BID: u32 = 0u;
const LOST: u32 = 1u;
const CLAIM: u32 = 2u;
const OVERFLOW: u32 = 3u;
const LEADERS: u32 = 4u;
const COUNTS: u32 = 8u;

const INVALID: u32 = 0xffffffffu;

// Per round: `seed` orders requests that share an acting patch; `round_seed`
// orders acting patches and must change every round; `request_count` is the
// number of requests; `flat_grid` and `patch_grid` are the workgroup counts
// `cavity_plan` writes while requests are pending.
struct RoundParams {
    seed: u32,
    round_seed: u32,
    request_count: u32,
    _pad: u32,
    flat_grid: vec2<u32>,
    patch_grid: vec2<u32>,
}

// Words of the plan buffer: the number of pending requests, then from
// CAVITY_PLAN_FLAT and CAVITY_PLAN_PATCHES, the three workgroup counts of an
// indirect dispatch for one invocation per request and for one workgroup
// per patch.
public const CAVITY_PLAN_WORDS: u32 = 12u;
public const CAVITY_PLAN_PENDING: u32 = 0u;
public const CAVITY_PLAN_FLAT: u32 = 4u;
public const CAVITY_PLAN_PATCHES: u32 = 8u;

// CAVITY_REQUEST_WORDS words per request; set the stage to CAVITY_PENDING
// and the corners before the first round. Corners that are not used must
// repeat a used corner.
@group(2) @binding(0) var<storage, read_write> requests: array<u32>;
// CAVITY_VERTEX_WORDS words per mesh vertex, cleared every round.
@group(2) @binding(1) var<storage, read_write> vertex_records: array<atomic<u32>>;
// CAVITY_PATCH_WORDS words per patch, cleared every round.
@group(2) @binding(2) var<storage, read_write> patch_words: array<atomic<u32>>;
// One word per mesh vertex: nonzero once a selected request used it. Keep
// it for every round of a batch.
@group(2) @binding(3) var<storage, read_write> taken: array<atomic<u32>>;
@group(2) @binding(4) var<uniform> round_params: RoundParams;
// CAVITY_PLAN_WORDS words. Set the pending count to the number of
// CAVITY_PENDING requests before the first round; the stages keep it
// current.
@group(2) @binding(5) var<storage, read_write> plan: array<atomic<u32>>;

fn stage_word(request: u32) -> u32 {
    return CAVITY_REQUEST_WORDS * request;
}

fn record_word(v: u32, field: u32) -> u32 {
    return CAVITY_VERTEX_WORDS * v + field;
}

fn corners(request: u32) -> vec4<u32> {
    let at = stage_word(request) + 2u;
    return vec4(requests[at], requests[at + 1u], requests[at + 2u], requests[at + 3u]);
}

// Returns acting patch `p`'s bid this round: nonzero, and different for
// every patch below 2^31.
fn patch_bid(p: u32) -> u32 {
    return flip_priority(p, round_params.round_seed | 0x80000000u);
}

// The three vertex slots of active face `f`, in the face's corner order.
fn face_vertex_slots(r: PatchRecord, f: u32) -> array<u32, 3> {
    var out: array<u32, 3>;
    for (var i = 0u; i < 3u; i++) {
        let fe = face_edge(r, f, i);
        out[i] = directed_start(edge_vertices(r, fe.x), fe.y != 0u);
    }
    return out;
}

// Counts one face owned by patch `tag - 1` around vertex `v`.
fn count_leader(v: u32, tag: u32) {
    var k = 0u;
    while (k < LEADER_SLOTS) {
        let slot = record_word(v, LEADERS + k);
        let seen = atomicLoad(&vertex_records[slot]);
        if (seen == 0u) {
            // Claim the empty slot; if that fails, read the slot again.
            if (!atomicCompareExchangeWeak(&vertex_records[slot], 0u, tag).exchanged) {
                continue;
            }
            atomicAdd(&vertex_records[record_word(v, COUNTS + k)], 1u);
            return;
        }
        if (seen == tag) {
            atomicAdd(&vertex_records[record_word(v, COUNTS + k)], 1u);
            return;
        }
        k++;
    }
    atomicStore(&vertex_records[record_word(v, OVERFLOW)], 1u);
}

// Stage 1 for patch `r`, from a workgroup of PATCH_LANES invocations.
public fn cavity_lead(lane: u32, r: PatchRecord) {
    let faces = slot_end(r, FACE);
    for (var f = lane; f < faces; f += PATCH_LANES) {
        if (!slot_active(r, FACE, f)) { continue; }
        let vertices = face_vertex_slots(r, f);
        let tag = owner(FACE, global_face(r, f)).x + 1u;
        for (var i = 0u; i < 3u; i++) {
            if (slot_owned(r, VERTEX, vertices[i])) {
                count_leader(global_vertex(r, vertices[i]), tag);
            }
        }
    }
}

// Returns how many faces around vertex `v` patch `tag - 1` owns, or 0 if
// the vertex has too many owners to count.
fn leader_count(v: u32, tag: u32) -> u32 {
    if (atomicLoad(&vertex_records[record_word(v, OVERFLOW)]) != 0u) { return 0u; }
    for (var k = 0u; k < LEADER_SLOTS; k++) {
        if (atomicLoad(&vertex_records[record_word(v, LEADERS + k)]) == tag) {
            return atomicLoad(&vertex_records[record_word(v, COUNTS + k)]);
        }
    }
    return 0u;
}

// Whichever owner of corners a and b owns more of the four corners, the
// lower ID on ties.
fn endpoint_actor(c: vec4<u32>) -> u32 {
    let owners = vec4(
        owner(VERTEX, c.x).x, owner(VERTEX, c.y).x, owner(VERTEX, c.z).x, owner(VERTEX, c.w).x,
    );
    let count_a = dot(vec4<u32>(owners == vec4(owners.x)), vec4(1u));
    let count_b = dot(vec4<u32>(owners == vec4(owners.y)), vec4(1u));
    if (count_b > count_a || (count_b == count_a && owners.y < owners.x)) {
        return owners.y;
    }
    return owners.x;
}

// Returns the patch that makes a cavity with corners `c`, after stage 1:
// the patch owning the most faces, summed over the corners' stars, the
// lower ID on ties. Stars with more than LEADER_SLOTS owners are left out;
// if all four are, the owner of a or b that owns more corners.
public fn cavity_actor(c: vec4<u32>) -> u32 {
    var best = INVALID;
    var best_score = 0u;
    for (var i = 0u; i < 4u; i++) {
        for (var k = 0u; k < LEADER_SLOTS; k++) {
            let tag = atomicLoad(&vertex_records[record_word(c[i], LEADERS + k)]);
            if (tag == 0u || leader_count(c[i], tag) == 0u) { continue; }
            var score = 0u;
            for (var j = 0u; j < 4u; j++) {
                score += leader_count(c[j], tag);
            }
            let p = tag - 1u;
            if (score > best_score || (score == best_score && p < best)) {
                best = p;
                best_score = score;
            }
        }
    }
    if (best == INVALID) { return endpoint_actor(c); }
    return best;
}

// Stage 2 for request `request`; skip requests at or above request_count.
public fn cavity_bid(request: u32) {
    let at = stage_word(request);
    if (requests[at] != CAVITY_PENDING) { return; }
    let c = corners(request);
    for (var i = 0u; i < 4u; i++) {
        if (atomicLoad(&taken[c[i]]) != 0u) {
            requests[at] = CAVITY_LOST;
            atomicSub(&plan[CAVITY_PLAN_PENDING], 1u);
            return;
        }
    }
    let actor = cavity_actor(c);
    requests[at + 1u] = actor;
    for (var i = 0u; i < 4u; i++) {
        atomicMax(&vertex_records[record_word(c[i], BID)], patch_bid(actor));
    }
}

// The patches owning face `f`, its three vertices and its three edges.
fn face_footprint(r: PatchRecord, f: u32, vertices: array<u32, 3>) -> array<u32, 7> {
    var out: array<u32, 7>;
    out[0] = owner(FACE, global_face(r, f)).x;
    for (var i = 0u; i < 3u; i++) {
        out[1u + i] = owner(VERTEX, global_vertex(r, vertices[i])).x;
        out[4u + i] = owner(EDGE, global_edge(r, face_edge(r, f, i).x)).x;
    }
    return out;
}

// Visits every active face of patch `r` around a vertex this patch owns
// that carries a bid. `forward` raises the footprint's lock words to the
// bid; otherwise, marks the vertex lost if any lock word differs from it.
fn relay(lane: u32, r: PatchRecord, forward: bool) {
    let faces = slot_end(r, FACE);
    for (var f = lane; f < faces; f += PATCH_LANES) {
        if (!slot_active(r, FACE, f)) { continue; }
        let vertices = face_vertex_slots(r, f);

        // The bids on the face's vertices that this patch owns.
        var bids = vec3(0u);
        var globals = vec3(0u);
        for (var i = 0u; i < 3u; i++) {
            if (!slot_owned(r, VERTEX, vertices[i])) { continue; }
            globals[i] = global_vertex(r, vertices[i]);
            bids[i] = atomicLoad(&vertex_records[record_word(globals[i], BID)]);
        }
        let top = max(bids.x, max(bids.y, bids.z));
        if (top == 0u) { continue; }

        // Each distinct patch of the face's footprint once.
        let patches = face_footprint(r, f, vertices);
        for (var k = 0u; k < 7u; k++) {
            var repeat = false;
            for (var j = 0u; j < k; j++) { repeat = repeat || patches[j] == patches[k]; }
            if (repeat) { continue; }
            let lock = CAVITY_PATCH_WORDS * patches[k] + CAVITY_LOCK_WORD;
            if (forward) {
                atomicMax(&patch_words[lock], top);
                continue;
            }
            let held = atomicLoad(&patch_words[lock]);
            for (var i = 0u; i < 3u; i++) {
                if (bids[i] != 0u && held != bids[i]) {
                    atomicStore(&vertex_records[record_word(globals[i], LOST)], 1u);
                }
            }
        }
    }
}

// Stage 3 for patch `r`, from a workgroup of PATCH_LANES invocations.
public fn cavity_relay_forward(lane: u32, r: PatchRecord) {
    relay(lane, r, true);
}

// Stage 4 for patch `r`, from a workgroup of PATCH_LANES invocations.
public fn cavity_relay_back(lane: u32, r: PatchRecord) {
    relay(lane, r, false);
}

// Stage 5 for request `request`.
public fn cavity_check(request: u32) {
    let at = stage_word(request);
    if (requests[at] != CAVITY_PENDING) { return; }
    let bid = patch_bid(requests[at + 1u]);
    let c = corners(request);
    for (var i = 0u; i < 4u; i++) {
        let won = atomicLoad(&vertex_records[record_word(c[i], BID)]) == bid;
        if (!won || atomicLoad(&vertex_records[record_word(c[i], LOST)]) != 0u) {
            return;
        }
    }
    requests[at] = CAVITY_WON;
}

// Stage 6a for request `request`.
public fn cavity_claim(request: u32) {
    if (requests[stage_word(request)] != CAVITY_WON) { return; }
    let c = corners(request);
    for (var i = 0u; i < 4u; i++) {
        atomicMax(&vertex_records[record_word(c[i], CLAIM)], flip_priority(request, round_params.seed));
    }
}

// Stage 6b for request `request`.
public fn cavity_resolve(request: u32) {
    let at = stage_word(request);
    if (requests[at] != CAVITY_WON) { return; }
    let c = corners(request);
    let priority = flip_priority(request, round_params.seed);
    for (var i = 0u; i < 4u; i++) {
        if (atomicLoad(&vertex_records[record_word(c[i], CLAIM)]) != priority) {
            requests[at] = CAVITY_PENDING;
            return;
        }
    }
    requests[at] = CAVITY_SELECTED;
    atomicSub(&plan[CAVITY_PLAN_PENDING], 1u);
    let actor = requests[at + 1u];
    for (var i = 0u; i < 4u; i++) {
        atomicStore(&taken[c[i]], 1u);
        atomicStore(&vertex_records[record_word(c[i], CAVITY_CORNER)], actor + 1u);
        let corner_owner = owner(VERTEX, c[i]).x;
        atomicStore(&patch_words[CAVITY_PATCH_WORDS * corner_owner + CAVITY_RING_WORD], 1u);
    }

    // Flag the acting patch, which every later stage of the round changes.
    atomicStore(&patch_words[CAVITY_PATCH_WORDS * actor + CAVITY_ACTING_WORD], 1u);
    atomicStore(&patch_words[CAVITY_PATCH_WORDS * actor + CAVITY_INVOLVED_WORD], 1u);
}

// After `cavity_preflight` from `topology::cavity_migrate`, for request
// `request`: a selected request whose acting patch failed becomes
// CAVITY_FAILED and frees its corners.
public fn cavity_conclude(request: u32) {
    let at = stage_word(request);
    if (requests[at] != CAVITY_SELECTED) { return; }
    let actor = requests[at + 1u];
    let failed = CAVITY_PATCH_WORDS * actor + CAVITY_FAILED_WORD;
    if (atomicLoad(&patch_words[failed]) == 0u) { return; }

    // Only requests selected this round carry this round's corner marks.
    let c = corners(request);
    let mark = atomicLoad(&vertex_records[record_word(c.x, CAVITY_CORNER)]);
    if (mark != actor + 1u) { return; }
    requests[at] = CAVITY_FAILED;
    for (var i = 0u; i < 4u; i++) {
        atomicStore(&taken[c[i]], 0u);
    }
}

// Before a round's first stage, from a single invocation: writes the
// round's indirect dispatch sizes into the plan buffer, the full grids
// while any request is pending and zero workgroups otherwise. Record every
// stage of the round, migration included, with these sizes.
public fn cavity_plan() {
    let busy = atomicLoad(&plan[CAVITY_PLAN_PENDING]) != 0u;
    let flat = select(vec2(0u, 1u), round_params.flat_grid, busy);
    let patches = select(vec2(0u, 1u), round_params.patch_grid, busy);
    atomicStore(&plan[CAVITY_PLAN_FLAT], flat.x);
    atomicStore(&plan[CAVITY_PLAN_FLAT + 1u], flat.y);
    atomicStore(&plan[CAVITY_PLAN_FLAT + 2u], 1u);
    atomicStore(&plan[CAVITY_PLAN_PATCHES], patches.x);
    atomicStore(&plan[CAVITY_PLAN_PATCHES + 1u], patches.y);
    atomicStore(&plan[CAVITY_PLAN_PATCHES + 2u], 1u);
}

// Returns the number of requests, for skipping spare invocations.
public fn cavity_request_count() -> u32 {
    return round_params.request_count;
}
