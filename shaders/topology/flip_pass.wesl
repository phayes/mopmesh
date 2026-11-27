// Flip many edges at once from your own compute shaders, with your own rule
// for which edges to flip. An edge flip replaces the shared edge of two
// triangles with the other diagonal of their four-vertex region.
//
// Use `EdgeFlipQuery::pass_layout` for group 1 in the pipeline layout and
// `EdgeFlipQuery::begin_pass` to create its bindings. Group 0 is available
// for your own data. Do not also import `topology::patch_arena`: it uses
// group 1 too.
//
// A batch is one dispatch with one workgroup of PATCH_LANES invocations
// (from `topology::patch`) per patch (`FlipPass::patch_grid`). Every active
// edge the patch owns is offered to your rule once, with its four corners;
// then the workgroup picks non-conflicting flips and applies them:
//
//     @compute @workgroup_size(PATCH_LANES)
//     fn flip(@builtin(workgroup_id) group: vec3<u32>,
//             @builtin(num_workgroups) groups: vec3<u32>,
//             @builtin(local_invocation_index) lane: u32) {
//         let p = flat_group(group, groups);
//         if (p >= flip_patch_count(lane)) { return; }
//         let r = flip_patch(p);
//         let edges = flip_edge_end(lane, r);
//         for (var first = 0u; first < edges; first += STAGED_EDGES) {
//             flip_stage(lane, r, first);
//             for (var e = first + lane; e < min(edges, first + STAGED_EDGES); e += PATCH_LANES) {
//                 let c = flip_candidate(r, first, e);
//                 flip_request(c, c.status != FLIP_DONE || my_rule(c.corners));
//             }
//         }
//         flip_finish(lane, r, edges);
//     }
//
// Call flip_patch_count, flip_edge_end, flip_stage and flip_finish from
// every invocation, in uniform control flow, exactly as above.
//
// After the dispatch, each edge's outcome is in the status buffer, indexed
// by mesh-wide edge ID. Flips that share any of their four vertices compete,
// and at most one of them succeeds. An edge that reports FLIP_LOST_CLAIM can
// be tried again in a later batch. Flips reuse their own edge and face
// slots, so they need no free slots.

import super::patch::{
    PatchRecord, EDGE, FACE, PATCH_COUNT_WORD, RECORDS_WORD,
    record_offset, decode_record, used_offset, mask_word, mask_test,
    face_edge_offset, edge_vertices_offset, global_vertex_offset, global_edge_offset,
    PATCH_LANES,
};
import super::flip::{
    FaceEdges, FlipQuad, flip_quad, flip_in_place_writes, flip_priority, runs_stored_order,
    FLIP_IN_PLACE_WRITES, FLIP_NOT_REQUESTED, FLIP_DONE, FLIP_BOUNDARY, FLIP_NOT_INTERIOR,
    FLIP_CONNECTED, FLIP_LOST_CLAIM,
};
import super::patch_staging::{clear_staged_edges, stage_face_edge, staged_edge_faces};
import super::edge::INVALID_FACE;

const INVALID: u32 = 0xffffffffu;

// Selection rounds per batch. Each round decides at least the request with
// the highest priority among those left, and usually far more. Requests
// still undecided after the last round report FLIP_LOST_CLAIM.
const MAX_ROUNDS: u32 = 64u;

struct PassParams { seed: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(1) @binding(0) var<storage, read_write> arena: array<u32>;
@group(1) @binding(1) var<storage, read_write> masks: array<atomic<u32>>;
// One bid per mesh vertex, reset each selection round.
@group(1) @binding(2) var<storage, read_write> claims: array<atomic<u32>>;
// Temporary flip data, STATE_WORDS zero-initialized words per mesh edge.
@group(1) @binding(3) var<storage, read_write> state: array<u32>;
@group(1) @binding(4) var<uniform> pass_params: PassParams;
// One FLIP_* outcome per mesh edge, zero-initialized.
@group(1) @binding(5) var<storage, read_write> statuses: array<u32>;
// Per mesh vertex: one more than the edge ID of the selected flip that uses
// this vertex, or zero. Zero-initialized.
@group(1) @binding(6) var<storage, read_write> partners: array<u32>;

// Words per edge: a stage marker, the two face slots, and the four corners'
// mesh-wide vertex IDs.
const STATE_WORDS: u32 = 7u;
const STATE_STAGE: u32 = 0u;
const STATE_FACES: u32 = 1u;
const STATE_CORNERS: u32 = 3u;
// Requested and eligible, not yet decided.
const STAGE_CANDIDATE: u32 = 1u;
// Won its corners this round; not yet checked for an existing c-d edge.
const STAGE_TENTATIVE: u32 = 2u;
// Won its corners, but c and d are already joined.
const STAGE_CONNECTED: u32 = 3u;
// Selected; applied at the end of the batch.
const STAGE_SELECTED: u32 = 4u;

var<workgroup> shared_word: u32;
var<workgroup> undecided: atomic<u32>;

// An edge offered to your rule by `flip_candidate`.
// `status == FLIP_DONE` means the edge can be requested; it hasn't flipped.
// Any other status says why it can't flip; only read `corners` when the
// status is FLIP_DONE.
//
// `corners` contains mesh-wide vertex IDs [a, b, c, d]. The current edge
// joins a to b, with a -> b in the edge's stored order; the replacement would
// join c to d. The two triangles are oriented a -> b -> c and b -> a -> d.
// `edge` is the mesh-wide edge ID. Pass the candidate unchanged to
// `flip_request`; its remaining fields are not inputs for your rule.
public struct FlipCandidate {
    status: u32,
    corners: array<u32, 4>,
    edge: u32,
    first_face: u32,
    second_face: u32,
}

// Shares a value from the first invocation with the whole workgroup.
fn broadcast(lane: u32, value: u32) -> u32 {
    if (lane == 0u) { shared_word = value; }
    return workgroupUniformLoad(&shared_word);
}

fn record(patch_id: u32) -> PatchRecord {
    let at = record_offset(arena[RECORDS_WORD], patch_id);
    var words: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { words[i] = arena[at + i]; }
    return decode_record(words);
}
fn bit(mask: u32, slot: u32) -> bool {
    return mask_test(atomicLoad(&masks[mask_word(mask, slot)]), slot);
}
fn face_edges(r: PatchRecord, slot: u32) -> FaceEdges {
    var face: FaceEdges;
    for (var i = 0u; i < 3u; i++) {
        let at = face_edge_offset(r, slot, i);
        face.edges[i] = arena[at];
        face.reversed[i] = arena[at + 1u];
    }
    return face;
}
fn edge_vertices(r: PatchRecord, slot: u32) -> vec2<u32> {
    let at = edge_vertices_offset(r, slot);
    return vec2(arena[at], arena[at + 1u]);
}
fn face_edge_vertices(r: PatchRecord, face: FaceEdges) -> array<vec2<u32>, 3> {
    return array(
        edge_vertices(r, face.edges.x),
        edge_vertices(r, face.edges.y),
        edge_vertices(r, face.edges.z),
    );
}
fn global_vertex(r: PatchRecord, slot: u32) -> u32 {
    return arena[global_vertex_offset(r, slot)];
}

// Builds the four-vertex region formed by the two triangles sharing an edge.
fn quad_of(r: PatchRecord, edge: u32, first: FaceEdges, second: FaceEdges) -> FlipQuad {
    return flip_quad(edge, first, face_edge_vertices(r, first), second, face_edge_vertices(r, second));
}

// Returns the number of patches, the same in every invocation. A workgroup
// whose index (parallel::grid::flat_group) is not below it must return
// without calling the other flip functions.
public fn flip_patch_count(lane: u32) -> u32 {
    return broadcast(lane, arena[PATCH_COUNT_WORD]);
}

// Returns the record of patch `p`, which must be less than flip_patch_count.
public fn flip_patch(p: u32) -> PatchRecord {
    return record(p);
}

// Returns the bound on the patch's edge slots, the same in every invocation,
// so it can bound the chunk loop.
public fn flip_edge_end(lane: u32, r: PatchRecord) -> u32 {
    return broadcast(lane, atomicLoad(&masks[used_offset(r, EDGE)]));
}

// Finds the two faces of every edge slot in the chunk of STAGED_EDGES slots
// starting at `first`, for flip_candidate.
public fn flip_stage(lane: u32, r: PatchRecord, first: u32) {
    // Wait until the previous chunk is no longer being read.
    workgroupBarrier();
    clear_staged_edges(lane, atomicLoad(&masks[used_offset(r, EDGE)]) - first);
    workgroupBarrier();
    let faces = atomicLoad(&masks[used_offset(r, FACE)]);
    for (var f = lane; f < faces; f += PATCH_LANES) {
        if (!bit(r.active_masks[FACE], f)) { continue; }
        let face = face_edges(r, f);
        for (var i = 0u; i < 3u; i++) {
            stage_face_edge(first, face.edges[i], face.reversed[i], f);
        }
    }
    workgroupBarrier();
}

// Checks edge slot `e` of the chunk staged at `first`. Skipped by
// flip_request when the slot is empty or another patch owns its edge.
// Otherwise the edge must join two triangles forming a proper four-sided
// region, and all four vertices must be interior to this patch. This check
// doesn't look at positions, competing flips, or an existing c-d edge;
// flip_finish checks the last two.
public fn flip_candidate(r: PatchRecord, first: u32, e: u32) -> FlipCandidate {
    var out: FlipCandidate;
    out.status = FLIP_NOT_REQUESTED;
    out.edge = INVALID;
    if (!bit(r.active_masks[EDGE], e) || !bit(r.owned_masks[EDGE], e)) { return out; }
    out.edge = arena[global_edge_offset(r, e)];

    // Both faces must exist. Put the face that runs the edge in its stored
    // order first, so each replacement face starts at the documented corner.
    let staged = staged_edge_faces(first, e);
    out.status = FLIP_BOUNDARY;
    if (staged.y == INVALID_FACE) { return out; }
    var faces = array(staged.x, staged.y);
    if (!runs_stored_order(face_edges(r, faces[0]), e)) {
        faces = array(faces[1], faces[0]);
    }
    let quad = quad_of(r, e, face_edges(r, faces[0]), face_edges(r, faces[1]));
    out.status = quad.status;
    if (quad.status != FLIP_DONE) { return out; }

    // All affected vertices must stay within the patch.
    let local = array(quad.a, quad.b, quad.c, quad.d);
    out.status = FLIP_NOT_INTERIOR;
    for (var k = 0u; k < 4u; k++) {
        if (!bit(r.interior_mask, local[k])) { return out; }
    }

    out.status = FLIP_DONE;
    for (var k = 0u; k < 4u; k++) { out.corners[k] = global_vertex(r, local[k]); }
    out.first_face = faces[0];
    out.second_face = faces[1];
    return out;
}

// Records your decision for a candidate. With `wanted` false, the edge is
// left alone and reports FLIP_NOT_REQUESTED. With `wanted` true, an edge
// that can't flip reports why, and an eligible edge becomes a request that
// flip_finish settles. Call once per candidate, after its flip_candidate
// call.
public fn flip_request(candidate: FlipCandidate, wanted: bool) {
    let edge = candidate.edge;
    if (edge == INVALID) { return; }
    if (!wanted) {
        statuses[edge] = FLIP_NOT_REQUESTED;
        return;
    }
    statuses[edge] = candidate.status;
    if (candidate.status != FLIP_DONE) { return; }

    // Save the flip for flip_finish.
    let base = edge * STATE_WORDS;
    state[base + STATE_FACES] = candidate.first_face;
    state[base + STATE_FACES + 1u] = candidate.second_face;
    for (var k = 0u; k < 4u; k++) { state[base + STATE_CORNERS + k] = candidate.corners[k]; }
    state[base + STATE_STAGE] = STAGE_CANDIDATE;
}

// Returns the mesh-wide ID of the edge in slot `e` if this patch owns it and
// its saved stage is `stage`, or INVALID.
fn edge_at_stage(r: PatchRecord, e: u32, stage: u32) -> u32 {
    if (!bit(r.active_masks[EDGE], e) || !bit(r.owned_masks[EDGE], e)) { return INVALID; }
    let edge = arena[global_edge_offset(r, e)];
    if (state[edge * STATE_WORDS + STATE_STAGE] != stage) { return INVALID; }
    return edge;
}

fn corner(edge: u32, k: u32) -> u32 {
    return state[edge * STATE_WORDS + STATE_CORNERS + k];
}

// Runs one selection round over the patch's requests and returns how many
// remain undecided, the same in every invocation.
fn select_round(lane: u32, r: PatchRecord, edges: u32) -> u32 {
    if (lane == 0u) { atomicStore(&undecided, 0u); }
    workgroupBarrier();

    // Clear the bids at every undecided request's corners.
    for (var e = lane; e < edges; e += PATCH_LANES) {
        let edge = edge_at_stage(r, e, STAGE_CANDIDATE);
        if (edge == INVALID) { continue; }
        for (var k = 0u; k < 4u; k++) { atomicStore(&claims[corner(edge, k)], 0u); }
    }
    storageBarrier();

    // A request touching a selected flip loses; the others bid.
    for (var e = lane; e < edges; e += PATCH_LANES) {
        let edge = edge_at_stage(r, e, STAGE_CANDIDATE);
        if (edge == INVALID) { continue; }
        var taken = false;
        for (var k = 0u; k < 4u; k++) { taken = taken || partners[corner(edge, k)] != 0u; }
        if (taken) {
            state[edge * STATE_WORDS + STATE_STAGE] = 0u;
            statuses[edge] = FLIP_LOST_CLAIM;
            continue;
        }
        let priority = flip_priority(edge, pass_params.seed);
        for (var k = 0u; k < 4u; k++) { atomicMax(&claims[corner(edge, k)], priority); }
    }
    storageBarrier();

    // A request with the highest bid at all four corners wins them for now,
    // and marks the ends of its new edge.
    for (var e = lane; e < edges; e += PATCH_LANES) {
        let edge = edge_at_stage(r, e, STAGE_CANDIDATE);
        if (edge == INVALID) { continue; }
        let priority = flip_priority(edge, pass_params.seed);
        var won = true;
        for (var k = 0u; k < 4u; k++) {
            won = won && atomicLoad(&claims[corner(edge, k)]) == priority;
        }
        if (won) {
            state[edge * STATE_WORDS + STATE_STAGE] = STAGE_TENTATIVE;
            partners[corner(edge, 2u)] = edge + 1u;
            partners[corner(edge, 3u)] = edge + 1u;
        }
    }
    storageBarrier();

    // Find winners whose new edge already exists. Their ends are interior to
    // this patch, so such an edge is in this patch with both ends interior;
    // other edges are skipped, so only this patch's vertices are looked up.
    for (var e = lane; e < edges; e += PATCH_LANES) {
        if (!bit(r.active_masks[EDGE], e)) { continue; }
        let ends = edge_vertices(r, e);
        if (!bit(r.interior_mask, ends.x) || !bit(r.interior_mask, ends.y)) { continue; }
        let u = global_vertex(r, ends.x);
        let w = global_vertex(r, ends.y);
        let marked = partners[u];
        if (marked == 0u) { continue; }
        let winner = marked - 1u;
        let base = winner * STATE_WORDS;
        if (state[base + STATE_STAGE] != STAGE_TENTATIVE) { continue; }
        let c = state[base + STATE_CORNERS + 2u];
        let d = state[base + STATE_CORNERS + 3u];
        if ((c == u && d == w) || (c == w && d == u)) {
            state[base + STATE_STAGE] = STAGE_CONNECTED;
        }
    }
    storageBarrier();

    // Drop winners whose new edge exists, freeing their corners, and select
    // the rest, marking all four of their corners.
    for (var e = lane; e < edges; e += PATCH_LANES) {
        if (!bit(r.active_masks[EDGE], e) || !bit(r.owned_masks[EDGE], e)) { continue; }
        let edge = arena[global_edge_offset(r, e)];
        let base = edge * STATE_WORDS;
        let stage = state[base + STATE_STAGE];
        if (stage == STAGE_CONNECTED) {
            state[base + STATE_STAGE] = 0u;
            statuses[edge] = FLIP_CONNECTED;
            partners[corner(edge, 2u)] = 0u;
            partners[corner(edge, 3u)] = 0u;
        } else if (stage == STAGE_TENTATIVE) {
            state[base + STATE_STAGE] = STAGE_SELECTED;
            partners[corner(edge, 0u)] = edge + 1u;
            partners[corner(edge, 1u)] = edge + 1u;
        } else if (stage == STAGE_CANDIDATE) {
            atomicAdd(&undecided, 1u);
        }
    }
    storageBarrier();
    workgroupBarrier();
    return broadcast(lane, atomicLoad(&undecided));
}

// Settles every request made in this patch and applies the selected flips,
// after every candidate has been passed to flip_request. Requests are
// selected in decreasing priority: a request goes ahead unless an
// already-selected flip uses one of its corners (FLIP_LOST_CLAIM) or c and d
// are already joined by an edge (FLIP_CONNECTED). Selected flips report
// FLIP_DONE. `edges` is the value returned by flip_edge_end.
public fn flip_finish(lane: u32, r: PatchRecord, edges: u32) {
    // Every flip that shares a corner with one in this patch is in this patch,
    // so the requests are complete once this workgroup has made them.
    storageBarrier();
    for (var round = 0u; round < MAX_ROUNDS; round++) {
        if (select_round(lane, r, edges) == 0u) { break; }
    }

    // Report requests still undecided, then apply the selected flips. They
    // share no vertices, so each reads and writes only its own rows.
    for (var e = lane; e < edges; e += PATCH_LANES) {
        if (!bit(r.active_masks[EDGE], e) || !bit(r.owned_masks[EDGE], e)) { continue; }
        let edge = arena[global_edge_offset(r, e)];
        let base = edge * STATE_WORDS;
        let stage = state[base + STATE_STAGE];
        if (stage == STAGE_CANDIDATE) {
            statuses[edge] = FLIP_LOST_CLAIM;
            continue;
        }
        if (stage != STAGE_SELECTED) { continue; }
        let first_face = state[base + STATE_FACES];
        let second_face = state[base + STATE_FACES + 1u];
        let first = face_edges(r, first_face);
        let second = face_edges(r, second_face);
        let quad = quad_of(r, e, first, second);
        let writes = flip_in_place_writes(
            r, e, first_face, second_face, quad, first, second,
            state[base + STATE_CORNERS + 2u], state[base + STATE_CORNERS + 3u],
        );
        for (var k = 0u; k < FLIP_IN_PLACE_WRITES; k++) {
            arena[writes.arena[k].offset] = writes.arena[k].value;
        }
    }
}
