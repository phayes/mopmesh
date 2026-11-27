# Milestone 3 plan: workgroup staging

Status: draft. Decided 26 November 2026: the static arena adopts the editable
header and patch records. Step 1 (the prototype) is done; see
[Step 1 results](#step-1-results). The remaining decisions are listed at the
end.

The direction behind it: follow RXMesh, editing a mesh is a headline feature,
and in-kernel local queries should be revisited for editing. The Milestone 5
session also relayed that the WESL package is a standalone library inside
consumers' own pipelines. Apart from the arena decision, the choices below
are defaults taken for the prototype and still need confirming; see
[Decisions for the user](#decisions-for-the-user).

## Bottom line

- **One workgroup per patch.** A staged kernel runs one workgroup per patch.
  The workgroup loads what it needs from its patch into workgroup memory and
  derives local relations there. This replaces the edge-record kernel, in which
  each owned edge scans every face its patch holds. That scan is the measured
  bottleneck: edge records are 5.8x slower than the flat layout at 2.1 M faces.
- **Edge records in O(1) per face.** Each active face writes its local face
  slot into `edge_faces[2 * local_edge + reversed]` in workgroup memory. The
  builder rejects non-manifold edges and inconsistent winding, so each slot
  has at most one writer and no atomics are needed. After a barrier, each
  owned edge reads its two slots, puts the lower local slot first, and maps
  slots to global face IDs. That is the order the current kernel produces.
  Boundary flags come from the same pass, and so does FF: an owned face's
  neighbour across an edge is the other slot.
- **The static arena takes the editable header and patch records, but not the
  owner tables.** Both mesh types then share one accessor layer and one set of
  staged kernels. Static patches have capacity equal to count, with every slot
  active. Net memory falls from about 128 to 82 B/face (see
  [Memory](#memory)). This changes the arena that the GPU build writes and the
  benchmark measures.
- **Value-based accessors, with bindings opt-in.** No module forces a bind
  group on consumers. Naga rejects storage pointers as function arguments, so
  the layout module can't take the arena by reference either.
  - `shaders/topology/patch.wesl`: layout constants, record decoding, and
    offset and mask-bit functions on plain values. It declares no bindings.
  - `shaders/topology/patch_arena.wesl`: opt-in readers over a read-only
    arena and mask buffer in group 1, built on `patch`.
  - `shaders/topology/patch_staging.wesl`: owns the workgroup arrays and
    takes values, so it works with either.
  - Editing kernels use `patch` and `patch_staging` with their own
    read-write bindings, so no `@if` variants are needed.
- **Static meshes get the full mask buffer.** Every active bit is set and the
  used counts equal the counts, so one reader works for both mesh types. This
  costs about 0.2 words per face.
- **Outputs unchanged.** Staged edge records, boundary flags and FF must match
  the current kernels word for word. Existing tests then hold as they are.

## Accessor layer

`shaders/topology/patch.wesl` (no bindings):

```wgsl
const VERTEX, EDGE, FACE, PATCH_COUNT_WORD, RECORDS_WORD, RECORD_WORDS;
struct PatchRecord { capacity: vec3<u32>, vertex_ids: u32, ev: u32, fe: u32,
    edge_ids: u32, face_ids: u32, active_masks: vec3<u32>,
    owned_masks: vec3<u32>, interior_mask: u32, used: u32 }
fn record_offset(records: u32, p: u32) -> u32;
fn decode_record(w: array<u32, 16>) -> PatchRecord;
fn used_offset(r: PatchRecord, kind: u32) -> u32;
fn mask_word(mask: u32, slot: u32) -> u32;
fn mask_test(word: u32, slot: u32) -> bool;
fn face_edge_offset(r: PatchRecord, face: u32, i: u32) -> u32;
fn edge_vertices_offset(r: PatchRecord, edge: u32) -> u32;
fn global_vertex_offset / global_edge_offset / global_face_offset(r, slot) -> u32;
```

`shaders/topology/patch_arena.wesl` (arena and masks read-only in group 1):
`patch_count`, `patch_record`, `slot_end`, `slot_active`, `slot_owned`,
`face_edge`, `edge_vertices`, `global_vertex`, `global_edge`, `global_face`.

`shaders/topology/patch_staging.wesl` (workgroup memory only):

```wgsl
const STAGED_EDGES: u32 = 1024u;              // 8 KB of edge-face slots
fn clear_staged_edges(lane: u32, count: u32);
fn stage_face_edge(first: u32, edge: u32, reversed: u32, face: u32);
fn staged_edge_faces(first: u32, edge: u32) -> vec2<u32>;  // lower slot first
```

The caller runs the barriers between clearing, staging and reading. Records
are read as scalar words, never as a storage vector. On Metal, parallel writes
to lanes of one storage vector lost updates.

## Dispatch

Staged kernels dispatch one workgroup per patch. Per-element queries, such as
face centroids, can still dispatch by element. The Milestone 5 plan dispatches
editable queries by global ID through the owner tables, and that stays as it
is. "One kernel set" means shared accessors and shared staged kernels. It does
not mean every kernel dispatches the same way.

## Limits

Workgroup arrays need a size known at shader compile time. The largest patches
are set by vertex valence, not patch size. A patch that owns a face at the
sphere's pole holds the whole pole star. Measured over the CPU build of the
benchmark meshes:

| Patch size | Most held edges in one patch: grid, torus, irregular | Sphere, 524 K faces | Sphere, 2.1 M faces |
|---:|---:|---:|---:|
| 32 | 137–176 | 1,535–1,578 | 2,985–3,031 |
| 128 | 360–502 | 1,745–1,791 | 3,169–3,269 |
| 512 | 1,104–1,477 | 2,406–2,591 | 3,859–4,116 |

Each range covers input order and shuffled IDs. The GPU build's input-order
split holds about twice as many faces as these splits, so its worst patches
are larger again.

A cap would reject the sphere at every patch size, so staging is chunked:

- A patch with more than `STAGED_EDGES` held edges is processed in several
  passes of `STAGED_EDGES` edges each.
- In each pass, every face scatters only the edges that fall inside the
  current chunk.
- A typical patch needs one pass. The 2.1 M-face sphere's pole patches need
  three to five, and pay for each by reading their faces again. The sphere's
  timings stay in line with the other meshes at patch sizes 128 and 512.

Override-sized workgroup arrays were not tried; the fixed size works on
every mesh measured.

## Memory

Words per face at patch size 128, on the input-order grid. Per face, the
patches hold 0.99 vertices, 2.61 edges and 1.63 faces.

| Part | Today | Proposed |
|---|---:|---:|
| Vertex IDs, EV, FE, face IDs | 17.6 | 17.6 |
| Face and edge task lists (4F + 7E) | 14.5 | – |
| Global edge IDs per held edge | – | 2.6 |
| Patch records and masks | – | 0.5 |
| **Total** | **32.1 (128 B)** | **20.7 (83 B)** |

- The arena today stores global edge IDs only in the edge tasks, so the
  proposal has to add an edge-ID region.
- The owner tables would add about 6 words per face (24 B). The static
  staged kernels don't need them; editable meshes keep them.
- The prototype measured the editable arena at capacity 1, owner tables
  included: 107 B/face at patch size 128 against 129 for today's arena, and
  89–99 B/face at patch size 512 against 114–122.
- The flat layout uses 36 B/face.

## Step 1 results

The prototype is `StagedEdgeQuery` in `src/gpu/staged.rs` (test-only), with
the kernel in `runtime_shaders/staged_edges.wesl`. It runs on
`GpuEditableMesh`, whose layout is the one the static arena will adopt.

- **Exact.** It matches `EdgeQuery` word for word on every valid fixture at
  every fixture patch size, and on the generated meshes in input and shuffled
  order, at capacity 1 and 1.5. That includes a sphere patch with more held
  edges than one staging chunk. The benchmark also checks every mesh it
  times, at 524 K and 2.1 M faces. That covers dispatch grids of up to
  123,736 patches, which need the second grid dimension, and pole patches
  that need up to five chunks.
- **Speed.** GPU time on an Apple M1 Max, one run of the whole benchmark with
  the other session idle. Ratios are of medians over 11 runs, at 2.1 M faces,
  across the grid, sphere, torus and irregular meshes. Run with
  `MOPMESH_BENCH=1 cargo test --release --lib staged_edge_benchmark -- --ignored --nocapture`.

| Patch size | Order | Staged vs current | Staged vs flat |
|---:|---|---:|---:|
| 32 | input | 3.5–4.1x slower | 8.8–15x slower |
| 32 | shuffled | 3.3–4.6x slower | 3.1–3.8x slower |
| 128 | input | 2.1–2.4x faster | 2.5–4.2x slower |
| 128 | shuffled | 1.6–2.0x faster | 1.1–1.6x slower |
| 512 | input | 11–12x faster | 1.5–1.9x slower |
| 512 | shuffled | 9.5–10.6x faster | 1.2–1.4x faster |

- At 524 K faces the pattern is the same, over both orders. Staged is
  1.8–2.3x faster than current at patch size 128 and about 10–11x at 512, and
  2.8–4.1x slower at 32. The input-order grid at patch size 128 is left out,
  because it was too noisy.
- Several 2.1 M-face rows were noisy, with medians up to 1.6x the minimum,
  mostly at patch size 128 and on shuffled meshes. Ratios of the minima fall
  in the same ranges.
- **Why patch size 32 is slow: unprofiled.** Limiting the clear to the
  chunk's edges helped only partly. Candidates: one 64-lane workgroup per
  patch is mostly idle when a patch holds about 60 faces, and every lane
  decodes the 16-word record. Packing several small patches into one
  workgroup would address the first; neither is tried yet.
- **Inactive slots are not exercised.** Capacity 1.5 pads past the used
  count, so every slot below it is active. Skipping inactive slots below the
  used count is untested until edits exist.
- **Against flat.** Staged edges are still slower than the flat layout in
  input order. On shuffled meshes at patch size 512 they are faster. The
  likely reason is that a patch keeps its reads local, but that isn't
  profiled.

The gate in step 1 is met at the default patch size of 128 and above.

## Steps

1. **Prototype the staged edge kernel — done.** See above.
2. **Move the static arena to the shared header and patch records.**
   - Change the CPU `arena_contents` and GPU build stage 8.
   - Add the global edge-ID region and the mask buffer to static uploads.
   - Change the face-vertex kernel to dispatch by patch instead of by face
     task.
   - Keep the word-for-word CPU/GPU arena tests.
3. **Land the staged edge, boundary and FF kernels** for both mesh types,
   then remove the old edge and FF kernels.
4. **Stage VV,** if the benchmark still shows VV as the order-sensitive cost.
   It needs per-vertex row offsets, so it comes after FF.
5. **Rerun the baseline** at all sizes and both orders, and update the report.

## Decisions for the user

1. Decided: the static arena adopts the editable header and patch records.
2. Proceed to step 2, given the step 1 results? The staged kernel loses at
   patch size 32.
3. Defaults taken in step 1, to confirm:
   - Value-based accessors, with the group-1 bindings in an opt-in module.
   - Chunked staging for oversized patches, rather than a `LimitsExceeded`
     error, which would reject the sphere at every patch size.
   - A full mask buffer for static meshes. The Milestone 5 session preferred
     a static `slot_end` that reads the capacity, with no mask buffer.
   - Gating step 2 on step 1's measured speed-up.
4. The patch size 32 slowdown: profile and fix it now, or record it as a
   follow-up?

## Open questions

- Does staging FE as well as edge faces help the later kernels enough to be
  worth its workgroup memory? Measure in step 3.
