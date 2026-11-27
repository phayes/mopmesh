# Milestone 3 plan: workgroup staging

Status: complete (27 November 2026). Steps 1 to 3 and 5 are done and step 4
was skipped; see [Step 1 results](#step-1-results),
[Step 2 results](#step-2-results), [Step 3 results](#step-3-results) and
[Step 5 results](#step-5-results). A later
[profile of the staged pass](#staged-pass-profile) led to 128-lane
workgroups, 512 staged edges and no workgroup zero-fill (`2dcee78`). The
step 2 and 3 speed tables are the measurements taken at the time; the
[baseline report](baseline_benchmark.md) has the latest clean rerun.
Decisions and remaining defaults are listed at the end.

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
  used counts equal the counts, so one reader works for both mesh types.
  Measured on the benchmark meshes, with no interior mask: 0.56–0.67 B/face at
  patch size 512, 0.82–1.09 at 128 and 1.6–2.4 at 32. The owned masks alone,
  which a static mesh needs in any case, are 0.53–0.62, 0.70–0.91 and 1.2–1.7.
  Skipping the active and used checks in the staged edge kernel saved about
  3% of GPU time at patch size 32, 5–6% at 128 and 7–15% at 512. Decided 26
  November 2026: static meshes get the full mask buffer. An opt-in
  all-active reader for static meshes could recover that time later.
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

### Face vertices without face tasks

Removing the face task list means `FaceVertexQuery`, which every per-face
query uses, can no longer dispatch one invocation per face task. The
prototype `StagedFaceVertexQuery` (test-only, `src/gpu/staged.rs`, kernel
`runtime_shaders/staged_face_vertices.wesl`) tries two replacements on the
editable layout at capacity 1. Both match `FaceVertexQuery` word for word on
the fixtures and generated meshes, and the benchmark checks every mesh it
times.

- **Per patch:** one workgroup per patch, over its owned face slots.
- **Per face:** one invocation per global face, through the face owner table.

GPU time at 2.1 M faces, medians of 11 runs on the M1 Max, over grid, sphere,
torus and irregular meshes, against today's face-task kernel:

| Patch size | Order | Per patch | Per face (owner table) | Flat |
|---:|---|---:|---:|---:|
| 32 | input | 2.1–2.5x slower | 1.3–1.4x slower | 2.0–2.2x faster |
| 32 | shuffled | 1.8–2.1x slower | 20x slower | 1.3x faster |
| 128 | input | 1.12–1.25x slower | 1.17–1.19x slower | 1.7x faster |
| 128 | shuffled | 1.15–1.19x slower | 20x slower | 1.2x faster |
| 512 | input | 2–6% faster | 1.1x slower | 1.6x faster |
| 512 | shuffled | 8–11% slower | 17x slower | 1.3x faster |

- At patch size 128 the per-patch kernel costs 0.07–0.20 ms more per call at
  2.1 M faces, against 3.6–5 ms saved on edge records.
- Per-face dispatch through the owner table is unusable on shuffled meshes.
  The table is in global face order, so neighbouring invocations read
  unrelated patches.
- Keeping a face task list for this alone would cost 16 B/face, raising the
  static estimate from about 83 to 99 B/face.

## Step 2 results

- **Layout.** `GpuMesh` stores the editable header and patch records, with
  capacity equal to count, and a mask buffer of used counts, all-set active
  masks and owned masks. Owner-table offsets and the record's interior offset
  hold `INVALID_ID`. `GpuMesh::upload` and the GPU build share
  `patch_layout::append_patch` with `GpuEditableMesh` (CPU), and the GPU build
  writes the same words (`runtime_shaders/build_arena.wesl`). The GPU build
  sizes its mask buffer from member totals, so it may end with unused zero
  words.
- **Queries.** `EdgeQuery` uses the staged kernel and `FaceVertexQuery` one
  workgroup per patch, both through `topology::patch_arena`. Every per-face
  query and all adjacency go through them. The step 1 prototype queries and
  their benchmarks are gone with the old kernels; `src/gpu/staged.rs` now
  checks that editable uploads at capacity 1 and 1.5 give the same edge
  records and face vertices as the static upload.
- **Tests.** The validator checks the records, masks and ownership in place of
  the task lists. GPU builds match the CPU arena word for word, and the masks
  up to the CPU length.
- **Memory**, from the baseline benchmark at patch size 128: 83–98 B/face in
  input order and 91–103 B/face shuffled, against 129–145 before. Flat: 36.
- **Speed**, the same benchmark at 2.1 M faces, median GPU time. This run was
  noisier than earlier ones, especially on shuffled meshes, where the flat
  baseline itself ran 18–34% slower than in Milestone 2. Rerun in step 5.

| Query, 2.1 M faces | Order | Milestone 2 patched | Now patched | Flat |
|---|---|---:|---:|---:|
| Edge records (grid) | input | 6.29 ms | 2.97 ms | 1.11 ms |
| Edge records (grid) | shuffled | 6.42 ms | 4.40 ms | 4.33 ms |
| VV + FF + diamonds + edges (grid) | input | 8.22 ms | 5.59 ms | 3.47 ms |
| Face vertices, four meshes | input | – | 0.66–0.91 ms | 0.35–0.40 ms |

## Step 3 results

- **Kernels.** `runtime_shaders/edges.wesl` has a second entry point,
  `adjacency`, which writes edge records, FF and diamonds in one staged pass.
  The owning patch of an edge holds both its faces, so it writes both faces'
  FF entries and both opposite vertices. `runtime_shaders/boundary.wesl` is
  now staged too: it writes the flags directly, without edge records. Both
  share `shaders/topology/patch_edge_staging.wesl`, a new opt-in public
  module that runs the per-chunk staging loop over `topology::patch_arena`.
- **Queries.** `AdjacencyQuery` no longer runs `EdgeQuery` or
  `FaceVertexQuery`; VV still runs as a global transpose of the staged edge
  records. `AdjacencyQuery` and `BoundaryQuery` gained `encode_patches`, and
  `PatchBuffers` gained `vertex_count`. The global FF kernel moved to
  `runtime_shaders/baseline_face_neighbors.wesl` for the flat benchmark path.
- **Tests.** `src/gpu/staged.rs` checks, word for word, that the staged
  edge records, FF and diamonds equal what the global FF kernel derives from
  the same edge records and face vertices. It also checks that the boundary
  flags equal a CPU derivation from the edge records, and that editable
  uploads at capacity 1 and 1.5 give the same results, VV rows included.
  That covers every fixture and patch size, plus generated meshes in both
  orders, including the 131K-face sphere that needs chunking. Breaking the
  opposite-vertex index or one vertex flag in a kernel makes it fail.
- **Speed**, the baseline benchmark at 2.1 M faces and patch size 128,
  median GPU time, before and after this step on the same machine. A build
  overlapped part of the "before" run, around the shuffled sphere.

| Query, 2.1 M faces | Order | Before | After | Flat |
|---|---|---:|---:|---:|
| VV + FF + diamonds + edges | input, four meshes | 4.70–5.48 ms | 4.47–5.28 ms | 2.82–2.93 ms |
| VV + FF + diamonds + edges | shuffled, four meshes | 12.36–12.94 ms | 9.69–10.07 ms | 13.14–13.25 ms |
| Boundary flags | input, four meshes | 2.84–5.66 ms | 3.17–3.82 ms | – |
| Boundary flags | shuffled, four meshes | 4.10–5.10 ms | 3.58–5.69 ms | – |

  Adjacency drops from 13 to 11 dispatches and now beats the flat layout by
  about 25% on shuffled meshes. Boundary flags drop from 2 dispatches to 1
  and its median wall time falls by 1.3–4.5 ms on every mesh, but its GPU
  time is noisy and on two input-order meshes rose by 0.3–0.4 ms. On
  input-order meshes it runs 0.5–0.6 ms above the staged edge-record pass
  (3.17 against 2.68 ms on the grid), though it writes less; that is not
  yet explained. The staged edge pass itself is still 2.5–3x the flat edge
  pass on input-ordered meshes, and it is the remaining gap to flat there.
  On shuffled meshes, VV timed on its own (below) is not the largest cost.
- **Staging FE as well (open question).** Not tried. The staged passes read
  FE from the patch's own contiguous region, and the gap above is in the
  staged pass as a whole, so FE staging is one candidate alongside the
  patch size 32 follow-up. It needs profiling before it is worth its
  workgroup memory.

## Step 5 results

The full baseline benchmark ran three times on commit `746a47f`, at
32 K–2.1 M faces, patch sizes 32, 128 and 512, in input and shuffled
order; the [baseline report](baseline_benchmark.md) has the tables and
method. At 2.1 M faces and patch size 128:

| Query | Order | Milestone 2 | Milestone 3 | Flat |
|---|---|---:|---:|---:|
| Edge records | input | 6.29–7.22 ms | 2.67–3.26 ms | 1.01–1.05 ms |
| Edge records | shuffled | 6.42–7.49 ms | 3.54–3.93 ms | 3.65–3.66 ms |
| VV + FF + diamonds + edges | input | 8.22–9.29 ms | 4.47–5.24 ms | 2.82–2.92 ms |
| VV + FF + diamonds + edges | shuffled | 15.09–16.28 ms | 9.67–10.07 ms | 13.12–13.24 ms |
| Boundary flags | both | 6.54–7.39 ms | 2.39–4.41 ms | – |
| Memory | both | 129–145 B/face | 83–103 B/face | 36 B/face |

- Patch size 512 is now the fastest setting measured, and the smallest:
  edge records 1.6–1.9 ms in input order and 2.0–2.2 ms shuffled, batched
  adjacency 3.4–3.8 and 8.9–9.2 ms, 65–75 B/face. The default is still 128.
- Patch size 32 is about 3x slower than in milestone 2 (follow-up, deferred).
- Per-face queries on the arena are 9–27% slower than in milestone 2.
- Smoothing on shuffled meshes takes either about 9 or 12–15 ms per ten
  steps, varying between runs. A later rerun gave the VV row its own
  adjacency, and the variation remained, so the shared buffer was not the
  cause.
- The [baseline report](baseline_benchmark.md) was rerun at `ab98f19`, after
  the default patch size became 512. Batched adjacency had regressed there
  by 1.1–1.2 ms in input order and 0.4–0.5 ms shuffled, after the
  adjacency kernel moved onto the shared `local_query` helpers. `e6f8663`
  fixed that, and a third rerun there put adjacency back within noise of
  `746a47f`.

## Staged pass profile

Profiled on 27 November 2026 at `42aad9e`, on an Apple M1 Max at 2.1 M
faces. Scratch copies of the edge-record kernel had parts removed, and
staging sizes and lane counts varied. GPU time is per compute pass, from
21 timed runs after 2 warm-ups, and every variant's records were checked
against the flat kernel. The tables give minimum times; medians agreed
except in noisy runs.

**Where the time goes.** At patch size 512, the shipped kernel takes
1.60 ms on the grid in input order. Staging alone takes 1.13 ms. The
writes are cheap: the record writes take 0.31 ms including the patch
record read, and the active-slot check under 0.1 ms. The 9–27% per-face
slowdown therefore doesn't come from these checks, at least in this kernel.

**The cost is workgroup memory, not staging work.** The grid at patch size
128 never needs a second chunk, since it has at most 369 edge slots. Even
so, its time tracks the declared size of `edge_faces`. There are two
effects:

1. wgpu zero-fills every workgroup's workgroup memory before it runs,
   because `zero_initialize_workgroup_memory` is on by default. That costs
   0.2 ms at patch size 512 and 0.74 ms at 128, where there are four times
   as many workgroups. The staging functions already clear what they read.
2. The 8 KB array limits how many workgroups run at once per GPU core. A
   4 KB array is faster even when patches at 512 then need two or three
   chunks, each rescanning the patch's faces. With 128 lanes per workgroup
   instead of 64, the same memory serves twice the invocations.

Edge records (ms; flat 1.02–1.06 in input order, 3.64–3.71 shuffled):

| Lanes, staged edges, zero-fill | Grid 128 | Grid 512 | Sphere 128 | Sphere 512 | Shuffled grid 128 | Shuffled grid 512 |
|---|---:|---:|---:|---:|---:|---:|
| 64, 1024, on (shipped) | 2.63 | 1.60 | 3.21 | 1.89 | 3.65 | 2.00 |
| 64, 1024, off | 1.89 | 1.40 | 2.29 | 1.67 | 2.46 | 1.67 |
| 64, 512, on | 1.55 | 1.24 | 1.98 | 1.51 | – | – |
| 64, 512, off | 1.25 | 1.17 | 1.61 | 1.43 | – | – |
| 128, 512, on | 1.31 | 0.97 | 1.71 | 1.20 | 1.94 | 1.44 |
| 128, 512, off | 1.06 | 0.92 | 1.42 | 1.15 | 1.61 | 1.36 |

Batched adjacency through `AdjacencyQuery`, zero-fill on (ms):

| Configuration | Grid 128 | Grid 512 | Sphere 128 | Sphere 512 | Shuffled grid 128 | Shuffled grid 512 |
|---|---:|---:|---:|---:|---:|---:|
| Flat | 2.78 | 2.78 | 2.80 | 2.84 | 13.10 | 13.10 |
| 64 lanes, 1024 (shipped) | 4.55 | 3.48 | 5.35 | 3.90 | 9.84 | 8.66 |
| 128 lanes, 512 | 2.82 | 2.45 | 3.35 | 2.79 | 9.58 | 9.13 |

With 128 lanes and 512 staged edges, edge records on the grid at patch size
512 beat the flat kernel. Batched adjacency beats it at patch size 512 on
the grid and roughly matches it on the sphere.

Caveats:

- The 128-lane rows come from one or two runs each, and the shipped rows
  from three. Sweep repeats agreed within 0.05 ms, apart from noisy bursts.
- The adjacency output wasn't checked in the 128-lane configuration.
- The shuffled grid at 512 came out 0.5 ms slower with 128 lanes, from a
  single run. Recheck it before changing anything.
- 256 lanes and 256 staged edges gave no gain over 128 and 512, each in
  one noisy run.
- Packing both local face slots of an edge into one word with workgroup
  atomics would halve the memory again. Not tried.

Other workgroup memory: `local_query` declares about 7 KB (`vv_entries`
and three 256-entry tables). A kernel using both it and the staging array
would hold about 15 KB per workgroup. If those kernels show the same
pattern, the per-patch flip and editing kernels likely pay the same cost.
Not measured.

All three fixes were applied together in `2dcee78`, agreed with the
Milestone 5 session:

1. The crate's compute pipelines no longer zero-fill workgroup memory
   (`pipeline::NO_ZERO_FILL`). Every kernel was checked to write its
   workgroup variables before reading them. Pipelines that users build keep
   wgpu's default, which is correct but slower.
2. `STAGED_EDGES` is 512, so `edge_faces` takes 4 KB.
3. Per-patch kernels run with `PATCH_LANES` (128) invocations, a public
   constant in `topology::patch`. It replaces the 64-lane contract of
   `patch_staging`, `patch_edge_staging`, `patch_queries`, `local_query`
   and `flip_pass`, and is used by boundary flags, edge records,
   adjacency, edge flips and the consumer examples.

Results, from three clean runs of the
[baseline benchmark](baseline_benchmark.md) at 2.1 M faces:

- Edge records are 1.4–3.2x faster than before, boundary flags 1.2–2.6x
  and batched adjacency 1.4–2.5x in input order. Shuffled adjacency is 5%
  faster at patch size 128 and 1–2% slower at 512.
- At patch size 512, edge records run at 0.92–1.16x of the flat layout in
  input order and 0.37–0.43x shuffled, and batched adjacency at 0.85–0.96x
  and 0.69–0.71x.
- Patch size 32 recovered most of its loss: edge records take 2.7–4.4 ms,
  from 8.8–11.0 ms.
- One flip batch requesting every edge is 13–40% faster, with identical
  outcomes, from the flip benchmark before and after (three runs each).
  The 2.1 M-face grid went from 37.3 to 28.4 ms.
- A side-by-side run with zero-fill turned back on put its share at about
  5%: edge records on the grid at patch size 512 took 1.00 ms with it and
  0.95 ms without. The lane count and smaller array give the rest.
- Smoothing, which uses no workgroup memory, measured slower in some runs.
  Alternating builds of the old code, this code and this code with
  zero-fill on gave overlapping ranges, so the change is not the cause.

## Steps

1. **Prototype the staged edge kernel — done.** See above.
2. **Move the static arena to the shared header and patch records — done.**
   See [Step 2 results](#step-2-results).
3. **Stage the boundary and FF kernels — done.** See
   [Step 3 results](#step-3-results).
4. **Stage VV — skipped** (decided 27 November 2026). The step was to run
   only if VV was still the order-sensitive cost. Measured on 26 November 2026 with the "VV (edge records given)" row, 2.1 M
   faces, patch size 128, median GPU time: 0.86–1.08 ms in input order and
   2.41–2.71 ms shuffled, out of 4.47–5.15 ms and 9.67–10.08 ms for the
   whole adjacency query. VV is a quarter of the shuffled cost; the staged
   edge, FF and diamond pass is the other 7.2–7.5 ms. Staging VV could save
   at most part of 2.5 ms shuffled, and staged kernels have so far run
   2.5–3x slower than flat ones in input order, where VV costs under 1 ms.
5. **Rerun the baseline and update the reports — done.** See
   [Step 5 results](#step-5-results). The checklist below was followed,
   with three runs instead of two because the first two disagreed in bursts.
   1. Before starting, ask the Milestone 5 session for a quiet GPU and CPU
      (no builds or tests) for the whole run, and tell it when done.
   2. Run on a clean committed tree:
      `MOPMESH_BENCH=1 MOPMESH_BENCH_FACES=32768,524288,2097152
      MOPMESH_BENCH_PATCHES=32,128,512 cargo test --release --lib
      gpu_baseline_benchmark -- --ignored --nocapture`, saving the output.
      This covers every generated mesh in input and shuffled order. Run it
      a second time and compare medians; rerun any row whose medians differ
      by more than about 10%, as earlier runs were noisy.
   3. Update [docs/baseline_benchmark.md](baseline_benchmark.md): every
      table, the memory figures (bytes per face and per held element), the
      dispatch, buffer and bind-group counts, and the new "VV (edge records
      given)" row. Rewrite the summary so it states where the arena now wins
      (shuffled adjacency) and loses (input order, patch size 32), and fix
      the text that still calls VV/FF global kernels (around line 38).
   4. Update the plan's Step 2 and Step 3 speed tables from the clean run,
      and mark this plan's status as complete.
   5. Update `ROADMAP.md`: the Milestone 3 entry under "Next execution
      milestones", the benchmark summary in item 2 ("slower on every query",
      the 5.8x, 2.9x and 3.6–4x figures), and, in agreement with the Milestone 5 session, the "Current
      position" paragraph and the test count.
   6. Run fmt, clippy, `cargo doc --all-features` and
      `cargo test --doc --all-features`, then ask the user to commit.

## Decisions for the user

Decided on 26 November 2026:

1. The static arena adopts the editable header and patch records.
2. Proceed to step 2.
3. Static meshes get the full mask buffer: used counts, all-set active masks
   and owned masks, with no interior mask or owner tables. Codex, the advisor
   and the Milestone 5 session all recommended it; owned masks inside the
   arena were the alternative.
4. The patch size 32 slowdown is a follow-up, not part of this milestone.

Defaults taken in step 1, not yet confirmed:

- Value-based accessors, with the group-1 bindings in an opt-in module.
- Chunked staging for oversized patches, rather than a `LimitsExceeded`
  error, which would reject the sphere at every patch size.

Default taken in step 3, not yet confirmed:

- A public opt-in module, `topology::patch_edge_staging`, for the staging
  loop over `topology::patch_arena`, so the runtime kernels stay thin
  wrappers over library functions.

## Open questions

- Does staging FE as well as edge faces help the later kernels enough to be
  worth its workgroup memory? Not tried in step 3; see
  [Step 3 results](#step-3-results). The
  [profile](#staged-pass-profile) shows that workgroup memory decides
  occupancy here, so staging FE would probably slow the pass down unless
  its memory came from somewhere else.
