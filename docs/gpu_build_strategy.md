# GPU mesh construction: strategy

Build the GPU connectivity arena on the GPU from raw positions and triangles,
instead of building a CPU `Mesh` and uploading it. The goals are a fast GPU
build and confidence that it is correct. Correct means "satisfies the
contract", not "equals the CPU builder": where the contract allows several
answers (the patch split, order within a patch), the GPU may choose a
different, faster one. See [testing](#testing) for how each output is checked.

## Why

On the 2.1 M-face grid (M1 Max, see [approximate_benchmark.md](approximate_benchmark.md)):

| Path | Time |
|---|---:|
| `MeshBuilder::build` (CPU) | 387 ms |
| `GpuMesh::upload` (arena + positions) | 90 ms |
| Upload of raw positions + triangles only (~37 MB) | estimate, not measured |

Every query that runs after the mesh is ready takes 4–16 ms, so construction
dominates any workload that builds once and runs a few queries. Local dynamics
(ROADMAP milestone 5) will also need patch rebuilds on the GPU, so this is work
we'd need eventually anyway.

## Ground rules

1. **Test against the contract, not the CPU.** Exact agreement with the CPU is
   required only where the API promises one answer: which inputs are rejected
   and with which error, and edge IDs in ascending (smaller, larger) vertex
   order. Everything else is checked by properties (see [testing](#testing)).
   Never slow the GPU build down only to reproduce a CPU choice.
2. **Same result on every run on the same device.** The GPU result must not
   depend on thread scheduling, so bugs reproduce and benchmarks compare.
   Atomics may count, or pick a minimum; final positions come from scans, or
   from ranks with an explicit sort key. The result may differ between the
   CPU and the GPU, and between devices.
3. **The arena layout doesn't change.** GPU construction writes the same arena
   that `GpuMesh::upload` writes today: header, then per-patch vertex IDs, EV,
   FE and face IDs, then face tasks, then edge tasks. Every existing query then
   works unchanged.
4. **Portable WebGPU only.** No 64-bit atomics, no subgroups, and no
   push-constant or other optional features. Workgroup size is 64, as in the
   existing queries. Check buffer and dispatch limits up front with the
   existing `pipeline::check_binding` / `workgroups` helpers, so a mesh that is
   too large returns `GpuError::LimitsExceeded`.
5. **One stage at a time**, each with its own acceptance test, landing
   separately. Kernels and their bindings go in `runtime_shaders/` (compiled
   in build.rs, like the queries). Value-level logic that editing or
   downstream code can reuse goes in the public `shaders/build/` package;
   portable WGSL cannot pass storage pointers, so these helpers take loaded
   values.

## Testing

Each output is checked by the strongest test its contract allows:

| Output | Contract | Test |
|---|---|---|
| Accept/reject and the error reported | One answer | Exact match with `MeshBuilder::build` |
| Global adjacency (FV, edges, EF, VF, …) | One answer, up to documented ordering freedoms | Query suite against Alum, on GPU-built meshes |
| Edge IDs | Ascending (smaller, larger) vertex pairs | Exact |
| Patch split | Any split: every face in one patch, at most `patch_size` faces per patch | Properties; quality metrics against the CPU splits |
| Order within a patch, task order | Unspecified, but the same every run | Arena validator |
| Owners | A patch that holds the element's full neighbourhood, chosen by the documented rule | Arena validator |
| Same result every run | Rule 2 | Build twice, compare |
| Speed, ribbon overhead | Not pass/fail | Benchmarks |

The **arena validator** (`gpu::validate_arena`, CPU, test-only) decodes any
arena and checks it against the input triangles, independent of numbering
choices. It accepts CPU-built and GPU-built arenas alike, so neither builder is
the definition of correct.

**Word-for-word comparison** remains a debugging aid: the CPU builder accepts
an explicit face→patch split, so it can rebuild the GPU's split and diff the
arenas, pointing at the first difference. It applies only while both builders
make the same choices downstream of the split; a faster GPU route that orders
things differently drops it for that stage.

## Pipeline

The original plan, CPU stage → GPU equivalent. Some GPU designs changed
while building them; [stages and acceptance](#stages-and-acceptance) describes
what was built.

| # | CPU today (`src/mesh.rs`, `src/mesh/build.rs`) | GPU kernels | Difficulty |
|---|---|---|---|
| 1 | Non-finite positions, missing vertices, degenerate faces (decision 4) | One thread per vertex or face; an atomic minimum records the first failing ID per error kind | Easy |
| 2 | Vertex→corner rows (`Rows::group`) | Count per vertex (atomic add) → scan → stable fill | Medium: stable fill |
| 3 | Duplicate faces (`find_duplicate_face`) | Per-row sort of the sorted vertex triples, in buckets keyed by smallest vertex; flag equal neighbours; atomic minimum of the later face | Medium |
| 4 | Edge IDs, edge faces, face edges, winding and non-manifold edges (`connect_edges`) | Bucket the corners by smaller endpoint (reuse stage 2's machinery), sort each row by (larger endpoint, corner), flag the first corner of each edge, scan the flags to get edge IDs, write the faces for each edge and the edges for each face; atomic minimum of the failing corner | Medium |
| 5 | Vertex fans (`is_single_fan`) | One thread per vertex, same walk | Easy |
| 6 | Faces grouped into patches (breadth-first growth) | **Needs a new algorithm** (see below) | Hard |
| 7 | Patch contents, local IDs, owners | One workgroup per patch: gather, deduplicate, sort in workgroup memory; count pass → scan of the per-patch sizes → write pass | Medium |
| 8 | Arena assembly (`GpuMesh::upload`) | Header and task lists from scans over owned-face and owned-edge counts per patch | Easy, once 7 exists |

Error reporting: each kernel writes the first failing ID into a small error
buffer, one slot per error kind. After submission a single readback turns that
buffer into a `BuildError`, using the precedence the CPU uses today: per-face
checks first, then the earlier of a duplicate-face error and an edge error,
then vertex errors.

### Shared GPU building blocks

- **Scan:** already exists (`src/gpu/scan.rs`, hierarchical 64-lane).
- **Bucket fill:** count, scan, scatter with atomics. Order within a row is
  then arbitrary; sort the row only where a later step needs it (patch rows
  are sorted for binary search; corner rows are not).
- **Radix sort:** stable 4-bit LSD passes over `[key, value]` pairs
  (`runtime_shaders/radix_sort.wesl`), used for the Morton split.
- **Small sorts within a row:** the same insertion or bitonic sort, reused by
  stages 2–4 and 7.
- **Flag + scan compaction:** for edge IDs, patch offsets and task lists.
- **First-error reduction:** `atomicMin` on `u32`.

## The patching decision

The CPU partition grows one patch at a time across neighbouring faces, in face
order. That is inherently serial and can't be reproduced in parallel, so the
GPU needs its own split. Under rule 1 the GPU split need not match any CPU
split; the CPU builder offers the same families (input order, Morton) for
comparison and accepts an explicit split for debugging.

| Option | Parallel | Patch quality | Same result every run | Notes |
|---|---|---|---|---|
| A. Morton order of face centroids, cut into runs of `patch_size` faces | Fully | OK on well-shaped meshes. Patches can be disconnected, with more border faces | Yes (sort key = Morton code, face ID) | Needs a GPU sort of 2 M keys, which is our own radix sort |
| B. Lloyd iterations (RXMesh: `prior_art/RXMesh/include/rxmesh/patcher/`) | Per iteration | Compact, close to equal size | Yes, if growth breaks ties by ID | Several iterations of parallel breadth-first growth plus seed updates. Patch sizes vary, so a patch can exceed `patch_size`; needs a split pass |
| C. Chunk faces in input order, `patch_size` at a time | Fully | Depends entirely on input order | Yes | Trivial; good only as a stepping stone |

**Recommendation:** use C as the stepping stone, so stages 7 and 8 can be built
and tested without a sort. Then A as the working default: it's the simplest
parallel scheme with a predictable `patch_size` bound. B must also be built and
compared before a final default is chosen. Compare all options on ribbon
overhead and query times against today's breadth-first patches.

## API shape

Proposed; to be confirmed before step 7 below:

```rust
// Records a GPU build. Inputs are caller-owned buffers of [f32; 4] positions and
// [u32; 4] triangles (fourth lane ignored, so WGSL vec4 layout works).
let pending = GpuMeshBuilder::new(&device)
    .patch_size(128)
    .encode(&device, &mut encoder, &positions, &triangles, vertex_count, face_count)?;
queue.submit([encoder.finish()]);
// Reads back the error buffer; Ok gives a GpuMesh identical to GpuMesh::upload.
let gpu_mesh: GpuMesh = pending.finish(&device)?;
```

- `GpuMesh` already contains no CPU connectivity, so queries don't change.
- A GPU-built mesh has no CPU `Mesh`, so `MeshId` must be created by the GPU
  build itself; vertex attributes then need a constructor that takes a
  `GpuMesh` instead of a `Mesh`.
- Optional later: `GpuMesh::read_back(&device) -> Mesh`, for callers who need
  CPU patches or handles.

## Stages and acceptance

0. **Harness.** Factor `GpuMesh::upload`'s arena assembly into a CPU function
   that returns the arena words, so tests can compare against them. Add a
   GPU-test helper that runs a build both ways on the same mesh set as
   `tests/gpu_topology.rs` (grids, holes, fans, tetrahedron, empty, invalid
   inputs) and compares the results.
   **Done:** `arena_contents` in `src/gpu.rs`; helpers in `src/gpu/test_support.rs`
   (device, readback, CPU reference, region-naming arena compare); shared
   meshes in `tests/support/fixtures.rs`. Today the harness only checks
   uploaded arenas against the CPU words.
1. **Validation + vertex→corner rows (stages 1–2).** Accept: per-face errors
   and the corner rows match the CPU output exactly.
   **Done:** `GpuBuilder::encode_elements` in `src/gpu/build.rs` with
   `runtime_shaders/build_elements.wesl`, checked against
   `mesh::build::check_elements` and `Rows::group` on every fixture. Corner
   rows were later left unsorted (step 6): nothing depends on their order.
2. **Edges (stages 3–4).** Accept: edge endpoints, edge faces and face edges
   match `connect_edges` exactly, and every duplicate, winding and
   non-manifold-edge case reports the CPU error.
   **Done:** `GpuBuilder::encode_connectivity` with `runtime_shaders/build_connectivity.wesl`,
   checked against `mesh::build::check_edges`. It needs no new sort: each face
   and corner searches the corner rows of its vertices, so cost grows with
   valence. Edge IDs come from per-vertex counts of the edges each vertex leads
   as the smaller endpoint, a scan, and a rank within the vertex. The edge
   count exists only on the GPU (the last word of the edge offsets), and the
   edge buffer is sized for 3F edges; later stages must read the count on the
   GPU or size their work from that bound.
3. **Fans (stage 5).** Accept: `NonManifoldVertex` and `IsolatedVertex` cases
   match.
   **Done:** the `check_fans` kernel runs at the end of `encode_connectivity`
   (it needs the same bindings), checked against `mesh::build::check_fans`.
   Every GPU check is now in place: the error buffer reproduces every
   `BuildError` a mesh can cause, in the CPU's order.
4. **Patch contents with a given partition (stage 7).** Upload the CPU's
   face→patch array as an input. Accept: patch vertex, edge and face lists and
   the owner handles match the CPU output.
   **Done**, in two phases (decision 5). Phase 1 (`encode_member_counts`,
   `runtime_shaders/build_patch_members.wesl`) builds S(v), the patches
   owning a face at each vertex, and counts members: a face is in every patch
   of S(a) ∪ S(b) ∪ S(c), an edge in those of its endpoints and opposite
   vertices, a vertex in those of itself and its neighbours. The error words,
   edge count and member totals go into one 15-word summary. Phase 2
   (`encode_patches`, `build_patch_rows.wesl`, `build_patch_local.wesl`)
   writes the members, groups them into per-patch rows (count, scan, fill,
   sort), and derives local EV/FE and owner handles by binary search in the
   rows. The totals give the arena size exactly:
   `4 + 4F + 7E + Mv + 2Me + 7Mf`. Each patch row is sorted by one 64-wide
   workgroup (a rank sort through workgroup-memory tiles), so cost per row
   still grows with the square of its length, shared by 64 invocations.
5. **Arena assembly (stage 8).** Accept: GPU arena == CPU arena, word for word,
   and every existing query result matches.
   **Done:** `GpuBuilder::encode_arena` with `runtime_shaders/build_arena.wesl`.
   A one-invocation kernel writes the four header words (mapping the arena
   at creation cost about 35 ms at 2.1 M faces); the GPU scans per-patch region sizes into region starts and owned-element
   flags into task indices, then each patch member writes its own words. The
   arena equals `arena_contents` word for word on every fixture and patch
   size. Queries read only the arena and the three counts, so identical
   arenas give identical query results; running the query suite on GPU-built
   meshes waits for the public API (step 7), which creates the `GpuMesh`.
6. **Partitioning (stage 6).** Make the CPU split pluggable (breadth-first,
   input order, Morton, explicit). Implement C, then A, on the GPU with fast
   f32 arithmetic. Accept: the split passes its property checks; the full GPU
   build passes the arena validator; the CPU builder given the GPU's split
   produces the same arena word for word; building twice gives the same
   arena. Benchmark ribbon overhead and query times against breadth-first.
   Then implement B (Lloyd iterations) on CPU and GPU and compare it with A
   before choosing the final default.
   **C and A done:** `GpuBuilder::encode_split` with
   `runtime_shaders/build_split.wesl` (input order; Morton keys of f32
   centroids over their bounding box) and `runtime_shaders/radix_sort.wesl`
   (stable 4-bit LSD radix sort, 8 passes). The CPU builder has the same
   methods (crate-private) and accepts an explicit split. The full GPU build,
   split included, passes the contract test on every fixture, patch size and
   method: split properties, the arena validator, the word-for-word diff
   against the CPU given the GPU's split, and a repeat build. The radix sort
   has its own exact test. Ribbon overhead (faces held per face) on the
   8,192-face grid: at patch size 128, Morton 1.47 against breadth-first
   1.60; at patch size 3, Morton 6.95 against 6.20. The corner-row sort from
   step 1 is gone: nothing depends on that order.
   **Benchmarked** ([results](approximate_benchmark.md#gpu-build)): the GPU
   build is 3.1x faster than CPU build + upload at 2.1 M faces, 3x slower at
   32 K. Morton patches are within 2–3% of breadth-first on ribbon overhead,
   with query times within noise except centroids (+19%), so option B stays
   deferred. The benchmark also found that one-dimensional dispatches capped
   every stage at about 4.2 M items; all build kernels and the scan now use
   two-dimensional grids (`pipeline::workgroup_grid`,
   `shaders/parallel/grid.wesl`), tested by building on a device limited to
   64 workgroups per dimension.
7. **Public API + docs + benchmark.** Add the GPU build row to
   `examples/bench_alum.rs` and the approximate benchmark doc.

Follow-ups and to-dos from this work are tracked in the
[ROADMAP](../ROADMAP.md#gpu-construction-follow-ups).

## Risks

- **Long rows.** High-valence vertices make the per-row sorts slow or need a
  fallback; test with the 129-face fan and something larger.
- **Changing the partition** moves every patch-dependent result. Owners and
  ribbons change, so tests that compare patch-level details need to compare
  against the CPU reference, not hard-coded values.
- **Memory.** Temporary buffers (corner records, sort scratch) are several
  times the arena size. On devices with a 128 MB binding limit, the largest
  mesh a GPU build can handle may be smaller than what the CPU path can upload.
- **Readback latency.** `finish` blocks on the GPU to learn whether the input
  was valid. Callers who trust their input may want a no-validation option
  later.

## Decisions (agreed 2026-11-26)

1. **Partition (tentative):** option C (input order) as the stepping stone,
   then option A (Morton order) for now. The GPU split need not match a CPU
   split (decision 6). Update: option B is deferred unless the benchmark
   shows Morton patches hurting (tracked in the ROADMAP). Originally: option B (RXMesh-style Lloyd
   iterations) must also be implemented and compared on ribbon overhead and
   query times before a default is chosen. Keep the partition step pluggable
   so the options can be swapped and benchmarked against each other.
2. **No readback in the first version.** A GPU build produces a `GpuMesh` only;
   `GpuMesh::read_back` can come later.
3. **Input:** caller-owned GPU buffers in vec4 layout (`[f32; 4]` positions,
   `[u32; 4]` triangles).

4. **Degenerate faces (agreed 2026-11-26):** a face is degenerate only when two
   of its corners have equal coordinates (`0.0` equals `-0.0`; a repeated
   vertex ID counts). Collinear faces are accepted. The earlier f64
   cross-product test could not be reproduced exactly in portable f32, so the
   CPU builder switched to this rule too. The GPU compares position bit
   patterns, with no float arithmetic.

5. **Two phases, one readback (agreed 2026-11-26):** buffer sizes depend on
   the mesh data (edge count, and each patch's faces, edges and vertices
   including ribbons), so one recording cannot size them. Phase 1 records
   validation, connectivity and per-patch member counts, then copies the
   error words and totals into one small buffer; the build submits it and
   reads that back. Phase 2 allocates exact buffers and records the rest. The
   build therefore takes the queue and submits by itself; this replaces the
   single-encoder API sketched above.

6. **Correct, not identical (agreed 2026-11-26):** the GPU build is tested
   against the contract, as in [testing](#testing). Exact CPU agreement is
   kept only for errors and edge IDs. Edge IDs stay in ascending
   (smaller, larger) vertex order. Results stay the same on every run on one
   device. Order within a patch and task order become unspecified in the
   documented API.

Steps 0–6 are done (option B deferred). Next: speed up the patch-contents
and arena stages, then step 7 (public API).
