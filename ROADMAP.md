# Implementation roadmap

Last reviewed: 26 November 2026.

This is the maintained execution checklist. The [technical design](MOPMesh_Technical_Design.md)
explains the architecture and intended dynamic transaction model; the
[README](README.md) documents the implemented API, buffer layouts and commands.
Checked items mean implemented and verified within the stated coverage, not
production readiness. Update this checklist with each milestone and keep API
contracts in the README alongside the code.

## Current position and evidence

The prototype has the planned static adjacency families, canonical vertex
attributes, hierarchical scans, and two runnable downstream WESL examples.
Geometry algorithms have begun; topology editing has not. The public API
builds and partitions on the CPU; a GPU build producing the same arena format
works internally and awaits its public API. Uploaded topology is immutable.

The static API review is complete. Its contract decisions are in the README's
shared contracts; its deferred findings are listed under
[static API review follow-ups](#static-api-review-follow-ups).

The last full verification run passed 28 tests: 18 hardware-independent tests
and ten opt-in GPU tests (one is the GPU build benchmark, which returns at
once unless `MOPMESH_BENCH` is set). Those test functions cover many fixtures and patch
sizes; the count is not a measure of exhaustive coverage. Clippy and formatting
checks passed, and both examples ran. GPU execution has only been verified on
Apple M1 Max / Metal; Naga validation uses no optional capabilities.

Evidence lives in [host topology tests](tests/static_topology.rs),
[GPU integration tests](tests/gpu_topology.rs), [scan tests](src/gpu/scan.rs), and
[downstream linking tests](tests/wesl_package.rs). Fixtures include empty meshes,
boundaries, a hole, disconnected components, a tetrahedron, a 129-face vertex
fan, and grids through 8,192 faces. Scans cover up to 262,145 values. Mesh
results are compared with Alum; integer scan results use arithmetic checks.

## Acceptance gates

- **Static viability — open.** Static query correctness is covered by the
  current fixtures. Measured advantage over an unpartitioned GPU baseline is
  still missing. No performance or memory-efficiency claim has been established.
- **Dynamic viability — not started.** Requires conflict-free local and
  cross-patch edits, preserved committed topology on failure, and bounded retries.
- **Production viability — not started.** Requires cross-vendor execution and
  performance evidence, reliable capacity handling, stable contracts and packaging.

## Phase checklist

The numeric-representation constraint below applies to every implementation phase.

### Initial-release numeric representation

The initial release uses native 32-bit GPU values: `u32`/`i32` for integer
data and `f32` for geometry. Connectivity indices and arena word offsets use
`u32`. Vectors and records may contain multiple independent 32-bit fields.
There is no 16-bit storage/arithmetic variant, subword numeric packing, native
64-bit GPU representation, emulated 64-bit integer arithmetic, or paired-float
(single-single/double-single) precision path.

Host-side `u64`/`usize` size calculations, overflow checks and wgpu API arguments
remain appropriate. Capacity beyond the
32-bit indexing or device limits must produce an explicit error, not a fallback
to wider or packed representations.

Packing and alternative numeric widths are outside the committed roadmap.
Reconsider them only for a measured need through a separate design decision.

### 0 — References: mostly complete

- [x] Record RXMesh and OpenMesh revisions and retain upstream license notices.
- [x] Use Alum as the independent CPU oracle and development dependency.
- [x] Download OpenMesh and build OpenMeshCore as a secondary reference.
- [x] Add synthetic correctness fixtures.
- [ ] Curate reproducible benchmark meshes, sizes, provenance and licenses.
- [ ] Configure CUDA RXMesh execution for matched reference benchmarks when hardware is available.

OpenMesh is not connected through Rust FFI; current comparison tests use Alum.
See [reference setup](prior_art/REFERENCES.md).

### 1 — Infrastructure: prototype foundation complete

- [x] Integrate Rust/wgpu with caller-owned devices, queues and command encoders.
- [x] Compile WESL and export a Cargo shader package for independent downstream linking.
- [x] Separate reusable `shaders/` modules from concrete `runtime_shaders/` bindings.
- [x] Provide an importable 64-lane scan collective and hierarchical GPU scan.
- [x] Validate shaders with Naga; run GPU tests explicitly, failing if no adapter is available.
- [x] Check arena/result sizes, scalar addressing and relevant dispatch limits.

Errors are consolidated into `GpuError` and pipeline setup shares one private
helper module. Device-profile selection and scratch/pipeline reuse across
queries still need measurement; these are follow-up hardening tasks, not
evidence of production readiness.

### 2 — Static topology: connectivity complete; broader acceptance open

- [x] Validate oriented manifold triangle inputs, including supported boundaries.
- [x] Build deterministic partitions, EV/FE connectivity, vertex-star ribbons and ownership.
- [x] Provide patch/local handles and canonical host/GPU vertex attributes.
- [x] Implement GPU FV, EV/EF, VF, VE/VV, FF and edge-diamond results.
- [x] Document canonical IDs, ordering, strides, sentinels and empty-buffer behavior.
- [x] Verify query results across patch sizes against Alum, including repeated execution.
- [x] Run downstream boundary and neighborhood compositions using the public WESL package.
- [ ] Add canonical edge and face attributes with identity checks and reference tests.
- [x] Review the static API and buffer contracts before expanding them further.
- [ ] Stage topology in workgroup memory and evaluate patch-local traversal.
- [ ] Demonstrate locality benefits against an unpartitioned GPU baseline.

### GPU construction follow-ups

Building the arena on the GPU is in progress; see the
[strategy](docs/gpu_build_strategy.md).

- [ ] **Port more build logic into `shaders/`** (corner grouping, the fan walk,
  patch membership, local-ID lookup) when editing shows the interfaces it
  needs. Only value-level helpers have moved so far (`shaders/build/`).
- [ ] **Move CPU code used only to check the GPU behind a feature.** The
  public Rust API is for running the GPU shaders, not for running the
  algorithms on the CPU. Comparison-only CPU code (the alternative split
  methods, the explicit split, the arena validator, and eventually
  `MeshBuilder` itself if the GPU build replaces it) should sit behind a
  non-default feature such as `cpu-reference`, used by the tests. Until then
  it stays crate-private.
- [ ] **Revisit the edge ID ordering promise.** Ascending (smaller, larger)
  vertex-pair order costs a per-vertex count, scan and rank pass on the GPU.
  Once the GPU build is benchmarked, measure that cost and check whether
  callers still need stable, face-order-independent IDs; if not, numbering
  edge leaders in corner order is cheaper.
- [ ] **Check whether an RXMesh-style split is needed.** Lloyd iterations
  (`prior_art/RXMesh/include/rxmesh/patcher/`) give compact, near-equal
  patches. Deferred: build it only if the GPU build benchmark shows Morton
  patches hurting ribbon overhead or query times. First benchmark (2.1 M
  faces): Morton within 2–3% of breadth-first on ribbon overhead, query times
  within noise except centroids (+19%); not needed yet. Re-check on real,
  irregular meshes.
- [ ] **Speed up the GPU build further.** Done so far: one workgroup per
  patch-row sort, and a header kernel instead of a mapped arena (2.1 M
  faces: 173 → 145 ms; 32 K faces: 24 → 12 ms). Remaining: patch contents
  is still 63 ms at 2.1 M faces, mostly outside the sort (member writes,
  grouping, local-ID binary searches); at 32 K faces the 8-pass Morton sort
  and per-dispatch overhead keep the GPU build slower than the CPU path and
  Alum. Do the profiling item below before the next round.
- [ ] **Profile the GPU build per kernel.** The benchmark's stage breakdown
  submits and waits per stage, which hides which kernel inside a stage is
  slow. Use wgpu timestamp queries (where the adapter supports them) around
  each dispatch, for each grid size and split, and record the results in the
  benchmark doc before choosing the next optimization.
- [ ] **Maybe: a fully deterministic GPU path that matches the CPU exactly.**
  An optional mode keeping every intermediate in canonical order (sorted
  corner and patch rows) with a split the CPU can reproduce bit for bit
  (input order, or integer-exact Morton keys). It would restore word-for-word
  CPU comparison at some speed cost, and could serve callers who need results
  identical to the CPU builder. Decide after the fast path is benchmarked.

### Static API review follow-ups

Deferred findings from the milestone 1 review, to settle with measurements:

- [ ] **Shared face-vertex intermediate (P2).** `VertexFaceQuery`,
  `AdjacencyQuery` and `FaceCentroidQuery` each compile and run their own
  `FaceVertexQuery` (and two compile their own scan). Measure before adding
  shared-intermediate inputs.
- [ ] **Per-call allocation (P3).** Every `encode` allocates outputs,
  scratch and uniform buffers; the scan allocates one uniform per level.
  Measure the cost, then decide on reusable buffers.
- [ ] **Selective adjacency (P4).** `AdjacencyQuery` always computes VE/VV,
  FF and diamonds together. Measure selective execution.
- [ ] **Explicit counts in two shaders (L3).** `face_centroids.wesl` and
  `face_neighbors.wesl` bound loops with `arrayLength`, so they need
  exact-size inputs. Switch to explicit counts when buffers are pooled or
  oversized.
- [ ] **`missing_docs` (D-1).** Enable it for the public API in the release
  phase.

### 3 — Static algorithms: started

- [x] Implement face centroids and a reusable edge-length helper.
- [x] Validate weighted boundary lengths and neighborhood statistics in downstream examples.
- [ ] Add face/vertex normals and face areas with documented numerical behavior.
- [ ] Add reductions and smoothing, including boundary policies and reference checks.
- [ ] Decide whether a standalone edge-length batch API is useful; the WESL primitive already exists.
- [ ] Add runnable downstream examples alongside each new operator family.

### 4 — Local dynamics: not started

- [ ] Introduce editable storage, active masks, capacity metadata and full conflict footprints.
- [ ] Implement cavity discovery/staging, a single interior edge flip, then parallel compatible flips.
- [ ] Test failure/rollback behavior and topology/attribute preservation against Alum operations.

### 5 — Boundary dynamics: not started

- [ ] Implement cross-patch claims, separate commit phases, bounded retry queues and explicit failures.
- [ ] Repair ownership and ribbons after edits; validate migration and handle invalidation.

### 6 — Growth: not started

- [ ] Implement splits/collapses, slot reservation/recycling and overflow handling.
- [ ] Add compaction and patch slicing; test capacity pressure and attribute preservation.

### 7 — Mesh algorithms: not started

- [ ] Build Delaunay flipping, isotropic remeshing and decimation on validated edit operators.
- [ ] Define geometric criteria and termination; add reference comparisons and downstream compositions.

### 8 — Optimization and portability: not started as a measured program

- [ ] Profile adapters and evaluate optional subgroups and dispatch fusion using native 32-bit values.
- [ ] Measure reusable scratch/output buffers, shared intermediates and selective adjacency execution.
- [ ] Validate additional GPU vendors/backends and record capabilities and regressions.

Baseline measurement and the first locality experiment happen before dynamics,
not at the end of the project. Later tuning depends on those measurements.

### 9 — Release: not started; initial documentation/examples exist

- [ ] Stabilize public Rust/WESL contracts, packaging and reproducible dependency setup.
- [ ] Establish automated CPU/shader checks and a practical GPU validation matrix.
- [ ] Publish benchmark methodology/results, examples, topology constraints and release notes.
- [ ] Validate the release package and publish to crates.io when acceptance criteria are met.

The planned 0.1 scope includes static queries/attributes/geometry and experimental
local cavities. The current Cargo version does not mean that release scope is complete.

## Next execution milestones

1. **Static API and architecture review — done.** It covered layouts,
   identity/device requirements, ordering, sentinels, error handling,
   pipeline setup, batched-query cost and temporary allocations. Fixes landed
   with regression tests; the deferred findings are listed under
   [static API review follow-ups](#static-api-review-follow-ups) and feed the
   benchmark baseline.
2. **Benchmark baseline.** Add representative meshes and an unpartitioned WESL
   baseline. Measure construction/upload separately from warm GPU execution;
   report allocations, dispatches, query throughput, bytes/triangle, scratch and
   ribbon overhead. Record adapter, limits, patch size and timing method.
   Acceptance: reproducible runs and a baseline report, with no assumed speedup.
   An [approximate comparison against Alum](docs/approximate_benchmark.md)
   exists. Host construction was rewritten after it showed construction about
   16x slower than Alum's; it is now about 2.4x slower. Construction on the
   GPU is in progress ([strategy](docs/gpu_build_strategy.md)); it is already
   1.3–1.4x faster than Alum's build at 522 K faces and above.
3. **Workgroup-local traversal experiment.** Select a bottleneck from the
   baseline; edge incidence is a candidate because it currently scans local
   faces per owned edge. Compare staged and existing paths with identical
   outputs. Acceptance: Alum parity and measured tradeoffs across mesh/patch
   sizes; retain or revise the approach based on evidence.
4. **Complete useful static functionality.** Add edge/face attributes and the
   remaining Phase 3 operators, with public WESL implementations and downstream
   examples. Acceptance: documented contracts, reference checks and measured costs.
5. **Begin local dynamics.** Use the reviewed storage/synchronization contracts
   for Phase 4. Acceptance starts with a correct single interior flip before
   parallel claims or cross-patch editing.

Do not begin cross-patch transactions or growth before the local-edit invariants
are tested. CUDA benchmark availability need not block the initial same-adapter
comparison against the unpartitioned WESL baseline.

## Discoveries and implementation consequences

- **Correctness is ahead of locality.** Patch/ribbon storage exists, but current
  topology traversal uses storage buffers directly; VF and VE/VV transpose
  globally. Edge incidence costs owned edges times local face count per patch.
  These are baseline algorithms, not demonstrated RXMesh-like performance.
- **Owned-face limits do not bound total patch size.** Full vertex-star ribbons
  can grow substantially around high-valence vertices. Measure actual patch
  footprints before selecting workgroup arrays or claiming a fixed memory budget.
- **Composition has allocation costs.** Query objects reuse pipelines, but each
  encode allocates outputs/scratch. `AdjacencyQuery` computes VE/VV, FF and
  diamonds together. Measure selective execution and scratch reuse before
  committing to a larger abstraction.
- **Ordering is part of the API.** Canonical edge IDs are constructed IDs, not
  Alum IDs. CSR rows are unordered; FF preserves input edge-slot order; diamonds
  pair opposites with incident faces rather than forming a cyclic cavity.
  Editing operators must reconstruct orientation explicitly.
- **WGSL layout needs explicit contracts.** Contiguous u32 FV/FF triples use 12-byte
  records, unlike 16-byte WGSL vec3 array strides. GPU attributes therefore
  support scalar/vec2/vec4 representations; positions use padded vec4 values.
  Empty buffers contain placeholders with zero logical records.
- **Portable helper APIs use values.** An early storage-pointer function
  approach failed baseline shader validation. Current helpers accept values;
  consumers load their own buffers. Collective scan calls require all 64 lanes
  in uniform control flow, with zero padding rather than early returns.
- **Attributes are snapshots.** Host edits require explicit uploads; GPU writes
  do not update host data. Queue writes affect the next submission, including
  commands already encoded. Attribute updates do not revalidate geometry.
- **Lifetimes and identities are distinct.** Keep the caller's queue alive while
  encoding/submitting work. Every attribute and query result reports a
  `MeshId`; built-in queries check typed attribute inputs, and downstream
  compositions compare IDs themselves. Device identity is documented as a
  caller responsibility and not checked. Handles are not generation-checked.
- **Reusable operators are core scope.** Both runnable examples import only
  public WESL symbols and choose their own bindings. Common operations belong
  in the library, and runtime wrappers use them. New families need linking and
  GPU-result checks, not just internal shader compilation.
- **Portability remains an evidence gap.** Baseline Naga validation and Metal
  execution do not establish Vulkan, DX12, browser or cross-vendor behavior.
  Capacity/edit failures and large real-world datasets are also untested.

## Beyond the initial release

Cross-patch remeshing/decimation and specialized performance paths remain in the
later phases above. Stable IDs across migration, richer visualization/inspection
tools and browser integration are possible extensions, not prerequisites for
the current prototype. Numerical solvers, automatic differentiation and full
CUDA API parity remain deferred. Downstream examples are required core work,
not stretch goals.
