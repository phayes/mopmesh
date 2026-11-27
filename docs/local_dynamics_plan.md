# Milestone 5 plan: local dynamics

Status: approved 26 November 2026. Nothing here is implemented yet.

## Direction: in-kernel local queries

The user's direction, given when approving this plan: editing is the
standout feature of this crate, and we definitely want to revisit in-kernel
local queries. On a separate occasion they also said to follow RXMesh.
Nothing more specific has been decided.

What follows is a proposal, not yet put to the user:

- Editable meshes could use RXMesh-style in-kernel local queries: each kernel
  loads its patch's current EV/FE into workgroup memory and derives adjacency
  on the spot, so nothing is kept that could go stale.
- Global query buffers would stay as the simple static API and the test
  oracle.
- The topology version check in this plan would then be a safeguard rather
  than the end state.
- One way to make Milestone 3's WESL staging usable by editing kernels:
  patch capacity, tombstoned slots, active masks, and kernels the consumer
  writes that call library staging functions, since WGSL has no callbacks.

Check these choices with the user before treating them as decided.

## Bottom line

- **New type `GpuEditableMesh`.** It holds a read-write connectivity arena with
  spare capacity per patch, active masks, owner tables and a topology version.
  `GpuMesh` and its arena stay immutable and unchanged.
- **Masks and capacity from the start.** A flip deletes two faces and one edge
  in their patch's local slots and adds the replacements in fresh local slots,
  in the RXMesh style. The global IDs are kept: the edge and both faces keep
  theirs, so attributes indexed by global ID keep their slots. Phase 6 adds a
  global free list for splits and collapses.
- **Free local slots are reused, as in RXMesh.** A new element claims the first
  inactive slot in its patch's active bitmask with an atomic try-set
  (`atomicOr`, keeping the slot only if its bit was clear). Slots freed by a
  batch become reusable from the next batch, so a rejected or interrupted edit
  never overwrites rows it still needs. Capacity only has to cover one batch's
  new elements, not every edit ever made.
- **Global IDs, with global capacity deferred.** Attributes and query results
  stay indexed by global ID. Flips never create or delete global IDs, so global
  spare capacity and global active masks arrive in Phase 6, together with the
  splits and collapses that need them. Until then, attributes keep their
  current sizes. Changed after approval, because no kernel in this milestone
  would read them.
- **Topology version, not just `MeshId`.** Each committed edit batch bumps a
  version, every time an edit is encoded, because only the GPU knows whether
  anything was accepted. Topology results (adjacency, edge connectivity, face
  vertices) record `(MeshId, version)`. Queries that take them return a new
  `GpuError::Stale` on a mismatch. Attributes the library computes from
  topology (boundary flags, normals, areas, centroids) are `GpuAttribute`s and
  get an optional topology stamp, checked the same way when one is present.
  Attributes the caller uploads, such as positions, have no stamp, because edits
  don't invalidate them.
- **Edge ordering promise narrowed.** Ascending vertex-pair order holds only for
  freshly built meshes. A flipped edge keeps its ID and attribute slot, and its
  value is left as it was.
- **Strictly interior flips only.** A flip is accepted only when every face
  incident to any of the four diamond vertices is owned by one patch. Then no
  other patch holds a ribbon copy of anything the flip reads or writes. Every
  other flip is rejected with a status, never applied partially. Flips across
  patch boundaries are Phase 5.
- **Order of work.** First one correct flip, then parallel compatible flips in
  one submission, with a claim pass. Stop there; migration and cross-patch
  claims are out of scope.
- **Coordination point.** Static query kernels must skip tombstoned local slots
  and read per-patch counts from a header. That touches shared WESL accessors,
  which the other session's Milestone 3 staging also changes. See
  [Coordination](#coordination).

## Why strict interior is enough to start

A patch stores every face incident to a vertex of one of its owned faces
(`MeshBuilder::build`). Suppose edge `ab` is flipped, with opposite vertices
`c` and `d`. The old faces `abc` and `abd` become `acd` and `bcd`, and both
use the same four vertices. Patch Q holds a copy of `abc` if Q owns a face at
`a`, `b` or `c`. Q needs `acd` if Q owns a face at `a`, `c` or `d`. So if
every face at `a`, `b`, `c` and `d` is owned by patch P:

- no other patch holds a copy of the old faces or edge, or needs the new ones;
- P's own ribbon set is unchanged, because the union of vertices of its owned
  faces is unchanged;
- ownership is unchanged: the new edge and faces are owned by P, and each
  vertex's owner (the minimum patch over its star) is still P.

A flip near a patch border fails this test even when the edge itself is owned
by P. That case is the main rejection test.

## Editable storage contract

The arena is one storage buffer in `u32` words, like the static arena.

- **Header:** global vertex, edge and face counts; the patch count; word
  offsets of the patch records and of the three owner tables.
- **Patch records:** one fixed-size record per patch, holding local slot
  capacities for vertices, edges and faces, the word offsets of that patch's
  arena regions, and the word offsets of its masks in the mask buffer.
- **Per-patch regions**, sized by capacity, not count: global vertex IDs, EV
  (two local vertex IDs per edge), FE (three `(local edge, reversed)` pairs per
  face), global edge IDs and global face IDs. Unused slots hold `INVALID_ID`.
- **Global owner tables:** for each global vertex, edge and face ID, the owner
  `(patch, local)` handle, as two words.
- **Mask buffer:** a second buffer with each patch's active, owned and
  interior bitmasks, one bit per local slot, packed into `u32` words.

The static arena's face and edge task lists are left out. Every task field is
a patch offset or a local index, which the owner handle and patch record
already give. So queries dispatch over global IDs through the owner tables,
and a flip commit has no task lists to rewrite. That also saves `4F + 7E`
words. Changed after approval.

Deleted local slots are tombstones: their active bit is cleared and their FE
and EV rows are overwritten with `INVALID_ID`. Capacity per patch is
`ceil(count * capacity_factor)`, where `capacity_factor` is set by the caller.
The default is 1.5, as in RXMesh's `capacity_factor`; the real need is unmeasured.

Several winning flips in one patch can set and clear bits in the same mask
word, and disjoint vertex claims don't protect it. So the active and owned
masks, both local and global, live in a separate mask buffer bound as
`array<atomic<u32>>`, and are only changed with `atomicOr` and `atomicAnd`.
Kernels that only read masks bind the same buffer read-only.

Parallel invocations must never write different components of the same
storage vector, such as the lanes of a `vec2<u32>` row. On Metal some of those
writes were lost, as if the whole vector had been stored. EV, FE, owner tables
and status outputs therefore use separate `u32` words, and every word that
concurrent invocations share is updated with atomics.

Each patch record also holds an interior-vertex bitmask, built at upload: a
vertex is interior when its whole star is owned by that patch. Interior-only
flips never change it, so the interior test is four bit lookups rather than a
scan of each star.

API sketch, names provisional:

```rust
pub struct EditCapacity { /* factor, checked finite and >= 1 by `new` */ }
pub struct GpuEditableMesh { /* arena, counts, capacities, MeshId, version */ }
impl GpuEditableMesh {
    pub fn upload(device: &wgpu::Device, mesh: &Mesh, capacity: EditCapacity)
        -> Result<Self, GpuError>;
    pub fn version(&self) -> u32;
}
pub struct EdgeFlipQuery { /* pipelines */ }
impl EdgeFlipQuery {
    /// Flips each edge whose request flag is nonzero; returns one status per edge.
    pub fn encode(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder,
        mesh: &mut GpuEditableMesh, requests: &GpuEdgeAttribute<u32>)
        -> Result<GpuEdgeAttribute<u32>, GpuError>;
}
```

Status values: not requested, flipped, boundary edge, not interior, opposite
vertices already connected, degenerate (`c == d`, or a repeated face), lost a
claim, and out of local capacity.

Existing queries take a shared `GpuTopology` trait (arena, counts, `MeshId`,
version), implemented by both mesh types. Whether editable meshes need
separate pipeline variants or shared kernels can read both arenas depends on
the accessor work below.

## Steps

The version check was step 1, but nothing can change a version before a flip
exists, so it couldn't be tested. The flip was then moved ahead of the query
step: the queries wait on the shared accessor layer, and the flip only needs
the editable arena. The version counter lands with the flip; `GpuError::Stale`
and its test land with the queries, the first results to carry a version.

**WESL first.** The WESL package is a standalone library, used as one stage
of larger pipelines. Calls from other WESL code matter more than calls from
Rust. So edit logic lives in public, value-based modules under `shaders/`
(`topology/flip.wesl` for the flip). Runtime kernels and Rust wrappers are
thin layers over those modules.

Decided (user, 26 November 2026): **value-based commit.** The library
returns the words to write, and the consumer's kernel stores them itself.
Concretely, `flip.wesl` will offer a function that takes the decoded patch
record (from `topology/patch.wesl`), the flip's quad and its old and new
slots. It returns a fixed-size list of arena writes, each an offset and a
value, plus a list of mask updates (offset, bits, and whether to set or
clear them). A consumer applies them with plain stores and
`atomicOr`/`atomicAnd` on its own bindings. The runtime flip kernel becomes
a loop over that list. Helpers for picking free slots from mask words are
value-based too.

**This API needs good documentation** (user, 26 November 2026). Applying a
list of writes correctly is easy to get wrong, so the docs must cover:
- a worked example kernel that tests a condition, flips, and applies the
  writes;
- which mask updates must be atomic, and why plain stores are safe for the
  arena words;
- what the caller must check first: the flip's status, slot availability,
  and claims once flips run in parallel;
- what happens if the writes are applied partially or twice.
The example should run in a test, not only compile, and the docs should be
reviewed before the step is called done.

Background: the commit step (writing the arena and masks) can't yet be a
library function. Naga 30 doesn't implement WGSL's
`unrestricted_pointer_parameters`, so a library function can't take a
caller's storage buffer as a parameter. Any library function that writes
the mesh would have to use bindings the library declares itself. The
options are:
- an opt-in module that declares the editable bindings at a group the
  consumer picks (for example through a WESL `@if` or a generated variant);
- a value-based commit that returns the words to write, which the consumer
  stores itself;
- leaving the commit in a library kernel and giving up a single-dispatch
  predicate-plus-flip in consumer code.

1. **Editable arena and upload.** A host-side layout builder with capacities,
   masks and owner tables, plus `GpuEditableMesh::upload`. One layout
   definition is used both to build the arena and to decode it in tests.
   Host-side tests:
   - decoding gives back the source triangles, at several patch sizes and
     for an empty mesh;
   - active bits match the counts, and owned bits match the owner tables;
   - the interior bitmask matches a brute-force scan of each star;
   - invalid capacity factors and device limits are rejected before
     anything is allocated.
2. **Single flip and the version counter.** Public helpers `flip_quad` and
   `flipped_faces` in `shaders/topology/flip.wesl`. `EdgeFlipQuery` flips one
   edge per call and bumps `GpuEditableMesh::version`. One kernel validates
   and then commits:
   - It finds the diamond from FE/EV in the owner patch, and rebuilds the
     orientation explicitly, because diamonds are not cyclic.
   - It checks boundary, interior, connected-opposites and degeneracy.
   - It picks free slots by reading the active masks. It doesn't reuse the
     slots it frees, and raises the used counts only at commit.
   - It tombstones the old slots, writes the new EV/FE rows, and updates the
     owner tables and active masks.
   - Rejections write a status and change nothing else, used counts
     included. In the parallel step, slots will be claimed with an atomic
     try-set and released on failure.
3. **Queries on the editable arena, and `Stale`.** A shared `GpuTopology`
   trait carries the arena, counts, `MeshId` and version, and results record
   the version. `GpuError::Stale`, the optional topology stamp on derived
   attributes, and the staleness test land here. Build on the other
   session's value-based accessors (`shaders/topology/patch.wesl`). Run the
   full existing query suite on unedited and edited meshes against Alum.
4. **Parallel compatible flips.** Add a claim dispatch: each candidate does an
   `atomicMin` of its priority on all four diamond vertices, and a later
   dispatch keeps only the candidates that won all four. The priority is a
   bijective hash of the global edge ID, reseeded on each call. Raw IDs would
   make grid diagonals resolve as a wavefront, about one row per round.
   Claiming vertices is conservative. It covers shared faces, shared edges and
   two flips creating the same new edge. No CPU readback happens between
   dispatches; the caller retries losers by calling again. Bounded retry
   rounds are the caller's choice for now.
5. **Example and docs.** A downstream example that flips grid diagonals by a
   geometric rule, such as a Delaunay-style angle test, written in public WESL.
   It runs queries before and after the flips. Update the README contracts and
   the ROADMAP.

## Tests

- **Oracle.** Apply Alum's `swap_edge_ccw` to the same edges, then compare face
  sets as vertex triples up to rotation. Then rebuild a fresh `Mesh` from the
  flipped triangles, and compare every query on the edited mesh with the fresh
  one. Compare by vertex pairs and sets, since edge IDs differ. Alum doesn't
  reject flips whose opposite vertices are already connected, so we check that
  case ourselves.
- **Rejection is a no-op.** For each rejection kind, read the arena back
  before and after. The bytes must be identical, apart from the status
  output.
- **Fixtures:**
  - an interior grid edge;
  - an edge next to a patch border at a small patch size, which is expected
    to be rejected as not interior;
  - a boundary edge;
  - a tetrahedron, where the opposite vertices are already connected;
  - every grid diagonal flipped in parallel on a single-patch mesh, retrying
    until no lost-claim statuses remain. A multi-patch version computes the
    expected not-interior set on the host;
  - a flip, then a flip back, with the topology compared to the original;
  - many rounds of flips at a capacity factor that only covers one batch,
    which shows that freed slots are reused;
  - running out of capacity at a factor of 1.0.
- **Staleness.** A result from before an edit, passed to a query after it,
  returns `Stale`.

## Out of scope

Cross-patch flips, ribbon repair, ownership migration, patch slicing, splits,
collapses, the global free list and compaction (Phases 5 and 6). Arg-min/max and
the other Milestone 4 leftovers are also out of scope.

## Coordination

Step 2 changes how kernels address local slots: they go through patch records
and skip inactive slots. The other session's Milestone 3 workgroup staging
reads the same patch data. Before step 2, agree on one accessor layer in
`shaders/topology/` that both use. Step 1 doesn't touch shared kernels or
shared Rust files, apart from one `mod` line in `src/gpu.rs`.

## Open questions

- Should static `GpuMesh` adopt the editable arena's header and patch records,
  so that there is a single kernel set? That would be simpler long-term, but it
  changes the arena the other session benchmarks.
- What does editable storage cost in memory? The static arena already uses
  129–145 B/face, against 36 B/face for flat global arrays (baseline
  benchmark, 2.1M faces). Capacity slack, masks, owner tables and the
  interior bitmask add to that. Report B/face for the editable arena next to
  the static one.
- Is a default capacity factor of 1.5 right? With slot reuse, a batch needs one
  free edge slot and two free face slots per flip in each patch. Measure
  once flips run.
