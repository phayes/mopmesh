# Static API and architecture review

Review date: 26 November 2026. Scope: roadmap milestone 1 — the host builder,
handles, patches, vertex attributes, every `mopmesh::gpu` query, the public WESL
package, the runtime shaders and both downstream examples, checked against the
README contracts.

Each finding has a disposition:

- **Fix now** — conventional answer; applied in this milestone with regression tests.
- **Decision** — more than one defensible contract; recorded below with the choice made.
- **Deferred** — needs measurement (milestone 2) or belongs to a later phase.

No correctness bug in the current query results was found. The GPU suite and
host suite pass on Apple M1 Max / Metal at the start of the review. The findings
are contract, consistency and cost issues.

## Error handling

| # | Finding | Where | Disposition |
|---|---------|-------|-------------|
| E1 | Two GPU error types: `GpuLimitError` (unit struct) for topology queries and `GpuAttributeError` for attributes/centroids, which duplicates `LimitsExceeded`. A caller composing both needs two error conversions, and a limit failure says nothing about which limit. | `src/gpu.rs:14`, `src/gpu/attribute.rs:24` | Fix now: one `GpuError` enum. |
| E2 | `FaceVertexQuery::encode` returns a bare buffer and checks no limits; it relies on checks made by `GpuMesh::upload`, possibly against another device. Every other query returns `Result`. | `src/gpu.rs:166` | Fix now: return `Result` and check its own output/dispatch limits. |
| E3 | `BuildError`'s `Display` prints its `Debug` form. | `src/mesh.rs:21` | Fix now: readable messages. |
| E4 | The limit expression `max_buffer_size.min(max_storage_buffer_binding_size)` and the 64-lane dispatch check are copied into six places; only `ScanPlan` can be tested without a GPU. | `src/gpu.rs:50,321`, `src/gpu/adjacency.rs:150`, `src/gpu/attribute.rs:117`, `src/gpu/edge.rs:59`, `src/gpu/scan.rs:45` | Fix now: shared private helpers, unit-tested against `wgpu::Limits`. |

## Identity and device requirements

| # | Finding | Where | Disposition |
|---|---------|-------|-------------|
| I1 | Nothing checks that a `GpuMesh`, attribute and query object share one device. A mismatch reaches wgpu validation, which panics or reports an uncaptured error instead of returning an error. `wgpu::Device` is `Clone + Eq`, so a check would be cheap. | `src/gpu.rs`, all `encode` methods | Decision D1. |
| I2 | Topology results (`EdgeConnectivity`, `MeshAdjacency`, vertex-face CSR) carry no mesh identity, so downstream operators cannot check that edge records and positions come from one mesh. The neighborhood example compares lengths as a proxy. | `src/gpu/edge.rs`, `src/gpu/adjacency.rs`, `examples/support/neighborhood.rs` | Decision D2. |
| I3 | Handles are not generation-checked and the README says so. | `src/handle.rs` | Deferred to Phase 4 (editable storage). |

## Result types and layouts

| # | Finding | Where | Disposition |
|---|---------|-------|-------------|
| L1 | `FaceVertexQuery` and `FaceCentroidQuery` return raw `wgpu::Buffer`; the other queries return typed results with logical counts. Callers must recompute counts and cannot tell a placeholder from data. | `src/gpu.rs:171`, `src/gpu/attribute.rs:161` | Fix now: `FaceVertices` and `FaceCentroids` result types. |
| L2 | Host `VertexAttribute` can only be filled with one value or the positions. There is no constructor from per-vertex values and no slice access, so both examples upload their weights as raw buffers instead of attributes. | `src/mesh.rs:335`, `examples/support/*.rs` | Fix now: `Mesh::vertex_attribute_from` and `VertexAttribute::as_slice`/`len`. |
| L3 | `face_centroids.wesl` and `face_neighbors.wesl` bound their loops with `arrayLength`, so they depend on exact-size inputs. That holds today, but would break once buffers are pooled or oversized. | `runtime_shaders/face_centroids.wesl:9`, `runtime_shaders/face_neighbors.wesl:12` | Deferred: switch to explicit counts when scratch reuse lands (milestone 2). |
| L4 | Layout contracts (12-byte FV/FF records, 16-byte edge and diamond records, CSR offset units, placeholders) match the README and the tests. | README, `tests/gpu_topology.rs` | No change. |

## Ordering and sentinels

| # | Finding | Where | Disposition |
|---|---------|-------|-------------|
| O1 | `0xffffffff` is spelled three ways: `INVALID_FACE`, a bare literal in `other_edge_vertex`, `face_edge_slot` and `opposite_vertex`, and `u32::MAX` on the host. There is no named vertex or slot sentinel. | `shaders/topology/edge.wesl`, `shaders/topology/face.wesl`, `runtime_shaders/face_neighbors.wesl` | Fix now: public `INVALID_VERTEX`/`INVALID_SLOT` WESL constants and a Rust `gpu::INVALID_ID`. |
| O2 | Ordering contracts (ascending edge endpoints, unordered CSR rows, FF input-slot order, side-aligned diamonds) match the README and are asserted by the tests. | README | No change. |

## Pipeline setup, batched cost and temporary allocations

| # | Finding | Where | Disposition |
|---|---------|-------|-------------|
| P1 | Pipeline, bind-group-layout, bind-group and dispatch boilerplate is repeated in every query. Labels such as `"count"` are reused by two queries, so captures are ambiguous. | `src/gpu.rs`, `src/gpu/*.rs` | Fix now: one private helper module with labelled pipelines. |
| P2 | `VertexFaceQuery`, `AdjacencyQuery` and `FaceCentroidQuery` each compile their own `FaceVertexQuery`, and two of them compile their own scan. One application using all three compiles FV three times and evaluates it three times per frame. | `src/gpu.rs:294`, `src/gpu/adjacency.rs:124`, `src/gpu/attribute.rs:146` | Deferred: measure in milestone 2 before adding shared-intermediate inputs. |
| P3 | Every `encode` allocates its outputs, scratch and uniform parameter buffers; the scan allocates one uniform buffer per hierarchy level. | `src/gpu/scan.rs:123`, `src/gpu/adjacency.rs:195`, `src/gpu.rs:341` | Deferred: measure allocation cost in milestone 2. |
| P4 | `AdjacencyQuery` always computes VE/VV, FF and diamonds together. | `src/gpu/adjacency.rs` | Deferred: measure selective execution in milestone 2. |

## Documentation

| # | Finding | Where | Disposition |
|---|---------|-------|-------------|
| D-1 | Many public host items (`Mesh`, `Patch` accessors, `MeshBuilder::build`) have no doc comments; `Patch::face_vertices` does not say it returns local IDs. | `src/mesh.rs`, `src/patch.rs` | Fix now for items touched here; enabling `missing_docs` is deferred to the release phase. |

## Contract decisions

- **D1 — device identity: document only.** Every resource passed to one call
  must come from the same `wgpu::Device`. MOPMesh does not check this; wgpu
  validation reports a mismatch. Documented in the `gpu` module docs and the
  README's shared contracts. Storing the device in each resource was considered
  and rejected for now.
- **D2 — result identity: public `MeshId`.** `Mesh::id()` returns an opaque,
  comparable ID. `VertexAttribute`, `GpuMesh`, `GpuVertexAttribute` and every
  query result expose `mesh_id()`. Built-in queries check typed attribute
  inputs; downstream operators compare IDs themselves, as both examples now do.
  The ID comes from a process-wide counter, so two separately built meshes
  never share an ID.

## Applied changes

- E1: `GpuLimitError` and `GpuAttributeError` are replaced by
  `GpuError { LimitsExceeded { resource }, MeshMismatch }`.
- E2: `FaceVertexQuery::encode` returns `Result<FaceVertices, GpuError>` and
  checks its own output and dispatch limits. `GpuMesh::upload` checks only the
  arena.
- E3: `BuildError` has readable `Display` messages.
- E4/P1: `src/gpu/pipeline.rs` holds the limit checks and wgpu setup helpers.
  Pipelines and passes have unique labels.
- L1: new `FaceVertices` and `FaceCentroids` result types with counts and mesh IDs.
- L2: `Mesh::vertex_attribute_from`, `VertexAttribute::{as_slice, len, is_empty, mesh_id}`.
  The neighborhood example now takes its weights as a `GpuVertexAttribute<f32>`.
- O1: WESL `INVALID_VERTEX` and `INVALID_SLOT` constants, and Rust `gpu::INVALID_ID`.

Regression tests: hardware-independent unit tests for binding limits,
workgroup counts and arena sizing; host tests for `vertex_attribute_from`,
`MeshId` and `BuildError` messages; downstream linking of the new constants;
GPU tests for mesh IDs on every result, typed result counts, per-vertex
attribute upload order and the face-vertex dispatch limit.
