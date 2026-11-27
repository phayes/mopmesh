# MOPMesh technical design and feasibility assessment

Rust / WESL / WebGPU • 26 November 2026

## Executive summary

MOPMesh (Mahmoud–Owens–Porumbescu Mesh) is an independent Rust/WESL prototype inspired by RXMesh, the Ribbon-matriX Mesh architecture introduced in 2021 and extended to dynamic topology in 2025. Its goal is GPU-resident triangle mesh processing on multiple GPU vendors, without a CUDA dependency.

The project is technically plausible, but performance parity with CUDA is unproven. Preserve patch locality, ribbon ownership, connectivity queries and cavity-based editing; redesign cross-workgroup coordination around explicit dispatch phases. Establish correctness and measure locality benefits before committing to full dynamic parity.

The static query families, vertex attributes, hierarchical scans and two downstream WESL examples are implemented. Geometry algorithms are partial; dynamic topology is still a design. The [maintained roadmap](ROADMAP.md) records completed work, remaining acceptance criteria, discoveries and the next execution sequence. This document describes both implemented architecture and explicitly planned extensions; the [README](README.md) is the current API and layout guide.

## Recommended initial scope

- Static partitioned mesh, EV/FE connectivity, ribbon ownership and typed attributes.

- Portable workgroup primitives. Alum is a best-effort CPU reference: we accept differences from it, and project CPU models serve as oracles where Alum's operations don't match ours.

- Reusable WESL library modules that downstream WESL projects can import and link independently of MOPMesh host execution.

- Local edge flips, then cross-patch edits, allocation, migration and slicing.

- Defer numerical solvers, automatic differentiation and full CUDA API parity.

## Decision gates

| **Gate**             | **Evidence required**                                                                     |
|----------------------|-------------------------------------------------------------------------------------------|
| Static viability     | Correct queries; measured advantage over a global-memory baseline.                        |
| Dynamic viability    | Conflict-free local and boundary edits; bounded retries; no topology corruption.          |
| Production viability | Cross-vendor performance, reliable capacity handling and documented topology constraints. |

All three gates remain open. Query correctness has fixture coverage, but static locality benefit has not been measured. Dynamic and production viability have not been demonstrated.

# 1. Upstream and licensing

RXMesh is maintained in owensgroup/RXMesh. The 2021 paper introduces the patch/ribbon structure; Dynamic Mesh Processing on the GPU (2025), by Ahmed H. Mahmoud, Serban D. Porumbescu and John D. Owens, introduces dynamic cavity operations. \[1–3\]

## BSD-2-Clause

The upstream repository uses BSD-2-Clause. Its copyright notice is “Copyright (c) 2022, owensgroup.” The license permits modification, translation, redistribution, commercial and proprietary use, subject to preservation of the original notice, conditions and disclaimer. Binary redistributions must reproduce these in accompanying materials. \[4\]

MOPMesh uses BSD-2-Clause; `LICENSE`, `THIRD_PARTY_NOTICES` and the README retain project licensing and reference information. A release dependency/asset audit remains future work. BSD-2-Clause does not expressly grant patent rights; avoid implying author endorsement through the project name.

## Running the reference without CUDA

| **Option**         | **Assessment**                                                                      |
|--------------------|-------------------------------------------------------------------------------------|
| Remote NVIDIA GPU  | Preferred for reproducible reference outputs and comparable benchmarks.             |
| SCALE              | CUDA compilation targeting AMD; inline PTX and library compatibility merit testing. |
| ZLUDA              | Experimental CUDA translation targeting AMD; compatibility unverified.              |
| HIP / ROCm or SYCL | Substantial additional porting work; not a drop-in solution.                        |
| CPU emulation      | Possible experimental route, unsuitable for performance comparison.                 |

No official non-CUDA RXMesh backend was identified. SCALE and ZLUDA are candidates for investigation, not validated RXMesh execution paths. \[5–6\]

# 2. Architecture and GPU representation

Rust owns device integration, CPU mesh construction/partitioning, resource lifetime and pipeline objects. The application owns the device, queue, command encoder and submission; queries record commands and expose GPU-resident outputs. WESL implements connectivity traversal and current geometry/parallel primitives. Topology transactions and patch maintenance are planned responsibilities, not implemented features.

## Reusable WESL library requirement

The shader code is a public library as well as the implementation behind the Rust API. Other WESL projects must be able to import its types, connectivity helpers, parallel primitives and operators without adopting MOPMesh compute entry points or host scheduling. Ship a `wesl.toml` and a Cargo WESL package, with explicit public symbols and documented module paths.

Keep reusable modules independent of application-specific binding numbers and entry points. Put concrete storage bindings, workgroup sizing and dispatch wrappers in separate integration modules. Document storage layouts, capacity requirements and synchronization preconditions as those APIs are introduced. Library linking must be covered by a downstream consumer test with its own bindings and entry point; syntax-only compilation of internal shaders is insufficient.

Common operations belong in the public WESL library as they are implemented: connectivity traversal, geometry primitives, scans and reductions, and operators such as normals, areas, edge lengths and smoothing. The Rust runtime must use these same implementations. Keep allocation, concrete bindings, pipeline configuration and dispatch scheduling in the runtime integration layer. Document any workgroup-size or collective-call requirements of public primitives.

Downstream example applications are part of the core implementation plan and API acceptance criteria, not a stretch goal. Add a small runnable example alongside each major operator family. Each example should import the packaged library, own its bindings and entry points, and compose public operations with application-specific data or behavior. Generally useful operations discovered while building an example should become library functions rather than duplicated example code. Use concrete consumers to guide abstractions instead of designing speculative interfaces.

Accept an operator family when its downstream example links using only the public package, runs with documented layouts and synchronization, and produces validated results. Use Alum as a best-effort mesh reference where it fits. If an example needs private runtime internals or duplicates a common operation, address the missing public API before considering the family complete. Keep downstream linking checks in the hardware-independent suite and execution checks in the explicit GPU suite.

| **Layer**     | **Responsibility**                                                      |
|---------------|-------------------------------------------------------------------------|
| Rust host     | Builder, partitioner, handles, typed attributes, scheduler, validation. |
| WESL core     | Native 32-bit types, buffer offsets, bitmasks, arenas, ownership resolution. |
| WESL parallel | Scan, reduce, histogram, compaction and optional subgroup variants.     |
| WESL topology | Patch/ribbon queries, cavity construction, claims, commit and repair.   |
| WESL operators | Reusable geometry operations, smoothing and, in later phases, remeshing and decimation operators. |
| Applications  | Compose public operators with application-specific data, policies and behavior. |

The layer responsibilities above describe the target architecture. Bitmasks, histograms, compaction, cavity edits and scheduling are not yet implemented. Numeric packing is outside the committed roadmap.

## Patch and ribbon model

Each patch owns mesh elements and stores replicated neighboring connectivity (ribbons), allowing local queries without accessing unrelated patches. Preserve EV (edge-to-two-vertices) and FE (face-to-three-directed-edges) as primary connectivity. Derive FV, VE, EF, VV, VF, FF and edge-diamond queries by local composition and transposition.

Use explicit patch/local handles initially; document invalidation after migration or slot reuse. Add generation counters or a separate stable-ID mapping where stale-handle detection is required. Ownership maps resolve ribbon elements to canonical owners.

## Storage layout

The initial release uses native `u32`/`i32` integer values and `f32` geometry on the GPU, with `u32` connectivity indices and word offsets. Vectors and records contain independent 32-bit fields. Do not introduce 16-bit formats or subword numeric packing, native or emulated 64-bit GPU values, or paired-float (single-single/double-single) precision schemes. Exceeding index/device limits is an error, not a reason to select a wider representation. Alternative widths or compression require a measured need and a separate design decision; they are not committed future work.

This constraint does not apply to host allocation arithmetic: retain `u64`/`usize` for buffer sizes, overflow checks and wgpu interfaces. This does not constitute a wider GPU representation.

The implemented static arena uses u32 word offsets and unpacked connectivity, with host-generated owned-face and owned-edge tasks. CPU patches carry ownership maps; the GPU arena does not yet contain general ownership metadata, active masks, free slots or edit capacities. Canonical vertex attributes and query results live in separate buffers. See the [current layouts](README.md#gpu-usage) before writing a consumer.

For dynamics, extend metadata with capacities, flags and ownership information and add separate logical regions for masks, transactions, scheduling and maintenance. Fixed-capacity patch allocations are a proposed starting point for editable storage, not a description of the current immutable upload.

Current handles use this representation; they are mesh-relative and not generation-checked:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexHandle {
    pub patch: u32,
    pub local: u32,
}
```

The WESL handle names the patch field `patch_id` because `patch` is reserved in WGSL. Rust handles are not currently advertised as directly uploadable GPU attribute elements.

# 3. CUDA-to-WebGPU adaptation

| **CUDA mechanism**         | **Portable replacement**                          |
|----------------------------|---------------------------------------------------|
| Thread blocks / threads    | Workgroups / invocations                          |
| Dynamic shared memory      | Pipeline-specialized workgroup arrays             |
| Warp primitives / CUB      | Portable scans and reductions; optional subgroups |
| Pointers / 16-bit indices  | Native u32 word offsets and connectivity indices |
| 64-bit atomics             | Redesign claims around native 32-bit atomics; no 64-bit emulation |
| Cross-block locks          | Nonblocking claims and separate dispatch phases   |
| Dynamic allocation         | Preallocated arenas, free lists, compaction       |
| Device lambdas / templates | WESL imports and specialization; Rust generics    |

## Memory budget

Select memory profiles from the actual adapter and requested device limits rather than treating a generic WebGPU limit as a guarantee. Current compute kernels use 64 invocations; the scan collective uses 256 bytes of workgroup scratch. Runtime queries use at most four storage bindings and one uniform binding; the neighborhood example uses five storage bindings. Pipeline/device profiles beyond those exercised on Metal remain to be validated. \[7–9\]

Budget for the actual 32-bit representation. The current static connectivity arena uses `4 + 4F + 7E + sum_p(Vp + 2Ep + 7Fp)` u32 words: a four-word header, owned face/edge tasks, and per-patch vertex IDs, EV, FE and face IDs. `F` and `E` are global counts; per-patch counts include ribbons. Multiply by four for arena bytes. This excludes attributes, result buffers, scan scratch, host storage and future edit metadata. The earlier 16-bit packing estimates do not apply and have been removed.

Begin experiments with 128–256 owned faces and 64–128 invocations, but measure full ribbon footprints: an owned-face limit does not bound total patch size, especially at high-valence vertices. Reduce duplication, adjust patch sizes and improve access patterns within the 32-bit representation before considering a separate format proposal.

Target a 16 KiB workgroup budget only on devices exposing at least that amount; reduce the budget or use a fallback otherwise. Request 32 KiB or higher only when supported. Never assume a fixed subgroup width or cross-workgroup forward progress. These byte budgets do not imply support for alternative numeric widths.

# 4. Dynamic topology design

Everything in this section is planned. No topology edits, cavity claims, rollback, migration or slicing are implemented yet.

Cavity processing removes a connected region and reconstructs its boundary. Edge flips, splits and collapses can use a common transaction model. Preserve local speculative execution where practical, but do not port CUDA spin-wait synchronization literally.

| **Phase**     | **Operation**                                             |
|---------------|-----------------------------------------------------------|
| 1\. Discover  | Evaluate geometric and topology predicates.               |
| 2\. Construct | Identify cavity, boundary and full conflict footprint.    |
| 3\. Claim     | Use atomic priorities to select compatible candidates.    |
| 4\. Reserve   | Allocate slots and capacity before staging edits.         |
| 5\. Stage     | Build replacement connectivity and attributes.            |
| 6\. Commit    | Publish validated changes; discard failed edits.          |
| 7\. Repair    | Update ribbons, canonical ownership and dirty patches.    |
| 8\. Retry     | Reschedule conflicts; slice or migrate oversized patches. |

## Two execution paths

Interior cavities: one workgroup per patch, local barriers and scratch, speculative reconstruction and local commit. Boundary cavities: queue work, resolve conflicts across patches, reserve resources and commit in separate dispatches. Repair affected ownership and ribbon state before subsequent queries.

A candidate can claim each conflicting resource using atomicMin with a unique priority, then validate in a subsequent dispatch. Claims must cover all conflicting reads/writes, ownership transitions and attribute changes. A winning subset need not be maximal; retry with bounded rounds and explicit failure reporting.

WGSL atomics are relaxed, and execution barriers do not synchronize all workgroups. Avoid global spinlocks and assumptions about simultaneous residency. Use double-buffered work queues and indirect dispatch arguments; no CPU readback is required between GPU phases. \[7\]

## Primary implementation risk

Cross-patch transactions, migration and patch slicing are the hardest components. Separate dispatches and smaller patches may reduce the speedup from locality. Establish a same-adapter unpartitioned GPU baseline before locality optimization; add original CUDA RXMesh measurements when compatible hardware is available. Missing CUDA execution need not block the first static experiment.

# 5. Implementation roadmap

See [ROADMAP.md](ROADMAP.md) for the maintained phase checklist, acceptance gates and implementation discoveries. It preserves phases 0–9 while distinguishing completed prototype work from unproven performance and release criteria.

Current position: the reference setup is mostly complete; infrastructure and static query coverage are in place; static algorithms have begun; dynamics and measured optimization have not. Downstream examples are core acceptance work, and both boundary and neighborhood compositions now run against the public package.

The static API review is complete; its deferred findings are tracked in the [roadmap](ROADMAP.md#static-api-review-follow-ups). The next sequence is **benchmark baseline → workgroup-local traversal experiment → remaining static attributes/algorithms → interior edge flips**. Baseline profiling has been moved ahead of dynamics rather than deferred to the late optimization phase. Broader cross-vendor tuning still belongs to later phases.

## Current project layout

```text
src/                       CPU construction, handles, patches, public GPU API
src/gpu/                   Attributes, adjacency, edge queries, scan orchestration, shared pipeline helpers
docs/                      Review findings and contract decisions
shaders/                   Public core, topology, geometry and parallel WESL modules
runtime_shaders/           Concrete bindings and dispatch entry points
examples/                  Runnable downstream applications
examples/shaders/           Downstream WESL with application-owned bindings
examples/support/           Example orchestration shared with GPU tests
tests/                     Host/GPU reference tests and downstream linking
prior_art/REFERENCES.md     Recorded upstream revisions and setup instructions
```

Dedicated scheduler/dynamic modules, benchmark infrastructure and editable storage remain planned. Public operations must remain reusable independently of runtime bindings as those modules arrive.

## Current host API and future direction

Construction is CPU-only and GPU upload is a separate step. The caller supplies the device/queue and owns submission. Cache query objects to reuse their pipelines; results remain on the GPU unless explicitly copied for readback.

```rust,ignore
use mopmesh::{MeshBuilder, gpu::{AdjacencyQuery, GpuMesh}};

let mesh = MeshBuilder::new()
    .positions(&positions)
    .triangles(&triangles)
    .patch_size(128)
    .build()?;
let gpu_mesh = GpuMesh::upload(&device, &mesh)?;
let query = AdjacencyQuery::new(&device);
let mut encoder = device.create_command_encoder(&Default::default());
let adjacency = query.encode(&device, &mut encoder, &gpu_mesh)?;
queue.submit([encoder.finish()]);
```

There are no `mesh.compute_normals()` or `mesh.remesh()` methods yet. Their eventual interfaces should follow the same explicit ownership/submission model rather than treating speculative examples as a frozen API.

Current runtime dependencies are `wgpu`, `wesl-core` and `bytemuck`; WESL compilation uses `wesl` at build time. Development tools include Alum, Naga, pollster and wesl. Exact resolved versions are in `Cargo.lock`; `glam` and `rayon` have not been added and should be justified by implementation needs.

# 6. Validation and performance

## Correctness

Use Alum as a best-effort Rust CPU oracle and development dependency. We accept differences from Alum, and never choose a slower or more complex design just to match it. Where an operation differs from Alum's, such as splits that keep IDs, in-place collapses or cross-patch scheduling, tests use project CPU models (for example the migration simulation's `Sim`) and invariant checks instead. Alum decides only between otherwise equal choices. Retain OpenMesh as additional prior art and an optional C++ reference. Compare canonical topology and attributes rather than relying on identical patch IDs or operation ordering. Define supported topology classes explicitly; start with oriented manifold triangle meshes and tested boundary cases.

- Every active edge references valid vertices; every face forms a closed directed triangle.

- Each logical element has one owner; ribbons resolve to active canonical elements.

- No capacity overflow, stale-handle aliasing or lost attributes during migration.

- Failed transactions preserve committed topology; concurrent edits have compatible footprints.

- Retry queues terminate or return explicit failure.

Implemented fixtures include triangle/quadrilateral host cases, a tetrahedron, grids with and without a hole, disconnected components, high-valence vertices, invalid-input rejection and empty meshes. GPU tests cover multiple patch sizes and repeated execution. Sphere/large real-world benchmark datasets, nearly full editable patches and conflicting boundary edits remain planned. The dynamic invariants above become acceptance tests when editing is introduced; they are not currently exercised.

## Four feasibility experiments

| **Experiment**          | **What it tests**                             |
|-------------------------|-----------------------------------------------|
| Static local queries    | Memory footprint and locality benefit.        |
| Parallel interior flips | Workgroup cavity logic and conflict handling. |
| Cross-patch flips       | Portable coordination and ownership repair.   |
| Repeated edge splits    | Capacity pressure, allocation and growth.     |

## Benchmark design

Compare CPU oracle, simple unpartitioned WESL mesh, patch-based MOPMesh and CUDA RXMesh. Match hardware, input, geometric criteria and termination when comparing CUDA. Test NVIDIA, AMD, Intel and Apple adapters where available.

Measure queries/s, successful edits/s, bytes/triangle, ribbon overhead, rejected cavities, scratch usage, dispatch overhead, maintenance cost and total remeshing time. Record adapter limits and kernel specialization. Functional parity is a goal; performance parity is not yet established.

# 7. Related work and prior art

| **Project**                          | **Relevance to MOPMesh**                                                                 |
|--------------------------------------|------------------------------------------------------------------------------------------|
| Parallel Halfedge Subdivision \[10\] | OpenGL compute; parallel topology reconstruction for Loop and Catmull–Clark subdivision. |
| Concurrent Binary Trees \[11\]       | Compact concurrent allocation and hierarchical refinement; not general topology editing. |
| Papageorgiou & Platis \[12\]         | OpenCL parallel edge-collapse simplification; evidence of non-CUDA dynamic editing.      |
| Zayer et al. \[13\]                  | Sparse-matrix connectivity for unstructured grids; direct conceptual predecessor.        |
| Hornet \[14\]                        | CUDA dynamic graph adjacency, incremental updates and memory allocation.                 |
| faimGraph \[15\]                     | CUDA autonomous allocation, recycling and dynamic connectivity.                          |
| PRAgMaTIc \[16\]                     | CPU OpenMP/MPI mesh adaptation; parallel conflict avoidance and topology reference.      |

Alum (https://github.com/ranjeethmahankali/alum), a Rust halfedge mesh library inspired by OpenMesh, is the selected best-effort CPU test reference and additional prior art. OpenMesh (https://www.graphics.rwth-aachen.de/software/openmesh/) is downloaded as a secondary reference. Neither is intended to replace MOPMesh's GPU patch/ribbon representation. Other potential references include tri-mesh, plexus and truck; meshopt is useful for simplification comparisons and parry for geometric queries. Their current APIs, licenses and suitability require separate review.

## Release scope

The initial-release GPU numeric contract is native 32-bit only, as specified under Storage layout. Numeric packing and 16/64-bit or paired-float paths are excluded from the committed roadmap.

Target MOPMesh 0.1 scope: GPU-resident static mesh, patch/ribbon representation, queries, typed attributes, geometry operations and experimental local cavities. The prototype's Cargo version is not a declaration that this scope is complete. Later releases target cross-patch topology, slicing, remeshing, decimation and reusable cavity operators. Numerical solvers and automatic differentiation remain out of scope until justified.

## Feasibility conclusion

Proceed with a bounded feasibility prototype using native 32-bit GPU representations. Address memory constraints through patch sizing, controlled ribbon duplication, selective storage access and resource reuse; synchronization requires a deliberate redesign. Prior art supports the feasibility of non-CUDA topology modification, but does not prove that RXMesh-level generality or performance can be attained in WebGPU. The first investment gate is measured static locality benefit and correct concurrent edge flips.

# 8. References and source map

**\[1\] RXMesh repository —** https://github.com/owensgroup/RXMesh

**\[2\] RXMesh: A GPU Mesh Data Structure (2021) —** https://doi.org/10.1145/3450626.3459748

**\[3\] Dynamic Mesh Processing on the GPU (2025) —** https://doi.org/10.1145/3731162

**\[4\] RXMesh license —** https://github.com/owensgroup/RXMesh/blob/main/LICENSE

**\[5\] SCALE documentation —** https://docs.scale-lang.com/

**\[6\] ZLUDA —** https://github.com/vosen/ZLUDA

**\[7\] WGSL specification —** https://www.w3.org/TR/WGSL/

**\[8\] wgpu Limits —** https://docs.rs/wgpu/latest/wgpu/struct.Limits.html

**\[9\] WESL documentation —** https://wesl-lang.dev/

**\[10\] Parallel Halfedge Subdivision —** https://github.com/kvanhoey/ParallelHalfedgeSubdivision

**\[11\] Concurrent Binary Trees —** https://github.com/jdupuy/libcbt

**\[12\] GPU-based triangle mesh simplification —** https://users.uop.gr/~nplatis/files/PapageorgiouPlatisGPU.pdf

**\[13\] A GPU-Adapted Structure for Unstructured Grids —** https://doi.org/10.1111/cgf.13144

**\[14\] Hornet —** https://github.com/hornet-gt/hornet

**\[15\] faimGraph —** https://hpdgp.mpi-inf.mpg.de/pappe/faimGraph.pdf

**\[16\] PRAgMaTIc —** https://github.com/meshadaptation/pragmatic

## Upstream source files to inspect at pinned revision

include/rxmesh/patch_info.h, handle.h, rxmesh_static.h, rxmesh_dynamic.h, cavity_manager.cuh, patch_scheduler.h; apps/ and tests/. Documentation: https://ahdhn.github.io/RXMeshDocs/.
