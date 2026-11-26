Portable GPU mesh processing \| Technical design and feasibility assessment

Rust / WESL / WebGPU • 26 November 2026

## Executive summary

MOPMesh (Mahmoud–Owens–Porumbescu Mesh) is a proposed independent Rust/WESL port of RXMesh, the Ribbon-matriX Mesh architecture introduced in 2021 and extended to dynamic topology in 2025. Its goal is GPU-resident triangle mesh processing on multiple GPU vendors, without a CUDA dependency.

The project is technically plausible, but performance parity with CUDA is unproven. Preserve patch locality, ribbon ownership, connectivity queries and cavity-based editing; redesign cross-workgroup coordination around explicit dispatch phases. Establish correctness and measure locality benefits before committing to full dynamic parity.

## Recommended initial scope

- Static partitioned mesh, EV/FE connectivity, ribbon ownership and typed attributes.

- Portable workgroup primitives and Alum as the CPU reference oracle; do not build a separate CPU reference mesh implementation.

- Reusable WESL library modules that downstream WESL projects can import and link independently of MOPMesh host execution.

- Local edge flips, then cross-patch edits, allocation, migration and slicing.

- Defer numerical solvers, automatic differentiation and full CUDA API parity.

## Decision gates

| **Gate**             | **Evidence required**                                                                     |
|----------------------|-------------------------------------------------------------------------------------------|
| Static viability     | Correct queries; measured advantage over a global-memory baseline.                        |
| Dynamic viability    | Conflict-free local and boundary edits; bounded retries; no topology corruption.          |
| Production viability | Cross-vendor performance, reliable capacity handling and documented topology constraints. |

# 1. Upstream and licensing

RXMesh is maintained in owensgroup/RXMesh. The 2021 paper introduces the patch/ribbon structure; Dynamic Mesh Processing on the GPU (2025), by Ahmed H. Mahmoud, Serban D. Porumbescu and John D. Owens, introduces dynamic cavity operations. \[1–3\]

## BSD-2-Clause

The upstream repository uses BSD-2-Clause. Its copyright notice is “Copyright (c) 2022, owensgroup.” The license permits modification, translation, redistribution, commercial and proprietary use, subject to preservation of the original notice, conditions and disclaimer. Binary redistributions must reproduce these in accompanying materials. \[4\]

Recommend retaining BSD-2-Clause for MOPMesh, adding a copyright notice for new contributions, retaining upstream notices and documenting both papers in the README. Audit third-party dependencies and assets separately. BSD-2-Clause does not expressly grant patent rights; avoid implying author endorsement through the project name.

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

Rust owns device integration, mesh construction, resource lifetime, pipeline caching and command submission. WESL owns connectivity traversal, parallel geometry algorithms, topology transactions and patch maintenance. Keep intermediate data GPU-resident; expose buffers for downstream rendering and compute.

## Reusable WESL library requirement

The shader code is a public library as well as the implementation behind the Rust API. Other WESL projects must be able to import its types, connectivity helpers, parallel primitives and operators without adopting MOPMesh compute entry points or host scheduling. Ship a `wesl.toml` and a Cargo WESL package, with explicit public symbols and documented module paths.

Keep reusable modules independent of application-specific binding numbers and entry points. Put concrete storage bindings, workgroup sizing and dispatch wrappers in separate integration modules. Document storage layouts, capacity requirements and synchronization preconditions as those APIs are introduced. Library linking must be covered by a downstream consumer test with its own bindings and entry point; syntax-only compilation of internal shaders is insufficient.

Common operations belong in the public WESL library as they are implemented: connectivity traversal, geometry primitives, scans and reductions, and operators such as normals, areas, edge lengths and smoothing. The Rust runtime must use these same implementations. Keep allocation, concrete bindings, pipeline configuration and dispatch scheduling in the runtime integration layer. Document any workgroup-size or collective-call requirements of public primitives.

Downstream example applications are part of the core implementation plan and API acceptance criteria, not a stretch goal. Add a small runnable example alongside each major operator family. Each example should import the packaged library, own its bindings and entry points, and compose public operations with application-specific data or behavior. Generally useful operations discovered while building an example should become library functions rather than duplicated example code. Use concrete consumers to guide abstractions instead of designing speculative interfaces.

Accept an operator family when its downstream example links using only the public package, runs with documented layouts and synchronization, and produces validated results. Use Alum as the independent mesh reference where applicable. If an example needs private runtime internals or duplicates a common operation, address the missing public API before considering the family complete. Keep downstream linking checks in the hardware-independent suite and execution checks in the explicit GPU suite.

| **Layer**     | **Responsibility**                                                      |
|---------------|-------------------------------------------------------------------------|
| Rust host     | Builder, partitioner, handles, typed attributes, scheduler, validation. |
| WESL core     | Packed types, buffer offsets, bitmasks, arenas, ownership resolution.   |
| WESL parallel | Scan, reduce, histogram, compaction and optional subgroup variants.     |
| WESL topology | Patch/ribbon queries, cavity construction, claims, commit and repair.   |
| WESL operators | Reusable geometry operations, smoothing and, in later phases, remeshing and decimation operators. |
| Applications  | Compose public operators with application-specific data, policies and behavior. |

## Patch and ribbon model

Each patch owns mesh elements and stores replicated neighboring connectivity (ribbons), allowing local queries without accessing unrelated patches. Preserve EV (edge-to-two-vertices) and FE (face-to-three-directed-edges) as primary connectivity. Derive FV, VE, EF, VV, VF, FF and edge-diamond queries by local composition and transposition.

Use explicit patch/local handles initially; document invalidation after migration or slot reuse. Add generation counters or a separate stable-ID mapping where stale-handle detection is required. Ownership maps resolve ribbon elements to canonical owners.

## Storage layout

Replace CUDA pointers with offsets into storage arenas. Patch metadata holds offsets, counts, capacities, flags and ownership information. Use separate logical regions for connectivity, active/owned masks, attributes, ownership, transactions, scheduling and maintenance. Fixed-capacity patch allocations simplify the first implementation; variable-sized arenas can follow profiling.

\#\[repr(C)\]  
\#\[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)\]  
pub struct VertexHandle {  
pub patch: u32,  
pub local: u32,  
}

# 3. CUDA-to-WebGPU adaptation

| **CUDA mechanism**         | **Portable replacement**                          |
|----------------------------|---------------------------------------------------|
| Thread blocks / threads    | Workgroups / invocations                          |
| Dynamic shared memory      | Pipeline-specialized workgroup arrays             |
| Warp primitives / CUB      | Portable scans and reductions; optional subgroups |
| Pointers / 16-bit indices  | Offsets; packed u32 words or unpacked u32         |
| 64-bit atomics             | 32-bit claims or redesigned multiword state       |
| Cross-block locks          | Nonblocking claims and separate dispatch phases   |
| Dynamic allocation         | Preallocated arenas, free lists, compaction       |
| Device lambdas / templates | WESL imports and specialization; Rust generics    |

## Memory budget

Standard wgpu defaults include 16 KiB workgroup storage, 256 invocations per workgroup, 128 MiB maximum storage-buffer binding size and eight storage buffers per shader stage. Query actual adapter limits and request supported enhanced profiles. \[7–9\]

| **Owned faces** | **Approx. connectivity + masks** |
|-----------------|----------------------------------|
| 64              | 1.5 KiB                          |
| 128             | 3.0 KiB                          |
| 256             | 6.0 KiB                          |
| 512             | 12.0 KiB                         |
| 1024            | 24.0 KiB                         |

Estimate assumes 30% ribbon overhead, 1.5× capacity, E ≈ 1.5F, V ≈ 0.5F, 16-bit connectivity and two masks per element. It excludes ownership maps, scratch, cavities, attributes and alignment. Begin with 128–256 owned faces and 64–128 invocations; measure actual memory consumption. Smaller patches increase ribbon duplication and boundary work.

Use 16 KiB as a portable profile, 32 KiB or higher only when supported. Packed writes require atomic updates or exclusive word ownership. Never assume a fixed subgroup width or cross-workgroup forward progress.

# 4. Dynamic topology design

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

Cross-patch transactions, migration and patch slicing are the hardest components. Separate dispatches and smaller patches may reduce the speedup from locality. Benchmark both an unpartitioned GPU baseline and original CUDA RXMesh before optimizing.

# 5. Implementation roadmap

| **Phase**             | **Deliverable and acceptance criterion**                        |
|-----------------------|-----------------------------------------------------------------|
| 0 — Reference         | Pin upstream commit; record license and benchmark fixtures.     |
| 1 — Infrastructure    | Rust/wgpu integration, WESL compilation, scans, CPU oracle and downstream package linking. |
| 2 — Static topology   | Patch partitioning, EV/FE, ribbons, ownership and query parity; downstream query-composition example. |
| 3 — Static algorithms | Public normals, areas, lengths, smoothing and reductions; runnable downstream examples with validated results. |
| 4 — Local dynamics    | Single and parallel interior edge flips; rollback tests.        |
| 5 — Boundary dynamics | Cross-patch claims, migration and ribbon repair.                |
| 6 — Growth            | Splits, collapses, free slots, overflow and patch slicing.      |
| 7 — Algorithms        | Delaunay flipping, isotropic remeshing, decimation.             |
| 8 — Optimization      | Packing, subgroups, dispatch fusion, adapter profiling.         |
| 9 — Release           | Documentation, benchmarks, examples and crates.io.              |

Introduce downstream examples with each operator family, including dynamic operators as those phases arrive. Phase 9 polishes and packages those examples; it does not defer their creation until release. The initial runnable example, `examples/weighted_boundary.rs`, consumes GPU edge connectivity and imports public WESL boundary and edge-length helpers to apply application-provided weights. Its shader has independent bindings and its results are checked against Alum by the GPU suite. Downstream linking and WGSL validation also run without hardware. Other operator families still require their own examples and result checks as they are implemented.

`examples/neighborhood.rs` extends this coverage to paired vertex-edge/vertex-neighbor CSR traversal, composing public CSR, boundary and length helpers with application weights. The runtime now also exposes face neighbors in input edge-slot order and edge diamonds with explicit boundary sentinels. These static query families are checked against Alum across patch sizes; workgroup-local execution and measured locality benefits remain outstanding acceptance work. Other operator families still need their corresponding downstream examples.

## Proposed project layout

src/ gpu, mesh, builder, partition, handle, patch,  
attribute, connectivity, scheduler, dynamic, validation  
shaders/ core/, parallel/, topology/, geometry/, dynamic/, algorithms/  
runtime_shaders/ concrete bindings and dispatch entry points  
examples/ tests/ benches/ docs/  
LICENSE THIRD_PARTY_NOTICES Cargo.toml build.rs

## Host API direction

Accept an existing wgpu Device and Queue. Record operations into caller-owned command encoders, keep readbacks explicit and cache pipelines by specialization. Expose reusable WESL modules for application-specific operators. Use Rust, wgpu, wesl-rs, bytemuck, glam and rayon as candidate dependencies; pin versions after compatibility tests.

let mesh = MeshBuilder::new()  
.positions(&positions).triangles(&triangles)  
.patch_size(128).build(&device, &queue)?;  
mesh.compute_normals(&mut encoder)?;  
mesh.remesh(&mut encoder, &settings)?;

# 6. Validation and performance

## Correctness

Use Alum as the independent Rust CPU oracle and development dependency, rather than implementing a separate CPU reference mesh. Retain OpenMesh as additional prior art and an optional C++ reference. Compare canonical topology and attributes rather than relying on identical patch IDs or operation ordering. Define supported topology classes explicitly; start with oriented manifold triangle meshes and tested boundary cases.

- Every active edge references valid vertices; every face forms a closed directed triangle.

- Each logical element has one owner; ribbons resolve to active canonical elements.

- No capacity overflow, stale-handle aliasing or lost attributes during migration.

- Failed transactions preserve committed topology; concurrent edits have compatible footprints.

- Retry queues terminate or return explicit failure.

Fixtures: single triangle, quadrilateral, tetrahedron, sphere, holes, high-valence vertices, degenerate and non-manifold inputs, nearly full patches and deliberately conflicting boundary edits.

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

Alum (https://github.com/ranjeethmahankali/alum), a Rust halfedge mesh library inspired by OpenMesh, is the selected CPU test reference and additional prior art. OpenMesh (https://www.graphics.rwth-aachen.de/software/openmesh/) is downloaded as a secondary reference. Neither is intended to replace MOPMesh's GPU patch/ribbon representation. Other potential references include tri-mesh, plexus and truck; meshopt is useful for simplification comparisons and parry for geometric queries. Their current APIs, licenses and suitability require separate review.

## Release scope

MOPMesh 0.1: GPU-resident static mesh, patch/ribbon representation, queries, typed attributes, geometry operations and experimental local cavities. Later releases: cross-patch topology, slicing, remeshing, decimation and reusable cavity operators. Numerical solvers and automatic differentiation remain out of scope until justified.

## Feasibility conclusion

Proceed with a bounded feasibility prototype. Memory constraints can be addressed by patch sizing, packing and selective storage access; synchronization requires a deliberate redesign. Prior art supports the feasibility of non-CUDA topology modification, but does not prove that RXMesh-level generality or performance can be attained in WebGPU. The first investment gate is measured static locality benefit and correct concurrent edge flips.

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
