# MOPMesh

Portable patch-based triangle mesh processing in Rust and WESL, following the
RXMesh patch/ribbon architecture. This is an early feasibility prototype with
static GPU connectivity queries.

## Implemented

- Validated host construction for oriented manifold triangle meshes, including boundaries.
- Deterministic breadth-first face partitioning, local EV/FE connectivity, vertex-star ribbons, and explicit canonical ownership.
- Patch/local vertex, edge, and face handles; typed canonical vertex attributes.
- Alum-backed reference tests for connectivity and ownership across multiple patch sizes.
- A reusable WESL package with initial handle types and directed-edge helpers, tested through a downstream consumer shader.
- A scalar connectivity arena, wgpu upload, and a GPU face-to-vertex query verified against Alum.
- GPU vertex-to-face CSR adjacency derived by count, prefix, and scatter passes, verified against Alum.
- GPU edge endpoints and incident faces, plus a downstream weighted-boundary operator verified against Alum.
- GPU vertex-edge/vertex-neighbor, face-neighbor and edge-diamond adjacency, plus a downstream neighborhood operator verified against Alum.
- A hierarchical GPU prefix scan and an importable 64-lane WESL scan collective.
- Typed canonical GPU vertex attributes with explicit updates, and GPU face centroids verified against Alum.

Alum is the CPU reference, not a separately implemented MOPMesh CPU oracle.
OpenMesh is retained as an optional secondary reference and prior art.

## Run

```sh
cargo run
cargo run --example weighted_boundary
cargo run --example neighborhood
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
# Requires a real GPU adapter; fails if no adapter is available.
cargo test --test gpu_topology -- --ignored --nocapture
# Include both GPU topology and scan tests:
cargo test --all-targets -- --include-ignored --nocapture
```

The example builds a two-triangle square with one owned face per patch.

```rust
use mopmesh::MeshBuilder;

let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
let triangles = [[0, 1, 2]];
let mesh = MeshBuilder::new()
    .positions(&positions)
    .triangles(&triangles)
    .patch_size(128)
    .build()?;
# Ok::<(), mopmesh::BuildError>(())
```

## Reusing the WESL library

The shader package is a first-class public API. Its sources live under
`shaders/`, with package configuration in `wesl.toml`. The Rust crate exports
`mopmesh::mopmesh::PACKAGE`, which downstream WESL linkers can register:

```rust,ignore
let mut resolver = wesl::resolver::StandardResolver::new("my_shaders");
resolver.add_package(&mopmesh::mopmesh::PACKAGE);
let wgsl = wesl::Compiler::default()
    .with_resolver(resolver)
    .compile_module(&"package::main".parse().unwrap())?;
```

```wesl
import mopmesh::core::handle::{VertexHandle, same_vertex};
import mopmesh::topology::directed_edge::{directed_start, directed_end};
import mopmesh::topology::face::face_vertices;
import mopmesh::topology::csr::{CsrRow, row_from_count, row_length};
import mopmesh::geometry::triangle::triangle_centroid;
import mopmesh::parallel::scan::exclusive_scan_64;
import mopmesh::topology::edge::{EdgeRecord, is_boundary_edge, edge_face_count};
import mopmesh::geometry::edge::edge_length;
```

These initial modules declare no resource bindings or compute entry points.
Consumers choose those themselves. See `tests/shaders/consumer.wesl` and
`tests/wesl_package.rs` for a complete linking example. Public module paths and
types are experimental. `face_vertices` accepts three EV records in face-edge
order and three orientation flags, returning local vertex IDs. Consumers load
records from their own buffers; no storage-pointer extensions are required.
The CSR helpers describe half-open row ranges independently of buffer bindings.
The geometry helper computes a triangle centroid from three position values.
`exclusive_scan_64(value, lane)` returns a u32 exclusive prefix and block total.
It is a collective: all 64 workgroup invocations must call it in uniform control
flow with their distinct `local_invocation_index`. Inactive lanes supply zero
instead of returning early. It uses 256 bytes of private workgroup scratch,
includes barriers, and supports sequential calls. Arithmetic wraps at u32.
The helper owns no resource bindings or entry points; a separate downstream
consumer demonstrates importing it and calling it twice.
Both the downstream shader and built-in query pass Naga validation with no
optional capabilities. Binding and dispatch wrappers live separately in
`runtime_shaders/`, outside the reusable package. These entry points implement
the Rust runtime's buffer bindings and dispatch layouts.

### Public operators and downstream examples

Common mesh operations will be implemented in the public WESL library, with the
Rust runtime using those same implementations. This includes connectivity
traversal, geometry primitives, scans/reductions, and planned operators such as
normals, areas, edge lengths and smoothing. Allocation, concrete bindings,
pipeline configuration and dispatch scheduling belong in the runtime layer.

A small runnable downstream example is an acceptance criterion for each major
operator family, not a stretch goal. Examples will import the packaged library,
choose their own bindings and entry points, and compose public operations with
application-specific data or behavior. If an example duplicates a common
operation or needs private runtime internals, we should improve the library API.
Actual consumer needs will guide the abstractions we introduce.

The first runnable example is `examples/weighted_boundary.rs`, with its own
shader in `examples/shaders/weighted_boundary.wesl`. It consumes edge connectivity
and canonical positions, then combines public boundary and edge-length helpers
with application-provided weights. The query and custom operator run in one
submission; only the final result is read back. On a two-triangle unit square it
reports five edges, four boundary edges, and weighted boundary length four when
all weights are one.

GPU tests execute the same example operator and compare connectivity, boundary
classification and weighted lengths against Alum across multiple fixtures and
patch sizes. The ordinary suite links and validates the example shader without
GPU hardware. Additional operator families will add their own runnable examples
and result checks as they are implemented.

`examples/neighborhood.rs` demonstrates VE/VV composition. Its shader imports
public CSR, boundary and length helpers, traverses each vertex's paired
edge/neighbor entries, and returns degree, incident boundary-edge count, mean
edge length, and an application-weighted mean. It uses five storage bindings and
chooses its own bindings, entry point and output policy. Its actual operator code
is exercised against Alum in the GPU tests; only the final results are read back.

## GPU usage

Use an existing wgpu device, queue, and command encoder:

```rust,ignore
use mopmesh::gpu::{FaceVertexQuery, GpuMesh};

let gpu_mesh = GpuMesh::upload(&device, &mesh)?;
let query = FaceVertexQuery::new(&device); // Cache once per device.
let mut encoder = device.create_command_encoder(&Default::default());
let result = query.encode(&device, &mut encoder, &gpu_mesh);
// Encode downstream compute or an explicit readback copy here.
queue.submit([encoder.finish()]);
```

The result is three u32 global vertex IDs per original input face, preserving
winding. It has STORAGE and COPY_SRC usage and stays on the GPU. Each owned
face is queried exactly once; ribbon copies do not duplicate output. Empty
meshes dispatch no work and return a four-byte placeholder. Resources must
belong to the same device. The query uses baseline compute limits: two storage
bindings and 64 invocations per workgroup.

`GpuMesh::arena()` exposes immutable scalar connectivity. All offsets are u32
word offsets, with no struct padding. The four-word header holds face count,
face-task offset, edge count, and edge-task offset. Patch regions contain global
vertex IDs, EV endpoint pairs, FE triples of `(local edge ID, reversed 0/1)`
pairs, and global face IDs. Each four-word face task stores EV offset, face
record offset, vertex offset, and global face ID. Each seven-word edge task stores
local edge ID, local face count, FE base, face-ID base, global edge ID,
vertex-ID base, and EV base.
This static ABI is experimental; it contains no edit capacities or active masks.
Upload checks device buffer, binding-size, and dispatch limits before allocation.

### Edge endpoints and incident faces

```rust,ignore
use mopmesh::gpu::EdgeQuery;

let query = EdgeQuery::new(&device);
let edges = query.encode(&device, &mut encoder, &gpu_mesh)?;
// Bind edges.buffer() as array<mopmesh::topology::edge::EdgeRecord>.
```

Each canonical edge has a 16-byte record: two global vertex IDs, followed by two
global face IDs. Endpoints are ascending; incident-face order is unspecified.
Boundary edges use `0xffffffff` for the missing face. Edge IDs are deterministic
constructed IDs, accessible through each patch's `edge_ids()` table; they need
not match another library's edge numbering. The result has STORAGE and COPY_SRC
usage, and an empty mesh returns a zeroed 16-byte placeholder with zero logical
records. Do not interpret that placeholder as an edge.

The current kernel scans the owner patch's local faces, including ribbons, once
per owned edge. This provides complete incident faces across partition borders
without duplicating output. It is a correctness baseline with work proportional
to owned edges times local face count per patch; workgroup staging and faster
traversal remain to be implemented. The query checks its own output and dispatch
limits before recording commands and uses two storage bindings.

### Remaining adjacency families

```rust,ignore
use mopmesh::gpu::AdjacencyQuery;

let query = AdjacencyQuery::new(&device); // Cache once per device.
let adjacency = query.encode(&device, &mut encoder, &gpu_mesh)?;
let ve_vv = adjacency.vertex_neighbors();
let ff = adjacency.face_neighbors();
let diamonds = adjacency.edge_diamonds();
```

`AdjacencyQuery` computes these four families together, sharing its edge and
face connectivity intermediates. It also exposes `edge_connectivity()` for
downstream composition. All result buffers have STORAGE and COPY_SRC usage.
The batch requires four storage bindings, one uniform binding, and 64 lanes.
All resources must belong to the same device. Buffer, scalar-index, scratch and
dispatch limits are checked before commands are recorded.

- **Vertex-edge and vertex-neighbor:** one CSR buffer with `V + 1` u32 offsets,
  followed by `2E` pairs `(canonical edge ID, neighbor vertex ID)`. Offsets count
  pairs, not scalar words. The pair region starts at
  `vertex_entries_byte_offset()`. Each edge contributes one pair to each
  endpoint. Row order is unspecified and is not a cyclic one-ring ordering.
- **Face-neighbor:** three tightly packed u32 face IDs per original input face.
  Slot `i` is across `(triangle[i], triangle[(i + 1) % 3])`, preserving input
  winding and corner order. `0xffffffff` denotes a boundary. Do not bind this
  12-byte-per-face layout as a WGSL array of vec3, whose stride would be 16 bytes.
- **Edge-diamond:** one 16-byte quartet per canonical edge:
  `[endpoint0, endpoint1, opposite0, opposite1]`. Endpoints are ascending and the
  opposite vertices correspond to the two incident-face slots in
  `edge_connectivity()`. A missing boundary-side opposite is `0xffffffff`.
  This quartet is not cyclic or oriented by side; consumers must use face
  winding when they need an oriented cavity boundary.

Empty meshes dispatch no work: VE/VV contains one zero offset, FF has a four-byte
zero placeholder, and edge/diamond buffers each have a 16-byte zero placeholder.
Their logical counts are zero; placeholders are not valid topology records.
VE/VV uses a global count/scan/scatter transpose. FF and diamonds compose EV/EF
with FV, with one writer per face-edge slot. No host adjacency or intermediate
readback is used. Workgroup-local staging and selective single-family execution
remain future optimizations.

### Vertex-to-face adjacency

```rust,ignore
use mopmesh::gpu::VertexFaceQuery;

let query = VertexFaceQuery::new(&device); // Cache once per device.
let mut encoder = device.create_command_encoder(&Default::default());
let adjacency = query.encode(&device, &mut encoder, &gpu_mesh)?;
// Bind adjacency.buffer() to a downstream compute shader or copy for readback.
queue.submit([encoder.finish()]);
```

The result stores `V + 1` u32 offsets followed by `3F` u32 global face IDs.
Offsets index the face-ID region, which begins at `entries_byte_offset()`.
Vertex `v` has entries in `offsets[v]..offsets[v + 1]`. Rows use original input
vertex IDs; entries use original input face IDs. Face order within a row is
unspecified, not cyclic. An empty mesh returns one zero offset and no entries.
The buffer has STORAGE and COPY_SRC usage; it requires no host readback.

The query evaluates FV from owned faces, counts incidences atomically, computes
prefix offsets, then scatters face IDs. Ribbon replicas do not duplicate entries.
Each encode allocates fresh scratch and output buffers, so a query can be reused
across meshes and submissions. Three storage bindings, one uniform binding, and
64 invocations per workgroup are required. Output limits are checked before any
commands are recorded. Prefixes use 64-lane workgroup scans, recursively scan
block totals, then propagate offsets downward through separate passes. Partial
blocks pad inactive lanes with zeros. Scratch and dispatch limits include every
scan level; no host readback or serial per-vertex prefix loop is needed. This
implementation has correctness coverage but has not been performance-tuned.
The transpose is global rather than staged per patch.

### Vertex attributes and face centroids

```rust,ignore
use mopmesh::gpu::FaceCentroidQuery;

let mut host_positions = mesh.position_attribute(); // [x, y, z, 0]
let mut positions = gpu_mesh.upload_vertex_attribute(&device, &host_positions)?;
let query = FaceCentroidQuery::new(&device);
let mut encoder = device.create_command_encoder(&Default::default());
let centroids = query.encode(&device, &mut encoder, &gpu_mesh, &positions)?;
queue.submit([encoder.finish()]);

// After editing host_positions through mesh vertex handles:
positions.write(&queue, &host_positions)?;
```

GPU attributes store one value per original input vertex. Resolve patch-local
vertices through the arena's global vertex-ID table; ribbon references share
the same canonical slot. Uploads, updates, and centroid queries reject attributes
from another mesh, even if the meshes have identical topology. Resources must
still belong to the same device. The buffer is exposed for downstream shaders
and has STORAGE, COPY_SRC, and COPY_DST usage.

Supported element types are `f32`, `u32`, `i32`, and arrays of two or four of
those scalars, matching WGSL scalar/vec2/vec4 storage-array strides. Three-element
Rust arrays are excluded because WGSL vec3 arrays require 16-byte stride.
An empty attribute has zero logical elements and one zero element of storage.

Host and GPU attributes are independent snapshots. Updates replace the full GPU
buffer on the next queue submission; submit earlier readers first, since queue
writes also affect commands already encoded but not submitted. Updating a
position attribute does not change `Mesh::positions()` or revalidate geometry.
GPU writes are not automatically copied back to host attributes.

The centroid result contains one `vec4<f32>` per original face ID, with centroid
xyz and zero w; input w lanes are ignored. Empty meshes return a zero vec4
placeholder without dispatch. The query uses three storage bindings and the
existing FV evaluation. Finite, reasonably scaled f32 positions are expected;
extreme coordinates remain subject to floating-point overflow and rounding.

GPU reference tests cover 128-face and 8,192-face grids, a grid with a hole, a closed tetrahedron, disconnected
triangles, a 129-face vertex fan, and empty meshes at patch sizes 1, 3, and 128,
including repeated queries, scalar/vector uploads, cross-mesh rejection, and
centroids before and after position updates through ribbon handles, edge
connectivity, VE/VV, FF slot order, boundary/interior diamonds, and both downstream
operators. Repeated adjacency execution is also checked. They passed
on Apple M1 Max / Metal; other backends
have not yet been exercised. The ordinary test suite runs shader validation
and scan allocation/dispatch-limit checks without requiring GPU hardware.
GPU scan tests cover empty, exact, and partial blocks, zeros, varying values,
wrapping arithmetic, and up to four hierarchy levels (262,145 values).

## Current constraints

`patch_size` limits owned faces only. Ribbons include the complete face stars
of vertices on owned faces, so high-valence vertices can cause substantial
duplication. Full one-ring queries are guaranteed for owned elements; resolve
ribbon elements to their owner before querying their full neighborhood.

Construction rejects non-finite coordinates, invalid indices, zero-area or
duplicate triangles, inconsistent winding, non-manifold edges or vertices,
and isolated vertices. Empty meshes and disconnected components are supported.
Exact zero area is tested in f64; near-degenerate triangles are not filtered.
Geometric self-intersections are not checked.

Handles are mesh-relative addresses and must not be used with another mesh.
Future migration and slot reuse invalidate them. Vertex attributes verify mesh
identity and share canonical values across ribbons. Edge/face attributes,
active masks, edit capacities, and dynamic edits remain
future work. The current GPU query walks EV/FE directly; it does not yet stage
patches in workgroup memory. There are no GPU performance claims yet.

## Reference sources

Pinned revisions, dependency roles, and OpenMesh build instructions are in
`prior_art/REFERENCES.md`. Original upstream license notices are retained in
`THIRD_PARTY_NOTICES`.

- [RXMesh: A GPU Mesh Data Structure (2021)](https://doi.org/10.1145/3450626.3459748)
- [Dynamic Mesh Processing on the GPU (2025)](https://doi.org/10.1145/3731162)
- [Alum](https://github.com/ranjeethmahankali/alum)
- [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/)

The planned static adjacency families are now implemented and checked against
Alum. Next milestones: static geometry operators (normals, areas and smoothing),
edge/face attributes, workgroup-local traversal, and locality benchmarks against
an unpartitioned baseline. Static performance viability remains unproven.
