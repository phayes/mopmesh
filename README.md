# MOPMesh

Portable patch-based triangle mesh processing in Rust and WESL, following the
RXMesh patch/ribbon architecture. This is an early feasibility prototype with
static GPU connectivity queries.

**Status:** the planned static query families are implemented and checked against
Alum. Static geometry algorithms are partial; topology editing has not started.
Execution has been verified on Metal, but locality benefits and cross-vendor
performance are unproven. See the [maintained roadmap](ROADMAP.md) for completed
work, acceptance criteria and next steps, and the
[technical design](MOPMesh_Technical_Design.md) for architecture and planned dynamics.

## Implemented

- Validated host construction for oriented manifold triangle meshes, including boundaries.
- Deterministic breadth-first face partitioning, local EV/FE connectivity, vertex-star ribbons, and explicit canonical ownership.
- Patch/local vertex, edge, and face handles; typed canonical vertex, edge and face attributes.
- Alum-backed reference tests for connectivity and ownership across multiple patch sizes.
- A reusable WESL package with handle types, topology/geometry helpers and a scan collective, tested through downstream consumers.
- A scalar connectivity arena, wgpu upload, and a GPU face-to-vertex query verified against Alum.
- GPU vertex-to-face CSR adjacency derived by count, prefix, and scatter passes, verified against Alum.
- GPU edge endpoints and incident faces, plus a downstream weighted-boundary operator verified against Alum.
- GPU vertex-edge/vertex-neighbor, face-neighbor and edge-diamond adjacency, plus a downstream neighborhood operator verified against Alum.
- A hierarchical GPU prefix scan and an importable 64-lane WESL scan collective.
- Typed canonical GPU vertex, edge and face attributes with explicit updates, and GPU face centroids verified against Alum.
- GPU face normals, face areas and area-weighted vertex normals, verified against Alum and an f64 reference.
- GPU boundary flags, whole-attribute reductions (sum, min, max, dot) and uniform Laplacian smoothing with optional fixed vertices, plus a downstream smoothing example.
- An editable GPU mesh with spare per-patch slots and masks, parallel edge flips inside a patch verified against Alum, and WESL flip APIs that let downstream kernels flip edges by their own rule, one at a time or in parallel batches (see `examples/delaunay_flips.rs`).
- GPU mesh construction (`GpuMeshBuilder`) from position and triangle buffers already on the GPU, with the same validation as the host builder and every query verified against Alum.

Alum is the CPU reference, not a separately implemented MOPMesh CPU oracle.
OpenMesh is retained as an optional secondary reference and prior art.

## Run

```sh
cargo run
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
# The following require a GPU adapter; missing hardware is a failure.
cargo run --example weighted_boundary
cargo run --example neighborhood
cargo run --example smoothing
cargo run --example delaunay_flips
# Rough timing against Alum; see docs/approximate_benchmark.md:
cargo run --release --example bench_alum
cargo test --test gpu_topology -- --ignored --nocapture
# Include both GPU topology and scan tests:
cargo test --all-targets -- --include-ignored --nocapture
```

`cargo run` is a CPU-only construction example: it builds a two-triangle square
with one owned face per patch. The named examples execute GPU operators.
The ordinary test suite skips the two explicit hardware tests; use
`--include-ignored` as shown above to run the full suite.

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
import mopmesh::topology::edge::{INVALID_FACE, INVALID_VERTEX};
import mopmesh::topology::face::INVALID_SLOT;
import mopmesh::geometry::edge::edge_length;
import mopmesh::topology::face::face_vertex_slot;
import mopmesh::build::exact::{is_finite_position, same_position, has_coincident_corners};
import mopmesh::build::edge_group::{EdgeGroup, empty_edge_group, edge_group_add};
import mopmesh::build::edge_group::{edge_group_misoriented, edge_group_error_corner};
import mopmesh::build::morton::{morton_cell, morton_code_30, morton_spread_10};
```

The `build` modules hold value-level pieces of mesh construction that
editing code can reuse: exact position checks on f32 bit patterns (load
positions with `bitcast<vec3<u32>>`), an `EdgeGroup` value that collects the
corners on one edge and reports the corner that makes it invalid, and Morton
codes for a 1024³ grid.

`INVALID_FACE`, `INVALID_VERTEX` and `INVALID_SLOT` all equal `0xffffffff`, the
same value as the Rust constant `mopmesh::gpu::INVALID_ID`. Helpers that look up
a missing face, endpoint or face slot return the matching constant.

These public modules take values and declare no resource bindings or compute
entry points, except the opt-in `topology::patch_arena` readers and
`topology::patch_edge_staging`, which bind the arena and masks in group 1. Consumers choose the rest themselves. See
`tests/shaders/consumer.wesl` and `tests/wesl_package.rs` for a complete
linking example. Public module paths and
types are experimental. `face_vertices` accepts three EV records in face-edge
order and three orientation flags, returning local vertex IDs. Consumers load
records from their own buffers; no storage-pointer extensions are required.
The CSR helpers describe half-open row ranges independently of buffer bindings.
The geometry helper computes a triangle centroid from three position values.
`exclusive_scan_64(value, lane)` returns a u32 exclusive prefix and block total.
It is a collective: all 64 workgroup invocations must call it in uniform control
flow with their distinct `local_invocation_index`. Inactive lanes supply zero
instead of returning early. It uses 256 bytes of module-owned workgroup scratch,
includes barriers, and supports sequential calls. Arithmetic wraps at u32.
The helper owns no resource bindings or entry points; a separate downstream
consumer demonstrates importing it and calling it twice.
The downstream shaders and built-in queries pass Naga validation with no
optional capabilities. Binding and dispatch wrappers live separately in
`runtime_shaders/`, outside the reusable package. These entry points implement
the Rust runtime's buffer bindings and dispatch layouts.

### Public operators and downstream examples

Common mesh operations will be implemented in the public WESL library, with the
Rust runtime using those same implementations. This includes connectivity
traversal, geometry primitives and parallel primitives. Centroids, edge length,
normals, areas, boundary flags, scans, reductions and smoothing exist.
Allocation, concrete bindings,
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
edge length, and an application-weighted mean. Its per-vertex weights are an
ordinary `GpuVertexAttribute<f32>`. It uses five storage bindings and chooses
its own bindings, entry point and output policy. Both example operators assert
that their inputs report the same `mesh_id()`. Its actual operator code is
exercised against Alum in the GPU tests; only the final results are read back.

`examples/smoothing.rs` composes the geometry operators. It smooths a bumpy
33 x 33 grid for 20 steps with the boundary fixed, using `BoundaryQuery`'s
vertex flags. Its own shader (`examples/shaders/bending.wesl`) imports the
public CSR helpers and `normalize_or_zero` and rates how far each vertex's
edges leave its tangent plane, from VV rows and `VertexNormalQuery` normals.
It wraps its output buffer with `GpuAttribute::from_buffer`, so the library's
`ReduceQuery` can total it, together with `FaceAreaQuery` areas. Everything
runs in one submission and only four totals are read back; it reports mean
bending falling from about 0.117 to 0.022 and total area from 1119 to 1035.
GPU tests check the bending operator against an f64 reference.

## GPU usage

Use an existing wgpu device, queue, and command encoder:

```rust,ignore
use mopmesh::gpu::{FaceVertexQuery, GpuMesh};

let gpu_mesh = GpuMesh::upload(&device, &mesh)?;
let query = FaceVertexQuery::new(&device); // Cache once per device.
let mut encoder = device.create_command_encoder(&Default::default());
let face_vertices = query.encode(&device, &mut encoder, &gpu_mesh)?;
// Bind face_vertices.buffer() downstream or copy it for readback.
queue.submit([encoder.finish()]);
```

The result is three u32 global vertex IDs per original input face, preserving
winding. It has STORAGE and COPY_SRC usage and stays on the GPU. Each owned
face is queried exactly once; ribbon copies do not duplicate output. Empty
meshes dispatch no work and return a four-byte placeholder; `face_count()` gives
the logical record count. The query uses baseline compute limits: two storage
bindings and 64 invocations per workgroup.

### Building on the GPU

When positions and triangles are already on the GPU, build the mesh there
instead of with `MeshBuilder`:

```rust,ignore
use mopmesh::gpu::{GpuMeshBuilder, GpuMeshInput};

let builder = GpuMeshBuilder::new(&device).patch_size(128); // Cache once per device.
let input = GpuMeshInput {
    positions: &positions,  // STORAGE buffer of [f32; 4], fourth lane ignored
    triangles: &triangles,  // STORAGE buffer of [u32; 4], fourth lane ignored
    vertex_count,
    face_count,
};
let gpu_mesh = builder.build(&device, &queue, &input)?;
let positions = GpuVertexAttribute::<[f32; 4]>::from_buffer(&gpu_mesh, &positions)?;
```

`build` submits its own work. It blocks until the input has been validated,
then submits the rest without waiting. It rejects exactly what `MeshBuilder`
rejects, returning `GpuBuildError::Input` with the same `BuildError`.
`GpuBuildError::Gpu` covers undersized buffers, missing `STORAGE` usage and
device limits, and `GpuBuildError::Device` covers a failed wait. Each buffer
must hold at least 16 bytes, even for an empty mesh.

The result is an ordinary `GpuMesh` with its own `MeshId` and no CPU `Mesh`.
Global vertex, face and edge IDs match a CPU build of the same input. Patches
come from a Morton-order split, so ownership and order within patches differ
from `GpuMesh::upload`. Create its attributes on the GPU:

- `GpuAttribute::from_buffer(&gpu_mesh, &buffer)` wraps an existing buffer
  without copying it.
- `GpuAttribute::new(&device, &gpu_mesh, &values)` uploads one value per
  vertex, edge or face.

Replace an attribute's values with `GpuAttribute::write_values`; `write` takes
a host attribute, which only CPU meshes have.

### Shared contracts

- **Errors.** Uploads and queries return `GpuError`. `LimitsExceeded { resource }`
  means a buffer, u32 index range or dispatch would exceed u32 addressing or the
  device limits; `resource` names it. `MeshMismatch` means typed inputs came from
  different meshes. `InvalidBuffer { resource }` means a buffer or slice you
  passed in is too small, has the wrong length, or lacks a required usage.
  On error nothing has been allocated or recorded. The GPU build wraps these
  in `GpuBuildError`.
- **Mesh identity.** `Mesh::id()` returns a `MeshId`. Host and GPU attributes,
  `GpuMesh` and every query result report it through `mesh_id()`. Separately
  built meshes never share an ID, even with identical input. Built-in queries
  check typed attribute inputs; downstream operators can compare `mesh_id()`
  values themselves, as both examples do.
- **Device identity.** Every resource passed to one call must come from the same
  `wgpu::Device`. This is not checked: a mismatch is reported by wgpu validation
  rather than as a `GpuError`.
- **Results.** Every query returns a typed result exposing `buffer()` or named
  buffers, logical counts and `mesh_id()`. Empty results hold zeroed placeholders
  so the buffer is always bindable; use the counts, never the buffer size.

`GpuMesh::arena()` and `GpuMesh::masks()` expose immutable scalar
connectivity in the same patch layout as `GpuEditableMesh`, with STORAGE and
COPY_SRC usage. All offsets are u32 word offsets, with no struct padding. The
eight-word header holds the vertex, edge, face and patch counts, the offset of
the patch records, and three owner-table offsets, which are `INVALID_ID` on a
static mesh. Each 16-word patch record gives the patch's slot counts and the
offsets of its global vertex IDs, EV endpoint pairs, FE triples of
`(local edge slot, reversed 0/1)` pairs, global edge IDs and global face IDs,
and of its used-slot counts and active and owned masks in the mask buffer. A
static mesh has no spare slots: every slot is active, and it has no interior
masks. Read the layout from WESL with `topology::patch` (value-based) or
`topology::patch_arena` (read-only bindings in group 1). To find the faces
around each edge of a patch in workgroup memory, use `topology::patch_staging`
with your own bindings, or `topology::patch_edge_staging` on top of
`topology::patch_arena`; `tests/shaders/patch_arena_consumer.wesl` shows the
second.
This static ABI is experimental. Upload checks the arena and masks against u32
addressing and the device buffer and binding-size limits before allocation.
Each query checks its own outputs and dispatches.

### Edge endpoints and incident faces

```rust,ignore
use mopmesh::gpu::EdgeQuery;

let query = EdgeQuery::new(&device);
let edges = query.encode(&device, &mut encoder, &gpu_mesh)?;
// Bind edges.buffer() as array<mopmesh::topology::edge::EdgeRecord>.
```

Each canonical edge has a 16-byte record: two global vertex IDs, followed by two
global face IDs. Endpoints are ascending; incident-face order is unspecified.
Boundary edges use `INVALID_ID` (`0xffffffff`) for the missing face. Edge IDs run from 0 in
ascending order of each edge's (smaller, larger) vertex ID pair, so they do not
depend on input face order. Each patch's `edge_ids()` table maps its local edges
to these IDs. They need not match another library's edge numbering. The result has STORAGE and COPY_SRC
usage, and an empty mesh returns a zeroed 16-byte placeholder with zero logical
records. Do not interpret that placeholder as an edge.

The current kernel scans the owner patch's local faces, including ribbons, once
per owned edge. This provides complete incident faces across partition borders
without duplicating output. It is a correctness baseline with work proportional
to owned edges times local face count per patch; workgroup staging and faster
traversal remain to be implemented. The query checks its own output and dispatch
limits before recording commands and uses two storage bindings.

### Vertex neighborhoods, face neighbors and edge diamonds

```rust,ignore
use mopmesh::gpu::AdjacencyQuery;

let query = AdjacencyQuery::new(&device); // Cache once per device.
let adjacency = query.encode(&device, &mut encoder, &gpu_mesh)?;
let ve_vv = adjacency.vertex_neighbors();
let ff = adjacency.face_neighbors();
let diamonds = adjacency.edge_diamonds();
```

`AdjacencyQuery` computes these four families together. It also exposes
`edge_connectivity()` for downstream composition. All result buffers have
STORAGE and COPY_SRC usage. The query needs up to five storage bindings per
shader stage, one uniform binding, and 64 lanes.
Buffer, scalar-index, scratch and dispatch limits are checked before commands
are recorded.

- **Vertex-edge and vertex-neighbor:** one CSR buffer with `V + 1` u32 offsets,
  followed by `2E` pairs `(canonical edge ID, neighbor vertex ID)`. Offsets count
  pairs, not scalar words. The pair region starts at
  `vertex_entries_byte_offset()`. Each edge contributes one pair to each
  endpoint. Row order is unspecified and is not a cyclic one-ring ordering.
- **Face-neighbor:** three contiguous u32 face IDs per original input face.
  Slot `i` is across `(triangle[i], triangle[(i + 1) % 3])`, preserving input
  winding and corner order. `INVALID_ID` denotes a boundary. Do not bind this
  12-byte-per-face layout as a WGSL array of vec3, whose stride would be 16 bytes.
- **Edge-diamond:** one 16-byte quartet per canonical edge:
  `[endpoint0, endpoint1, opposite0, opposite1]`. Endpoints are ascending and the
  opposite vertices correspond to the two incident-face slots in
  `edge_connectivity()`. A missing boundary-side opposite is `INVALID_ID`.
  This quartet is not cyclic or oriented by side; consumers must use face
  winding when they need an oriented cavity boundary.

Empty meshes dispatch no work: VE/VV contains one zero offset, FF has a four-byte
zero placeholder, and edge/diamond buffers each have a 16-byte zero placeholder.
Their logical counts are zero; placeholders are not valid topology records.
Edge records, FF and diamonds come from one pass with one workgroup per
patch, which finds each edge's faces in workgroup memory. VE/VV uses a global
count/scan/scatter transpose of the edge records. No host adjacency or
intermediate readback is used. Selective single-family execution remains a
future optimization.

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

### Attributes and geometry queries

```rust,ignore
use mopmesh::gpu::FaceCentroidQuery;

let mut host_positions = mesh.position_attribute(); // [x, y, z, 0]
let mut positions = gpu_mesh.upload_attribute(&device, &host_positions)?;
let query = FaceCentroidQuery::new(&device);
let mut encoder = device.create_command_encoder(&Default::default());
let centroids = query.encode(&device, &mut encoder, &gpu_mesh, &positions)?;
queue.submit([encoder.finish()]);

// After editing host_positions through mesh vertex handles:
positions.write(&queue, &host_positions)?;

// Per-vertex application data, in input vertex order:
let weights = VertexAttribute::from_values(&mesh, vec![1.0f32; mesh.positions().len()]).unwrap();
let weights = gpu_mesh.upload_attribute(&device, &weights)?;

// Per-face and per-edge data work the same way:
let labels = FaceAttribute::new(&mesh, 0u32);
let labels = gpu_mesh.upload_attribute(&device, &labels)?;
```

`Attribute<E, T>` holds one value per element of kind `E`, which is a handle
type: `VertexAttribute<T>`, `EdgeAttribute<T>` and `FaceAttribute<T>` are
aliases for `Attribute<VertexHandle, T>`, `Attribute<EdgeHandle, T>` and
`Attribute<FaceHandle, T>`. Values are in canonical order: input order for
vertices and faces, canonical edge ID order for edges. Create one with
`Attribute::new(&mesh, value)` or `Attribute::from_values(&mesh, values)`,
which returns `None` unless there is exactly one value per element;
`Mesh::position_attribute()` gives the positions as vec4 values. `get` and
`get_mut` take a handle; every ribbon copy of an element reads and writes the
same value.

`GpuAttribute<E, T>` (aliases `GpuVertexAttribute`, `GpuEdgeAttribute`,
`GpuFaceAttribute`) stores the same canonical order on the GPU. Create one
with `GpuMesh::upload_attribute`, with `GpuAttribute::new(&device, &gpu_mesh,
&values)` (for meshes built with `GpuMeshBuilder`, which have no host
attributes), or around an existing buffer with `GpuAttribute::from_buffer`.
Replace its contents with `write(&queue, &host_attribute)` or
`write_values(&queue, &values)`. Uploads, updates, and queries reject
attributes from another mesh, even if the meshes have identical topology.
Buffers this crate creates have STORAGE, COPY_SRC, and COPY_DST usage.
Query results such as centroids and normals are GPU attributes too, so they
can be passed straight to later queries.

Supported element types are `f32`, `u32`, `i32`, and arrays of two or four of
those scalars, matching WGSL scalar/vec2/vec4 storage-array strides. Three-element
Rust arrays are excluded because WGSL vec3 arrays require 16-byte stride.
An empty attribute has zero logical elements and one zero element of storage.

Host and GPU attributes are independent snapshots. Updates replace the full GPU
buffer on the next queue submission; submit earlier readers first, since queue
writes also affect commands already encoded but not submitted. Updating a
position attribute does not change `Mesh::positions()` or revalidate geometry.
GPU writes are not automatically copied back to host attributes.

Geometry queries read a `GpuVertexAttribute<[f32; 4]>` of positions (the w
lane is ignored) and return GPU attributes:

| Query | Result | Value |
|---|---|---|
| `FaceCentroidQuery` | `GpuFaceAttribute<[f32; 4]>` | average of the three corners, `[x, y, z, 0]` |
| `FaceNormalQuery` | `GpuFaceAttribute<[f32; 4]>` | unit normal, `[x, y, z, 0]`; corners appear counter-clockwise from the side it points to |
| `FaceAreaQuery` | `GpuFaceAttribute<f32>` | triangle area, never negative |
| `VertexNormalQuery` | `GpuVertexAttribute<[f32; 4]>` | normalized sum of the incident faces' normals, each scaled by twice its face's area, `[x, y, z, 0]` |

Faces with zero area (collinear corners) get a zero normal and zero area,
not NaN, and add nothing to vertex normals. A vertex whose weighted face
normals cancel gets a zero normal. Vertex normals sum each vertex's faces in
an unspecified order, so they may differ in the last bits between runs.
Results are checked against Alum (face normals and areas) and an f64
reference (vertex normals) to 1e-5. Empty meshes return a zero placeholder
without dispatch. Finite, reasonably scaled f32 positions are expected;
extreme coordinates remain subject to floating-point overflow and rounding.
The matching WESL helpers are `triangle_area_vector`, `triangle_normal`,
`triangle_area` and `normalize_or_zero` in `mopmesh::geometry::triangle`.

### Boundary flags, reductions and smoothing

```rust,ignore
use mopmesh::gpu::{BoundaryQuery, ReduceOp, ReduceQuery, SmoothQuery, Smoothing};

let boundary = BoundaryQuery::new(&device).encode(&device, &mut encoder, &gpu_mesh)?;
// boundary.edges: GpuEdgeAttribute<u32>; boundary.vertices: GpuVertexAttribute<u32>.
let smoothed = SmoothQuery::new(&device).encode(
    &device, &mut encoder, &adjacency, &positions,
    Some(&boundary.vertices), // keep boundary vertices in place
    Smoothing { step: 0.5, iterations: 10 },
)?;
let total = ReduceQuery::<f32>::new(&device, ReduceOp::Sum)
    .encode(&device, &mut encoder, &face_areas)?;
// total.buffer() holds one f32 at byte 0.
```

`BoundaryQuery` flags each edge and vertex with 1 on the boundary and 0
elsewhere: a boundary edge has one adjoining face, and a boundary vertex
touches a boundary edge. Closed meshes have no boundary.

`ReduceQuery<T>` combines every value of an `f32`, `u32` or `i32` attribute of
any element kind with `ReduceOp::Sum`, `Min` or `Max`; `DotQuery` sums the
element-wise product of two `f32` attributes of the same mesh. The result is a
`GpuReduction<T>`, one value at byte 0 of a 4-byte buffer with STORAGE,
COPY_SRC and COPY_DST usage, so later GPU work can read it without a
readback. Integer sums wrap. Empty attributes give 0 for sums and dot
products, +infinity or the largest integer for `Min`, and -infinity, 0 or
the smallest `i32` for `Max`. The combination order is fixed for a given
length, so repeated `f32` results on one device agree. NaN inputs give
unspecified results. Reductions combine 64 values per workgroup per level,
use a two-dimensional dispatch grid when needed, and allocate a scratch
buffer and one uniform per level on each call.

`SmoothQuery` runs uniform Laplacian steps: each step moves every free vertex
`step` of the way toward the average of its edge neighbors, read from a
`MeshAdjacency`. Vertices whose `fixed` flag is non-zero, and vertices with no
neighbors, stay in place; `w` lanes are copied unchanged. The input positions
are not modified. Zero iterations return a copy, which needs `COPY_SRC` usage
on the input. Smoothing shrinks surfaces and, without fixed boundary
vertices, open boundaries too. Neighbors are summed in an unspecified order,
so results may differ in the last bits between runs. Tests compare against an
f64 host reference after up to four steps, with and without fixed boundaries.

### Editing: edge flips

Topology edits work on a `GpuEditableMesh`, a separate upload whose patches
have spare slots for new edges and faces. `GpuMesh` stays read-only.

```rust,ignore
use mopmesh::gpu::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh};

// Each patch gets 1.5 times as many slots as it has elements.
let mut editable = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default())?;
let flips = EdgeFlipQuery::new(&device);
// Request any number of edges; they are flipped in parallel.
let statuses = flips.encode(&device, &mut encoder, &mut editable, &edge_ids)?;
// After submitting, read statuses.buffer(): one word per requested edge.
// FlipStatus::from_word(word) == Some(FlipStatus::Flipped)
```

Flipping an edge shared by two triangles replaces it with the other diagonal
of the four-sided region they form. For edge `a-b` with `a < b`, triangles
`a -> b -> c` and `b -> a -> d` become `a -> d -> c` and `d -> b -> c`, each
keeping its face ID and starting at those corners. The flipped edge keeps its
edge ID, and its endpoints `c` and `d` are stored in ascending order like every
other edge, so edge and face attributes keep their slots. Their values are left as they were; recompute
them if they depend on the changed connectivity. Edge IDs on an edited mesh
no longer follow the ascending vertex-pair order of a freshly built mesh.

A flip is rejected, and the mesh left exactly as it was, when:
- the edge is on the boundary (`Boundary`);
- any face around its four corners is owned by another patch (`NotInterior`);
  flips across patch borders aren't supported yet;
- the two new corners are already joined by an edge (`Connected`);
- the two faces don't form a proper four-sided region (`Degenerate`);
- the patch has no free edge slot or two free face slots (`OutOfCapacity`);
- another requested flip shares one of its four corners and won it
  (`LostClaim`);
- the edge ID is out of range (`NotAnEdge`).

`LostClaim` and `OutOfCapacity` can depend on the other flips in the same
call, so request those edges again in a later call to retry them. Which of two
conflicting flips wins is pseudo-random and changes from call to call. List
each edge at most once per call: a repeated edge flips once, and requesting
its losing copies again would flip it back.

Slots freed by a flip are reused by flips in later calls, so capacity only has
to cover one call's flips. Calls recorded one after another apply in order.
Each `encode` call increases `GpuEditableMesh::version()`, even when the GPU
rejects every flip. Results computed before an edit are not updated; recompute
them afterwards. Static queries don't accept a `GpuEditableMesh` yet.

**Flipping from your own WESL kernel.** `topology::flip` exposes the flip as
plain functions, so a pipeline can test its own condition and flip in the same
dispatch, using its own bindings. Bind `GpuEditableMesh::arena()` as
`array<u32>` (read-write) and `GpuEditableMesh::masks()` as
`array<atomic<u32>>`, decode records with `topology::patch`, then:

1. Look up the edge's patch and local slot in the edge owner table
   (`owner_offset(arena[OWNERS_WORD + EDGE], edge)`), and decode that patch's
   record.
2. Find the edge's two active faces in that patch and call `flip_quad`, passing
   first the face for which `runs_stored_order(face, edge)` is true. Continue
   only if its `status` is `FLIP_DONE`.
3. Apply your own condition, for example "the new edge is shorter".
4. Check the library's preconditions: all four corners set in the patch's
   interior mask, no active edge already joining `c` and `d`, and one free edge
   slot and two free face slots (`free_bit` on active-mask words, below the
   record's capacity).
5. Call `flip_writes`, with the mesh-wide vertex IDs of `c` and `d` in
   `FlipSlots::global_c` and `global_d`, and apply every entry: store each `ArenaWrite`, and
   apply each `MaskUpdate` with `atomicOr` (`MASK_SET`), `atomicAnd` of the
   complement (`MASK_CLEAR`) or `atomicMax` (`MASK_MAX`).

Rules for step 5:
- **All or nothing.** Nothing on the GPU checks the list, so do every check
  before the first write. A partial commit leaves the patch inconsistent.
- **Once only.** Applying a list twice corrupts the patch.
- **Order doesn't matter.** No two entries touch the same word.
- **Arena writes can be plain stores, but mask updates must be atomic.** A
  flip writes only its own slots, while several slots share each mask word.
- **One flip per patch per dispatch.** Two concurrent flips in one patch
  could share a corner or pick the same free slot. To flip many edges at
  once, use the three-dispatch protocol below.

**Flipping many edges at once from WESL.** Use `topology::flip_pass` to
choose edges with your own shader rule while the library handles competing
requests and mesh updates. Give each request a unique index, list each edge
at most once, and run three entry points in order: claim, reserve, then commit.
Each stage handles one request per invocation and skips indices at or beyond
`flip_pass_count()`. No CPU readback is needed between stages.

The claim stage below uses two application-defined functions:
`edge_for_request(i)` supplies a mesh-wide edge ID, and `my_rule` decides
whether its four vertices need a flip.

```wgsl
import mopmesh::topology::flip_pass::{
    flip_pass_count, flip_candidate, flip_claim, flip_reserve, flip_commit,
};
import mopmesh::topology::flip::{FLIP_DONE, FLIP_NOT_REQUESTED};
import mopmesh::parallel::grid::flat_index;

@group(0) @binding(0) var<storage, read_write> status: array<u32>;

@compute @workgroup_size(64)
fn claim(@builtin(global_invocation_id) id: vec3<u32>,
         @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(id, groups);
    if (i >= flip_pass_count()) { return; }
    let candidate = flip_candidate(i, edge_for_request(i));
    var outcome = candidate.status;
    if (outcome == FLIP_DONE && my_rule(candidate.corners)) {
        flip_claim(i, candidate);
    } else if (outcome == FLIP_DONE) {
        outcome = FLIP_NOT_REQUESTED;
    }
    status[i] = outcome;
}
// reserve: if status[i] == FLIP_DONE, set status[i] = flip_reserve(i).
// commit: call flip_commit(i).
```

`candidate.corners` contains mesh-wide vertex IDs `[a, b, c, d]`: the
current edge joins `a` to `b`, and the replacement would join `c` to `d`.
Read these only when the candidate status is `FLIP_DONE`. At this stage,
that status means the edge is eligible, not that it has already flipped.

In the host pipeline layout, use your application bindings as group 0 and
`EdgeFlipQuery::pass_layout()` as group 1. For each batch, call
`EdgeFlipQuery::begin_pass(device, &mut mesh, count)` and bind the returned
`FlipPass::bind_group()` at group 1 for all three dispatches. Use
`FlipPass::grid()` for the x and y workgroup counts and 1 for z with the
64-invocation workgroups shown above. Preparing a pass increases the mesh
version even if no dispatch follows. Do not also import
`topology::patch_arena`, which uses group 1 too.

After all claims finish, call `flip_reserve(i)` for each eligible request
and store its returned status. After all reservations finish, call
`flip_commit(i)` once per request. Only successfully reserved requests
change the mesh. Final statuses follow the same rules as
`EdgeFlipQuery::encode`: requests that share a vertex may report
`FLIP_LOST_CLAIM`, and requests without replacement storage report
`FLIP_OUT_OF_CAPACITY`. Both can be retried in a fresh batch; retrying does
not create additional storage.

[`examples/delaunay_flips.rs`](examples/delaunay_flips.rs) and its
[shader](examples/shaders/delaunay.wesl) demonstrate this approach on a flat,
skewed grid. They request a flip when the angles opposite an edge sum to
more than 180 degrees, allowing a small tolerance, and repeat batches until
no flips or retries remain. This is the local angle test for a Delaunay
triangulation. The library's [edge-flip kernel](runtime_shaders/edge_flip.wesl)
uses the same stages without an application rule.

If group 1 is already in use, the header of `topology::flip` explains how to
manage the three stages with your own buffers and bindings.
[`tests/shaders/flip_consumer.wesl`](tests/shaders/flip_consumer.wesl) provides
a smaller example of a single flip with custom bindings.

The editable arena is one `u32` buffer: a header, a 16-word record per patch,
owner tables mapping each global ID to its owning patch and local slot, and
per-patch rows sized by capacity, with `INVALID_ID` in unused slots. The mask
buffer holds per-patch active, owned and interior bitmasks and used-slot
counts. `GpuEditableMesh::arena` and `masks` document the word layout. It is
experimental and may change.

## Validation coverage

GPU reference tests cover 128-face and 8,192-face grids, a grid with a hole,
a closed tetrahedron, disconnected triangles, a 129-face vertex fan, and empty
meshes at patch sizes 1, 3, and 128,
including repeated queries, scalar/vector uploads, cross-mesh rejection, and
centroids before and after position updates through ribbon handles, edge
connectivity, VE/VV, FF slot order, boundary/interior diamonds, and both downstream
operators. Repeated adjacency execution is also checked. They passed on
Apple M1 Max / Metal; other backends have not yet been exercised.
The ordinary test suite runs shader validation, arena sizing, binding and
dispatch-limit checks, and scan allocation checks without requiring GPU
hardware. GPU tests also check mesh IDs on every result and per-query limit
errors on devices with reduced limits.
GPU scan tests cover empty, exact, and partial blocks, zeros, varying values,
wrapping arithmetic, and up to four hierarchy levels (262,145 values).
The latest recorded full run passed 16 tests (14 hardware-independent and two
opt-in GPU tests); both examples, Clippy and formatting checks also passed.
These are correctness results, not benchmarks or exhaustive topology coverage.

## Current constraints

The initial release supports native 32-bit GPU values: `u32`/`i32` integer data
and `f32` geometry, including vectors/records made of independent 32-bit fields.
It excludes 16-bit formats, subword numeric packing, native or emulated 64-bit
GPU values, and paired-float precision schemes. Packing and alternative widths
are outside the committed roadmap. Host `u64`/`usize` size checks remain in
use; they do not introduce wider GPU representations.

The order of elements within a patch (its local IDs) and which patch owns
which faces are unspecified. They
are the same on every build of the same input on the same device, but may
change between releases and differ between `GpuMesh::upload` and
`GpuMeshBuilder`. Global vertex, face and edge IDs do not depend on them.

`patch_size` limits owned faces only. Ribbons include the complete face stars
of vertices on owned faces, so high-valence vertices can cause substantial
duplication. Full one-ring queries are guaranteed for owned elements; resolve
ribbon elements to their owner before querying their full neighborhood.

Construction rejects non-finite coordinates, invalid indices, degenerate or
duplicate triangles, inconsistent winding, non-manifold edges or vertices,
and isolated vertices. Empty meshes and disconnected components are supported.
A triangle is degenerate when two of its corners have equal coordinates
(including a repeated vertex ID; `0.0` equals `-0.0`). Triangles whose three
distinct corners are collinear, or nearly so, are accepted. Geometric
self-intersections are not checked.

Handles are mesh-relative addresses and must not be used with another mesh.
Future migration and slot reuse invalidate them. Vertex attributes verify mesh
identity and share canonical values across ribbons, as do edge and face
attributes. Edits are limited to edge flips inside one patch on a
`GpuEditableMesh`; cross-patch edits, splits and collapses remain future work.
GPU topology queries currently access storage buffers directly; the scan uses
workgroup scratch, but topology is not yet staged in workgroup memory. Query
objects reuse pipelines while encode calls allocate fresh output/scratch
buffers. The batched adjacency API computes all four families together.
Their performance and allocation tradeoffs remain to be measured.

Built-in queries check mesh identity for typed attribute inputs. Downstream
compositions can compare `mesh_id()` values but must enforce that themselves;
device identity is never checked. Keep the caller's queue
alive while encoding and submitting work. No automatic device selection,
resource pooling, generation-checked handles or migration exists yet.

## Reference sources

Recorded revisions, dependency roles, and OpenMesh build instructions are in
[reference setup](prior_art/REFERENCES.md). Original upstream license notices are
retained in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). The downloaded source
checkouts are optional and ignored by the top-level repository; Cargo tests use
Alum and do not require those checkouts.

- [RXMesh: A GPU Mesh Data Structure (2021)](https://doi.org/10.1145/3450626.3459748)
- [Dynamic Mesh Processing on the GPU (2025)](https://doi.org/10.1145/3731162)
- [Alum](https://github.com/ranjeethmahankali/alum)
- [OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/)

## Next steps

Follow the [execution milestones](ROADMAP.md#next-execution-milestones). The
static API review is complete. Next,
establish an unpartitioned GPU benchmark baseline, then evaluate
workgroup-local traversal. Complete remaining static attributes and geometry
operators before starting interior edge flips. Static performance viability
remains an open gate rather than an assumed property of the patch representation.
