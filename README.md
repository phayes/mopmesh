# MOPMesh

MOPMesh is a WESL library for GPU triangle-mesh processing. Import topology,
geometry, and parallel helpers into your own compute shaders to traverse
neighborhoods, calculate mesh properties, and flip edges. You choose the
application buffers, entry points, and output.

The accompanying Rust/wgpu runtime builds and uploads meshes, prepares
connectivity buffers, and provides ready-made queries. The WESL package lives in
[`shaders/`](shaders/); the runtime's entry points live separately in
[`runtime_shaders/`](runtime_shaders/).

MOPMesh is experimental: public APIs and buffer layouts may change. It supports
oriented manifold triangle meshes, including boundaries and disconnected
components. Topology edits currently support edge flips within a patch.

## Start with a shader

You can use MOPMesh in two ways:

- **Read connectivity buffers.** Bind edge records or adjacency rows and compose
  value-based helpers with your own data. This is the easiest starting point.
- **Traverse patches inside your kernel.** Bind the mesh arena and masks, then
  use workgroup-local queries to find neighbors and edge diamonds without
  materializing global adjacency buffers.

For example, this complete shader computes a weighted boundary length and an
incident-face count for each edge:

```wesl
import mopmesh::topology::edge::{EdgeRecord, is_boundary_edge, edge_face_count};
import mopmesh::geometry::edge::edge_length;

@group(0) @binding(0) var<storage, read> edges: array<EdgeRecord>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<vec2<f32>>;

@compute @workgroup_size(64)
fn weighted_boundary(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&weights)) { return; }
    let edge = edges[id.x];
    var contribution = 0.0;
    if (is_boundary_edge(edge.faces)) {
        contribution = weights[id.x] * edge_length(
            positions[edge.vertices.x].xyz, positions[edge.vertices.y].xyz);
    }
    output[id.x] = vec2(contribution, f32(edge_face_count(edge.faces)));
}
```

Bind one `EdgeRecord`, weight, and output element per mesh edge, and one
`vec4<f32>` position per mesh vertex (`w` is unused). An `EdgeRecord` contains
`vertices: vec2<u32>` and `faces: vec2<u32>`: mesh-wide IDs, with `0xffffffff`
for a missing boundary face. Dispatch `ceil(edge_count / 64)` workgroups;
skip the dispatch for an empty mesh.

The Rust runtime's `EdgeQuery` produces the edge buffer. Your shader can consume
it in the same command submission, with no CPU readback between the query and
your operator. The complete [shader](examples/shaders/weighted_boundary.wesl),
[host integration](examples/support/boundary.rs), and
[runnable example](examples/weighted_boundary.rs) are included:

```sh
cargo run --example weighted_boundary
```

This runs on a two-triangle unit square and reports five edges, four boundary
edges, and a weighted boundary length of four with unit weights. It requires a
GPU adapter.

## Import and compile WESL

Imports use the `mopmesh` package name:

```wesl
import mopmesh::geometry::triangle::{triangle_area, triangle_normal};
import mopmesh::topology::csr::{CsrRow, row_from_count, row_length};
import mopmesh::parallel::scan::exclusive_scan_64;
```

Most helpers take values and declare no resource bindings or entry points.
You can use them with your own buffer layouts. The patch readers and flip-pass
helpers also provide an opt-in binding layout, described below.

For a Rust consumer, add MOPMesh from your local checkout and the WESL compiler:

```toml
[dependencies]
mopmesh = { path = "../mopmesh" }
wesl = "0.6.0"
```

Save the first shader as `my_shaders/weighted_boundary.wesl`, register the
exported package, and compile it to WGSL:

```rust
let mut resolver = wesl::resolver::StandardResolver::new("my_shaders");
resolver.add_package(&mopmesh::mopmesh::PACKAGE);
let wgsl = wesl::Compiler::default()
    .with_resolver(resolver)
    .compile_module(&"package::weighted_boundary".parse().unwrap())?
    .to_string();
```

Pass the resulting string to your GPU API's shader-module creation function.
[`examples/support/boundary.rs`](examples/support/boundary.rs) shows the wgpu
pipeline, bindings, and dispatch. The package and example shaders are linked
and validated as WGSL by [`tests/wesl_package.rs`](tests/wesl_package.rs).

## Traverse a mesh inside your kernel

MOPMesh divides a mesh into patches. Each patch contains owned elements plus
copies of nearby elements, called its *ribbon*, so neighborhoods of owned
elements are complete. Process only owned elements when writing one result per
mesh element.

A **local slot** addresses an element within a patch. A **global ID** addresses
its canonical mesh-wide attribute value. Convert local slots with
`global_vertex`, `global_edge`, or `global_face` before indexing attributes.
Ribbon copies share the same global IDs as their owners.

The following kernel counts each vertex's neighbors using workgroup-local
traversal:

```wesl
import mopmesh::topology::patch::VERTEX;
import mopmesh::topology::patch_arena::{
    patch_count, patch_record, slot_active, slot_owned, global_vertex,
};
import mopmesh::topology::patch_queries::{
    patch_slot_end, stage_vertex_neighbors, vv_begin, vv_valid, vv_next,
};
import mopmesh::parallel::grid::flat_group;

@group(0) @binding(0) var<storage, read_write> degree: array<u32>;

@compute @workgroup_size(64)
fn vertex_degree(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let p = flat_group(group, groups);
    if (p >= patch_count()) { return; }
    let r = patch_record(p);
    let vertices = patch_slot_end(lane, r, VERTEX);
    var first = 0u;
    while (first < vertices) {
        let end = stage_vertex_neighbors(lane, r, first);
        for (var v = first + lane; v < end; v += 64u) {
            if (!slot_active(r, VERTEX, v) || !slot_owned(r, VERTEX, v)) { continue; }
            var count = 0u;
            for (var it = vv_begin(r, first, v); vv_valid(it); it = vv_next(r, it)) {
                count++;
            }
            degree[global_vertex(r, v)] = count;
        }
        first = end;
    }
}
```

Bind your output at group 0, binding 0, with one `u32` per mesh vertex.
`patch_queries` imports `patch_arena`, which declares two read-only storage
bindings: the mesh's `arena()` at group 1, binding 0, and `masks()` at group 1,
binding 1. Dispatch one 64-lane workgroup per patch. These readers work with
both static and editable meshes.

Call `patch_slot_end` and `stage_vertex_neighbors` from every invocation in the
workgroup, in uniform control flow: they contain synchronization. The iterator
is valid until the next staging call. To read a neighbor's attribute, import
`vv_vertex` and use `global_vertex(r, vv_vertex(r, it))` as the attribute index.
Neighbor order is unspecified.

For edge diamonds, stage an edge chunk with
`topology::patch_edge_staging::stage_patch_edges`, then call
`topology::patch_queries::edge_diamond`. It returns two endpoints and two
opposite vertices as local slots; a missing boundary-side vertex is
`INVALID_VERTEX`. Check that sentinel before converting to a global ID.
The [local-query consumer](tests/shaders/local_query_consumer.wesl) demonstrates
both traversals in full.

If you need custom bindings, use the value-based layout helpers in
[`topology::patch`](shaders/topology/patch.wesl) and the staging primitives in
[`topology::local_query`](shaders/topology/local_query.wesl) and
[`topology::patch_staging`](shaders/topology/patch_staging.wesl).

## Flip edges with your own rule

Use `topology::flip_pass` to select edges in WESL while MOPMesh coordinates
competing edits. The [Delaunay shader](examples/shaders/delaunay.wesl) requests a
flip when the angles opposite an edge sum to more than 180 degrees:

```sh
cargo run --example delaunay_flips
```

A batch is one dispatch with one 64-lane workgroup per patch, as in RXMesh's
cavity manager. The workgroup stages the patch's edges, offers each candidate
from `flip_candidate` to your rule, and passes your decision to
`flip_request`. Then `flip_finish` selects a maximal set of non-conflicting
flips and applies them.

A candidate with `status == FLIP_DONE` is eligible for your rule; it has not
flipped yet. Its `corners` are mesh-wide vertex IDs `[a, b, c, d]`, with current
diagonal `a-b` and proposed diagonal `c-d`. Read them only for eligible
candidates. Follow the example's collective calls and synchronization when
adapting the kernel.

Requested flips are selected in decreasing priority. A flip that shares a
vertex with an already selected one reports `FLIP_LOST_CLAIM` and can be
retried in a later batch. A flip whose `c-d` edge already exists reports
`FLIP_CONNECTED` and blocks nothing. Each flip rewrites its own edge and face
slots, so flips need no spare capacity.

On the host, create a `GpuEditableMesh` and an `EdgeFlipQuery`. Use your own
bindings at group 0 and `EdgeFlipQuery::pass_layout()` at group 1. For each batch,
call `begin_pass`, bind its `bind_group()`, and dispatch `patch_grid()`
workgroups; the results are in `statuses()`.
Do not combine `flip_pass` with `patch_arena` in the same shader: both use group 1.
See the [host integration](examples/support/delaunay.rs).

Flips preserve global edge and face IDs. Attribute slots remain valid, but
values derived from connectivity need to be recomputed. Boundary edges,
cross-patch neighborhoods, existing replacement diagonals, and degenerate
configurations are rejected and reported per edge. Recompute query results
after edits and submit commands in recording order.

For lower-level control with your own bindings,
[`topology::flip`](shaders/topology/flip.wesl) exposes precondition and write-list
helpers. The [single-flip consumer](tests/shaders/flip_consumer.wesl) demonstrates
this path; its caller is responsible for validating and applying the complete
edit exactly once.

## More shader building blocks

- **Geometry:** [`geometry::triangle`](shaders/geometry/triangle.wesl) provides
  centroids, areas, area vectors, normals, and `normalize_or_zero`;
  [`geometry::edge`](shaders/geometry/edge.wesl) provides edge lengths.
- **Connectivity:** [`topology::face`](shaders/topology/face.wesl),
  [`topology::directed_edge`](shaders/topology/directed_edge.wesl), and
  [`topology::edge`](shaders/topology/edge.wesl) decode face vertices, edge
  orientation, and boundary information.
- **Adjacency rows:** [`topology::csr`](shaders/topology/csr.wesl) describes
  half-open row ranges for traversing connectivity buffers. The
  [neighborhood shader](examples/shaders/neighborhood.wesl) combines these with
  vertex weights and geometry helpers.
- **Parallel operations:** [`parallel::scan`](shaders/parallel/scan.wesl)
  provides `exclusive_scan_64(value, lane)`, returning an exclusive prefix and
  block total. All 64 invocations must call it uniformly with distinct
  `local_invocation_index` values; inactive lanes supply zero. Arithmetic wraps
  at `u32`. [`parallel::grid`](shaders/parallel/grid.wesl) flattens dispatch IDs.
- **Construction:** [`build/`](shaders/build/) provides exact position checks,
  edge-group validation, and Morton-code helpers.

Try the other application examples:

```sh
cargo run --example neighborhood
cargo run --example smoothing
```

The neighborhood example computes degrees, boundary-edge counts, and weighted
edge lengths. The smoothing example combines built-in smoothing and normal
queries with a [custom bending shader](examples/shaders/bending.wesl), then
reduces the results on the GPU.

## Rust host setup

The runtime uses wgpu 30 and works with a device and queue supplied by your
application. To build a mesh on the CPU and prepare edge connectivity:

```rust
use mopmesh::MeshBuilder;
use mopmesh::gpu::{EdgeQuery, GpuMesh};

let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]];
let triangles = [[0, 1, 2]];
let mesh = MeshBuilder::new()
    .positions(&positions)
    .triangles(&triangles)
    .patch_size(512)
    .build()?;
let gpu_mesh = GpuMesh::upload(&device, &mesh)?;
let query = EdgeQuery::new(&device); // Cache query pipelines per device.
let mut encoder = device.create_command_encoder(&Default::default());
let edges = query.encode(&device, &mut encoder, &gpu_mesh)?;
// Bind edges.buffer() in your operator and record its dispatch here.
queue.submit([encoder.finish()]);
```

If positions and triangles already reside on the GPU, use `GpuMeshBuilder` with
`GpuMeshInput`. Positions use `[f32; 4]` and triangles use `[u32; 4]`, with the
fourth lane ignored. Building validates the input before producing a `GpuMesh`.

Typed vertex, edge, and face attributes store one value per global ID.
`GpuAttribute::from_buffer` wraps an existing GPU buffer, and
`GpuAttribute::new` uploads values. Query results remain on the GPU and expose
buffers and logical counts for downstream shaders. Use logical counts rather
than buffer sizes: empty results still have bindable placeholder storage.

Ready-made queries cover face vertices, edge connectivity, vertex/face
adjacency, centroids, normals, areas, boundary flags, reductions, and Laplacian
smoothing. Generate the Rust API reference with:

```sh
cargo doc --no-deps --open
```

Keep inputs on the same device and mesh. Built-in operations check mesh identity
for typed inputs; custom operators should compare `mesh_id()` themselves.
Connectivity-derived results from editable meshes carry a topology version and
must be refreshed after edits.

## Supported inputs and limits

- Construction rejects non-finite positions, invalid indices, duplicate or
  degenerate triangles, inconsistent winding, non-manifold edges or vertices,
  and isolated vertices. Empty meshes are supported.
- Degeneracy checks reject coincident corners. Distinct collinear corners are
  accepted; geometric self-intersections are not checked.
- GPU data uses 32-bit `u32`, `i32`, and `f32` values. Geometry helpers expect
  finite, reasonably scaled inputs.
- Patch size limits owned faces, not ribbon size. High-valence vertices can
  substantially increase duplication. Full neighborhoods are guaranteed for
  owned elements; ribbon copies may have incomplete neighborhoods.
- Patch ownership and local slot order are implementation details. Handles are
  mesh-relative and must not be used with another mesh.
- Cross-patch edits, edge splits, and collapses are not currently supported.
  Performance and portability beyond the tested Metal backend remain under
  evaluation.

## Development and references

Run the hardware-independent tests, including downstream WESL linking and WGSL
validation:

```sh
cargo test --all-targets
```

Include the GPU tests on a machine with a GPU adapter:

```sh
cargo test --all-targets -- --include-ignored --nocapture
```

See the [roadmap](ROADMAP.md) for implementation plans, the
[technical design](MOPMesh_Technical_Design.md) for architecture, and the
[benchmark notes](docs/approximate_benchmark.md) for timing methodology.

MOPMesh follows RXMesh's patch/ribbon architecture and uses Alum as a CPU
reference in tests. [Reference setup](prior_art/REFERENCES.md) records upstream
sources and revisions. MOPMesh is licensed under [BSD-2-Clause](LICENSE);
upstream notices are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
