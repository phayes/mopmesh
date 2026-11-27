//! Checks the in-kernel local queries of `topology::patch_queries` against
//! the library's global queries: edge diamonds word for word, and vertex
//! neighbor rows as sorted lists. Covers every fixture on static and
//! editable uploads, a vertex with too many neighbors for shared memory, and
//! a mesh after many parallel flips with slot reuse.
#[path = "../examples/support/delaunay.rs"]
mod delaunay;
#[path = "support/fixtures.rs"]
mod fixtures;
#[path = "support/fresh_flips.rs"]
mod fresh_flips;
use mopmesh::{
    Mesh, MeshBuilder,
    gpu::{
        AdjacencyQuery, EdgeFlipQuery, EditCapacity, GpuEditableMesh, GpuMesh, GpuTopology,
        INVALID_ID,
    },
};
use wgpu::util::DeviceExt;

/// Rows a workgroup builds at once, and the neighbor entries that fit in
/// shared memory, as in `topology::local_query`.
const VV_ROWS: usize = 256;
const VV_ENTRIES: u32 = 1024;

/// What one comparison exercised.
#[derive(Default)]
struct Seen {
    /// A vertex row too large for shared memory.
    oversized: bool,
}

struct LocalQueries {
    adjacency: AdjacencyQuery,
    own: wgpu::BindGroupLayout,
    patches: wgpu::BindGroupLayout,
    diamonds: wgpu::ComputePipeline,
    neighbors: wgpu::ComputePipeline,
}
impl LocalQueries {
    fn new(device: &wgpu::Device) -> Self {
        let mut resolver = wesl::resolver::StandardResolver::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/shaders"
        ));
        resolver.add_package(&mopmesh::mopmesh::PACKAGE);
        let wgsl = wesl::Compiler::default()
            .with_resolver(resolver)
            .compile_module(&"package::local_query_consumer".parse().unwrap())
            .expect("link the local query consumer")
            .to_string();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("local query consumer"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // Group 0 holds this test's buffers; group 1 the read-only arena and
        // masks that `topology::patch_arena` expects.
        let layout = |label, read_only: &[bool]| {
            let entries: Vec<_> = read_only
                .iter()
                .enumerate()
                .map(|(i, &read_only)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            })
        };
        let own = layout("local query outputs", &[true, false, false, false]);
        let patches = layout("local query patches", &[true, true]);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("local query consumer"),
            bind_group_layouts: &[Some(&own), Some(&patches)],
            immediate_size: 0,
        });
        let stage = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            adjacency: AdjacencyQuery::new(device),
            diamonds: stage("edge_diamonds"),
            neighbors: stage("vertex_neighbors"),
            own,
            patches,
        }
    }

    /// Runs the local queries and the global adjacency query on `mesh`, then
    /// asserts that they agree.
    fn compare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &impl GpuTopology,
        context: &str,
    ) -> Seen {
        let (vertices, edges) = (
            mesh.vertex_capacity() as usize,
            mesh.edge_capacity() as usize,
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        let global = self.adjacency.encode(device, &mut encoder, mesh).unwrap();

        // Zeroed outputs, laid out like the global results.
        let output = |label, words: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (4 * words as u64).max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let neighbors = output("local neighbors", 2 * edges);
        let rows = output("local rows", 3 * vertices);
        // Diamonds start as INVALID_ID, as the global query leaves the rows
        // of unused edge IDs.
        let diamonds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local diamonds"),
            contents: bytemuck::cast_slice(&vec![INVALID_ID; (4 * edges).max(1)]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bind = |layout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            })
        };
        let own = bind(
            &self.own,
            &[global.vertex_neighbors(), &neighbors, &rows, &diamonds],
        );
        let patches = bind(&self.patches, &[mesh.arena(), mesh.masks()]);
        for pipeline in [&self.diamonds, &self.neighbors] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &own, &[]);
            pass.set_bind_group(1, &patches, &[]);
            pass.dispatch_workgroups(mesh.patch_count(), 1, 1);
        }
        queue.submit([encoder.finish()]);

        // Diamonds match word for word, including the face order.
        let words = |b: &wgpu::Buffer| delaunay::read_words(device, queue, b);
        if edges > 0 {
            assert_eq!(
                words(&diamonds)[..4 * edges],
                words(global.edge_diamonds())[..4 * edges],
                "{context}: diamonds"
            );
        }

        // Each vertex's neighbors match as a sorted list, and its size and
        // oversized flag agree with the row.
        let csr = words(global.vertex_neighbors());
        let (offsets, pairs) = csr.split_at(vertices + 1);
        let (ours, ours_rows) = (words(&neighbors), words(&rows));
        let mut seen = Seen::default();
        for v in 0..vertices {
            let range = offsets[v] as usize..offsets[v + 1] as usize;
            let mut expected: Vec<u32> = range.clone().map(|k| pairs[2 * k + 1]).collect();
            let row = &ours_rows[3 * v..3 * v + 3];
            assert_eq!(
                row[0] as usize,
                expected.len(),
                "{context}: vertex {v} written"
            );
            assert_eq!(
                row[1] as usize,
                expected.len(),
                "{context}: vertex {v} vv_size"
            );
            assert_eq!(
                row[2] == 1,
                expected.len() as u32 > VV_ENTRIES,
                "{context}: vertex {v} oversized"
            );
            seen.oversized |= row[2] == 1;
            let mut actual = ours[range].to_vec();
            actual.sort();
            expected.sort();
            assert_eq!(actual, expected, "{context}: vertex {v} neighbors");
        }
        seen
    }
}

fn build(positions: &[[f32; 3]], triangles: &[[u32; 3]], patch_size: usize) -> Mesh {
    MeshBuilder::new()
        .patch_size(patch_size)
        .positions(positions)
        .triangles(triangles)
        .build()
        .unwrap()
}

fn chunked_vertices(mesh: &Mesh) -> bool {
    mesh.patches()
        .iter()
        .any(|p| p.vertex_ids().len() > VV_ROWS)
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_local_query -- --ignored --nocapture"]
fn local_queries_match_the_global_queries_on_every_fixture() {
    let (device, queue) = gpu_device();
    let q = LocalQueries::new(&device);
    let mut chunked = false;
    for fixture in fixtures::valid() {
        for patch_size in fixtures::PATCH_SIZES.into_iter().chain([512]) {
            let mesh = build(&fixture.positions, &fixture.triangles, patch_size);
            let context = format!("{} at patch size {patch_size}", fixture.name);
            chunked |= chunked_vertices(&mesh);
            let fixed = GpuMesh::upload(&device, &mesh).unwrap();
            q.compare(&device, &queue, &fixed, &format!("{context}, static"));
            let spare = EditCapacity::new(1.5).unwrap();
            let editable = GpuEditableMesh::upload(&device, &mesh, spare).unwrap();
            q.compare(&device, &queue, &editable, &format!("{context}, editable"));
        }
    }
    assert!(
        chunked,
        "no patch needed more than one chunk of vertex rows"
    );
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_local_query -- --ignored --nocapture"]
fn a_vertex_with_more_neighbors_than_shared_memory_holds_is_found_by_scanning() {
    // A fan around vertex 0 whose rim has more vertices than VV_ENTRIES,
    // in one patch, so its edges also need several staging chunks.
    let rim = VV_ENTRIES + 100;
    let mut positions = vec![[0.0, 0.0, 0.0]];
    for i in 0..rim {
        let angle = i as f32 / rim as f32 * std::f32::consts::TAU;
        positions.push([angle.cos(), angle.sin(), 0.0]);
    }
    let triangles: Vec<[u32; 3]> = (1..rim).map(|i| [0, i, i + 1]).collect();
    let mesh = build(&positions, &triangles, 4096);
    assert_eq!(mesh.patches().len(), 1);
    let (device, queue) = gpu_device();
    let q = LocalQueries::new(&device);
    let fixed = GpuMesh::upload(&device, &mesh).unwrap();
    let seen = q.compare(&device, &queue, &fixed, "fan");
    assert!(seen.oversized, "the fan's center row fit in shared memory");
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_local_query -- --ignored --nocapture"]
fn local_queries_follow_edits_with_reused_slots() {
    let (device, queue) = gpu_device();
    let q = LocalQueries::new(&device);
    let (mesh, positions, _) = delaunay::skewed_grid(17, 64);
    assert!(mesh.patches().len() > 1);
    let mut gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::new(1.1).unwrap()).unwrap();
    let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("positions"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &query);

    // Flip some edges into free slots first, so patches have empty and
    // reused slots below their used counts.
    let some_edges: Vec<u32> = (0..mesh.edge_count() as u32).step_by(5).collect();
    let fresh = fresh_flips::FreshFlips::new(&device).flip(
        &device,
        &queue,
        &mut gpu,
        &positions,
        &some_edges,
    );
    let fresh_flipped = fresh
        .iter()
        .filter(|&&s| s == Some(mopmesh::gpu::FlipStatus::Flipped))
        .count();
    assert!(
        fresh_flipped > mesh.patches().len(),
        "only {fresh_flipped} flips into free slots"
    );
    q.compare(&device, &queue, &gpu, "after flips into free slots");

    // Then flip in batches until one flips nothing, checking the local
    // queries after every batch.
    let mut flipped = 0;
    for round in 0..64 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let status = flips
            .encode(&device, &mut encoder, &query, &mut gpu, &positions)
            .unwrap();
        queue.submit([encoder.finish()]);
        let done = delaunay::Round::from_statuses(&delaunay::read_words(&device, &queue, &status));
        q.compare(&device, &queue, &gpu, &format!("after round {round}"));
        if done.flipped == 0 {
            break;
        }
        flipped += done.flipped;
    }
    assert!(
        flipped > mesh.patches().len() as u32,
        "only {flipped} flips in batches"
    );
}

fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    pollster::block_on(adapter.request_device(&Default::default())).unwrap()
}
