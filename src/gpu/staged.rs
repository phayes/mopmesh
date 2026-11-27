//! Computes edge vertices and neighboring faces from an editable GPU mesh.
//! Each GPU workgroup handles one patch, sharing temporary results among its
//! invocations to build the edge records.
use super::pipeline::{self, check_binding, group_grid};
use super::{EdgeConnectivity, GpuEditableMesh, GpuError};
use crate::Mesh;

/// Queries edge vertices and neighboring faces using shared GPU memory.
/// For the same mesh, returns the same records as [`EdgeQuery`](super::EdgeQuery).
pub(super) struct StagedEdgeQuery {
    pipeline: wgpu::ComputePipeline,
}

impl StagedEdgeQuery {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh staged edges",
            include_str!(concat!(env!("OUT_DIR"), "/staged_edges.wgsl")),
        );
        Self {
            pipeline: pipeline::compute_pipeline(
                device,
                "MOPMesh staged edges",
                &module,
                "main",
                None,
            ),
        }
    }

    /// Adds the edge query to `encoder` and returns its GPU output buffer.
    /// Submit the encoder before reading the results. `gpu` must have been
    /// uploaded from `mesh`; a mesh identity mismatch panics.
    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &Mesh,
        gpu: &GpuEditableMesh,
    ) -> Result<EdgeConnectivity, GpuError> {
        const RESOURCE: &str = "staged edge connectivity";
        assert_eq!(mesh.id(), gpu.mesh_id());
        let limits = device.limits();
        let bytes = (u64::from(gpu.edge_count()) * 16).max(16);
        check_binding(&limits, bytes, RESOURCE)?;
        let grid = group_grid(&limits, mesh.patches().len() as u32, RESOURCE)?;
        let result = EdgeConnectivity {
            buffer: pipeline::output_buffer(device, "MOPMesh staged edge records", bytes),
            edge_count: gpu.edge_count(),
            mesh: gpu.mesh_id(),
        };
        if gpu.edge_count() != 0 {
            let output = pipeline::bind_buffers(
                device,
                "MOPMesh staged edge output",
                &self.pipeline.get_bind_group_layout(0),
                &[&result.buffer],
            );
            let arena = pipeline::bind_buffers(
                device,
                "MOPMesh staged edge arena",
                &self.pipeline.get_bind_group_layout(1),
                &[gpu.arena(), gpu.masks()],
            );
            pipeline::dispatch_grid_groups(
                encoder,
                "MOPMesh staged edges",
                &self.pipeline,
                &[&output, &arena],
                grid,
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::bench_meshes;
    use crate::gpu::test_support::{self, fixtures};
    use crate::gpu::{EdgeQuery, EditCapacity, GpuMesh};

    /// Checks that every output word matches EdgeQuery, both with no extra
    /// editing capacity and with 50% extra capacity requested.
    fn check(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        queries: (&StagedEdgeQuery, &EdgeQuery),
        mesh: &Mesh,
        context: &str,
    ) {
        let run = |record: &dyn Fn(&mut wgpu::CommandEncoder) -> EdgeConnectivity| {
            let mut encoder = device.create_command_encoder(&Default::default());
            let result = record(&mut encoder);
            test_support::read_words(device, queue, encoder, result.buffer())
        };
        let gpu = GpuMesh::upload(device, mesh).unwrap();
        let expected = run(&|e| queries.1.encode(device, e, &gpu).unwrap());
        for factor in [1.0, 1.5] {
            let capacity = EditCapacity::new(factor).unwrap();
            let editable = GpuEditableMesh::upload(device, mesh, capacity).unwrap();
            let actual = run(&|e| queries.0.encode(device, e, mesh, &editable).unwrap());
            assert_eq!(actual, expected, "{context}, capacity {factor}");
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn staged_edges_match_the_current_kernel() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let staged = StagedEdgeQuery::new(&device);
        let current = EdgeQuery::new(&device);
        let limits = device.limits();

        // Cover all valid test meshes at every configured patch size.
        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                let context = format!("{} at patch size {patch_size}", fixture.name);
                check(&device, &queue, (&staged, &current), &mesh, &context);
            }
        }

        // Test generated meshes with both original and shuffled face order.
        // The large sphere also checks patches whose edges need several chunks.
        let mut chunked = false;
        for (faces, patch_size) in [(2000, 32), (2000, 512), (131072, 32)] {
            for base in bench_meshes::generated(faces) {
                if faces > 2000 && !base.name.starts_with("sphere") {
                    continue;
                }
                let shuffled = bench_meshes::shuffled(&base);
                for m in [base, shuffled] {
                    let mesh = MeshBuilder::new()
                        .positions(&m.positions)
                        .triangles(&m.triangles)
                        .patch_size(patch_size)
                        .build()
                        .unwrap();
                    chunked |= mesh.patches().iter().any(|p| p.edge_ids().len() > 1024);
                    let context = format!("{} at patch size {patch_size}", m.name);
                    check(&device, &queue, (&staged, &current), &mesh, &context);
                }
            }
        }
        assert!(chunked, "no patch exercised chunked staging");
    }

    /// Compares GPU time for three ways to build edge records: EdgeQuery,
    /// shared-memory staging with no extra editing capacity, and a flat mesh
    /// representation without patches. Requires GPU timestamp support.
    /// Run: `MOPMESH_BENCH=1 cargo test --release --lib staged_edge_benchmark -- --ignored --nocapture`.
    /// Set `MOPMESH_BENCH_FACES` to comma-separated target face counts (default
    /// `524288,2097152`) and `MOPMESH_BENCH_PATCHES` to comma-separated patch
    /// sizes (default `32,128,512`).
    #[test]
    #[ignore = "benchmark; run in release with --ignored --nocapture"]
    fn staged_edge_benchmark() {
        use crate::gpu::baseline::{FlatMesh, FlatQueries};
        use crate::gpu::bench_baseline::{Timer, WARMUP, sample, spread};
        if std::env::var_os("MOPMESH_BENCH").is_none() {
            println!("set MOPMESH_BENCH=1 to run the staged edge benchmark");
            return;
        }
        let numbers = |name: &str, default: &str| -> Vec<u32> {
            std::env::var(name)
                .unwrap_or_else(|_| default.into())
                .split(',')
                .map(|s| s.trim().parse().unwrap())
                .collect()
        };
        let faces = numbers("MOPMESH_BENCH_FACES", "524288,2097152");
        let patches = numbers("MOPMESH_BENCH_PATCHES", "32,128,512");
        const RUNS: usize = 11;

        // Use the GPU adapter's limits and enable timestamps when supported.
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        let timer = Timer::new(&device, &queue);
        println!(
            "adapter: {}; GPU time min / median / max ms",
            adapter.get_info().name
        );
        let staged = StagedEdgeQuery::new(&device);
        let current = EdgeQuery::new(&device);
        let flat_queries = FlatQueries::new(&device);

        for target in faces {
            for base in bench_meshes::generated(target) {
                let shuffled = bench_meshes::shuffled(&base);
                for m in [base, shuffled] {
                    for &patch_size in &patches {
                        let mesh = MeshBuilder::new()
                            .positions(&m.positions)
                            .triangles(&m.triangles)
                            .patch_size(patch_size as usize)
                            .build()
                            .unwrap();
                        let gpu = GpuMesh::upload(&device, &mesh).unwrap();
                        let capacity = EditCapacity::new(1.0).unwrap();
                        let editable = GpuEditableMesh::upload(&device, &mesh, capacity).unwrap();
                        let flat = FlatMesh::upload(&device, &mesh);

                        // Verify matching edge records for this mesh and patch size before timing.
                        let read =
                            |record: &dyn Fn(&mut wgpu::CommandEncoder) -> EdgeConnectivity| {
                                let mut encoder =
                                    device.create_command_encoder(&Default::default());
                                let result = record(&mut encoder);
                                test_support::read_words(&device, &queue, encoder, result.buffer())
                            };
                        assert_eq!(
                            read(&|e| staged.encode(&device, e, &mesh, &editable).unwrap()),
                            read(&|e| current.encode(&device, e, &gpu).unwrap()),
                            "{} at patch size {patch_size}",
                            m.name
                        );

                        // Measure each of the three queries once per round, discarding warmup runs.
                        let records: [&dyn Fn(&mut wgpu::CommandEncoder); 3] = [
                            &|e| drop(current.encode(&device, e, &gpu).unwrap()),
                            &|e| drop(staged.encode(&device, e, &mesh, &editable).unwrap()),
                            &|e| drop(flat_queries.edges(&device, e, &flat).unwrap()),
                        ];
                        let mut times = [const { Vec::new() }; 3];
                        for run in 0..WARMUP + RUNS {
                            for (k, record) in records.iter().enumerate() {
                                let (_, gpu_time) =
                                    sample(&device, &queue, timer.as_ref(), *record);
                                if run >= WARMUP {
                                    times[k].push(
                                        gpu_time.expect("benchmark needs GPU timestamp queries"),
                                    );
                                }
                            }
                        }
                        let [c, s, f] = times.map(|mut t| spread(&mut t));
                        let max_edges = mesh
                            .patches()
                            .iter()
                            .map(|p| p.edge_ids().len())
                            .max()
                            .unwrap();
                        let bytes = |b: u64| b as f64 / mesh.face_count() as f64;
                        println!(
                            "{} p{patch_size}: F {} | current {c} | staged {s} | flat {f} | \
                             B/face static {:.0} editable {:.0} | max held edges {max_edges} | \
                             patches {}",
                            m.name,
                            mesh.face_count(),
                            bytes(gpu.arena().size()),
                            bytes(editable.arena().size() + editable.masks().size()),
                            mesh.patches().len(),
                        );
                    }
                }
            }
        }
    }
}
