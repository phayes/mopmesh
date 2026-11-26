//! Downstream VE/VV composition, shared with GPU result tests.
use mopmesh::gpu::{GpuVertexAttribute, MeshAdjacency};

pub struct NeighborhoodOperator {
    pipeline: wgpu::ComputePipeline,
}
impl NeighborhoodOperator {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut resolver = wesl::resolver::StandardResolver::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/shaders"
        ));
        resolver.add_package(&mopmesh::mopmesh::PACKAGE);
        let compiled = wesl::Compiler::default()
            .with_resolver(resolver)
            .compile_module(&"package::neighborhood".parse().unwrap())
            .expect("link downstream neighborhood operator");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downstream neighborhood"),
            source: wgpu::ShaderSource::Wgsl(compiled.to_string().into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("downstream neighborhood"),
                layout: None,
                module: &module,
                entry_point: Some("neighborhood"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }
    // Record four values per vertex: number of edges touching it, number of those
    // edges on the boundary, mean edge length, and weight times mean edge length.
    // Adjacency, positions, and weights must have the same mesh ID; a mismatch panics.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        adjacency: &MeshAdjacency,
        positions: &GpuVertexAttribute<[f32; 4]>,
        weights: &GpuVertexAttribute<f32>,
    ) -> wgpu::Buffer {
        assert_eq!(adjacency.mesh_id(), positions.mesh_id());
        assert_eq!(adjacency.mesh_id(), weights.mesh_id());
        let size = (u64::from(adjacency.vertex_count()) * 16).max(16);
        let limits = device.limits();
        assert!(
            size <= limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size)
        );
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("neighborhood result"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if adjacency.vertex_count() == 0 {
            return output;
        }
        let buffers = [
            adjacency.vertex_neighbors(),
            adjacency.edge_connectivity().buffer(),
            positions.buffer(),
            weights.buffer(),
            &output,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("neighborhood application bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("neighborhood operator"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(adjacency.vertex_count().div_ceil(64), 1, 1);
        }
        output
    }
}
