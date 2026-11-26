//! Downstream VE/VV composition, shared with GPU result tests.
use mopmesh::gpu::{GpuVertexAttribute, MeshAdjacency};
use wgpu::util::DeviceExt;

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
    // Caller supplies matching mesh resources. Output lanes: degree, boundary
    // incident edges, mean edge length, application weight times mean length.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        adjacency: &MeshAdjacency,
        positions: &GpuVertexAttribute<[f32; 4]>,
        weights: &[f32],
    ) -> wgpu::Buffer {
        assert_eq!(weights.len(), adjacency.vertex_count() as usize);
        assert_eq!(positions.len(), adjacency.vertex_count());
        let size = (weights.len() as u64 * 16).max(16);
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
        if weights.is_empty() {
            return output;
        }
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("application vertex weights"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffers = [
            adjacency.vertex_neighbors(),
            adjacency.edge_connectivity().buffer(),
            positions.buffer(),
            &weights,
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
