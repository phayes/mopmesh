//! Application integration, shared with tests so they execute the actual example.
use mopmesh::gpu::{EdgeConnectivity, GpuVertexAttribute};
use wgpu::util::DeviceExt;

pub fn compile_shader() -> String {
    let mut resolver = wesl::resolver::StandardResolver::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/shaders"
    ));
    resolver.add_package(&mopmesh::mopmesh::PACKAGE);
    wesl::Compiler::default()
        .with_resolver(resolver)
        .compile_module(&"package::weighted_boundary".parse().unwrap())
        .expect("link downstream weighted-boundary operator")
        .to_string()
}

pub struct BoundaryOperator {
    pipeline: wgpu::ComputePipeline,
}
impl BoundaryOperator {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downstream weighted boundary"),
            source: wgpu::ShaderSource::Wgsl(compile_shader().into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("downstream weighted boundary"),
                layout: None,
                module: &module,
                entry_point: Some("weighted_boundary"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }

    // Record one [weighted boundary length, adjoining-face count] result per edge.
    // Edge records and positions must have the same mesh ID, and weights must
    // contain one value per canonical edge ID. Invalid inputs panic.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        edges: &EdgeConnectivity,
        positions: &GpuVertexAttribute<[f32; 4]>,
        weights: &[f32],
    ) -> wgpu::Buffer {
        assert_eq!(edges.mesh_id(), positions.mesh_id());
        assert_eq!(weights.len(), edges.edge_count() as usize);
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weighted boundary result"),
            size: (weights.len() as u64 * 8).max(8),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if weights.is_empty() {
            return output;
        }
        let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("application edge weights"),
            contents: bytemuck::cast_slice(weights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffers = [edges.buffer(), positions.buffer(), &weights, &output];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("application bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("weighted boundary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(edges.edge_count().div_ceil(64), 1, 1);
        }
        output
    }
}
