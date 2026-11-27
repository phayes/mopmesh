//! Application integration, shared with tests so they execute the actual example.
use mopmesh::gpu::{GpuMesh, GpuVertexAttribute, MeshAdjacency};

pub fn compile_shader() -> String {
    let mut resolver = wesl::resolver::StandardResolver::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/shaders"
    ));
    resolver.add_package(&mopmesh::mopmesh::PACKAGE);
    wesl::Compiler::default()
        .with_resolver(resolver)
        .compile_module(&"package::bending".parse().unwrap())
        .expect("link downstream bending operator")
        .to_string()
}

/// A downstream operator that rates how much the surface bends at each
/// vertex, from library vertex neighbors and vertex normals.
pub struct BendingOperator {
    pipeline: wgpu::ComputePipeline,
}
impl BendingOperator {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downstream bending"),
            source: wgpu::ShaderSource::Wgsl(compile_shader().into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("downstream bending"),
                layout: None,
                module: &module,
                entry_point: Some("bending_metric"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }

    // Record one bending value per vertex. The result is a library vertex
    // attribute, so library queries such as reductions accept it. All inputs
    // must belong to `mesh`; mismatches panic. `mesh` must not be empty.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        adjacency: &MeshAdjacency,
        positions: &GpuVertexAttribute<[f32; 4]>,
        normals: &GpuVertexAttribute<[f32; 4]>,
    ) -> GpuVertexAttribute<f32> {
        assert_eq!(adjacency.mesh_id(), mesh.mesh_id());
        assert_eq!(positions.mesh_id(), mesh.mesh_id());
        assert_eq!(normals.mesh_id(), mesh.mesh_id());
        let vertices = mesh.vertex_count();
        assert!(vertices > 0, "the bending example needs a non-empty mesh");
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bending result"),
            size: u64::from(vertices) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffers = [
            adjacency.vertex_neighbors(),
            positions.buffer(),
            normals.buffer(),
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
            label: Some("bending bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bending operator"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(vertices.div_ceil(64), 1, 1);
        }
        GpuVertexAttribute::from_buffer(mesh, &output).expect("bending output fits the mesh")
    }
}
