use super::{GpuLimitError, GpuMesh};

/// One 16-byte WESL EdgeRecord per canonical edge ID: two global vertex IDs
/// followed by two global face IDs. A missing face is u32::MAX. Incident-face
/// order is unspecified; endpoint IDs are ascending. Usage: STORAGE | COPY_SRC.
pub struct EdgeConnectivity {
    buffer: wgpu::Buffer,
    edge_count: u32,
}
impl EdgeConnectivity {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }
}

/// Evaluate EV and EF for each owned edge, including faces across patch borders.
/// The current kernel scans each owner patch's local faces. This is a correctness
/// baseline; it does not yet stage topology in workgroup memory.
pub struct EdgeQuery {
    pipeline: wgpu::ComputePipeline,
}
impl EdgeQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh edge connectivity"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/edges.wgsl")).into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MOPMesh edge connectivity"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Record into a caller-owned encoder. Checks output and dispatch limits
    /// before allocating or recording anything. Uses two storage bindings and
    /// 64 invocations per workgroup. All resources must belong to the same device.
    /// Empty meshes return a 16-byte zero placeholder without dispatch.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> Result<EdgeConnectivity, GpuLimitError> {
        let size = (u64::from(mesh.edge_count) * 16).max(16);
        let limits = device.limits();
        if size
            > limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size)
            || mesh.edge_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(GpuLimitError);
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh edge records"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if mesh.edge_count != 0 {
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MOPMesh edge bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: mesh.arena.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MOPMesh edge query"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(mesh.edge_count.div_ceil(64), 1, 1);
        }
        Ok(EdgeConnectivity {
            buffer,
            edge_count: mesh.edge_count,
        })
    }
}
