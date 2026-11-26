use super::{EdgeConnectivity, EdgeQuery, FaceVertexQuery, GpuLimitError, GpuMesh, scan};
use wgpu::util::DeviceExt;

/// Batched VE, VV, FF and edge-diamond results. All buffers have STORAGE and
/// COPY_SRC usage. IDs are canonical; resources belong to the supplied device.
pub struct MeshAdjacency {
    edges: EdgeConnectivity,
    vertices: wgpu::Buffer,
    faces: wgpu::Buffer,
    diamonds: wgpu::Buffer,
    vertex_count: u32,
    face_count: u32,
}
impl MeshAdjacency {
    /// V+1 u32 offsets followed by 2E pairs (edge ID, neighbor vertex ID).
    /// Offsets count pairs, not words. Row order is unspecified, not cyclic.
    /// Each edge contributes one pair to each endpoint's row.
    pub fn vertex_neighbors(&self) -> &wgpu::Buffer {
        &self.vertices
    }
    /// Three u32 neighbor face IDs per input face. Slot i is across the edge
    /// (input_triangle[i], input_triangle[(i+1)%3]); u32::MAX means boundary.
    pub fn face_neighbors(&self) -> &wgpu::Buffer {
        &self.faces
    }
    /// Four u32 vertex IDs per edge: [endpoint0, endpoint1, opposite0, opposite1].
    /// Opposites correspond to edge_connectivity().faces slots; u32::MAX means
    /// no face on that side. This is not a cyclic or orientation-sorted quartet.
    pub fn edge_diamonds(&self) -> &wgpu::Buffer {
        &self.diamonds
    }
    pub fn edge_connectivity(&self) -> &EdgeConnectivity {
        &self.edges
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
    pub fn edge_count(&self) -> u32 {
        self.edges.edge_count()
    }
    pub fn vertex_entry_count(&self) -> u32 {
        2 * self.edge_count()
    }
    pub fn vertex_entries_byte_offset(&self) -> u64 {
        4 * (u64::from(self.vertex_count) + 1)
    }
}

/// Compute the remaining static adjacency families together, sharing EV/EF and
/// FV intermediates. Cache once per device. VE/VV use a global GPU transpose;
/// FF and diamonds compose EV/EF with FV. No CPU adjacency or readback is used.
pub struct AdjacencyQuery {
    edges: EdgeQuery,
    faces: FaceVertexQuery,
    scan: scan::ScanPipeline,
    vertex_layout: wgpu::BindGroupLayout,
    vertex_pipelines: [wgpu::ComputePipeline; 3],
    face_pipeline: wgpu::ComputePipeline,
}
impl AdjacencyQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh vertex edge/neighbor adjacency"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/vertex_neighbors.wgsl")).into(),
            ),
        });
        let entries = [
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
            wgpu::BufferBindingType::Storage { read_only: false },
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect::<Vec<_>>();
        let vertex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MOPMesh vertex adjacency layout"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh vertex adjacency pipeline layout"),
            bind_group_layouts: &[Some(&vertex_layout)],
            immediate_size: 0,
        });
        let vertex_pipelines = ["count", "initialize", "scatter"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh face neighbors and diamonds"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/face_neighbors.wgsl")).into(),
            ),
        });
        let face_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MOPMesh face neighbors and diamonds"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            edges: EdgeQuery::new(device),
            faces: FaceVertexQuery::new(device),
            scan: scan::ScanPipeline::new(device),
            vertex_layout,
            vertex_pipelines,
            face_pipeline,
        }
    }

    /// Checks all new buffer, scalar-index, and dispatch limits before recording.
    /// Requires four storage bindings, one uniform binding and 64 lanes. Use one
    /// device for all resources. Empty meshes dispatch nothing; CSR is [0],
    /// FF has a four-byte zero placeholder, and diamonds/edges each have 16 bytes.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> Result<MeshAdjacency, GpuLimitError> {
        let limits = device.limits();
        let plan = scan::ScanPlan::new(mesh.vertex_count, &limits)?;
        let vertex_words = u64::from(mesh.vertex_count) + 1 + 4 * u64::from(mesh.edge_count);
        let face_bytes = (12 * u64::from(mesh.face_count)).max(4);
        let edge_bytes = (16 * u64::from(mesh.edge_count)).max(16);
        let max_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if vertex_words > u64::from(u32::MAX)
            || vertex_words * 4 > max_bytes
            || face_bytes > max_bytes
            || edge_bytes > max_bytes
            || mesh.edge_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(GpuLimitError);
        }
        let allocate = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let vertices = allocate("MOPMesh vertex edge/neighbor CSR", vertex_words * 4);
        let faces = allocate("MOPMesh face neighbors", face_bytes);
        let diamonds = allocate("MOPMesh edge diamonds", edge_bytes);
        let edges = self.edges.encode(device, encoder, mesh)?;
        let result = MeshAdjacency {
            edges,
            vertices,
            faces,
            diamonds,
            vertex_count: mesh.vertex_count,
            face_count: mesh.face_count,
        };
        if mesh.edge_count == 0 {
            return Ok(result);
        }
        let fv = self.faces.encode(device, encoder, mesh);
        let bindings = bind_buffers(
            device,
            &self.face_pipeline.get_bind_group_layout(0),
            &[result.edges.buffer(), &fv, &result.faces, &result.diamonds],
        );
        dispatch(
            encoder,
            &self.face_pipeline,
            &bindings,
            mesh.edge_count.div_ceil(64),
        );

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh vertex adjacency sizes"),
            contents: bytemuck::cast_slice(&[mesh.vertex_count, mesh.edge_count]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh vertex adjacency counts/scan"),
            size: plan.byte_size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bindings = bind_buffers(
            device,
            &self.vertex_layout,
            &[&params, result.edges.buffer(), &cursors, &result.vertices],
        );
        dispatch(
            encoder,
            &self.vertex_pipelines[0],
            &bindings,
            mesh.edge_count.div_ceil(64),
        );
        self.scan.encode(device, encoder, &cursors, &plan);
        dispatch(
            encoder,
            &self.vertex_pipelines[1],
            &bindings,
            mesh.vertex_count.div_ceil(64),
        );
        dispatch(
            encoder,
            &self.vertex_pipelines[2],
            &bindings,
            mesh.edge_count.div_ceil(64),
        );
        Ok(result)
    }
}

fn bind_buffers(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MOPMesh adjacency bindings"),
        layout,
        entries: &entries,
    })
}
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    groups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MOPMesh adjacency pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}
