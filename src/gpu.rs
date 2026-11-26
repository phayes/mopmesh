//! Immutable GPU connectivity and explicit, caller-encoded topology queries.
use crate::Mesh;
use wgpu::util::DeviceExt;
mod adjacency;
mod attribute;
mod edge;
mod scan;
pub use adjacency::{AdjacencyQuery, MeshAdjacency};
pub use attribute::{FaceCentroidQuery, GpuAttributeError, GpuValue, GpuVertexAttribute};
pub use edge::{EdgeConnectivity, EdgeQuery};

/// A mesh exceeded the scalar arena, binding, or dispatch limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLimitError;

impl std::fmt::Display for GpuLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("mesh exceeds GPU arena, binding, or dispatch limits")
    }
}
impl std::error::Error for GpuLimitError {}

/// Static connectivity, including ribbons, uploaded to one read-only arena.
///
/// The caller owns the device, queue, submission, and optional readback.
pub struct GpuMesh {
    arena: wgpu::Buffer,
    face_count: u32,
    vertex_count: u32,
    edge_count: u32,
    identity: std::sync::Arc<()>,
}

impl GpuMesh {
    pub fn upload(device: &wgpu::Device, mesh: &Mesh) -> Result<Self, GpuLimitError> {
        let limits = device.limits();
        let count = u32::try_from(mesh.face_count()).map_err(|_| GpuLimitError)?;
        let vertex_count = u32::try_from(mesh.positions().len()).map_err(|_| GpuLimitError)?;
        let edge_count = u32::try_from(mesh.edge_count()).map_err(|_| GpuLimitError)?;
        let mut word_count = 4u64 + 4 * u64::from(count) + 7 * u64::from(edge_count);
        for p in mesh.patches() {
            word_count = word_count
                .checked_add(p.vertex_ids().len() as u64)
                .and_then(|n| n.checked_add(2 * p.edge_vertices().len() as u64))
                .and_then(|n| n.checked_add(7 * p.face_edges().len() as u64))
                .ok_or(GpuLimitError)?;
        }
        let max_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if word_count > u64::from(u32::MAX)
            || word_count * 4 > max_bytes
            || (u64::from(count) * 12).max(4) > max_bytes
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(GpuLimitError);
        }
        let mut words = Vec::with_capacity(word_count as usize);
        words.extend([count, 0, edge_count, 0]);
        let mut tasks = Vec::with_capacity(count as usize);
        let mut edge_tasks = Vec::with_capacity(edge_count as usize);
        for p in mesh.patches() {
            let vertex_offset = words.len() as u32;
            words.extend_from_slice(p.vertex_ids());
            let ev_offset = words.len() as u32;
            for edge in p.edge_vertices() {
                words.extend_from_slice(edge);
            }
            let faces_offset = words.len() as u32;
            for (local, face) in p.face_edges().iter().enumerate() {
                let face_offset = words.len() as u32;
                for edge in face {
                    words.extend([edge.edge, u32::from(edge.reversed)]);
                }
                if p.face_owners()[local].patch == p.id() {
                    tasks.push([ev_offset, face_offset, vertex_offset, p.face_ids()[local]]);
                }
            }
            let face_ids_offset = words.len() as u32;
            words.extend_from_slice(p.face_ids());
            for (local, owner) in p.edge_owners().iter().enumerate() {
                if owner.patch == p.id() {
                    edge_tasks.push([
                        local as u32,
                        p.face_ids().len() as u32,
                        faces_offset,
                        face_ids_offset,
                        p.edge_ids()[local],
                        vertex_offset,
                        ev_offset,
                    ]);
                }
            }
        }
        words[1] = words.len() as u32;
        for task in tasks {
            words.extend(task);
        }
        words[3] = words.len() as u32;
        for task in edge_tasks {
            words.extend(task);
        }
        let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh static connectivity"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(Self {
            arena,
            face_count: count,
            vertex_count,
            edge_count,
            identity: mesh.identity.clone(),
        })
    }

    /// Scalar arena header: [face count, face task offset, edge count, edge task offset].
    /// Face tasks: [EV word offset, FE word offset, vertex word offset, global face ID].
    /// Edge tasks: [local edge, local face count, FE base, face-ID base,
    /// global edge ID, vertex-ID base, EV base]. Patch data also stores global face IDs.
    /// EV stores two local vertex IDs per edge; FE stores three pairs of
    /// (local edge ID, reversed 0/1) per face; vertices stores global IDs.
    /// All offsets are measured in u32 words. This ABI is experimental.
    pub fn arena(&self) -> &wgpu::Buffer {
        &self.arena
    }
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }
}

/// Compile once per device and reuse across meshes uploaded to that device.
pub struct FaceVertexQuery {
    pipeline: wgpu::ComputePipeline,
}

impl FaceVertexQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh face vertices"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/face_vertices.wgsl")).into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MOPMesh face vertices"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Record one invocation per owned face. Returns tightly packed u32 triples
    /// indexed by original input face ID, with STORAGE and COPY_SRC usage.
    /// The empty result has a four-byte placeholder and dispatches no work.
    /// All resources and the encoder must belong to the same device.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> wgpu::Buffer {
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh face vertices result"),
            size: (u64::from(mesh.face_count) * 12).max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if mesh.face_count == 0 {
            return output;
        }
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MOPMesh face query bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh.arena.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MOPMesh face query"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(mesh.face_count.div_ceil(64), 1, 1);
        }
        output
    }
}

/// GPU CSR adjacency indexed by original input vertex ID.
///
/// The buffer contains V + 1 u32 offsets followed by 3F u32 face IDs. Offsets
/// are relative to the face-ID region. A vertex's incident faces occupy
/// offsets[v]..offsets[v + 1]; ordering within each row is unspecified.
/// The buffer has STORAGE and COPY_SRC usage and remains on the GPU.
pub struct VertexFaceAdjacency {
    buffer: wgpu::Buffer,
    vertex_count: u32,
    entry_count: u32,
}

impl VertexFaceAdjacency {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    pub fn entry_count(&self) -> u32 {
        self.entry_count
    }
    pub fn entries_byte_offset(&self) -> u64 {
        (u64::from(self.vertex_count) + 1) * 4
    }
}

/// Transpose face vertices on the GPU using count, prefix, and scatter passes.
/// Cache once per device. Prefixes use a hierarchical workgroup scan with
/// recursive block sums. No host adjacency or GPU readback is used.
pub struct VertexFaceQuery {
    face_vertices: FaceVertexQuery,
    scan: scan::ScanPipeline,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
}

impl VertexFaceQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh vertex faces"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/vertex_faces.wgsl")).into(),
            ),
        });
        // A shared explicit layout allows all three entry points to bind the
        // same resources even when an entry point uses only a subset.
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MOPMesh vertex face layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh vertex face pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["count", "initialize", "scatter"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self {
            face_vertices: FaceVertexQuery::new(device),
            scan: scan::ScanPipeline::new(device),
            layout,
            pipelines,
        }
    }

    /// Record FV evaluation, incidence counts, hierarchical prefix scans,
    /// offset initialization, and scatter passes. Limits are
    /// checked before allocating or recording anything. Requires three storage
    /// bindings, one uniform binding, and 64 invocations per workgroup.
    /// Empty meshes return a single zero offset and dispatch no work.
    /// All resources must belong to the same device.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> Result<VertexFaceAdjacency, GpuLimitError> {
        let entries = u64::from(mesh.face_count) * 3;
        let words = u64::from(mesh.vertex_count) + 1 + entries;
        let limits = device.limits();
        let scan_plan = scan::ScanPlan::new(mesh.vertex_count, &limits)?;
        if words > u64::from(u32::MAX)
            || words * 4
                > limits
                    .max_buffer_size
                    .min(limits.max_storage_buffer_binding_size)
        {
            return Err(GpuLimitError);
        }
        // WebGPU initializes fresh buffers to zero, including the cursor counts.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh vertex face CSR"),
            size: words * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let result = VertexFaceAdjacency {
            buffer,
            vertex_count: mesh.vertex_count,
            entry_count: entries as u32,
        };
        if mesh.face_count == 0 {
            return Ok(result);
        }
        let faces = self.face_vertices.encode(device, encoder, mesh);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh transpose sizes"),
            contents: bytemuck::cast_slice(&[mesh.vertex_count, mesh.face_count]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh transpose cursors"),
            size: scan_plan.byte_size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buffers = [&params, &faces, &cursors, &result.buffer];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MOPMesh transpose bindings"),
            layout: &self.layout,
            entries: &entries,
        });
        for (index, (pipeline, workgroups)) in self
            .pipelines
            .iter()
            .zip([
                mesh.face_count.div_ceil(64),
                mesh.vertex_count.div_ceil(64),
                mesh.face_count.div_ceil(64),
            ])
            .enumerate()
        {
            if index == 1 {
                self.scan.encode(device, encoder, &cursors, &scan_plan);
            }
            // Separate passes establish visibility between counts, prefix, and scatter.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MOPMesh transpose pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        Ok(result)
    }
}
