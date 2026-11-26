use super::pipeline::{self, WORKGROUP_SIZE, check_binding, workgroups};
use super::{EdgeConnectivity, EdgeQuery, FaceVertexQuery, GpuError, GpuMesh, scan};
use crate::MeshId;
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
    /// Returns three packed `u32` neighbor face IDs per input face (12 bytes per face).
    /// For input triangle `[a, b, c]`, the entries are the faces across edges
    /// `(a, b)`, `(b, c)`, and `(c, a)`. A boundary edge has no neighbor and
    /// uses [`super::INVALID_ID`].
    pub fn face_neighbors(&self) -> &wgpu::Buffer {
        &self.faces
    }
    /// Returns four packed `u32` vertex IDs per canonical edge (16 bytes per edge):
    /// `[endpoint0, endpoint1, opposite0, opposite1]`. An opposite vertex is the
    /// third vertex of an adjoining triangle, outside the shared edge.
    /// The opposite vertices follow the face order in [`Self::edge_connectivity`];
    /// a missing boundary face gives [`super::INVALID_ID`]. The four vertices
    /// are not ordered around the edge.
    pub fn edge_diamonds(&self) -> &wgpu::Buffer {
        &self.diamonds
    }
    pub fn edge_connectivity(&self) -> &EdgeConnectivity {
        &self.edges
    }
    pub fn mesh_id(&self) -> MeshId {
        self.edges.mesh_id()
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
        let module = pipeline::shader_module(
            device,
            "MOPMesh vertex neighbors",
            include_str!(concat!(env!("OUT_DIR"), "/vertex_neighbors.wgsl")),
        );
        let (vertex_layout, layout) = pipeline::buffer_layout(
            device,
            "MOPMesh vertex neighbors layout",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let vertex_pipelines = [
            ("MOPMesh vertex neighbors count", "count"),
            ("MOPMesh vertex neighbors initialize", "initialize"),
            ("MOPMesh vertex neighbors scatter", "scatter"),
        ]
        .map(|(label, entry)| {
            pipeline::compute_pipeline(device, label, &module, entry, Some(&layout))
        });
        let module = pipeline::shader_module(
            device,
            "MOPMesh face neighbors and diamonds",
            include_str!(concat!(env!("OUT_DIR"), "/face_neighbors.wgsl")),
        );
        let face_pipeline = pipeline::compute_pipeline(
            device,
            "MOPMesh face neighbors and diamonds",
            &module,
            "main",
            None,
        );
        Self {
            edges: EdgeQuery::new(device),
            faces: FaceVertexQuery::new(device),
            scan: scan::ScanPipeline::new(device),
            vertex_layout,
            vertex_pipelines,
            face_pipeline,
        }
    }

    /// Records queries for vertex neighbors, neighboring faces, edge endpoints,
    /// and opposite vertices. See [`MeshAdjacency`] for each result's buffer layout.
    /// Returns [`GpuError::LimitsExceeded`] before allocating or recording commands
    /// if a required buffer or dispatch is too large. An empty mesh records no
    /// GPU work: vertex offsets contain `[0]`, and the face, diamond, and edge
    /// buffers contain zero placeholders of 4, 16, and 16 bytes respectively.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> Result<MeshAdjacency, GpuError> {
        // Validate sizes for results and temporary buffers before allocating or recording work.
        const RESOURCE: &str = "mesh adjacency";
        let limits = device.limits();
        let plan = scan::ScanPlan::new(mesh.vertex_count, &limits)?;
        let vertex_bytes = 4 * (u64::from(mesh.vertex_count) + 1 + 4 * u64::from(mesh.edge_count));
        let face_bytes = (12 * u64::from(mesh.face_count)).max(4);
        let edge_bytes = (16 * u64::from(mesh.edge_count)).max(16);
        for bytes in [vertex_bytes, face_bytes, edge_bytes] {
            check_binding(&limits, bytes, RESOURCE)?;
        }
        let edge_groups = workgroups(&limits, mesh.edge_count, RESOURCE)?;
        workgroups(&limits, mesh.face_count, RESOURCE)?;
        let vertex_groups = mesh.vertex_count.div_ceil(WORKGROUP_SIZE);

        let vertices =
            pipeline::output_buffer(device, "MOPMesh vertex edge/neighbor CSR", vertex_bytes);
        let faces = pipeline::output_buffer(device, "MOPMesh face neighbors", face_bytes);
        let diamonds = pipeline::output_buffer(device, "MOPMesh edge diamonds", edge_bytes);
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

        // For each edge, find the neighboring faces and their opposite vertices.
        // Write the neighbor entry for each face that touches the edge.
        let fv = self.faces.encode(device, encoder, mesh)?;
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh face neighbors bindings",
            &self.face_pipeline.get_bind_group_layout(0),
            &[
                result.edges.buffer(),
                fv.buffer(),
                &result.faces,
                &result.diamonds,
            ],
        );
        pipeline::dispatch(
            encoder,
            "MOPMesh face neighbors and diamonds",
            &self.face_pipeline,
            &bindings,
            edge_groups,
        );

        // Build a neighbor list for each vertex: count its edges, turn the counts
        // into row offsets, then fill the rows with edge and neighbor vertex IDs.
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh vertex neighbors sizes"),
            contents: bytemuck::cast_slice(&[mesh.vertex_count, mesh.edge_count]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh vertex neighbors cursors"),
            size: plan.byte_size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh vertex neighbors bindings",
            &self.vertex_layout,
            &[&params, result.edges.buffer(), &cursors, &result.vertices],
        );
        let [count, initialize, scatter] = &self.vertex_pipelines;
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex neighbors count",
            count,
            &bindings,
            edge_groups,
        );
        self.scan.encode(device, encoder, &cursors, &plan);
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex neighbors initialize",
            initialize,
            &bindings,
            vertex_groups,
        );
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex neighbors scatter",
            scatter,
            &bindings,
            edge_groups,
        );
        Ok(result)
    }
}
