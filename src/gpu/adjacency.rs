use super::pipeline::{self, WORKGROUP_SIZE, check_binding, group_grid, workgroups};
use super::{EdgeConnectivity, GpuError, GpuMesh, PatchBuffers, scan};
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

/// Computes vertex neighbors, face neighbors, edge diamonds and edge records
/// together on the GPU; see [`MeshAdjacency`] for each result's layout.
/// Create once per device and reuse it for that device's meshes.
pub struct AdjacencyQuery {
    edge_pipeline: wgpu::ComputePipeline,
    scan: scan::ScanPipeline,
    vertex_layout: wgpu::BindGroupLayout,
    vertex_pipelines: [wgpu::ComputePipeline; 3],
}
impl AdjacencyQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh edge adjacency",
            include_str!(concat!(env!("OUT_DIR"), "/edges.wgsl")),
        );
        let edge_pipeline = pipeline::compute_pipeline(
            device,
            "MOPMesh edge adjacency",
            &module,
            "adjacency",
            None,
        );
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
        Self {
            edge_pipeline,
            scan: scan::ScanPipeline::new(device),
            vertex_layout,
            vertex_pipelines,
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
        self.encode_patches(device, encoder, mesh.buffers())
    }

    /// Records the same queries as [`encode`](Self::encode), using the
    /// shared buffer layout of static and editable meshes.
    pub(crate) fn encode_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: PatchBuffers,
    ) -> Result<MeshAdjacency, GpuError> {
        // Validate every size before allocating.
        let limits = device.limits();
        let plan =
            AdjacencyPlan::new(&limits, mesh.vertex_count, mesh.face_count, mesh.edge_count)?;
        let grid = group_grid(&limits, mesh.patch_count, AdjacencyPlan::RESOURCE)?;
        let edges = EdgeConnectivity {
            buffer: pipeline::output_buffer(device, "MOPMesh edge records", plan.edge_bytes),
            edge_count: mesh.edge_count,
            mesh: mesh.mesh,
        };
        let result = Self::outputs(device, &plan, edges);
        if plan.edge_count == 0 {
            return Ok(result);
        }

        // Write edge records, face neighbors and diamonds in one pass per
        // patch, then build the vertex neighbor lists from the edge records.
        pipeline::dispatch_patches(
            device,
            encoder,
            "MOPMesh edge adjacency",
            &self.edge_pipeline,
            &[result.edges.buffer(), &result.faces, &result.diamonds],
            mesh,
            grid,
        );
        self.record_vertex_neighbors(device, encoder, &plan, &result);
        Ok(result)
    }

    /// Allocates zeroed results for the mesh `plan` was checked for.
    /// `edges` becomes the result's edge records; its contents may be
    /// recorded before or after this call.
    pub(super) fn outputs(
        device: &wgpu::Device,
        plan: &AdjacencyPlan,
        edges: EdgeConnectivity,
    ) -> MeshAdjacency {
        let vertices = pipeline::output_buffer(
            device,
            "MOPMesh vertex edge/neighbor CSR",
            plan.vertex_bytes,
        );
        let faces = pipeline::output_buffer(device, "MOPMesh face neighbors", plan.face_bytes);
        let diamonds = pipeline::output_buffer(device, "MOPMesh edge diamonds", plan.edge_bytes);
        MeshAdjacency {
            edges,
            vertices,
            faces,
            diamonds,
            vertex_count: plan.vertex_count,
            face_count: plan.face_count,
        }
    }

    /// Records the vertex neighbor lists of `result` from its edge records,
    /// which must be recorded first. Records no GPU work for a mesh without
    /// edges.
    pub(super) fn record_vertex_neighbors(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        plan: &AdjacencyPlan,
        result: &MeshAdjacency,
    ) {
        if plan.edge_count == 0 {
            return;
        }
        let edge_groups = plan.edge_groups;
        let vertex_groups = plan.vertex_count.div_ceil(WORKGROUP_SIZE);

        // Build a neighbor list for each vertex: count its edges, turn the counts
        // into row offsets, then fill the rows with edge and neighbor vertex IDs.
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh vertex neighbors sizes"),
            contents: bytemuck::cast_slice(&[plan.vertex_count, plan.edge_count]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh vertex neighbors cursors"),
            size: plan.scan.byte_size(),
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
        self.scan.encode(device, encoder, &cursors, &plan.scan);
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
    }
}

/// Result sizes and dispatch counts for one [`AdjacencyQuery`] run, checked
/// against the device limits.
pub(super) struct AdjacencyPlan {
    scan: scan::ScanPlan,
    vertex_bytes: u64,
    face_bytes: u64,
    edge_bytes: u64,
    edge_groups: u32,
    vertex_count: u32,
    face_count: u32,
    edge_count: u32,
}
impl AdjacencyPlan {
    /// The resource named in this query's limit errors.
    pub(super) const RESOURCE: &'static str = "mesh adjacency";

    /// Checks the results, temporary buffers and dispatches for a mesh with
    /// these counts. Returns [`GpuError::LimitsExceeded`] if any is too large.
    pub(super) fn new(
        limits: &wgpu::Limits,
        vertex_count: u32,
        face_count: u32,
        edge_count: u32,
    ) -> Result<Self, GpuError> {
        const RESOURCE: &str = AdjacencyPlan::RESOURCE;
        let scan = scan::ScanPlan::new(vertex_count, limits)?;
        let vertex_bytes = 4 * (u64::from(vertex_count) + 1 + 4 * u64::from(edge_count));
        let face_bytes = (12 * u64::from(face_count)).max(4);
        let edge_bytes = (16 * u64::from(edge_count)).max(16);
        for bytes in [vertex_bytes, face_bytes, edge_bytes] {
            check_binding(limits, bytes, RESOURCE)?;
        }
        let edge_groups = workgroups(limits, edge_count, RESOURCE)?;
        Ok(Self {
            scan,
            vertex_bytes,
            face_bytes,
            edge_bytes,
            edge_groups,
            vertex_count,
            face_count,
            edge_count,
        })
    }
}
