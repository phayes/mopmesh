use super::pipeline::{self, group_grid};
use super::{GpuEdgeAttribute, GpuError, GpuTopology, GpuVertexAttribute, PatchBuffers, buffers};

/// Which edges and vertices lie on the mesh boundary, as `u32` flags: 1 on
/// the boundary, 0 elsewhere. A boundary edge has only one adjoining face; a
/// boundary vertex touches at least one boundary edge. A closed mesh has no
/// boundary, so every flag is 0.
pub struct MeshBoundary {
    pub edges: GpuEdgeAttribute<u32>,
    pub vertices: GpuVertexAttribute<u32>,
}

/// Finds boundary edges and vertices. Cache one per device.
pub struct BoundaryQuery {
    pipeline: wgpu::ComputePipeline,
}
impl BoundaryQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh boundary",
            include_str!(concat!(env!("OUT_DIR"), "/boundary.wgsl")),
        );
        Self {
            pipeline: pipeline::compute_pipeline(device, "MOPMesh boundary", &module, "main", None),
        }
    }

    /// Records a query for the boundary flags of every edge and vertex of
    /// `mesh`. Returns [`GpuError::LimitsExceeded`] before allocating or
    /// recording commands if a buffer or dispatch is too large. An empty mesh
    /// returns empty results without recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &impl GpuTopology,
    ) -> Result<MeshBoundary, GpuError> {
        self.encode_patches(device, encoder, buffers(mesh))
    }

    /// Records the same query as [`encode`](Self::encode), using the shared
    /// buffer layout of static and editable meshes.
    pub(crate) fn encode_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: PatchBuffers,
    ) -> Result<MeshBoundary, GpuError> {
        // Check sizes, then allocate zeroed flags.
        const RESOURCE: &str = "boundary flags";
        let limits = device.limits();
        GpuEdgeAttribute::<u32>::check_output(&limits, mesh.edge_count, RESOURCE)?;
        GpuVertexAttribute::<u32>::check_output(&limits, mesh.vertex_count, RESOURCE)?;
        let grid = group_grid(&limits, mesh.patch_count, RESOURCE)?;
        let result = MeshBoundary {
            edges: GpuEdgeAttribute::derived(
                device,
                mesh.topology,
                mesh.edge_count,
                "MOPMesh boundary edges",
            ),
            vertices: GpuVertexAttribute::derived(
                device,
                mesh.topology,
                mesh.vertex_count,
                "MOPMesh boundary vertices",
            ),
        };
        if mesh.edge_count == 0 {
            return Ok(result);
        }

        // Mark each boundary edge and its two endpoints.
        pipeline::dispatch_patches(
            device,
            encoder,
            "MOPMesh boundary",
            &self.pipeline,
            &[result.edges.buffer(), result.vertices.buffer()],
            mesh,
            grid,
        );
        Ok(result)
    }
}
