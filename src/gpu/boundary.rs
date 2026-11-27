use super::pipeline::{self, workgroups};
use super::{EdgeQuery, GpuEdgeAttribute, GpuError, GpuMesh, GpuVertexAttribute};
use wgpu::util::DeviceExt;

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
    edges: EdgeQuery,
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
            edges: EdgeQuery::new(device),
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
        mesh: &GpuMesh,
    ) -> Result<MeshBoundary, GpuError> {
        // Check sizes, then allocate zeroed flags.
        const RESOURCE: &str = "boundary flags";
        let limits = device.limits();
        GpuEdgeAttribute::<u32>::check_output(&limits, mesh.edge_count, RESOURCE)?;
        GpuVertexAttribute::<u32>::check_output(&limits, mesh.vertex_count, RESOURCE)?;
        let groups = workgroups(&limits, mesh.edge_count, RESOURCE)?;
        let result = MeshBoundary {
            edges: GpuEdgeAttribute::output(
                device,
                mesh.mesh,
                mesh.edge_count,
                "MOPMesh boundary edges",
            ),
            vertices: GpuVertexAttribute::output(
                device,
                mesh.mesh,
                mesh.vertex_count,
                "MOPMesh boundary vertices",
            ),
        };
        if mesh.edge_count == 0 {
            return Ok(result);
        }

        // Mark each boundary edge and its two endpoints.
        let edges = self.edges.encode(device, encoder, mesh)?;
        // Padded to 16 bytes, the minimum uniform struct size.
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh boundary sizes"),
            contents: bytemuck::cast_slice(&[mesh.edge_count, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh boundary bindings",
            &self.pipeline.get_bind_group_layout(0),
            &[
                &params,
                edges.buffer(),
                result.edges.buffer(),
                result.vertices.buffer(),
            ],
        );
        pipeline::dispatch(
            encoder,
            "MOPMesh boundary",
            &self.pipeline,
            &bindings,
            groups,
        );
        Ok(result)
    }
}
