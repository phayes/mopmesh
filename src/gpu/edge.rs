use super::pipeline::{self, check_binding, workgroups};
use super::{GpuError, GpuMesh};
use crate::MeshId;

/// Edge endpoints and the faces that touch each edge.
/// The buffer contains one 16-byte WESL `EdgeRecord` per canonical edge ID:
/// two input vertex IDs in ascending order, then two input face IDs in
/// unspecified order. A boundary edge's missing face is [`super::INVALID_ID`].
/// The buffer supports `STORAGE | COPY_SRC`.
pub struct EdgeConnectivity {
    buffer: wgpu::Buffer,
    edge_count: u32,
    mesh: MeshId,
}
impl EdgeConnectivity {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
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
        let module = pipeline::shader_module(
            device,
            "MOPMesh edge connectivity",
            include_str!(concat!(env!("OUT_DIR"), "/edges.wgsl")),
        );
        Self {
            pipeline: pipeline::compute_pipeline(
                device,
                "MOPMesh edge connectivity",
                &module,
                "main",
                None,
            ),
        }
    }

    /// Records a query for each edge's endpoints and adjoining faces in `mesh`.
    /// Returns [`GpuError::LimitsExceeded`] before allocating or recording commands
    /// if the result or dispatch is too large. An empty mesh returns zero edges
    /// with a 16-byte zero placeholder buffer and records no GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> Result<EdgeConnectivity, GpuError> {
        const RESOURCE: &str = "edge connectivity";
        let limits = device.limits();
        let bytes = (u64::from(mesh.edge_count) * 16).max(16);
        check_binding(&limits, bytes, RESOURCE)?;
        let groups = workgroups(&limits, mesh.edge_count, RESOURCE)?;
        let result = EdgeConnectivity {
            buffer: pipeline::output_buffer(device, "MOPMesh edge records", bytes),
            edge_count: mesh.edge_count,
            mesh: mesh.mesh,
        };
        if mesh.edge_count != 0 {
            let bindings = pipeline::bind_buffers(
                device,
                "MOPMesh edge bindings",
                &self.pipeline.get_bind_group_layout(0),
                &[&mesh.arena, &result.buffer],
            );
            pipeline::dispatch(
                encoder,
                "MOPMesh edge connectivity",
                &self.pipeline,
                &bindings,
                groups,
            );
        }
        Ok(result)
    }
}
