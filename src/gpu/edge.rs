use super::pipeline::{self, check_binding, group_grid};
use super::{GpuError, GpuMesh, PatchBuffers};
use crate::MeshId;

/// Edge endpoints and the faces that touch each edge.
/// The buffer contains one 16-byte WESL `EdgeRecord` per canonical edge ID:
/// two input vertex IDs in ascending order, then two input face IDs in
/// unspecified order. A boundary edge's missing face is [`super::INVALID_ID`].
/// The buffer supports `STORAGE | COPY_SRC`.
pub struct EdgeConnectivity {
    pub(super) buffer: wgpu::Buffer,
    pub(super) edge_count: u32,
    pub(super) mesh: MeshId,
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

/// Queries each edge's two endpoints and adjoining faces, including faces
/// across patch boundaries. Create once per device and reuse for its meshes.
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
        self.encode_patches(device, encoder, mesh.buffers())
    }

    /// Records the same edge query as [`encode`](Self::encode), using the
    /// shared buffer layout of static and editable meshes.
    pub(crate) fn encode_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: PatchBuffers,
    ) -> Result<EdgeConnectivity, GpuError> {
        const RESOURCE: &str = "edge connectivity";
        let limits = device.limits();
        let bytes = (u64::from(mesh.edge_count) * 16).max(16);
        check_binding(&limits, bytes, RESOURCE)?;
        let grid = group_grid(&limits, mesh.patch_count, RESOURCE)?;
        let result = EdgeConnectivity {
            buffer: pipeline::output_buffer(device, "MOPMesh edge records", bytes),
            edge_count: mesh.edge_count,
            mesh: mesh.mesh,
        };
        if mesh.edge_count != 0 {
            pipeline::dispatch_patches(
                device,
                encoder,
                "MOPMesh edge connectivity",
                &self.pipeline,
                &[&result.buffer],
                mesh,
                grid,
            );
        }
        Ok(result)
    }
}
