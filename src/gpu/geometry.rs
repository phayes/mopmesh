use super::attribute::face_kernel;
use super::pipeline::{self, workgroups};
use super::{
    FaceVertexQuery, GpuError, GpuFaceAttribute, GpuMesh, GpuVertexAttribute, VertexFaceQuery,
};
use wgpu::util::DeviceExt;

/// Builds the pipeline for one of the compiled per-face geometry kernels.
fn face_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &'static str,
) -> wgpu::ComputePipeline {
    let module = pipeline::shader_module(device, label, source);
    pipeline::compute_pipeline(device, label, &module, "main", None)
}

/// Computes the unit normal of each triangle. Cache one per device.
///
/// A normal points to the side from which the triangle's corners appear
/// counter-clockwise, in input order. Faces with zero area (for example,
/// collinear corners) get the zero vector instead of a normal.
pub struct FaceNormalQuery {
    face_vertices: FaceVertexQuery,
    pipeline: wgpu::ComputePipeline,
}
impl FaceNormalQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            face_vertices: FaceVertexQuery::new(device),
            pipeline: face_pipeline(
                device,
                "MOPMesh face normals",
                include_str!(concat!(env!("OUT_DIR"), "/face_normals.wgsl")),
            ),
        }
    }

    /// Records a query for face normals from the supplied vertex positions.
    /// Each position is `[x, y, z, w]`; `w` is ignored. The result holds one
    /// `[x, y, z, 0]` per face. Returns [`GpuError::MeshMismatch`] unless
    /// `positions` belongs to `mesh`, or [`GpuError::LimitsExceeded`] if a
    /// required buffer or dispatch is too large. Errors occur before allocating
    /// or recording commands. An empty mesh returns an empty result without
    /// recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        positions: &GpuVertexAttribute<[f32; 4]>,
    ) -> Result<GpuFaceAttribute<[f32; 4]>, GpuError> {
        face_kernel(
            device,
            encoder,
            mesh,
            positions,
            &self.face_vertices,
            &self.pipeline,
            "face normals",
        )
    }
}

/// Computes the area of each triangle. Cache one per device.
/// Areas are never negative; faces with collinear corners have zero area.
pub struct FaceAreaQuery {
    face_vertices: FaceVertexQuery,
    pipeline: wgpu::ComputePipeline,
}
impl FaceAreaQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            face_vertices: FaceVertexQuery::new(device),
            pipeline: face_pipeline(
                device,
                "MOPMesh face areas",
                include_str!(concat!(env!("OUT_DIR"), "/face_areas.wgsl")),
            ),
        }
    }

    /// Records a query for face areas from the supplied vertex positions.
    /// Each position is `[x, y, z, w]`; `w` is ignored. The result holds one
    /// `f32` per face. Errors and empty meshes behave as in
    /// [`FaceNormalQuery::encode`].
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        positions: &GpuVertexAttribute<[f32; 4]>,
    ) -> Result<GpuFaceAttribute<f32>, GpuError> {
        face_kernel(
            device,
            encoder,
            mesh,
            positions,
            &self.face_vertices,
            &self.pipeline,
            "face areas",
        )
    }
}

/// Computes area-weighted vertex normals. Cache one per device.
///
/// Each vertex normal is the normalized sum, over the faces touching the
/// vertex, of the face normal scaled by twice the face's area. Larger faces
/// therefore count for more, and zero-area faces count for nothing. A vertex
/// whose weighted face normals cancel out gets the zero vector. Because the
/// faces around a vertex are summed in an unspecified order, results may
/// differ in the last bits between runs.
pub struct VertexNormalQuery {
    vertex_faces: VertexFaceQuery,
    face_vertices: FaceVertexQuery,
    pipeline: wgpu::ComputePipeline,
}
impl VertexNormalQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            vertex_faces: VertexFaceQuery::new(device),
            face_vertices: FaceVertexQuery::new(device),
            pipeline: face_pipeline(
                device,
                "MOPMesh vertex normals",
                include_str!(concat!(env!("OUT_DIR"), "/vertex_normals.wgsl")),
            ),
        }
    }

    /// Records a query for vertex normals from the supplied vertex positions.
    /// Each position is `[x, y, z, w]`; `w` is ignored. The result holds one
    /// `[x, y, z, 0]` per vertex. Returns [`GpuError::MeshMismatch`] unless
    /// `positions` belongs to `mesh`, or [`GpuError::LimitsExceeded`] if a
    /// required buffer or dispatch is too large. Errors occur before allocating
    /// or recording commands. An empty mesh returns an empty result without
    /// recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        positions: &GpuVertexAttribute<[f32; 4]>,
    ) -> Result<GpuVertexAttribute<[f32; 4]>, GpuError> {
        // Validate the mesh ID and the result and dispatch sizes before GPU work.
        const RESOURCE: &str = "vertex normals";
        if mesh.mesh != positions.mesh_id() {
            return Err(GpuError::MeshMismatch);
        }
        let limits = device.limits();
        GpuVertexAttribute::<[f32; 4]>::check_output(&limits, mesh.vertex_count, RESOURCE)?;
        let groups = workgroups(&limits, mesh.vertex_count, RESOURCE)?;
        let result = GpuVertexAttribute::output(
            device,
            mesh.mesh,
            mesh.vertex_count,
            "MOPMesh vertex normals",
        );
        if mesh.vertex_count == 0 {
            return Ok(result);
        }

        // Gather each vertex's faces, then sum their area vectors.
        let vertex_faces = self.vertex_faces.encode(device, encoder, mesh)?;
        let faces = self.face_vertices.encode(device, encoder, mesh)?;
        // Padded to 16 bytes, the minimum uniform struct size.
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh vertex normals sizes"),
            contents: bytemuck::cast_slice(&[mesh.vertex_count, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh vertex normals bindings",
            &self.pipeline.get_bind_group_layout(0),
            &[
                &params,
                faces.buffer(),
                positions.buffer(),
                vertex_faces.buffer(),
                result.buffer(),
            ],
        );
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex normals",
            &self.pipeline,
            &bindings,
            groups,
        );
        Ok(result)
    }
}
