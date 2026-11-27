use super::pipeline::{self, check_binding, workgroups};
use super::{FaceVertexQuery, GpuError, GpuMesh};
use crate::{MeshId, VertexAttribute};
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

mod sealed {
    pub trait Sealed {}
    macro_rules! values {
        ($($ty:ty),*) => { $(impl Sealed for $ty {})* };
    }
    values!(
        f32, u32, i32, [f32; 2], [u32; 2], [i32; 2], [f32; 4], [u32; 4], [i32; 4]
    );
}

/// Supported storage-array elements: f32/u32/i32 scalars and their vec2/vec4
/// equivalents. Rust arrays of length three are deliberately excluded: WGSL
/// vec3 array stride is 16 bytes, whereas a Rust [f32; 3] occupies 12 bytes.
/// This trait is sealed to keep the storage ABI explicit.
pub trait GpuValue: sealed::Sealed + bytemuck::Pod {}
impl<T: sealed::Sealed + bytemuck::Pod> GpuValue for T {}

/// A GPU buffer with one attribute value per vertex, in original input order.
/// Bind it as an array of the WGSL scalar or vector type matching `T`. Buffers
/// this crate creates support `STORAGE | COPY_SRC | COPY_DST`; a buffer from
/// [`GpuMesh::vertex_attribute_from_buffer`] keeps its own usage and may be
/// longer than the attribute. Direct writes must preserve the element layout
/// and vertex order. An empty attribute has a zero placeholder the size of one
/// element, but its length is zero.
pub struct GpuVertexAttribute<T: GpuValue> {
    buffer: wgpu::Buffer,
    mesh: MeshId,
    len: u32,
    marker: PhantomData<T>,
}

impl GpuMesh {
    /// Copies a CPU vertex attribute to a new GPU buffer. Later CPU changes
    /// do not update the GPU copy; use [`GpuVertexAttribute::write`] to replace it.
    /// The attribute must belong to the same mesh as this upload. Returns
    /// [`GpuError::MeshMismatch`] otherwise, or [`GpuError::LimitsExceeded`]
    /// if the buffer would be too large. Errors occur before GPU allocation.
    pub fn upload_vertex_attribute<T: GpuValue>(
        &self,
        device: &wgpu::Device,
        attribute: &VertexAttribute<T>,
    ) -> Result<GpuVertexAttribute<T>, GpuError> {
        if self.mesh != attribute.mesh {
            return Err(GpuError::MeshMismatch);
        }
        self.create_vertex_attribute(device, &attribute.values)
    }

    /// Copies one value per vertex, in input vertex order, to a new GPU buffer
    /// with `STORAGE | COPY_SRC | COPY_DST` usage. Use this for meshes built
    /// with [`GpuMeshBuilder`](super::GpuMeshBuilder), which have no CPU
    /// attributes. Returns [`GpuError::InvalidBuffer`] unless `values` has
    /// exactly [`vertex_count`](Self::vertex_count) entries, or
    /// [`GpuError::LimitsExceeded`] if the buffer would be too large. Errors
    /// occur before GPU allocation.
    pub fn create_vertex_attribute<T: GpuValue>(
        &self,
        device: &wgpu::Device,
        values: &[T],
    ) -> Result<GpuVertexAttribute<T>, GpuError> {
        const RESOURCE: &str = "vertex attribute";
        if values.len() != self.vertex_count as usize {
            return Err(GpuError::InvalidBuffer { resource: RESOURCE });
        }
        let bytes = bytemuck::cast_slice(values);
        let size = bytes.len().max(std::mem::size_of::<T>()) as u64;
        check_binding(&device.limits(), size, RESOURCE)?;
        let zero = T::zeroed();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh canonical vertex attribute"),
            contents: if bytes.is_empty() {
                bytemuck::bytes_of(&zero)
            } else {
                bytes
            },
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Ok(GpuVertexAttribute {
            buffer,
            mesh: self.mesh,
            len: self.vertex_count,
            marker: PhantomData,
        })
    }

    /// Uses an existing buffer as a vertex attribute of this mesh, without
    /// copying it: GPU writes to `buffer` are seen by queries that read the
    /// attribute. For example, pass the positions buffer given to
    /// [`GpuMeshBuilder::build`](super::GpuMeshBuilder::build) to compute face
    /// centroids. The buffer must hold one `T` per vertex in input vertex order,
    /// starting at byte 0, and at least one `T` for an empty mesh. Returns
    /// [`GpuError::InvalidBuffer`] if it is smaller or lacks `STORAGE` usage.
    pub fn vertex_attribute_from_buffer<T: GpuValue>(
        &self,
        buffer: &wgpu::Buffer,
    ) -> Result<GpuVertexAttribute<T>, GpuError> {
        let needed = std::mem::size_of::<T>() as u64 * u64::from(self.vertex_count.max(1));
        if buffer.size() < needed || !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(GpuError::InvalidBuffer {
                resource: "vertex attribute",
            });
        }
        Ok(GpuVertexAttribute {
            buffer: buffer.clone(),
            mesh: self.mesh,
            len: self.vertex_count,
            marker: PhantomData,
        })
    }
}

impl<T: GpuValue> GpuVertexAttribute<T> {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Queue a full replacement from a host attribute of the same mesh. Host
    /// and GPU attributes are snapshots; neither automatically updates the other.
    /// Queue writes become visible on the next submission, including to commands
    /// already encoded. Submit earlier readers before calling this method.
    /// Returns [`GpuError::MeshMismatch`] if `attribute` belongs to another mesh
    /// (always the case for a mesh from [`GpuMeshBuilder`](super::GpuMeshBuilder)),
    /// or [`GpuError::InvalidBuffer`] if the buffer lacks `COPY_DST` usage.
    pub fn write(
        &mut self,
        queue: &wgpu::Queue,
        attribute: &VertexAttribute<T>,
    ) -> Result<(), GpuError> {
        if self.mesh != attribute.mesh {
            return Err(GpuError::MeshMismatch);
        }
        if !self.buffer.usage().contains(wgpu::BufferUsages::COPY_DST) {
            return Err(GpuError::InvalidBuffer {
                resource: "vertex attribute",
            });
        }
        if !self.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&attribute.values));
        }
        Ok(())
    }
}

/// The center of each triangle, computed as the average of its three positions.
/// The buffer contains one `vec4<f32>` per input face ID (16 bytes per face):
/// `[x, y, z, 0]`. It supports `STORAGE | COPY_SRC`. An empty mesh has one zero
/// vector as a placeholder and a face count of zero.
pub struct FaceCentroids {
    buffer: wgpu::Buffer,
    face_count: u32,
    mesh: MeshId,
}
impl FaceCentroids {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
}

/// A first attribute-consuming geometry query. Cache once per device.
pub struct FaceCentroidQuery {
    face_vertices: FaceVertexQuery,
    pipeline: wgpu::ComputePipeline,
}
impl FaceCentroidQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh face centroids",
            include_str!(concat!(env!("OUT_DIR"), "/face_centroids.wgsl")),
        );
        Self {
            face_vertices: FaceVertexQuery::new(device),
            pipeline: pipeline::compute_pipeline(
                device,
                "MOPMesh face centroids",
                &module,
                "main",
                None,
            ),
        }
    }

    /// Records a query for triangle centers using the supplied vertex positions.
    /// Each position is `[x, y, z, w]`; `w` is ignored. `positions` must belong
    /// to `mesh`, or the call returns [`GpuError::MeshMismatch`].
    /// Returns [`GpuError::LimitsExceeded`] if a required buffer or dispatch is
    /// too large. Errors occur before allocating or recording commands. An empty
    /// mesh returns an empty [`FaceCentroids`] result without recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        positions: &GpuVertexAttribute<[f32; 4]>,
    ) -> Result<FaceCentroids, GpuError> {
        // Validate the mesh ID and sizes, including the temporary face-vertex result,
        // before allocating buffers or recording commands.
        const RESOURCE: &str = "face centroids";
        if mesh.mesh != positions.mesh {
            return Err(GpuError::MeshMismatch);
        }
        let limits = device.limits();
        let bytes = (u64::from(mesh.face_count) * 16).max(16);
        check_binding(&limits, bytes, RESOURCE)?;
        check_binding(&limits, (u64::from(mesh.face_count) * 12).max(4), RESOURCE)?;
        let groups = workgroups(&limits, mesh.face_count, RESOURCE)?;

        let result = FaceCentroids {
            buffer: pipeline::output_buffer(device, "MOPMesh face centroid result", bytes),
            face_count: mesh.face_count,
            mesh: mesh.mesh,
        };
        if mesh.face_count == 0 {
            return Ok(result);
        }
        let faces = self.face_vertices.encode(device, encoder, mesh)?;
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh face centroid bindings",
            &self.pipeline.get_bind_group_layout(0),
            &[faces.buffer(), positions.buffer(), &result.buffer],
        );
        pipeline::dispatch(
            encoder,
            "MOPMesh face centroids",
            &self.pipeline,
            &bindings,
            groups,
        );
        Ok(result)
    }
}
