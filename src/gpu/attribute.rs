use super::pipeline::{self, check_binding, workgroups};
use super::{FaceVertexQuery, GpuError, GpuMesh};
use crate::{Attribute, EdgeHandle, Element, ElementKind, FaceHandle, MeshId, VertexHandle};
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

/// A GPU buffer with one value per vertex, edge or face (chosen by `E`), in
/// canonical order: input order for vertices and faces, canonical edge ID
/// order for edges. Bind it as an array of the WGSL scalar or vector type
/// matching `T`.
///
/// Buffers this crate creates support `STORAGE | COPY_SRC | COPY_DST`; a
/// buffer from [`GpuAttribute::from_buffer`] keeps its own usage and may be
/// longer than the attribute. Direct writes must preserve the element layout
/// and order. An empty attribute has a zero placeholder the size of one
/// element, but its length is zero.
pub struct GpuAttribute<E: Element, T: GpuValue> {
    buffer: wgpu::Buffer,
    mesh: MeshId,
    len: u32,
    marker: PhantomData<(E, T)>,
}

/// One GPU value per vertex, in input vertex order.
pub type GpuVertexAttribute<T> = GpuAttribute<VertexHandle, T>;
/// One GPU value per edge, in canonical edge ID order.
pub type GpuEdgeAttribute<T> = GpuAttribute<EdgeHandle, T>;
/// One GPU value per face, in input face order.
pub type GpuFaceAttribute<T> = GpuAttribute<FaceHandle, T>;

/// Names the attribute buffer in errors: "vertex attribute", "edge
/// attribute" or "face attribute".
fn resource(kind: ElementKind) -> &'static str {
    match kind {
        ElementKind::Vertex => "vertex attribute",
        ElementKind::Edge => "edge attribute",
        ElementKind::Face => "face attribute",
    }
}

impl GpuMesh {
    /// Returns the number of vertices, edges or faces.
    pub fn element_count(&self, kind: ElementKind) -> u32 {
        match kind {
            ElementKind::Vertex => self.vertex_count,
            ElementKind::Edge => self.edge_count,
            ElementKind::Face => self.face_count,
        }
    }

    /// Copies a CPU attribute to a new GPU buffer. Later CPU changes do not
    /// update the GPU copy; use [`GpuAttribute::write`] to replace it.
    /// The attribute must belong to the same mesh as this upload. Returns
    /// [`GpuError::MeshMismatch`] otherwise, or [`GpuError::LimitsExceeded`]
    /// if the buffer would be too large. Errors occur before GPU allocation.
    pub fn upload_attribute<E: Element, T: GpuValue>(
        &self,
        device: &wgpu::Device,
        attribute: &Attribute<E, T>,
    ) -> Result<GpuAttribute<E, T>, GpuError> {
        if self.mesh != attribute.mesh {
            return Err(GpuError::MeshMismatch);
        }
        GpuAttribute::new(device, self, &attribute.values)
    }
}

impl<E: Element, T: GpuValue> GpuAttribute<E, T> {
    /// Copies one value per element of `mesh`, in canonical order, to a new
    /// GPU buffer with `STORAGE | COPY_SRC | COPY_DST` usage. Use this for
    /// meshes built with [`GpuMeshBuilder`](super::GpuMeshBuilder), which have
    /// no CPU attributes. Returns [`GpuError::InvalidBuffer`] unless `values`
    /// has exactly [`GpuMesh::element_count`] entries, or
    /// [`GpuError::LimitsExceeded`] if the buffer would be too large. Errors
    /// occur before GPU allocation.
    pub fn new(device: &wgpu::Device, mesh: &GpuMesh, values: &[T]) -> Result<Self, GpuError> {
        let resource = resource(E::KIND);
        let len = mesh.element_count(E::KIND);
        if values.len() != len as usize {
            return Err(GpuError::InvalidBuffer { resource });
        }
        let bytes = bytemuck::cast_slice(values);
        let size = bytes.len().max(std::mem::size_of::<T>()) as u64;
        check_binding(&device.limits(), size, resource)?;
        let zero = T::zeroed();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh canonical attribute"),
            contents: if bytes.is_empty() {
                bytemuck::bytes_of(&zero)
            } else {
                bytes
            },
            usage: Self::USAGE,
        });
        Ok(Self::wrap(buffer, mesh, len))
    }

    /// Uses an existing buffer as an attribute of `mesh`, without copying it:
    /// GPU writes to `buffer` are seen by queries that read the attribute. For
    /// example, pass the positions buffer given to
    /// [`GpuMeshBuilder::build`](super::GpuMeshBuilder::build) to compute face
    /// centroids. The buffer must hold one `T` per element in canonical order,
    /// starting at byte 0, and at least one `T` for an empty attribute. Returns
    /// [`GpuError::InvalidBuffer`] if it is smaller or lacks `STORAGE` usage.
    pub fn from_buffer(mesh: &GpuMesh, buffer: &wgpu::Buffer) -> Result<Self, GpuError> {
        let len = mesh.element_count(E::KIND);
        let needed = std::mem::size_of::<T>() as u64 * u64::from(len.max(1));
        if buffer.size() < needed || !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(GpuError::InvalidBuffer {
                resource: resource(E::KIND),
            });
        }
        Ok(Self::wrap(buffer.clone(), mesh, len))
    }

    /// Checks that an output attribute for `mesh` fits the device limits.
    pub(crate) fn check_output(
        limits: &wgpu::Limits,
        len: u32,
        resource: &'static str,
    ) -> Result<(), GpuError> {
        check_binding(limits, Self::byte_size(len), resource)
    }

    /// Allocates a zeroed output attribute of `len` elements for mesh `mesh`.
    /// `len` must be the mesh's element count for `E`. Check the size with
    /// [`check_output`](Self::check_output) first.
    pub(crate) fn output(device: &wgpu::Device, mesh: MeshId, len: u32, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: Self::byte_size(len),
            usage: Self::USAGE,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            mesh,
            len,
            marker: PhantomData,
        }
    }

    const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
        .union(wgpu::BufferUsages::COPY_SRC)
        .union(wgpu::BufferUsages::COPY_DST);

    fn byte_size(len: u32) -> u64 {
        std::mem::size_of::<T>() as u64 * u64::from(len.max(1))
    }

    fn wrap(buffer: wgpu::Buffer, mesh: &GpuMesh, len: u32) -> Self {
        Self {
            buffer,
            mesh: mesh.mesh,
            len,
            marker: PhantomData,
        }
    }

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
        attribute: &Attribute<E, T>,
    ) -> Result<(), GpuError> {
        if self.mesh != attribute.mesh {
            return Err(GpuError::MeshMismatch);
        }
        self.write_values(queue, &attribute.values)
    }

    /// Queue a full replacement from one value per element, in canonical
    /// order. Use this for meshes from [`GpuMeshBuilder`](super::GpuMeshBuilder).
    /// Visibility rules match [`write`](Self::write). Returns
    /// [`GpuError::InvalidBuffer`] unless `values` has exactly
    /// [`len`](Self::len) entries and the buffer has `COPY_DST` usage.
    pub fn write_values(&mut self, queue: &wgpu::Queue, values: &[T]) -> Result<(), GpuError> {
        if values.len() != self.len as usize
            || !self.buffer.usage().contains(wgpu::BufferUsages::COPY_DST)
        {
            return Err(GpuError::InvalidBuffer {
                resource: resource(E::KIND),
            });
        }
        if !self.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(values));
        }
        Ok(())
    }
}

/// Computes the center of each triangle, as the average of its three
/// positions. Cache one per device.
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
    /// Each position is `[x, y, z, w]`; `w` is ignored. The result holds one
    /// `[x, y, z, 0]` per face. `positions` must belong to `mesh`, or the call
    /// returns [`GpuError::MeshMismatch`]. Returns [`GpuError::LimitsExceeded`]
    /// if a required buffer or dispatch is too large. Errors occur before
    /// allocating or recording commands. An empty mesh returns an empty result
    /// without recording GPU work.
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
            "face centroids",
        )
    }
}

/// Records a per-face kernel that reads face vertices and positions and
/// writes one `T` per face, after the shared mesh-ID and size checks.
pub(super) fn face_kernel<T: GpuValue>(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    mesh: &GpuMesh,
    positions: &GpuVertexAttribute<[f32; 4]>,
    face_vertices: &FaceVertexQuery,
    pipeline: &wgpu::ComputePipeline,
    resource: &'static str,
) -> Result<GpuFaceAttribute<T>, GpuError> {
    // Validate the mesh ID and sizes, including the temporary face-vertex
    // result, before allocating buffers or recording commands.
    if mesh.mesh != positions.mesh {
        return Err(GpuError::MeshMismatch);
    }
    let limits = device.limits();
    GpuFaceAttribute::<T>::check_output(&limits, mesh.face_count, resource)?;
    check_binding(&limits, (u64::from(mesh.face_count) * 12).max(4), resource)?;
    let groups = workgroups(&limits, mesh.face_count, resource)?;

    // Allocate the result, then run the face-vertex query and the kernel.
    let result = GpuFaceAttribute::<T>::output(device, mesh.mesh, mesh.face_count, resource);
    if mesh.face_count == 0 {
        return Ok(result);
    }
    let faces = face_vertices.encode(device, encoder, mesh)?;
    let bindings = pipeline::bind_buffers(
        device,
        resource,
        &pipeline.get_bind_group_layout(0),
        &[faces.buffer(), positions.buffer(), result.buffer()],
    );
    pipeline::dispatch(encoder, resource, pipeline, &bindings, groups);
    Ok(result)
}
