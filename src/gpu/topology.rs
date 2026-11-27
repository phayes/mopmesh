//! Connectivity sources that GPU queries accept, and the stamps that tie
//! query results to the connectivity they were computed from.
use super::{GpuEditableMesh, GpuError, GpuMesh, PatchBuffers};
use crate::{ElementKind, MeshId};

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::GpuMesh {}
    impl Sealed for super::GpuEditableMesh {}
}

/// Identifies the connectivity a GPU result was computed from.
///
/// Results such as [`FaceVertices`](super::FaceVertices) and
/// [`EdgeConnectivity`](super::EdgeConnectivity) record the stamp of the mesh
/// they came from. Queries that take such a result compare it with the
/// current stamp of the mesh you pass, and return [`GpuError::Stale`] when
/// the mesh has been edited since. Every [`GpuMesh`] upload of the same CPU
/// mesh has the same stamp, because its connectivity never changes. Each
/// [`GpuEditableMesh`] upload has its own stamp, which changes with every
/// recorded edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopologyStamp {
    mesh: MeshId,
    upload: u64,
    version: u64,
}

impl TopologyStamp {
    /// The stamp of a static upload of `mesh`.
    pub(crate) fn fixed(mesh: MeshId) -> Self {
        Self {
            mesh,
            upload: 0,
            version: 0,
        }
    }

    /// The stamp of editable upload `upload` of `mesh` at `version`.
    /// `upload` must be nonzero and unique to the upload.
    pub(crate) fn editable(mesh: MeshId, upload: u64, version: u64) -> Self {
        debug_assert_ne!(upload, 0);
        Self {
            mesh,
            upload,
            version,
        }
    }

    /// Returns the ID of the CPU mesh the connectivity was uploaded from.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }

    /// Returns the topology version: always 0 for a [`GpuMesh`], and
    /// [`GpuEditableMesh::version`] for an editable mesh.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Checks that a result stamped `self` can be used with a mesh whose
    /// current stamp is `current`. Returns [`GpuError::MeshMismatch`] if they
    /// come from different CPU meshes, or [`GpuError::Stale`] if they come
    /// from different uploads or versions of the same mesh.
    pub(crate) fn check(self, current: TopologyStamp) -> Result<(), GpuError> {
        if self.mesh != current.mesh {
            Err(GpuError::MeshMismatch)
        } else if self != current {
            Err(GpuError::Stale)
        } else {
            Ok(())
        }
    }
}

/// Mesh connectivity on the GPU that queries can read: a [`GpuMesh`] or a
/// [`GpuEditableMesh`]. Query methods take `&impl GpuTopology`, so pass
/// either kind of mesh. This trait can't be implemented outside this crate.
pub trait GpuTopology: sealed::Sealed {
    /// Returns the ID of the CPU mesh used for the upload.
    fn mesh_id(&self) -> MeshId;
    /// Returns the stamp of the connectivity as of the last recorded edit.
    fn topology(&self) -> TopologyStamp;
    /// Returns the connectivity buffer; see [`GpuMesh::arena`].
    fn arena(&self) -> &wgpu::Buffer;
    /// Returns the slot-state buffer; see [`GpuMesh::masks`].
    fn masks(&self) -> &wgpu::Buffer;
    /// Returns the number of face groups used to process the mesh on the GPU.
    fn patch_count(&self) -> u32;
    fn vertex_count(&self) -> u32;
    fn edge_count(&self) -> u32;
    fn face_count(&self) -> u32;
    /// Returns the number of vertices, edges or faces.
    fn element_count(&self, kind: ElementKind) -> u32 {
        match kind {
            ElementKind::Vertex => self.vertex_count(),
            ElementKind::Edge => self.edge_count(),
            ElementKind::Face => self.face_count(),
        }
    }
}

/// Borrows `mesh`'s connectivity buffers, counts and current stamp for a query.
pub(crate) fn buffers(mesh: &impl GpuTopology) -> PatchBuffers<'_> {
    PatchBuffers {
        arena: mesh.arena(),
        masks: mesh.masks(),
        patch_count: mesh.patch_count(),
        vertex_count: mesh.vertex_count(),
        face_count: mesh.face_count(),
        edge_count: mesh.edge_count(),
        topology: mesh.topology(),
    }
}

impl GpuTopology for GpuMesh {
    fn mesh_id(&self) -> MeshId {
        GpuMesh::mesh_id(self)
    }
    fn topology(&self) -> TopologyStamp {
        TopologyStamp::fixed(GpuMesh::mesh_id(self))
    }
    fn arena(&self) -> &wgpu::Buffer {
        GpuMesh::arena(self)
    }
    fn masks(&self) -> &wgpu::Buffer {
        GpuMesh::masks(self)
    }
    fn patch_count(&self) -> u32 {
        GpuMesh::patch_count(self)
    }
    fn vertex_count(&self) -> u32 {
        GpuMesh::vertex_count(self)
    }
    fn edge_count(&self) -> u32 {
        GpuMesh::edge_count(self)
    }
    fn face_count(&self) -> u32 {
        GpuMesh::face_count(self)
    }
}

impl GpuTopology for GpuEditableMesh {
    fn mesh_id(&self) -> MeshId {
        GpuEditableMesh::mesh_id(self)
    }
    fn topology(&self) -> TopologyStamp {
        GpuEditableMesh::topology(self)
    }
    fn arena(&self) -> &wgpu::Buffer {
        GpuEditableMesh::arena(self)
    }
    fn masks(&self) -> &wgpu::Buffer {
        GpuEditableMesh::masks(self)
    }
    fn patch_count(&self) -> u32 {
        GpuEditableMesh::patch_count(self)
    }
    fn vertex_count(&self) -> u32 {
        GpuEditableMesh::vertex_count(self)
    }
    fn edge_count(&self) -> u32 {
        GpuEditableMesh::edge_count(self)
    }
    fn face_count(&self) -> u32 {
        GpuEditableMesh::face_count(self)
    }
}
