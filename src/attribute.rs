use crate::{EdgeHandle, FaceHandle, Mesh, MeshId, Patch, VertexHandle};
use std::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
    impl Sealed for crate::VertexHandle {}
    impl Sealed for crate::EdgeHandle {}
    impl Sealed for crate::FaceHandle {}
}

/// The kind of mesh element an attribute stores values for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementKind {
    Vertex,
    Edge,
    Face,
}

/// A mesh element handle ([`VertexHandle`], [`EdgeHandle`] or [`FaceHandle`]).
/// It selects which elements an [`Attribute`] or GPU attribute holds values
/// for. This trait is sealed: only those three handle types implement it.
pub trait Element: sealed::Sealed + Copy {
    /// Which element kind this handle addresses.
    const KIND: ElementKind;

    /// Returns the canonical element ID that `self` addresses in `patch`, or
    /// `None` if `self.local` is out of range for that patch.
    #[doc(hidden)]
    fn canonical_id(self, patch: &Patch) -> Option<u32>;

    /// Returns the patch this handle addresses.
    #[doc(hidden)]
    fn patch(self) -> u32;
}

impl Element for VertexHandle {
    const KIND: ElementKind = ElementKind::Vertex;
    fn canonical_id(self, patch: &Patch) -> Option<u32> {
        patch.vertex_ids().get(self.local as usize).copied()
    }
    fn patch(self) -> u32 {
        self.patch
    }
}
impl Element for EdgeHandle {
    const KIND: ElementKind = ElementKind::Edge;
    fn canonical_id(self, patch: &Patch) -> Option<u32> {
        patch.edge_ids().get(self.local as usize).copied()
    }
    fn patch(self) -> u32 {
        self.patch
    }
}
impl Element for FaceHandle {
    const KIND: ElementKind = ElementKind::Face;
    fn patch(self) -> u32 {
        self.patch
    }
    fn canonical_id(self, patch: &Patch) -> Option<u32> {
        patch.face_ids().get(self.local as usize).copied()
    }
}

/// One value per vertex, edge or face of a [`Mesh`], chosen by `E`.
///
/// Values are stored once per element, in canonical order: input order for
/// vertices and faces, canonical edge ID order for edges. Every handle that
/// addresses the same element, including ribbon copies in other patches,
/// reads and writes the same value. An attribute only answers handles of the
/// mesh that created it.
#[derive(Debug, Clone)]
pub struct Attribute<E: Element, T> {
    pub(crate) values: Vec<T>,
    pub(crate) mesh: MeshId,
    marker: PhantomData<E>,
}

/// One value per vertex, in input vertex order.
pub type VertexAttribute<T> = Attribute<VertexHandle, T>;
/// One value per edge, in canonical edge ID order.
pub type EdgeAttribute<T> = Attribute<EdgeHandle, T>;
/// One value per face, in input face order.
pub type FaceAttribute<T> = Attribute<FaceHandle, T>;

impl<E: Element, T> Attribute<E, T> {
    /// Creates an attribute holding a clone of `value` for every element of `mesh`.
    pub fn new(mesh: &Mesh, value: T) -> Self
    where
        T: Clone,
    {
        Self::wrap(mesh, vec![value; mesh.element_count(E::KIND)])
    }

    /// Creates an attribute from one value per element, in canonical order.
    /// Returns `None` unless `values` has exactly one value per element.
    pub fn from_values(mesh: &Mesh, values: Vec<T>) -> Option<Self> {
        (values.len() == mesh.element_count(E::KIND)).then(|| Self::wrap(mesh, values))
    }

    pub(crate) fn wrap(mesh: &Mesh, values: Vec<T>) -> Self {
        Self {
            values,
            mesh: mesh.id(),
            marker: PhantomData,
        }
    }

    /// Returns the ID of the mesh that created this attribute.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    /// Borrows the values in canonical element order.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    /// Mutably borrows the values in canonical element order.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the value for the element `h` addresses in `mesh`. Returns
    /// `None` if `mesh` did not create this attribute or `h` is out of range.
    pub fn get(&self, mesh: &Mesh, h: E) -> Option<&T> {
        self.values.get(self.index(mesh, h)?)
    }
    /// Like [`get`](Self::get), but returns a mutable reference.
    pub fn get_mut(&mut self, mesh: &Mesh, h: E) -> Option<&mut T> {
        let index = self.index(mesh, h)?;
        self.values.get_mut(index)
    }

    fn index(&self, mesh: &Mesh, h: E) -> Option<usize> {
        if self.mesh != mesh.id() {
            return None;
        }
        let patch = mesh.patches().get(h.patch() as usize)?;
        Some(h.canonical_id(patch)? as usize)
    }
}
