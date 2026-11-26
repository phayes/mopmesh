use crate::{EdgeHandle, FaceHandle, VertexHandle};

/// An edge index and direction relative to its stored EV endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectedEdge {
    pub edge: u32,
    pub reversed: bool,
}

/// Immutable local connectivity including vertex-star ribbons.
///
/// Full one-ring queries are guaranteed for owned elements. Queries starting
/// from ribbon elements may be incomplete; resolve their owner first.
/// Global IDs refer to input vertices/faces and deterministic constructed edges.
#[derive(Debug)]
pub struct Patch {
    pub(crate) id: u32,
    pub(crate) vertices: Vec<u32>,
    pub(crate) edges: Vec<u32>,
    pub(crate) faces: Vec<u32>,
    pub(crate) ev: Vec<[u32; 2]>,
    pub(crate) fe: Vec<[DirectedEdge; 3]>,
    pub(crate) vertex_owners: Vec<VertexHandle>,
    pub(crate) edge_owners: Vec<EdgeHandle>,
    pub(crate) face_owners: Vec<FaceHandle>,
}

impl Patch {
    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn vertex_ids(&self) -> &[u32] {
        &self.vertices
    }
    pub fn edge_ids(&self) -> &[u32] {
        &self.edges
    }
    pub fn face_ids(&self) -> &[u32] {
        &self.faces
    }
    pub fn edge_vertices(&self) -> &[[u32; 2]] {
        &self.ev
    }
    pub fn face_edges(&self) -> &[[DirectedEdge; 3]] {
        &self.fe
    }
    pub fn vertex_owners(&self) -> &[VertexHandle] {
        &self.vertex_owners
    }
    pub fn edge_owners(&self) -> &[EdgeHandle] {
        &self.edge_owners
    }
    pub fn face_owners(&self) -> &[FaceHandle] {
        &self.face_owners
    }

    pub fn face_vertices(&self, local: u32) -> Option<[u32; 3]> {
        Some(
            self.fe
                .get(local as usize)?
                .map(|d| self.ev[d.edge as usize][usize::from(d.reversed)]),
        )
    }

    /// Local face IDs incident on a local vertex.
    pub fn vertex_faces(&self, local: u32) -> impl Iterator<Item = u32> + '_ {
        (0..self.fe.len() as u32)
            .filter(move |&f| self.face_vertices(f).is_some_and(|vs| vs.contains(&local)))
    }

    pub fn owned_face_count(&self) -> usize {
        self.face_owners
            .iter()
            .filter(|h| h.patch == self.id)
            .count()
    }
}
