use crate::{DirectedEdge, EdgeHandle, FaceHandle, Patch, VertexHandle};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identifies a [`Mesh`] and its attributes, GPU uploads, and query results.
/// Compare IDs before combining resources to check that they belong to the same mesh.
/// Building another mesh assigns a new ID, even if its input data is identical.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshId(u64);
impl MeshId {
    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// An input error reported by [`MeshBuilder::build`]. Vertex and face IDs are
/// zero-based indices into the positions and triangles supplied to the builder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    InvalidPatchSize,
    TooManyElements,
    NonFinitePosition(u32),
    InvalidVertex { face: u32, vertex: u32 },
    DegenerateFace(u32),
    DuplicateFace(u32),
    NonManifoldEdge([u32; 2]),
    InconsistentWinding([u32; 2]),
    NonManifoldVertex(u32),
    IsolatedVertex(u32),
}
impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPatchSize => {
                f.write_str("patch size must be between 1 and u32::MAX faces")
            }
            Self::TooManyElements => f.write_str("too many vertices or faces for u32 indices"),
            Self::NonFinitePosition(v) => write!(f, "vertex {v} has a non-finite coordinate"),
            Self::InvalidVertex { face, vertex } => {
                write!(f, "face {face} references missing vertex {vertex}")
            }
            Self::DegenerateFace(face) => write!(f, "face {face} has zero area"),
            Self::DuplicateFace(face) => write!(f, "face {face} repeats an earlier face"),
            Self::NonManifoldEdge([a, b]) => {
                write!(f, "edge ({a}, {b}) is shared by more than two faces")
            }
            Self::InconsistentWinding([a, b]) => {
                write!(f, "faces sharing edge ({a}, {b}) have inconsistent winding")
            }
            Self::NonManifoldVertex(v) => write!(f, "vertex {v} is non-manifold"),
            Self::IsolatedVertex(v) => write!(f, "vertex {v} is not used by any face"),
        }
    }
}
impl std::error::Error for BuildError {}

/// Constructs storage for oriented manifold triangles, with or without boundary.
/// Empty meshes are accepted; isolated vertices and zero-area faces are rejected.
#[derive(Debug)]
pub struct MeshBuilder<'a> {
    positions: &'a [[f32; 3]],
    triangles: &'a [[u32; 3]],
    patch_size: usize,
}
impl Default for MeshBuilder<'_> {
    fn default() -> Self {
        Self {
            positions: &[],
            triangles: &[],
            patch_size: 128,
        }
    }
}
impl<'a> MeshBuilder<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn positions(mut self, positions: &'a [[f32; 3]]) -> Self {
        self.positions = positions;
        self
    }
    pub fn triangles(mut self, triangles: &'a [[u32; 3]]) -> Self {
        self.triangles = triangles;
        self
    }
    /// Maximum owned faces per patch, excluding ribbons.
    pub fn patch_size(mut self, size: usize) -> Self {
        self.patch_size = size;
        self
    }

    pub fn build(self) -> Result<Mesh, BuildError> {
        if self.patch_size == 0 || self.patch_size > u32::MAX as usize {
            return Err(BuildError::InvalidPatchSize);
        }
        if self.positions.len() > u32::MAX as usize || self.triangles.len() > u32::MAX as usize / 3
        {
            return Err(BuildError::TooManyElements);
        }
        for (v, p) in self.positions.iter().enumerate() {
            if p.iter().any(|x| !x.is_finite()) {
                return Err(BuildError::NonFinitePosition(v as u32));
            }
        }
        let mut edge_map = BTreeMap::<[u32; 2], u32>::new();
        let mut edges = Vec::new();
        let mut edge_faces: Vec<Vec<(u32, bool)>> = Vec::new();
        let mut face_edges = Vec::new();
        let mut vertex_faces = vec![Vec::new(); self.positions.len()];
        let mut unique_faces = BTreeSet::new();
        for (face, &vs) in self.triangles.iter().enumerate() {
            let face = face as u32;
            for &v in &vs {
                if v as usize >= self.positions.len() {
                    return Err(BuildError::InvalidVertex { face, vertex: v });
                }
            }
            let [a, b, c] = vs.map(|v| self.positions[v as usize].map(f64::from));
            let u: [f64; 3] = std::array::from_fn(|i| b[i] - a[i]);
            let v: [f64; 3] = std::array::from_fn(|i| c[i] - a[i]);
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            if cross == [0.0; 3] {
                return Err(BuildError::DegenerateFace(face));
            }
            let mut key = vs;
            key.sort_unstable();
            if !unique_faces.insert(key) {
                return Err(BuildError::DuplicateFace(face));
            }
            let mut fe = [DirectedEdge {
                edge: 0,
                reversed: false,
            }; 3];
            for i in 0..3 {
                vertex_faces[vs[i] as usize].push(face);
                let (a, b) = (vs[i], vs[(i + 1) % 3]);
                let key = [a.min(b), a.max(b)];
                let e = *edge_map.entry(key).or_insert_with(|| {
                    let id = edges.len() as u32;
                    edges.push(key);
                    edge_faces.push(Vec::new());
                    id
                });
                let reversed = a > b;
                let incident = &mut edge_faces[e as usize];
                if incident.len() == 2 {
                    return Err(BuildError::NonManifoldEdge(key));
                }
                if incident.first().is_some_and(|&(_, r)| r == reversed) {
                    return Err(BuildError::InconsistentWinding(key));
                }
                incident.push((face, reversed));
                fe[i] = DirectedEdge { edge: e, reversed };
            }
            face_edges.push(fe);
        }
        // Each vertex link must be one path or one cycle, not a bow tie.
        for (v, incident) in vertex_faces.iter().enumerate() {
            if incident.is_empty() {
                return Err(BuildError::IsolatedVertex(v as u32));
            }
            let mut link = BTreeMap::<u32, Vec<u32>>::new();
            for &f in incident {
                let others: Vec<_> = self.triangles[f as usize]
                    .iter()
                    .copied()
                    .filter(|&x| x != v as u32)
                    .collect();
                link.entry(others[0]).or_default().push(others[1]);
                link.entry(others[1]).or_default().push(others[0]);
            }
            let mut seen = BTreeSet::new();
            let mut stack = vec![*link.first_key_value().unwrap().0];
            while let Some(x) = stack.pop() {
                if seen.insert(x) {
                    stack.extend(&link[&x]);
                }
            }
            let ends = link.values().filter(|n| n.len() == 1).count();
            if seen.len() != link.len()
                || link.values().any(|n| n.len() > 2)
                || (ends != 0 && ends != 2)
            {
                return Err(BuildError::NonManifoldVertex(v as u32));
            }
        }
        // Deterministic breadth-first face partitioning across shared edges.
        let mut face_patch = vec![u32::MAX; self.triangles.len()];
        let mut groups = Vec::<Vec<u32>>::new();
        for seed in 0..self.triangles.len() {
            if face_patch[seed] != u32::MAX {
                continue;
            }
            let id = groups.len() as u32;
            let mut group = vec![seed as u32];
            let mut queue = VecDeque::from([seed as u32]);
            face_patch[seed] = id;
            while let Some(f) = queue.pop_front() {
                for d in face_edges[f as usize] {
                    for &(n, _) in &edge_faces[d.edge as usize] {
                        if group.len() < self.patch_size && face_patch[n as usize] == u32::MAX {
                            face_patch[n as usize] = id;
                            group.push(n);
                            queue.push_back(n);
                        }
                    }
                }
            }
            groups.push(group);
        }
        let vertex_patch: Vec<_> = vertex_faces
            .iter()
            .map(|fs| fs.iter().map(|&f| face_patch[f as usize]).min().unwrap())
            .collect();
        let edge_patch: Vec<_> = edge_faces
            .iter()
            .map(|fs| {
                fs.iter()
                    .map(|&(f, _)| face_patch[f as usize])
                    .min()
                    .unwrap()
            })
            .collect();
        let mut patches = Vec::new();
        for (id, group) in groups.iter().enumerate() {
            let mut fs = BTreeSet::new();
            for &f in group {
                for &v in &self.triangles[f as usize] {
                    fs.extend(&vertex_faces[v as usize]);
                }
            }
            let faces: Vec<u32> = fs.into_iter().collect();
            let es: BTreeSet<_> = faces
                .iter()
                .flat_map(|&f| face_edges[f as usize].map(|d| d.edge))
                .collect();
            let edges_local: Vec<_> = es.into_iter().collect();
            let vs: BTreeSet<_> = edges_local
                .iter()
                .flat_map(|&e| edges[e as usize])
                .collect();
            let vertices: Vec<_> = vs.into_iter().collect();
            let ev = edges_local
                .iter()
                .map(|&e| edges[e as usize].map(|v| vertices.binary_search(&v).unwrap() as u32))
                .collect();
            let fe = faces
                .iter()
                .map(|&f| {
                    face_edges[f as usize].map(|d| DirectedEdge {
                        edge: edges_local.binary_search(&d.edge).unwrap() as u32,
                        reversed: d.reversed,
                    })
                })
                .collect();
            patches.push(Patch {
                id: id as u32,
                vertices,
                edges: edges_local,
                faces,
                ev,
                fe,
                vertex_owners: Vec::new(),
                edge_owners: Vec::new(),
                face_owners: Vec::new(),
            });
        }
        let vertex_owners: Vec<_> = vertex_patch
            .iter()
            .enumerate()
            .map(|(v, &p)| VertexHandle {
                patch: p,
                local: patches[p as usize]
                    .vertices
                    .binary_search(&(v as u32))
                    .unwrap() as u32,
            })
            .collect();
        let edge_owners: Vec<_> = edge_patch
            .iter()
            .enumerate()
            .map(|(e, &p)| EdgeHandle {
                patch: p,
                local: patches[p as usize]
                    .edges
                    .binary_search(&(e as u32))
                    .unwrap() as u32,
            })
            .collect();
        let face_owners: Vec<_> = face_patch
            .iter()
            .enumerate()
            .map(|(f, &p)| FaceHandle {
                patch: p,
                local: patches[p as usize]
                    .faces
                    .binary_search(&(f as u32))
                    .unwrap() as u32,
            })
            .collect();
        for p in &mut patches {
            p.vertex_owners = p
                .vertices
                .iter()
                .map(|&v| vertex_owners[v as usize])
                .collect();
            p.edge_owners = p.edges.iter().map(|&e| edge_owners[e as usize]).collect();
            p.face_owners = p.faces.iter().map(|&f| face_owners[f as usize]).collect();
        }
        Ok(Mesh {
            positions: self.positions.to_vec(),
            edge_count: edges.len(),
            face_count: self.triangles.len(),
            patches,
            id: MeshId::next(),
        })
    }
}

/// A validated triangle mesh created by [`MeshBuilder`].
/// Stores vertex positions and fixed connections between vertices, edges, and faces,
/// divided into patches for local access.
#[derive(Debug)]
pub struct Mesh {
    positions: Vec<[f32; 3]>,
    edge_count: usize,
    face_count: usize,
    patches: Vec<Patch>,
    id: MeshId,
}
impl Mesh {
    /// Returns this mesh's ID, also reported by its attributes, GPU uploads, and query results.
    pub fn id(&self) -> MeshId {
        self.id
    }
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }
    pub fn face_count(&self) -> usize {
        self.face_count
    }
    pub fn patches(&self) -> &[Patch] {
        &self.patches
    }
    pub fn vertex_owner(&self, h: VertexHandle) -> Option<VertexHandle> {
        self.patches
            .get(h.patch as usize)?
            .vertex_owners
            .get(h.local as usize)
            .copied()
    }
    pub fn edge_owner(&self, h: EdgeHandle) -> Option<EdgeHandle> {
        self.patches
            .get(h.patch as usize)?
            .edge_owners
            .get(h.local as usize)
            .copied()
    }
    pub fn face_owner(&self, h: FaceHandle) -> Option<FaceHandle> {
        self.patches
            .get(h.patch as usize)?
            .face_owners
            .get(h.local as usize)
            .copied()
    }
    /// Creates an attribute with a clone of `value` for each input vertex.
    pub fn vertex_attribute<T: Clone>(&self, value: T) -> VertexAttribute<T> {
        VertexAttribute {
            values: vec![value; self.positions.len()],
            mesh: self.id,
        }
    }
    /// Creates an attribute from values in the original input vertex order.
    /// Returns `None` unless `values` contains exactly one value per vertex.
    pub fn vertex_attribute_from<T>(&self, values: Vec<T>) -> Option<VertexAttribute<T>> {
        (values.len() == self.positions.len()).then_some(VertexAttribute {
            values,
            mesh: self.id,
        })
    }
    /// Original positions as WGSL-compatible vec4 values; the fourth lane is zero.
    pub fn position_attribute(&self) -> VertexAttribute<[f32; 4]> {
        VertexAttribute {
            values: self
                .positions
                .iter()
                .map(|&[x, y, z]| [x, y, z, 0.])
                .collect(),
            mesh: self.id,
        }
    }
}

/// Canonical vertex values: ribbon reads/writes access the same value.
/// Attributes reject access through a different mesh. Handles themselves are
/// mesh-relative addresses, not globally unique or generation-checked IDs.
#[derive(Debug)]
pub struct VertexAttribute<T> {
    pub(crate) values: Vec<T>,
    pub(crate) mesh: MeshId,
}
impl<T> VertexAttribute<T> {
    /// Returns the ID of the mesh that created this attribute.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    /// Borrows the attribute values in the original input vertex order.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    fn index(&self, mesh: &Mesh, h: VertexHandle) -> Option<usize> {
        if self.mesh != mesh.id {
            return None;
        }
        Some(
            *mesh
                .patches
                .get(h.patch as usize)?
                .vertices
                .get(h.local as usize)? as usize,
        )
    }
    pub fn get(&self, mesh: &Mesh, h: VertexHandle) -> Option<&T> {
        self.values.get(self.index(mesh, h)?)
    }
    pub fn get_mut(&mut self, mesh: &Mesh, h: VertexHandle) -> Option<&mut T> {
        let index = self.index(mesh, h)?;
        self.values.get_mut(index)
    }
}
