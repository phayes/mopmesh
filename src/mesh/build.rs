//! Connectivity passes used by `MeshBuilder::build`. Inputs have already been
//! checked for out-of-range vertex IDs.
use super::{BuildError, Partition};
use crate::DirectedEdge;
use std::collections::VecDeque;

/// Marks a missing face on a boundary edge.
pub(super) const NO_FACE: u32 = u32::MAX;

/// Checks each position and each face on its own, in this order:
/// the first vertex with a non-finite coordinate, then the first face that
/// references a missing vertex (its first such corner) or has two corners at
/// equal coordinates (`0.0` equals `-0.0`). Repeated vertex IDs count as equal
/// coordinates; three distinct collinear points do not.
pub(crate) fn check_elements(
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> Result<(), BuildError> {
    if let Some(v) = positions
        .iter()
        .position(|p| p.iter().any(|x| !x.is_finite()))
    {
        return Err(BuildError::NonFinitePosition(v as u32));
    }
    for (face, &vs) in triangles.iter().enumerate() {
        let face = face as u32;
        if let Some(&vertex) = vs.iter().find(|&&v| v as usize >= positions.len()) {
            return Err(BuildError::InvalidVertex { face, vertex });
        }
        let [a, b, c] = vs.map(|v| positions[v as usize]);
        if a == b || b == c || a == c {
            return Err(BuildError::DegenerateFace(face));
        }
    }
    Ok(())
}

/// Items `0..count` grouped by a key below `rows`, stored contiguously.
/// Row `r` is `entries[offsets[r]..offsets[r + 1]]`; items keep ascending order
/// within each row.
pub(crate) struct Rows {
    offsets: Vec<u32>,
    entries: Vec<u32>,
}
impl Rows {
    /// `count` must fit in `u32`, and `key` must return values below `rows`.
    pub(crate) fn group(rows: usize, count: usize, key: impl Fn(usize) -> u32) -> Self {
        // Count items per row, then turn the counts into row start offsets.
        let mut offsets = vec![0u32; rows + 1];
        for item in 0..count {
            offsets[key(item) as usize + 1] += 1;
        }
        for r in 0..rows {
            offsets[r + 1] += offsets[r];
        }
        // Place each item at the next free slot of its row.
        let mut next = offsets[..rows].to_vec();
        let mut entries = vec![0; count];
        for item in 0..count {
            let slot = &mut next[key(item) as usize];
            entries[*slot as usize] = item as u32;
            *slot += 1;
        }
        Self { offsets, entries }
    }
    /// Row start offsets, one per row plus the total item count.
    #[cfg(test)]
    pub(crate) fn offsets(&self) -> &[u32] {
        &self.offsets
    }
    pub(crate) fn row(&self, r: usize) -> &[u32] {
        &self.entries[self.offsets[r] as usize..self.offsets[r + 1] as usize]
    }
    fn row_mut(&mut self, r: usize) -> &mut [u32] {
        &mut self.entries[self.offsets[r] as usize..self.offsets[r + 1] as usize]
    }
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// Finds duplicate faces and connects edges, reporting the first error in face
/// order: a duplicate face, or the face whose corner first causes an edge error.
/// A duplicate wins a tie. `triangles` must have passed [`check_elements`].
pub(crate) fn check_edges(
    vertex_count: usize,
    triangles: &[[u32; 3]],
) -> Result<Edges, BuildError> {
    let duplicate = find_duplicate_face(vertex_count, triangles);
    let edges = connect_edges(vertex_count, triangles);
    match (duplicate, edges) {
        (Some(d), Err((face, _))) if d <= face => Err(BuildError::DuplicateFace(d)),
        (Some(d), Ok(_)) => Err(BuildError::DuplicateFace(d)),
        (_, edges) => edges.map_err(|(_, error)| error),
    }
}

/// Returns the lowest face ID that repeats the vertex set of an earlier face.
fn find_duplicate_face(vertex_count: usize, triangles: &[[u32; 3]]) -> Option<u32> {
    let keys: Vec<[u32; 3]> = triangles
        .iter()
        .map(|&t| {
            let mut key = t;
            key.sort_unstable();
            key
        })
        .collect();
    // Bucket faces by smallest vertex, then sort each small bucket by the rest.
    let mut buckets = Rows::group(vertex_count, triangles.len(), |f| keys[f][0]);
    let mut first = None;
    for r in 0..buckets.len() {
        let row = buckets.row_mut(r);
        row.sort_unstable_by_key(|&f| (keys[f as usize], f));
        for pair in row.windows(2) {
            if keys[pair[0] as usize] == keys[pair[1] as usize] {
                first = Some(first.map_or(pair[1], |f: u32| f.min(pair[1])));
            }
        }
    }
    first
}

/// Edge connectivity derived from the input triangles.
pub(crate) struct Edges {
    /// Ascending endpoint pair per edge ID.
    pub(crate) endpoints: Vec<[u32; 2]>,
    /// The one or two faces using each edge, in input face order; [`NO_FACE`]
    /// fills the second slot on a boundary.
    pub(crate) faces: Vec<[u32; 2]>,
    /// For each face and slot `i`, the edge from vertex `i` to vertex `i + 1`.
    pub(crate) face_edges: Vec<[DirectedEdge; 3]>,
}

/// Numbers edges in ascending order of their (smaller, larger) endpoint pair.
/// On failure, returns the error caused by the lowest face corner, paired with
/// that corner's face ID, matching a scan of the faces in input order.
fn connect_edges(vertex_count: usize, triangles: &[[u32; 3]]) -> Result<Edges, (u32, BuildError)> {
    // Corner `c` is slot `c % 3` of face `c / 3` and starts the edge to the next corner.
    let start = |c: usize| triangles[c / 3][c % 3];
    let end = |c: usize| triangles[c / 3][(c + 1) % 3];
    let low = |c: usize| start(c).min(end(c));
    let high = |c: usize| start(c).max(end(c));

    // Bucket corners by smaller endpoint, then sort each bucket by larger
    // endpoint and corner, so equal edges are adjacent and in face order.
    let mut buckets = Rows::group(vertex_count, 3 * triangles.len(), low);
    let mut endpoints = Vec::new();
    let mut faces = Vec::new();
    let mut face_edges = vec![
        [DirectedEdge {
            edge: 0,
            reversed: false
        }; 3];
        triangles.len()
    ];
    let mut error: Option<(u32, BuildError)> = None;
    for r in 0..buckets.len() {
        let row = buckets.row_mut(r);
        row.sort_unstable_by_key(|&c| (high(c as usize), c));
        for group in row.chunk_by(|&a, &b| high(a as usize) == high(b as usize)) {
            let key = [r as u32, high(group[0] as usize)];
            let reversed = |c: u32| start(c as usize) > end(c as usize);

            // Record the first problem this edge causes, by corner order.
            let problem = if group.len() >= 2 && reversed(group[0]) == reversed(group[1]) {
                Some((group[1], BuildError::InconsistentWinding(key)))
            } else if group.len() >= 3 {
                Some((group[2], BuildError::NonManifoldEdge(key)))
            } else {
                None
            };
            if let Some(problem) = problem {
                if error.as_ref().is_none_or(|(c, _)| problem.0 < *c) {
                    error = Some(problem);
                }
                continue;
            }

            // Assign the next edge ID and link it with its faces.
            let edge = endpoints.len() as u32;
            endpoints.push(key);
            faces.push([group[0] / 3, group.get(1).map_or(NO_FACE, |c| c / 3)]);
            for &c in group {
                face_edges[c as usize / 3][c as usize % 3] = DirectedEdge {
                    edge,
                    reversed: reversed(c),
                };
            }
        }
    }
    match error {
        Some((corner, error)) => Err((corner / 3, error)),
        None => Ok(Edges {
            endpoints,
            faces,
            face_edges,
        }),
    }
}

/// Splits the faces into patches of at most `patch_size` owned faces using
/// `method`, returning each face's patch. Patch IDs run from 0 without gaps.
/// `face_edges` and `edge_faces` must come from [`check_edges`].
pub(super) fn split_faces(
    method: Partition,
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
    face_edges: &[[DirectedEdge; 3]],
    edge_faces: &[[u32; 2]],
    patch_size: usize,
) -> Vec<u32> {
    match method {
        Partition::BreadthFirst => breadth_first(face_edges, edge_faces, patch_size),
        Partition::InputOrder => (0..triangles.len())
            .map(|f| (f / patch_size) as u32)
            .collect(),
        Partition::Morton => {
            let mut order: Vec<(u32, u32)> = morton_codes(positions, triangles)
                .into_iter()
                .zip(0..)
                .collect();
            order.sort_unstable();
            let mut face_patch = vec![0; triangles.len()];
            for (rank, (_, f)) in order.into_iter().enumerate() {
                face_patch[f as usize] = (rank / patch_size) as u32;
            }
            face_patch
        }
    }
}

/// Grows each patch across shared edges, breadth first, from the lowest
/// unassigned face.
fn breadth_first(
    face_edges: &[[DirectedEdge; 3]],
    edge_faces: &[[u32; 2]],
    patch_size: usize,
) -> Vec<u32> {
    let mut face_patch = vec![u32::MAX; face_edges.len()];
    let mut patches = 0;
    for seed in 0..face_edges.len() {
        if face_patch[seed] != u32::MAX {
            continue;
        }
        let id = patches;
        patches += 1;
        let mut size = 1;
        let mut queue = VecDeque::from([seed as u32]);
        face_patch[seed] = id;
        while let Some(f) = queue.pop_front() {
            for d in face_edges[f as usize] {
                for &n in &edge_faces[d.edge as usize] {
                    if n != NO_FACE && size < patch_size && face_patch[n as usize] == u32::MAX {
                        face_patch[n as usize] = id;
                        size += 1;
                        queue.push_back(n);
                    }
                }
            }
        }
    }
    face_patch
}

/// The 30-bit Morton code of each face's centroid in a 1024³ grid over the
/// centroids' bounding box. Matches `shaders/build/morton.wesl` in intent; the
/// GPU may round differently.
fn morton_codes(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<u32> {
    let centroids: Vec<[f32; 3]> = triangles
        .iter()
        .map(|t| std::array::from_fn(|i| t.iter().map(|&v| positions[v as usize][i] / 3.0).sum()))
        .collect();
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for c in &centroids {
        for i in 0..3 {
            low[i] = low[i].min(c[i]);
            high[i] = high[i].max(c[i]);
        }
    }
    let spread = |x: u32| {
        let mut v = x & 0x3ff;
        v = (v | (v << 16)) & 0x0300_00ff;
        v = (v | (v << 8)) & 0x0300_f00f;
        v = (v | (v << 4)) & 0x030c_30c3;
        (v | (v << 2)) & 0x0924_9249
    };
    centroids
        .iter()
        .map(|c| {
            let cell: [u32; 3] = std::array::from_fn(|i| {
                let extent = (high[i] - low[i]).max(1e-30);
                ((c[i] - low[i]) / extent * 1024.0).clamp(0.0, 1023.0) as u32
            });
            spread(cell[0]) | (spread(cell[1]) << 1) | (spread(cell[2]) << 2)
        })
        .collect()
}

/// Reports the lowest vertex that no face uses, or whose faces do not form one
/// fan. `vertex_corners` must group the corners of `triangles` by vertex, and
/// `edges` must come from [`check_edges`] on the same triangles.
pub(crate) fn check_fans(
    vertex_corners: &Rows,
    triangles: &[[u32; 3]],
    edges: &Edges,
) -> Result<(), BuildError> {
    for v in 0..vertex_corners.len() {
        let corners = vertex_corners.row(v);
        if corners.is_empty() {
            return Err(BuildError::IsolatedVertex(v as u32));
        }
        if !is_single_fan(v as u32, corners, triangles, edges) {
            return Err(BuildError::NonManifoldVertex(v as u32));
        }
    }
    Ok(())
}

/// Returns true if the faces at vertex `v` form one fan: a closed ring, or a
/// single strip between two boundary edges. `corners` lists every corner at `v`
/// and must not be empty. `edges` must come from the same triangles.
fn is_single_fan(v: u32, corners: &[u32], triangles: &[[u32; 3]], edges: &Edges) -> bool {
    // The face across the edge in slot `slot` of `face`, or NO_FACE on a boundary.
    let across = |face: u32, slot: usize| {
        let [a, b] = edges.faces[edges.face_edges[face as usize][slot].edge as usize];
        if a == face { b } else { a }
    };
    let corner_of = |face: u32| {
        triangles[face as usize]
            .iter()
            .position(|&x| x == v)
            .unwrap()
    };

    // Turn around `v` from the first face, leaving each face through the edge
    // chosen by `slot`, until reaching a boundary or returning to the start.
    // Returns the number of other faces passed and whether the walk closed.
    let walk = |slot: fn(usize) -> usize| {
        let first = corners[0] / 3;
        let mut face = first;
        let mut passed = 0;
        loop {
            let next = across(face, slot(corner_of(face)));
            if next == NO_FACE {
                return (passed, false);
            }
            if next == first || passed == corners.len() {
                return (passed, true);
            }
            face = next;
            passed += 1;
        }
    };
    // With consistent winding, the edge leaving `v` is in the corner's own slot
    // and the edge arriving at `v` is in the previous slot.
    let (forward, closed) = walk(|p| p);
    if closed {
        return forward + 1 == corners.len();
    }
    let (backward, _) = walk(|p| (p + 2) % 3);
    forward + backward + 1 == corners.len()
}
