//! Connectivity passes used by `MeshBuilder::build`. Inputs have already been
//! checked for out-of-range vertex IDs.
use super::BuildError;
use crate::DirectedEdge;

/// Marks a missing face on a boundary edge.
pub(super) const NO_FACE: u32 = u32::MAX;

/// Items `0..count` grouped by a key below `rows`, stored contiguously.
/// Row `r` is `entries[offsets[r]..offsets[r + 1]]`; items keep ascending order
/// within each row.
pub(super) struct Rows {
    offsets: Vec<u32>,
    entries: Vec<u32>,
}
impl Rows {
    /// `count` must fit in `u32`, and `key` must return values below `rows`.
    pub(super) fn group(rows: usize, count: usize, key: impl Fn(usize) -> u32) -> Self {
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
    pub(super) fn row(&self, r: usize) -> &[u32] {
        &self.entries[self.offsets[r] as usize..self.offsets[r + 1] as usize]
    }
    fn row_mut(&mut self, r: usize) -> &mut [u32] {
        &mut self.entries[self.offsets[r] as usize..self.offsets[r + 1] as usize]
    }
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// Returns the lowest face ID that repeats the vertex set of an earlier face.
pub(super) fn find_duplicate_face(vertex_count: usize, triangles: &[[u32; 3]]) -> Option<u32> {
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
pub(super) struct Edges {
    /// Ascending endpoint pair per edge ID.
    pub(super) endpoints: Vec<[u32; 2]>,
    /// The one or two faces using each edge, in input face order; [`NO_FACE`]
    /// fills the second slot on a boundary.
    pub(super) faces: Vec<[u32; 2]>,
    /// For each face and slot `i`, the edge from vertex `i` to vertex `i + 1`.
    pub(super) face_edges: Vec<[DirectedEdge; 3]>,
}

/// Numbers edges in ascending order of their (smaller, larger) endpoint pair.
/// On failure, returns the error caused by the lowest face corner, paired with
/// that corner's face ID, matching a scan of the faces in input order.
pub(super) fn connect_edges(
    vertex_count: usize,
    triangles: &[[u32; 3]],
) -> Result<Edges, (u32, BuildError)> {
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

/// Returns true if the faces at vertex `v` form one fan: a closed ring, or a
/// single strip between two boundary edges. `corners` lists every corner at `v`
/// and must not be empty. `edges` must come from the same triangles.
pub(super) fn is_single_fan(
    v: u32,
    corners: &[u32],
    triangles: &[[u32; 3]],
    edges: &Edges,
) -> bool {
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
