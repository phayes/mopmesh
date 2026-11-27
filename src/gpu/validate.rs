//! A test-only checker for connectivity arenas, independent of how they were
//! built. It accepts any arena that meets the contract for the given input
//! triangles, whatever patch split or order within patches the builder chose.
use std::collections::{BTreeMap, BTreeSet};

/// The sizes of one patch's region: vertices, edges and faces, including ribbons.
#[derive(Clone, Copy, Debug)]
pub struct PatchSizes {
    pub vertices: u32,
    pub edges: u32,
    pub faces: u32,
}

/// One decoded patch region.
struct Region<'a> {
    start: usize,
    vertex_ids: &'a [u32],
    ev: &'a [u32],
    fe: &'a [u32],
    face_ids: &'a [u32],
}

impl Region<'_> {
    fn ev_offset(&self) -> usize {
        self.start + self.vertex_ids.len()
    }
    fn fe_offset(&self) -> usize {
        self.ev_offset() + self.ev.len()
    }
    fn face_ids_offset(&self) -> usize {
        self.fe_offset() + self.fe.len()
    }
    /// The global endpoints of local edge `e`, in stored order.
    fn edge_vertices(&self, e: usize) -> [u32; 2] {
        [0, 1].map(|i| self.vertex_ids[self.ev[2 * e + i] as usize])
    }
    /// The global vertices of local face `f`, in winding order.
    fn face_vertices(&self, f: usize) -> [u32; 3] {
        [0, 1, 2].map(|i| {
            let edge = self.fe[6 * f + 2 * i] as usize;
            let reversed = self.fe[6 * f + 2 * i + 1] as usize;
            self.vertex_ids[self.ev[2 * edge + reversed] as usize]
        })
    }
    /// For each local edge, the local faces that use it. Local edge IDs must
    /// already be checked to be in range.
    fn edge_faces(&self) -> Vec<Vec<usize>> {
        let mut faces = vec![Vec::new(); self.ev.len() / 2];
        for f in 0..self.face_ids.len() {
            for i in 0..3 {
                faces[self.fe[6 * f + 2 * i] as usize].push(f);
            }
        }
        faces
    }
}

/// Checks `arena` against `triangles` (over `vertex_count` vertices) and the
/// claimed per-patch sizes, in arena order. Returns a description of the first
/// violation. Checks:
/// - the header, region layout and total length;
/// - every patch's IDs are in range and distinct, and its local EV/FE map back
///   to the input faces' vertices in winding order;
/// - every patch holds the full vertex star of each vertex of its owned faces;
/// - each face and each edge has exactly one task, and edge IDs follow
///   ascending (smaller, larger) vertex pairs;
/// - each edge is owned by the lowest patch owning one of its faces, which
///   holds all of its faces;
/// - no patch owns more than `patch_size` faces.
pub fn validate_arena(
    arena: &[u32],
    vertex_count: usize,
    triangles: &[[u32; 3]],
    patches: &[PatchSizes],
    patch_size: usize,
) -> Result<(), String> {
    // The expected edges, numbered in ascending pair order, and their faces.
    let mut edge_faces = BTreeMap::<[u32; 2], Vec<u32>>::new();
    for (f, t) in triangles.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            edge_faces
                .entry([a.min(b), a.max(b)])
                .or_default()
                .push(f as u32);
        }
    }
    let edges: Vec<[u32; 2]> = edge_faces.keys().copied().collect();
    let mut vertex_faces = vec![BTreeSet::new(); vertex_count];
    for (f, t) in triangles.iter().enumerate() {
        for &v in t {
            vertex_faces[v as usize].insert(f as u32);
        }
    }

    // Header and layout.
    let face_count = triangles.len();
    let regions_end = 4 + patches
        .iter()
        .map(|p| (p.vertices + 2 * p.edges + 7 * p.faces) as usize)
        .sum::<usize>();
    let face_tasks = regions_end;
    let edge_tasks = face_tasks + 4 * face_count;
    let expected_header = [face_count, face_tasks, edges.len(), edge_tasks].map(|x| x as u32);
    if arena.get(..4) != Some(&expected_header[..]) {
        return Err(format!(
            "header {:?}, expected {expected_header:?}",
            arena.get(..4)
        ));
    }
    if arena.len() != edge_tasks + 7 * edges.len() {
        return Err(format!(
            "length {}, expected {}",
            arena.len(),
            edge_tasks + 7 * edges.len()
        ));
    }

    // Decode and check each region on its own.
    let mut regions = Vec::new();
    let mut start = 4;
    for (p, sizes) in patches.iter().enumerate() {
        let take = |from: usize, len: u32| &arena[from..from + len as usize];
        let vertex_ids = take(start, sizes.vertices);
        let ev = take(start + vertex_ids.len(), 2 * sizes.edges);
        let fe = take(start + vertex_ids.len() + ev.len(), 6 * sizes.faces);
        let face_ids = take(start + vertex_ids.len() + ev.len() + fe.len(), sizes.faces);
        let region = Region {
            start,
            vertex_ids,
            ev,
            fe,
            face_ids,
        };
        check_region(p, &region, vertex_count, triangles, &edge_faces)?;
        start = region.face_ids_offset() + face_ids.len();
        regions.push(region);
    }

    // Face tasks: one per face (there are exactly as many tasks as faces, so
    // rejecting repeats covers every face), pointing at a patch that holds it.
    let by_start: BTreeMap<usize, usize> = regions
        .iter()
        .enumerate()
        .map(|(p, r)| (r.start, p))
        .collect();
    let mut face_patch = vec![u32::MAX; face_count];
    let mut owned = vec![0usize; patches.len()];
    for task in arena[face_tasks..edge_tasks].chunks(4) {
        let &[ev, fe, vertices, face] = task else {
            unreachable!()
        };
        let p = *by_start
            .get(&(vertices as usize))
            .ok_or(format!("face task {task:?} has no patch"))?;
        let region = &regions[p];
        let local = (fe as usize)
            .checked_sub(region.fe_offset())
            .filter(|x| x.is_multiple_of(6) && x / 6 < region.face_ids.len())
            .map(|x| x / 6)
            .ok_or(format!("face task {task:?} FE offset outside patch {p}"))?;
        if ev as usize != region.ev_offset() || region.face_ids[local] != face {
            return Err(format!("face task {task:?} does not match patch {p}"));
        }
        let owner = face_patch
            .get_mut(face as usize)
            .ok_or(format!("face task for missing face {face}"))?;
        if *owner != u32::MAX {
            return Err(format!("face {face} has two tasks"));
        }
        *owner = p as u32;
        owned[p] += 1;
    }
    if let Some(p) = owned.iter().position(|&n| n > patch_size) {
        return Err(format!("patch {p} owns {} faces", owned[p]));
    }

    // Ribbons: each owned face's vertex stars are present in its patch.
    let held: Vec<BTreeSet<u32>> = regions
        .iter()
        .map(|r| r.face_ids.iter().copied().collect())
        .collect();
    for (f, &p) in face_patch.iter().enumerate() {
        for &v in &triangles[f] {
            if !vertex_faces[v as usize].is_subset(&held[p as usize]) {
                return Err(format!(
                    "patch {p} owns face {f} but lacks part of vertex {v}'s star"
                ));
            }
        }
    }

    // Edge tasks: one per edge ID, owned by the lowest patch owning a face of
    // the edge, which holds all of the edge's faces.
    let local_edge_faces: Vec<_> = regions.iter().map(Region::edge_faces).collect();
    let mut seen = vec![false; edges.len()];
    for task in arena[edge_tasks..].chunks(7) {
        let &[local, faces, fe, face_ids, edge, vertices, ev] = task else {
            unreachable!()
        };
        let pair = *edges
            .get(edge as usize)
            .ok_or(format!("edge task for missing edge {edge}"))?;
        if std::mem::replace(&mut seen[edge as usize], true) {
            return Err(format!("edge {edge} has two tasks"));
        }
        let expected_owner = edge_faces[&pair]
            .iter()
            .map(|&f| face_patch[f as usize])
            .min()
            .unwrap() as usize;
        let region = &regions[expected_owner];
        let layout = [
            faces as usize == region.face_ids.len(),
            fe as usize == region.fe_offset(),
            face_ids as usize == region.face_ids_offset(),
            vertices as usize == region.start,
            ev as usize == region.ev_offset(),
            (local as usize) < region.ev.len() / 2,
        ];
        if layout.contains(&false) {
            return Err(format!(
                "edge task {task:?} does not match owner patch {expected_owner}"
            ));
        }
        if region.edge_vertices(local as usize) != pair {
            return Err(format!("edge {edge} task names the wrong local edge"));
        }
        let mut faces_held: Vec<u32> = local_edge_faces[expected_owner][local as usize]
            .iter()
            .map(|&f| region.face_ids[f])
            .collect();
        faces_held.sort_unstable();
        if faces_held != edge_faces[&pair] {
            return Err(format!(
                "owner patch {expected_owner} of edge {edge} holds faces {faces_held:?}"
            ));
        }
    }
    Ok(())
}

/// Checks one region's IDs and local connectivity against the input.
fn check_region(
    p: usize,
    region: &Region,
    vertex_count: usize,
    triangles: &[[u32; 3]],
    edge_faces: &BTreeMap<[u32; 2], Vec<u32>>,
) -> Result<(), String> {
    let distinct = |ids: &[u32], limit: usize, what: &str| {
        let set: BTreeSet<_> = ids.iter().collect();
        if set.len() != ids.len() || ids.iter().any(|&x| x as usize >= limit) {
            return Err(format!("patch {p} {what} IDs are out of range or repeated"));
        }
        Ok(())
    };
    distinct(region.vertex_ids, vertex_count, "vertex")?;
    distinct(region.face_ids, triangles.len(), "face")?;

    // Local edges are input edges with ascending global endpoints.
    let (vertices, edges) = (region.vertex_ids.len(), region.ev.len() / 2);
    if region.ev.iter().any(|&v| v as usize >= vertices) {
        return Err(format!("patch {p} EV names a missing local vertex"));
    }
    let mut pairs = BTreeSet::new();
    for e in 0..edges {
        let [a, b] = region.edge_vertices(e);
        if a >= b || !edge_faces.contains_key(&[a, b]) || !pairs.insert([a, b]) {
            return Err(format!("patch {p} local edge {e} ({a}, {b}) is invalid"));
        }
    }

    // Local faces name local edges, and map back to the input faces in winding order.
    for f in 0..region.face_ids.len() {
        let record = &region.fe[6 * f..6 * f + 6];
        if record.chunks(2).any(|d| d[0] as usize >= edges || d[1] > 1) {
            return Err(format!("patch {p} FE of local face {f} is out of range"));
        }
    }
    if let Some(e) = region.edge_faces().iter().position(Vec::is_empty) {
        return Err(format!("patch {p} local edge {e} is used by no face"));
    }
    for f in 0..region.face_ids.len() {
        let global = region.face_ids[f];
        if region.face_vertices(f) != triangles[global as usize] {
            return Err(format!(
                "patch {p} local face {f} does not match face {global}"
            ));
        }
    }
    let used: BTreeSet<u32> = region.ev.iter().copied().collect();
    if used.len() != vertices {
        return Err(format!("patch {p} has a local vertex on no edge"));
    }
    Ok(())
}
