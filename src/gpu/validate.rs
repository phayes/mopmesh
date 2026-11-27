//! Test helpers that check static GPU connectivity against input triangles.
//! Checks storage, connections and ownership without requiring the builder to
//! choose a particular grouping of faces or order of elements within a patch.
use super::INVALID_ID;
use super::patch_layout::{self as layout, PatchRecord, mask_words};
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
    vertex_ids: &'a [u32],
    ev: &'a [u32],
    fe: &'a [u32],
    edge_ids: &'a [u32],
    face_ids: &'a [u32],
}

impl Region<'_> {
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

/// Reads one boolean per slot from a packed bitmask. `mask` is an offset in
/// `u32` words; returns `None` if the requested words are outside the buffer.
fn bits(masks: &[u32], mask: u32, slots: u32) -> Option<Vec<bool>> {
    let words = masks.get(mask as usize..(mask + mask_words(slots)) as usize)?;
    Some(
        (0..slots)
            .map(|s| words[s as usize / 32] & (1 << (s % 32)) != 0)
            .collect(),
    )
}

/// Checks a static mesh's connectivity and masks against its input triangles.
/// Pass per-patch sizes in arena order and the maximum owned-face count as
/// `patch_size`. Input triangle indices must be valid for `vertex_count`.
/// Returns `Ok(())` on success or a description of the first failed check.
///
/// Checks buffer layout and lengths, valid element IDs, edge endpoints and
/// face vertex order. All slots must be active, and every element must have
/// exactly one owning patch. Edges and vertices belong to the lowest-numbered
/// patch owning an adjoining face. Each patch must include all faces touching
/// the vertices of its owned faces, so local queries can see their neighbors.
/// No patch may own more than `patch_size` faces. Extra mask words are allowed
/// only if they are zero.
pub fn validate_arena(
    arena: &[u32],
    masks: &[u32],
    vertex_count: usize,
    triangles: &[[u32; 3]],
    patches: &[PatchSizes],
    patch_size: usize,
) -> Result<(), String> {
    // Reconstruct edges from the input triangles. Sort endpoint pairs to
    // assign global edge IDs, and retain the faces touching each edge.
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
    let edge_index: BTreeMap<[u32; 2], u32> = edges
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i as u32))
        .collect();
    let mut vertex_faces = vec![BTreeSet::new(); vertex_count];
    for (f, t) in triangles.iter().enumerate() {
        for &v in t {
            vertex_faces[v as usize].insert(f as u32);
        }
    }

    // Check mesh counts and the offsets required by the static layout.
    let face_count = triangles.len();
    let expected_header = [
        vertex_count as u32,
        edges.len() as u32,
        face_count as u32,
        patches.len() as u32,
        layout::HEADER_WORDS as u32,
        INVALID_ID,
        INVALID_ID,
        INVALID_ID,
    ];
    if arena.get(..layout::HEADER_WORDS) != Some(&expected_header[..]) {
        return Err(format!(
            "header {:?}, expected {expected_header:?}",
            arena.get(..layout::HEADER_WORDS)
        ));
    }

    // Check each patch record and its data. Regions must follow one another
    // in patch order, with no gaps or overlap.
    let mut regions = Vec::new();
    let mut records = Vec::new();
    let mut start = layout::HEADER_WORDS + layout::RECORD_WORDS * patches.len();
    let mut mask_start = 0u32;
    for (p, sizes) in patches.iter().enumerate() {
        let at = layout::HEADER_WORDS + layout::RECORD_WORDS * p;
        let record = PatchRecord::from_words(
            arena
                .get(at..at + layout::RECORD_WORDS)
                .ok_or("arena too short for its records")?,
        );
        let n = [sizes.vertices, sizes.edges, sizes.faces];
        let w = n.map(mask_words);
        let o = mask_start + 3 + w[0] + w[1] + w[2];
        let mut expected = PatchRecord {
            capacity: n,
            vertex_ids: start as u32,
            ev: 0,
            fe: 0,
            edge_ids: 0,
            face_ids: 0,
            active: [
                mask_start + 3,
                mask_start + 3 + w[0],
                mask_start + 3 + w[0] + w[1],
            ],
            owned: [o, o + w[0], o + w[0] + w[1]],
            interior: INVALID_ID,
            used: mask_start,
        };
        expected.ev = expected.vertex_ids + n[0];
        expected.fe = expected.ev + layout::EV_WORDS * n[1];
        expected.edge_ids = expected.fe + layout::FE_WORDS * n[2];
        expected.face_ids = expected.edge_ids + n[1];
        if record != expected {
            return Err(format!(
                "patch {p} record {record:?}, expected {expected:?}"
            ));
        }
        let end = (expected.face_ids + n[2]) as usize;
        let take = |from: u32, len: u32| arena.get(from as usize..(from + len) as usize);
        let region = (|| {
            Some(Region {
                vertex_ids: take(record.vertex_ids, n[0])?,
                ev: take(record.ev, 2 * n[1])?,
                fe: take(record.fe, 6 * n[2])?,
                edge_ids: take(record.edge_ids, n[1])?,
                face_ids: take(record.face_ids, n[2])?,
            })
        })()
        .ok_or(format!("arena too short for patch {p}"))?;
        check_region(p, &region, vertex_count, triangles, &edge_index)?;
        regions.push(region);
        records.push(record);
        start = end;
        mask_start = o + w[0] + w[1] + w[2];
    }
    if arena.len() != start {
        return Err(format!("arena length {}, expected {start}", arena.len()));
    }
    let tail = masks.get(mask_start as usize..).ok_or(format!(
        "masks have {} words, expected at least {mask_start}",
        masks.len()
    ))?;
    if tail.iter().any(|&w| w != 0) {
        return Err("masks have a nonzero word past the last patch".into());
    }

    // Static meshes have no spare slots: slot bounds equal capacities,
    // and every slot must be marked active.
    let mut owned = Vec::new();
    for (p, (record, sizes)) in records.iter().zip(patches).enumerate() {
        let n = [sizes.vertices, sizes.edges, sizes.faces];
        if masks[record.used as usize..record.used as usize + 3] != n {
            return Err(format!("patch {p} used counts differ from its slot counts"));
        }
        // Unused bits in the final mask word must be zero.
        let padding = |mask: u32, slots: u32| match slots % 32 {
            0 => 0,
            used => masks[(mask + slots / 32) as usize] >> used,
        };
        for k in 0..3 {
            let all = bits(masks, record.active[k], n[k]).unwrap();
            if all.contains(&false) || padding(record.active[k], n[k]) != 0 {
                return Err(format!(
                    "patch {p} active mask {k} is not exactly the used slots"
                ));
            }
            if padding(record.owned[k], n[k]) != 0 {
                return Err(format!("patch {p} owned mask {k} has bits past its slots"));
            }
            owned.push(bits(masks, record.owned[k], n[k]).unwrap());
        }
    }
    let owned = |p: usize, kind: usize| &owned[3 * p + kind];

    // Require exactly one owner per face and enforce the per-patch face limit.
    let mut face_patch = vec![u32::MAX; face_count];
    for (p, region) in regions.iter().enumerate() {
        let mut count = 0;
        for (local, &face) in region.face_ids.iter().enumerate() {
            if owned(p, 2)[local] {
                if face_patch[face as usize] != u32::MAX {
                    return Err(format!("face {face} is owned twice"));
                }
                face_patch[face as usize] = p as u32;
                count += 1;
            }
        }
        if count > patch_size {
            return Err(format!("patch {p} owns {count} faces"));
        }
    }
    if let Some(f) = face_patch.iter().position(|&p| p == u32::MAX) {
        return Err(format!("face {f} has no owner"));
    }

    // Require all faces touching each vertex of an owned face to be present
    // in the same patch. These neighboring copies support local queries.
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

    // Each edge must belong to the lowest-numbered patch owning an adjoining
    // face. That patch must contain every face touching the edge.
    let mut edge_seen = vec![false; edges.len()];
    for (p, region) in regions.iter().enumerate() {
        let local_faces = region.edge_faces();
        for (local, &edge) in region.edge_ids.iter().enumerate() {
            if !owned(p, 1)[local] {
                continue;
            }
            if std::mem::replace(&mut edge_seen[edge as usize], true) {
                return Err(format!("edge {edge} is owned twice"));
            }
            let pair = edges[edge as usize];
            let expected_owner = edge_faces[&pair]
                .iter()
                .map(|&f| face_patch[f as usize])
                .min()
                .unwrap() as usize;
            if p != expected_owner {
                return Err(format!(
                    "edge {edge} is owned by patch {p}, expected {expected_owner}"
                ));
            }
            let mut faces_held: Vec<u32> = local_faces[local]
                .iter()
                .map(|&f| region.face_ids[f])
                .collect();
            faces_held.sort_unstable();
            if faces_held != edge_faces[&pair] {
                return Err(format!(
                    "owner patch {p} of edge {edge} holds faces {faces_held:?}"
                ));
            }
        }
    }
    if let Some(e) = edge_seen.iter().position(|&s| !s) {
        return Err(format!("edge {e} has no owner"));
    }

    // Each vertex must have one owner: the lowest-numbered patch owning
    // any face that touches it.
    let mut vertex_seen = vec![false; vertex_count];
    for (p, region) in regions.iter().enumerate() {
        for (local, &v) in region.vertex_ids.iter().enumerate() {
            if !owned(p, 0)[local] {
                continue;
            }
            if std::mem::replace(&mut vertex_seen[v as usize], true) {
                return Err(format!("vertex {v} is owned twice"));
            }
            let expected_owner = vertex_faces[v as usize]
                .iter()
                .map(|&f| face_patch[f as usize])
                .min()
                .unwrap() as usize;
            if p != expected_owner {
                return Err(format!(
                    "vertex {v} is owned by patch {p}, expected {expected_owner}"
                ));
            }
        }
    }
    if let Some(v) = vertex_seen.iter().position(|&s| !s) {
        return Err(format!("vertex {v} has no owner"));
    }
    Ok(())
}

/// Checks one region's IDs and local connectivity against the input.
fn check_region(
    p: usize,
    region: &Region,
    vertex_count: usize,
    triangles: &[[u32; 3]],
    edge_index: &BTreeMap<[u32; 2], u32>,
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

    // Check that each local edge connects valid vertices, stores the smaller
    // global vertex ID first, and matches the input edge's global ID.
    let (vertices, edges) = (region.vertex_ids.len(), region.ev.len() / 2);
    if region.ev.iter().any(|&v| v as usize >= vertices) {
        return Err(format!("patch {p} EV names a missing local vertex"));
    }
    let mut pairs = BTreeSet::new();
    for e in 0..edges {
        let [a, b] = region.edge_vertices(e);
        if a >= b || !pairs.insert([a, b]) {
            return Err(format!("patch {p} local edge {e} ({a}, {b}) is invalid"));
        }
        if edge_index.get(&[a, b]) != Some(&region.edge_ids[e]) {
            return Err(format!(
                "patch {p} local edge {e} ({a}, {b}) has the wrong global ID"
            ));
        }
    }

    // Check face references and directions, then verify that their vertices
    // reproduce the input triangle order.
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
