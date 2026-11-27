//! Storage layout shared by CPU uploads and GPU mesh queries.
//! A patch is a group of faces with copies of nearby elements for local queries.
//! The arena stores connectivity; the mask buffer tracks which slots contain
//! elements and which patch owns each element. Offsets count `u32` words, not bytes.
//! The WESL module `shaders/topology/patch.wesl` reads this same layout.
use super::INVALID_ID;
use crate::Patch;

/// Header words: vertex, edge, face and patch counts, then the offsets of
/// the patch records and of the vertex, edge and face owner tables.
pub const HEADER_WORDS: usize = 8;
pub const VERTEX_COUNT: usize = 0;
pub const EDGE_COUNT: usize = 1;
pub const FACE_COUNT: usize = 2;
pub const PATCH_COUNT: usize = 3;
pub const RECORDS: usize = 4;
pub const OWNERS: [usize; 3] = [5, 6, 7];

/// Words in each patch record.
pub const RECORD_WORDS: usize = 16;
/// Words per local slot in each patch region.
pub const EV_WORDS: u32 = 2;
pub const FE_WORDS: u32 = 6;
/// Words per owner-table entry: patch, then local slot.
pub const OWNER_WORDS: u32 = 2;

/// One patch's capacities and the offsets of its regions.
///
/// `capacity`, `active` and `owned` are indexed vertex, edge, face. Region
/// offsets point into the arena; mask offsets point into the mask buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchRecord {
    pub capacity: [u32; 3],
    pub vertex_ids: u32,
    pub ev: u32,
    pub fe: u32,
    pub edge_ids: u32,
    pub face_ids: u32,
    pub active: [u32; 3],
    pub owned: [u32; 3],
    pub interior: u32,
    /// Mask-buffer offset of three words: one past the highest vertex,
    /// edge and face slot that may be active.
    pub used: u32,
}

impl PatchRecord {
    pub fn to_words(self) -> [u32; RECORD_WORDS] {
        let [cv, ce, cf] = self.capacity;
        let [av, ae, af] = self.active;
        let [ov, oe, of] = self.owned;
        [
            cv,
            ce,
            cf,
            self.vertex_ids,
            self.ev,
            self.fe,
            self.edge_ids,
            self.face_ids,
            av,
            ae,
            af,
            ov,
            oe,
            of,
            self.interior,
            self.used,
        ]
    }
    #[cfg(test)]
    pub fn from_words(w: &[u32]) -> Self {
        Self {
            capacity: [w[0], w[1], w[2]],
            vertex_ids: w[3],
            ev: w[4],
            fe: w[5],
            edge_ids: w[6],
            face_ids: w[7],
            active: [w[8], w[9], w[10]],
            owned: [w[11], w[12], w[13]],
            interior: w[14],
            used: w[15],
        }
    }
}

/// Words in a bitmask with one bit per slot.
pub fn mask_words(slots: u32) -> u32 {
    slots.div_ceil(32)
}

/// Returns the mask-buffer size in `u32` words for vertex, edge and face
/// capacities `[v, e, f]`. Includes three slot bounds and bitmasks marking
/// active and owned slots. Set `interior` to include a mask of vertices whose
/// surrounding faces all belong to this patch.
pub fn patch_mask_words([v, e, f]: [u32; 3], interior: bool) -> u64 {
    let masks = u64::from(mask_words(v)) + u64::from(mask_words(e)) + u64::from(mask_words(f));
    3 + 2 * masks
        + if interior {
            u64::from(mask_words(v))
        } else {
            0
        }
}

/// Returns the connectivity storage size in `u32` words for capacities
/// `[vertices, edges, faces]`. Includes global IDs, each edge's two endpoints,
/// and each face's three edges with their directions; excludes the patch record.
pub fn patch_region_words([v, e, f]: [u32; 3]) -> u64 {
    u64::from(v) + u64::from(EV_WORDS + 1) * u64::from(e) + u64::from(FE_WORDS + 1) * u64::from(f)
}

/// Returns the owning patch ID for each local vertex, edge and face, in
/// that order. Ownership identifies the one patch responsible for an element
/// that may also appear in neighboring patches.
pub fn owner_patches(p: &Patch) -> [Vec<u32>; 3] {
    [
        p.vertex_owners().iter().map(|h| h.patch).collect(),
        p.edge_owners().iter().map(|h| h.patch).collect(),
        p.face_owners().iter().map(|h| h.patch).collect(),
    ]
}

/// Adds a patch's connectivity and slot masks to the two buffers and returns
/// a record containing their offsets. Spare connectivity slots contain `INVALID_ID`.
///
/// `interior_patch` maps each global vertex to the patch owning all its
/// surrounding faces, or `INVALID_ID` if no such patch exists. Passing `None`
/// omits the interior-vertex mask and sets its offset to `INVALID_ID`.
pub fn append_patch(
    arena: &mut Vec<u32>,
    masks: &mut Vec<u32>,
    p: &Patch,
    [cv, ce, cf]: [u32; 3],
    interior_patch: Option<&[u32]>,
) -> PatchRecord {
    fn region(arena: &mut Vec<u32>, words: impl IntoIterator<Item = u32>, len: u32) -> u32 {
        let start = arena.len();
        arena.extend(words);
        arena.resize(start + len as usize, INVALID_ID);
        start as u32
    }
    fn mask(masks: &mut Vec<u32>, bits: impl Iterator<Item = bool>, slots: u32) -> u32 {
        let start = masks.len();
        masks.resize(start + mask_words(slots) as usize, 0);
        for (slot, set) in bits.enumerate() {
            if set {
                masks[start + slot / 32] |= 1 << (slot % 32);
            }
        }
        start as u32
    }

    // Store IDs and connections, leaving space for every reserved slot.
    let vertex_ids = region(arena, p.vertex_ids().iter().copied(), cv);
    let ev = region(
        arena,
        p.edge_vertices().iter().flatten().copied(),
        EV_WORDS * ce,
    );
    let fe_words = p
        .face_edges()
        .iter()
        .flatten()
        .flat_map(|d| [d.edge, u32::from(d.reversed)]);
    let fe = region(arena, fe_words, FE_WORDS * cf);
    let edge_ids = region(arena, p.edge_ids().iter().copied(), ce);
    let face_ids = region(arena, p.face_ids().iter().copied(), cf);

    // Existing elements fill the first slots. Mark those slots active, then
    // mark the elements owned here and vertices whose surrounding faces are owned here.
    let id = p.id();
    let lens = [p.vertex_ids().len(), p.edge_ids().len(), p.face_ids().len()];
    let caps = [cv, ce, cf];
    let used = masks.len() as u32;
    masks.extend(lens.map(|n| n as u32));
    let active = [0, 1, 2].map(|k| mask(masks, (0..lens[k]).map(|_| true), caps[k]));
    let owners = owner_patches(p);
    let owned = [0, 1, 2].map(|k| mask(masks, owners[k].iter().map(|&o| o == id), caps[k]));
    let interior = match interior_patch {
        Some(patch_of) => mask(
            masks,
            p.vertex_ids().iter().map(|&v| patch_of[v as usize] == id),
            cv,
        ),
        None => INVALID_ID,
    };
    PatchRecord {
        capacity: caps,
        vertex_ids,
        ev,
        fe,
        edge_ids,
        face_ids,
        active,
        owned,
        interior,
        used,
    }
}
