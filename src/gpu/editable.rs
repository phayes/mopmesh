//! Editable connectivity: patch storage with spare local slots and bitmasks.
use super::pipeline::check_binding;
use super::{GpuError, INVALID_ID};
use crate::{Mesh, MeshId, Patch};
use wgpu::util::DeviceExt;

/// How much spare room each patch gets for edits.
///
/// Each patch gets `factor` times as many local vertex, edge and face slots as
/// it has elements, rounded up. Edits that need more free slots in a patch than
/// it has are rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EditCapacity {
    factor: f32,
}

impl EditCapacity {
    /// Returns `None` unless `factor` is finite and at least 1.
    pub fn new(factor: f32) -> Option<Self> {
        (factor.is_finite() && factor >= 1.0).then_some(Self { factor })
    }
    pub fn factor(self) -> f32 {
        self.factor
    }
    /// Returns the slot count for `count` elements, or `None` if it exceeds `u32`.
    fn slots(self, count: usize) -> Option<u32> {
        let slots = (count as f64 * f64::from(self.factor)).ceil();
        (slots <= f64::from(u32::MAX)).then_some(slots as u32)
    }
}

/// A factor of 1.5.
impl Default for EditCapacity {
    fn default() -> Self {
        Self { factor: 1.5 }
    }
}

/// Word layout of the editable arena and mask buffer, shared by the builder
/// and the test decoder. All offsets are in `u32` words.
pub(super) mod layout {
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
}

use layout::{PatchRecord, mask_words};

/// Arena and mask-buffer contents for an editable upload.
pub(super) struct EditableWords {
    pub arena: Vec<u32>,
    pub masks: Vec<u32>,
}

/// Per-patch slot capacities and the total arena and mask sizes in words,
/// computed before anything is allocated.
struct Sizes {
    capacities: Vec<[u32; 3]>,
    arena: u64,
    masks: u64,
}

fn sizes(mesh: &Mesh, capacity: EditCapacity) -> Option<Sizes> {
    let owners = u64::from(layout::OWNER_WORDS)
        * (mesh.positions().len() + mesh.edge_count() + mesh.face_count()) as u64;
    let mut arena =
        (layout::HEADER_WORDS + layout::RECORD_WORDS * mesh.patches().len()) as u64 + owners;
    let mut masks = 0u64;
    let mut capacities = Vec::with_capacity(mesh.patches().len());
    for p in mesh.patches() {
        let [v, e, f] = [
            capacity.slots(p.vertex_ids().len())?,
            capacity.slots(p.edge_ids().len())?,
            capacity.slots(p.face_ids().len())?,
        ];
        let words = u64::from(v)
            + u64::from(layout::EV_WORDS + 1) * u64::from(e)
            + u64::from(layout::FE_WORDS + 1) * u64::from(f);
        arena = arena.checked_add(words)?;
        // Used-slot counts, active and owned masks for each kind, and the
        // interior-vertex mask.
        let mask = 3
            + 3 * u64::from(mask_words(v))
            + 2 * u64::from(mask_words(e))
            + 2 * u64::from(mask_words(f));
        masks = masks.checked_add(mask)?;
        capacities.push([v, e, f]);
    }
    Some(Sizes {
        capacities,
        arena,
        masks,
    })
}

/// Returns the editable arena and masks for `mesh`, or an error if either
/// exceeds the device's buffer limits or `u32` word offsets.
pub(super) fn editable_contents(
    mesh: &Mesh,
    capacity: EditCapacity,
    limits: &wgpu::Limits,
) -> Result<EditableWords, GpuError> {
    // Check every size before allocating.
    const ARENA: &str = "editable connectivity arena";
    const MASKS: &str = "editable connectivity masks";
    let too_large = GpuError::LimitsExceeded { resource: ARENA };
    let sizes = sizes(mesh, capacity).ok_or(too_large)?;
    check_binding(limits, sizes.arena.checked_mul(4).ok_or(too_large)?, ARENA)?;
    let too_large = GpuError::LimitsExceeded { resource: MASKS };
    check_binding(
        limits,
        sizes.masks.max(1).checked_mul(4).ok_or(too_large)?,
        MASKS,
    )?;

    // Header, then placeholder records and owner tables filled in below.
    let counts = [mesh.positions().len(), mesh.edge_count(), mesh.face_count()];
    let mut arena = vec![0u32; layout::HEADER_WORDS];
    arena[layout::VERTEX_COUNT] = counts[0] as u32;
    arena[layout::EDGE_COUNT] = counts[1] as u32;
    arena[layout::FACE_COUNT] = counts[2] as u32;
    arena[layout::PATCH_COUNT] = mesh.patches().len() as u32;
    arena[layout::RECORDS] = arena.len() as u32;
    arena.resize(
        layout::HEADER_WORDS + layout::RECORD_WORDS * mesh.patches().len(),
        0,
    );
    let mut owner_offsets = [0u32; 3];
    for (kind, &count) in counts.iter().enumerate() {
        owner_offsets[kind] = arena.len() as u32;
        arena[layout::OWNERS[kind]] = arena.len() as u32;
        arena.resize(
            arena.len() + layout::OWNER_WORDS as usize * count,
            INVALID_ID,
        );
    }

    // Mark each vertex whose faces all belong to one patch.
    let interior_patch = interior_patches(mesh);

    // Store each patch's regions and masks, and its owned elements in the owner tables.
    let mut masks = Vec::with_capacity(sizes.masks as usize);
    for (p, &capacities) in mesh.patches().iter().zip(&sizes.capacities) {
        let record = append_patch(&mut arena, &mut masks, p, capacities, &interior_patch);
        let start = layout::HEADER_WORDS + layout::RECORD_WORDS * p.id() as usize;
        arena[start..start + layout::RECORD_WORDS].copy_from_slice(&record.to_words());
        let globals = [p.vertex_ids(), p.edge_ids(), p.face_ids()];
        for (kind, owners) in owner_patches(p).iter().enumerate() {
            for (local, (&global, &owner)) in globals[kind].iter().zip(owners).enumerate() {
                if owner == p.id() {
                    let at = (owner_offsets[kind] + layout::OWNER_WORDS * global) as usize;
                    arena[at..at + 2].copy_from_slice(&[p.id(), local as u32]);
                }
            }
        }
    }
    debug_assert_eq!(arena.len() as u64, sizes.arena);
    debug_assert_eq!(masks.len() as u64, sizes.masks);
    if masks.is_empty() {
        masks.push(0);
    }
    Ok(EditableWords { arena, masks })
}

/// Returns, for each global vertex, the patch owning all of its faces, or
/// `INVALID_ID` if its faces belong to more than one patch.
fn interior_patches(mesh: &Mesh) -> Vec<u32> {
    const UNSET: u32 = INVALID_ID - 1;
    let mut patch_of = vec![UNSET; mesh.positions().len()];
    for p in mesh.patches() {
        for local in 0..p.face_ids().len() as u32 {
            if p.face_owners()[local as usize].patch != p.id() {
                continue;
            }
            for v in p.face_vertices(local).unwrap() {
                let state = &mut patch_of[p.vertex_ids()[v as usize] as usize];
                if *state == UNSET {
                    *state = p.id();
                } else if *state != p.id() {
                    *state = INVALID_ID;
                }
            }
        }
    }
    patch_of
}

/// Returns the owner patch of each local vertex, edge and face of `p`.
fn owner_patches(p: &Patch) -> [Vec<u32>; 3] {
    [
        p.vertex_owners().iter().map(|h| h.patch).collect(),
        p.edge_owners().iter().map(|h| h.patch).collect(),
        p.face_owners().iter().map(|h| h.patch).collect(),
    ]
}

/// Appends one patch's regions to `arena` and its masks to `masks`, and
/// returns its record. Slots past the element count hold `INVALID_ID`.
fn append_patch(
    arena: &mut Vec<u32>,
    masks: &mut Vec<u32>,
    p: &Patch,
    [cv, ce, cf]: [u32; 3],
    interior_patch: &[u32],
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

    // Connectivity regions, padded to capacity.
    let vertex_ids = region(arena, p.vertex_ids().iter().copied(), cv);
    let ev = region(
        arena,
        p.edge_vertices().iter().flatten().copied(),
        layout::EV_WORDS * ce,
    );
    let fe_words = p
        .face_edges()
        .iter()
        .flatten()
        .flat_map(|d| [d.edge, u32::from(d.reversed)]);
    let fe = region(arena, fe_words, layout::FE_WORDS * cf);
    let edge_ids = region(arena, p.edge_ids().iter().copied(), ce);
    let face_ids = region(arena, p.face_ids().iter().copied(), cf);

    // Active slots are the first `count` of each kind; owned and interior bits follow ownership.
    let id = p.id();
    let lens = [p.vertex_ids().len(), p.edge_ids().len(), p.face_ids().len()];
    let caps = [cv, ce, cf];
    let used = masks.len() as u32;
    masks.extend(lens.map(|n| n as u32));
    let active = [0, 1, 2].map(|k| mask(masks, (0..lens[k]).map(|_| true), caps[k]));
    let owners = owner_patches(p);
    let owned = [0, 1, 2].map(|k| mask(masks, owners[k].iter().map(|&o| o == id), caps[k]));
    let interior = mask(
        masks,
        p.vertex_ids()
            .iter()
            .map(|&v| interior_patch[v as usize] == id),
        cv,
    );
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

/// Mesh connectivity uploaded with spare room for topology edits.
///
/// Holds two storage buffers: the connectivity arena and its bitmasks. Both
/// support `STORAGE | COPY_SRC`. The layouts are experimental and may change.
pub struct GpuEditableMesh {
    arena: wgpu::Buffer,
    masks: wgpu::Buffer,
    vertex_count: u32,
    edge_count: u32,
    face_count: u32,
    mesh: MeshId,
}

impl GpuEditableMesh {
    /// Uploads `mesh`'s connectivity with the spare room `capacity` asks for.
    /// Returns [`GpuError::LimitsExceeded`] before allocating GPU buffers if
    /// the connectivity is too large for the device.
    pub fn upload(
        device: &wgpu::Device,
        mesh: &Mesh,
        capacity: EditCapacity,
    ) -> Result<Self, GpuError> {
        let words = editable_contents(mesh, capacity, &device.limits())?;
        let buffer = |label, words: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        Ok(Self {
            arena: buffer("MOPMesh editable connectivity", &words.arena),
            masks: buffer("MOPMesh editable connectivity masks", &words.masks),
            vertex_count: words.arena[layout::VERTEX_COUNT],
            edge_count: words.arena[layout::EDGE_COUNT],
            face_count: words.arena[layout::FACE_COUNT],
            mesh: mesh.id(),
        })
    }

    /// Header: vertex, edge, face and patch counts, then the word offsets of
    /// the patch records and of the vertex, edge and face owner tables.
    ///
    /// Each 16-word patch record holds the vertex, edge and face slot
    /// capacities; the arena offsets of its global vertex IDs, edge vertices
    /// (two local vertex slots per edge), face edges (three pairs of local edge
    /// slot and reversed 0/1 per face), global edge IDs and global face IDs;
    /// then the mask-buffer offsets of its active vertex, edge and face masks,
    /// its owned vertex, edge and face masks, its interior-vertex mask, and its
    /// used-slot counts. Each owner-table entry is the patch and local slot
    /// that owns that global ID.
    ///
    /// Unused slots hold [`INVALID_ID`](super::INVALID_ID). All offsets are in
    /// `u32` words.
    pub fn arena(&self) -> &wgpu::Buffer {
        &self.arena
    }
    /// Per-patch slot state: bitmasks with one bit per local slot, packed into
    /// `u32` words starting at the low bit, and used-slot counts.
    /// [`arena`](Self::arena) gives each entry's word offset.
    ///
    /// An active slot holds an element, an owned slot holds the element's
    /// owning copy, and an interior vertex has every face around it owned by
    /// that patch. The used-slot counts are three words, for vertices, edges
    /// and faces: every active slot is below its count. For a mesh with no
    /// patches, the buffer is one zero word.
    pub fn masks(&self) -> &wgpu::Buffer {
        &self.masks
    }
    /// Returns the ID of the CPU mesh used for this upload.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
}

#[cfg(test)]
mod tests {
    use super::layout::{self, PatchRecord};
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::test_support::{self, fixtures};

    /// One patch as stored in the editable arena.
    struct Decoded {
        record: PatchRecord,
        used: [u32; 3],
    }

    fn bit(masks: &[u32], offset: u32, slot: u32) -> bool {
        masks[(offset + slot / 32) as usize] & (1 << (slot % 32)) != 0
    }

    fn patches(words: &EditableWords) -> Vec<Decoded> {
        let a = &words.arena;
        let start = a[layout::RECORDS] as usize;
        (0..a[layout::PATCH_COUNT] as usize)
            .map(|p| {
                let at = start + p * layout::RECORD_WORDS;
                let record = PatchRecord::from_words(&a[at..at + layout::RECORD_WORDS]);
                let u = record.used as usize;
                let used = [0, 1, 2].map(|k| words.masks[u + k]);
                Decoded { record, used }
            })
            .collect()
    }

    /// Rebuilds each global face's input vertex IDs from the owned face slots.
    fn decode_triangles(words: &EditableWords) -> Vec<[u32; 3]> {
        let a = &words.arena;
        let mut triangles = vec![[INVALID_ID; 3]; a[layout::FACE_COUNT] as usize];
        for Decoded { record: r, .. } in patches(words) {
            for f in 0..r.capacity[2] {
                if !bit(&words.masks, r.active[2], f) || !bit(&words.masks, r.owned[2], f) {
                    continue;
                }
                let global = a[(r.face_ids + f) as usize] as usize;
                triangles[global] = [0, 1, 2].map(|i| {
                    let at = (r.fe + layout::FE_WORDS * f + 2 * i) as usize;
                    let (edge, reversed) = (a[at], a[at + 1]);
                    let local = a[(r.ev + layout::EV_WORDS * edge + reversed) as usize];
                    a[(r.vertex_ids + local) as usize]
                });
            }
        }
        triangles
    }

    fn capacities() -> [EditCapacity; 3] {
        [1.0, 1.5, 2.25].map(|f| EditCapacity::new(f).unwrap())
    }

    #[test]
    fn editable_arena_decodes_to_the_source_triangles() {
        let limits = wgpu::Limits::default();
        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                for capacity in capacities() {
                    let context = format!(
                        "{} at patch size {patch_size}, factor {}",
                        fixture.name,
                        capacity.factor()
                    );
                    let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                    let words = editable_contents(&mesh, capacity, &limits).unwrap();
                    assert_eq!(decode_triangles(&words), fixture.triangles, "{context}");
                }
            }
        }
    }

    #[test]
    fn editable_slots_masks_and_owners_match_the_mesh() {
        let limits = wgpu::Limits::default();
        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                for capacity in capacities() {
                    let context = format!(
                        "{} at patch size {patch_size}, factor {}",
                        fixture.name,
                        capacity.factor()
                    );
                    let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                    let words = editable_contents(&mesh, capacity, &limits).unwrap();
                    check_slots(&mesh, capacity, &words, &context);
                }
            }
        }
    }

    /// Checks capacities, used counts, active bits, unused rows, and that each
    /// owner-table entry points back at an owned slot with the same global ID.
    fn check_slots(mesh: &Mesh, capacity: EditCapacity, words: &EditableWords, context: &str) {
        let (a, m) = (&words.arena, &words.masks);
        let decoded = patches(words);
        assert_eq!(decoded.len(), mesh.patches().len(), "{context}");
        for (p, d) in mesh.patches().iter().zip(&decoded) {
            let r = &d.record;
            let lens = [p.vertex_ids().len(), p.edge_ids().len(), p.face_ids().len()];
            let row_words = [1, layout::EV_WORDS, layout::FE_WORDS];
            let rows = [r.vertex_ids, r.ev, r.fe];
            for k in 0..3 {
                let len = lens[k] as u32;
                assert_eq!(r.capacity[k], capacity.slots(lens[k]).unwrap(), "{context}");
                assert!(r.capacity[k] >= len, "{context}");
                assert_eq!(d.used[k], len, "{context}");
                for slot in 0..r.capacity[k] {
                    assert_eq!(
                        bit(m, r.active[k], slot),
                        slot < len,
                        "{context}: active bit"
                    );
                    if slot >= len {
                        let at = (rows[k] + row_words[k] * slot) as usize;
                        assert!(
                            a[at..at + row_words[k] as usize]
                                .iter()
                                .all(|&w| w == INVALID_ID),
                            "{context}: unused row"
                        );
                    }
                }
            }
        }

        // Every global ID has exactly one owner entry, at an owned slot holding that ID.
        let counts = [mesh.positions().len(), mesh.edge_count(), mesh.face_count()];
        for k in 0..3 {
            let mut owned = vec![0u32; counts[k]];
            for (pid, d) in decoded.iter().enumerate() {
                let r = &d.record;
                let ids = [r.vertex_ids, r.edge_ids, r.face_ids][k];
                for slot in 0..d.used[k] {
                    if bit(m, r.owned[k], slot) {
                        let global = a[(ids + slot) as usize];
                        owned[global as usize] += 1;
                        let at = (a[layout::OWNERS[k]] + layout::OWNER_WORDS * global) as usize;
                        assert_eq!(a[at..at + 2], [pid as u32, slot], "{context}: owner table");
                    }
                }
            }
            assert!(
                owned.iter().all(|&n| n == 1),
                "{context}: one owned copy per element"
            );
        }
    }

    #[test]
    fn interior_bits_match_a_scan_of_each_star() {
        let limits = wgpu::Limits::default();
        let mut seen = [0usize; 2];
        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                let context = format!("{} at patch size {patch_size}", fixture.name);
                let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                let words = editable_contents(&mesh, EditCapacity::default(), &limits).unwrap();
                let a = &words.arena;

                // Each face's owner patch from the CPU mesh, then the patches around each vertex.
                let mut face_patch = vec![INVALID_ID; mesh.face_count()];
                for p in mesh.patches() {
                    for (&f, owner) in p.face_ids().iter().zip(p.face_owners()) {
                        if owner.patch == p.id() {
                            face_patch[f as usize] = p.id();
                        }
                    }
                }
                let mut star_patches = vec![Vec::new(); mesh.positions().len()];
                for (f, triangle) in fixture.triangles.iter().enumerate() {
                    for &v in triangle {
                        star_patches[v as usize].push(face_patch[f]);
                    }
                }
                for (pid, d) in patches(&words).iter().enumerate() {
                    let r = &d.record;
                    for slot in 0..d.used[0] {
                        let v = a[(r.vertex_ids + slot) as usize] as usize;
                        let expected = star_patches[v].iter().all(|&q| q == pid as u32);
                        assert_eq!(
                            bit(&words.masks, r.interior, slot),
                            expected,
                            "{context}: vertex {v}"
                        );
                        seen[usize::from(expected)] += 1;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|&n| n > 0),
            "fixtures cover interior and non-interior slots"
        );
    }

    #[test]
    fn edit_capacity_rejects_factors_below_one_and_non_finite() {
        for factor in [0.0, 0.99, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(EditCapacity::new(factor), None, "{factor}");
        }
        assert_eq!(EditCapacity::new(1.0).unwrap().factor(), 1.0);
        assert_eq!(EditCapacity::default().factor(), 1.5);
    }

    #[test]
    fn editable_upload_checks_limits_before_allocating() {
        let (positions, triangles) = fixtures::grid(9);
        let mesh = MeshBuilder::new()
            .positions(&positions)
            .triangles(&triangles)
            .build()
            .unwrap();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 64,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            editable_contents(&mesh, EditCapacity::default(), &limits).err(),
            Some(GpuError::LimitsExceeded {
                resource: "editable connectivity arena"
            })
        );

        // A huge factor overflows the u32 slot count instead of wrapping.
        let huge = EditCapacity::new(f32::MAX).unwrap();
        assert_eq!(
            editable_contents(&mesh, huge, &wgpu::Limits::default()).err(),
            Some(GpuError::LimitsExceeded {
                resource: "editable connectivity arena"
            })
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn editable_upload_matches_host_contents() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let limits = device.limits();
        for fixture in fixtures::valid() {
            let (mesh, _) = test_support::reference(&fixture, 3, &limits).unwrap();
            let words = editable_contents(&mesh, EditCapacity::default(), &limits).unwrap();
            let gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
            let read = |buffer| {
                let encoder = device.create_command_encoder(&Default::default());
                test_support::read_words(&device, &queue, encoder, buffer)
            };
            assert_eq!(read(gpu.arena()), words.arena, "{}", fixture.name);
            assert_eq!(read(gpu.masks()), words.masks, "{}", fixture.name);
            assert_eq!(gpu.mesh_id(), mesh.id());
            assert_eq!(gpu.face_count() as usize, mesh.face_count());
        }
    }
}
