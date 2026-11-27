//! Editable connectivity: patch storage with spare local slots and bitmasks.
use super::pipeline::check_binding;
use super::{GpuError, INVALID_ID};
use crate::{Mesh, MeshId};
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

pub(super) use super::patch_layout as layout;

use layout::{append_patch, owner_patches};

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
        arena = arena.checked_add(layout::patch_region_words([v, e, f]))?;
        masks = masks.checked_add(layout::patch_mask_words([v, e, f], true))?;
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
        let record = append_patch(&mut arena, &mut masks, p, capacities, Some(&interior_patch));
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
    version: u64,
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
            version: 0,
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
    /// Returns the topology version: 0 after upload, and one higher for each
    /// edit recorded since, whether or not the GPU accepts it.
    pub fn version(&self) -> u64 {
        self.version
    }
    /// Counts one recorded edit.
    pub(super) fn bump_version(&mut self) {
        self.version += 1;
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

/// Test helpers that read an editable arena back as sets of global vertex IDs,
/// so an edited arena can be compared with one built from scratch.
#[cfg(test)]
pub(super) mod inspect {
    use super::INVALID_ID;
    use super::layout::{self, PatchRecord};
    use std::collections::BTreeSet;

    /// One patch's active elements, each as global vertex IDs. Faces are
    /// rotated to start at their smallest vertex, keeping their winding;
    /// edges are in stored order, which is ascending on a valid arena.
    #[derive(Debug, PartialEq, Eq, Default)]
    pub struct PatchView {
        pub vertices: BTreeSet<u32>,
        pub edges: BTreeSet<[u32; 2]>,
        pub faces: BTreeSet<[u32; 3]>,
        pub owned_vertices: BTreeSet<u32>,
        pub owned_edges: BTreeSet<[u32; 2]>,
        pub owned_faces: BTreeSet<[u32; 3]>,
        pub interior: BTreeSet<u32>,
    }

    fn bit(masks: &[u32], offset: u32, slot: u32) -> bool {
        masks[(offset + slot / 32) as usize] & (1 << (slot % 32)) != 0
    }

    pub fn records(arena: &[u32]) -> Vec<PatchRecord> {
        let start = arena[layout::RECORDS] as usize;
        (0..arena[layout::PATCH_COUNT] as usize)
            .map(|p| {
                let at = start + p * layout::RECORD_WORDS;
                PatchRecord::from_words(&arena[at..at + layout::RECORD_WORDS])
            })
            .collect()
    }

    fn rotate([a, b, c]: [u32; 3]) -> [u32; 3] {
        let m = a.min(b).min(c);
        if m == a {
            [a, b, c]
        } else if m == b {
            [b, c, a]
        } else {
            [c, a, b]
        }
    }

    /// Decodes every patch. Panics if a slot below a used count but inactive
    /// holds anything other than `INVALID_ID`, or if an active slot is at or
    /// above its used count.
    pub fn patch_views(arena: &[u32], masks: &[u32]) -> Vec<PatchView> {
        records(arena)
            .iter()
            .enumerate()
            .map(|(p, r)| {
                let used = [0, 1, 2].map(|k| masks[r.used as usize + k]);
                let row_words = [1, layout::EV_WORDS, layout::FE_WORDS];
                let rows = [r.vertex_ids, r.ev, r.fe];
                let ids = [r.vertex_ids, r.edge_ids, r.face_ids];
                for k in 0..3 {
                    for slot in 0..r.capacity[k] {
                        let active = bit(masks, r.active[k], slot);
                        assert!(
                            !active || slot < used[k],
                            "patch {p}: active slot past used count"
                        );
                        if !active {
                            let at = (rows[k] + row_words[k] * slot) as usize;
                            let row = &arena[at..at + row_words[k] as usize];
                            assert!(row.iter().all(|&w| w == INVALID_ID), "patch {p}: stale row");
                            assert_eq!(
                                arena[(ids[k] + slot) as usize],
                                INVALID_ID,
                                "patch {p}: stale ID"
                            );
                        }
                    }
                }
                let vertex = |local: u32| arena[(r.vertex_ids + local) as usize];
                let edge = |slot: u32| {
                    let at = (r.ev + layout::EV_WORDS * slot) as usize;
                    [vertex(arena[at]), vertex(arena[at + 1])]
                };
                let face = |slot: u32| {
                    rotate([0, 1, 2].map(|i| {
                        let at = (r.fe + layout::FE_WORDS * slot + 2 * i) as usize;
                        let ev = (r.ev + layout::EV_WORDS * arena[at] + arena[at + 1]) as usize;
                        vertex(arena[ev])
                    }))
                };
                let mut view = PatchView::default();
                for slot in 0..used[0] {
                    if bit(masks, r.active[0], slot) {
                        view.vertices.insert(vertex(slot));
                        if bit(masks, r.owned[0], slot) {
                            view.owned_vertices.insert(vertex(slot));
                        }
                        if bit(masks, r.interior, slot) {
                            view.interior.insert(vertex(slot));
                        }
                    }
                }
                for slot in 0..used[1] {
                    if bit(masks, r.active[1], slot) {
                        view.edges.insert(edge(slot));
                        if bit(masks, r.owned[1], slot) {
                            view.owned_edges.insert(edge(slot));
                        }
                    }
                }
                for slot in 0..used[2] {
                    if bit(masks, r.active[2], slot) {
                        view.faces.insert(face(slot));
                        if bit(masks, r.owned[2], slot) {
                            view.owned_faces.insert(face(slot));
                        }
                    }
                }
                view
            })
            .collect()
    }

    /// Returns the global endpoints of global edge `edge` in stored order,
    /// found through its owner-table entry.
    pub fn edge_endpoints(arena: &[u32], edge: u32) -> [u32; 2] {
        let at = (arena[layout::OWNERS[1]] + layout::OWNER_WORDS * edge) as usize;
        let r = &records(arena)[arena[at] as usize];
        let ev = (r.ev + layout::EV_WORDS * arena[at + 1]) as usize;
        [arena[ev], arena[ev + 1]].map(|v| arena[(r.vertex_ids + v) as usize])
    }

    /// Returns the global vertices of global face `face` in stored corner
    /// order, found through its owner-table entry.
    pub fn face_vertices(arena: &[u32], face: u32) -> [u32; 3] {
        let at = (arena[layout::OWNERS[2]] + layout::OWNER_WORDS * face) as usize;
        let r = &records(arena)[arena[at] as usize];
        let slot = arena[at + 1];
        [0, 1, 2].map(|i| {
            let fe = (r.fe + layout::FE_WORDS * slot + 2 * i) as usize;
            let ev = (r.ev + layout::EV_WORDS * arena[fe] + arena[fe + 1]) as usize;
            arena[(r.vertex_ids + arena[ev]) as usize]
        })
    }

    /// Panics unless every owner-table entry points at an active, owned slot
    /// holding that global ID, and every owned slot has an entry.
    pub fn check_owner_tables(arena: &[u32], masks: &[u32]) {
        let recs = records(arena);
        let counts =
            [layout::VERTEX_COUNT, layout::EDGE_COUNT, layout::FACE_COUNT].map(|i| arena[i]);
        for k in 0..3 {
            let mut owned = 0;
            for global in 0..counts[k] {
                let at = (arena[layout::OWNERS[k]] + layout::OWNER_WORDS * global) as usize;
                let (p, slot) = (arena[at], arena[at + 1]);
                let r = &recs[p as usize];
                let ids = [r.vertex_ids, r.edge_ids, r.face_ids][k];
                assert!(
                    bit(masks, r.active[k], slot),
                    "kind {k} ID {global}: owner slot inactive"
                );
                assert!(
                    bit(masks, r.owned[k], slot),
                    "kind {k} ID {global}: owner slot not owned"
                );
                assert_eq!(
                    arena[(ids + slot) as usize],
                    global,
                    "kind {k}: owner slot holds another ID"
                );
            }
            for r in &recs {
                for slot in 0..r.capacity[k] {
                    owned += u32::from(bit(masks, r.owned[k], slot));
                }
            }
            assert_eq!(owned, counts[k], "kind {k}: one owned slot per global ID");
        }
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
