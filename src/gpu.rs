//! Upload mesh connectivity to the GPU and query connections between mesh elements.
//!
//! Query `encode` methods record commands into your encoder. Submit those commands
//! before reading the results, or use the result buffers in later commands.
//! Create each query, encoder, and input resource on the same `wgpu::Device` passed
//! to the call. Device mismatches are reported by wgpu validation, not [`GpuError`].
use crate::{Mesh, MeshId};
use wgpu::util::DeviceExt;
mod adjacency;
mod attribute;
#[cfg(test)]
mod baseline;
#[cfg(test)]
mod bench;
#[cfg(test)]
mod bench_baseline;
#[cfg(test)]
mod bench_flips;
#[cfg(test)]
mod bench_meshes;
#[cfg(test)]
mod bench_migration;
mod boundary;
mod build;
mod cavity_migrate;
mod cavity_rounds;
#[cfg(test)]
mod cross_patch_tests;
mod edge;
mod editable;
mod flip;
mod geometry;
mod patch_layout;
#[cfg(test)]
mod patch_map_tests;
mod pipeline;
mod reduce;
mod scan;
mod smooth;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod staged;
#[cfg(test)]
mod test_support;
mod topology;
#[cfg(test)]
mod validate;
pub use adjacency::{AdjacencyQuery, MeshAdjacency};
pub use attribute::{
    FaceCentroidQuery, GpuAttribute, GpuEdgeAttribute, GpuFaceAttribute, GpuValue,
    GpuVertexAttribute,
};
pub use boundary::{BoundaryQuery, MeshBoundary};
pub use build::{GpuBuildError, GpuMeshBuilder, GpuMeshInput};
pub use edge::{EdgeConnectivity, EdgeQuery};
pub use editable::{EditCapacity, GpuEditableMesh};
pub use flip::{EdgeFlipQuery, FlipPass, FlipStatus, GpuFlipStatuses};
pub use geometry::{FaceAreaQuery, FaceNormalQuery, VertexNormalQuery};
use pipeline::{WORKGROUP_SIZE, check_binding, workgroups};
pub use reduce::{DotQuery, GpuReduction, ReduceOp, ReduceQuery, ReduceValue};
pub use smooth::{SmoothQuery, Smoothing};
use topology::buffers;
pub use topology::{GpuTopology, TopologyStamp};

/// Marks a missing element in GPU results, with the value `u32::MAX`.
/// For example, an edge on the mesh boundary has no second face. Never use this
/// value as an index into a vertex, face, or edge buffer.
pub const INVALID_ID: u32 = u32::MAX;

/// An input or size error from a GPU upload or query.
/// These errors are returned before the call allocates GPU buffers or records commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuError {
    /// A required buffer or compute dispatch exceeds the device limits, or
    /// its data cannot be indexed with `u32`. `resource` identifies the affected
    /// operation or buffer.
    LimitsExceeded { resource: &'static str },
    /// The inputs belong to different meshes. Use resources with matching mesh IDs.
    MeshMismatch,
    /// A result passed in was computed from connectivity that has changed
    /// since, such as an editable mesh before a later edit, or from a
    /// different editable upload of the same mesh. Recompute it from the
    /// mesh you pass. See [`TopologyStamp`].
    Stale,
    /// A buffer or slice you passed in has too few values, the wrong length,
    /// or lacks a required buffer usage. `resource` identifies which one.
    InvalidBuffer { resource: &'static str },
    /// A slice you passed in holds an element ID that is not less than the
    /// mesh's count of that element. `resource` identifies which slice.
    ElementOutOfRange { resource: &'static str },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitsExceeded { resource } => {
                write!(f, "{resource} exceeds GPU buffer, index or dispatch limits")
            }
            Self::MeshMismatch => f.write_str("GPU resources belong to different meshes"),
            Self::Stale => f.write_str(
                "GPU result was computed from an older version or a different upload of the mesh",
            ),
            Self::InvalidBuffer { resource } => {
                write!(f, "{resource} has the wrong size or buffer usage")
            }
            Self::ElementOutOfRange { resource } => {
                write!(f, "{resource} contains an ID outside the mesh")
            }
        }
    }
}
impl std::error::Error for GpuError {}

/// Mesh connectivity uploaded for GPU queries. The mesh is divided into patches,
/// groups of faces that also include neighboring elements needed by queries.
/// Connectivity is fixed after upload; upload positions and other attributes separately.
///
/// Record queries with your command encoder, then submit them on your queue.
/// Results remain on the GPU unless you copy them back to the CPU.
pub struct GpuMesh {
    arena: wgpu::Buffer,
    masks: wgpu::Buffer,
    face_count: u32,
    vertex_count: u32,
    edge_count: u32,
    patch_count: u32,
    mesh: MeshId,
}

/// Borrows connectivity buffers, mesh counts and the current topology stamp
/// for queries. Static and editable meshes use the same buffer layout.
#[derive(Clone, Copy)]
pub(crate) struct PatchBuffers<'a> {
    pub arena: &'a wgpu::Buffer,
    pub masks: &'a wgpu::Buffer,
    pub patch_count: u32,
    pub vertex_capacity: u32,
    pub face_capacity: u32,
    pub edge_capacity: u32,
    /// Whether some IDs below the capacities may be unused, so outputs must
    /// mark them.
    pub unused_ids: bool,
    pub topology: TopologyStamp,
}

/// CPU-side `u32` data for a static mesh's connectivity and slot masks.
pub(crate) struct ArenaWords {
    pub arena: Vec<u32>,
    pub masks: Vec<u32>,
}

/// Identifies the connectivity buffer in [`GpuError::LimitsExceeded`].
const ARENA_RESOURCE: &str = "connectivity arena";
/// Identifies the slot-state buffer in [`GpuError::LimitsExceeded`].
const MASKS_RESOURCE: &str = "connectivity masks";

/// Returns each patch's `[vertices, edges, faces]` slot counts, including
/// copies of neighboring elements. The mesh builder's size limits ensure
/// these counts fit in `u32`.
fn patch_counts(mesh: &Mesh) -> impl Iterator<Item = [u32; 3]> + '_ {
    mesh.patches()
        .iter()
        .map(|p| [p.vertex_ids().len(), p.edge_ids().len(), p.face_ids().len()].map(|n| n as u32))
}

/// Returns `[arena, masks]` storage sizes in `u32` words. Fails if either
/// buffer would exceed device limits or the supported index range. An empty
/// mask buffer is checked as a one-word allocation but reported as zero words.
fn arena_words(mesh: &Mesh, limits: &wgpu::Limits) -> Result<[u64; 2], GpuError> {
    let mut arena =
        (patch_layout::HEADER_WORDS + patch_layout::RECORD_WORDS * mesh.patches().len()) as u64;
    let mut masks = 0u64;
    for counts in patch_counts(mesh) {
        arena = arena
            .checked_add(patch_layout::patch_region_words(counts))
            .ok_or(GpuError::LimitsExceeded {
                resource: ARENA_RESOURCE,
            })?;
        masks += patch_layout::patch_mask_words(counts, false);
    }
    check_binding(limits, arena.saturating_mul(4), ARENA_RESOURCE)?;
    check_binding(limits, masks.max(1).saturating_mul(4), MASKS_RESOURCE)?;
    Ok([arena, masks])
}

/// Builds the CPU-side words used by [`GpuMesh::upload`], following the layout
/// on [`GpuMesh::arena`]. Returns an error if the required buffers exceed limits.
/// An empty mask buffer receives one zero word so it can be bound to a shader.
fn arena_contents(mesh: &Mesh, limits: &wgpu::Limits) -> Result<ArenaWords, GpuError> {
    use patch_layout as layout;
    let [arena_len, masks_len] = arena_words(mesh, limits)?;

    // Write mesh counts and reserve space for patch records. Mark the
    // owner-table offsets as absent; static queries use ownership masks.
    let mut arena = vec![0u32; layout::HEADER_WORDS];
    let [v, e, f] = layout::ID_CAPACITY;
    arena[v] = mesh.positions().len() as u32;
    arena[e] = mesh.edge_count() as u32;
    arena[f] = mesh.face_count() as u32;
    arena[layout::PATCH_COUNT] = mesh.patches().len() as u32;
    arena[layout::RECORDS] = layout::HEADER_WORDS as u32;
    for owners in layout::OWNERS {
        arena[owners] = INVALID_ID;
    }
    arena.resize(
        layout::HEADER_WORDS + layout::RECORD_WORDS * mesh.patches().len(),
        0,
    );
    arena.reserve(arena_len as usize - arena.len());

    // Store each patch's connectivity and masks without reserving spare slots.
    let mut masks = Vec::with_capacity(masks_len as usize);
    for (p, counts) in mesh.patches().iter().zip(patch_counts(mesh)) {
        let record = layout::append_patch(&mut arena, &mut masks, p, counts, None);
        let start = layout::HEADER_WORDS + layout::RECORD_WORDS * p.id() as usize;
        arena[start..start + layout::RECORD_WORDS].copy_from_slice(&record.to_words());
    }
    debug_assert_eq!(arena.len() as u64, arena_len);
    debug_assert_eq!(masks.len() as u64, masks_len);
    if masks.is_empty() {
        masks.push(0);
    }
    Ok(ArenaWords { arena, masks })
}

impl GpuMesh {
    /// Uploads `mesh`'s fixed connectivity for use by GPU queries.
    /// Upload attributes, such as positions, separately with
    /// [`GpuMesh::upload_attribute`]. Returns [`GpuError::LimitsExceeded`]
    /// before allocating GPU buffers if the connectivity is too large.
    pub fn upload(device: &wgpu::Device, mesh: &Mesh) -> Result<Self, GpuError> {
        let words = arena_contents(mesh, &device.limits())?;
        let buffer = |label, words: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        Ok(Self {
            arena: buffer("MOPMesh static connectivity", &words.arena),
            masks: buffer("MOPMesh static connectivity masks", &words.masks),
            face_count: mesh.face_count() as u32,
            vertex_count: mesh.positions().len() as u32,
            edge_count: mesh.edge_count() as u32,
            patch_count: mesh.patches().len() as u32,
            mesh: mesh.id(),
        })
    }

    /// Returns the buffer describing which vertices, edges and faces are connected.
    /// Use WESL `topology::patch` or `topology::patch_arena` to read it in a shader.
    /// The layout is experimental; offsets count `u32` words, not bytes.
    /// The buffer supports `STORAGE | COPY_SRC`.
    ///
    /// The eight-word header contains vertex, edge, face and patch counts, the
    /// patch-record offset, and three absent owner-table offsets ([`INVALID_ID`]).
    /// Each 16-word patch record contains, in order:
    ///
    /// - vertex, edge and face slot counts;
    /// - arena offsets for global vertex IDs, edge endpoints, face edges, global
    ///   edge IDs and global face IDs;
    /// - mask-buffer offsets for active vertex, edge and face masks, then owned masks;
    /// - [`INVALID_ID`] for the absent interior-vertex mask;
    /// - the mask-buffer offset for three used-slot counts.
    ///
    /// Edge endpoints are two local vertex slots. Each face stores three pairs of
    /// local edge slot and direction: 0 follows the edge, 1 reverses it.
    /// The layout matches [`GpuEditableMesh::arena`], with no spare slots,
    /// owner tables or interior-vertex masks.
    pub fn arena(&self) -> &wgpu::Buffer {
        &self.arena
    }
    /// Returns the buffer tracking active slots and element ownership in each patch.
    /// A slot is a position reserved for a vertex, edge or face. Every slot in this
    /// static mesh is active. Elements may appear in several patches, but exactly
    /// one copy is marked owned, allowing queries to process each element once.
    ///
    /// Each patch stores three used-slot counts, then active and owned bitmasks
    /// in vertex, edge and face order, as in [`GpuEditableMesh::masks`]. Used-slot
    /// counts equal capacities. Trailing zero words may be unused.
    /// The buffer supports `STORAGE | COPY_SRC`.
    pub fn masks(&self) -> &wgpu::Buffer {
        &self.masks
    }
    /// Returns the number of face groups used to process this mesh on the GPU.
    pub fn patch_count(&self) -> u32 {
        self.patch_count
    }
    /// Returns the ID of the CPU mesh used for this upload.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }
}

/// The three input vertex IDs of each face, preserving the input triangle order.
/// On an edited mesh, flipped faces follow the corner order given at
/// [`FlipStatus::Flipped`].
/// The buffer contains packed `u32` triples indexed by input face ID, with 12 bytes
/// per face. It supports `STORAGE | COPY_SRC`. For an empty mesh, the buffer is a
/// four-byte zero placeholder; `face_count()` is zero.
pub struct FaceVertices {
    buffer: wgpu::Buffer,
    face_count: u32,
    topology: TopologyStamp,
}
impl FaceVertices {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// Returns the number of face rows: the mesh's face ID capacity, which
    /// may include unused IDs.
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
    pub fn mesh_id(&self) -> MeshId {
        self.topology.mesh_id()
    }
    /// Returns the stamp of the connectivity this result was computed from.
    pub fn topology(&self) -> TopologyStamp {
        self.topology
    }
}

/// Compile once per device and reuse across meshes uploaded to that device.
pub struct FaceVertexQuery {
    pipeline: wgpu::ComputePipeline,
    fill: pipeline::Fill,
}

impl FaceVertexQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh face vertices",
            include_str!(concat!(env!("OUT_DIR"), "/face_vertices.wgsl")),
        );
        Self {
            pipeline: pipeline::compute_pipeline(
                device,
                "MOPMesh face vertices",
                &module,
                "main",
                None,
            ),
            fill: pipeline::Fill::new(device),
        }
    }

    /// Records a query for the three vertex IDs of every face in `mesh`.
    /// Rows of unused face IDs hold [`INVALID_ID`].
    /// Returns [`GpuError::LimitsExceeded`] before allocating or recording commands
    /// if the result or dispatch is too large. An empty mesh returns an empty
    /// [`FaceVertices`] result without recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &impl GpuTopology,
    ) -> Result<FaceVertices, GpuError> {
        self.encode_patches(device, encoder, buffers(mesh))
    }

    /// Records the same face-vertex query as [`encode`](Self::encode), using
    /// the shared buffer layout of static and editable meshes.
    pub(crate) fn encode_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: PatchBuffers,
    ) -> Result<FaceVertices, GpuError> {
        const RESOURCE: &str = "face vertices";
        let limits = device.limits();
        let bytes = (u64::from(mesh.face_capacity) * 12).max(4);
        check_binding(&limits, bytes, RESOURCE)?;
        let grid = pipeline::group_grid(&limits, mesh.patch_count, RESOURCE)?;
        let result = FaceVertices {
            buffer: pipeline::output_buffer(device, "MOPMesh face vertices result", bytes),
            face_count: mesh.face_capacity,
            topology: mesh.topology,
        };
        if mesh.face_capacity != 0 {
            if mesh.unused_ids {
                let words = 3 * mesh.face_capacity;
                self.fill
                    .encode(device, encoder, &result.buffer, words, INVALID_ID, RESOURCE)?;
            }
            pipeline::dispatch_patches(
                device,
                encoder,
                "MOPMesh face vertices",
                &self.pipeline,
                &[&result.buffer],
                mesh,
                grid,
            );
        }
        Ok(result)
    }
}

/// GPU CSR adjacency indexed by original input vertex ID.
///
/// The buffer contains V + 1 u32 offsets followed by room for 3F u32 face
/// IDs, where V and F are the vertex and face ID capacities. Offsets are
/// relative to the face-ID region. A vertex's incident faces occupy
/// `offsets[v]..offsets[v + 1]`; ordering within each row is unspecified.
/// The last offset is the number of entries written, and unused vertex IDs
/// have empty rows.
/// The buffer has STORAGE and COPY_SRC usage and remains on the GPU.
pub struct VertexFaceAdjacency {
    buffer: wgpu::Buffer,
    vertex_count: u32,
    entry_count: u32,
    topology: TopologyStamp,
}

impl VertexFaceAdjacency {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn mesh_id(&self) -> MeshId {
        self.topology.mesh_id()
    }
    /// Returns the stamp of the connectivity this result was computed from.
    pub fn topology(&self) -> TopologyStamp {
        self.topology
    }
    /// Returns the number of vertex rows: the mesh's vertex ID capacity,
    /// which may include unused IDs.
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
    /// Returns the room for face IDs after the offsets: three per face ID.
    /// The entries written number the last offset, which is smaller when
    /// some face IDs are unused.
    pub fn entry_count(&self) -> u32 {
        self.entry_count
    }
    pub fn entries_byte_offset(&self) -> u64 {
        (u64::from(self.vertex_count) + 1) * 4
    }
}

/// Transpose face vertices on the GPU using count, prefix, and scatter passes.
/// Cache once per device. Prefixes use a hierarchical workgroup scan with
/// recursive block sums. No host adjacency or GPU readback is used.
pub struct VertexFaceQuery {
    face_vertices: FaceVertexQuery,
    scan: scan::ScanPipeline,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
}

impl VertexFaceQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh vertex faces",
            include_str!(concat!(env!("OUT_DIR"), "/vertex_faces.wgsl")),
        );
        let (layout, pipeline_layout) = pipeline::buffer_layout(
            device,
            "MOPMesh vertex faces layout",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let pipelines = [
            ("MOPMesh vertex faces count", "count"),
            ("MOPMesh vertex faces initialize", "initialize"),
            ("MOPMesh vertex faces scatter", "scatter"),
        ]
        .map(|(label, entry)| {
            pipeline::compute_pipeline(device, label, &module, entry, Some(&pipeline_layout))
        });
        Self {
            face_vertices: FaceVertexQuery::new(device),
            scan: scan::ScanPipeline::new(device),
            layout,
            pipelines,
        }
    }

    /// Records a query listing the faces that touch each vertex in `mesh`.
    /// See [`VertexFaceAdjacency`] for the buffer layout and ordering.
    /// Returns [`GpuError::LimitsExceeded`] before allocating or recording commands
    /// if a required buffer or dispatch is too large. An empty mesh returns a
    /// single zero offset without recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &impl GpuTopology,
    ) -> Result<VertexFaceAdjacency, GpuError> {
        self.encode_patches(device, encoder, buffers(mesh))
    }

    /// Records the same query as [`encode`](Self::encode) from borrowed buffers.
    pub(crate) fn encode_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: PatchBuffers,
    ) -> Result<VertexFaceAdjacency, GpuError> {
        // Validate result and temporary buffer sizes, plus dispatch limits, before GPU work.
        const RESOURCE: &str = "vertex-face CSR";
        let limits = device.limits();
        let entries = u64::from(mesh.face_capacity) * 3;
        let words = u64::from(mesh.vertex_capacity) + 1 + entries;
        // The scan covers one more entry than vertices, whose prefix is the
        // total entry count.
        let scan_plan = scan::ScanPlan::new(mesh.vertex_capacity + 1, &limits)?;
        check_binding(&limits, words * 4, RESOURCE)?;
        check_binding(
            &limits,
            (u64::from(mesh.face_capacity) * 12).max(4),
            RESOURCE,
        )?;
        let face_groups = workgroups(&limits, mesh.face_capacity, RESOURCE)?;
        let vertex_groups = (mesh.vertex_capacity + 1).div_ceil(WORKGROUP_SIZE);

        let result = VertexFaceAdjacency {
            buffer: pipeline::output_buffer(device, "MOPMesh vertex faces CSR", words * 4),
            vertex_count: mesh.vertex_capacity,
            entry_count: entries as u32,
            topology: mesh.topology,
        };
        if mesh.face_capacity == 0 {
            return Ok(result);
        }
        let faces = self.face_vertices.encode_patches(device, encoder, mesh)?;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh vertex faces sizes"),
            contents: bytemuck::cast_slice(&[mesh.vertex_capacity, mesh.face_capacity]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // The counting pass starts from zero, as guaranteed for new WebGPU buffers.
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh vertex faces cursors"),
            size: scan_plan.byte_size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh vertex faces bindings",
            &self.layout,
            &[&params, faces.buffer(), &cursors, &result.buffer],
        );

        // Count faces per vertex, convert counts to row offsets, then fill each row with face IDs.
        let [count, initialize, scatter] = &self.pipelines;
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex faces count",
            count,
            &bindings,
            face_groups,
        );
        self.scan.encode(device, encoder, &cursors, &scan_plan);
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex faces initialize",
            initialize,
            &bindings,
            vertex_groups,
        );
        pipeline::dispatch(
            encoder,
            "MOPMesh vertex faces scatter",
            scatter,
            &bindings,
            face_groups,
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;

    #[test]
    fn arena_size_matches_the_documented_word_count() {
        let mesh = MeshBuilder::new()
            .positions(&[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
            .triangles(&[[0, 1, 2], [0, 2, 3]])
            .patch_size(1)
            .build()
            .unwrap();
        // Both patches store the whole square: one owned face and its neighbor.
        // Storage needs 8 header words, 16 per patch record, then 1 per vertex,
        // 3 per edge and 7 per face in each patch. Each patch also needs
        // 3 slot counts and 6 mask words (active and owned for each element kind).
        let expected = 8 + 2 * 16 + 2 * (4 + 3 * 5 + 7 * 2);
        let masks = 2 * (3 + 2 * 3);
        let mut limits = wgpu::Limits::default();
        assert_eq!(arena_words(&mesh, &limits), Ok([expected, masks]));
        limits.max_storage_buffer_binding_size = 4 * expected;
        assert_eq!(arena_words(&mesh, &limits), Ok([expected, masks]));
        limits.max_storage_buffer_binding_size = 4 * expected - 4;
        assert_eq!(
            arena_words(&mesh, &limits),
            Err(GpuError::LimitsExceeded {
                resource: "connectivity arena"
            })
        );
    }

    #[test]
    fn arena_contents_have_the_computed_word_count() {
        let limits = wgpu::Limits::default();
        for fixture in test_support::fixtures::valid() {
            for patch_size in test_support::fixtures::PATCH_SIZES {
                let (mesh, words) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                let [arena, masks] = arena_words(&mesh, &limits).unwrap();
                let context = format!("{}, patch size {patch_size}", fixture.name);
                assert_eq!(words.arena.len() as u64, arena, "{context}");
                assert_eq!(words.masks.len() as u64, masks.max(1), "{context}");
            }
        }
    }

    // Checks the comparison harness end to end: the uploaded arena, read back
    // from the GPU, equals the CPU reference words for every fixture.
    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn uploaded_arena_matches_the_cpu_reference() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let limits = device.limits();
        for fixture in test_support::fixtures::valid() {
            for patch_size in test_support::fixtures::PATCH_SIZES {
                let context = format!("{}, patch size {patch_size}", fixture.name);
                let (mesh, expected) =
                    test_support::reference(&fixture, patch_size, &limits).unwrap();
                let gpu = GpuMesh::upload(&device, &mesh).unwrap();
                let read = |buffer| {
                    let encoder = device.create_command_encoder(&Default::default());
                    test_support::read_words(&device, &queue, encoder, buffer)
                };
                let (arena, masks) = (read(gpu.arena()), read(gpu.masks()));
                test_support::assert_arena_eq(&context, &mesh, &arena, &masks, &expected);
            }
        }
        // Every invalid fixture is rejected by the CPU reference.
        for fixture in test_support::fixtures::invalid() {
            assert!(
                test_support::reference(&fixture, 128, &limits).is_err(),
                "{}",
                fixture.name
            );
        }
    }

    /// The per-patch region sizes of `mesh`, in arena order.
    fn patch_sizes(mesh: &Mesh) -> Vec<validate::PatchSizes> {
        mesh.patches()
            .iter()
            .map(|p| validate::PatchSizes {
                vertices: p.vertex_ids().len() as u32,
                edges: p.edge_ids().len() as u32,
                faces: p.face_ids().len() as u32,
            })
            .collect()
    }

    #[test]
    fn validator_accepts_cpu_arenas_and_rejects_corrupted_ones() {
        let limits = wgpu::Limits::default();
        let methods = [
            crate::mesh::Partition::BreadthFirst,
            crate::mesh::Partition::InputOrder,
            crate::mesh::Partition::Morton,
        ];
        for fixture in test_support::fixtures::valid() {
            for (patch_size, method) in test_support::fixtures::PATCH_SIZES
                .into_iter()
                .flat_map(|size| methods.map(|method| (size, method)))
            {
                let name = format!("{}, patch size {patch_size}, {method:?}", fixture.name);
                let builder = MeshBuilder::new().patch_size(patch_size).partition(method);
                let (mesh, words) =
                    test_support::reference_with(&fixture, builder, &limits).unwrap();
                let sizes = patch_sizes(&mesh);
                let check = |arena: &[u32], masks: &[u32]| {
                    validate::validate_arena(
                        arena,
                        masks,
                        fixture.positions.len(),
                        &fixture.triangles,
                        &sizes,
                        patch_size,
                    )
                };
                assert_eq!(check(&words.arena, &words.masks), Ok(()), "{name}");

                // Check that corrupting representative arena words produces an error.
                // Limit these checks to small meshes to keep the test quick.
                if method != crate::mesh::Partition::BreadthFirst
                    || mesh.face_count() == 0
                    || mesh.face_count() > 1000
                {
                    continue;
                }
                let records =
                    patch_layout::HEADER_WORDS + patch_layout::RECORD_WORDS * mesh.patches().len();
                for index in [
                    3,
                    patch_layout::HEADER_WORDS + 5,
                    records,
                    words.arena.len() - 1,
                ] {
                    let mut corrupted = words.arena.clone();
                    corrupted[index] ^= 1;
                    assert!(
                        check(&corrupted, &words.masks).is_err(),
                        "{name}: arena word {index}"
                    );
                }
                // Zero padding is allowed. Corrupted slot counts, active or owned
                // bits, and nonzero padding must all be rejected.
                let first = patch_layout::PatchRecord::from_words(
                    &words.arena[patch_layout::HEADER_WORDS..records],
                );
                let mut extended = words.masks.clone();
                extended.push(0);
                assert_eq!(check(&words.arena, &extended), Ok(()), "{name}: zero tail");
                for index in [
                    first.used as usize,
                    first.active[2] as usize,
                    first.owned[2] as usize,
                    extended.len() - 1,
                ] {
                    let mut corrupted = extended.clone();
                    corrupted[index] ^= 1;
                    assert!(
                        check(&words.arena, &corrupted).is_err(),
                        "{name}: mask word {index}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "arena word 40 differs (patch 0 FE, word 0)")]
    fn arena_comparison_names_the_first_differing_region() {
        // Face-edge data starts at word 40: 8 header words, a 16-word patch
        // record, 4 vertex IDs and 12 words describing edge endpoints.
        let fixture = test_support::fixtures::valid()
            .into_iter()
            .find(|f| f.name == "tetrahedron")
            .unwrap();
        let (mesh, expected) =
            test_support::reference(&fixture, 128, &wgpu::Limits::default()).unwrap();
        let mut actual = expected.arena.clone();
        actual[40] ^= 1;
        test_support::assert_arena_eq("tetrahedron", &mesh, &actual, &expected.masks, &expected);
    }
}
