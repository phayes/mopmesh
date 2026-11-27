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
mod bench_meshes;
mod boundary;
mod build;
mod edge;
mod editable;
mod flip;
mod geometry;
mod pipeline;
mod reduce;
mod scan;
mod smooth;
#[cfg(test)]
mod staged;
#[cfg(test)]
mod test_support;
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
pub use flip::{EdgeFlipQuery, FlipStatus, GpuFlipStatus};
pub use geometry::{FaceAreaQuery, FaceNormalQuery, VertexNormalQuery};
use pipeline::{WORKGROUP_SIZE, check_binding, workgroups};
pub use reduce::{DotQuery, GpuReduction, ReduceOp, ReduceQuery, ReduceValue};
pub use smooth::{SmoothQuery, Smoothing};

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
    /// A buffer or slice you passed in has too few values, the wrong length,
    /// or lacks a required buffer usage. `resource` identifies which one.
    InvalidBuffer { resource: &'static str },
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LimitsExceeded { resource } => {
                write!(f, "{resource} exceeds GPU buffer, index or dispatch limits")
            }
            Self::MeshMismatch => f.write_str("GPU resources belong to different meshes"),
            Self::InvalidBuffer { resource } => {
                write!(f, "{resource} has the wrong size or buffer usage")
            }
        }
    }
}
impl std::error::Error for GpuError {}

/// Static connectivity, including ribbons, uploaded to one read-only arena.
///
/// The caller owns the device, queue, submission, and optional readback.
pub struct GpuMesh {
    arena: wgpu::Buffer,
    face_count: u32,
    vertex_count: u32,
    edge_count: u32,
    mesh: MeshId,
}

/// Returns the connectivity buffer size in `u32` words, or an error if it exceeds
/// the device's buffer limits or the supported index range.
fn arena_words(mesh: &Mesh, limits: &wgpu::Limits) -> Result<u64, GpuError> {
    const RESOURCE: &str = "connectivity arena";
    let too_large = GpuError::LimitsExceeded { resource: RESOURCE };
    let mut words = 4 + 4 * mesh.face_count() as u64 + 7 * mesh.edge_count() as u64;
    for p in mesh.patches() {
        words = words
            .checked_add(p.vertex_ids().len() as u64)
            .and_then(|n| n.checked_add(2 * p.edge_vertices().len() as u64))
            .and_then(|n| n.checked_add(7 * p.face_edges().len() as u64))
            .ok_or(too_large)?;
    }
    check_binding(limits, words.checked_mul(4).ok_or(too_large)?, RESOURCE)?;
    Ok(words)
}

/// Returns the connectivity arena for `mesh`, word for word as [`GpuMesh::upload`]
/// stores it (layout documented on [`GpuMesh::arena`]), or an error if it exceeds
/// the device's buffer limits.
fn arena_contents(mesh: &Mesh, limits: &wgpu::Limits) -> Result<Vec<u32>, GpuError> {
    let word_count = arena_words(mesh, limits)?;
    // Counts fit u32: the builder allows at most u32::MAX / 3 faces,
    // each contributing at most three edges, and at most u32::MAX vertices.
    let count = mesh.face_count() as u32;
    let edge_count = mesh.edge_count() as u32;

    // Store each patch's vertex IDs, edge endpoints, face edges, and face IDs.
    // Collect one query task per owned face or edge to avoid duplicate results.
    let mut words = Vec::with_capacity(word_count as usize);
    words.extend([count, 0, edge_count, 0]);
    let mut tasks = Vec::with_capacity(count as usize);
    let mut edge_tasks = Vec::with_capacity(edge_count as usize);
    for p in mesh.patches() {
        let vertex_offset = words.len() as u32;
        words.extend_from_slice(p.vertex_ids());
        let ev_offset = words.len() as u32;
        for edge in p.edge_vertices() {
            words.extend_from_slice(edge);
        }
        let faces_offset = words.len() as u32;
        for (local, face) in p.face_edges().iter().enumerate() {
            let face_offset = words.len() as u32;
            for edge in face {
                words.extend([edge.edge, u32::from(edge.reversed)]);
            }
            if p.face_owners()[local].patch == p.id() {
                tasks.push([ev_offset, face_offset, vertex_offset, p.face_ids()[local]]);
            }
        }
        let face_ids_offset = words.len() as u32;
        words.extend_from_slice(p.face_ids());
        for (local, owner) in p.edge_owners().iter().enumerate() {
            if owner.patch == p.id() {
                edge_tasks.push([
                    local as u32,
                    p.face_ids().len() as u32,
                    faces_offset,
                    face_ids_offset,
                    p.edge_ids()[local],
                    vertex_offset,
                    ev_offset,
                ]);
            }
        }
    }

    // Append the face and edge tasks, and store their starting word offsets in the header.
    words[1] = words.len() as u32;
    for task in tasks {
        words.extend(task);
    }
    words[3] = words.len() as u32;
    for task in edge_tasks {
        words.extend(task);
    }
    Ok(words)
}

impl GpuMesh {
    /// Uploads `mesh`'s fixed connectivity for use by GPU queries.
    /// Upload attributes, such as positions, separately with
    /// [`GpuMesh::upload_attribute`]. Returns [`GpuError::LimitsExceeded`]
    /// before allocating GPU buffers if the connectivity is too large.
    pub fn upload(device: &wgpu::Device, mesh: &Mesh) -> Result<Self, GpuError> {
        let words = arena_contents(mesh, &device.limits())?;
        let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh static connectivity"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Ok(Self {
            arena,
            face_count: words[0],
            vertex_count: mesh.positions().len() as u32,
            edge_count: words[2],
            mesh: mesh.id(),
        })
    }

    /// Scalar arena header: [face count, face task offset, edge count, edge task offset].
    /// Face tasks: [EV word offset, FE word offset, vertex word offset, global face ID].
    /// Edge tasks: [local edge, local face count, FE base, face-ID base,
    /// global edge ID, vertex-ID base, EV base]. Patch data also stores global face IDs.
    /// EV stores two local vertex IDs per edge; FE stores three pairs of
    /// (local edge ID, reversed 0/1) per face; vertices stores global IDs.
    /// All offsets are measured in u32 words. This ABI is experimental.
    /// The buffer supports `STORAGE | COPY_SRC`.
    pub fn arena(&self) -> &wgpu::Buffer {
        &self.arena
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
/// The buffer contains packed `u32` triples indexed by input face ID, with 12 bytes
/// per face. It supports `STORAGE | COPY_SRC`. For an empty mesh, the buffer is a
/// four-byte zero placeholder; `face_count()` is zero.
pub struct FaceVertices {
    buffer: wgpu::Buffer,
    face_count: u32,
    mesh: MeshId,
}
impl FaceVertices {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn face_count(&self) -> u32 {
        self.face_count
    }
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
}

/// Compile once per device and reuse across meshes uploaded to that device.
pub struct FaceVertexQuery {
    pipeline: wgpu::ComputePipeline,
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
        }
    }

    /// Records a query for the three vertex IDs of every face in `mesh`.
    /// Returns [`GpuError::LimitsExceeded`] before allocating or recording commands
    /// if the result or dispatch is too large. An empty mesh returns an empty
    /// [`FaceVertices`] result without recording GPU work.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
    ) -> Result<FaceVertices, GpuError> {
        const RESOURCE: &str = "face vertices";
        let limits = device.limits();
        let bytes = (u64::from(mesh.face_count) * 12).max(4);
        check_binding(&limits, bytes, RESOURCE)?;
        let groups = workgroups(&limits, mesh.face_count, RESOURCE)?;
        let result = FaceVertices {
            buffer: pipeline::output_buffer(device, "MOPMesh face vertices result", bytes),
            face_count: mesh.face_count,
            mesh: mesh.mesh,
        };
        if mesh.face_count != 0 {
            let bindings = pipeline::bind_buffers(
                device,
                "MOPMesh face vertices bindings",
                &self.pipeline.get_bind_group_layout(0),
                &[&mesh.arena, &result.buffer],
            );
            pipeline::dispatch(
                encoder,
                "MOPMesh face vertices",
                &self.pipeline,
                &bindings,
                groups,
            );
        }
        Ok(result)
    }
}

/// GPU CSR adjacency indexed by original input vertex ID.
///
/// The buffer contains V + 1 u32 offsets followed by 3F u32 face IDs. Offsets
/// are relative to the face-ID region. A vertex's incident faces occupy
/// `offsets[v]..offsets[v + 1]`; ordering within each row is unspecified.
/// The buffer has STORAGE and COPY_SRC usage and remains on the GPU.
pub struct VertexFaceAdjacency {
    buffer: wgpu::Buffer,
    vertex_count: u32,
    entry_count: u32,
    mesh: MeshId,
}

impl VertexFaceAdjacency {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
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
        mesh: &GpuMesh,
    ) -> Result<VertexFaceAdjacency, GpuError> {
        // Validate result and temporary buffer sizes, plus dispatch limits, before GPU work.
        const RESOURCE: &str = "vertex-face CSR";
        let limits = device.limits();
        let entries = u64::from(mesh.face_count) * 3;
        let words = u64::from(mesh.vertex_count) + 1 + entries;
        let scan_plan = scan::ScanPlan::new(mesh.vertex_count, &limits)?;
        check_binding(&limits, words * 4, RESOURCE)?;
        check_binding(&limits, (u64::from(mesh.face_count) * 12).max(4), RESOURCE)?;
        let face_groups = workgroups(&limits, mesh.face_count, RESOURCE)?;
        let vertex_groups = mesh.vertex_count.div_ceil(WORKGROUP_SIZE);

        let result = VertexFaceAdjacency {
            buffer: pipeline::output_buffer(device, "MOPMesh vertex faces CSR", words * 4),
            vertex_count: mesh.vertex_count,
            entry_count: entries as u32,
            mesh: mesh.mesh,
        };
        if mesh.face_count == 0 {
            return Ok(result);
        }
        let faces = self.face_vertices.encode(device, encoder, mesh)?;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh vertex faces sizes"),
            contents: bytemuck::cast_slice(&[mesh.vertex_count, mesh.face_count]),
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
        // Count u32 words: 4 for the header, 4 per face task, 7 per edge task,
        // and (vertices + 2 * edges + 7 * faces) per patch. Each patch owns one
        // face and includes the other as a neighbor, so both store the whole square.
        let expected = 4 + 4 * 2 + 7 * 5 + 2 * (4 + 2 * 5 + 7 * 2);
        let mut limits = wgpu::Limits::default();
        assert_eq!(arena_words(&mesh, &limits), Ok(expected));
        limits.max_storage_buffer_binding_size = 4 * expected;
        assert_eq!(arena_words(&mesh, &limits), Ok(expected));
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
                assert_eq!(
                    words.len() as u64,
                    arena_words(&mesh, &limits).unwrap(),
                    "{}, patch size {patch_size}",
                    fixture.name
                );
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
                let encoder = device.create_command_encoder(&Default::default());
                let actual = test_support::read_words(&device, &queue, encoder, gpu.arena());
                test_support::assert_arena_eq(&context, &mesh, &actual, &expected);
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
                let check = |words: &[u32]| {
                    validate::validate_arena(
                        words,
                        fixture.positions.len(),
                        &fixture.triangles,
                        &sizes,
                        patch_size,
                    )
                };
                assert_eq!(check(&words), Ok(()), "{name}");

                // Changing words in each region must be caught; small meshes
                // are enough, and keep the test fast.
                if method != crate::mesh::Partition::BreadthFirst
                    || mesh.face_count() == 0
                    || mesh.face_count() > 1000
                {
                    continue;
                }
                for index in [
                    4,
                    words[1] as usize - 1,
                    words[1] as usize + 3,
                    words.len() - 1,
                ] {
                    let mut corrupted = words.clone();
                    corrupted[index] ^= 1;
                    assert!(check(&corrupted).is_err(), "{name}: word {index}");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "arena differs at word 20 (patch 0 FE, word 0)")]
    fn arena_comparison_names_the_first_differing_region() {
        // The tetrahedron in one patch: 4 header words, 4 vertex IDs and 12 EV
        // words come before the FE records.
        let fixture = test_support::fixtures::valid()
            .into_iter()
            .find(|f| f.name == "tetrahedron")
            .unwrap();
        let (mesh, expected) =
            test_support::reference(&fixture, 128, &wgpu::Limits::default()).unwrap();
        let mut actual = expected.clone();
        actual[20] ^= 1;
        test_support::assert_arena_eq("tetrahedron", &mesh, &actual, &expected);
    }
}
