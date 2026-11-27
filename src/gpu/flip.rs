//! Edge flips on editable meshes.
use super::{GpuEditableMesh, GpuError, pipeline};
use wgpu::util::DeviceExt;

/// The outcome of one requested edge flip, as written by the GPU.
///
/// Flipping an edge shared by two triangles replaces it with the other
/// diagonal of the four-sided region they form. Every outcome except
/// [`Flipped`](Self::Flipped) leaves the mesh unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipStatus {
    /// The edge was flipped. It keeps its edge ID and now joins the other two
    /// corners, with endpoints in ascending order. If the edge was `a-b`
    /// (`a < b`) with faces `a -> b -> c` and `b -> a -> d`, those faces keep
    /// their IDs and become `a -> d -> c` and `d -> b -> c`, starting at those
    /// corners. Flipping the edge back restores the triangles but swaps the
    /// two faces' IDs.
    Flipped,
    /// The edge is on the mesh boundary, with only one face.
    Boundary,
    /// A face around one of the four affected vertices belongs to another
    /// patch. Only edges well inside one patch can be flipped.
    NotInterior,
    /// The two vertices the new edge would join are already joined by an edge.
    Connected,
    /// The two faces on the edge don't form a proper four-sided region.
    Degenerate,
    /// A flip selected in the same batch uses one of the affected vertices.
    /// Flips are selected in decreasing priority, and one that reports
    /// [`Connected`](Self::Connected) blocks no other flip.
    LostClaim,
    /// The patch has no free slot for the new edge or faces. Reported by your
    /// own WESL kernels that place flips in free slots with `flip_writes`;
    /// [`EdgeFlipQuery`] and `topology::flip_pass` reuse each flip's own
    /// slots, so they never report it.
    OutOfCapacity,
    /// The requested edge ID is not less than the mesh's edge count. Reported
    /// by your own WESL kernels that use the `FLIP_NOT_AN_EDGE` constant;
    /// [`EdgeFlipQuery::encode`] returns an error for such IDs instead.
    NotAnEdge,
}

impl FlipStatus {
    /// Decodes a status word written by the GPU, or by WESL code using the
    /// `FLIP_*` constants in the `topology::flip` module. Returns `None` for 0
    /// (no flip requested) and for unknown values.
    pub fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            1 => Self::Flipped,
            2 => Self::Boundary,
            3 => Self::NotInterior,
            4 => Self::Connected,
            5 => Self::Degenerate,
            6 => Self::LostClaim,
            7 => Self::OutOfCapacity,
            8 => Self::NotAnEdge,
            _ => return None,
        })
    }
}

/// GPU results for a batch of edge flips: one status word per mesh-wide edge
/// ID.
///
/// After submitting and completing the batch's commands, read the first
/// [`len`](Self::len) `u32` words and decode them with
/// [`FlipStatus::from_word`]. An edge that wasn't requested reads 0, which
/// decodes to `None`. The buffer supports `STORAGE | COPY_SRC`, so it can be
/// used by another shader or copied to a CPU-readable buffer.
#[derive(Clone)]
pub struct GpuFlipStatuses {
    buffer: wgpu::Buffer,
    len: u32,
}
impl GpuFlipStatuses {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// Number of results to read: the mesh's edge count.
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Bindings, results and dispatch sizes for one batch of edge flips chosen by
/// your own WESL rule, using `topology::flip_pass`.
///
/// Create a pass with [`EdgeFlipQuery::begin_pass`], bind it at group 1, and
/// record one dispatch of [`patch_grid`](Self::patch_grid) workgroups. Use a
/// fresh pass for each batch; its state must not be reused.
pub struct FlipPass {
    bind_group: wgpu::BindGroup,
    statuses: GpuFlipStatuses,
    patch_grid: [u32; 2],
}
impl FlipPass {
    /// The bindings to set at group 1 for this batch's dispatch.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    /// The batch's results, one per mesh-wide edge ID, written by the
    /// `topology::flip_pass` functions.
    pub fn statuses(&self) -> &GpuFlipStatuses {
        &self.statuses
    }
    /// Workgroup counts `[x, y]` for the dispatch: one workgroup per patch,
    /// declared with `@workgroup_size(PATCH_LANES)` from `topology::patch`.
    /// Dispatch with a z count of 1. Spare workgroups,
    /// whose `flat_group` index is not below `flip_patch_count`, must return
    /// at once.
    pub fn patch_grid(&self) -> [u32; 2] {
        self.patch_grid
    }
}

/// Records batches of edge flips on a [`GpuEditableMesh`].
/// Create one per device and reuse it for subsequent batches.
pub struct EdgeFlipQuery {
    requests: wgpu::BindGroupLayout,
    pass: wgpu::BindGroupLayout,
    flip: wgpu::ComputePipeline,
}

impl EdgeFlipQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::BufferBindingType::{Storage, Uniform};
        let module = pipeline::shader_module(
            device,
            "MOPMesh edge flip",
            include_str!(concat!(env!("OUT_DIR"), "/edge_flip.wgsl")),
        );

        // Keep the request mask in group 0. Group 1 matches the
        // `topology::flip_pass` bindings used by application shaders too.
        let write = Storage { read_only: false };
        let (requests, _) = pipeline::buffer_layout(
            device,
            "MOPMesh edge flip requests",
            &[Storage { read_only: true }],
        );
        let (pass, _) = pipeline::buffer_layout(
            device,
            "MOPMesh edge flip pass",
            &[write, write, write, write, Uniform, write, write],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh edge flip"),
            bind_group_layouts: &[Some(&requests), Some(&pass)],
            immediate_size: 0,
        });
        Self {
            flip: pipeline::compute_pipeline(
                device,
                "MOPMesh edge flip",
                &module,
                "flip",
                Some(&pipeline_layout),
            ),
            requests,
            pass,
        }
    }

    /// The bind group layout required by WESL shaders importing
    /// `topology::flip_pass`. Place it at index 1 in the pipeline layout;
    /// use index 0 for application bindings.
    pub fn pass_layout(&self) -> &wgpu::BindGroupLayout {
        &self.pass
    }

    /// Prepares bindings and a zeroed status buffer for one batch of flips on
    /// `mesh`, chosen by application shaders that import
    /// `topology::flip_pass`. This does not record GPU work. Record the
    /// batch's dispatch with the returned pass before starting another batch
    /// on the same mesh.
    ///
    /// Every successful call increases [`GpuEditableMesh::version`], including
    /// a pass that is never dispatched. Later batches may choose different
    /// winners among flips that share vertices.
    ///
    /// Returns [`GpuError::LimitsExceeded`] if the batch's buffers or
    /// dispatches exceed the device limits; on error the version is unchanged.
    pub fn begin_pass(
        &self,
        device: &wgpu::Device,
        mesh: &mut GpuEditableMesh,
    ) -> Result<FlipPass, GpuError> {
        // Check buffer sizes and dispatch dimensions before allocating or counting an edit.
        const RESOURCE: &str = "edge flips";
        let limits = device.limits();
        let edges = u64::from(mesh.edge_count());
        let state_bytes = edges * u64::from(STATE_WORDS) * 4;
        for bytes in [state_bytes, edges * 4, u64::from(mesh.vertex_count()) * 4] {
            pipeline::check_binding(&limits, bytes, RESOURCE)?;
        }
        let patch_grid = pipeline::group_grid(&limits, mesh.patch_count(), RESOURCE)?;

        // Fresh buffers provide zeroed claims, partners, state and statuses.
        // Use the current version as the priority seed, then count this batch
        // as an edit. The three trailing parameter words pad the uniform.
        let seed = mesh.version() as u32;
        mesh.mark_edited();
        let words = |n: u64| (n * 4).max(4);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh edge flip parameters"),
            contents: bytemuck::cast_slice(&[seed, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let vertices = u64::from(mesh.vertex_count());
        let claims = pipeline::output_buffer(device, "MOPMesh edge flip claims", words(vertices));
        let partners =
            pipeline::output_buffer(device, "MOPMesh edge flip partners", words(vertices));
        let state = pipeline::output_buffer(device, "MOPMesh edge flip state", state_bytes.max(4));
        let statuses = GpuFlipStatuses {
            buffer: pipeline::output_buffer(device, "MOPMesh edge flip statuses", words(edges)),
            len: mesh.edge_count(),
        };
        let bind_group = pipeline::bind_buffers(
            device,
            "MOPMesh edge flip pass",
            &self.pass,
            &[
                mesh.arena(),
                mesh.masks(),
                &claims,
                &state,
                &params,
                statuses.buffer(),
                &partners,
            ],
        );
        Ok(FlipPass {
            bind_group,
            statuses,
            patch_grid,
        })
    }

    /// Records a batch of flips for the mesh-wide edge IDs in `edges`.
    /// Returns a GPU status buffer with one result per mesh edge, indexed by
    /// edge ID; edges not listed read 0. Submit the encoder's commands to run
    /// the flips and produce the results. Listing an edge more than once is
    /// the same as listing it once.
    ///
    /// Flips in one batch must not share any of their four corner vertices.
    /// Each valid flip gets a priority, and flips are selected in decreasing
    /// priority; one that shares a vertex with an already selected flip
    /// reports [`FlipStatus::LostClaim`] and may be retried later. Priorities
    /// depend on the edge ID and the mesh version, so later calls can select
    /// different winners. Flips reuse their own edge and face slots, so they
    /// need no spare capacity.
    ///
    /// Batches recorded into the encoder run in order. Every successful call
    /// increases [`GpuEditableMesh::version`] when recorded, including an empty
    /// batch or one whose flips are all rejected. An empty batch records no
    /// GPU work.
    ///
    /// Returns [`GpuError::ElementOutOfRange`] if an ID is not less than the
    /// mesh's edge count, or [`GpuError::LimitsExceeded`] if the batch
    /// exceeds the device's buffer or dispatch limits. On error, no work is
    /// recorded and the version is unchanged.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &mut GpuEditableMesh,
        edges: &[u32],
    ) -> Result<GpuFlipStatuses, GpuError> {
        // Check the IDs, then prepare the pass, which counts an edit even
        // when there are no requests.
        const RESOURCE: &str = "edge flip requests";
        let edge_count = mesh.edge_count();
        if edges.iter().any(|&e| e >= edge_count) {
            return Err(GpuError::ElementOutOfRange { resource: RESOURCE });
        }
        let pass = self.begin_pass(device, mesh)?;
        if edges.is_empty() {
            return Ok(pass.statuses);
        }

        // Mark each requested edge in a bit mask.
        let mut mask = vec![0u32; edge_count.div_ceil(32) as usize];
        for &e in edges {
            mask[(e / 32) as usize] |= 1 << (e % 32);
        }
        let requests = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh edge flip requests"),
            contents: bytemuck::cast_slice(&mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bindings = pipeline::bind_buffers(device, RESOURCE, &self.requests, &[&requests]);

        pipeline::dispatch_grid_groups(
            encoder,
            "MOPMesh edge flip",
            &self.flip,
            &[&bindings, pass.bind_group()],
            pass.patch_grid(),
        );
        Ok(pass.statuses)
    }
}

/// Number of `u32` words allocated for each edge's temporary flip state.
/// Keep this equal to `STATE_WORDS` in `shaders/topology/flip_pass.wesl`.
const STATE_WORDS: u32 = 7;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::editable::inspect::{
        PatchView, check_invariants, edge_endpoints as edge_endpoints_after, face_vertices,
        patch_views,
    };
    use crate::gpu::editable::{EditCapacity, editable_contents};
    use crate::gpu::test_support::{self, fixtures};
    use crate::{Mesh, MeshBuilder};
    use alum::{EditableTopology, Handle, HasIterators, HasTopology, PolyMeshF32, Vec3};

    /// What the host expects a flip of one edge to do.
    struct Expected {
        status: FlipStatus,
        /// For a flip: the triangles after it, with face IDs unchanged.
        triangles: Option<Vec<[u32; 3]>>,
        /// For a flip: the sorted endpoints of the new edge.
        new_edge: [u32; 2],
    }

    /// Each global face's owning patch.
    fn face_patches(mesh: &Mesh) -> Vec<u32> {
        let mut patch = vec![u32::MAX; mesh.face_count()];
        for p in mesh.patches() {
            for (&f, owner) in p.face_ids().iter().zip(p.face_owners()) {
                if owner.patch == p.id() {
                    patch[f as usize] = p.id();
                }
            }
        }
        patch
    }

    /// The global endpoints of each global edge.
    fn edge_endpoints(mesh: &Mesh) -> Vec<[u32; 2]> {
        let mut ends = vec![[0; 2]; mesh.edge_count()];
        for p in mesh.patches() {
            for (&e, ev) in p.edge_ids().iter().zip(p.edge_vertices()) {
                ends[e as usize] = ev.map(|v| p.vertex_ids()[v as usize]);
            }
        }
        ends
    }

    /// A flip that passes the CPU reference checks made before claiming:
    /// two faces, all four corners interior to one patch. Whether c and d are
    /// already joined, conflicts with other flips and storage space are
    /// settled afterwards.
    struct Valid {
        /// Patch that stores the edge and contains all affected faces.
        patch: u32,
        /// Mesh-wide vertex IDs: the original edge endpoints, then the new endpoints.
        corners: [u32; 4],
        /// Whether an edge already joins the new endpoints. Such a flip still
        /// claims its corners, and reports Connected only if it wins them.
        connected: bool,
    }

    /// Checks whether edge `u-v` can flip within one patch of the test mesh.
    /// Assumes valid input triangles; storage limits and competing requests
    /// are checked separately.
    fn classify(
        mesh: &Mesh,
        triangles: &[[u32; 3]],
        [u, v]: [u32; 2],
    ) -> Result<Valid, FlipStatus> {
        let tris = triangles;
        let on_edge: Vec<usize> = (0..tris.len())
            .filter(|&f| tris[f].contains(&u) && tris[f].contains(&v))
            .collect();
        if on_edge.len() < 2 {
            return Err(FlipStatus::Boundary);
        }
        let opposite = |f: usize| *tris[f].iter().find(|&&w| w != u && w != v).unwrap();
        let (c, d) = (opposite(on_edge[0]), opposite(on_edge[1]));

        // A flip must stay within one patch, including every face touching its corners.
        let patch = face_patches(mesh);
        let p = patch[on_edge[0]];
        let interior = [u, v, c, d].iter().all(|&w| {
            (0..tris.len())
                .filter(|&f| tris[f].contains(&w))
                .all(|f| patch[f] == p)
        });
        if !interior {
            return Err(FlipStatus::NotInterior);
        }
        Ok(Valid {
            patch: p,
            corners: [u, v, c, d],
            connected: tris.iter().any(|t| t.contains(&c) && t.contains(&d)),
        })
    }

    /// Builds expected triangles by applying the requested flips with Alum,
    /// an independent mesh library. Face IDs remain aligned with the input.
    fn alum_flips(
        positions: &[[f32; 3]],
        triangles: &[[u32; 3]],
        edges: &[[u32; 2]],
    ) -> Vec<[u32; 3]> {
        let mut reference = PolyMeshF32::new();
        for &[x, y, z] in positions {
            reference.add_vertex(Vec3(x, y, z)).unwrap();
        }
        for t in triangles {
            reference.add_face(&t.map(Into::into)).unwrap();
        }
        for &[u, v] in edges {
            let edge = reference.find_halfedge(u.into(), v.into()).unwrap().edge();
            reference.swap_edge_ccw(edge).unwrap();
        }
        reference
            .faces()
            .map(|f| {
                let vs: Vec<u32> = reference.fv_ccw_iter(f).map(|v| v.index()).collect();
                [vs[0], vs[1], vs[2]]
            })
            .collect()
    }

    /// Applies flips by the documented face-ID rule: for edge `[a, b]` in
    /// stored (ascending) order, the face running a -> b -> c becomes
    /// a -> d -> c and the face running b -> a -> d becomes d -> b -> c,
    /// starting at those corners. Triangles are indexed by face ID, and
    /// unflipped triangles keep their input corner order.
    fn documented_flips(triangles: &[[u32; 3]], edges: &[[u32; 2]]) -> Vec<[u32; 3]> {
        let mut tris = triangles.to_vec();
        let runs = |t: &[u32; 3], a, b| (0..3).any(|k| t[k] == a && t[(k + 1) % 3] == b);
        for &[a, b] in edges {
            let first = tris.iter().position(|t| runs(t, a, b)).unwrap();
            let second = tris.iter().position(|t| runs(t, b, a)).unwrap();
            let opposite = |t: &[u32; 3]| *t.iter().find(|&&x| x != a && x != b).unwrap();
            let (c, d) = (opposite(&tris[first]), opposite(&tris[second]));
            tris[first] = [a, d, c];
            tris[second] = [d, b, c];
        }
        tris
    }

    /// Asserts that every global face in `arena` holds the triangle at its ID.
    fn assert_faces_by_id(arena: &[u32], expected: &[[u32; 3]], context: &str) {
        for (face, &t) in expected.iter().enumerate() {
            assert_eq!(
                face_vertices(arena, face as u32),
                t,
                "{context}: face {face}"
            );
        }
    }

    /// Predicts the flip of the edge `u-v` from the input triangles and patch
    /// assignment, using Alum for the flipped triangles. Flips reuse their
    /// own slots, so spare capacity doesn't matter.
    fn expect(mesh: &Mesh, fixture: &fixtures::Fixture, [u, v]: [u32; 2]) -> Expected {
        let reject = |status| Expected {
            status,
            triangles: None,
            new_edge: [0; 2],
        };
        let Valid {
            corners: [_, _, c, d],
            connected,
            ..
        } = match classify(mesh, &fixture.triangles, [u, v]) {
            Ok(valid) => valid,
            Err(status) => return reject(status),
        };
        // A lone request always wins its corners.
        if connected {
            return reject(FlipStatus::Connected);
        }

        Expected {
            status: FlipStatus::Flipped,
            triangles: Some(alum_flips(
                &fixture.positions,
                &fixture.triangles,
                &[[u, v]],
            )),
            new_edge: [c.min(d), c.max(d)],
        }
    }

    /// Views of the editable arena for `triangles`, split into the same
    /// patches as `mesh`.
    fn rebuilt_views(
        mesh: &Mesh,
        fixture: &fixtures::Fixture,
        triangles: &[[u32; 3]],
    ) -> Vec<PatchView> {
        let patches = face_patches(mesh);
        let rebuilt = MeshBuilder::new()
            .positions(&fixture.positions)
            .triangles(triangles)
            .face_patches(&patches)
            .build()
            .unwrap();
        let words =
            editable_contents(&rebuilt, EditCapacity::default(), &wgpu::Limits::default()).unwrap();
        patch_views(&words.arena, &words.masks)
    }

    struct Gpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
        query: EdgeFlipQuery,
    }
    impl Gpu {
        fn new() -> Self {
            let (device, queue) = test_support::device(wgpu::Limits::default());
            let query = EdgeFlipQuery::new(&device);
            Self {
                device,
                queue,
                query,
            }
        }
        fn read(&self, buffer: &wgpu::Buffer) -> Vec<u32> {
            let encoder = self.device.create_command_encoder(&Default::default());
            test_support::read_words(&self.device, &self.queue, encoder, buffer)
        }
        fn flip(&self, mesh: &mut GpuEditableMesh, edge: u32) -> FlipStatus {
            self.flip_all(mesh, &[edge])[0]
        }
        /// Flips the listed edges in one batch and returns each listed edge's
        /// status, in list order. Asserts that unlisted edges report nothing.
        fn flip_all(&self, mesh: &mut GpuEditableMesh, edges: &[u32]) -> Vec<FlipStatus> {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let statuses = self
                .query
                .encode(&self.device, &mut encoder, mesh, edges)
                .unwrap();
            assert_eq!(statuses.len(), mesh.edge_count());
            let words =
                test_support::read_words(&self.device, &self.queue, encoder, statuses.buffer());
            for (e, &w) in words[..statuses.len() as usize].iter().enumerate() {
                assert!(
                    edges.contains(&(e as u32)) || w == 0,
                    "unlisted edge {e} reported {w}"
                );
            }
            edges
                .iter()
                .map(|&e| {
                    let w = words[e as usize];
                    FlipStatus::from_word(w).unwrap_or_else(|| panic!("edge {e}: status {w}"))
                })
                .collect()
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn each_edge_flips_like_alum_or_is_rejected_unchanged() {
        let gpu = Gpu::new();
        let mut seen = std::collections::BTreeSet::new();
        let small = fixtures::valid()
            .into_iter()
            .filter(|f| f.triangles.len() <= 200);
        for fixture in small {
            for patch_size in fixtures::PATCH_SIZES {
                for factor in [1.0, 1.5] {
                    let capacity = EditCapacity::new(factor).unwrap();
                    let context = format!(
                        "{} at patch size {patch_size}, factor {factor}",
                        fixture.name
                    );
                    let mesh = MeshBuilder::new()
                        .patch_size(patch_size)
                        .positions(&fixture.positions)
                        .triangles(&fixture.triangles)
                        .build()
                        .unwrap();
                    let ends = edge_endpoints(&mesh);
                    let mut editable = None;
                    for (edge, &uv) in ends.iter().enumerate() {
                        // Reuse the upload until a flip changes it.
                        let gpu_mesh = editable.get_or_insert_with(|| {
                            GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap()
                        });
                        let before = (gpu.read(gpu_mesh.arena()), gpu.read(gpu_mesh.masks()));
                        let expected = expect(&mesh, &fixture, uv);
                        let status = gpu.flip(gpu_mesh, edge as u32);
                        assert_eq!(status, expected.status, "{context}: edge {edge} {uv:?}");
                        seen.insert(format!("{status:?}"));
                        let (arena, masks) =
                            (gpu.read(gpu_mesh.arena()), gpu.read(gpu_mesh.masks()));
                        match expected.triangles {
                            None => {
                                assert!(
                                    arena == before.0 && masks == before.1,
                                    "{context}: rejected flip of edge {edge} changed the mesh"
                                );
                            }
                            Some(triangles) => {
                                check_invariants(&arena, &masks, &gpu.read(gpu_mesh.maps()));
                                assert_slots_kept(&before.0, &before.1, &arena, &masks, &context);
                                assert_eq!(
                                    patch_views(&arena, &masks),
                                    rebuilt_views(&mesh, &fixture, &triangles),
                                    "{context}: edge {edge} {uv:?}"
                                );
                                // The flipped edge keeps its ID and now joins the opposite corners.
                                assert_eq!(
                                    edge_endpoints_after(&arena, edge as u32),
                                    expected.new_edge,
                                    "{context}: edge {edge}"
                                );
                                assert_faces_by_id(
                                    &arena,
                                    &documented_flips(&fixture.triangles, &[uv]),
                                    &context,
                                );
                                // Upload afresh for the next edge.
                                editable = None;
                            }
                        }
                    }
                }
            }
        }
        for status in ["Flipped", "Boundary", "NotInterior", "Connected"] {
            assert!(seen.contains(status), "fixtures never produced {status}");
        }
    }

    /// The owner tables of an editable arena: two words per mesh-wide vertex,
    /// edge and face.
    fn owner_tables(arena: &[u32]) -> Vec<u32> {
        (0..3)
            .flat_map(|kind| {
                let at = arena[5 + kind] as usize;
                arena[at..at + 2 * arena[kind] as usize].iter().copied()
            })
            .collect()
    }

    /// Asserts that flips kept every slot: the masks and owner tables are
    /// unchanged.
    fn assert_slots_kept(
        arena_before: &[u32],
        masks_before: &[u32],
        arena: &[u32],
        masks: &[u32],
        context: &str,
    ) {
        assert!(masks == masks_before, "{context}: masks changed");
        assert!(
            owner_tables(arena) == owner_tables(arena_before),
            "{context}: owner tables changed"
        );
    }

    /// A 9 x 9 grid in one patch, and its central interior edge.
    fn single_patch_grid() -> (fixtures::Fixture, Mesh, u32) {
        let (positions, triangles) = fixtures::grid(9);
        let fixture = fixtures::Fixture {
            name: "grid 9",
            positions,
            triangles,
        };
        let mesh = MeshBuilder::new()
            .patch_size(128)
            .positions(&fixture.positions)
            .triangles(&fixture.triangles)
            .build()
            .unwrap();
        assert_eq!(mesh.patches().len(), 1);
        // The diagonal of the square whose lower-left corner is (4, 4).
        let diagonal = [4 * 9 + 5, 5 * 9 + 4];
        let edge = edge_endpoints(&mesh)
            .iter()
            .position(|&[a, b]| [a.min(b), a.max(b)] == diagonal)
            .unwrap() as u32;
        (fixture, mesh, edge)
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn flipping_twice_restores_the_mesh_and_counts_versions() {
        let gpu = Gpu::new();
        let capacity = EditCapacity::default();
        let (fixture, mesh, edge) = single_patch_grid();
        let mut editable = GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
        let original = patch_views(&gpu.read(editable.arena()), &gpu.read(editable.masks()));
        let ab = edge_endpoints(&mesh)[edge as usize];
        assert_eq!(editable.version(), 0);
        assert_eq!(gpu.flip(&mut editable, edge), FlipStatus::Flipped);
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        assert_ne!(patch_views(&arena, &masks), original);
        let cd = edge_endpoints_after(&arena, edge);
        assert_faces_by_id(
            &arena,
            &documented_flips(&fixture.triangles, &[ab]),
            "first flip",
        );

        // Flipping back restores the mesh, but swaps the two faces' IDs.
        assert_eq!(gpu.flip(&mut editable, edge), FlipStatus::Flipped);
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_invariants(&arena, &masks, &gpu.read(editable.maps()));
        assert_eq!(patch_views(&arena, &masks), original);
        assert_faces_by_id(
            &arena,
            &documented_flips(&fixture.triangles, &[ab, cd]),
            "second flip",
        );

        // Rejected flips count as edits too: the bottom-left grid edge has
        // only one face.
        let boundary = edge_endpoints(&mesh)
            .iter()
            .position(|&[a, b]| [a.min(b), a.max(b)] == [0, 1])
            .unwrap() as u32;
        assert_eq!(gpu.flip(&mut editable, boundary), FlipStatus::Boundary);
        assert_eq!(editable.version(), 3);

        // An ID outside the mesh is refused before anything is recorded,
        // and doesn't count as an edit.
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let refused = gpu
            .query
            .encode(&gpu.device, &mut encoder, &mut editable, &[edge, u32::MAX]);
        assert!(matches!(refused, Err(GpuError::ElementOutOfRange { .. })));
        assert_eq!(editable.version(), 3);
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn repeated_flips_need_no_spare_slots() {
        // With no spare slots at all, each flip rewrites its own edge and
        // faces, so the masks and owner tables never change.
        let gpu = Gpu::new();
        let capacity = EditCapacity::new(1.0).unwrap();
        let (fixture, mesh, edge) = single_patch_grid();
        let mut editable = GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
        let before = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        let mut flipped = vec![edge_endpoints(&mesh)[edge as usize]];
        for round in 0..20 {
            assert_eq!(
                gpu.flip(&mut editable, edge),
                FlipStatus::Flipped,
                "round {round}"
            );
            // Track the edge's endpoints so its next flip can be replayed.
            flipped.push(edge_endpoints_after(&gpu.read(editable.arena()), edge));
        }
        flipped.pop();
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_invariants(&arena, &masks, &gpu.read(editable.maps()));
        assert_slots_kept(&before.0, &before.1, &arena, &masks, "after 20 flips");
        assert_faces_by_id(
            &arena,
            &documented_flips(&fixture.triangles, &flipped),
            "after 20 flips",
        );
        assert_eq!(
            patch_views(&arena, &masks),
            rebuilt_views(&mesh, &fixture, &fixture.triangles),
            "an even number of flips restores the mesh"
        );
    }

    /// Reproduces the shader's priority for an edge ID on the CPU to predict
    /// winners.
    fn priority(edge: u32, seed: u32) -> u32 {
        let mut x = edge ^ seed;
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    }

    /// Predicted results of one batch.
    struct Prediction {
        /// A status per listed edge, in list order.
        statuses: Vec<FlipStatus>,
        /// The original endpoints of edges expected to flip, each once.
        flipped: Vec<[u32; 2]>,
        /// Number of flips that share a corner with a higher-priority flip
        /// reporting Connected, and still go ahead.
        flipped_past_connected: usize,
    }

    /// Predicts batch results for the original mesh. Eligible flips are
    /// visited in decreasing priority: one whose corner is taken by an
    /// already selected flip reports LostClaim, one whose new edge already
    /// exists reports Connected and takes nothing, and the rest are selected
    /// and take their four corners.
    fn predict_batch(mesh: &Mesh, triangles: &[[u32; 3]], edges: &[u32], seed: u32) -> Prediction {
        let ends = edge_endpoints(mesh);
        let requested: std::collections::BTreeSet<u32> = edges.iter().copied().collect();
        let checked: std::collections::BTreeMap<u32, _> = requested
            .iter()
            .map(|&e| (e, classify(mesh, triangles, ends[e as usize])))
            .collect();

        // Ineligible requests report why; the others are decided greedily.
        let mut outcome = std::collections::BTreeMap::<u32, FlipStatus>::new();
        let mut eligible = Vec::new();
        for (&e, valid) in &checked {
            match valid {
                Err(status) => {
                    outcome.insert(e, *status);
                }
                Ok(v) => eligible.push((e, v)),
            }
        }
        eligible.sort_by_key(|&(e, _)| std::cmp::Reverse(priority(e, seed)));
        let mut taken = std::collections::HashSet::new();
        let mut connected_corners = std::collections::HashSet::new();
        let (mut flipped, mut flipped_past_connected) = (Vec::new(), 0);
        for (e, v) in eligible {
            let status = if v.corners.iter().any(|c| taken.contains(c)) {
                FlipStatus::LostClaim
            } else if v.connected {
                connected_corners.extend(v.corners);
                FlipStatus::Connected
            } else {
                taken.extend(v.corners);
                flipped.push([v.corners[0], v.corners[1]]);
                if v.corners.iter().any(|c| connected_corners.contains(c)) {
                    flipped_past_connected += 1;
                }
                FlipStatus::Flipped
            };
            outcome.insert(e, status);
        }
        Prediction {
            statuses: edges.iter().map(|e| outcome[e]).collect(),
            flipped,
            flipped_past_connected,
        }
    }

    /// Builds small test meshes with two patch sizes to exercise batches
    /// within a single patch and across multiple patches.
    fn batch_meshes() -> Vec<(String, fixtures::Fixture, Mesh)> {
        let small = fixtures::valid()
            .into_iter()
            .filter(|f| f.triangles.len() <= 200);
        small
            .flat_map(|fixture| {
                [32, 128].map(|patch_size| {
                    let mesh = MeshBuilder::new()
                        .patch_size(patch_size)
                        .positions(&fixture.positions)
                        .triangles(&fixture.triangles)
                        .build()
                        .unwrap();
                    (
                        format!("{} at patch size {patch_size}", fixture.name),
                        fixtures::Fixture {
                            name: fixture.name,
                            positions: fixture.positions.clone(),
                            triangles: fixture.triangles.clone(),
                        },
                        mesh,
                    )
                })
            })
            .collect()
    }

    /// A 7 x 7 grid with one interior triangle split in three around a new
    /// vertex. Flipping any edge at that vertex would duplicate an edge of the
    /// original triangle, so those flips report Connected, next to ordinary
    /// grid flips that share their corners.
    fn split_grid() -> fixtures::Fixture {
        let (mut positions, mut triangles) = fixtures::grid(7);
        let [a, b, c] = [24, 25, 31];
        let at = triangles.iter().position(|t| *t == [a, b, c]).unwrap();
        let v = positions.len() as u32;
        let centroid = |i: usize| {
            (positions[a as usize][i] + positions[b as usize][i] + positions[c as usize][i]) / 3.0
        };
        positions.push([centroid(0), centroid(1), centroid(2)]);
        triangles[at] = [a, b, v];
        triangles.extend([[b, c, v], [c, a, v]]);
        fixtures::Fixture {
            name: "split grid",
            positions,
            triangles,
        }
    }

    /// Requests every edge in a reordered list, then adds a duplicate, which
    /// must behave as if listed once.
    fn batch_request(mesh: &Mesh) -> Vec<u32> {
        let count = mesh.edge_count() as u32;
        let mut edges: Vec<u32> = (0..count).map(|i| (i * 7 + 3) % count).collect();
        if count.is_multiple_of(7) {
            edges = (0..count).rev().collect();
        }
        edges.extend(edges.get(count as usize / 2).copied());
        edges
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn batches_flip_exactly_the_predicted_winners_like_alum() {
        let gpu = Gpu::new();
        let capacity = EditCapacity::new(1.0).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        let mut most_in_one_patch = 0;
        let mut flipped_past_connected = 0;
        for (context, fixture, mesh) in batch_meshes() {
            let edges = batch_request(&mesh);
            let mut editable = GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
            let before = (gpu.read(editable.arena()), gpu.read(editable.masks()));
            // Exercise priority selection with a mesh version greater than zero.
            editable.mark_edited();
            let seed = editable.version() as u32;
            let Prediction {
                statuses: expected,
                flipped,
                flipped_past_connected: past,
            } = predict_batch(&mesh, &fixture.triangles, &edges, seed);
            flipped_past_connected += past;
            let statuses = gpu.flip_all(&mut editable, &edges);
            assert_eq!(statuses, expected, "{context}");
            seen.extend(statuses.iter().map(|s| format!("{s:?}")));

            // The selection is maximal: every flip that lost shares a corner
            // with one that flipped.
            let ends = edge_endpoints(&mesh);
            let flipped_corners: std::collections::HashSet<u32> = flipped
                .iter()
                .flat_map(|&uv| classify(&mesh, &fixture.triangles, uv).unwrap().corners)
                .collect();
            for (&edge, status) in edges.iter().zip(&statuses) {
                if *status == FlipStatus::LostClaim {
                    let valid = classify(&mesh, &fixture.triangles, ends[edge as usize]).unwrap();
                    assert!(
                        valid.corners.iter().any(|c| flipped_corners.contains(c)),
                        "{context}: edge {edge} lost to no flip"
                    );
                }
            }

            // Count the flips per patch; the batches have no spare slots.
            let mut per_patch = std::collections::HashMap::<u32, usize>::new();
            for uv in &flipped {
                let valid = classify(&mesh, &fixture.triangles, *uv).unwrap();
                *per_patch.entry(valid.patch).or_default() += 1;
            }
            most_in_one_patch = per_patch
                .values()
                .copied()
                .fold(most_in_one_patch, usize::max);

            // Compare the GPU result with the same flips performed by Alum.
            let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
            check_invariants(&arena, &masks, &gpu.read(editable.maps()));
            assert_slots_kept(&before.0, &before.1, &arena, &masks, &context);
            let triangles = alum_flips(&fixture.positions, &fixture.triangles, &flipped);
            assert_eq!(
                patch_views(&arena, &masks),
                rebuilt_views(&mesh, &fixture, &triangles),
                "{context}"
            );
            assert_faces_by_id(
                &arena,
                &documented_flips(&fixture.triangles, &flipped),
                &context,
            );
            for (&edge, status) in edges.iter().zip(&statuses) {
                if *status == FlipStatus::Flipped {
                    let valid = classify(&mesh, &fixture.triangles, ends[edge as usize]).unwrap();
                    let [_, _, c, d] = valid.corners;
                    assert_eq!(
                        edge_endpoints_after(&arena, edge),
                        [c.min(d), c.max(d)],
                        "{context}: edge {edge}"
                    );
                }
            }
        }
        // Find a batch, by seed, in which a flip goes ahead although a
        // higher-priority flip sharing a corner reported Connected, and check
        // the GPU selects exactly as predicted there too.
        if flipped_past_connected == 0 {
            let split = split_grid();
            let split_mesh = MeshBuilder::new()
                .patch_size(128)
                .positions(&split.positions)
                .triangles(&split.triangles)
                .build()
                .unwrap();
            let candidates = [("split grid".to_owned(), split, split_mesh)];
            'search: for (context, fixture, mesh) in candidates.into_iter().chain(batch_meshes()) {
                let edges = batch_request(&mesh);
                for seed in 2..256 {
                    let prediction = predict_batch(&mesh, &fixture.triangles, &edges, seed);
                    if prediction.flipped_past_connected == 0 {
                        continue;
                    }
                    let mut editable =
                        GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
                    let before = (gpu.read(editable.arena()), gpu.read(editable.masks()));
                    for _ in 0..seed {
                        editable.mark_edited();
                    }
                    let context = format!("{context}, seed {seed}");
                    assert_eq!(
                        gpu.flip_all(&mut editable, &edges),
                        prediction.statuses,
                        "{context}"
                    );
                    let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
                    check_invariants(&arena, &masks, &gpu.read(editable.maps()));
                    assert_slots_kept(&before.0, &before.1, &arena, &masks, &context);
                    let flipped = &prediction.flipped;
                    assert_eq!(
                        patch_views(&arena, &masks),
                        rebuilt_views(
                            &mesh,
                            &fixture,
                            &alum_flips(&fixture.positions, &fixture.triangles, flipped)
                        ),
                        "{context}"
                    );
                    assert_faces_by_id(
                        &arena,
                        &documented_flips(&fixture.triangles, flipped),
                        &context,
                    );
                    flipped_past_connected += prediction.flipped_past_connected;
                    break 'search;
                }
            }
        }
        for status in [
            "Flipped",
            "LostClaim",
            "Boundary",
            "NotInterior",
            "Connected",
        ] {
            assert!(seen.contains(status), "batches never produced {status}");
        }
        assert!(
            most_in_one_patch > 1,
            "no batch flipped two edges in one patch"
        );
        assert!(
            flipped_past_connected > 0,
            "no flip shared a corner with a higher-priority Connected flip"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn batches_in_patches_larger_than_one_staging_chunk_flip_as_predicted() {
        // A 31 x 31 grid in one patch holds 2,610 edges: several chunks of
        // the edge slots the claim stage stages at a time.
        use crate::gpu::test_support::STAGED_EDGES;
        let gpu = Gpu::new();
        let (positions, triangles) = fixtures::grid(31);
        let fixture = fixtures::Fixture {
            name: "grid 31",
            positions,
            triangles,
        };
        let mesh = MeshBuilder::new()
            .patch_size(4096)
            .positions(&fixture.positions)
            .triangles(&fixture.triangles)
            .build()
            .unwrap();
        assert_eq!(mesh.patches().len(), 1);
        let slots = mesh.patches()[0].edge_ids();
        assert!(slots.len() > 2 * STAGED_EDGES);

        // Every edge is requested; the winners must be exactly the predicted ones.
        let mut editable =
            GpuEditableMesh::upload(&gpu.device, &mesh, EditCapacity::new(3.0).unwrap()).unwrap();
        editable.mark_edited();
        let seed = editable.version() as u32;
        let edges = batch_request(&mesh);
        let prediction = predict_batch(&mesh, &fixture.triangles, &edges, seed);
        assert_eq!(gpu.flip_all(&mut editable, &edges), prediction.statuses);

        // Flips were predicted in every staging chunk of the patch.
        let ends = edge_endpoints(&mesh);
        let chunks: std::collections::BTreeSet<usize> = prediction
            .flipped
            .iter()
            .map(|uv| {
                let edge = ends.iter().position(|e| e == uv).unwrap() as u32;
                slots.iter().position(|&g| g == edge).unwrap() / STAGED_EDGES
            })
            .collect();
        assert_eq!(chunks.len(), slots.len().div_ceil(STAGED_EDGES));

        // The mesh matches the same flips made by Alum.
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_invariants(&arena, &masks, &gpu.read(editable.maps()));
        assert_eq!(
            patch_views(&arena, &masks),
            rebuilt_views(
                &mesh,
                &fixture,
                &alum_flips(&fixture.positions, &fixture.triangles, &prediction.flipped)
            )
        );
        assert_faces_by_id(
            &arena,
            &documented_flips(&fixture.triangles, &prediction.flipped),
            "grid 31",
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn retrying_lost_claims_eventually_flips_every_valid_diagonal() {
        let gpu = Gpu::new();
        let (positions, triangles) = fixtures::grid(9);
        for patch_size in [32, 128] {
            let mesh = MeshBuilder::new()
                .patch_size(patch_size)
                .positions(&positions)
                .triangles(&triangles)
                .build()
                .unwrap();
            let ends = edge_endpoints(&mesh);
            let is_diagonal = |[a, b]: [u32; 2]| a % 9 != b % 9 && a / 9 != b / 9;
            let diagonals: Vec<u32> = (0..ends.len() as u32)
                .filter(|&e| is_diagonal(ends[e as usize]))
                .collect();
            let valid: Vec<u32> = diagonals
                .iter()
                .copied()
                .filter(|&e| {
                    matches!(classify(&mesh, &triangles, ends[e as usize]), Ok(v) if !v.connected)
                })
                .collect();
            assert!(!valid.is_empty());

            // Retry requests blocked by competing flips or unavailable storage.
            // Remove successful requests so their edges are not flipped back.
            let mut editable =
                GpuEditableMesh::upload(&gpu.device, &mesh, EditCapacity::default()).unwrap();
            let mut pending = diagonals.clone();
            let mut flipped = Vec::new();
            let mut rounds = 0;
            while !pending.is_empty() {
                rounds += 1;
                assert!(rounds <= 64, "patch size {patch_size}: no convergence");
                let statuses = gpu.flip_all(&mut editable, &pending);
                let mut retry = Vec::new();
                for (&edge, status) in pending.iter().zip(&statuses) {
                    match status {
                        FlipStatus::Flipped => flipped.push(edge),
                        FlipStatus::LostClaim => retry.push(edge),
                        FlipStatus::NotInterior => assert!(!valid.contains(&edge)),
                        other => panic!("patch size {patch_size}: edge {edge}: {other:?}"),
                    }
                }
                assert!(
                    retry.len() < pending.len(),
                    "patch size {patch_size}: round {rounds} made no progress"
                );
                pending = retry;
            }
            flipped.sort();
            assert_eq!(flipped, valid, "patch size {patch_size}");
            if patch_size == 128 {
                assert!(rounds > 1, "the whole grid flipped in one round");
            }

            let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
            check_invariants(&arena, &masks, &gpu.read(editable.maps()));
            let flipped_ends: Vec<_> = flipped.iter().map(|&e| ends[e as usize]).collect();
            let fixture = fixtures::Fixture {
                name: "grid 9",
                positions: positions.clone(),
                triangles: triangles.clone(),
            };
            assert_eq!(
                patch_views(&arena, &masks),
                rebuilt_views(
                    &mesh,
                    &fixture,
                    &alum_flips(&positions, &triangles, &flipped_ends)
                ),
                "patch size {patch_size}"
            );
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn batches_without_spare_slots_flip_as_predicted() {
        // With no spare slots, every flip rewrites its own edge and faces.
        let gpu = Gpu::new();
        let (fixture, mesh, _) = single_patch_grid();
        let mut editable =
            GpuEditableMesh::upload(&gpu.device, &mesh, EditCapacity::new(1.0).unwrap()).unwrap();
        let before = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        let edges = batch_request(&mesh);
        let prediction = predict_batch(&mesh, &fixture.triangles, &edges, 0);
        assert!(prediction.statuses.contains(&FlipStatus::Flipped));
        assert_eq!(gpu.flip_all(&mut editable, &edges), prediction.statuses);

        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_invariants(&arena, &masks, &gpu.read(editable.maps()));
        assert_slots_kept(&before.0, &before.1, &arena, &masks, "capacity 1.0");
        assert_eq!(
            patch_views(&arena, &masks),
            rebuilt_views(
                &mesh,
                &fixture,
                &alum_flips(&fixture.positions, &fixture.triangles, &prediction.flipped)
            )
        );
        assert_faces_by_id(
            &arena,
            &documented_flips(&fixture.triangles, &prediction.flipped),
            "capacity 1.0",
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn empty_batches_record_nothing_but_count_a_version() {
        let gpu = Gpu::new();
        let (_, mesh, _) = single_patch_grid();
        let mut editable =
            GpuEditableMesh::upload(&gpu.device, &mesh, EditCapacity::default()).unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let statuses = gpu
            .query
            .encode(&gpu.device, &mut encoder, &mut editable, &[])
            .unwrap();
        // One status per edge, all reporting that nothing was requested.
        assert_eq!(statuses.len(), mesh.edge_count() as u32);
        let words = test_support::read_words(&gpu.device, &gpu.queue, encoder, statuses.buffer());
        assert!(words[..statuses.len() as usize].iter().all(|&w| w == 0));
        assert_eq!(editable.version(), 1);
    }
}
