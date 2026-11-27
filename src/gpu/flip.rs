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
    /// Another flip in the same batch claimed one of the affected vertices.
    LostClaim,
    /// The patch has no free slot for the new edge or faces.
    OutOfCapacity,
    /// The requested edge ID is not less than the mesh's edge count.
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

/// GPU results for a batch of edge flips, in the same order as the requests.
///
/// After submitting and completing the commands from [`EdgeFlipQuery::encode`],
/// read the first [`len`](Self::len) `u32` words and decode them with
/// [`FlipStatus::from_word`]. The buffer supports `STORAGE | COPY_SRC`, so it
/// can be used by another shader or copied to a CPU-readable buffer.
pub struct GpuFlipStatuses {
    buffer: wgpu::Buffer,
    len: u32,
}
impl GpuFlipStatuses {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// Number of results to read, including rejected requests.
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Bindings and dispatch dimensions for one batch of application-defined
/// edge flips using `topology::flip_pass`.
///
/// Create a pass with [`EdgeFlipQuery::begin_pass`]. Bind it at group 1 for
/// three ordered dispatches: claim, reserve, then commit. Use a fresh pass
/// for each batch; its request state must not be reused.
pub struct FlipPass {
    bind_group: wgpu::BindGroup,
    count: u32,
    grid: [u32; 2],
}
impl FlipPass {
    /// The bindings to set at group 1 in all three dispatches for this batch.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    /// Number of requests, also available in WESL as `flip_pass_count()`.
    pub fn count(&self) -> u32 {
        self.count
    }
    /// Workgroup counts `[x, y]` for shaders using `@workgroup_size(64)` and
    /// `parallel::grid::flat_index`. Dispatch with a z count of 1, and skip
    /// invocations whose index is at least [`count`](Self::count).
    /// Returns `[0, 0]` for an empty batch.
    pub fn grid(&self) -> [u32; 2] {
        self.grid
    }
}

/// Records batches of edge flips on a [`GpuEditableMesh`].
/// Create one per device and reuse it for subsequent batches.
pub struct EdgeFlipQuery {
    requests: wgpu::BindGroupLayout,
    pass: wgpu::BindGroupLayout,
    claim: wgpu::ComputePipeline,
    reserve: wgpu::ComputePipeline,
    commit: wgpu::ComputePipeline,
}

impl EdgeFlipQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::BufferBindingType::{Storage, Uniform};
        let module = pipeline::shader_module(
            device,
            "MOPMesh edge flip",
            include_str!(concat!(env!("OUT_DIR"), "/edge_flip.wgsl")),
        );

        // Keep request IDs and results in group 0. Group 1 matches the
        // `topology::flip_pass` bindings used by application shaders too.
        let write = Storage { read_only: false };
        let (requests, _) = pipeline::buffer_layout(
            device,
            "MOPMesh edge flip requests",
            &[Storage { read_only: true }, write],
        );
        let (pass, _) = pipeline::buffer_layout(
            device,
            "MOPMesh edge flip pass",
            &[write, write, write, write, Uniform],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh edge flip"),
            bind_group_layouts: &[Some(&requests), Some(&pass)],
            immediate_size: 0,
        });
        let stage = |entry| {
            pipeline::compute_pipeline(
                device,
                "MOPMesh edge flip",
                &module,
                entry,
                Some(&pipeline_layout),
            )
        };
        Self {
            claim: stage("claim"),
            reserve: stage("reserve"),
            commit: stage("commit"),
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

    /// Prepares bindings for `count` requests on `mesh` using application
    /// shaders that import `topology::flip_pass`. This does not record GPU work.
    /// Record claim, reserve, and commit dispatches with the returned pass,
    /// in that order, before starting another batch on the same mesh.
    ///
    /// Every successful call increases [`GpuEditableMesh::version`], including
    /// an empty batch or a pass that is never dispatched. Later batches may
    /// choose different winners among requests that share vertices.
    ///
    /// Returns [`GpuError::LimitsExceeded`] if the requests or mesh exceed
    /// the device's buffer or dispatch limits; on error the version is unchanged.
    pub fn begin_pass(
        &self,
        device: &wgpu::Device,
        mesh: &mut GpuEditableMesh,
        count: u32,
    ) -> Result<FlipPass, GpuError> {
        // Check buffer sizes and dispatch dimensions before allocating or counting an edit.
        const RESOURCE: &str = "edge flips";
        let limits = device.limits();
        let slot_bytes = u64::from(count) * u64::from(SLOT_WORDS) * 4;
        pipeline::check_binding(&limits, slot_bytes, RESOURCE)?;
        pipeline::check_binding(&limits, u64::from(mesh.vertex_count()) * 4, RESOURCE)?;
        let grid = pipeline::workgroup_grid(&limits, count, RESOURCE)?;

        // Fresh buffers provide zeroed claims and request state. Use the current
        // version as the priority seed, then count this batch as an edit.
        // The two trailing parameter words pad the shader uniform layout.
        let seed = mesh.version() as u32;
        mesh.bump_version();
        let words = |n: u64| (n * 4).max(4);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh edge flip parameters"),
            contents: bytemuck::cast_slice(&[count, seed, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let claims = pipeline::output_buffer(
            device,
            "MOPMesh edge flip claims",
            words(u64::from(mesh.vertex_count())),
        );
        let slots = pipeline::output_buffer(device, "MOPMesh edge flip slots", slot_bytes.max(4));
        let bind_group = pipeline::bind_buffers(
            device,
            "MOPMesh edge flip pass",
            &self.pass,
            &[mesh.arena(), mesh.masks(), &claims, &slots, &params],
        );
        Ok(FlipPass {
            bind_group,
            count,
            grid,
        })
    }

    /// Records a batch of flips for the mesh-wide edge IDs in `edges`.
    /// Returns a GPU status buffer with one result per request, in input order.
    /// Submit the encoder's commands to run the flips and produce the results.
    ///
    /// Flips in one batch must not share any of their four corner vertices.
    /// Each valid request gets a priority and proceeds only if it wins all
    /// four corners; otherwise it reports [`FlipStatus::LostClaim`]. Priorities
    /// depend on request order and the mesh version, so later calls can select
    /// different winners.
    ///
    /// Each winner needs one spare edge slot and two spare face slots in its
    /// patch. If it cannot reserve them, it reports [`FlipStatus::OutOfCapacity`].
    /// When space is limited, GPU execution order can affect which winners
    /// get slots. Requests with either status make no changes themselves and
    /// may be retried later; a patch without spare slots still needs more capacity.
    ///
    /// List each edge at most once. At most one duplicate request can flip it,
    /// and retrying another copy after a success can flip the edge back.
    ///
    /// Batches recorded into the encoder run in order. Every successful call
    /// increases [`GpuEditableMesh::version`] when recorded, including an empty
    /// batch or one whose flips are all rejected. An empty batch records no
    /// GPU work. Returns [`GpuError::LimitsExceeded`] if the requests or mesh
    /// exceed the device's buffer or dispatch limits, without recording work
    /// or increasing the version.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &mut GpuEditableMesh,
        edges: &[u32],
    ) -> Result<GpuFlipStatuses, GpuError> {
        // Preparing the pass counts an edit even when there are no requests.
        // An empty result still needs a nonzero buffer size, but has no readable statuses.
        let count = u32::try_from(edges.len()).map_err(|_| GpuError::LimitsExceeded {
            resource: "edge flips",
        })?;
        let pass = self.begin_pass(device, mesh, count)?;
        let status = pipeline::output_buffer(
            device,
            "MOPMesh edge flip statuses",
            (u64::from(count) * 4).max(4),
        );
        if count == 0 {
            return Ok(GpuFlipStatuses {
                buffer: status,
                len: 0,
            });
        }
        let requests = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh edge flip requests"),
            contents: bytemuck::cast_slice(edges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh edge flip requests",
            &self.requests,
            &[&requests, &status],
        );

        // Separate dispatches let each stage read the completed results
        // of the previous stage.
        for (label, stage) in [
            ("MOPMesh edge flip claim", &self.claim),
            ("MOPMesh edge flip reserve", &self.reserve),
            ("MOPMesh edge flip commit", &self.commit),
        ] {
            pipeline::dispatch_grid_groups(
                encoder,
                label,
                stage,
                &[&bindings, pass.bind_group()],
                pass.grid(),
            );
        }
        Ok(GpuFlipStatuses {
            buffer: status,
            len: count,
        })
    }
}

/// Number of `u32` words allocated for each request's temporary state.
/// Keep this equal to `SLOT_WORDS` in `shaders/topology/flip_pass.wesl`.
const SLOT_WORDS: u32 = 7;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::editable::inspect::{
        PatchView, check_owner_tables, edge_endpoints as edge_endpoints_after, face_vertices,
        patch_views, records,
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

    /// A flip that passes the CPU reference checks, before checking storage
    /// space or conflicts with other requests.
    struct Valid {
        /// Patch that stores the edge and contains all affected faces.
        patch: u32,
        /// Mesh-wide vertex IDs: the original edge endpoints, then the new endpoints.
        corners: [u32; 4],
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
        if tris.iter().any(|t| t.contains(&c) && t.contains(&d)) {
            return Err(FlipStatus::Connected);
        }
        Ok(Valid {
            patch: p,
            corners: [u, v, c, d],
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
    /// assignment, using Alum for the flipped triangles.
    fn expect(
        mesh: &Mesh,
        fixture: &fixtures::Fixture,
        capacity: EditCapacity,
        [u, v]: [u32; 2],
    ) -> Expected {
        let reject = |status| Expected {
            status,
            triangles: None,
            new_edge: [0; 2],
        };
        let Valid {
            patch: p,
            corners: [_, _, c, d],
        } = match classify(mesh, &fixture.triangles, [u, v]) {
            Ok(valid) => valid,
            Err(status) => return reject(status),
        };
        let record = records(
            &editable_contents(mesh, capacity, &wgpu::Limits::default())
                .unwrap()
                .arena,
        )[p as usize];
        let record_patch = &mesh.patches()[p as usize];
        let free_edges = record.capacity[1] as usize - record_patch.edge_ids().len();
        let free_faces = record.capacity[2] as usize - record_patch.face_ids().len();
        if free_edges < 1 || free_faces < 2 {
            return reject(FlipStatus::OutOfCapacity);
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
        fn flip_all(&self, mesh: &mut GpuEditableMesh, edges: &[u32]) -> Vec<FlipStatus> {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let statuses = self
                .query
                .encode(&self.device, &mut encoder, mesh, edges)
                .unwrap();
            let words =
                test_support::read_words(&self.device, &self.queue, encoder, statuses.buffer());
            words[..edges.len()]
                .iter()
                .map(|&w| FlipStatus::from_word(w).unwrap_or_else(|| panic!("unknown status {w}")))
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
                        let expected = expect(&mesh, &fixture, capacity, uv);
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
                                check_owner_tables(&arena, &masks);
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
        for status in [
            "Flipped",
            "Boundary",
            "NotInterior",
            "Connected",
            "OutOfCapacity",
        ] {
            assert!(seen.contains(status), "fixtures never produced {status}");
        }
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
        check_owner_tables(&arena, &masks);
        assert_eq!(patch_views(&arena, &masks), original);
        assert_faces_by_id(
            &arena,
            &documented_flips(&fixture.triangles, &[ab, cd]),
            "second flip",
        );

        // Rejected flips count as edits too.
        assert_eq!(gpu.flip(&mut editable, u32::MAX), FlipStatus::NotAnEdge);
        assert_eq!(editable.version(), 3);
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn freed_slots_are_reused_by_later_flips() {
        // 128 faces and 208 edges at factor 1.05 leave 7 spare face slots
        // and 11 spare edge slots, enough for three flips without reuse.
        let gpu = Gpu::new();
        let capacity = EditCapacity::new(1.05).unwrap();
        let (fixture, mesh, edge) = single_patch_grid();
        let mut editable = GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
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
        check_owner_tables(&arena, &masks);
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

    /// Reproduces the shader's request priorities on the CPU to predict winners.
    fn priority(index: u32, seed: u32) -> u32 {
        let mut x = index ^ seed;
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x
    }

    /// Predicts batch results for the original mesh, assuming enough spare storage.
    /// Returns a status per request and the original endpoints of edges expected
    /// to flip. A request wins only if it has the highest priority at every corner.
    fn predict_batch(
        mesh: &Mesh,
        triangles: &[[u32; 3]],
        edges: &[u32],
        seed: u32,
    ) -> (Vec<FlipStatus>, Vec<[u32; 2]>) {
        let ends = edge_endpoints(mesh);
        let checked: Vec<_> = edges
            .iter()
            .map(|&e| match ends.get(e as usize) {
                None => Err(FlipStatus::NotAnEdge),
                Some(&uv) => classify(mesh, triangles, uv),
            })
            .collect();

        // Record the highest valid request priority at each affected vertex.
        let mut claims = std::collections::HashMap::<u32, u32>::new();
        for (i, valid) in checked.iter().enumerate() {
            for &corner in valid.iter().flat_map(|v| &v.corners) {
                let bid = claims.entry(corner).or_default();
                *bid = (*bid).max(priority(i as u32, seed));
            }
        }

        // Only requests with the highest priority at all four corners can flip.
        let mut flipped = Vec::new();
        let statuses = checked
            .iter()
            .enumerate()
            .map(|(i, valid)| match valid {
                Err(status) => *status,
                Ok(v)
                    if v.corners
                        .iter()
                        .all(|c| claims[c] == priority(i as u32, seed)) =>
                {
                    flipped.push([v.corners[0], v.corners[1]]);
                    FlipStatus::Flipped
                }
                Ok(_) => FlipStatus::LostClaim,
            })
            .collect();
        (statuses, flipped)
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

    /// Requests every edge in a reordered list, then adds a duplicate when
    /// possible and an invalid ID to check how those requests are handled.
    fn batch_request(mesh: &Mesh) -> Vec<u32> {
        let count = mesh.edge_count() as u32;
        let mut edges: Vec<u32> = (0..count).map(|i| (i * 7 + 3) % count).collect();
        if count.is_multiple_of(7) {
            edges = (0..count).rev().collect();
        }
        edges.extend(edges.get(count as usize / 2).copied());
        edges.push(count);
        edges
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn batches_flip_exactly_the_predicted_winners_like_alum() {
        let gpu = Gpu::new();
        let capacity = EditCapacity::new(3.0).unwrap();
        let mut seen = std::collections::BTreeSet::new();
        let mut most_in_one_patch = 0;
        for (context, fixture, mesh) in batch_meshes() {
            let edges = batch_request(&mesh);
            let mut editable = GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
            // Exercise priority selection with a mesh version greater than zero.
            editable.bump_version();
            let seed = editable.version() as u32;
            let (expected, flipped) = predict_batch(&mesh, &fixture.triangles, &edges, seed);
            let statuses = gpu.flip_all(&mut editable, &edges);
            assert_eq!(statuses, expected, "{context}");
            seen.extend(statuses.iter().map(|s| format!("{s:?}")));

            // Confirm the prediction's assumption: every patch has one spare edge
            // slot and two spare face slots for each predicted winner.
            let ends = edge_endpoints(&mesh);
            let records = records(
                &editable_contents(&mesh, capacity, &wgpu::Limits::default())
                    .unwrap()
                    .arena,
            );
            let mut per_patch = std::collections::HashMap::<u32, usize>::new();
            for uv in &flipped {
                let valid = classify(&mesh, &fixture.triangles, *uv).unwrap();
                *per_patch.entry(valid.patch).or_default() += 1;
            }
            for (&p, &n) in &per_patch {
                let patch = &mesh.patches()[p as usize];
                let free_edges = records[p as usize].capacity[1] as usize - patch.edge_ids().len();
                let free_faces = records[p as usize].capacity[2] as usize - patch.face_ids().len();
                assert!(
                    n <= free_edges && 2 * n <= free_faces,
                    "{context}: patch {p} too full to predict"
                );
                most_in_one_patch = most_in_one_patch.max(n);
            }

            // Compare the GPU result with the same flips performed by Alum.
            let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
            check_owner_tables(&arena, &masks);
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
        for status in [
            "Flipped",
            "LostClaim",
            "Boundary",
            "NotInterior",
            "Connected",
            "NotAnEdge",
        ] {
            assert!(seen.contains(status), "batches never produced {status}");
        }
        assert!(
            most_in_one_patch > 1,
            "no batch flipped two edges in one patch"
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
                .filter(|&e| classify(&mesh, &triangles, ends[e as usize]).is_ok())
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
                        FlipStatus::LostClaim | FlipStatus::OutOfCapacity => retry.push(edge),
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
            check_owner_tables(&arena, &masks);
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
    fn batches_without_spare_slots_reject_every_winner_unchanged() {
        let gpu = Gpu::new();
        let (_, mesh, _) = single_patch_grid();
        let (_, triangles) = fixtures::grid(9);
        let mut editable =
            GpuEditableMesh::upload(&gpu.device, &mesh, EditCapacity::new(1.0).unwrap()).unwrap();
        let before = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        let edges = batch_request(&mesh);
        let (predicted, _) = predict_batch(&mesh, &triangles, &edges, 0);
        let expected: Vec<_> = predicted
            .into_iter()
            .map(|s| match s {
                FlipStatus::Flipped => FlipStatus::OutOfCapacity,
                other => other,
            })
            .collect();
        assert!(expected.contains(&FlipStatus::OutOfCapacity));
        assert_eq!(gpu.flip_all(&mut editable, &edges), expected);
        assert!(
            (gpu.read(editable.arena()), gpu.read(editable.masks())) == before,
            "rejected flips changed the mesh"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn batches_near_capacity_stay_consistent() {
        // Five percent extra storage allows at most three flips in this batch.
        // More requests win their corners, so the test accepts whichever
        // subset gets storage and checks that the resulting mesh is consistent.
        let gpu = Gpu::new();
        let (fixture, mesh, _) = single_patch_grid();
        let mut editable =
            GpuEditableMesh::upload(&gpu.device, &mesh, EditCapacity::new(1.05).unwrap()).unwrap();
        let edges = batch_request(&mesh);
        let (predicted, _) = predict_batch(&mesh, &fixture.triangles, &edges, 0);
        let statuses = gpu.flip_all(&mut editable, &edges);
        let ends = edge_endpoints(&mesh);
        let mut flipped = Vec::new();
        for (i, (&status, &expected)) in statuses.iter().zip(&predicted).enumerate() {
            if expected == FlipStatus::Flipped {
                assert!(
                    matches!(status, FlipStatus::Flipped | FlipStatus::OutOfCapacity),
                    "candidate {i}: {status:?}"
                );
            } else {
                assert_eq!(status, expected, "candidate {i}");
            }
            if status == FlipStatus::Flipped {
                flipped.push(ends[edges[i] as usize]);
            }
        }
        assert!(flipped.len() <= 3, "more flips than spare slots");
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_owner_tables(&arena, &masks);
        assert_eq!(
            patch_views(&arena, &masks),
            rebuilt_views(
                &mesh,
                &fixture,
                &alum_flips(&fixture.positions, &fixture.triangles, &flipped)
            )
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
        assert!(statuses.is_empty());
        assert_eq!(editable.version(), 1);
    }
}
