//! Edge flips on editable meshes.
use super::{GpuEditableMesh, pipeline};
use wgpu::util::DeviceExt;

/// The outcome of one requested edge flip, as written by the GPU.
///
/// Flipping an edge shared by two triangles replaces it with the other
/// diagonal of the four-sided region they form. Every outcome except
/// [`Flipped`](Self::Flipped) leaves the mesh unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipStatus {
    /// The edge was flipped. It keeps its edge ID, and the two new faces
    /// take the two old faces' IDs.
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

/// One `u32` status word written by an edge flip; decode it with
/// [`FlipStatus::from_word`]. The buffer supports `STORAGE | COPY_SRC`.
pub struct GpuFlipStatus {
    buffer: wgpu::Buffer,
}
impl GpuFlipStatus {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Flips single edges of a [`GpuEditableMesh`]. Cache one per device.
pub struct EdgeFlipQuery {
    pipeline: wgpu::ComputePipeline,
}

impl EdgeFlipQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh edge flip",
            include_str!(concat!(env!("OUT_DIR"), "/edge_flip.wgsl")),
        );
        Self {
            pipeline: pipeline::compute_pipeline(
                device,
                "MOPMesh edge flip",
                &module,
                "main",
                None,
            ),
        }
    }

    /// Records a flip of `edge`, a global edge ID of `mesh`, and returns the
    /// status the GPU will write. Flips recorded one after another apply in
    /// order. Each call increases [`GpuEditableMesh::version`], even if the
    /// flip is then rejected.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &mut GpuEditableMesh,
        edge: u32,
    ) -> GpuFlipStatus {
        // Padded to 16 bytes, the minimum uniform struct size.
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh edge flip request"),
            contents: bytemuck::cast_slice(&[edge, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let status = pipeline::output_buffer(device, "MOPMesh edge flip status", 4);
        let bindings = pipeline::bind_buffers(
            device,
            "MOPMesh edge flip bindings",
            &self.pipeline.get_bind_group_layout(0),
            &[&params, mesh.arena(), mesh.masks(), &status],
        );
        pipeline::dispatch(encoder, "MOPMesh edge flip", &self.pipeline, &bindings, 1);
        mesh.bump_version();
        GpuFlipStatus { buffer: status }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::editable::inspect::{
        PatchView, check_owner_tables, edge_endpoints as edge_endpoints_after, patch_views, records,
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
        let tris = &fixture.triangles;
        let on_edge: Vec<usize> = (0..tris.len())
            .filter(|&f| tris[f].contains(&u) && tris[f].contains(&v))
            .collect();
        if on_edge.len() < 2 {
            return reject(FlipStatus::Boundary);
        }
        let opposite = |f: usize| *tris[f].iter().find(|&&w| w != u && w != v).unwrap();
        let (c, d) = (opposite(on_edge[0]), opposite(on_edge[1]));

        // Every face around the four corners must belong to the same patch.
        let patch = face_patches(mesh);
        let p = patch[on_edge[0]];
        let interior = [u, v, c, d].iter().all(|&w| {
            (0..tris.len())
                .filter(|&f| tris[f].contains(&w))
                .all(|f| patch[f] == p)
        });
        if !interior {
            return reject(FlipStatus::NotInterior);
        }
        if tris.iter().any(|t| t.contains(&c) && t.contains(&d)) {
            return reject(FlipStatus::Connected);
        }
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

        // Flip with Alum. Its faces keep their indices, so face IDs line up.
        let mut reference = PolyMeshF32::new();
        for &[x, y, z] in &fixture.positions {
            reference.add_vertex(Vec3(x, y, z)).unwrap();
        }
        for t in tris {
            reference.add_face(&t.map(Into::into)).unwrap();
        }
        let edge = reference.find_halfedge(u.into(), v.into()).unwrap().edge();
        reference.swap_edge_ccw(edge).unwrap();
        let triangles = reference
            .faces()
            .map(|f| {
                let vs: Vec<u32> = reference.fv_ccw_iter(f).map(|v| v.index()).collect();
                [vs[0], vs[1], vs[2]]
            })
            .collect();
        Expected {
            status: FlipStatus::Flipped,
            triangles: Some(triangles),
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
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let status = self.query.encode(&self.device, &mut encoder, mesh, edge);
            let word =
                test_support::read_words(&self.device, &self.queue, encoder, status.buffer())[0];
            FlipStatus::from_word(word).unwrap_or_else(|| panic!("unknown status {word}"))
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
        let (_, mesh, edge) = single_patch_grid();
        let mut editable = GpuEditableMesh::upload(&gpu.device, &mesh, capacity).unwrap();
        let original = patch_views(&gpu.read(editable.arena()), &gpu.read(editable.masks()));
        assert_eq!(editable.version(), 0);
        assert_eq!(gpu.flip(&mut editable, edge), FlipStatus::Flipped);
        let flipped = patch_views(&gpu.read(editable.arena()), &gpu.read(editable.masks()));
        assert_ne!(flipped, original);
        assert_eq!(gpu.flip(&mut editable, edge), FlipStatus::Flipped);
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_owner_tables(&arena, &masks);
        assert_eq!(patch_views(&arena, &masks), original);

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
        for round in 0..20 {
            assert_eq!(
                gpu.flip(&mut editable, edge),
                FlipStatus::Flipped,
                "round {round}"
            );
        }
        let (arena, masks) = (gpu.read(editable.arena()), gpu.read(editable.masks()));
        check_owner_tables(&arena, &masks);
        assert_eq!(
            patch_views(&arena, &masks),
            rebuilt_views(&mesh, &fixture, &fixture.triangles),
            "an even number of flips restores the mesh"
        );
    }
}
