use alum::{Handle, HasIterators, HasTopology, PolyMeshF32, Vec3};
use mopmesh::{BuildError, EdgeHandle, FaceHandle, MeshBuilder, VertexHandle};
use std::collections::BTreeSet;

fn check_against_alum(positions: &[[f32; 3]], triangles: &[[u32; 3]]) {
    let mut reference = PolyMeshF32::new();
    for &[x, y, z] in positions {
        reference.add_vertex(Vec3(x, y, z)).unwrap();
    }
    for tri in triangles {
        reference.add_face(&tri.map(Into::into)).unwrap();
    }
    reference.check_topology().unwrap();
    let reference_edges: BTreeSet<_> = reference
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(&reference);
            [a.index().min(b.index()), a.index().max(b.index())]
        })
        .collect();

    for size in [1, 2, 3, 8, 128] {
        let mesh = MeshBuilder::new()
            .positions(positions)
            .triangles(triangles)
            .patch_size(size)
            .build()
            .unwrap();
        assert_eq!(mesh.positions().len(), reference.num_vertices());
        assert_eq!(mesh.edge_count(), reference.num_edges());
        assert_eq!(mesh.face_count(), reference.num_faces());
        // Edge IDs follow ascending (smaller, larger) endpoint order.
        let mut endpoints_by_id = vec![[0; 2]; mesh.edge_count()];
        for p in mesh.patches() {
            for (local, &global) in p.edge_ids().iter().enumerate() {
                endpoints_by_id[global as usize] =
                    p.edge_vertices()[local].map(|v| p.vertex_ids()[v as usize]);
            }
        }
        assert!(endpoints_by_id.windows(2).all(|w| w[0] < w[1]));
        let mut owned_vertices = vec![0; positions.len()];
        let mut owned_edges = vec![0; mesh.edge_count()];
        let mut owned_faces = vec![0; triangles.len()];
        let mut actual_edges = BTreeSet::new();
        for p in mesh.patches() {
            assert!((1..=size).contains(&p.owned_face_count()));
            for (local, &global) in p.vertex_ids().iter().enumerate() {
                let h = VertexHandle {
                    patch: p.id(),
                    local: local as u32,
                };
                let owner = mesh.vertex_owner(h).unwrap();
                assert_eq!(mesh.vertex_owner(owner), Some(owner));
                assert_eq!(
                    mesh.patches()[owner.patch as usize].vertex_ids()[owner.local as usize],
                    global
                );
                if owner == h {
                    owned_vertices[global as usize] += 1;
                    let actual: BTreeSet<_> = p
                        .vertex_faces(local as u32)
                        .map(|f| p.face_ids()[f as usize])
                        .collect();
                    let expected: BTreeSet<_> = reference
                        .vf_ccw_iter(global.into())
                        .map(|f| f.index())
                        .collect();
                    assert_eq!(actual, expected, "owned vertex {global}, patch size {size}");
                }
            }
            for (local, &global) in p.edge_ids().iter().enumerate() {
                let h = EdgeHandle {
                    patch: p.id(),
                    local: local as u32,
                };
                let owner = mesh.edge_owner(h).unwrap();
                assert_eq!(mesh.edge_owner(owner), Some(owner));
                assert_eq!(
                    mesh.patches()[owner.patch as usize].edge_ids()[owner.local as usize],
                    global
                );
                let endpoints = p.edge_vertices()[local].map(|v| p.vertex_ids()[v as usize]);
                assert!(reference_edges.contains(&endpoints));
                if owner == h {
                    owned_edges[global as usize] += 1;
                    actual_edges.insert(endpoints);
                    let expected: BTreeSet<_> = reference
                        .edges()
                        .find_map(|e| {
                            let (a, b) = e.vertices(&reference);
                            ([a.index().min(b.index()), a.index().max(b.index())] == endpoints)
                                .then(|| {
                                    let (a, b) = e.halfedges();
                                    [a.face(&reference), b.face(&reference)]
                                        .into_iter()
                                        .flatten()
                                        .map(|f| f.index())
                                        .collect()
                                })
                        })
                        .unwrap();
                    let actual: BTreeSet<_> = p
                        .face_edges()
                        .iter()
                        .enumerate()
                        .filter(|(_, fe)| fe.iter().any(|d| d.edge == local as u32))
                        .map(|(f, _)| p.face_ids()[f])
                        .collect();
                    assert_eq!(actual, expected);
                }
            }
            for (local, &global) in p.face_ids().iter().enumerate() {
                let h = FaceHandle {
                    patch: p.id(),
                    local: local as u32,
                };
                let owner = mesh.face_owner(h).unwrap();
                assert_eq!(mesh.face_owner(owner), Some(owner));
                assert_eq!(
                    mesh.patches()[owner.patch as usize].face_ids()[owner.local as usize],
                    global
                );
                if owner == h {
                    owned_faces[global as usize] += 1;
                }
                let actual = p
                    .face_vertices(local as u32)
                    .unwrap()
                    .map(|v| p.vertex_ids()[v as usize]);
                let expected: Vec<_> = reference
                    .fv_ccw_iter(global.into())
                    .map(|v| v.index())
                    .collect();
                assert!((0..3).any(|shift| (0..3).all(|i| actual[i] == expected[(i + shift) % 3])));
                // Directed edges must also end at the next edge's start.
                for i in 0..3 {
                    let d = p.face_edges()[local][i];
                    let end = p.edge_vertices()[d.edge as usize][usize::from(!d.reversed)];
                    assert_eq!(p.vertex_ids()[end as usize], actual[(i + 1) % 3]);
                }
            }
        }
        assert_eq!(actual_edges, reference_edges);
        assert!(
            owned_vertices
                .iter()
                .chain(&owned_edges)
                .chain(&owned_faces)
                .all(|&n| n == 1)
        );
    }
}

#[test]
fn triangle_and_quad_match_alum() {
    let p = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
    check_against_alum(&p[..3], &[[0, 1, 2]]);
    check_against_alum(&p, &[[0, 1, 2], [0, 2, 3]]);
}

#[test]
fn closed_tetrahedron_matches_alum() {
    check_against_alum(
        &[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        &[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    );
}

#[test]
fn grid_with_hole_matches_alum() {
    let mut positions = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            positions.push([x as f32, y as f32, 0.]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..4 {
        for x in 0..4 {
            if x == 1 && y == 1 {
                continue;
            }
            let v = y * 5 + x;
            triangles.extend([[v, v + 1, v + 6], [v, v + 6, v + 5]]);
        }
    }
    check_against_alum(&positions, &triangles);
}

#[test]
fn high_valence_vertex_matches_alum() {
    let n = 64u32;
    let mut positions = vec![[0., 0., 0.]];
    for i in 0..n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        positions.push([angle.cos(), angle.sin(), 0.]);
    }
    let triangles: Vec<_> = (0..n).map(|i| [0, i + 1, (i + 1) % n + 1]).collect();
    check_against_alum(&positions, &triangles);
}

#[test]
fn disconnected_components_match_alum() {
    check_against_alum(
        &[
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [3., 0., 0.],
            [4., 0., 0.],
            [3., 1., 0.],
        ],
        &[[0, 1, 2], [3, 4, 5]],
    );
}

#[test]
fn ribbon_attributes_share_canonical_values() {
    let positions = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
    let triangles = [[0, 1, 2], [0, 2, 3]];
    let build = || {
        MeshBuilder::new()
            .positions(&positions)
            .triangles(&triangles)
            .patch_size(1)
            .build()
            .unwrap()
    };
    let mesh = build();
    let mut attr = mesh.vertex_attribute(0u32);
    let ribbon = VertexHandle { patch: 1, local: 0 };
    let owner = mesh.vertex_owner(ribbon).unwrap();
    assert_ne!(ribbon, owner);
    *attr.get_mut(&mesh, ribbon).unwrap() = 42;
    assert_eq!(attr.get(&mesh, owner), Some(&42));
    let other = build();
    assert_ne!(other.id(), mesh.id());
    assert_eq!(attr.get(&other, owner), None);
    assert_eq!(attr.mesh_id(), mesh.id());

    // Per-vertex construction uses input vertex order and requires one value each.
    let ramp = mesh.vertex_attribute_from(vec![10u32, 11, 12, 13]).unwrap();
    assert_eq!(ramp.as_slice(), [10, 11, 12, 13]);
    assert_eq!(ramp.len(), 4);
    assert!(!ramp.is_empty());
    for p in mesh.patches() {
        for (local, &global) in p.vertex_ids().iter().enumerate() {
            let h = VertexHandle {
                patch: p.id(),
                local: local as u32,
            };
            assert_eq!(ramp.get(&mesh, h), Some(&(10 + global)));
        }
    }
    assert!(mesh.vertex_attribute_from(vec![0u32; 3]).is_none());
    assert!(mesh.vertex_attribute_from(vec![0u32; 5]).is_none());
    assert_eq!(
        mesh.vertex_owner(VertexHandle {
            patch: u32::MAX,
            local: 0
        }),
        None
    );
    assert_eq!(
        attr.get(
            &mesh,
            VertexHandle {
                patch: 0,
                local: u32::MAX
            }
        ),
        None
    );
}

#[test]
fn invalid_inputs_are_rejected() {
    let p = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., -1., 0.],
    ];
    let build = |p: &[[f32; 3]], t: &[[u32; 3]]| {
        MeshBuilder::new()
            .positions(p)
            .triangles(t)
            .build()
            .unwrap_err()
    };
    assert_eq!(
        build(&p, &[[0, 1, 9]]),
        BuildError::InvalidVertex { face: 0, vertex: 9 }
    );
    assert_eq!(build(&p, &[[0, 0, 1]]), BuildError::DegenerateFace(0));
    // Distinct vertices at equal coordinates are degenerate; -0.0 equals 0.0.
    assert_eq!(
        build(&[[0., 0., 0.], [1., 0., 0.], [-0., 0., 0.]], &[[0, 1, 2]]),
        BuildError::DegenerateFace(0)
    );
    // Three distinct collinear corners are accepted.
    assert!(
        MeshBuilder::new()
            .positions(&[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
            .triangles(&[[0, 1, 2]])
            .build()
            .is_ok()
    );
    assert_eq!(
        build(&p, &[[0, 1, 2], [2, 1, 0]]),
        BuildError::DuplicateFace(1)
    );
    // A same-direction repeat also breaks winding; the duplicate is reported.
    assert_eq!(
        build(&p, &[[0, 1, 2], [0, 1, 2]]),
        BuildError::DuplicateFace(1)
    );
    assert_eq!(
        build(&p, &[[0, 1, 2], [0, 1, 4]]),
        BuildError::InconsistentWinding([0, 1])
    );
    assert_eq!(
        build(&p, &[[0, 1, 2], [0, 3, 4]]),
        BuildError::NonManifoldVertex(0)
    );
    assert_eq!(build(&p, &[[0, 1, 2]]), BuildError::IsolatedVertex(3));
    assert_eq!(
        build(&[[f32::NAN, 0., 0.]], &[]),
        BuildError::NonFinitePosition(0)
    );
    assert_eq!(
        MeshBuilder::new().patch_size(0).build().unwrap_err(),
        BuildError::InvalidPatchSize
    );
    let p = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
    ];
    assert_eq!(
        build(&p, &[[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
        BuildError::NonManifoldEdge([0, 1])
    );
    // Two closed surfaces meeting at a single vertex.
    let p = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
    ];
    let tetrahedron = |[a, b, c, d]: [u32; 4]| [[a, c, b], [a, b, d], [b, c, d], [c, a, d]];
    let faces: Vec<_> = tetrahedron([0, 1, 2, 3])
        .into_iter()
        .chain(tetrahedron([0, 4, 5, 6]))
        .collect();
    assert_eq!(build(&p, &faces), BuildError::NonManifoldVertex(0));
    // Messages name the offending input element rather than a Debug dump.
    assert_eq!(
        BuildError::InvalidVertex { face: 0, vertex: 9 }.to_string(),
        "face 0 references missing vertex 9"
    );
    assert_eq!(
        BuildError::NonManifoldEdge([0, 1]).to_string(),
        "edge (0, 1) is shared by more than two faces"
    );
}

#[test]
fn empty_mesh_matches_alum() {
    check_against_alum(&[], &[]);
}
