//! Host attributes for every element kind: canonical values shared by ribbon
//! copies, construction rules and mesh identity.
use mopmesh::{
    EdgeAttribute, EdgeHandle, ElementKind, FaceAttribute, FaceHandle, Mesh, MeshBuilder,
    VertexAttribute,
};

#[path = "support/fixtures.rs"]
mod fixtures;

fn build() -> Mesh {
    let (positions, triangles) = fixtures::grid(5);
    MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .patch_size(3)
        .build()
        .unwrap()
}

#[test]
fn element_counts_size_each_attribute_kind() {
    let mesh = build();
    assert_eq!(mesh.element_count(ElementKind::Vertex), 25);
    assert_eq!(mesh.element_count(ElementKind::Edge), mesh.edge_count());
    assert_eq!(mesh.element_count(ElementKind::Face), 32);
    assert_eq!(VertexAttribute::new(&mesh, 0u8).len(), 25);
    assert_eq!(EdgeAttribute::new(&mesh, 0u8).len(), mesh.edge_count());
    assert_eq!(FaceAttribute::new(&mesh, 0u8).len(), 32);
    assert!(FaceAttribute::from_values(&mesh, vec![0u8; 31]).is_none());
    assert!(EdgeAttribute::from_values(&mesh, vec![0u8; 32]).is_none());
}

#[test]
fn edge_and_face_ribbon_copies_share_one_value() {
    let mesh = build();
    let mut edges = EdgeAttribute::new(&mesh, 0u32);
    let mut faces = FaceAttribute::new(&mesh, 0u32);

    // Find an edge and a face held by more than one patch.
    let ribbon_edge = mesh
        .patches()
        .iter()
        .flat_map(|p| {
            (0..p.edge_ids().len() as u32).map(|local| EdgeHandle {
                patch: p.id(),
                local,
            })
        })
        .find(|&h| mesh.edge_owner(h) != Some(h))
        .expect("a ribbon edge");
    let ribbon_face = mesh
        .patches()
        .iter()
        .flat_map(|p| {
            (0..p.face_ids().len() as u32).map(|local| FaceHandle {
                patch: p.id(),
                local,
            })
        })
        .find(|&h| mesh.face_owner(h) != Some(h))
        .expect("a ribbon face");

    // Writes through a ribbon copy are read through the owner, and vice versa.
    *edges.get_mut(&mesh, ribbon_edge).unwrap() = 7;
    assert_eq!(
        edges.get(&mesh, mesh.edge_owner(ribbon_edge).unwrap()),
        Some(&7)
    );
    *faces
        .get_mut(&mesh, mesh.face_owner(ribbon_face).unwrap())
        .unwrap() = 9;
    assert_eq!(faces.get(&mesh, ribbon_face), Some(&9));
    assert_eq!(edges.as_slice().iter().filter(|&&v| v == 7).count(), 1);
    assert_eq!(faces.as_slice().iter().filter(|&&v| v == 9).count(), 1);
}

#[test]
fn handles_resolve_to_canonical_ids() {
    let mesh = build();
    let edges = EdgeAttribute::from_values(&mesh, (0..mesh.edge_count() as u32).collect()).unwrap();
    let faces = FaceAttribute::from_values(&mesh, (0..mesh.face_count() as u32).collect()).unwrap();
    for p in mesh.patches() {
        for (local, &id) in p.edge_ids().iter().enumerate() {
            let h = EdgeHandle {
                patch: p.id(),
                local: local as u32,
            };
            assert_eq!(edges.get(&mesh, h), Some(&id));
        }
        for (local, &id) in p.face_ids().iter().enumerate() {
            let h = FaceHandle {
                patch: p.id(),
                local: local as u32,
            };
            assert_eq!(faces.get(&mesh, h), Some(&id));
        }
    }

    // Out-of-range handles and other meshes find nothing.
    let local = mesh.patches()[0].face_ids().len() as u32;
    assert_eq!(faces.get(&mesh, FaceHandle { patch: 0, local }), None);
    assert_eq!(
        faces.get(
            &mesh,
            FaceHandle {
                patch: u32::MAX,
                local: 0
            }
        ),
        None
    );
    let other = build();
    assert_eq!(faces.get(&other, FaceHandle { patch: 0, local: 0 }), None);
    assert_eq!(faces.mesh_id(), mesh.id());
}
