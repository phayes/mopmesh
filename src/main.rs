use mopmesh::MeshBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let positions = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
    let triangles = [[0, 1, 2], [0, 2, 3]];
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .patch_size(1)
        .build()?;
    println!(
        "{} vertices, {} edges, {} faces, {} patches",
        mesh.positions().len(),
        mesh.edge_count(),
        mesh.face_count(),
        mesh.patches().len()
    );
    Ok(())
}
