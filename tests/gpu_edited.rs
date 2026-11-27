//! Checks GPU queries on edited meshes. After many parallel flips, every
//! query on the edited upload must match the same query on a fresh upload of
//! the flipped triangles, and results from before an edit must be rejected.
#[path = "../examples/support/delaunay.rs"]
mod delaunay;
#[path = "support/fresh_flips.rs"]
mod fresh_flips;
#[path = "support/replay.rs"]
mod replay;
use mopmesh::{
    Mesh, MeshBuilder,
    gpu::{
        AdjacencyQuery, BoundaryQuery, DotQuery, EdgeFlipQuery, EdgeQuery, EditCapacity,
        FaceAreaQuery, FaceCentroidQuery, FaceNormalQuery, FaceVertexQuery, FlipStatus,
        GpuEditableMesh, GpuError, GpuMesh, GpuTopology, GpuVertexAttribute, ReduceOp, ReduceQuery,
        SmoothQuery, Smoothing, VertexFaceQuery, VertexNormalQuery,
    },
};
use std::collections::HashMap;

/// Every query this test compares.
struct Queries {
    edges: EdgeQuery,
    faces: FaceVertexQuery,
    vertex_faces: VertexFaceQuery,
    adjacency: AdjacencyQuery,
    boundary: BoundaryQuery,
    face_normals: FaceNormalQuery,
    face_areas: FaceAreaQuery,
    centroids: FaceCentroidQuery,
    vertex_normals: VertexNormalQuery,
    smooth: SmoothQuery,
    sum_f32: ReduceQuery<f32>,
    sum_u32: ReduceQuery<u32>,
}
impl Queries {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            edges: EdgeQuery::new(device),
            faces: FaceVertexQuery::new(device),
            vertex_faces: VertexFaceQuery::new(device),
            adjacency: AdjacencyQuery::new(device),
            boundary: BoundaryQuery::new(device),
            face_normals: FaceNormalQuery::new(device),
            face_areas: FaceAreaQuery::new(device),
            centroids: FaceCentroidQuery::new(device),
            vertex_normals: VertexNormalQuery::new(device),
            smooth: SmoothQuery::new(device),
            sum_f32: ReduceQuery::new(device, ReduceOp::Sum),
            sum_u32: ReduceQuery::new(device, ReduceOp::Sum),
        }
    }
}

/// Every query result for one upload, read back as `u32` words.
struct Results {
    edges: Vec<u32>,
    faces: Vec<u32>,
    vertex_faces: Vec<u32>,
    vertex_neighbors: Vec<u32>,
    face_neighbors: Vec<u32>,
    diamonds: Vec<u32>,
    boundary_edges: Vec<u32>,
    boundary_vertices: Vec<u32>,
    face_normals: Vec<f32>,
    face_areas: Vec<f32>,
    centroids: Vec<f32>,
    vertex_normals: Vec<f32>,
    smoothed: Vec<f32>,
    area_sum: f32,
    boundary_edge_count: u32,
}

const SMOOTHING: Smoothing = Smoothing {
    step: 0.5,
    iterations: 3,
};

/// Records every query on `mesh` with `positions`, checks that each result
/// computed from connectivity carries the mesh's current stamp, and reads
/// the results back.
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    q: &Queries,
    mesh: &impl GpuTopology,
    positions: &[[f32; 4]],
) -> Results {
    let positions = GpuVertexAttribute::new(device, mesh, positions).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let e = &mut encoder;
    let edges = q.edges.encode(device, e, mesh).unwrap();
    let faces = q.faces.encode(device, e, mesh).unwrap();
    let vertex_faces = q.vertex_faces.encode(device, e, mesh).unwrap();
    let adjacency = q.adjacency.encode(device, e, mesh).unwrap();
    let boundary = q.boundary.encode(device, e, mesh).unwrap();
    let face_normals = q.face_normals.encode(device, e, mesh, &positions).unwrap();
    let face_areas = q.face_areas.encode(device, e, mesh, &positions).unwrap();
    let centroids = q.centroids.encode(device, e, mesh, &positions).unwrap();
    let vertex_normals = q
        .vertex_normals
        .encode(device, e, mesh, &positions)
        .unwrap();
    let smoothed = q
        .smooth
        .encode(
            device,
            e,
            mesh,
            &adjacency,
            &positions,
            Some(&boundary.vertices),
            SMOOTHING,
        )
        .unwrap();
    let area_sum = q.sum_f32.encode(device, e, &face_areas).unwrap();
    let boundary_edge_count = q.sum_u32.encode(device, e, &boundary.edges).unwrap();
    queue.submit([encoder.finish()]);

    // Results computed from connectivity record the stamp they were computed from.
    let current = mesh.topology();
    for stamp in [
        edges.topology(),
        faces.topology(),
        vertex_faces.topology(),
        adjacency.topology(),
    ] {
        assert_eq!(stamp, current);
    }
    for stamp in [
        boundary.edges.topology(),
        boundary.vertices.topology(),
        face_normals.topology(),
        face_areas.topology(),
        centroids.topology(),
        vertex_normals.topology(),
    ] {
        assert_eq!(stamp, Some(current));
    }
    assert_eq!(
        smoothed.topology(),
        None,
        "smoothed positions are the caller's"
    );

    let words = |b: &wgpu::Buffer| delaunay::read_words(device, queue, b);
    let floats = |b: &wgpu::Buffer| bytemuck::cast_slice(&words(b)).to_vec();
    Results {
        edges: words(edges.buffer()),
        faces: words(faces.buffer()),
        vertex_faces: words(vertex_faces.buffer()),
        vertex_neighbors: words(adjacency.vertex_neighbors()),
        face_neighbors: words(adjacency.face_neighbors()),
        diamonds: words(adjacency.edge_diamonds()),
        boundary_edges: words(boundary.edges.buffer()),
        boundary_vertices: words(boundary.vertices.buffer()),
        face_normals: floats(face_normals.buffer()),
        face_areas: floats(face_areas.buffer()),
        centroids: floats(centroids.buffer()),
        vertex_normals: floats(vertex_normals.buffer()),
        smoothed: floats(smoothed.buffer()),
        area_sum: f32::from_bits(words(area_sum.buffer())[0]),
        boundary_edge_count: words(boundary_edge_count.buffer())[0],
    }
}

/// Asserts that two float lists agree to within `tolerance`, scaled by size.
fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "{what}[{i}]: {a} vs {e}"
        );
    }
}

/// Returns each row of a CSR list with `rows` rows and `width`-word entries,
/// sorted, after mapping each entry with `map`.
fn csr_rows(
    words: &[u32],
    rows: usize,
    width: usize,
    map: impl Fn(&[u32]) -> Vec<u32>,
) -> Vec<Vec<Vec<u32>>> {
    let (offsets, entries) = words.split_at(rows + 1);
    offsets
        .windows(2)
        .map(|w| {
            let mut row: Vec<Vec<u32>> = entries[width * w[0] as usize..width * w[1] as usize]
                .chunks(width)
                .map(&map)
                .collect();
            row.sort();
            row
        })
        .collect()
}

/// Compares every query on the edited upload with the same query on a fresh
/// upload of the replayed triangles. `endpoints` are the replayed edge
/// endpoints by edited edge ID, and `triangles` the replayed triangles by
/// face ID.
fn compare(
    edited: &Results,
    fresh: &Results,
    fresh_mesh: &Mesh,
    endpoints: &[[u32; 2]],
    triangles: &[[u32; 3]],
) {
    let (vertices, faces) = (fresh_mesh.positions().len(), fresh_mesh.face_count());

    // Edge records: endpoints in ascending order and as replayed, then a
    // one-to-one map from edited to fresh edge IDs by endpoint pair.
    let fresh_ids: HashMap<[u32; 2], u32> = fresh
        .edges
        .chunks(4)
        .enumerate()
        .map(|(id, r)| ([r[0], r[1]], id as u32))
        .collect();
    assert_eq!(fresh_ids.len(), endpoints.len(), "fresh edges are distinct");
    assert_eq!(edited.edges.len(), fresh.edges.len(), "edge counts");
    let mut to_fresh = Vec::new();
    for (id, record) in edited.edges.chunks(4).enumerate() {
        let ends = [record[0], record[1]];
        assert!(
            ends[0] < ends[1],
            "edge {id}: endpoints {ends:?} not ascending"
        );
        assert_eq!(ends, endpoints[id], "edge {id}: endpoints");
        to_fresh.push(fresh_ids[&ends]);
    }
    let mut seen = to_fresh.clone();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), to_fresh.len(), "edge map is one-to-one");
    let fresh_edge = |e: u32| to_fresh[e as usize] as usize;
    for (e, record) in edited.edges.chunks(4).enumerate() {
        let f = &fresh.edges[4 * fresh_edge(e as u32)..][..4];
        let mut actual = [record[2], record[3]];
        let mut expected = [f[2], f[3]];
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected, "edge {e}: adjoining faces");
    }

    // Face vertices match the replayed triangles corner for corner, so the
    // per-corner face neighbors must match exactly too.
    assert_eq!(edited.faces, triangles.as_flattened(), "face vertices");
    assert_eq!(edited.faces, fresh.faces, "face vertices vs fresh upload");
    assert_eq!(
        edited.face_neighbors, fresh.face_neighbors,
        "face neighbors"
    );

    // Vertex rows, with edge IDs mapped to the fresh upload's IDs.
    assert_eq!(
        csr_rows(&edited.vertex_faces, vertices, 1, |f| f.to_vec()),
        csr_rows(&fresh.vertex_faces, vertices, 1, |f| f.to_vec()),
        "vertex faces"
    );
    assert_eq!(
        csr_rows(&edited.vertex_neighbors, vertices, 2, |p| vec![
            fresh_edge(p[0]) as u32,
            p[1]
        ]),
        csr_rows(&fresh.vertex_neighbors, vertices, 2, |p| p.to_vec()),
        "vertex neighbors"
    );

    // Diamonds: same endpoints, and the same (face, opposite vertex) pairs.
    for (e, d) in edited.diamonds.chunks(4).enumerate() {
        let fe = fresh_edge(e as u32);
        let f = &fresh.diamonds[4 * fe..][..4];
        assert_eq!(d[..2], f[..2], "edge {e}: diamond endpoints");
        let pairs = |faces: &[u32], d: &[u32]| {
            let mut p = [[faces[2], d[2]], [faces[3], d[3]]];
            p.sort();
            p
        };
        assert_eq!(
            pairs(&edited.edges[4 * e..], d),
            pairs(&fresh.edges[4 * fe..], f),
            "edge {e}: diamond opposites"
        );
    }

    // Boundary flags, by vertex ID and through the edge map.
    assert_eq!(
        edited.boundary_vertices, fresh.boundary_vertices,
        "boundary vertices"
    );
    for (e, &flag) in edited.boundary_edges.iter().enumerate() {
        assert_eq!(
            flag,
            fresh.boundary_edges[fresh_edge(e as u32)],
            "edge {e}: boundary"
        );
    }
    assert_eq!(
        edited.boundary_edge_count, fresh.boundary_edge_count,
        "boundary count"
    );

    // Geometry and smoothing, by preserved face and vertex IDs.
    assert_close(
        &edited.face_normals,
        &fresh.face_normals,
        1e-6,
        "face normals",
    );
    assert_close(&edited.face_areas, &fresh.face_areas, 1e-6, "face areas");
    assert_close(&edited.centroids, &fresh.centroids, 1e-6, "centroids");
    assert_close(
        &edited.vertex_normals,
        &fresh.vertex_normals,
        1e-5,
        "vertex normals",
    );
    assert_close(
        &edited.smoothed,
        &fresh.smoothed,
        1e-5,
        "smoothed positions",
    );
    assert_close(&[edited.area_sum], &[fresh.area_sum], 1e-5, "area sum");
    assert_eq!(edited.face_areas.len(), faces);
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_edited -- --ignored --nocapture"]
fn queries_on_an_edited_mesh_match_a_fresh_upload_of_the_flipped_triangles() {
    let (device, queue) = gpu_device();
    let q = Queries::new(&device);
    let flip = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &flip);

    // Several patches, with so little spare room that most flips must reuse
    // slots freed by earlier batches. Renumbering the grid's vertices makes
    // the new edges' endpoints arrive in both orders.
    let (grid, grid_positions, grid_triangles) = delaunay::skewed_grid(17, 64);
    let renumber = |v: u32| v * 37 % grid_positions.len() as u32;
    let mut positions = grid_positions.clone();
    for (v, &p) in grid_positions.iter().enumerate() {
        positions[renumber(v as u32) as usize] = p;
    }
    let mut triangles: Vec<[u32; 3]> = grid_triangles.iter().map(|t| t.map(renumber)).collect();
    let xyz: Vec<[f32; 3]> = positions.iter().map(|&[x, y, z, _]| [x, y, z]).collect();
    let mesh = MeshBuilder::new()
        .patch_size(64)
        .positions(&xyz)
        .triangles(&triangles)
        .build()
        .unwrap();
    assert_eq!(mesh.face_count(), grid.face_count());
    assert!(mesh.patches().len() > 1);
    let capacity = EditCapacity::new(1.02).unwrap();
    let mut editable = GpuEditableMesh::upload(&device, &mesh, capacity).unwrap();
    let position_buffer = wgpu::util::DeviceExt::create_buffer_init(
        &device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("edited positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE,
        },
    );

    // First flip some edges into free slots, one at a time, so the patches
    // have empty slots below their used counts and reused slots. Replay
    // each flip on the CPU.
    let mut endpoints = replay::edge_endpoints(&mesh);
    let some_edges: Vec<u32> = (0..mesh.edge_count() as u32).step_by(5).collect();
    let fresh = fresh_flips::FreshFlips::new(&device).flip(
        &device,
        &queue,
        &mut editable,
        &position_buffer,
        &some_edges,
    );
    let mut fresh_flipped = 0;
    for (&edge, status) in some_edges.iter().zip(&fresh) {
        if *status == Some(FlipStatus::Flipped) {
            replay::replay(&mut triangles, &mut endpoints, edge as usize);
            fresh_flipped += 1;
        }
    }
    assert!(
        fresh_flipped > mesh.patches().len(),
        "only {fresh_flipped} flips into free slots"
    );
    assert!(
        fresh.contains(&Some(FlipStatus::OutOfCapacity)),
        "free slots never ran out"
    );

    // Then flip in batches until one flips nothing, replaying each reported
    // flip on the CPU.
    let (mut flipped, mut stalled) = (0, false);
    for _ in 0..1000 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let status = flips
            .encode(
                &device,
                &mut encoder,
                &flip,
                &mut editable,
                &position_buffer,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let words = delaunay::read_words(&device, &queue, &status);
        for (edge, &word) in words.iter().enumerate() {
            if FlipStatus::from_word(word) == Some(FlipStatus::Flipped) {
                replay::replay(&mut triangles, &mut endpoints, edge);
                flipped += 1;
            }
        }
        if delaunay::Round::from_statuses(&words).flipped == 0 {
            stalled = true;
            break;
        }
    }
    assert!(stalled, "flipping never stopped");
    assert!(
        flipped > mesh.patches().len(),
        "only {flipped} flips in batches"
    );

    // Compare against a fresh static upload of the replayed triangles.
    let fresh_mesh = MeshBuilder::new()
        .patch_size(64)
        .positions(mesh.positions())
        .triangles(&triangles)
        .build()
        .unwrap();
    let fresh = GpuMesh::upload(&device, &fresh_mesh).unwrap();
    let edited_results = run(&device, &queue, &q, &editable, &positions);
    let fresh_results = run(&device, &queue, &q, &fresh, &positions);
    compare(
        &edited_results,
        &fresh_results,
        &fresh_mesh,
        &endpoints,
        &triangles,
    );
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_edited -- --ignored --nocapture"]
fn results_from_before_an_edit_are_rejected_as_stale() {
    let (device, queue) = gpu_device();
    let (mesh, positions, _) = delaunay::skewed_grid(5, 256);
    let flip = EdgeFlipQuery::new(&device);
    let adjacency_query = AdjacencyQuery::new(&device);
    let boundary_query = BoundaryQuery::new(&device);
    let areas = FaceAreaQuery::new(&device);
    let smooth = SmoothQuery::new(&device);
    let dot = DotQuery::new(&device);
    let mut editable = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
    let gpu_positions = GpuVertexAttribute::new(&device, &editable, &positions).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let e = &mut encoder;

    // Results computed at version 0 work at version 0.
    let adjacency = adjacency_query.encode(&device, e, &editable).unwrap();
    let mut boundary = boundary_query.encode(&device, e, &editable).unwrap();
    let old_areas = areas.encode(&device, e, &editable, &gpu_positions).unwrap();
    let smoothing = Smoothing {
        step: 0.5,
        iterations: 1,
    };
    let smooth_with = |e: &mut wgpu::CommandEncoder,
                       mesh: &GpuEditableMesh,
                       adjacency: &mopmesh::gpu::MeshAdjacency,
                       fixed: Option<&GpuVertexAttribute<u32>>| {
        smooth
            .encode(
                &device,
                e,
                mesh,
                adjacency,
                &gpu_positions,
                fixed,
                smoothing,
            )
            .map(drop)
    };
    assert_eq!(
        smooth_with(e, &editable, &adjacency, Some(&boundary.vertices)),
        Ok(())
    );

    // An empty batch still counts as an edit, so every earlier result is stale.
    flip.encode(&device, e, &mut editable, &[]).unwrap();
    assert_eq!(editable.version(), 1);
    assert_eq!(
        smooth_with(e, &editable, &adjacency, None),
        Err(GpuError::Stale)
    );
    let fresh_adjacency = adjacency_query.encode(&device, e, &editable).unwrap();
    assert_eq!(
        smooth_with(e, &editable, &fresh_adjacency, Some(&boundary.vertices)),
        Err(GpuError::Stale),
        "computed boundary flags are stale too"
    );
    assert_eq!(smooth_with(e, &editable, &fresh_adjacency, None), Ok(()));
    let new_areas = areas.encode(&device, e, &editable, &gpu_positions).unwrap();
    assert_eq!(
        dot.encode(&device, e, &old_areas, &new_areas).err(),
        Some(GpuError::Stale)
    );
    assert!(dot.encode(&device, e, &new_areas, &new_areas).is_ok());

    // An edit counted by the caller's own kernel makes results stale the same way.
    editable.mark_edited();
    assert_eq!(editable.version(), 2);
    assert_eq!(
        smooth_with(e, &editable, &fresh_adjacency, None),
        Err(GpuError::Stale)
    );
    let fresh_adjacency = adjacency_query.encode(&device, e, &editable).unwrap();

    // Replacing computed values makes them the caller's, which edits don't affect.
    let flags = vec![0u32; positions.len()];
    boundary.vertices.write_values(&queue, &flags).unwrap();
    assert_eq!(boundary.vertices.topology(), None);
    assert_eq!(
        smooth_with(e, &editable, &fresh_adjacency, Some(&boundary.vertices)),
        Ok(())
    );

    // Another editable upload of the same mesh, even at the same version,
    // doesn't accept this upload's results.
    let mut other = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
    flip.encode(&device, e, &mut other, &[]).unwrap();
    other.mark_edited();
    assert_eq!(other.version(), editable.version());
    assert_eq!(
        smooth_with(e, &other, &fresh_adjacency, None),
        Err(GpuError::Stale)
    );

    // A result from another mesh is a mesh mismatch, even when it is also old.
    let (other_mesh, _, _) = delaunay::skewed_grid(5, 256);
    let unrelated = GpuEditableMesh::upload(&device, &other_mesh, EditCapacity::default()).unwrap();
    let unrelated_adjacency = adjacency_query.encode(&device, e, &unrelated).unwrap();
    assert_eq!(
        smooth_with(e, &editable, &unrelated_adjacency, None),
        Err(GpuError::MeshMismatch)
    );
    drop(encoder);

    // Static uploads of one mesh never change, so they share results.
    let first = GpuMesh::upload(&device, &mesh).unwrap();
    let second = GpuMesh::upload(&device, &mesh).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let shared = adjacency_query
        .encode(&device, &mut encoder, &first)
        .unwrap();
    let static_positions = GpuVertexAttribute::new(&device, &second, &positions).unwrap();
    assert!(
        smooth
            .encode(
                &device,
                &mut encoder,
                &second,
                &shared,
                &static_positions,
                None,
                smoothing
            )
            .is_ok()
    );
}

fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    pollster::block_on(adapter.request_device(&Default::default())).unwrap()
}
