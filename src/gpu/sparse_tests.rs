//! Queries and edits on editable meshes whose used IDs have unused IDs
//! between them, compared with a static upload of the same mesh.
use super::editable::{editable_contents, inspect};
use super::test_support::{self, fixtures};
use super::{
    AdjacencyQuery, DotQuery, EdgeFlipQuery, EditCapacity, FaceAreaQuery, FaceCentroidQuery,
    FaceNormalQuery, FlipStatus, GpuEditableMesh, GpuError, GpuMesh, GpuVertexAttribute, ReduceOp,
    ReduceQuery, SmoothQuery, Smoothing, VertexFaceQuery, VertexNormalQuery,
};
use crate::{Mesh, MeshBuilder};

struct Queries {
    normals: FaceNormalQuery,
    areas: FaceAreaQuery,
    centroids: FaceCentroidQuery,
    vertex_normals: VertexNormalQuery,
    vertex_faces: VertexFaceQuery,
    adjacency: AdjacencyQuery,
    smooth: SmoothQuery,
    sum: ReduceQuery<f32>,
    min: ReduceQuery<f32>,
    max: ReduceQuery<f32>,
    dot: DotQuery,
}

/// Per-ID results of every geometry query, read back as words.
struct Results {
    normals: Vec<u32>,
    areas: Vec<u32>,
    centroids: Vec<u32>,
    vertex_normals: Vec<u32>,
    vertex_faces: Vec<u32>,
    smoothed: Vec<u32>,
    sum: f32,
    min: f32,
    max: f32,
    dot: f32,
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let encoder = device.create_command_encoder(&Default::default());
    test_support::read_words(device, queue, encoder, buffer)
}

/// Runs every query on `mesh` with `positions`, one `[x, y, z, 1]` per
/// vertex ID.
fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    q: &Queries,
    mesh: &impl super::GpuTopology,
    positions: &[[f32; 4]],
) -> Results {
    let positions = GpuVertexAttribute::new(device, mesh, positions).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let e = &mut encoder;
    let normals = q.normals.encode(device, e, mesh, &positions).unwrap();
    let areas = q.areas.encode(device, e, mesh, &positions).unwrap();
    let centroids = q.centroids.encode(device, e, mesh, &positions).unwrap();
    let vertex_normals = q
        .vertex_normals
        .encode(device, e, mesh, &positions)
        .unwrap();
    let vertex_faces = q.vertex_faces.encode(device, e, mesh).unwrap();
    let adjacency = q.adjacency.encode(device, e, mesh).unwrap();
    let settings = Smoothing {
        step: 0.5,
        iterations: 3,
    };
    let smoothed = q
        .smooth
        .encode(device, e, mesh, &adjacency, &positions, None, settings)
        .unwrap();
    let sum = q.sum.encode(device, e, mesh, &areas).unwrap();
    let min = q.min.encode(device, e, mesh, &areas).unwrap();
    let max = q.max.encode(device, e, mesh, &areas).unwrap();
    let dot = q.dot.encode(device, e, mesh, &areas, &areas).unwrap();
    queue.submit([encoder.finish()]);
    let value = |b: &wgpu::Buffer| f32::from_bits(read(device, queue, b)[0]);
    Results {
        normals: read(device, queue, normals.buffer()),
        areas: read(device, queue, areas.buffer()),
        centroids: read(device, queue, centroids.buffer()),
        vertex_normals: read(device, queue, vertex_normals.buffer()),
        vertex_faces: read(device, queue, vertex_faces.buffer()),
        smoothed: read(device, queue, smoothed.buffer()),
        sum: value(sum.buffer()),
        min: value(min.buffer()),
        max: value(max.buffer()),
        dot: value(dot.buffer()),
    }
}

/// Checks that `words`, with `width` words per ID, hold `expected`'s row
/// `i` at row `spread(i)`, within `tolerance` as floats, and zeros in every
/// other row below `capacity`.
fn check_rows(
    words: &[u32],
    expected: &[u32],
    width: usize,
    capacity: u32,
    tolerance: f32,
    name: &str,
) {
    let count = expected.len() / width;
    let mut used = vec![false; capacity as usize];
    for i in 0..count as u32 {
        used[inspect::spread(i) as usize] = true;
    }
    for (new, row) in words.chunks(width).take(capacity as usize).enumerate() {
        if !used[new] {
            assert!(row.iter().all(|&w| w == 0), "{name}: unused row {new}");
            continue;
        }
        let i = (new - 1) / 2;
        let want = &expected[i * width..(i + 1) * width];
        for (a, b) in row.iter().zip(want) {
            let (a, b) = (f32::from_bits(*a), f32::from_bits(*b));
            assert!((a - b).abs() <= tolerance, "{name}: row {i}: {a} vs {b}");
        }
    }
}

/// Returns the sorted face IDs around each vertex of a vertex-face CSR with
/// `vertices` rows, renumbering IDs with `back`.
fn face_rows(words: &[u32], vertices: usize, back: impl Fn(u32) -> u32) -> Vec<Vec<u32>> {
    let (offsets, faces) = words.split_at(vertices + 1);
    (0..vertices)
        .map(|v| {
            let mut row: Vec<u32> = faces[offsets[v] as usize..offsets[v + 1] as usize]
                .iter()
                .map(|&f| back(f))
                .collect();
            row.sort();
            row
        })
        .collect()
}

fn check(device: &wgpu::Device, queue: &wgpu::Queue, q: &Queries, mesh: &Mesh, context: &str) {
    // The static upload, positions indexed by vertex ID.
    let gpu = GpuMesh::upload(device, mesh).unwrap();
    let positions: Vec<[f32; 4]> = mesh
        .positions()
        .iter()
        .map(|&[x, y, z]| [x, y, z, 1.])
        .collect();
    let expected = run(device, queue, q, &gpu, &positions);

    // The same mesh with spread IDs; unused vertex IDs hold NaN positions,
    // which no query may read.
    let mut words = editable_contents(mesh, EditCapacity::default(), &device.limits()).unwrap();
    inspect::spread_ids(&mut words);
    let editable = GpuEditableMesh::from_words(device, mesh.id(), &words);
    let [vertices, _, faces] = inspect::id_capacities(&words.arena);
    let mut spread_positions = vec![[f32::NAN; 4]; vertices as usize];
    for (v, &p) in positions.iter().enumerate() {
        spread_positions[inspect::spread(v as u32) as usize] = p;
    }
    let actual = run(device, queue, q, &editable, &spread_positions);

    // Per-face results match exactly; per-vertex sums may round differently.
    check_rows(&actual.normals, &expected.normals, 4, faces, 0., context);
    check_rows(&actual.areas, &expected.areas, 1, faces, 0., context);
    check_rows(
        &actual.centroids,
        &expected.centroids,
        4,
        faces,
        0.,
        context,
    );
    check_rows(
        &actual.vertex_normals,
        &expected.vertex_normals,
        4,
        vertices,
        1e-5,
        context,
    );

    // Smoothed positions match for used vertices; unused ones keep NaN.
    let n = positions.len();
    for (v, want) in expected.smoothed.chunks(4).take(n).enumerate() {
        let at = 4 * inspect::spread(v as u32) as usize;
        for (a, b) in actual.smoothed[at..at + 4].iter().zip(want) {
            let (a, b) = (f32::from_bits(*a), f32::from_bits(*b));
            assert!((a - b).abs() <= 1e-5, "{context}: smoothed vertex {v}");
        }
    }

    // Vertex-face rows match once renumbered; unused vertices have none.
    let back = |f: u32| (f - 1) / 2;
    let rows = face_rows(&actual.vertex_faces, vertices as usize, back);
    let want = face_rows(&expected.vertex_faces, n, |f| f);
    for (v, row) in rows.iter().enumerate() {
        if v % 2 == 1 && (v as u32 - 1) / 2 < n as u32 {
            assert_eq!(*row, want[(v - 1) / 2], "{context}: faces of vertex {v}");
        } else {
            assert!(row.is_empty(), "{context}: unused vertex {v} has faces");
        }
    }
    let total = actual.vertex_faces[vertices as usize];
    assert_eq!(
        total as usize,
        3 * mesh.face_count(),
        "{context}: CSR total"
    );

    // Reductions skip unused face IDs.
    let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * b.abs().max(1.);
    assert!(close(actual.sum, expected.sum), "{context}: sum");
    assert!(close(actual.dot, expected.dot), "{context}: dot");
    assert_eq!(actual.min, expected.min, "{context}: min");
    assert_eq!(actual.max, expected.max, "{context}: max");

    // An attribute sized for another ID capacity is rejected.
    let short = GpuVertexAttribute::new(device, &gpu, &positions).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    assert_eq!(
        q.normals
            .encode(device, &mut encoder, &editable, &short)
            .err(),
        Some(GpuError::InvalidBuffer {
            resource: "vertex attribute"
        }),
        "{context}"
    );
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn queries_skip_unused_ids_between_used_ones() {
    let (device, queue) = test_support::device(wgpu::Limits::default());
    let q = Queries {
        normals: FaceNormalQuery::new(&device),
        areas: FaceAreaQuery::new(&device),
        centroids: FaceCentroidQuery::new(&device),
        vertex_normals: VertexNormalQuery::new(&device),
        vertex_faces: VertexFaceQuery::new(&device),
        adjacency: AdjacencyQuery::new(&device),
        smooth: SmoothQuery::new(&device),
        sum: ReduceQuery::new(&device, ReduceOp::Sum),
        min: ReduceQuery::new(&device, ReduceOp::Min),
        max: ReduceQuery::new(&device, ReduceOp::Max),
        dot: DotQuery::new(&device),
    };
    let limits = device.limits();
    for fixture in fixtures::valid() {
        if fixture.triangles.is_empty() {
            continue;
        }
        for patch_size in fixtures::PATCH_SIZES {
            let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
            let context = format!("{} at patch size {patch_size}", fixture.name);
            check(&device, &queue, &q, &mesh, &context);
        }
    }
    for base in super::bench_meshes::generated(2000) {
        let mesh = MeshBuilder::new()
            .positions(&base.positions)
            .triangles(&base.triangles)
            .patch_size(64)
            .build()
            .unwrap();
        check(&device, &queue, &q, &mesh, &base.name);
    }
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn flips_report_unused_edge_ids_and_keep_the_mesh_valid() {
    let (device, queue) = test_support::device(wgpu::Limits::default());
    let flip = EdgeFlipQuery::new(&device);
    let base = &super::bench_meshes::generated(2000)[0];
    let mesh = MeshBuilder::new()
        .positions(&base.positions)
        .triangles(&base.triangles)
        .patch_size(64)
        .build()
        .unwrap();
    let mut words = editable_contents(&mesh, EditCapacity::default(), &device.limits()).unwrap();
    inspect::spread_ids(&mut words);
    let mut editable = GpuEditableMesh::from_words(&device, mesh.id(), &words);

    // Request every edge ID: the unused ones report NotAnEdge, the used ones
    // anything else.
    let edges: Vec<u32> = (0..editable.edge_capacity()).collect();
    let mut encoder = device.create_command_encoder(&Default::default());
    let statuses = flip
        .encode(&device, &mut encoder, &mut editable, &edges)
        .unwrap();
    queue.submit([encoder.finish()]);
    let words_out = read(&device, &queue, statuses.buffer());
    let mut flipped = 0;
    for (e, &w) in words_out.iter().take(edges.len()).enumerate() {
        let status = FlipStatus::from_word(w).unwrap_or_else(|| panic!("edge {e}: status {w}"));
        let used = e % 2 == 1 && (e as u32 - 1) / 2 < mesh.edge_count() as u32;
        if used {
            assert_ne!(status, FlipStatus::NotAnEdge, "used edge {e}");
            flipped += u32::from(status == FlipStatus::Flipped);
        } else {
            assert_eq!(status, FlipStatus::NotAnEdge, "unused edge {e}");
        }
    }
    assert!(flipped > 0, "some edges flip");

    // The edited arena keeps every invariant, with the same unused IDs.
    let arena = read(&device, &queue, editable.arena());
    let masks = read(&device, &queue, editable.masks());
    let maps = read(&device, &queue, editable.maps());
    inspect::check_invariants(&arena, &masks, &maps);
    for e in (0..editable.edge_capacity()).step_by(2) {
        assert!(!inspect::alive(&arena, 1, e), "edge {e} stays unused");
    }
}
