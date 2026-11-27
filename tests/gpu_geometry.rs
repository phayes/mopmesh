//! GPU geometry queries (face normals, face areas, vertex normals) and
//! boundary flags checked against Alum and an f64 reference, on uploaded and
//! GPU-built meshes.
use alum::{Handle, HasTopology, PolyMeshF32, Vec3};
use mopmesh::{
    FaceAttribute, MeshBuilder,
    gpu::{
        BoundaryQuery, FaceAreaQuery, FaceNormalQuery, GpuError, GpuFaceAttribute, GpuMesh,
        GpuMeshBuilder, GpuMeshInput, GpuVertexAttribute, VertexNormalQuery,
    },
};
use wgpu::util::DeviceExt;

#[path = "support/fixtures.rs"]
mod fixtures;

struct Queries {
    face_normals: FaceNormalQuery,
    face_areas: FaceAreaQuery,
    vertex_normals: VertexNormalQuery,
    boundary: BoundaryQuery,
}

// Explicit opt-in: a missing adapter is a failure, never a silent successful test.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_geometry -- --ignored --nocapture"]
fn gpu_geometry_matches_reference() {
    let (device, queue) = gpu_device();
    let queries = Queries {
        face_normals: FaceNormalQuery::new(&device),
        face_areas: FaceAreaQuery::new(&device),
        vertex_normals: VertexNormalQuery::new(&device),
        boundary: BoundaryQuery::new(&device),
    };
    let mut builder = GpuMeshBuilder::new(&device);
    for fixture in geometry_fixtures() {
        eprintln!("checking {}", fixture.name);
        let reference = alum_mesh(&fixture.positions, &fixture.triangles);
        for patch_size in fixtures::PATCH_SIZES {
            // A CPU-built mesh, uploaded.
            let mesh = MeshBuilder::new()
                .positions(&fixture.positions)
                .triangles(&fixture.triangles)
                .patch_size(patch_size)
                .build()
                .unwrap();
            let gpu = GpuMesh::upload(&device, &mesh).unwrap();
            let positions = gpu
                .upload_attribute(&device, &mesh.position_attribute())
                .unwrap();
            check(
                &device, &queue, &queries, &reference, &fixture, &gpu, &positions,
            );

            // The same mesh built on the GPU, reading the build's positions.
            builder = builder.patch_size(patch_size as u32);
            let (position_buffer, triangle_buffer) =
                input_buffers(&device, &fixture.positions, &fixture.triangles);
            let input = GpuMeshInput {
                positions: &position_buffer,
                triangles: &triangle_buffer,
                vertex_count: fixture.positions.len() as u32,
                face_count: fixture.triangles.len() as u32,
            };
            let gpu = builder.build(&device, &queue, &input).unwrap();
            let positions =
                GpuVertexAttribute::<[f32; 4]>::from_buffer(&gpu, &position_buffer).unwrap();
            check(
                &device, &queue, &queries, &reference, &fixture, &gpu, &positions,
            );
        }
    }

    // Positions from another mesh are rejected before any GPU work.
    let (positions, triangles) = fixtures::grid(3);
    let build = || {
        MeshBuilder::new()
            .positions(&positions)
            .triangles(&triangles)
            .build()
            .unwrap()
    };
    let (mesh, other) = (build(), build());
    let gpu = GpuMesh::upload(&device, &mesh).unwrap();
    let other_gpu = GpuMesh::upload(&device, &other).unwrap();
    let wrong = other_gpu
        .upload_attribute(&device, &other.position_attribute())
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let mismatch = Some(GpuError::MeshMismatch);
    let encoder = &mut encoder;
    assert_eq!(
        queries
            .face_normals
            .encode(&device, encoder, &gpu, &wrong)
            .err(),
        mismatch
    );
    assert_eq!(
        queries
            .face_areas
            .encode(&device, encoder, &gpu, &wrong)
            .err(),
        mismatch
    );
    assert_eq!(
        queries
            .vertex_normals
            .encode(&device, encoder, &gpu, &wrong)
            .err(),
        mismatch
    );
}

/// Edge and face attributes upload, round-trip and accept slice writes.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_geometry -- --ignored --nocapture"]
fn gpu_edge_and_face_attributes_round_trip() {
    let (device, queue) = gpu_device();
    let (positions, triangles) = fixtures::grid(5);
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .patch_size(3)
        .build()
        .unwrap();
    let gpu = GpuMesh::upload(&device, &mesh).unwrap();

    // Uploads keep canonical order for both element kinds.
    let face_ramp: Vec<u32> = (0..mesh.face_count() as u32).map(|f| 2 * f + 1).collect();
    let faces = FaceAttribute::from_values(&mesh, face_ramp.clone()).unwrap();
    let gpu_faces = gpu.upload_attribute(&device, &faces).unwrap();
    assert_eq!(gpu_faces.len() as usize, mesh.face_count());
    assert_eq!(read(&device, &queue, gpu_faces.buffer()), face_ramp);
    let edges = mopmesh::EdgeAttribute::new(&mesh, 9u32);
    let mut gpu_edges = gpu.upload_attribute(&device, &edges).unwrap();
    assert_eq!(gpu_edges.len() as usize, mesh.edge_count());
    assert_eq!(
        read(&device, &queue, gpu_edges.buffer()),
        vec![9; mesh.edge_count()]
    );

    // Slice writes replace every value and check the length.
    let edge_ramp: Vec<u32> = (0..mesh.edge_count() as u32).collect();
    gpu_edges.write_values(&queue, &edge_ramp).unwrap();
    assert_eq!(read(&device, &queue, gpu_edges.buffer()), edge_ramp);
    assert_eq!(
        gpu_edges.write_values(&queue, &edge_ramp[1..]),
        Err(GpuError::InvalidBuffer {
            resource: "edge attribute"
        })
    );

    // Direct construction checks the element count for its kind.
    assert_eq!(
        GpuFaceAttribute::new(&device, &gpu, &vec![0u32; mesh.edge_count()]).err(),
        Some(GpuError::InvalidBuffer {
            resource: "face attribute"
        })
    );
}

/// Runs the three queries on `gpu` and compares them with the references.
fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &Queries,
    reference: &PolyMeshF32,
    fixture: &fixtures::Fixture,
    gpu: &GpuMesh,
    positions: &GpuVertexAttribute<[f32; 4]>,
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let face_normals = queries
        .face_normals
        .encode(device, &mut encoder, gpu, positions)
        .unwrap();
    let face_areas = queries
        .face_areas
        .encode(device, &mut encoder, gpu, positions)
        .unwrap();
    let vertex_normals = queries
        .vertex_normals
        .encode(device, &mut encoder, gpu, positions)
        .unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(face_normals.len() as usize, fixture.triangles.len());
    assert_eq!(face_areas.len() as usize, fixture.triangles.len());
    assert_eq!(vertex_normals.len() as usize, fixture.positions.len());
    for result in [
        face_normals.mesh_id(),
        face_areas.mesh_id(),
        vertex_normals.mesh_id(),
    ] {
        assert_eq!(result, gpu.mesh_id());
    }

    // Face normals and areas against Alum. Alum gives NaN normals for
    // zero-area faces, where MOPMesh documents the zero vector.
    let normals: Vec<f32> = bits(read(device, queue, face_normals.buffer()));
    let areas: Vec<f32> = bits(read(device, queue, face_areas.buffer()));
    for face in reference.faces() {
        let f = face.index() as usize;
        let Vec3(x, y, z) = reference.try_calc_face_normal(face).unwrap();
        let expected = if x.is_nan() { [0.; 3] } else { [x, y, z] };
        assert_vector(&normals[4 * f..4 * f + 4], expected, "face normal", f);
        let expected = reference.try_calc_face_area(face).unwrap();
        let actual = areas[f];
        assert!(
            (actual - expected).abs() <= 1e-5 * expected.abs().max(1.),
            "face area {f}: {actual} vs {expected}"
        );
    }

    // Vertex normals against an f64 area-weighted sum over each vertex's faces.
    let normals: Vec<f32> = bits(read(device, queue, vertex_normals.buffer()));
    for (v, expected) in reference_vertex_normals(fixture).into_iter().enumerate() {
        assert_vector(&normals[4 * v..4 * v + 4], expected, "vertex normal", v);
    }

    // Boundary flags against Alum. Canonical edge IDs number the edges in
    // ascending (smaller, larger) vertex-pair order.
    let mut encoder = device.create_command_encoder(&Default::default());
    let boundary = queries.boundary.encode(device, &mut encoder, gpu).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(boundary.edges.len() as usize, reference.edges().len());
    assert_eq!(boundary.vertices.len() as usize, fixture.positions.len());
    let edge_flags = read(device, queue, boundary.edges.buffer());
    let mut pairs: Vec<[u32; 2]> = reference
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(reference);
            [a.index().min(b.index()), a.index().max(b.index())]
        })
        .collect();
    let alum_edges: Vec<_> = reference.edges().collect();
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    order.sort_by_key(|&i| pairs[i]);
    pairs.sort();
    for (id, &i) in order.iter().enumerate() {
        let expected = u32::from(alum_edges[i].is_boundary(reference));
        assert_eq!(edge_flags[id], expected, "boundary edge {:?}", pairs[id]);
    }
    let vertex_flags = read(device, queue, boundary.vertices.buffer());
    for v in reference.vertices() {
        let expected = u32::from(v.is_boundary(reference));
        assert_eq!(
            vertex_flags[v.index() as usize],
            expected,
            "boundary vertex {v:?}"
        );
    }
}

/// Area-weighted vertex normals in f64: the normalized sum of each incident
/// face's cross product, or zero when that sum is zero.
fn reference_vertex_normals(fixture: &fixtures::Fixture) -> Vec<[f32; 3]> {
    let p = |v: u32| fixture.positions[v as usize].map(f64::from);
    let mut sums = vec![[0f64; 3]; fixture.positions.len()];
    for &[a, b, c] in &fixture.triangles {
        let (pa, pb, pc) = (p(a), p(b), p(c));
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let w = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let cross = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        for v in [a, b, c] {
            for i in 0..3 {
                sums[v as usize][i] += cross[i];
            }
        }
    }
    sums.into_iter()
        .map(|s| {
            let length = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            if length == 0. {
                [0.; 3]
            } else {
                s.map(|x| (x / length) as f32)
            }
        })
        .collect()
}

/// Checks one `[x, y, z, 0]` result against `expected`. A zero expectation
/// must be exactly zero.
fn assert_vector(actual: &[f32], expected: [f32; 3], what: &str, index: usize) {
    assert_eq!(actual[3], 0., "{what} {index}: w lane");
    for i in 0..3 {
        let ok = if expected == [0.; 3] {
            actual[i] == 0.
        } else {
            (actual[i] - expected[i]).abs() <= 1e-5
        };
        assert!(ok, "{what} {index}: {:?} vs {expected:?}", &actual[..3]);
    }
}

/// The shared fixtures, plus copies lifted off the z = 0 plane so that normals
/// vary from face to face.
fn geometry_fixtures() -> Vec<fixtures::Fixture> {
    let mut all = fixtures::valid();
    let lifted: Vec<_> = all
        .iter()
        .filter(|f| !f.positions.is_empty() && f.name != "collinear triangle")
        .map(|f| fixtures::Fixture {
            name: "lifted",
            positions: f
                .positions
                .iter()
                .map(|&[x, y, z]| [x, y, z + 0.5 * (0.7 * x).sin() * (1.3 * y).cos()])
                .collect(),
            triangles: f.triangles.clone(),
        })
        .collect();
    all.extend(lifted);
    all
}

fn alum_mesh(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> PolyMeshF32 {
    let mut reference = PolyMeshF32::new();
    for &[x, y, z] in positions {
        reference.add_vertex(Vec3(x, y, z)).unwrap();
    }
    for tri in triangles {
        reference.add_face(&tri.map(Into::into)).unwrap();
    }
    reference
}

fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    pollster::block_on(adapter.request_device(&Default::default())).unwrap()
}

/// Uploads positions and triangles in the vec4 layout [`GpuMeshBuilder`]
/// reads, each buffer at least 16 bytes.
fn input_buffers(
    device: &wgpu::Device,
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> (wgpu::Buffer, wgpu::Buffer) {
    let mut positions: Vec<[f32; 4]> = positions.iter().map(|&[x, y, z]| [x, y, z, 0.]).collect();
    positions.resize(positions.len().max(1), [0.; 4]);
    let mut triangles: Vec<[u32; 4]> = triangles.iter().map(|&[a, b, c]| [a, b, c, 0]).collect();
    triangles.resize(triangles.len().max(1), [0; 4]);
    let create = |label, contents| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    (
        create("test positions", bytemuck::cast_slice(&positions)),
        create("test triangles", bytemuck::cast_slice(&triangles)),
    )
}

fn bits(words: Vec<u32>) -> Vec<f32> {
    words.into_iter().map(f32::from_bits).collect()
}

/// Copies `buffer` back to the host after all submitted work.
fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let words = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    words
}
