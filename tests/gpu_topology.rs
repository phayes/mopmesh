use alum::{Handle, HasIterators, HasTopology, PolyMeshF32, Vec3};
use mopmesh::{
    BuildError, MeshBuilder, VertexAttribute, VertexHandle,
    gpu::{
        AdjacencyQuery, EdgeQuery, FaceCentroidQuery, FaceVertexQuery, GpuBuildError, GpuError,
        GpuMesh, GpuMeshBuilder, GpuMeshInput, GpuVertexAttribute, INVALID_ID, VertexFaceQuery,
    },
};
use std::collections::{BTreeMap, BTreeSet};
use wgpu::util::DeviceExt;
#[path = "../examples/support/boundary.rs"]
mod boundary;
#[path = "support/fixtures.rs"]
mod fixtures;
#[path = "../examples/support/neighborhood.rs"]
mod neighborhood;

/// Requests the default adapter and a device with `limits`, failing the test
/// if there is no adapter: a missing GPU is never a silent success.
fn gpu_device(limits: wgpu::Limits) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    eprintln!("GPU: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap()
}

// Explicit opt-in: a missing adapter is a failure, never a silent successful test.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_topology -- --ignored --nocapture"]
fn gpu_topology_queries_match_alum() {
    let (device, queue) = gpu_device(Default::default());
    let queries = Queries::new(&device);
    let mut builder = GpuMeshBuilder::new(&device);
    for fixture in fixtures::valid() {
        eprintln!("checking {}", fixture.name);
        let mut reference = alum_mesh(&fixture.positions, &fixture.triangles);
        check_uploaded(&device, &queue, &queries, &mut reference, &fixture);
        for patch_size in fixtures::PATCH_SIZES {
            builder = builder.patch_size(patch_size as u32);
            check_gpu_built(
                &device,
                &queue,
                &queries,
                &builder,
                &mut reference,
                &fixture,
            );
        }
    }
    // Reject oversized uploads and builds before creating a GPU buffer.
    let (positions, triangles) = fixtures::grid(9);
    let (small_device, small_queue) = gpu_device(wgpu::Limits {
        max_storage_buffer_binding_size: 64,
        ..Default::default()
    });
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap();
    assert!(matches!(
        GpuMesh::upload(&small_device, &mesh),
        Err(GpuError::LimitsExceeded {
            resource: "connectivity arena"
        })
    ));
    let (position_buffer, triangle_buffer) = input_buffers(&small_device, &positions, &triangles);
    let input = GpuMeshInput {
        positions: &position_buffer,
        triangles: &triangle_buffer,
        vertex_count: positions.len() as u32,
        face_count: triangles.len() as u32,
    };
    assert!(matches!(
        GpuMeshBuilder::new(&small_device).build(&small_device, &small_queue, &input),
        Err(GpuBuildError::Gpu(GpuError::LimitsExceeded { .. }))
    ));
    // Face work can fit while edge work exceeds the dispatch limit.
    let (limited_device, _limited_queue) = gpu_device(wgpu::Limits {
        max_compute_workgroups_per_dimension: 1,
        ..Default::default()
    });
    let positions: Vec<_> = (0..22)
        .flat_map(|i| {
            let x = i as f32 * 3.;
            [[x, 0., 0.], [x + 1., 0., 0.], [x, 1., 0.]]
        })
        .collect();
    let faces: Vec<_> = (0..22).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&faces)
        .build()
        .unwrap();
    let gpu = GpuMesh::upload(&limited_device, &mesh).unwrap();
    let mut encoder = limited_device.create_command_encoder(&Default::default());
    assert!(matches!(
        EdgeQuery::new(&limited_device).encode(&limited_device, &mut encoder, &gpu),
        Err(GpuError::LimitsExceeded { .. })
    ));
    assert!(matches!(
        AdjacencyQuery::new(&limited_device).encode(&limited_device, &mut encoder, &gpu),
        Err(GpuError::LimitsExceeded { .. })
    ));
    // Face work now checks its own dispatch limit instead of relying on upload.
    let positions: Vec<_> = (0..65)
        .flat_map(|i| {
            let x = i as f32 * 3.;
            [[x, 0., 0.], [x + 1., 0., 0.], [x, 1., 0.]]
        })
        .collect();
    let faces: Vec<_> = (0..65).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&faces)
        .build()
        .unwrap();
    let gpu = GpuMesh::upload(&limited_device, &mesh).unwrap();
    assert!(matches!(
        FaceVertexQuery::new(&limited_device).encode(&limited_device, &mut encoder, &gpu),
        Err(GpuError::LimitsExceeded {
            resource: "face vertices"
        })
    ));
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_topology -- --ignored --nocapture"]
fn gpu_build_errors_match_the_cpu_builder() {
    let (device, queue) = gpu_device(Default::default());
    let mut builder = GpuMeshBuilder::new(&device);

    // Invalid mesh data is rejected with the CPU builder's error.
    for fixture in fixtures::invalid() {
        let expected = MeshBuilder::new()
            .positions(&fixture.positions)
            .triangles(&fixture.triangles)
            .build()
            .unwrap_err();
        let (positions, triangles) = input_buffers(&device, &fixture.positions, &fixture.triangles);
        let input = GpuMeshInput {
            positions: &positions,
            triangles: &triangles,
            vertex_count: fixture.positions.len() as u32,
            face_count: fixture.triangles.len() as u32,
        };
        assert_eq!(
            builder.build(&device, &queue, &input).err(),
            Some(GpuBuildError::Input(expected)),
            "{}",
            fixture.name
        );
    }

    // Patch size and element counts are checked before the buffers.
    let (positions, triangles) = fixtures::grid(3);
    let (position_buffer, triangle_buffer) = input_buffers(&device, &positions, &triangles);
    let input = GpuMeshInput {
        positions: &position_buffer,
        triangles: &triangle_buffer,
        vertex_count: positions.len() as u32,
        face_count: triangles.len() as u32,
    };
    builder = builder.patch_size(0);
    assert_eq!(
        builder.build(&device, &queue, &input).err(),
        Some(GpuBuildError::Input(BuildError::InvalidPatchSize))
    );
    builder = builder.patch_size(128);
    let too_many = GpuMeshInput {
        face_count: u32::MAX / 3 + 1,
        ..input
    };
    assert_eq!(
        builder.build(&device, &queue, &too_many).err(),
        Some(GpuBuildError::Input(BuildError::TooManyElements))
    );

    // Input buffers must hold every value and have STORAGE usage.
    let invalid = |resource| Some(GpuBuildError::Gpu(GpuError::InvalidBuffer { resource }));
    let short_positions = GpuMeshInput {
        vertex_count: positions.len() as u32 + 1,
        ..input
    };
    assert_eq!(
        builder.build(&device, &queue, &short_positions).err(),
        invalid("GPU build positions")
    );
    let short_triangles = GpuMeshInput {
        face_count: triangles.len() as u32 + 1,
        ..input
    };
    assert_eq!(
        builder.build(&device, &queue, &short_triangles).err(),
        invalid("GPU build triangles")
    );
    let not_storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test non-storage buffer"),
        size: position_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let wrong_usage = GpuMeshInput {
        positions: &not_storage,
        ..input
    };
    assert_eq!(
        builder.build(&device, &queue, &wrong_usage).err(),
        invalid("GPU build positions")
    );

    // Attributes of a GPU-built mesh need one value per vertex and STORAGE usage.
    let gpu = builder.build(&device, &queue, &input).unwrap();
    let attribute_error = GpuError::InvalidBuffer {
        resource: "vertex attribute",
    };
    let short = storage_only(&device, positions.len() - 1);
    assert_eq!(
        GpuVertexAttribute::<[f32; 4]>::from_buffer(&gpu, &short).err(),
        Some(attribute_error)
    );
    assert_eq!(
        GpuVertexAttribute::<[f32; 4]>::from_buffer(&gpu, &not_storage).err(),
        Some(attribute_error)
    );
    assert_eq!(
        GpuVertexAttribute::new(&device, &gpu, &vec![0u32; positions.len() - 1]).err(),
        Some(attribute_error)
    );
    let scalars = GpuVertexAttribute::new(&device, &gpu, &vec![7u32; positions.len()]).unwrap();
    assert_eq!(scalars.mesh_id(), gpu.mesh_id());
    assert_eq!(scalars.len() as usize, positions.len());

    // Two builds of the same data are different meshes.
    let other = builder.build(&device, &queue, &input).unwrap();
    assert_ne!(other.mesh_id(), gpu.mesh_id());
    let gpu_positions =
        GpuVertexAttribute::<[f32; 4]>::from_buffer(&gpu, &position_buffer).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(matches!(
        FaceCentroidQuery::new(&device).encode(&device, &mut encoder, &other, &gpu_positions),
        Err(GpuError::MeshMismatch)
    ));

    // Writing from the host needs COPY_DST usage on a wrapped buffer.
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap();
    let uploaded = GpuMesh::upload(&device, &mesh).unwrap();
    let mut wrapped = GpuVertexAttribute::<[f32; 4]>::from_buffer(
        &uploaded,
        &storage_only(&device, positions.len()),
    )
    .unwrap();
    assert_eq!(
        wrapped.write(&queue, &mesh.position_attribute()),
        Err(attribute_error)
    );
}

/// A `STORAGE`-only buffer that holds `count` `[f32; 4]` values.
fn storage_only(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test storage-only attribute"),
        size: 16 * count as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

/// Query pipelines shared by every check, created once per device.
struct Queries {
    face_vertices: FaceVertexQuery,
    vertex_faces: VertexFaceQuery,
    centroids: FaceCentroidQuery,
    edges: EdgeQuery,
    boundary: boundary::BoundaryOperator,
    adjacency: AdjacencyQuery,
    neighborhood: neighborhood::NeighborhoodOperator,
}

impl Queries {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            face_vertices: FaceVertexQuery::new(device),
            vertex_faces: VertexFaceQuery::new(device),
            centroids: FaceCentroidQuery::new(device),
            edges: EdgeQuery::new(device),
            boundary: boundary::BoundaryOperator::new(device),
            adjacency: AdjacencyQuery::new(device),
            neighborhood: neighborhood::NeighborhoodOperator::new(device),
        }
    }
}

/// The mesh data a GPU mesh was made from, and an Alum mesh holding the
/// positions the GPU attribute currently has.
struct Reference<'a> {
    mesh: &'a PolyMeshF32,
    triangles: &'a [[u32; 3]],
    vertex_count: usize,
}

/// A GPU mesh under test: its positions, and the edge ID of each
/// (smaller, larger) vertex pair. `label` names it in failure messages.
struct Subject<'a> {
    gpu: &'a GpuMesh,
    positions: &'a GpuVertexAttribute<[f32; 4]>,
    edge_ids: &'a BTreeMap<[u32; 2], u32>,
    label: &'a str,
}

fn alum_mesh(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> PolyMeshF32 {
    let mut reference = PolyMeshF32::new();
    for &[x, y, z] in positions {
        reference.add_vertex(Vec3(x, y, z)).unwrap();
    }
    for tri in triangles {
        reference.add_face(&tri.map(Into::into)).unwrap();
    }
    reference.check_topology().unwrap();
    reference
}

/// Sets every Alum vertex to its input position.
fn reset_points(reference: &mut PolyMeshF32, positions: &[[f32; 3]]) {
    for (v, &[x, y, z]) in positions.iter().enumerate() {
        reference
            .set_point((v as u32).into(), Vec3(x, y, z))
            .unwrap();
    }
}

/// The documented edge numbering: edges in ascending (smaller, larger)
/// vertex-pair order.
fn ascending_edge_ids(triangles: &[[u32; 3]]) -> BTreeMap<[u32; 2], u32> {
    let pairs: BTreeSet<[u32; 2]> = triangles
        .iter()
        .flat_map(|t| {
            (0..3).map(move |i| {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                [a.min(b), a.max(b)]
            })
        })
        .collect();
    pairs.into_iter().zip(0..).collect()
}

/// Uploads positions and triangles in the vec4 layout [`GpuMeshBuilder`]
/// reads, each buffer at least 16 bytes.
fn input_buffers(
    device: &wgpu::Device,
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> (wgpu::Buffer, wgpu::Buffer) {
    let mut positions: Vec<[f32; 4]> = positions.iter().map(|&[x, y, z]| [x, y, z, 0.]).collect();
    if positions.is_empty() {
        positions.push([0.; 4]);
    }
    let mut triangles: Vec<[u32; 4]> = triangles.iter().map(|&[a, b, c]| [a, b, c, 0]).collect();
    if triangles.is_empty() {
        triangles.push([0; 4]);
    }
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

/// Uploads CPU-built meshes at every patch size and checks every query,
/// plus the host attribute API.
fn check_uploaded(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &Queries,
    reference: &mut PolyMeshF32,
    fixture: &fixtures::Fixture,
) {
    let (positions, triangles) = (&fixture.positions[..], &fixture.triangles[..]);
    for patch_size in fixtures::PATCH_SIZES {
        let mesh = MeshBuilder::new()
            .positions(positions)
            .triangles(triangles)
            .patch_size(patch_size)
            .build()
            .unwrap();
        let gpu = GpuMesh::upload(device, &mesh).unwrap();
        assert_eq!(gpu.mesh_id(), mesh.id());
        let other_mesh = MeshBuilder::new()
            .positions(positions)
            .triangles(triangles)
            .build()
            .unwrap();
        assert_ne!(other_mesh.id(), mesh.id());
        let wrong_attribute = other_mesh.position_attribute();
        assert!(matches!(
            gpu.upload_attribute(device, &wrong_attribute),
            Err(GpuError::MeshMismatch)
        ));
        let mut host_positions = mesh.position_attribute();
        let mut gpu_positions = gpu.upload_attribute(device, &host_positions).unwrap();
        assert_eq!(gpu_positions.len() as usize, positions.len());
        assert_eq!(gpu_positions.is_empty(), positions.is_empty());
        assert_eq!(gpu_positions.mesh_id(), mesh.id());
        assert_eq!(
            gpu_positions.write(queue, &wrong_attribute),
            Err(GpuError::MeshMismatch)
        );
        let other_gpu = GpuMesh::upload(device, &other_mesh).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(matches!(
            queries
                .centroids
                .encode(device, &mut encoder, &other_gpu, &gpu_positions),
            Err(GpuError::MeshMismatch)
        ));
        // Both scalar and vector storage use original input vertex IDs.
        let scalar = gpu
            .upload_attribute(device, &VertexAttribute::new(&mesh, 42u32))
            .unwrap();
        let encoder = device.create_command_encoder(&Default::default());
        let values = read_words(device, queue, encoder, scalar.buffer());
        assert_eq!(&values[..positions.len()], vec![42u32; positions.len()]);
        // Per-vertex values keep input vertex order on the GPU.
        let ramp = VertexAttribute::from_values(
            &mesh,
            (0..positions.len() as u32).map(|v| 3 * v + 1).collect(),
        )
        .unwrap();
        let ramp = gpu.upload_attribute(device, &ramp).unwrap();
        let encoder = device.create_command_encoder(&Default::default());
        let values = read_words(device, queue, encoder, ramp.buffer());
        let expected_ramp: Vec<_> = (0..positions.len() as u32).map(|v| 3 * v + 1).collect();
        assert_eq!(&values[..positions.len()], expected_ramp);
        for scale in [1., 2.] {
            if scale != 1. {
                // Write through every patch, including ribbon handles. The same
                // global vertex must still address one canonical value.
                for p in mesh.patches() {
                    for (local, &global) in p.vertex_ids().iter().enumerate() {
                        let [x, y, z] = positions[global as usize];
                        *host_positions
                            .get_mut(
                                &mesh,
                                VertexHandle {
                                    patch: p.id(),
                                    local: local as u32,
                                },
                            )
                            .unwrap() = [scale * x + 1., scale * y - 2., scale * z + 3., 99.];
                    }
                }
                gpu_positions.write(queue, &host_positions).unwrap();
            }
            for (v, &[x, y, z]) in positions.iter().enumerate() {
                let pos = if scale == 1. {
                    Vec3(x, y, z)
                } else {
                    Vec3(scale * x + 1., scale * y - 2., scale * z + 3.)
                };
                reference.set_point((v as u32).into(), pos).unwrap();
            }
            check_centroids(device, queue, queries, reference, &gpu, &gpu_positions);
        }
        // Every patch, ribbons included, gives an edge the same global ID, and
        // the IDs follow ascending vertex-pair order.
        let mut edge_ids = BTreeMap::new();
        for p in mesh.patches() {
            for (local, &global) in p.edge_ids().iter().enumerate() {
                let key = p.edge_vertices()[local].map(|v| p.vertex_ids()[v as usize]);
                assert_eq!(*edge_ids.entry(key).or_insert(global), global);
            }
        }
        assert_eq!(edge_ids, ascending_edge_ids(triangles));
        let label = format!("uploaded, patch size {patch_size}");
        check_queries(
            device,
            queue,
            queries,
            &Reference {
                mesh: reference,
                triangles,
                vertex_count: positions.len(),
            },
            &Subject {
                gpu: &gpu,
                positions: &gpu_positions,
                edge_ids: &edge_ids,
                label: &label,
            },
        );
    }
}

/// Builds the mesh on the GPU with `builder`'s patch size and checks every
/// query, using the positions buffer the build read as the positions attribute.
fn check_gpu_built(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &Queries,
    builder: &GpuMeshBuilder,
    reference: &mut PolyMeshF32,
    fixture: &fixtures::Fixture,
) {
    let (positions, triangles) = (&fixture.positions[..], &fixture.triangles[..]);
    reset_points(reference, positions);
    let (position_buffer, triangle_buffer) = input_buffers(device, positions, triangles);
    let input = GpuMeshInput {
        positions: &position_buffer,
        triangles: &triangle_buffer,
        vertex_count: positions.len() as u32,
        face_count: triangles.len() as u32,
    };
    let gpu = builder.build(device, queue, &input).unwrap();
    assert_eq!(gpu.vertex_count() as usize, positions.len());

    // A repeat build is a new mesh with the same connectivity, word for word.
    let again = builder.build(device, queue, &input).unwrap();
    assert_ne!(again.mesh_id(), gpu.mesh_id());
    let read = |buffer| {
        let encoder = device.create_command_encoder(&Default::default());
        read_words(device, queue, encoder, buffer)
    };
    assert_eq!(read(gpu.arena()), read(again.arena()));

    // The build's own positions buffer serves as the positions attribute.
    let gpu_positions =
        GpuVertexAttribute::<[f32; 4]>::from_buffer(&gpu, &position_buffer).unwrap();
    assert_eq!(gpu_positions.mesh_id(), gpu.mesh_id());
    assert_eq!(gpu_positions.len() as usize, positions.len());
    check_centroids(device, queue, queries, reference, &gpu, &gpu_positions);
    let label = format!("GPU-built, {}", fixture.name);
    check_queries(
        device,
        queue,
        queries,
        &Reference {
            mesh: reference,
            triangles,
            vertex_count: positions.len(),
        },
        &Subject {
            gpu: &gpu,
            positions: &gpu_positions,
            edge_ids: &ascending_edge_ids(triangles),
            label: &label,
        },
    );
}

/// Checks face centroids against Alum's at its current positions.
fn check_centroids(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &Queries,
    reference: &PolyMeshF32,
    gpu: &GpuMesh,
    positions: &GpuVertexAttribute<[f32; 4]>,
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let output = queries
        .centroids
        .encode(device, &mut encoder, gpu, positions)
        .unwrap();
    assert_eq!(output.len() as usize, reference.faces().count());
    assert_eq!(output.mesh_id(), gpu.mesh_id());
    let words = read_words(device, queue, encoder, output.buffer());
    if gpu.vertex_count() == 0 {
        assert_eq!(words, [0; 4]);
    }
    for face in reference.faces() {
        let Vec3(x, y, z) = reference.try_calc_face_centroid(face).unwrap();
        for (i, expected) in [x, y, z, 0.].into_iter().enumerate() {
            let actual = f32::from_bits(words[4 * face.index() as usize + i]);
            assert!(
                (actual - expected).abs() <= 2e-5 * expected.abs().max(1.),
                "centroid {face:?}: {actual} vs {expected}"
            );
        }
    }
}

/// Checks the edge, boundary, adjacency, face-vertex and vertex-face queries
/// on `subject` against Alum.
fn check_queries(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &Queries,
    reference: &Reference,
    subject: &Subject,
) {
    let (alum, triangles) = (reference.mesh, reference.triangles);
    let (gpu, label) = (subject.gpu, subject.label);
    let edge_count = subject.edge_ids.len();
    let reference_edges: BTreeMap<_, _> = alum
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(alum);
            ([a.index().min(b.index()), a.index().max(b.index())], e)
        })
        .collect();
    let expected: Vec<Vec<u32>> = alum
        .faces()
        .map(|f| alum.fv_ccw_iter(f).map(|v| v.index()).collect())
        .collect();
    let expected_inverse: Vec<Vec<u32>> = alum
        .vertices()
        .map(|v| {
            let mut faces: Vec<_> = alum.vf_ccw_iter(v).map(|f| f.index()).collect();
            faces.sort_unstable();
            faces
        })
        .collect();

    // Edge records and the boundary-length example operator.
    let mut encoder = device.create_command_encoder(&Default::default());
    let edge_records = queries.edges.encode(device, &mut encoder, gpu).unwrap();
    assert_eq!(edge_records.mesh_id(), gpu.mesh_id());
    assert_eq!(edge_records.edge_count() as usize, reference_edges.len());
    assert_eq!(edge_records.edge_count(), gpu.edge_count());
    let weights: Vec<_> = (0..edge_count).map(|i| (i % 5) as f32 * 0.25).collect();
    let boundary_output = queries.boundary.encode(
        device,
        &mut encoder,
        &edge_records,
        subject.positions,
        &weights,
    );
    let records = read_words(device, queue, encoder, edge_records.buffer());
    let encoder = device.create_command_encoder(&Default::default());
    let boundary_words = read_words(device, queue, encoder, &boundary_output);
    if edge_count == 0 {
        assert_eq!(records, [0; 4]);
        assert_eq!(boundary_words, [0; 2]);
    }
    for (&endpoints, &edge) in subject.edge_ids {
        let edge = edge as usize;
        let actual = &records[edge * 4..edge * 4 + 4];
        assert_eq!(actual[..2], endpoints, "edge {edge}, {label}");
        let reference_edge = reference_edges[&endpoints];
        let (a, b) = reference_edge.halfedges();
        let mut expected_faces: Vec<_> = [a.face(alum), b.face(alum)]
            .into_iter()
            .flatten()
            .map(|f| f.index())
            .collect();
        expected_faces.sort_unstable();
        let mut actual_faces: Vec<_> = actual[2..]
            .iter()
            .copied()
            .filter(|&f| f != INVALID_ID)
            .collect();
        actual_faces.sort_unstable();
        assert_eq!(actual_faces, expected_faces, "edge {edge}, {label}");
        let expected_length = if reference_edge.is_boundary(alum) {
            alum.try_calc_edge_length(reference_edge).unwrap() * weights[edge]
        } else {
            0.
        };
        let actual_length = f32::from_bits(boundary_words[edge * 2]);
        assert!((actual_length - expected_length).abs() <= 2e-5 * expected_length.abs().max(1.));
        assert_eq!(
            f32::from_bits(boundary_words[edge * 2 + 1]),
            expected_faces.len() as f32
        );
    }
    check_remaining_adjacency(device, queue, queries, reference, subject);

    // Face vertices, in input order up to rotation.
    assert_eq!(gpu.face_count() as usize, expected.len());
    let mut encoder = device.create_command_encoder(&Default::default());
    let output = queries
        .face_vertices
        .encode(device, &mut encoder, gpu)
        .unwrap();
    assert_eq!(output.face_count() as usize, triangles.len());
    assert_eq!(output.mesh_id(), gpu.mesh_id());
    let words = read_words(device, queue, encoder, output.buffer());
    for (face, expected) in expected.iter().enumerate() {
        let actual = &words[3 * face..3 * face + 3];
        assert!(
            (0..3).any(|rotation| (0..3).all(|i| actual[i] == expected[(i + rotation) % 3])),
            "face {face}, {label}: {actual:?} vs {expected:?}"
        );
    }

    // Vertex faces. Reuse the same pipeline and mesh, including across submissions.
    let vertex_count = reference.vertex_count;
    for _ in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let adjacency = queries
            .vertex_faces
            .encode(device, &mut encoder, gpu)
            .unwrap();
        assert_eq!(adjacency.mesh_id(), gpu.mesh_id());
        assert_eq!(adjacency.vertex_count() as usize, vertex_count);
        assert_eq!(adjacency.entry_count() as usize, 3 * triangles.len());
        assert_eq!(
            adjacency.entries_byte_offset(),
            4 * (vertex_count as u64 + 1)
        );
        let words = read_words(device, queue, encoder, adjacency.buffer());
        let (offsets, faces) = words.split_at(vertex_count + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap() as usize, faces.len());
        for (vertex, expected) in expected_inverse.iter().enumerate() {
            assert!(offsets[vertex] <= offsets[vertex + 1]);
            let mut actual = faces[offsets[vertex] as usize..offsets[vertex + 1] as usize].to_vec();
            actual.sort_unstable();
            assert_eq!(&actual, expected, "vertex {vertex}, {label}");
        }
    }
}

fn check_remaining_adjacency(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    queries: &Queries,
    reference: &Reference,
    subject: &Subject,
) {
    let (alum, triangles, vertex_count) =
        (reference.mesh, reference.triangles, reference.vertex_count);
    let (gpu, edge_ids) = (subject.gpu, subject.edge_ids);
    let edge_count = edge_ids.len();
    let reference_edges: BTreeMap<_, _> = alum
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(alum);
            ([a.index().min(b.index()), a.index().max(b.index())], e)
        })
        .collect();
    let weights: Vec<_> = (0..vertex_count)
        .map(|v| (v % 7) as f32 * 0.5 - 1.5)
        .collect();
    let gpu_weights = GpuVertexAttribute::new(device, gpu, &weights).unwrap();
    for _ in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let adjacency = queries.adjacency.encode(device, &mut encoder, gpu).unwrap();
        assert_eq!(adjacency.mesh_id(), gpu.mesh_id());
        assert_eq!(adjacency.vertex_count() as usize, vertex_count);
        assert_eq!(adjacency.edge_count() as usize, edge_count);
        assert_eq!(adjacency.face_count() as usize, triangles.len());
        assert_eq!(adjacency.vertex_entry_count() as usize, 2 * edge_count);
        assert_eq!(
            adjacency.vertex_entries_byte_offset(),
            (vertex_count as u64 + 1) * 4
        );
        let summary = queries.neighborhood.encode(
            device,
            &mut encoder,
            &adjacency,
            subject.positions,
            &gpu_weights,
        );
        let vertices = read_words(device, queue, encoder, adjacency.vertex_neighbors());
        let read = |buffer| {
            read_words(
                device,
                queue,
                device.create_command_encoder(&Default::default()),
                buffer,
            )
        };
        let faces = read(adjacency.face_neighbors());
        let diamonds = read(adjacency.edge_diamonds());
        let edges = read(adjacency.edge_connectivity().buffer());
        let summaries = read(&summary);
        let (offsets, entries) = vertices.split_at(vertex_count + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap() as usize, 2 * edge_count);
        assert_eq!(entries.len(), 4 * edge_count);
        if triangles.is_empty() {
            assert_eq!(faces, [0]);
            assert_eq!(diamonds, [0; 4]);
            assert_eq!(edges, [0; 4]);
            assert_eq!(summaries, [0; 4]);
        }
        for vertex in alum.vertices() {
            let v = vertex.index() as usize;
            assert!(offsets[v] <= offsets[v + 1]);
            let mut actual: Vec<_> = entries[2 * offsets[v] as usize..2 * offsets[v + 1] as usize]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0], p[1]))
                .collect();
            actual.sort_unstable();
            let mut expected: Vec<_> = alum
                .ve_ccw_iter(vertex)
                .map(|e| {
                    let (a, b) = e.vertices(alum);
                    let key = [a.index().min(b.index()), a.index().max(b.index())];
                    (
                        edge_ids[&key],
                        if a == vertex { b.index() } else { a.index() },
                    )
                })
                .collect();
            expected.sort_unstable();
            assert_eq!(actual, expected, "VE/VV at vertex {v}");
            let mut neighbors: Vec<_> = actual.iter().map(|p| p.1).collect();
            neighbors.sort_unstable();
            let mut expected_neighbors: Vec<_> =
                alum.vv_ccw_iter(vertex).map(|v| v.index()).collect();
            expected_neighbors.sort_unstable();
            assert_eq!(neighbors, expected_neighbors);
            let degree = expected.len();
            let boundary = alum
                .ve_ccw_iter(vertex)
                .filter(|e| e.is_boundary(alum))
                .count();
            let mean = alum
                .ve_ccw_iter(vertex)
                .map(|e| alum.try_calc_edge_length(e).unwrap())
                .sum::<f32>()
                / degree as f32;
            for (lane, expected) in [degree as f32, boundary as f32, mean, weights[v] * mean]
                .into_iter()
                .enumerate()
            {
                let actual = f32::from_bits(summaries[4 * v + lane]);
                assert!(
                    (actual - expected).abs() <= 2e-5 * expected.abs().max(1.),
                    "neighborhood vertex {v}, lane {lane}: {actual} vs {expected}"
                );
            }
        }
        for (face, triangle) in triangles.iter().enumerate() {
            for side in 0..3 {
                let a = triangle[side];
                let b = triangle[(side + 1) % 3];
                let edge = reference_edges[&[a.min(b), a.max(b)]];
                let (a, b) = edge.halfedges();
                let expected = [a.face(alum), b.face(alum)]
                    .into_iter()
                    .flatten()
                    .map(|f| f.index())
                    .find(|&f| f != face as u32)
                    .unwrap_or(u32::MAX);
                assert_eq!(
                    faces[3 * face + side],
                    expected,
                    "FF at face {face}, slot {side}"
                );
            }
        }
        for (&endpoints, &global) in edge_ids {
            let i = global as usize * 4;
            assert_eq!(&diamonds[i..i + 2], &endpoints);
            assert_eq!(&edges[i..i + 2], &endpoints);
            let reference_edge = reference_edges[&endpoints];
            let (a, b) = reference_edge.halfedges();
            let mut expected_faces: Vec<_> = [a.face(alum), b.face(alum)]
                .into_iter()
                .flatten()
                .map(|f| f.index())
                .collect();
            expected_faces.sort_unstable();
            let mut actual_faces: Vec<_> = edges[i + 2..i + 4]
                .iter()
                .copied()
                .filter(|&f| f != u32::MAX)
                .collect();
            actual_faces.sort_unstable();
            assert_eq!(actual_faces, expected_faces);
            for side in 0..2 {
                let face = edges[i + 2 + side];
                let expected = if face == u32::MAX {
                    u32::MAX
                } else {
                    alum.fv_ccw_iter(face.into())
                        .map(|v| v.index())
                        .find(|v| !endpoints.contains(v))
                        .unwrap()
                };
                assert_eq!(
                    diamonds[i + 2 + side],
                    expected,
                    "diamond edge {global}, side {side}"
                );
            }
        }
    }
}

fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    output: &wgpu::Buffer,
) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(output, 0, &readback, 0, output.size());
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
    let words = {
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        words.to_vec()
    };
    readback.unmap();
    words
}
