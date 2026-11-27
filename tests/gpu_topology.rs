use alum::{Handle, HasIterators, HasTopology, PolyMeshF32, Vec3};
use mopmesh::{
    MeshBuilder, VertexHandle,
    gpu::{
        AdjacencyQuery, EdgeQuery, FaceCentroidQuery, FaceVertexQuery, GpuError, GpuMesh,
        INVALID_ID, VertexFaceQuery,
    },
};
#[path = "../examples/support/boundary.rs"]
mod boundary;
#[path = "support/fixtures.rs"]
mod fixtures;
#[path = "../examples/support/neighborhood.rs"]
mod neighborhood;

// Explicit opt-in: a missing adapter is a failure, never a silent successful test.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_topology -- --ignored --nocapture"]
fn gpu_topology_queries_match_alum() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    eprintln!("GPU: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let query = FaceVertexQuery::new(&device);
    for fixture in fixtures::valid() {
        eprintln!("checking {}", fixture.name);
        check(
            &device,
            &queue,
            &query,
            &fixture.positions,
            &fixture.triangles,
        );
    }
    // Reject oversized uploads before creating a GPU buffer.
    let (positions, triangles) = fixtures::grid(9);
    let (small_device, _) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_storage_buffer_binding_size: 64,
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
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
    // Face work can fit while edge work exceeds the dispatch limit.
    let (limited_device, _limited_queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_compute_workgroups_per_dimension: 1,
                ..Default::default()
            },
            ..Default::default()
        }))
        .unwrap();
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

fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    query: &FaceVertexQuery,
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
) {
    let inverse = VertexFaceQuery::new(device);
    let centroids = FaceCentroidQuery::new(device);
    let edge_query = EdgeQuery::new(device);
    let boundary_operator = boundary::BoundaryOperator::new(device);
    let adjacency_query = AdjacencyQuery::new(device);
    let neighborhood_operator = neighborhood::NeighborhoodOperator::new(device);
    let mut reference = PolyMeshF32::new();
    for &[x, y, z] in positions {
        reference.add_vertex(Vec3(x, y, z)).unwrap();
    }
    for tri in triangles {
        reference.add_face(&tri.map(Into::into)).unwrap();
    }
    reference.check_topology().unwrap();
    let reference_edges: std::collections::BTreeMap<_, _> = reference
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(&reference);
            ([a.index().min(b.index()), a.index().max(b.index())], e)
        })
        .collect();
    let expected: Vec<Vec<u32>> = reference
        .faces()
        .map(|f| reference.fv_ccw_iter(f).map(|v| v.index()).collect())
        .collect();
    let expected_inverse: Vec<Vec<u32>> = reference
        .vertices()
        .map(|v| {
            let mut faces: Vec<_> = reference.vf_ccw_iter(v).map(|f| f.index()).collect();
            faces.sort_unstable();
            faces
        })
        .collect();
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
            gpu.upload_vertex_attribute(device, &wrong_attribute),
            Err(GpuError::MeshMismatch)
        ));
        let mut host_positions = mesh.position_attribute();
        let mut gpu_positions = gpu
            .upload_vertex_attribute(device, &host_positions)
            .unwrap();
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
            centroids.encode(device, &mut encoder, &other_gpu, &gpu_positions),
            Err(GpuError::MeshMismatch)
        ));
        // Both scalar and vector storage use original input vertex IDs.
        let scalar = gpu
            .upload_vertex_attribute(device, &mesh.vertex_attribute(42u32))
            .unwrap();
        let encoder = device.create_command_encoder(&Default::default());
        let values = read_words(device, queue, encoder, scalar.buffer());
        assert_eq!(&values[..positions.len()], vec![42u32; positions.len()]);
        // Per-vertex values keep input vertex order on the GPU.
        let ramp = mesh
            .vertex_attribute_from((0..positions.len() as u32).map(|v| 3 * v + 1).collect())
            .unwrap();
        let ramp = gpu.upload_vertex_attribute(device, &ramp).unwrap();
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
            let mut encoder = device.create_command_encoder(&Default::default());
            let output = centroids
                .encode(device, &mut encoder, &gpu, &gpu_positions)
                .unwrap();
            assert_eq!(output.face_count() as usize, triangles.len());
            assert_eq!(output.mesh_id(), mesh.id());
            let words = read_words(device, queue, encoder, output.buffer());
            if positions.is_empty() {
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
        let mut canonical_endpoints = vec![[0; 2]; mesh.edge_count()];
        for p in mesh.patches() {
            for (local, &global) in p.edge_ids().iter().enumerate() {
                if p.edge_owners()[local].patch == p.id() {
                    canonical_endpoints[global as usize] =
                        p.edge_vertices()[local].map(|v| p.vertex_ids()[v as usize]);
                }
            }
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        let edge_records = edge_query.encode(device, &mut encoder, &gpu).unwrap();
        assert_eq!(edge_records.mesh_id(), mesh.id());
        assert_eq!(edge_records.edge_count() as usize, reference_edges.len());
        assert_eq!(edge_records.edge_count(), gpu.edge_count());
        let weights: Vec<_> = (0..mesh.edge_count())
            .map(|i| (i % 5) as f32 * 0.25)
            .collect();
        let boundary_output = boundary_operator.encode(
            device,
            &mut encoder,
            &edge_records,
            &gpu_positions,
            &weights,
        );
        let records = read_words(device, queue, encoder, edge_records.buffer());
        let encoder = device.create_command_encoder(&Default::default());
        let boundary_words = read_words(device, queue, encoder, &boundary_output);
        if mesh.edge_count() == 0 {
            assert_eq!(records, [0; 4]);
            assert_eq!(boundary_words, [0; 2]);
        }
        for (edge, &endpoints) in canonical_endpoints.iter().enumerate() {
            let actual = &records[edge * 4..edge * 4 + 4];
            assert_eq!(
                actual[..2],
                endpoints,
                "edge {edge}, patch size {patch_size}"
            );
            let reference_edge = reference_edges[&endpoints];
            let (a, b) = reference_edge.halfedges();
            let mut expected_faces: Vec<_> = [a.face(&reference), b.face(&reference)]
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
            assert_eq!(
                actual_faces, expected_faces,
                "edge {edge}, patch size {patch_size}"
            );
            let expected_length = if reference_edge.is_boundary(&reference) {
                reference.try_calc_edge_length(reference_edge).unwrap() * weights[edge]
            } else {
                0.
            };
            let actual_length = f32::from_bits(boundary_words[edge * 2]);
            assert!(
                (actual_length - expected_length).abs() <= 2e-5 * expected_length.abs().max(1.)
            );
            assert_eq!(
                f32::from_bits(boundary_words[edge * 2 + 1]),
                expected_faces.len() as f32
            );
        }
        check_remaining_adjacency(
            device,
            queue,
            &reference,
            (&mesh, triangles),
            &gpu,
            &gpu_positions,
            (&adjacency_query, &neighborhood_operator),
        );
        assert_eq!(gpu.face_count() as usize, expected.len());
        let mut encoder = device.create_command_encoder(&Default::default());
        let output = query.encode(device, &mut encoder, &gpu).unwrap();
        assert_eq!(output.face_count() as usize, triangles.len());
        assert_eq!(output.mesh_id(), mesh.id());
        let words = read_words(device, queue, encoder, output.buffer());
        for (face, expected) in expected.iter().enumerate() {
            let actual = &words[3 * face..3 * face + 3];
            assert!(
                (0..3).any(|rotation| (0..3).all(|i| actual[i] == expected[(i + rotation) % 3])),
                "face {face}, patch size {patch_size}: {actual:?} vs {expected:?}"
            );
        }
        // Reuse the same pipeline and mesh, including across submissions.
        for _ in 0..2 {
            let mut encoder = device.create_command_encoder(&Default::default());
            let adjacency = inverse.encode(device, &mut encoder, &gpu).unwrap();
            assert_eq!(adjacency.mesh_id(), mesh.id());
            assert_eq!(adjacency.vertex_count() as usize, positions.len());
            assert_eq!(adjacency.entry_count() as usize, 3 * triangles.len());
            assert_eq!(
                adjacency.entries_byte_offset(),
                4 * (positions.len() as u64 + 1)
            );
            let words = read_words(device, queue, encoder, adjacency.buffer());
            let (offsets, faces) = words.split_at(positions.len() + 1);
            assert_eq!(offsets[0], 0);
            assert_eq!(*offsets.last().unwrap() as usize, faces.len());
            for (vertex, expected) in expected_inverse.iter().enumerate() {
                assert!(offsets[vertex] <= offsets[vertex + 1]);
                let mut actual =
                    faces[offsets[vertex] as usize..offsets[vertex + 1] as usize].to_vec();
                actual.sort_unstable();
                assert_eq!(
                    &actual, expected,
                    "vertex {vertex}, patch size {patch_size}"
                );
            }
        }
    }
}

fn check_remaining_adjacency(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    reference: &PolyMeshF32,
    mesh_data: (&mopmesh::Mesh, &[[u32; 3]]),
    gpu: &GpuMesh,
    positions: &mopmesh::gpu::GpuVertexAttribute<[f32; 4]>,
    queries: (&AdjacencyQuery, &neighborhood::NeighborhoodOperator),
) {
    let (mesh, triangles) = mesh_data;
    let (query, operator) = queries;
    let mut edge_ids = std::collections::BTreeMap::new();
    for p in mesh.patches() {
        for (local, &global) in p.edge_ids().iter().enumerate() {
            edge_ids.insert(
                p.edge_vertices()[local].map(|v| p.vertex_ids()[v as usize]),
                global,
            );
        }
    }
    let reference_edges: std::collections::BTreeMap<_, _> = reference
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(reference);
            ([a.index().min(b.index()), a.index().max(b.index())], e)
        })
        .collect();
    let weights: Vec<_> = (0..mesh.positions().len())
        .map(|v| (v % 7) as f32 * 0.5 - 1.5)
        .collect();
    let gpu_weights = gpu
        .upload_vertex_attribute(
            device,
            &mesh.vertex_attribute_from(weights.clone()).unwrap(),
        )
        .unwrap();
    for _ in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let adjacency = query.encode(device, &mut encoder, gpu).unwrap();
        assert_eq!(adjacency.mesh_id(), mesh.id());
        assert_eq!(adjacency.vertex_count() as usize, mesh.positions().len());
        assert_eq!(adjacency.edge_count() as usize, mesh.edge_count());
        assert_eq!(adjacency.face_count() as usize, triangles.len());
        assert_eq!(
            adjacency.vertex_entry_count() as usize,
            2 * mesh.edge_count()
        );
        assert_eq!(
            adjacency.vertex_entries_byte_offset(),
            (mesh.positions().len() as u64 + 1) * 4
        );
        let summary = operator.encode(device, &mut encoder, &adjacency, positions, &gpu_weights);
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
        let (offsets, entries) = vertices.split_at(mesh.positions().len() + 1);
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap() as usize, 2 * mesh.edge_count());
        assert_eq!(entries.len(), 4 * mesh.edge_count());
        if triangles.is_empty() {
            assert_eq!(faces, [0]);
            assert_eq!(diamonds, [0; 4]);
            assert_eq!(edges, [0; 4]);
            assert_eq!(summaries, [0; 4]);
        }
        for vertex in reference.vertices() {
            let v = vertex.index() as usize;
            assert!(offsets[v] <= offsets[v + 1]);
            let mut actual: Vec<_> = entries[2 * offsets[v] as usize..2 * offsets[v + 1] as usize]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0], p[1]))
                .collect();
            actual.sort_unstable();
            let mut expected: Vec<_> = reference
                .ve_ccw_iter(vertex)
                .map(|e| {
                    let (a, b) = e.vertices(reference);
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
                reference.vv_ccw_iter(vertex).map(|v| v.index()).collect();
            expected_neighbors.sort_unstable();
            assert_eq!(neighbors, expected_neighbors);
            let degree = expected.len();
            let boundary = reference
                .ve_ccw_iter(vertex)
                .filter(|e| e.is_boundary(reference))
                .count();
            let mean = reference
                .ve_ccw_iter(vertex)
                .map(|e| reference.try_calc_edge_length(e).unwrap())
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
                let expected = [a.face(reference), b.face(reference)]
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
        for (&endpoints, &global) in &edge_ids {
            let i = global as usize * 4;
            assert_eq!(&diamonds[i..i + 2], &endpoints);
            assert_eq!(&edges[i..i + 2], &endpoints);
            let reference_edge = reference_edges[&endpoints];
            let (a, b) = reference_edge.halfedges();
            let mut expected_faces: Vec<_> = [a.face(reference), b.face(reference)]
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
                    reference
                        .fv_ccw_iter(face.into())
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
