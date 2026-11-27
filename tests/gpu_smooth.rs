//! Uniform Laplacian smoothing checked against an f64 host reference, with
//! and without fixed boundary vertices, plus the downstream bending operator
//! from the smoothing example.
use alum::{Handle, HasTopology, PolyMeshF32, Vec3};
use mopmesh::{
    MeshBuilder, VertexAttribute,
    gpu::{
        AdjacencyQuery, BoundaryQuery, GpuError, GpuMesh, GpuVertexAttribute, SmoothQuery,
        Smoothing, VertexNormalQuery,
    },
};
use std::collections::BTreeSet;

#[path = "../examples/support/bending.rs"]
mod bending;
#[path = "support/fixtures.rs"]
mod fixtures;

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_smooth -- --ignored --nocapture"]
fn downstream_bending_operator_matches_reference() {
    let (device, queue) = gpu_device();
    let (grid_positions, triangles) = fixtures::grid(9);
    let positions: Vec<[f32; 4]> = grid_positions
        .iter()
        .map(|&[x, y, _]| [x, y, 0.5 * (0.9 * x).sin() * (0.7 * y).cos(), 0.])
        .collect();
    let mesh = MeshBuilder::new()
        .positions(&grid_positions)
        .triangles(&triangles)
        .patch_size(3)
        .build()
        .unwrap();
    let gpu = GpuMesh::upload(&device, &mesh).unwrap();
    let gpu_positions = gpu
        .upload_attribute(
            &device,
            &VertexAttribute::from_values(&mesh, positions.clone()).unwrap(),
        )
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let adjacency = AdjacencyQuery::new(&device)
        .encode(&device, &mut encoder, &gpu)
        .unwrap();
    let normals = VertexNormalQuery::new(&device)
        .encode(&device, &mut encoder, &gpu, &gpu_positions)
        .unwrap();
    let result = bending::BendingOperator::new(&device).encode(
        &device,
        &mut encoder,
        &gpu,
        &adjacency,
        &gpu_positions,
        &normals,
    );
    queue.submit([encoder.finish()]);
    assert_eq!(result.mesh_id(), gpu.mesh_id());
    let normals = read_positions(&device, &queue, &normals);
    let actual = read_words(&device, &queue, result.buffer());

    // Recompute from the GPU normals in f64: mean |cos| between the normal
    // and each edge direction.
    let neighbors = neighbor_sets(positions.len(), &triangles);
    let mut checked_bend = false;
    for v in 0..positions.len() {
        let p = positions[v].map(f64::from);
        let n = normals[v].map(f64::from);
        let total: f64 = neighbors[v]
            .iter()
            .map(|&u| {
                let d: Vec<f64> = (0..3).map(|i| f64::from(positions[u][i]) - p[i]).collect();
                let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                ((n[0] * d[0] + n[1] * d[1] + n[2] * d[2]) / length).abs()
            })
            .sum();
        let expected = total / neighbors[v].len() as f64;
        let actual = f64::from(f32::from_bits(actual[v]));
        assert!(
            (actual - expected).abs() <= 1e-5,
            "bending {v}: {actual} vs {expected}"
        );
        checked_bend |= expected > 0.01;
    }
    assert!(checked_bend, "the fixture should bend somewhere");
}

/// Each vertex's edge neighbors.
fn neighbor_sets(vertices: usize, triangles: &[[u32; 3]]) -> Vec<BTreeSet<usize>> {
    let mut neighbors = vec![BTreeSet::new(); vertices];
    for t in triangles {
        for i in 0..3 {
            let (a, b) = (t[i] as usize, t[(i + 1) % 3] as usize);
            neighbors[a].insert(b);
            neighbors[b].insert(a);
        }
    }
    neighbors
}

// Explicit opt-in: a missing adapter is a failure, never a silent successful test.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_smooth -- --ignored --nocapture"]
fn gpu_smoothing_matches_reference() {
    let (device, queue) = gpu_device();
    let adjacency_query = AdjacencyQuery::new(&device);
    let boundary_query = BoundaryQuery::new(&device);
    let smooth = SmoothQuery::new(&device);
    for fixture in fixtures::valid() {
        eprintln!("checking {}", fixture.name);
        // Lift the fixture off the z = 0 plane and give each w lane a marker.
        let positions: Vec<[f32; 4]> = fixture
            .positions
            .iter()
            .enumerate()
            .map(|(v, &[x, y, z])| [x, y, z + 0.5 * (0.7 * x).sin() * (1.3 * y).cos(), v as f32])
            .collect();
        let alum_boundary = boundary_vertices(&fixture.positions, &fixture.triangles);
        for patch_size in fixtures::PATCH_SIZES {
            let mesh = MeshBuilder::new()
                .positions(&fixture.positions)
                .triangles(&fixture.triangles)
                .patch_size(patch_size)
                .build()
                .unwrap();
            let gpu = GpuMesh::upload(&device, &mesh).unwrap();
            let gpu_positions = gpu
                .upload_attribute(
                    &device,
                    &VertexAttribute::from_values(&mesh, positions.clone()).unwrap(),
                )
                .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let adjacency = adjacency_query.encode(&device, &mut encoder, &gpu).unwrap();
            let boundary = boundary_query.encode(&device, &mut encoder, &gpu).unwrap();
            queue.submit([encoder.finish()]);

            for iterations in [0, 1, 4] {
                for fix_boundary in [false, true] {
                    let settings = Smoothing {
                        step: 0.5,
                        iterations,
                    };
                    let fixed = fix_boundary.then_some(&boundary.vertices);
                    let mut encoder = device.create_command_encoder(&Default::default());
                    let result = smooth
                        .encode(
                            &device,
                            &mut encoder,
                            &gpu,
                            &adjacency,
                            &gpu_positions,
                            fixed,
                            settings,
                        )
                        .unwrap();
                    queue.submit([encoder.finish()]);
                    assert_eq!(result.mesh_id(), gpu.mesh_id());
                    assert_eq!(result.len() as usize, positions.len());
                    let actual = read_positions(&device, &queue, &result);
                    let fixed = if fix_boundary {
                        alum_boundary.clone()
                    } else {
                        vec![false; positions.len()]
                    };
                    let expected =
                        reference_smoothing(&positions, &fixture.triangles, &fixed, settings);
                    for v in 0..positions.len() {
                        let (a, e) = (actual[v], expected[v]);
                        if fixed[v] || iterations == 0 {
                            assert_eq!(a, positions[v], "vertex {v} must not move");
                        }
                        assert_eq!(a[3], positions[v][3], "vertex {v}: w lane");
                        for i in 0..3 {
                            let tolerance = 1e-5 * e[i].abs().max(1.) * iterations.max(1) as f32;
                            assert!(
                                (a[i] - e[i]).abs() <= tolerance,
                                "vertex {v} after {iterations} steps: {a:?} vs {e:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_smooth -- --ignored --nocapture"]
fn smoothing_rejects_mismatched_inputs() {
    let (device, _queue) = gpu_device();
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
    let mut encoder = device.create_command_encoder(&Default::default());
    let adjacency = AdjacencyQuery::new(&device)
        .encode(&device, &mut encoder, &gpu)
        .unwrap();
    let positions = gpu
        .upload_attribute(&device, &mesh.position_attribute())
        .unwrap();
    let wrong_positions = other_gpu
        .upload_attribute(&device, &other.position_attribute())
        .unwrap();
    let wrong_flags = GpuVertexAttribute::new(&device, &other_gpu, &[0u32; 9]).unwrap();
    let settings = Smoothing {
        step: 0.5,
        iterations: 1,
    };
    let smooth = SmoothQuery::new(&device);
    let mut run = |positions, fixed| {
        smooth
            .encode(
                &device,
                &mut encoder,
                &gpu,
                &adjacency,
                positions,
                fixed,
                settings,
            )
            .err()
    };
    assert_eq!(run(&wrong_positions, None), Some(GpuError::MeshMismatch));
    assert_eq!(
        run(&positions, Some(&wrong_flags)),
        Some(GpuError::MeshMismatch)
    );
}

/// Smooths in f64: each step moves every free vertex `step` of the way toward
/// the average of its edge neighbors.
fn reference_smoothing(
    positions: &[[f32; 4]],
    triangles: &[[u32; 3]],
    fixed: &[bool],
    settings: Smoothing,
) -> Vec<[f32; 4]> {
    let neighbors = neighbor_sets(positions.len(), triangles);
    let mut current: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| [p[0], p[1], p[2]].map(f64::from))
        .collect();
    let step = f64::from(settings.step);
    for _ in 0..settings.iterations {
        let previous = current.clone();
        for v in 0..positions.len() {
            if fixed[v] || neighbors[v].is_empty() {
                continue;
            }
            let count = neighbors[v].len() as f64;
            for i in 0..3 {
                let average = neighbors[v].iter().map(|&n| previous[n][i]).sum::<f64>() / count;
                current[v][i] = previous[v][i] + step * (average - previous[v][i]);
            }
        }
    }
    current
        .iter()
        .zip(positions)
        .map(|(c, p)| [c[0] as f32, c[1] as f32, c[2] as f32, p[3]])
        .collect()
}

/// Alum's boundary flag for each vertex.
fn boundary_vertices(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<bool> {
    let mut reference = PolyMeshF32::new();
    for &[x, y, z] in positions {
        reference.add_vertex(Vec3(x, y, z)).unwrap();
    }
    for tri in triangles {
        reference.add_face(&tri.map(Into::into)).unwrap();
    }
    let mut flags = vec![false; positions.len()];
    for v in reference.vertices() {
        flags[v.index() as usize] = v.is_boundary(&reference);
    }
    flags
}

fn read_positions(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    attribute: &GpuVertexAttribute<[f32; 4]>,
) -> Vec<[f32; 4]> {
    bytemuck::cast_slice(&read_words(device, queue, attribute.buffer())).to_vec()
}

/// Copies `buffer` back to the host after all submitted work.
fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
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
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let values: Vec<u32> =
        bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    values
}

fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    pollster::block_on(adapter.request_device(&Default::default())).unwrap()
}
