//! Whole-attribute reductions (sum, min, max, dot) checked against host
//! arithmetic, across one to four reduction levels.
use mopmesh::{
    EdgeAttribute, FaceAttribute, Mesh, MeshBuilder, VertexAttribute,
    gpu::{DotQuery, GpuError, GpuMesh, GpuReduction, ReduceOp, ReduceQuery, ReduceValue},
};

#[path = "support/fixtures.rs"]
mod fixtures;

struct Queries {
    f32: [ReduceQuery<f32>; 3],
    u32: [ReduceQuery<u32>; 3],
    i32: [ReduceQuery<i32>; 3],
    dot: DotQuery,
}
impl Queries {
    fn new(device: &wgpu::Device) -> Self {
        let ops = [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max];
        Self {
            f32: ops.map(|op| ReduceQuery::new(device, op)),
            u32: ops.map(|op| ReduceQuery::new(device, op)),
            i32: ops.map(|op| ReduceQuery::new(device, op)),
            dot: DotQuery::new(device),
        }
    }
}

// Explicit opt-in: a missing adapter is a failure, never a silent successful test.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_reduce -- --ignored --nocapture"]
fn gpu_reductions_match_host_arithmetic() {
    // The default device, then one whose dispatch grid needs two dimensions.
    for limit in [None, Some(128)] {
        let (device, queue) = gpu_device(limit);
        let queries = Queries::new(&device);
        // Vertex counts 0, 4, 81, 4,225 and 263,169: up to four levels.
        let meshes = [
            mesh(vec![], vec![]),
            tetrahedron(),
            grid(9),
            grid(65),
            grid(513),
        ];
        for mesh in &meshes {
            eprintln!(
                "checking {} vertices, device limit {limit:?}",
                mesh.positions().len()
            );
            check(&device, &queue, &queries, mesh);
        }
    }
}

/// Reduces vertex, edge and face attributes of `mesh` with every operation.
fn check(device: &wgpu::Device, queue: &wgpu::Queue, queries: &Queries, mesh: &Mesh) {
    let gpu = GpuMesh::upload(device, mesh).unwrap();
    let vertices = mesh.positions().len();

    // f32 vertex values, mixed signs, compared with an f64 sum.
    let floats: Vec<f32> = (0..vertices)
        .map(|v| ((v * 37 % 101) as f32 - 50.) * 0.25)
        .collect();
    let attribute = gpu
        .upload_attribute(
            device,
            &VertexAttribute::from_values(mesh, floats.clone()).unwrap(),
        )
        .unwrap();
    let [sum, min, max] = &queries.f32;
    let expected: f64 = floats.iter().map(|&x| f64::from(x)).sum();
    let bound: f64 = floats.iter().map(|&x| f64::from(x).abs()).sum::<f64>() * 1e-6;
    let actual = reduce(device, queue, &gpu, sum, &attribute);
    assert!(
        (f64::from(actual) - expected).abs() <= bound.max(1e-6),
        "f32 sum {actual} vs {expected}"
    );
    let lowest = floats.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = floats.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(reduce(device, queue, &gpu, min, &attribute), lowest);
    assert_eq!(reduce(device, queue, &gpu, max, &attribute), highest);

    // The dot product of the attribute with itself.
    let mut encoder = device.create_command_encoder(&Default::default());
    let dot = queries
        .dot
        .encode(device, &mut encoder, &gpu, &attribute, &attribute)
        .unwrap();
    queue.submit([encoder.finish()]);
    let expected: f64 = floats.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let actual = f64::from(read(device, queue, &dot));
    assert!(
        (actual - expected).abs() <= expected * 1e-6 + 1e-6,
        "dot {actual} vs {expected}"
    );

    // u32 edge values large enough to wrap the sum.
    let edges = mesh.edge_count();
    let words: Vec<u32> = (0..edges as u32)
        .map(|e| e.wrapping_mul(2_654_435_761))
        .collect();
    let attribute = gpu
        .upload_attribute(
            device,
            &EdgeAttribute::from_values(mesh, words.clone()).unwrap(),
        )
        .unwrap();
    let [sum, min, max] = &queries.u32;
    let wrapped = words.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    assert_eq!(reduce(device, queue, &gpu, sum, &attribute), wrapped);
    assert_eq!(
        reduce(device, queue, &gpu, min, &attribute),
        words.iter().copied().min().unwrap_or(u32::MAX)
    );
    assert_eq!(
        reduce(device, queue, &gpu, max, &attribute),
        words.iter().copied().max().unwrap_or(0)
    );

    // i32 face values on both sides of zero.
    let faces = mesh.face_count();
    let ints: Vec<i32> = (0..faces as i64)
        .map(|f| ((f * 7919) % 20011 - 10005) as i32)
        .collect();
    let attribute = gpu
        .upload_attribute(
            device,
            &FaceAttribute::from_values(mesh, ints.clone()).unwrap(),
        )
        .unwrap();
    let [sum, min, max] = &queries.i32;
    assert_eq!(
        reduce(device, queue, &gpu, sum, &attribute),
        ints.iter().fold(0i32, |a, &b| a.wrapping_add(b))
    );
    assert_eq!(
        reduce(device, queue, &gpu, min, &attribute),
        ints.iter().copied().min().unwrap_or(i32::MAX)
    );
    assert_eq!(
        reduce(device, queue, &gpu, max, &attribute),
        ints.iter().copied().max().unwrap_or(i32::MIN)
    );

    // Empty f32 attributes give the documented identities.
    if vertices == 0 {
        let attribute = gpu
            .upload_attribute(device, &VertexAttribute::new(mesh, 1f32))
            .unwrap();
        let [_, min, max] = &queries.f32;
        assert_eq!(reduce(device, queue, &gpu, min, &attribute), f32::INFINITY);
        assert_eq!(
            reduce(device, queue, &gpu, max, &attribute),
            f32::NEG_INFINITY
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_reduce -- --ignored --nocapture"]
fn dot_product_rejects_attributes_of_different_meshes() {
    let (device, _queue) = gpu_device(None);
    let (a, b) = (grid(3), grid(3));
    let upload = |mesh: &Mesh| {
        let gpu = GpuMesh::upload(&device, mesh).unwrap();
        let attribute = gpu
            .upload_attribute(&device, &VertexAttribute::new(mesh, 1f32))
            .unwrap();
        (gpu, attribute)
    };
    let ((gpu, a), (_, b)) = (upload(&a), upload(&b));
    let mut encoder = device.create_command_encoder(&Default::default());
    assert_eq!(
        DotQuery::new(&device)
            .encode(&device, &mut encoder, &gpu, &a, &b)
            .err(),
        Some(GpuError::MeshMismatch)
    );
}

/// Records and submits one reduction, then reads its value back.
fn reduce<E: mopmesh::Element, T: ReduceValue>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &GpuMesh,
    query: &ReduceQuery<T>,
    attribute: &mopmesh::gpu::GpuAttribute<E, T>,
) -> T {
    let mut encoder = device.create_command_encoder(&Default::default());
    let result = query.encode(device, &mut encoder, gpu, attribute).unwrap();
    assert_eq!(result.mesh_id(), attribute.mesh_id());
    queue.submit([encoder.finish()]);
    read(device, queue, &result)
}

fn read<T: ReduceValue>(device: &wgpu::Device, queue: &wgpu::Queue, result: &GpuReduction<T>) -> T {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(result.buffer(), 0, &readback, 0, 4);
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
    let value = bytemuck::pod_read_unaligned(&readback.slice(..).get_mapped_range().unwrap());
    readback.unmap();
    value
}

fn mesh(positions: Vec<[f32; 3]>, triangles: Vec<[u32; 3]>) -> Mesh {
    MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap()
}

fn grid(n: u32) -> Mesh {
    let (positions, triangles) = fixtures::grid(n);
    mesh(positions, triangles)
}

fn tetrahedron() -> Mesh {
    mesh(
        vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    )
}

/// Requests a device, optionally limiting workgroups per dispatch dimension.
fn gpu_device(workgroup_limit: Option<u32>) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    let mut limits = wgpu::Limits::default();
    if let Some(limit) = workgroup_limit {
        limits.max_compute_workgroups_per_dimension = limit;
    }
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap()
}
