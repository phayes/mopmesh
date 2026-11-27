//! Smooths a bumpy grid with its boundary held in place, then compares how
//! much the surface bends and its total area before and after.
//!
//! Run: cargo run --example smoothing
#[path = "support/bending.rs"]
mod bending;
use mopmesh::{
    MeshBuilder,
    gpu::{
        AdjacencyQuery, BoundaryQuery, FaceAreaQuery, GpuMesh, GpuReduction, ReduceOp, ReduceQuery,
        SmoothQuery, Smoothing, VertexNormalQuery,
    },
};

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&Default::default())
        .await
        .expect("GPU adapter required");
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    // A 33 x 33 grid with a bumpy height field.
    let n = 33u32;
    let mut positions = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let (fx, fy) = (x as f32, y as f32);
            positions.push([fx, fy, 0.8 * (0.9 * fx).sin() * (0.7 * fy).cos()]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let a = y * n + x;
            triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]);
        }
    }
    let mesh = MeshBuilder::new()
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap();
    let gpu = GpuMesh::upload(&device, &mesh).unwrap();
    let before = gpu
        .upload_attribute(&device, &mesh.position_attribute())
        .unwrap();

    // Library queries, cached once per device, and the downstream operator.
    let adjacency_query = AdjacencyQuery::new(&device);
    let boundary_query = BoundaryQuery::new(&device);
    let smooth = SmoothQuery::new(&device);
    let normals = VertexNormalQuery::new(&device);
    let areas = FaceAreaQuery::new(&device);
    let sum = ReduceQuery::<f32>::new(&device, ReduceOp::Sum);
    let bending = bending::BendingOperator::new(&device);

    // Smooth with the boundary fixed, then measure both surfaces, all in one
    // submission and without reading anything back in between.
    let mut encoder = device.create_command_encoder(&Default::default());
    let adjacency = adjacency_query.encode(&device, &mut encoder, &gpu).unwrap();
    let boundary = boundary_query.encode(&device, &mut encoder, &gpu).unwrap();
    let settings = Smoothing {
        step: 0.5,
        iterations: 20,
    };
    let after = smooth
        .encode(
            &device,
            &mut encoder,
            &gpu,
            &adjacency,
            &before,
            Some(&boundary.vertices),
            settings,
        )
        .unwrap();
    let mut measure = |positions| {
        let vertex_normals = normals
            .encode(&device, &mut encoder, &gpu, positions)
            .unwrap();
        let per_vertex = bending.encode(
            &device,
            &mut encoder,
            &gpu,
            &adjacency,
            positions,
            &vertex_normals,
        );
        let face_areas = areas
            .encode(&device, &mut encoder, &gpu, positions)
            .unwrap();
        (
            sum.encode(&device, &mut encoder, &per_vertex).unwrap(),
            sum.encode(&device, &mut encoder, &face_areas).unwrap(),
        )
    };
    let (bending_before, area_before) = measure(&before);
    let (bending_after, area_after) = measure(&after);
    queue.submit([encoder.finish()]);

    // Read the four totals back and report them.
    let vertex_count = mesh.positions().len() as f32;
    let [bending_before, area_before, bending_after, area_after] =
        [&bending_before, &area_before, &bending_after, &area_after]
            .map(|total| read(&device, &queue, total));
    println!(
        "mean bending: {:.4} -> {:.4}",
        bending_before / vertex_count,
        bending_after / vertex_count
    );
    println!("total area: {area_before:.2} -> {area_after:.2}");
    assert!(
        bending_after < bending_before,
        "smoothing should reduce bending"
    );
    assert!(
        area_after < area_before,
        "smoothing a bumpy surface should reduce its area"
    );
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, total: &GpuReduction<f32>) -> f32 {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoothing example readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(total.buffer(), 0, &readback, 0, 4);
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
