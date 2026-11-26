//! Run: cargo run --example weighted_boundary
#[path = "support/boundary.rs"]
mod boundary;
use mopmesh::{
    MeshBuilder,
    gpu::{EdgeQuery, GpuMesh},
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
    let mesh = MeshBuilder::new()
        .positions(&[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        .triangles(&[[0, 1, 2], [0, 2, 3]])
        .patch_size(1)
        .build()
        .unwrap();
    let gpu = GpuMesh::upload(&device, &mesh).unwrap();
    let positions = gpu
        .upload_vertex_attribute(&device, &mesh.position_attribute())
        .unwrap();
    let query = EdgeQuery::new(&device);
    let operator = boundary::BoundaryOperator::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    let edges = query.encode(&device, &mut encoder, &gpu).unwrap();
    let weights = vec![1.; mesh.edge_count()];
    let output = operator.encode(&device, &mut encoder, &edges, &positions, &weights);
    // No intermediate readback: the downstream operator directly consumes EF/EV.
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("example final readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
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
    {
        let bytes = readback.slice(..).get_mapped_range().unwrap();
        let rows: &[[f32; 2]] = bytemuck::cast_slice(&bytes);
        let boundary_count = rows.iter().filter(|r| r[1] == 1.).count();
        let weighted_length: f32 = rows.iter().map(|r| r[0]).sum();
        assert_eq!(boundary_count, 4);
        assert!((weighted_length - 4.).abs() < 1e-5);
        println!(
            "{} edges, {boundary_count} boundary edges, weighted boundary length {weighted_length}",
            edges.edge_count()
        );
    }
    readback.unmap();
}
