//! Run: cargo run --example neighborhood
#[path = "support/neighborhood.rs"]
mod neighborhood;
use mopmesh::{
    MeshBuilder,
    gpu::{AdjacencyQuery, GpuMesh},
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
    let query = AdjacencyQuery::new(&device);
    let operator = neighborhood::NeighborhoodOperator::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    let adjacency = query.encode(&device, &mut encoder, &gpu).unwrap();
    let output = operator.encode(&device, &mut encoder, &adjacency, &positions, &[1.; 4]);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neighborhood final readback"),
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
        let rows: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        assert_eq!(rows.iter().map(|r| r[0]).sum::<f32>(), 10.);
        assert_eq!(rows.iter().map(|r| r[1]).sum::<f32>(), 8.);
        for (vertex, row) in rows.iter().enumerate() {
            println!(
                "vertex {vertex}: degree {}, boundary edges {}, mean length {:.6}",
                row[0], row[1], row[2]
            );
        }
    }
    readback.unmap();
}
