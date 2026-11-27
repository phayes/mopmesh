//! Demonstrates an application WESL rule that improves a flat triangulation
//! by flipping edges whose opposite angles exceed 180 degrees, with a small
//! tolerance. Repeats batches on a skewed grid until no flips or retries remain.
//!
//! Run with `cargo run --example delaunay_flips` on a machine with a GPU adapter.
#[path = "support/delaunay.rs"]
mod delaunay;
use mopmesh::gpu::{EdgeFlipQuery, EditCapacity, GpuEditableMesh};
use wgpu::util::DeviceExt;

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

    // Create an 11 x 11 grid and reserve GPU storage for replacement edges and faces.
    let (mesh, positions, _) = delaunay::skewed_grid(11);
    let mut gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
    let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("positions"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &query);

    // Each batch flips edges that share no vertices. Recheck all edges in the
    // next batch, including requests that lost a shared vertex to another flip.
    let mut total = 0;
    for round in 1..=64 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let status = flips
            .encode(&device, &mut encoder, &query, &mut gpu, &positions)
            .unwrap();
        queue.submit([encoder.finish()]);
        let result =
            delaunay::Round::from_statuses(&delaunay::read_words(&device, &queue, &status));
        println!(
            "round {round}: {} flipped, {} to retry",
            result.flipped, result.retry
        );
        total += result.flipped;
        if result.flipped == 0 && result.retry == 0 {
            println!(
                "Delaunay after {total} flips, mesh version {}",
                gpu.version()
            );
            assert!(total > 0, "the skewed grid should need flips");
            return;
        }
    }
    panic!("flipping did not settle within 64 rounds");
}
