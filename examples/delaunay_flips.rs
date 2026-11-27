//! Demonstrates an application WESL rule that improves a flat triangulation
//! by flipping edges whose opposite angles exceed 180 degrees, with a small
//! tolerance. Repeats batches on a skewed grid until no flips or retries remain,
//! then runs queries on the edited mesh: results from before the flips are
//! rejected, and recomputed ones describe the new triangles.
//!
//! Run with `cargo run --example delaunay_flips` on a machine with a GPU adapter.
#[path = "support/delaunay.rs"]
mod delaunay;
use mopmesh::gpu::{
    AdjacencyQuery, EdgeFlipQuery, EditCapacity, FaceAreaQuery, GpuEditableMesh, GpuError,
    GpuVertexAttribute, ReduceOp, ReduceQuery, SmoothQuery, Smoothing,
};
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

    // Create a 33 x 33 grid in several patches, with spare storage for the
    // copies that flips across patch borders need.
    // Positions have one entry per vertex ID, so pad them to the mesh's
    // vertex ID capacity.
    let (mesh, mut positions, _) = delaunay::skewed_grid(33, 512);
    let mut gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
    positions.resize(gpu.vertex_capacity() as usize, [0.; 4]);
    let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("positions"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &query);

    // Queries accept the editable mesh like a static one. Record the total
    // area and the vertex neighbors before flipping.
    let attribute = GpuVertexAttribute::from_buffer(&gpu, &positions).unwrap();
    let areas = FaceAreaQuery::new(&device);
    let sum = ReduceQuery::<f32>::new(&device, ReduceOp::Sum);
    let total_area = |gpu: &GpuEditableMesh| {
        let mut encoder = device.create_command_encoder(&Default::default());
        let face_areas = areas
            .encode(&device, &mut encoder, gpu, &attribute)
            .unwrap();
        let total = sum.encode(&device, &mut encoder, gpu, &face_areas).unwrap();
        queue.submit([encoder.finish()]);
        f32::from_bits(delaunay::read_words(&device, &queue, total.buffer())[0])
    };
    let area_before = total_area(&gpu);
    let adjacency_query = AdjacencyQuery::new(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    let adjacency_before = adjacency_query.encode(&device, &mut encoder, &gpu).unwrap();
    queue.submit([encoder.finish()]);

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
            "round {round}: {} flipped, {} to retry, {} out of room",
            result.flipped, result.retry, result.out_of_room
        );
        total += result.flipped;
        assert_eq!(
            result.out_of_room, 0,
            "a patch ran out of room; upload with a larger EditCapacity"
        );
        if result.flipped == 0 && result.retry == 0 {
            println!(
                "Delaunay after {total} flips, mesh version {}",
                gpu.version()
            );
            assert!(total > 0, "the skewed grid should need flips");
            after_flips(&device, &queue, &gpu, &attribute, &adjacency_before);

            // Flips rearrange triangles inside the same outline, so the total
            // area stays the same.
            let area_after = total_area(&gpu);
            println!("total area {area_before} before flips, {area_after} after");
            assert!((area_after - area_before).abs() <= 1e-3 * area_before);
            return;
        }
    }
    panic!("flipping did not settle within 64 rounds");
}

/// Smooths the flipped mesh. Adjacency from before the flips is rejected as
/// stale, so it is recomputed from the edited mesh first.
fn after_flips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu: &GpuEditableMesh,
    positions: &GpuVertexAttribute<[f32; 4]>,
    adjacency_before: &mopmesh::gpu::MeshAdjacency,
) {
    let smooth = SmoothQuery::new(device);
    let settings = Smoothing {
        step: 0.5,
        iterations: 1,
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    let stale = smooth.encode(
        device,
        &mut encoder,
        gpu,
        adjacency_before,
        positions,
        None,
        settings,
    );
    assert_eq!(stale.err(), Some(GpuError::Stale));
    println!("adjacency from before the flips is stale; recomputing it");
    let adjacency = AdjacencyQuery::new(device)
        .encode(device, &mut encoder, gpu)
        .unwrap();
    smooth
        .encode(
            device,
            &mut encoder,
            gpu,
            &adjacency,
            positions,
            None,
            settings,
        )
        .unwrap();
    queue.submit([encoder.finish()]);
}
