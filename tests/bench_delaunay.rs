//! Delaunay settle benchmark: runs the Delaunay example's flip rule on large
//! skewed grids, batch after batch, until a batch flips nothing. Reports the
//! number of batches, the edges flipped, and the time spent running batches
//! (submit to completion; reading statuses back is not counted).
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --test bench_delaunay -- --ignored --nocapture`.
//! Settings, all optional:
//! - `MOPMESH_BENCH_SIDES`: grid sides in vertices, default `513,1025`
//!   (about 524 K and 2.1 M faces).
//! - `MOPMESH_BENCH_PATCHES`: patch sizes, default `512`.
//! - `MOPMESH_BENCH_RUNS`: runs per grid, default 3; the median is reported.
#[path = "../examples/support/delaunay.rs"]
mod delaunay;
use mopmesh::gpu::{EdgeFlipQuery, EditCapacity, GpuEditableMesh};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

/// Batches allowed before a run is reported as not settled.
const MAX_BATCHES: u32 = 1000;

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn delaunay_settle_benchmark() {
    if std::env::var_os("MOPMESH_BENCH").is_none() {
        println!("set MOPMESH_BENCH=1 to run the Delaunay settle benchmark");
        return;
    }
    let numbers = |name: &str, default: &str| -> Vec<u32> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .expect("a comma-separated list of integers")
            })
            .collect()
    };
    let sides = numbers("MOPMESH_BENCH_SIDES", "513,1025");
    let patches = numbers("MOPMESH_BENCH_PATCHES", "512");
    let runs = numbers("MOPMESH_BENCH_RUNS", "3")[0] as usize;

    // Open the default adapter with its own limits.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let query = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &query);
    println!("# MOPMesh Delaunay settle benchmark");
    if cfg!(debug_assertions) {
        println!("WARNING: debug build; timings are not representative");
    }
    println!("adapter: {}", adapter.get_info().name);
    println!("median of {runs} runs; batch time is submit to completion, summed over batches");
    println!("| grid | faces | patch size | batches | flipped | left out of room | batch time |");
    println!("|---|---:|---:|---:|---:|---:|---:|");

    for &side in &sides {
        for &patch_size in &patches {
            let (mesh, positions, _) = delaunay::skewed_grid(side, patch_size as usize);
            let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("positions"),
                contents: bytemuck::cast_slice(&positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

            // Settle a fresh upload in each run.
            let mut results = Vec::new();
            for _ in 0..runs {
                let mut gpu =
                    GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
                let (mut batches, mut flipped, mut time) = (0, 0, Duration::ZERO);
                let out_of_room = loop {
                    batches += 1;
                    assert!(batches <= MAX_BATCHES, "{side}x{side}: not settled");
                    let start = Instant::now();
                    let mut encoder = device.create_command_encoder(&Default::default());
                    let status = flips
                        .encode(&device, &mut encoder, &query, &mut gpu, &positions)
                        .unwrap();
                    queue.submit([encoder.finish()]);
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(120)),
                        })
                        .unwrap();
                    time += start.elapsed();
                    let round = delaunay::Round::from_statuses(&delaunay::read_words(
                        &device, &queue, &status,
                    ));
                    if round.flipped == 0 {
                        break round.out_of_room;
                    }
                    flipped += round.flipped;
                };
                results.push((time, batches, flipped, out_of_room));
            }
            results.sort();
            let (time, batches, flipped, out_of_room) = results[results.len() / 2];
            println!(
                "| {side}x{side} | {} | {patch_size} | {batches} | {flipped} | {out_of_room} | {:.1} ms |",
                mesh.face_count(),
                time.as_secs_f64() * 1e3
            );
        }
    }
}
