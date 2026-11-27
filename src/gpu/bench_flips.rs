//! Flip benchmark: one batch requesting every edge of each benchmark mesh,
//! on a fresh editable upload for every run, timing the claim, reserve and
//! commit dispatches.
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --lib gpu_flip_benchmark -- --ignored --nocapture`.
//! Without `MOPMESH_BENCH` it returns at once. Settings, all optional:
//! - `MOPMESH_BENCH_FACES`: comma-separated target face counts, default
//!   `524288,2097152`.
//! - `MOPMESH_BENCH_PATCHES`: comma-separated patch sizes, default `512`.
//! - `MOPMESH_BENCH_RUNS`: timed runs per mesh, default 11, after the
//!   warm-up runs.
use super::bench::{ms, wait};
use super::bench_baseline::{Timer, WARMUP, sample, spread};
use super::bench_meshes::{self, BenchMesh};
use super::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh};
use crate::MeshBuilder;
use std::cell::RefCell;

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn gpu_flip_benchmark() {
    if std::env::var_os("MOPMESH_BENCH").is_none() {
        println!("set MOPMESH_BENCH=1 to run the flip benchmark");
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
    let faces = numbers("MOPMESH_BENCH_FACES", "524288,2097152");
    let patches = numbers("MOPMESH_BENCH_PATCHES", "512");
    let runs = numbers("MOPMESH_BENCH_RUNS", "11")[0] as usize;

    // Open the default adapter with its own limits and timestamps.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    let timer = Timer::new(&device, &queue);
    let query = EdgeFlipQuery::new(&device);
    let info = adapter.get_info();
    println!("# MOPMesh flip benchmark");
    if cfg!(debug_assertions) {
        println!("WARNING: debug build; timings are not representative");
    }
    println!("adapter: {} ({:?})", info.name, info.backend);
    println!(
        "one batch requesting every edge, fresh upload at capacity 1.5 per run; \
         {runs} timed runs after {WARMUP} warm-up; GPU time is the sum of the \
         batch's compute passes; times are min / median / max in ms"
    );
    println!("| mesh | patch size | edges | flipped | lost claim | GPU | wall (median) |");
    println!("|---|---:|---:|---:|---:|---|---:|");

    // Each mesh, then a shuffled copy of it.
    for mesh in faces.into_iter().flat_map(bench_meshes::generated) {
        let shuffled = bench_meshes::shuffled(&mesh);
        for mesh in [&mesh, &shuffled] {
            for &patch_size in &patches {
                row(
                    &device,
                    &queue,
                    timer.as_ref(),
                    &query,
                    mesh,
                    patch_size,
                    runs,
                );
            }
        }
    }
}

/// Times one batch over every edge of `bench` and prints one table row.
fn row(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    timer: Option<&Timer>,
    query: &EdgeFlipQuery,
    bench: &BenchMesh,
    patch_size: u32,
    runs: usize,
) {
    let mesh = MeshBuilder::new()
        .patch_size(patch_size as usize)
        .positions(&bench.positions)
        .triangles(&bench.triangles)
        .build()
        .unwrap();
    let edges: Vec<u32> = (0..mesh.edge_count() as u32).collect();

    // Each run flips a fresh upload, made before the timed batch starts.
    let (mut gpu_times, mut wall_times) = (Vec::new(), Vec::new());
    let mut words = Vec::new();
    for run in 0..WARMUP + runs {
        let editable =
            RefCell::new(GpuEditableMesh::upload(device, &mesh, EditCapacity::default()).unwrap());
        wait(device);
        let statuses = RefCell::new(None);
        let (wall, gpu) = sample(device, queue, timer, &|encoder| {
            let result = query
                .encode(device, encoder, &mut editable.borrow_mut(), &edges)
                .unwrap();
            *statuses.borrow_mut() = Some(result);
        });
        if run >= WARMUP {
            wall_times.push(wall);
            gpu_times.extend(gpu);
        }
        if run + 1 == WARMUP + runs {
            let statuses = statuses.into_inner().unwrap();
            words = read(device, queue, statuses.buffer(), edges.len());
        }
    }

    // Requests are every edge in ID order, so request index and edge ID agree.
    let count = |status| {
        words
            .iter()
            .filter(|&&w| FlipStatus::from_word(w) == Some(status))
            .count()
    };
    let gpu = if gpu_times.is_empty() {
        "-".to_owned()
    } else {
        spread(&mut gpu_times)
    };
    wall_times.sort();
    println!(
        "| {} | {patch_size} | {} | {} | {} | {gpu} | {} |",
        bench.name,
        edges.len(),
        count(FlipStatus::Flipped),
        count(FlipStatus::LostClaim),
        ms(wall_times[wall_times.len() / 2]),
    );
}

/// Copies the first `len` words of `buffer` back to the CPU.
fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, len: usize) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flip benchmark readback"),
        size: 4 * len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, 4 * len as u64);
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
    wait(device);
    let words = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    words
}
