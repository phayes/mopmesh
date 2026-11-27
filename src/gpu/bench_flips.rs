//! Flip benchmark: one batch requesting every edge of each benchmark mesh,
//! on a fresh editable upload for every run, timing each compute pass the
//! batch records.
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --lib gpu_flip_benchmark -- --ignored --nocapture`.
//! Without `MOPMESH_BENCH` it returns at once. Settings, all optional:
//! - `MOPMESH_BENCH_FACES`: comma-separated target face counts, default
//!   `524288,2097152`.
//! - `MOPMESH_BENCH_PATCHES`: comma-separated patch sizes, default `512`.
//! - `MOPMESH_BENCH_RUNS`: timed runs per mesh, default 11, after the
//!   warm-up runs.
use super::bench::{ms, wait};
use super::bench_baseline::WARMUP;
use super::bench_meshes::{self, BenchMesh};
use super::pipeline::instrument::{PASS_TIMESTAMPS, PassTimestamps};
use super::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh};
use crate::MeshBuilder;
use std::cell::RefCell;
use std::time::{Duration, Instant};

/// The most compute passes one flip batch records.
const MAX_PASSES: u32 = 8;

/// Timestamp queries for the compute passes of one batch.
struct PassTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f64,
}
impl PassTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16 * u64::from(MAX_PASSES),
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("flip benchmark timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * MAX_PASSES,
            }),
            resolve: buffer(
                "flip benchmark resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "flip benchmark timestamps",
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ),
            period_ns: f64::from(queue.get_timestamp_period()),
        }
    }

    /// Records, submits and waits for `record`, returning the wall time and
    /// the GPU time of each compute pass it recorded, in order.
    fn sample(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        record: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> (Duration, Vec<Duration>) {
        // Time every pass the batch records.
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        PASS_TIMESTAMPS.set(Some(PassTimestamps {
            set: self.set.clone(),
            capacity: MAX_PASSES,
            passes: 0,
        }));
        record(&mut encoder);
        let passes = PASS_TIMESTAMPS.take().map_or(0, |t| t.passes);
        assert!(passes <= MAX_PASSES, "{passes} passes; raise MAX_PASSES");
        if passes > 0 {
            encoder.resolve_query_set(&self.set, 0..2 * passes, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &self.resolve,
                0,
                &self.readback,
                0,
                16 * u64::from(passes),
            );
        }
        queue.submit([encoder.finish()]);
        wait(device);
        let wall = start.elapsed();
        if passes == 0 {
            return (wall, Vec::new());
        }

        // Read each pass's start and end after the wall clock stops.
        let slice = self.readback.slice(..16 * u64::from(passes));
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        wait(device);
        let times = bytemuck::cast_slice::<u8, [u64; 2]>(&slice.get_mapped_range().unwrap())
            .iter()
            .map(|[start, end]| {
                Duration::from_nanos((end.saturating_sub(*start) as f64 * self.period_ns) as u64)
            })
            .collect();
        self.readback.unmap();
        (wall, times)
    }
}

/// The median of `times`, formatted in milliseconds.
fn median(times: &mut [Duration]) -> String {
    times.sort();
    ms(times[times.len() / 2])
}

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
    assert!(
        device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        "the flip benchmark needs timestamp queries"
    );
    let timer = PassTimer::new(&device, &queue);
    let query = EdgeFlipQuery::new(&device);
    let info = adapter.get_info();
    println!("# MOPMesh flip benchmark");
    if cfg!(debug_assertions) {
        println!("WARNING: debug build; timings are not representative");
    }
    println!("adapter: {} ({:?})", info.name, info.backend);
    println!(
        "one batch requesting every edge, fresh upload at capacity 1.5 per run; \
         {runs} timed runs after {WARMUP} warm-up; medians in ms: GPU time of each \
         compute pass in recording order, their sum, and wall time"
    );
    println!("| mesh | patch size | edges | flipped | lost claim | passes | GPU sum | wall |");
    println!("|---|---:|---:|---:|---:|---|---:|---:|");

    // Each mesh, then a shuffled copy of it.
    for mesh in faces.into_iter().flat_map(bench_meshes::generated) {
        let shuffled = bench_meshes::shuffled(&mesh);
        for mesh in [&mesh, &shuffled] {
            for &patch_size in &patches {
                row(&device, &queue, &timer, &query, mesh, patch_size, runs);
            }
        }
    }
}

/// Times one batch over every edge of `bench` and prints one table row.
fn row(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    timer: &PassTimer,
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
    let (mut passes, mut sums, mut walls) = (Vec::<Vec<Duration>>::new(), Vec::new(), Vec::new());
    let mut words = Vec::new();
    for run in 0..WARMUP + runs {
        let mut editable = GpuEditableMesh::upload(device, &mesh, EditCapacity::default()).unwrap();
        wait(device);
        let statuses = RefCell::new(None);
        let (wall, times) = timer.sample(device, queue, |encoder| {
            let result = query
                .encode(device, encoder, &mut editable, &edges)
                .unwrap();
            *statuses.borrow_mut() = Some(result);
        });
        if run >= WARMUP {
            passes.resize(times.len(), Vec::new());
            for (pass, &t) in passes.iter_mut().zip(&times) {
                pass.push(t);
            }
            sums.push(times.iter().sum::<Duration>());
            walls.push(wall);
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
    let per_pass: Vec<String> = passes.iter_mut().map(|p| median(p)).collect();
    println!(
        "| {} | {patch_size} | {} | {} | {} | {} | {} | {} |",
        bench.name,
        edges.len(),
        count(FlipStatus::Flipped),
        count(FlipStatus::LostClaim),
        per_pass.join(", "),
        median(&mut sums),
        median(&mut walls),
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
