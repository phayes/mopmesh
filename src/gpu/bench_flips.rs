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
//! - `MOPMESH_BENCH_STAGES`: if set, also prints each row's median GPU time
//!   per pass label, summed over the batch.
use super::bench::{ms, wait};
use super::bench_baseline::WARMUP;
use super::bench_meshes::{self, BenchMesh};
use super::pipeline::instrument::{PASS_TIMESTAMPS, PassTimestamps};
use super::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh};
use crate::MeshBuilder;
use std::cell::RefCell;
use std::time::{Duration, Instant};

/// The most compute passes one flip batch records.
const MAX_PASSES: u32 = 256;

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

    /// Records, submits and waits for `record`. Returns the wall time; the
    /// GPU span from the first compute pass's start to the last one's end,
    /// which also covers buffer clears and copies between passes; and the
    /// label and GPU time of each compute pass, in order.
    fn sample(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        record: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> (Duration, Duration, Vec<(String, Duration)>) {
        // Time every pass the batch records.
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        PASS_TIMESTAMPS.set(Some(PassTimestamps {
            set: self.set.clone(),
            capacity: MAX_PASSES,
            passes: 0,
            labels: Vec::new(),
        }));
        record(&mut encoder);
        let (passes, labels) = PASS_TIMESTAMPS
            .take()
            .map_or((0, Vec::new()), |t| (t.passes, t.labels));
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
            return (wall, Duration::ZERO, Vec::new());
        }

        // Read each pass's start and end after the wall clock stops.
        let slice = self.readback.slice(..16 * u64::from(passes));
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        wait(device);
        let stamps =
            bytemuck::cast_slice::<u8, [u64; 2]>(&slice.get_mapped_range().unwrap()).to_vec();
        self.readback.unmap();
        let duration = |start: u64, end: u64| {
            Duration::from_nanos((end.saturating_sub(start) as f64 * self.period_ns) as u64)
        };
        let span = duration(stamps[0][0], stamps[stamps.len() - 1][1]);
        let times = stamps
            .iter()
            .zip(labels)
            .map(|([start, end], label)| (label, duration(*start, *end)))
            .collect();
        (wall, span, times)
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
        "one batch requesting every edge, fresh upload at the default capacity per run; \
         {runs} timed runs after {WARMUP} warm-up; medians in ms: GPU time of the \
         interior pass, of the cross-patch passes after it (with their count), \
         their sum, the GPU span from the first pass's start to the last one's end \
         (clears and copies included), and wall time"
    );
    println!(
        "| mesh | patch size | edges | flipped | lost claim | retry | out of room | \
         interior | cross-patch | GPU sum | GPU span | wall |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|");

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
    let (mut interiors, mut crosses, mut sums, mut walls) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut cross_passes = 0;
    let mut words = Vec::new();
    let mut stages: Vec<(String, Vec<Duration>)> = Vec::new();
    let mut rounds: Vec<Vec<Duration>> = Vec::new();
    let mut spans = Vec::new();
    for run in 0..WARMUP + runs {
        let mut editable = GpuEditableMesh::upload(device, &mesh, EditCapacity::default()).unwrap();
        wait(device);
        let statuses = RefCell::new(None);
        let (wall, span, times) = timer.sample(device, queue, |encoder| {
            let result = query
                .encode(device, encoder, &mut editable, &edges)
                .unwrap();
            *statuses.borrow_mut() = Some(result);
        });
        if run >= WARMUP {
            let ((_, interior), cross) = times.split_first().expect("the batch records passes");
            interiors.push(*interior);
            crosses.push(cross.iter().map(|(_, t)| *t).sum::<Duration>());
            cross_passes = cross.len();
            sums.push(times.iter().map(|(_, t)| *t).sum::<Duration>());
            spans.push(span);
            walls.push(wall);

            // Sum this run's passes by label, labels in first-recorded order,
            // and by cross-patch round, each starting at its plan pass.
            let mut run_stages: Vec<(String, Duration)> = Vec::new();
            let mut run_rounds: Vec<Duration> = Vec::new();
            for (label, time) in times {
                if label == "plan" {
                    run_rounds.push(Duration::ZERO);
                }
                if let Some(total) = run_rounds.last_mut() {
                    *total += time;
                }
                match run_stages.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, total)) => *total += time,
                    None => run_stages.push((label, time)),
                }
            }
            for (label, total) in run_stages {
                match stages.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, totals)) => totals.push(total),
                    None => stages.push((label, vec![total])),
                }
            }
            rounds.resize(rounds.len().max(run_rounds.len()), Vec::new());
            for (totals, total) in rounds.iter_mut().zip(run_rounds) {
                totals.push(total);
            }
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
    println!(
        "| {} | {patch_size} | {} | {} | {} | {} | {} | {} | {} ({cross_passes}) | {} | {} | {} |",
        bench.name,
        edges.len(),
        count(FlipStatus::Flipped),
        count(FlipStatus::LostClaim),
        count(FlipStatus::Retry),
        count(FlipStatus::OutOfCapacity),
        median(&mut interiors),
        median(&mut crosses),
        median(&mut sums),
        median(&mut spans),
        median(&mut walls),
    );

    // With MOPMESH_BENCH_STAGES, each pass label's median total per batch.
    if std::env::var_os("MOPMESH_BENCH_STAGES").is_some() {
        let breakdown: Vec<String> = stages
            .iter_mut()
            .map(|(label, times)| format!("{label} {}", median(times)))
            .collect();
        println!("  stages: {}", breakdown.join(", "));
        let by_round: Vec<String> = rounds.iter_mut().map(|times| median(times)).collect();
        println!("  rounds: {}", by_round.join(", "));
    }
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
