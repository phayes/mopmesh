//! Release-mode baseline benchmark (ROADMAP milestone 2): setup costs, memory,
//! and warm query times on the patch arena against the unpartitioned
//! [`FlatMesh`] twins, on generated, shuffled and OBJ meshes. Not a pass/fail test.
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --lib gpu_baseline_benchmark -- --ignored --nocapture`.
//! Without `MOPMESH_BENCH` it returns at once. Settings, all optional:
//! - `MOPMESH_BENCH_FACES`: comma-separated target face counts for the
//!   generated meshes, default `32768,524288,2097152`.
//! - `MOPMESH_BENCH_PATCHES`: comma-separated patch sizes, default `128`.
//! - `MOPMESH_BENCH_OBJ`: comma-separated OBJ paths, benchmarked with a
//!   shuffled copy each.
//! - `MOPMESH_BENCH_RUNS`: timed runs per query, default 11, after 2 warm-up runs.
//!
//! Patched and flat runs of a query alternate within each round, so machine
//! drift affects both. Wall time covers encoding (including the buffers each
//! `encode` allocates), submitting and waiting. GPU time, where the adapter
//! supports timestamp queries, is the sum of each compute pass's own time;
//! it leaves out gaps between passes.
use super::baseline::{FlatMesh, FlatQueries};
use super::bench::{ms, wait};
use super::bench_meshes::{self, BenchMesh};
use super::pipeline::instrument::{DISPATCHES, PASS_TIMESTAMPS, PassTimestamps};
use super::test_support::{self, fixtures};
use super::{
    AdjacencyQuery, BoundaryQuery, DotQuery, EdgeQuery, FaceAreaQuery, FaceCentroidQuery,
    FaceNormalQuery, FaceVertexQuery, GpuMesh, GpuMeshBuilder, GpuMeshInput, ReduceOp, ReduceQuery,
    SmoothQuery, Smoothing, VertexFaceQuery, VertexNormalQuery,
};
use crate::{Mesh, MeshBuilder};
use std::hint::black_box;
use std::time::{Duration, Instant};

pub(super) const WARMUP: usize = 2;
const SETUP_RUNS: usize = 3;
const SMOOTHING: Smoothing = Smoothing {
    step: 0.5,
    iterations: 10,
};

/// Records one query run into the encoder.
type Record<'a> = Box<dyn Fn(&mut wgpu::CommandEncoder) + 'a>;

/// Runs one setup step to completion, including waiting for the GPU.
type Step<'a> = Box<dyn FnMut() + 'a>;

/// The most compute passes one timed run can time.
const MAX_PASSES: u32 = wgpu::QUERY_SET_MAX_QUERIES / 2;

/// Start and end timestamps for every compute pass of a timed run.
pub(super) struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f64,
}

impl Timer {
    /// Returns a timer, or `None` if the device lacks timestamp queries.
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16 * u64::from(MAX_PASSES),
                usage,
                mapped_at_creation: false,
            })
        };
        Some(Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("benchmark timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * MAX_PASSES,
            }),
            resolve: buffer(
                "benchmark timestamp resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "benchmark timestamp readback",
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            ),
            period_ns: f64::from(queue.get_timestamp_period()),
        })
    }

    /// Returns the summed duration of the first `passes` timed passes, after
    /// the GPU has finished them.
    fn read(&self, device: &wgpu::Device, passes: u32) -> Duration {
        let slice = self.readback.slice(..16 * u64::from(passes));
        slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
        wait(device);
        let ticks: u64 = bytemuck::cast_slice::<u8, [u64; 2]>(&slice.get_mapped_range().unwrap())
            .iter()
            .map(|[start, end]| end.saturating_sub(*start))
            .sum();
        self.readback.unmap();
        Duration::from_nanos((ticks as f64 * self.period_ns) as u64)
    }
}

/// Times one run of `record`: wall clock, and if `timer` is given, the
/// summed GPU time of its compute passes. That sum leaves out gaps between
/// passes, and is `None` if the run recorded no passes or too many to time.
pub(super) fn sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    timer: Option<&Timer>,
    record: &dyn Fn(&mut wgpu::CommandEncoder),
) -> (Duration, Option<Duration>) {
    // Encode, submit and wait, timing every compute pass.
    let start = Instant::now();
    let mut encoder = device.create_command_encoder(&Default::default());
    if let Some(t) = timer {
        PASS_TIMESTAMPS.set(Some(PassTimestamps {
            set: t.set.clone(),
            capacity: MAX_PASSES,
            passes: 0,
        }));
    }
    record(&mut encoder);
    let passes = PASS_TIMESTAMPS
        .take()
        .map_or(0, |t| t.passes)
        .min(MAX_PASSES + 1);
    let timed = timer.filter(|_| (1..=MAX_PASSES).contains(&passes));
    if let Some(t) = timed {
        encoder.resolve_query_set(&t.set, 0..2 * passes, &t.resolve, 0);
        encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, 16 * u64::from(passes));
    }
    queue.submit([encoder.finish()]);
    wait(device);
    let wall = start.elapsed();

    // Read the timestamps after the wall clock stops.
    (wall, timed.map(|t| t.read(device, passes)))
}

/// Dispatches, GPU buffers and bind groups that one run of `record` creates.
fn counts(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    record: &dyn Fn(&mut wgpu::CommandEncoder),
) -> (u64, isize, isize) {
    let hal = || device.get_internal_counters().hal;
    let (before, dispatches) = (hal(), DISPATCHES.get());
    let mut encoder = device.create_command_encoder(&Default::default());
    record(&mut encoder);
    let after = hal();
    let result = (
        DISPATCHES.get() - dispatches,
        after.buffers.read() - before.buffers.read(),
        after.bind_groups.read() - before.bind_groups.read(),
    );
    queue.submit([encoder.finish()]);
    wait(device);
    result
}

/// Minimum, median and maximum of `times`, in milliseconds.
pub(super) fn spread(times: &mut [Duration]) -> String {
    times.sort();
    format!(
        "{} / {} / {}",
        ms(times[0]),
        ms(times[times.len() / 2]),
        ms(times[times.len() - 1])
    )
}

/// Times `run` `SETUP_RUNS` times after one warm-up run.
fn setup_spread(mut run: impl FnMut()) -> String {
    run();
    let mut times: Vec<Duration> = (0..SETUP_RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    spread(&mut times)
}

/// Shared settings and cached pipelines for every mesh.
struct Bench {
    device: wgpu::Device,
    queue: wgpu::Queue,
    timer: Option<Timer>,
    runs: usize,
    face_vertices: FaceVertexQuery,
    edges: EdgeQuery,
    centroids: FaceCentroidQuery,
    adjacency: AdjacencyQuery,
    flat: FlatQueries,
    smooth: SmoothQuery,
    vertex_faces: VertexFaceQuery,
    face_normals: FaceNormalQuery,
    face_areas: FaceAreaQuery,
    vertex_normals: VertexNormalQuery,
    boundary: BoundaryQuery,
    sum: ReduceQuery<f32>,
    dot: DotQuery,
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn gpu_baseline_benchmark() {
    if std::env::var_os("MOPMESH_BENCH").is_none() {
        println!("set MOPMESH_BENCH=1 to run the baseline benchmark");
        return;
    }
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect()
    };
    let numbers = |name, default| -> Vec<u32> {
        list(name, default)
            .iter()
            .map(|s| {
                s.parse()
                    .unwrap_or_else(|_| panic!("{name} must list integers"))
            })
            .collect()
    };
    let faces = numbers("MOPMESH_BENCH_FACES", "32768,524288,2097152");
    let patches = numbers("MOPMESH_BENCH_PATCHES", "128");
    let runs = numbers("MOPMESH_BENCH_RUNS", "11")[0] as usize;
    let bench = Bench::new(runs);

    // Each mesh, then a shuffled copy of it.
    let mut meshes: Vec<BenchMesh> = faces
        .into_iter()
        .flat_map(bench_meshes::generated)
        .collect();
    meshes.extend(
        list("MOPMESH_BENCH_OBJ", "")
            .iter()
            .map(|p| bench_meshes::load_obj(p)),
    );
    for mesh in meshes {
        let shuffled = bench_meshes::shuffled(&mesh);
        for mesh in [mesh, shuffled] {
            for &patch_size in &patches {
                bench.mesh(&mesh, patch_size);
            }
        }
    }
}

impl Bench {
    /// Opens the default adapter with its own limits and, where available,
    /// timestamps inside encoders, and prints the setup.
    fn new(runs: usize) -> Self {
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

        // Record what produced the numbers.
        let info = adapter.get_info();
        let limits = device.limits();
        println!("# MOPMesh baseline benchmark");
        if cfg!(debug_assertions) {
            println!("WARNING: debug build; timings are not representative");
        }
        println!(
            "adapter: {} ({:?}, {:?}), driver {} {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        println!(
            "limits: storage binding {} B, buffer {} B, workgroups/dim {}",
            limits.max_storage_buffer_binding_size,
            limits.max_buffer_size,
            limits.max_compute_workgroups_per_dimension
        );
        match &timer {
            Some(t) => println!(
                "GPU timestamps: per compute pass, period {} ns",
                t.period_ns
            ),
            None => println!("GPU timestamps: unavailable; wall clock only"),
        }
        println!(
            "runs: {runs} timed after {WARMUP} warm-up, patched and flat alternating; \
             setup {SETUP_RUNS} after 1 warm-up; times are min / median / max"
        );
        Self {
            face_vertices: FaceVertexQuery::new(&device),
            edges: EdgeQuery::new(&device),
            centroids: FaceCentroidQuery::new(&device),
            adjacency: AdjacencyQuery::new(&device),
            flat: FlatQueries::new(&device),
            smooth: SmoothQuery::new(&device),
            vertex_faces: VertexFaceQuery::new(&device),
            face_normals: FaceNormalQuery::new(&device),
            face_areas: FaceAreaQuery::new(&device),
            vertex_normals: VertexNormalQuery::new(&device),
            boundary: BoundaryQuery::new(&device),
            sum: ReduceQuery::new(&device, ReduceOp::Sum),
            dot: DotQuery::new(&device),
            device,
            queue,
            timer,
            runs,
        }
    }

    /// Benchmarks setup, memory and queries for one mesh and patch size.
    fn mesh(&self, input: &BenchMesh, patch_size: u32) {
        let device = &self.device;
        let build = || {
            MeshBuilder::new()
                .positions(&input.positions)
                .triangles(&input.triangles)
                .patch_size(patch_size as usize)
                .build()
                .unwrap()
        };
        let mesh = build();
        println!(
            "\n## {}: {} vertices, {} faces, {} edges, patch size {patch_size}",
            input.name,
            input.positions.len(),
            mesh.face_count(),
            mesh.edge_count()
        );
        self.setup(input, &mesh, patch_size, build);

        // Memory: connectivity per triangle and ribbon overhead.
        let gpu = GpuMesh::upload(device, &mesh).unwrap();
        let flat = FlatMesh::upload(device, &mesh);
        let faces = mesh.face_count() as f64;
        let held =
            |count: fn(&crate::Patch) -> usize| -> usize { mesh.patches().iter().map(count).sum() };
        println!(
            "memory: arena {:.1} B/face, flat {:.1} B/face; per patch-held element: \
             faces {:.3}, vertices {:.3}, edges {:.3}; {} patches",
            gpu.arena().size() as f64 / faces,
            flat.bytes() as f64 / faces,
            held(|p| p.face_ids().len()) as f64 / faces,
            held(|p| p.vertex_ids().len()) as f64 / input.positions.len() as f64,
            held(|p| p.edge_ids().len()) as f64 / mesh.edge_count() as f64,
            mesh.patches().len()
        );
        self.queries(&mesh, &gpu, &flat);
    }

    /// Times host construction and each upload or GPU build path.
    fn setup(&self, input: &BenchMesh, mesh: &Mesh, patch_size: u32, build: impl Fn() -> Mesh) {
        let device = &self.device;
        let fixture = fixtures::Fixture {
            name: "bench",
            positions: input.positions.clone(),
            triangles: input.triangles.clone(),
        };
        let (positions, triangles) = test_support::input_buffers(device, &fixture);
        let gpu_input = GpuMeshInput {
            positions: &positions,
            triangles: &triangles,
            vertex_count: fixture.positions.len() as u32,
            face_count: fixture.triangles.len() as u32,
        };
        let builder = GpuMeshBuilder::new(device).patch_size(patch_size);
        let gpu = GpuMesh::upload(device, mesh).unwrap();
        let rows: [(&str, Step); 6] = [
            ("CPU build", Box::new(|| drop(black_box(build())))),
            (
                "upload arena",
                Box::new(|| {
                    black_box(GpuMesh::upload(device, mesh).unwrap());
                    wait(device);
                }),
            ),
            (
                "upload flat (host derive + upload)",
                Box::new(|| {
                    black_box(FlatMesh::upload(device, mesh));
                    wait(device);
                }),
            ),
            (
                "upload positions",
                Box::new(|| {
                    black_box(
                        gpu.upload_attribute(device, &mesh.position_attribute())
                            .unwrap(),
                    );
                    wait(device);
                }),
            ),
            (
                "upload raw GPU-build input",
                Box::new(|| {
                    black_box(test_support::input_buffers(device, &fixture));
                    wait(device);
                }),
            ),
            (
                "GPU build (from uploaded input)",
                Box::new(|| {
                    black_box(builder.build(device, &self.queue, &gpu_input).unwrap());
                    wait(device);
                }),
            ),
        ];
        for (name, mut run) in rows {
            println!("setup {name:<36} {}", setup_spread(&mut run));
        }
    }
}

impl Bench {
    /// Times every query on `gpu`, and its flat twin on `flat` where one exists.
    fn queries(&self, mesh: &Mesh, gpu: &GpuMesh, flat: &FlatMesh) {
        let d = &self.device;
        let positions = gpu.upload_attribute(d, &mesh.position_attribute()).unwrap();

        // Inputs for the queries that start from other results.
        let mut encoder = d.create_command_encoder(&Default::default());
        let adjacency = self.adjacency.encode(d, &mut encoder, gpu).unwrap();
        let areas = self
            .face_areas
            .encode(d, &mut encoder, gpu, &positions)
            .unwrap();
        self.queue.submit([encoder.finish()]);
        wait(d);

        // Queries with an unpartitioned twin: (name, patched, flat).
        let paired: Vec<(&str, Record, Record)> = vec![
            (
                "face vertices",
                Box::new(|e| drop(self.face_vertices.encode(d, e, gpu).unwrap())),
                Box::new(|e| drop(self.flat.face_vertices(d, e, flat).unwrap())),
            ),
            (
                "edges (EV + EF)",
                Box::new(|e| drop(self.edges.encode(d, e, gpu).unwrap())),
                Box::new(|e| drop(self.flat.edges(d, e, flat).unwrap())),
            ),
            (
                "face centroids",
                Box::new(|e| drop(self.centroids.encode(d, e, gpu, &positions).unwrap())),
                Box::new(|e| drop(self.flat.centroids(d, e, flat, &positions).unwrap())),
            ),
            (
                "VV + FF + diamonds + edges",
                Box::new(|e| drop(self.adjacency.encode(d, e, gpu).unwrap())),
                Box::new(|e| drop(self.flat.adjacency(d, e, flat).unwrap())),
            ),
            (
                "adjacency + smooth x10",
                Box::new(|e| {
                    let a = self.adjacency.encode(d, e, gpu).unwrap();
                    drop(self.smooth.encode(d, e, &a, &positions, None, SMOOTHING));
                }),
                Box::new(|e| {
                    let a = self.flat.adjacency(d, e, flat).unwrap();
                    drop(self.smooth.encode(d, e, &a, &positions, None, SMOOTHING));
                }),
            ),
        ];

        // Queries that only read global results, or have no twin.
        let single: Vec<(&str, Record)> = vec![
            (
                "vertex faces",
                Box::new(|e| drop(self.vertex_faces.encode(d, e, gpu).unwrap())),
            ),
            (
                "face normals",
                Box::new(|e| drop(self.face_normals.encode(d, e, gpu, &positions).unwrap())),
            ),
            (
                "face areas",
                Box::new(|e| drop(self.face_areas.encode(d, e, gpu, &positions).unwrap())),
            ),
            (
                "vertex normals",
                Box::new(|e| drop(self.vertex_normals.encode(d, e, gpu, &positions).unwrap())),
            ),
            (
                "boundary flags",
                Box::new(|e| drop(self.boundary.encode(d, e, gpu).unwrap())),
            ),
            (
                "smooth x10 (adjacency given)",
                Box::new(|e| {
                    drop(
                        self.smooth
                            .encode(d, e, &adjacency, &positions, None, SMOOTHING)
                            .unwrap(),
                    )
                }),
            ),
            (
                "sum of face areas",
                Box::new(|e| drop(self.sum.encode(d, e, &areas).unwrap())),
            ),
            (
                "dot of face areas",
                Box::new(|e| drop(self.dot.encode(d, e, &areas, &areas).unwrap())),
            ),
        ];

        println!(
            "\n| Query | Path | Wall ms (min / med / max) | GPU ms (min / med / max) | \
             Dispatches | Buffers | Bind groups |"
        );
        println!("|---|---|---|---|---:|---:|---:|");
        for (name, patched, flat) in &paired {
            self.row(name, &[("patched", patched), ("flat", flat)]);
        }
        for (name, patched) in &single {
            self.row(name, &[("patched", patched)]);
        }
    }

    /// Prints one table row per variant, timing the variants alternately.
    fn row(&self, name: &str, variants: &[(&str, &Record)]) {
        let (d, q) = (&self.device, &self.queue);
        let counted: Vec<_> = variants.iter().map(|(_, r)| counts(d, q, r)).collect();
        let mut wall = vec![Vec::new(); variants.len()];
        let mut gpu = vec![Vec::new(); variants.len()];
        for round in 0..WARMUP + self.runs {
            for (i, (_, record)) in variants.iter().enumerate() {
                let (w, g) = sample(d, q, self.timer.as_ref(), record);
                if round >= WARMUP {
                    wall[i].push(w);
                    gpu[i].extend(g);
                }
            }
        }
        for (i, (path, _)) in variants.iter().enumerate() {
            let gpu_time = if gpu[i].is_empty() {
                "–".into()
            } else {
                spread(&mut gpu[i])
            };
            let (dispatches, buffers, bind_groups) = counted[i];
            println!(
                "| {name} | {path} | {} | {gpu_time} | {dispatches} | {buffers} | {bind_groups} |",
                spread(&mut wall[i])
            );
        }
    }
}
