//! Release-mode benchmark of the GPU build against the CPU build plus upload,
//! and of query times on each patch split. Not a pass/fail test.
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --lib gpu_build_benchmark -- --ignored --nocapture`.
//! Without `MOPMESH_BENCH` it returns at once, so `--include-ignored` test runs
//! stay fast.
//! Grid sizes come from `MOPMESH_BENCH_GRIDS` (comma-separated vertex counts per
//! side), default `128,512,1024`. Times are wall clock, median of 7 runs after 2
//! warm-up runs, and include waiting for the GPU.
use super::build::{FirstPhase, GpuBuilder, GpuMeshInput, GpuSplit, Summary};
use super::test_support::{self, fixtures};
use super::{AdjacencyQuery, FaceCentroidQuery, GpuMesh, VertexFaceQuery};
use crate::{Mesh, MeshBuilder};
use std::hint::black_box;
use std::time::{Duration, Instant};

const WARMUP: usize = 2;
const RUNS: usize = 7;
const PATCH_SIZE: u32 = 128;

fn median(mut run: impl FnMut()) -> Duration {
    for _ in 0..WARMUP {
        run();
    }
    let mut times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[RUNS / 2]
}

fn wait(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(120)),
        })
        .unwrap();
}

fn ms(time: Duration) -> String {
    format!("{:.2} ms", time.as_secs_f64() * 1e3)
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn gpu_build_benchmark() {
    if std::env::var_os("MOPMESH_BENCH").is_none() {
        println!("set MOPMESH_BENCH=1 to run the GPU build benchmark");
        return;
    }
    // The adapter's own limits, so the largest grids fit.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    println!(
        "GPU: {} ({:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );
    let builder = GpuBuilder::new(&device);
    let centroids = FaceCentroidQuery::new(&device);
    let vertex_faces = VertexFaceQuery::new(&device);
    let adjacency = AdjacencyQuery::new(&device);

    let grids: Vec<u32> = std::env::var("MOPMESH_BENCH_GRIDS")
        .unwrap_or_else(|_| "128,512,1024".into())
        .split(',')
        .map(|s| s.trim().parse().expect("grid sizes must be integers"))
        .collect();
    for n in grids {
        let (positions, triangles) = fixtures::grid(n);
        let fixture = fixtures::Fixture {
            name: "grid",
            positions,
            triangles,
        };
        let faces = fixture.triangles.len();
        println!(
            "\ngrid {n}x{n}: {} vertices, {faces} faces, patch size {PATCH_SIZE}",
            fixture.positions.len()
        );

        // CPU build (breadth-first) and upload of the arena and positions.
        let cpu_build = |builder: MeshBuilder| {
            builder
                .positions(&fixture.positions)
                .triangles(&fixture.triangles)
                .patch_size(PATCH_SIZE as usize)
                .build()
                .unwrap()
        };
        let build_time = median(|| {
            black_box(cpu_build(MeshBuilder::new()));
        });
        let mesh = cpu_build(MeshBuilder::new());
        let upload_time = median(|| {
            let gpu = GpuMesh::upload(&device, &mesh).unwrap();
            black_box(
                gpu.upload_vertex_attribute(&device, &mesh.position_attribute())
                    .unwrap(),
            );
            wait(&device);
        });
        println!("CPU build (breadth-first)      {}", ms(build_time));
        println!("upload arena + positions       {}", ms(upload_time));
        println!(
            "CPU path total                 {}",
            ms(build_time + upload_time)
        );

        // Raw input upload, then the GPU build from those buffers.
        let input_time = median(|| {
            black_box(test_support::input_buffers(&device, &fixture));
            wait(&device);
        });
        println!("upload raw inputs              {}", ms(input_time));
        let (positions, triangles) = test_support::input_buffers(&device, &fixture);
        let input = GpuMeshInput {
            positions: &positions,
            triangles: &triangles,
            vertex_count: fixture.positions.len() as u32,
            face_count: faces as u32,
        };
        let mut splits = Vec::new();
        for method in [GpuSplit::Morton, GpuSplit::InputOrder] {
            let time = median(|| {
                black_box(
                    builder
                        .build(&device, &queue, &input, method, PATCH_SIZE)
                        .unwrap(),
                );
                wait(&device);
            });
            let build = builder
                .build(&device, &queue, &input, method, PATCH_SIZE)
                .unwrap();
            let encoder = device.create_command_encoder(&Default::default());
            let mut face_patch =
                test_support::read_words(&device, &queue, encoder, &build.split.face_patch);
            face_patch.truncate(faces);
            println!(
                "GPU build ({method:?}){:pad$}{}   GPU path total {}",
                "",
                ms(time),
                ms(input_time + time),
                pad = 20 - format!("{method:?}").len()
            );
            splits.push((method, face_patch, build.summary));
        }
        stage_breakdown(&device, &queue, &builder, &input);

        // Ribbon overhead and query times on each split.
        println!("split           faces held/face  centroids  vertex-face  VV+FF+edges");
        let report = |name: &str, mesh: &Mesh| {
            let held: usize = mesh.patches().iter().map(|p| p.face_ids().len()).sum();
            let gpu = GpuMesh::upload(&device, mesh).unwrap();
            let positions = gpu
                .upload_vertex_attribute(&device, &mesh.position_attribute())
                .unwrap();
            let query = |encode: &dyn Fn(&mut wgpu::CommandEncoder)| {
                median(|| {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    wait(&device);
                })
            };
            let c = query(&|e| {
                black_box(centroids.encode(&device, e, &gpu, &positions).unwrap());
            });
            let v = query(&|e| {
                black_box(vertex_faces.encode(&device, e, &gpu).unwrap());
            });
            let a = query(&|e| {
                black_box(adjacency.encode(&device, e, &gpu).unwrap());
            });
            println!(
                "{name:<15} {:>15.3}  {:>9}  {:>11}  {:>11}",
                held as f64 / faces as f64,
                ms(c),
                ms(v),
                ms(a)
            );
        };
        report("breadth-first", &mesh);
        for (method, face_patch, summary) in &splits {
            // The CPU builder given the GPU's split makes the same arena.
            let mesh = cpu_build(MeshBuilder::new().face_patches(face_patch));
            let held: usize = mesh.patches().iter().map(|p| p.face_ids().len()).sum();
            assert_eq!(held, summary.face_members as usize);
            report(&format!("GPU {method:?}"), &mesh);
        }
    }
}

/// Times each GPU build stage on its own, submitting and waiting after each.
fn stage_breakdown(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    builder: &GpuBuilder,
    input: &GpuMeshInput,
) {
    let mut times = Vec::new();
    let mut stage = |name: &'static str, record: &mut dyn FnMut(&mut wgpu::CommandEncoder)| {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        record(&mut encoder);
        queue.submit([encoder.finish()]);
        wait(device);
        times.push((name, start.elapsed()));
    };
    let method = GpuSplit::Morton;
    let mut elements = None;
    stage("elements", &mut |e| {
        elements = Some(builder.encode_elements(device, e, input).unwrap());
    });
    let elements = elements.unwrap();
    let mut connectivity = None;
    stage("connectivity", &mut |e| {
        connectivity = Some(
            builder
                .encode_connectivity(device, e, input, &elements)
                .unwrap(),
        );
    });
    let connectivity = connectivity.unwrap();
    let mut split = None;
    stage("Morton split", &mut |e| {
        split = Some(
            builder
                .encode_split(device, e, input, method, PATCH_SIZE)
                .unwrap(),
        );
    });
    let split = split.unwrap();
    let partition = split.partition();
    let mut counts = None;
    stage("member counts", &mut |e| {
        counts = Some(
            builder
                .encode_member_counts(device, e, input, &elements, &connectivity, &partition)
                .unwrap(),
        );
    });
    let counts = counts.unwrap();
    let start = Instant::now();
    let encoder = device.create_command_encoder(&Default::default());
    let words = test_support::read_words(device, queue, encoder, &counts.summary);
    let summary = Summary::decode(words.try_into().unwrap());
    let readback = start.elapsed();
    let first = FirstPhase {
        elements: &elements,
        connectivity: &connectivity,
        partition: split.partition(),
        counts: &counts,
        summary: &summary,
    };
    let mut patches = None;
    stage("patch contents", &mut |e| {
        patches = Some(builder.encode_patches(device, e, input, &first).unwrap());
    });
    let patches = patches.unwrap();
    stage("arena", &mut |e| {
        black_box(
            builder
                .encode_arena(device, e, input, &partition, &summary, &patches)
                .unwrap(),
        );
    });
    times.insert(4, ("summary readback", readback));
    let line: Vec<String> = times
        .iter()
        .map(|(n, t)| format!("{n} {}", ms(*t)))
        .collect();
    println!("  stages (one run, synced): {}", line.join(", "));
}
