//! Approximate wall-clock comparison of MOPMesh GPU queries against Alum on the CPU.
//!
//! Run: cargo run --release --example bench_alum [grid sizes...]
//!
//! Each grid size `n` builds an n x n vertex grid with 2 (n - 1)^2 triangles.
//! GPU times cover encode, submit and wait (no readback), so they include the
//! per-call buffer allocation each query makes. CPU times are single-threaded
//! Alum traversals that produce the same data in plain `Vec`s. Each figure is
//! the median of several runs after warm-up.
use alum::{Handle, HasIterators, HasTopology, PolyMeshF32, Vec3};
use mopmesh::{
    Mesh, MeshBuilder,
    gpu::{
        AdjacencyQuery, FaceCentroidQuery, GpuMesh, GpuMeshBuilder, GpuMeshInput, VertexFaceQuery,
    },
};
use std::hint::black_box;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WARMUP: usize = 2;
const RUNS: usize = 7;

fn main() {
    let sizes: Vec<u32> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("grid sizes must be integers"))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![128, 512, 1024]
    } else {
        sizes
    };

    // Request the adapter's own limits so the largest grids fit.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .unwrap();
    println!(
        "GPU: {}  ({:?})",
        adapter.get_info().name,
        adapter.get_info().backend
    );
    let centroid_query = FaceCentroidQuery::new(&device);
    let vf_query = VertexFaceQuery::new(&device);
    let adjacency_query = AdjacencyQuery::new(&device);
    let gpu_builder = GpuMeshBuilder::new(&device);

    for n in sizes {
        let (positions, triangles) = grid(n);
        println!(
            "\ngrid {n}x{n}: {} vertices, {} faces",
            positions.len(),
            triangles.len()
        );
        row("workload", "mopmesh", "alum", "alum/mopmesh");

        // Host mesh construction.
        let build = median(|| {
            black_box(build_mopmesh(&positions, &triangles));
        });
        let alum_build = median(|| {
            black_box(build_alum(&positions, &triangles));
        });
        compare("build (CPU)", build, alum_build);

        // GPU construction, including uploading the raw positions and triangles.
        let gpu_build = median(|| {
            black_box(build_on_gpu(
                &device,
                &queue,
                &gpu_builder,
                &positions,
                &triangles,
            ));
            wait(&device, &queue);
        });
        compare("build (GPU, + input)", gpu_build, alum_build);

        // GPU upload has no Alum counterpart.
        let mesh = build_mopmesh(&positions, &triangles);
        let upload = median(|| {
            let gpu = GpuMesh::upload(&device, &mesh).unwrap();
            black_box(
                gpu.upload_attribute(&device, &mesh.position_attribute())
                    .unwrap(),
            );
            wait(&device, &queue);
        });
        row("upload mesh+positions", &ms(upload), "-", "-");

        let gpu = GpuMesh::upload(&device, &mesh).unwrap();
        let gpu_positions = gpu
            .upload_attribute(&device, &mesh.position_attribute())
            .unwrap();
        let reference = build_alum(&positions, &triangles);

        // Face centroids.
        let gpu_time = gpu_median(&device, &queue, |encoder| {
            black_box(
                centroid_query
                    .encode(&device, encoder, &gpu, &gpu_positions)
                    .unwrap(),
            );
        });
        let cpu_time = median(|| {
            black_box(alum_centroids(&reference));
        });
        compare("face centroids", gpu_time, cpu_time);

        // Vertex-to-face CSR.
        let gpu_time = gpu_median(&device, &queue, |encoder| {
            black_box(vf_query.encode(&device, encoder, &gpu).unwrap());
        });
        let cpu_time = median(|| {
            black_box(alum_vertex_faces(&reference));
        });
        compare("vertex-face CSR", gpu_time, cpu_time);

        // Vertex neighbors, face neighbors and edge endpoints together.
        let gpu_time = gpu_median(&device, &queue, |encoder| {
            black_box(adjacency_query.encode(&device, encoder, &gpu).unwrap());
        });
        let cpu_time = median(|| {
            black_box(alum_adjacency(&reference));
        });
        compare("VV + FF + edges", gpu_time, cpu_time);
    }
}

// Mesh generation and construction.

fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let positions = (0..n)
        .flat_map(|y| (0..n).map(move |x| [x as f32, y as f32, 0.]))
        .collect();
    let triangles = (0..n - 1)
        .flat_map(|y| {
            (0..n - 1).flat_map(move |x| {
                let a = y * n + x;
                [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
            })
        })
        .collect();
    (positions, triangles)
}

fn build_mopmesh(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> Mesh {
    MeshBuilder::new()
        .positions(positions)
        .triangles(triangles)
        .build()
        .unwrap()
}

/// Uploads the grid in the vec4 layout [`GpuMeshBuilder`] reads and builds it
/// on the GPU. Grids are never empty, so the buffers need no padding.
fn build_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    builder: &GpuMeshBuilder,
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> GpuMesh {
    let positions4: Vec<[f32; 4]> = positions.iter().map(|&[x, y, z]| [x, y, z, 0.]).collect();
    let triangles4: Vec<[u32; 4]> = triangles.iter().map(|&[a, b, c]| [a, b, c, 0]).collect();
    let create = |label, contents| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let position_buffer = create("bench positions", bytemuck::cast_slice(&positions4));
    let triangle_buffer = create("bench triangles", bytemuck::cast_slice(&triangles4));
    let input = GpuMeshInput {
        positions: &position_buffer,
        triangles: &triangle_buffer,
        vertex_count: positions.len() as u32,
        face_count: triangles.len() as u32,
    };
    builder.build(device, queue, &input).unwrap()
}

fn build_alum(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> PolyMeshF32 {
    let mut mesh = PolyMeshF32::with_capacity(
        positions.len(),
        positions.len() + triangles.len(),
        triangles.len(),
    );
    for &[x, y, z] in positions {
        mesh.add_vertex(Vec3(x, y, z)).unwrap();
    }
    for tri in triangles {
        mesh.add_face(&tri.map(Into::into)).unwrap();
    }
    mesh
}

// Alum counterparts producing the same data as the GPU queries.

fn alum_centroids(mesh: &PolyMeshF32) -> Vec<[f32; 4]> {
    let points = mesh.points();
    let points = points.try_borrow().unwrap();
    mesh.faces()
        .map(|f| {
            let Vec3(x, y, z) = mesh.calc_face_centroid(f, &points);
            [x, y, z, 0.]
        })
        .collect()
}

fn csr(rows: impl Iterator<Item = impl Iterator<Item = u32>>) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0];
    let mut entries = Vec::new();
    for row in rows {
        entries.extend(row);
        offsets.push(entries.len() as u32);
    }
    (offsets, entries)
}

fn alum_vertex_faces(mesh: &PolyMeshF32) -> (Vec<u32>, Vec<u32>) {
    csr(mesh
        .vertices()
        .map(|v| mesh.vf_ccw_iter(v).map(|f| f.index())))
}

// Vertex-neighbor CSR, three face neighbors per face, and edge endpoints.
type Adjacency = ((Vec<u32>, Vec<u32>), Vec<[u32; 3]>, Vec<[u32; 2]>);

fn alum_adjacency(mesh: &PolyMeshF32) -> Adjacency {
    let vv = csr(mesh
        .vertices()
        .map(|v| mesh.vv_ccw_iter(v).map(|u| u.index())));
    let ff = mesh
        .faces()
        .map(|f| {
            let mut row = [u32::MAX; 3];
            for (slot, g) in row.iter_mut().zip(mesh.ff_ccw_iter(f)) {
                *slot = g.index();
            }
            row
        })
        .collect();
    let edges = mesh
        .edges()
        .map(|e| {
            let (a, b) = e.vertices(mesh);
            [a.index(), b.index()]
        })
        .collect();
    (vv, ff, edges)
}

// Timing.

fn median(mut f: impl FnMut()) -> Duration {
    for _ in 0..WARMUP {
        f();
    }
    let mut times: Vec<_> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[RUNS / 2]
}

fn gpu_median(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut record: impl FnMut(&mut wgpu::CommandEncoder),
) -> Duration {
    median(|| {
        let mut encoder = device.create_command_encoder(&Default::default());
        record(&mut encoder);
        queue.submit([encoder.finish()]);
        wait(device, queue);
    })
}

fn wait(device: &wgpu::Device, queue: &wgpu::Queue) {
    queue.submit([]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        })
        .unwrap();
}

// Output.

fn ms(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1e3)
}

fn row(name: &str, a: &str, b: &str, c: &str) {
    println!("  {name:<24}{a:>12}{b:>12}{c:>14}");
}

fn compare(name: &str, mopmesh: Duration, alum: Duration) {
    let ratio = alum.as_secs_f64() / mopmesh.as_secs_f64();
    row(name, &ms(mopmesh), &ms(alum), &format!("{ratio:.1}x"));
}
