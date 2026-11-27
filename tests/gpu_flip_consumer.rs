//! Checks application shaders that use the public WESL edge-flip API.
//! The shortening rule in `tests/shaders/flip_consumer.wesl` is compared with
//! `EdgeFlipQuery`. The Delaunay example is checked by replaying its reported
//! flips on the CPU and testing the resulting triangles.
#[path = "../examples/support/delaunay.rs"]
mod delaunay;
#[path = "support/replay.rs"]
mod replay;
use mopmesh::{
    Mesh, MeshBuilder,
    gpu::{EdgeFlipQuery, EdgeQuery, EditCapacity, FaceVertexQuery, FlipStatus, GpuEditableMesh},
};
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_flip_consumer -- --ignored --nocapture"]
fn downstream_kernel_flips_only_shortening_edges_like_the_library() {
    let (device, queue) = gpu_device();
    let consumer = consumer_pipeline(&device);
    let query = EdgeFlipQuery::new(&device);
    // Shearing the grid makes one diagonal of each square shorter than the other.
    for (shear, shortens) in [(-0.3, true), (0.3, false)] {
        let (mesh, positions, edge) = sheared_grid(shear);
        let upload = || GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
        let (by_consumer, mut by_library) = (upload(), upload());
        let original = read_words(&device, &queue, by_consumer.arena());

        // The downstream kernel, with its own bindings and rule.
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("consumer positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let request = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("consumer request"),
            contents: bytemuck::cast_slice(&[edge, u32::MAX, 0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("consumer bindings"),
            layout: &consumer.get_bind_group_layout(0),
            entries: &[
                by_consumer.arena(),
                by_consumer.masks(),
                &positions,
                &request,
            ]
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let maps = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("consumer maps"),
            layout: &consumer.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 7,
                resource: by_consumer.maps().as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&consumer);
            pass.set_bind_group(0, &bindings, &[]);
            pass.set_bind_group(1, &maps, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let status = read_words(&device, &queue, &request)[1];

        if shortens {
            // The same flip through the library's own kernel.
            let mut encoder = device.create_command_encoder(&Default::default());
            let library = query
                .encode(&device, &mut encoder, &mut by_library, &[edge])
                .unwrap();
            queue.submit([encoder.finish()]);
            let library = read_words(&device, &queue, library.buffer())[edge as usize];
            assert_eq!(FlipStatus::from_word(library), Some(FlipStatus::Flipped));
            assert_eq!(FlipStatus::from_word(status), Some(FlipStatus::Flipped));

            // The consumer places the flip in free slots and the library
            // rewrites its own slots, so compare the resulting meshes by
            // global ID: every face's corners, and every edge's endpoints and
            // pair of faces. The order of an edge's two faces follows their
            // slots, so it is compared as a set.
            let topology = |mesh: &GpuEditableMesh| {
                let mut encoder = device.create_command_encoder(&Default::default());
                let faces = FaceVertexQuery::new(&device)
                    .encode(&device, &mut encoder, mesh)
                    .unwrap();
                let edges = EdgeQuery::new(&device)
                    .encode(&device, &mut encoder, mesh)
                    .unwrap();
                queue.submit([encoder.finish()]);
                let mut records = read_words(&device, &queue, edges.buffer());
                for record in records.chunks_mut(4) {
                    record[2..].sort();
                }
                (read_words(&device, &queue, faces.buffer()), records)
            };
            assert_eq!(topology(&by_consumer), topology(&by_library));
            assert_ne!(read_words(&device, &queue, by_consumer.arena()), original);
        } else {
            assert_eq!(status, 0, "kept: the flip would not shorten the edge");
            assert_eq!(read_words(&device, &queue, by_consumer.arena()), original);
        }
    }
}

/// Checks that the example reports flips with no shared vertices per batch,
/// and that replaying those flips satisfies its angle rule within 64 batches.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_flip_consumer -- --ignored --nocapture"]
fn delaunay_example_settles_on_a_delaunay_mesh_with_disjoint_flips() {
    let (device, queue) = gpu_device();
    let (mesh, positions, mut triangles) = delaunay::skewed_grid(11, 256);
    let mut gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("delaunay positions"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &query);

    // Track edge endpoints on the CPU so reported GPU flips can be replayed
    // and checked without using the GPU mesh storage as the reference.
    let mut endpoints = replay::edge_endpoints(&mesh);
    assert!(
        violations(&triangles, &positions) > 0,
        "the fixture should start with non-Delaunay edges"
    );
    let mut flipped_total = 0;
    for batch in 1..=64 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let status = flips
            .encode(&device, &mut encoder, &query, &mut gpu, &buffer)
            .unwrap();
        queue.submit([encoder.finish()]);
        let words = delaunay::read_words(&device, &queue, &status);
        let round = delaunay::Round::from_statuses(&words);

        // A vertex may belong to at most one successful flip in this batch.
        let mut corners = std::collections::HashSet::new();
        for (id, &word) in words.iter().enumerate() {
            if FlipStatus::from_word(word) != Some(FlipStatus::Flipped) {
                continue;
            }
            let quad = replay::replay(&mut triangles, &mut endpoints, id);
            for v in quad {
                assert!(corners.insert(v), "two flips in one batch share vertex {v}");
            }
        }
        flipped_total += round.flipped;
        if round == delaunay::Round::default() {
            assert!(flipped_total > 0);
            assert_eq!(violations(&triangles, &positions), 0);
            // Each batch selects a maximal set of compatible flips, so this
            // grid settles in fewer batches than the 11 that one selection
            // round per batch took (8 on the machine this was measured on).
            assert!(batch < 11, "settled only after {batch} batches");
            return;
        }
    }
    panic!("flipping did not settle within 64 rounds");
}

/// The outcome of `settle`.
struct Settled {
    triangles: Vec<[u32; 3]>,
    positions: Vec<[f32; 4]>,
    /// Endpoints of each edge ID, after the replayed flips.
    endpoints: Vec<[u32; 2]>,
    /// The last batch's status words and counts.
    statuses: Vec<u32>,
    last: delaunay::Round,
    batches: usize,
}

/// Runs the example's rule, cross-patch rounds included, on a `side` x
/// `side` skewed grid until a batch flips and retries nothing, replaying
/// each reported flip on the CPU.
fn settle(side: u32, patch_size: usize) -> Settled {
    let (device, queue) = gpu_device();
    let (mesh, positions, mut triangles) = delaunay::skewed_grid(side, patch_size);
    let mut gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("delaunay positions"),
        contents: bytemuck::cast_slice(&positions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let query = EdgeFlipQuery::new(&device);
    let flips = delaunay::DelaunayFlips::new(&device, &query);
    let mut endpoints = replay::edge_endpoints(&mesh);
    for batch in 1..=64 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let status = flips
            .encode(&device, &mut encoder, &query, &mut gpu, &buffer)
            .unwrap();
        queue.submit([encoder.finish()]);
        let words = delaunay::read_words(&device, &queue, &status);
        let round = delaunay::Round::from_statuses(&words);

        // A vertex may belong to at most one successful flip in this batch.
        let mut corners = std::collections::HashSet::new();
        for (id, &word) in words.iter().enumerate() {
            if FlipStatus::from_word(word) == Some(FlipStatus::Flipped) {
                for v in replay::replay(&mut triangles, &mut endpoints, id) {
                    assert!(corners.insert(v), "two flips in one batch share vertex {v}");
                }
            }
        }
        if round.flipped == 0 && round.retry == 0 {
            return Settled {
                triangles,
                positions,
                endpoints,
                statuses: words,
                last: round,
                batches: batch,
            };
        }
    }
    panic!("flipping did not settle within 64 batches");
}

/// Counts edges that join faces in different patches and fail the
/// example's angle test.
fn crossing_violations(side: u32, patch_size: usize) -> usize {
    let (mesh, positions, triangles) = delaunay::skewed_grid(side, patch_size);
    let mut patch_of = vec![0; triangles.len()];
    for p in mesh.patches() {
        for (&f, owner) in p.face_ids().iter().zip(p.face_owners()) {
            if owner.patch == p.id() {
                patch_of[f as usize] = p.id();
            }
        }
    }
    let mut faces_of: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
    for (f, t) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            faces_of.entry((a.min(b), a.max(b))).or_default().push(f);
        }
    }
    faces_of
        .values()
        .filter(|fs| fs.len() == 2 && patch_of[fs[0]] != patch_of[fs[1]])
        .filter(|fs| violations(&[triangles[fs[0]], triangles[fs[1]]], &positions) > 0)
        .count()
}

/// At the default patch size and capacity, flips across patch borders let
/// the example settle multi-patch grids completely.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_flip_consumer -- --ignored --nocapture"]
fn delaunay_example_settles_across_patch_borders() {
    for side in [65, 129] {
        assert!(
            crossing_violations(side, 512) > 0,
            "{side}: some violation crosses patches"
        );
        let s = settle(side, 512);
        assert_eq!(s.last.out_of_room, 0, "{side}: no patch runs out of room");
        assert_eq!(violations(&s.triangles, &s.positions), 0, "{side}: settled");
        println!(
            "{side}x{side} at patch size 512: settled in {} batches",
            s.batches
        );
    }
}

/// Small patches can run out of room, which must be reported rather than
/// look settled, and must leave a valid mesh.
#[test]
#[ignore = "requires a GPU adapter; run cargo test --test gpu_flip_consumer -- --ignored --nocapture"]
fn delaunay_example_reports_patches_out_of_room() {
    let s = settle(65, 16);
    let left = violating_edges(&s.triangles, &s.positions);
    println!(
        "65x65 at patch size 16: {} batches, {} violations, {:?}",
        s.batches,
        left.len(),
        s.last
    );
    assert!(s.last.out_of_room > 0, "some patch runs out of room");
    assert!(
        !left.is_empty(),
        "edges left out of room still break the rule"
    );

    // Every edge still breaking the rule reported OutOfCapacity.
    for (id, &[a, b]) in s.endpoints.iter().enumerate() {
        if left.contains(&(a.min(b), a.max(b))) {
            assert_eq!(
                FlipStatus::from_word(s.statuses[id]),
                Some(FlipStatus::OutOfCapacity),
                "edge {id}"
            );
        }
    }
}

/// Counts interior edges that fail the example's Delaunay angle test:
/// the two opposite angles sum to more than 180 degrees plus 0.001 radians.
/// Positions are indexed by mesh-wide vertex ID; triangles must have
/// consistent orientation and nonzero edge lengths.
fn violations(triangles: &[[u32; 3]], positions: &[[f32; 4]]) -> usize {
    violating_edges(triangles, positions).len()
}

/// The edges counted by `violations`, as `(a, b)` with `a < b`.
fn violating_edges(
    triangles: &[[u32; 3]],
    positions: &[[f32; 4]],
) -> std::collections::HashSet<(u32, u32)> {
    let angle = |at: u32, p: u32, q: u32| {
        let o = positions[at as usize];
        let (p, q) = (positions[p as usize], positions[q as usize]);
        let u = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        let w = [q[0] - o[0], q[1] - o[1], q[2] - o[2]];
        let dot = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
        let len = |x: [f32; 3]| (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        (dot / (len(u) * len(w))).clamp(-1.0, 1.0).acos()
    };
    // Store each triangle's opposite angle under its directed edge. The
    // reverse edge supplies the angle in the neighboring triangle.
    let mut opposite = std::collections::HashMap::new();
    for t in triangles {
        for k in 0..3 {
            let (a, b, c) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
            opposite.insert((a, b), angle(c, a, b));
        }
    }
    opposite
        .iter()
        .filter(|&(&(a, b), &theta)| {
            a < b
                && opposite
                    .get(&(b, a))
                    .is_some_and(|&other| theta + other > std::f32::consts::PI + 1e-3)
        })
        .map(|(&edge, _)| edge)
        .collect()
}

/// A 9 x 9 grid in one patch with x shifted by `shear * y`, its positions as
/// vec4, and the global ID of the diagonal in the square at (4, 4).
fn sheared_grid(shear: f32) -> (Mesh, Vec<[f32; 4]>, u32) {
    let n = 9u32;
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    for y in 0..n {
        for x in 0..n {
            positions.push([x as f32 + shear * y as f32, y as f32, 0.0]);
        }
    }
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let a = y * n + x;
            triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]);
        }
    }
    let mesh = MeshBuilder::new()
        .patch_size(128)
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap();
    let diagonal = [4 * n + 5, 5 * n + 4];
    let p = &mesh.patches()[0];
    let edge = *p
        .edge_ids()
        .iter()
        .zip(p.edge_vertices())
        .find(|(_, ev)| {
            let [a, b] = ev.map(|v| p.vertex_ids()[v as usize]);
            [a.min(b), a.max(b)] == diagonal
        })
        .unwrap()
        .0;
    let vec4 = positions.iter().map(|&[x, y, z]| [x, y, z, 1.0]).collect();
    (mesh, vec4, edge)
}

fn consumer_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let mut resolver = wesl::resolver::StandardResolver::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/shaders"
    ));
    resolver.add_package(&mopmesh::mopmesh::PACKAGE);
    let wgsl = wesl::Compiler::default()
        .with_resolver(resolver)
        .compile_module(&"package::flip_consumer".parse().unwrap())
        .unwrap()
        .to_string();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flip consumer"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flip consumer"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
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
    let words = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    words
}

fn gpu_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    pollster::block_on(adapter.request_device(&Default::default())).unwrap()
}
