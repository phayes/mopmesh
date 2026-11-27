//! Runs the downstream flip kernel in tests/shaders/flip_consumer.wesl, which
//! flips an edge with the public WESL API only when that shortens it, and
//! checks that it edits the mesh exactly as `EdgeFlipQuery` does.
use mopmesh::{
    Mesh, MeshBuilder,
    gpu::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh},
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
            contents: bytemuck::cast_slice(&[edge, u32::MAX]),
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
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&consumer);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let status = read_words(&device, &queue, &request)[1];

        if shortens {
            // The same flip through the library's own kernel.
            let mut encoder = device.create_command_encoder(&Default::default());
            let library = query.encode(&device, &mut encoder, &mut by_library, edge);
            queue.submit([encoder.finish()]);
            let library = read_words(&device, &queue, library.buffer())[0];
            assert_eq!(FlipStatus::from_word(library), Some(FlipStatus::Flipped));
            assert_eq!(FlipStatus::from_word(status), Some(FlipStatus::Flipped));
            for (ours, theirs) in [
                (by_consumer.arena(), by_library.arena()),
                (by_consumer.masks(), by_library.masks()),
            ] {
                assert_eq!(
                    read_words(&device, &queue, ours),
                    read_words(&device, &queue, theirs)
                );
            }
            assert_ne!(read_words(&device, &queue, by_consumer.arena()), original);
        } else {
            assert_eq!(status, 0, "kept: the flip would not shorten the edge");
            assert_eq!(read_words(&device, &queue, by_consumer.arena()), original);
        }
    }
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
