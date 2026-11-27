//! Flips edges one at a time with the downstream single-flip kernel in
//! `tests/shaders/flip_consumer.wesl`, which places each flip in free slots.
//! `EdgeFlipQuery` rewrites a flip's own slots instead, so only these flips
//! leave empty slots below a patch's used count, for tests that check that
//! queries skip such holes.
#![allow(dead_code)]
use mopmesh::gpu::{EdgeFlipQuery, FlipStatus, GpuEditableMesh};
use wgpu::util::DeviceExt;

pub struct FreshFlips {
    pipeline: wgpu::ComputePipeline,
}
impl FreshFlips {
    pub fn new(device: &wgpu::Device) -> Self {
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
            label: Some("fresh-slot flips"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fresh-slot flips"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Tries to flip each edge of `edges` in turn, one dispatch each, and
    /// returns their statuses in order. `positions` holds one `vec4<f32>` per
    /// mesh vertex. Counts one edit on `mesh` through `query`, so results
    /// computed before are reported stale.
    pub fn flip(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        query: &EdgeFlipQuery,
        mesh: &mut GpuEditableMesh,
        positions: &wgpu::Buffer,
        edges: &[u32],
    ) -> Vec<Option<FlipStatus>> {
        // Preparing an unused pass is how a custom edit counts a version.
        query.begin_pass(device, mesh).unwrap();
        let layout = self.pipeline.get_bind_group_layout(0);
        let mut requests = Vec::new();
        let mut encoder = device.create_command_encoder(&Default::default());
        for &edge in edges {
            let request = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fresh-slot flip request"),
                contents: bytemuck::cast_slice(&[edge, 0, 1]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
            let entries: Vec<_> = [mesh.arena(), mesh.masks(), positions, &request]
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            requests.push(request);
        }
        queue.submit([encoder.finish()]);

        // Read every request's status word.
        requests
            .iter()
            .map(|request| FlipStatus::from_word(read_words(device, queue, request)[1]))
            .collect()
    }
}

fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fresh-slot flip readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
    queue.submit([encoder.finish()]);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .unwrap();
    let view = readback.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&view).to_vec()
}
