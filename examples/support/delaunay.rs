//! Shared setup and GPU dispatch helpers for the Delaunay example and its tests.
use mopmesh::{
    Mesh, MeshBuilder,
    gpu::{EdgeFlipQuery, FlipStatus, GpuEditableMesh, GpuError},
};

pub fn compile_shader() -> String {
    let mut resolver = wesl::resolver::StandardResolver::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/shaders"
    ));
    resolver.add_package(&mopmesh::mopmesh::PACKAGE);
    wesl::Compiler::default()
        .with_resolver(resolver)
        .compile_module(&"package::delaunay".parse().unwrap())
        .expect("link downstream Delaunay rule")
        .to_string()
}

/// Builds a flat grid with `n` vertices along each side, distorted so it needs
/// edge flips, split into patches of about `patch_size` faces. Returns the
/// mesh, positions as `[x, y, z, 1]` values, and triangles as mesh-wide vertex
/// IDs. Use `n >= 2`; the example uses 11 with patch size 256 to fit one patch.
pub fn skewed_grid(n: u32, patch_size: usize) -> (Mesh, Vec<[f32; 4]>, Vec<[u32; 3]>) {
    let mut positions = Vec::new();
    for y in 0..n {
        for x in 0..n {
            // Offset points slightly and repeatably to avoid many equal-angle choices.
            let h = (x * 7919 + y * 104_729) % 1000;
            let jitter = (h as f32 / 1000.0 - 0.5) * 0.2;
            positions.push([x as f32 - 0.4 * y as f32 + jitter, y as f32 - jitter, 0.0]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let a = y * n + x;
            triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]);
        }
    }
    let mesh = MeshBuilder::new()
        .patch_size(patch_size)
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap();
    let vec4 = positions.iter().map(|&[x, y, z]| [x, y, z, 1.0]).collect();
    (mesh, vec4, triangles)
}

/// Runs the example's angle-based flip rule on a flat mesh.
/// Each batch considers every edge and flips a compatible subset whose
/// opposite angles exceed 180 degrees by the shader's tolerance. Repeat
/// batches to reconsider edges that could not flip together.
pub struct DelaunayFlips {
    layout: wgpu::BindGroupLayout,
    flip: wgpu::ComputePipeline,
}
impl DelaunayFlips {
    pub fn new(device: &wgpu::Device, query: &EdgeFlipQuery) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("downstream Delaunay flips"),
            source: wgpu::ShaderSource::Wgsl(compile_shader().into()),
        });

        // Bind the example positions and results at group 0, and the library pass at group 1.
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Delaunay flips"),
            entries: &[storage(0, true)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Delaunay flips"),
            bind_group_layouts: &[Some(&layout), Some(query.pass_layout())],
            immediate_size: 0,
        });
        let flip = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Delaunay flips"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("flip"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { flip, layout }
    }

    /// Records one batch and returns a GPU buffer with one `u32` status per
    /// mesh-wide edge ID. Submit the encoder before reading results. Supply
    /// `positions` as a STORAGE buffer with one `vec4<f32>` per mesh-wide vertex ID.
    /// An eligible edge that needs no flip returns FLIP_NOT_REQUESTED (zero).
    ///
    /// Returns an error if `begin_pass` exceeds device limits. A successful call
    /// increases the mesh version, even when no edges flip.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        query: &EdgeFlipQuery,
        mesh: &mut GpuEditableMesh,
        positions: &wgpu::Buffer,
    ) -> Result<wgpu::Buffer, GpuError> {
        let pass = query.begin_pass(device, mesh)?;
        let own = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Delaunay flips"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: positions.as_entire_binding(),
            }],
        });

        // One dispatch, one workgroup per patch.
        let [x, y] = pass.patch_grid();
        let mut compute = encoder.begin_compute_pass(&Default::default());
        compute.set_pipeline(&self.flip);
        compute.set_bind_group(0, &own, &[]);
        compute.set_bind_group(1, pass.bind_group(), &[]);
        compute.dispatch_workgroups(x, y, 1);
        drop(compute);
        Ok(pass.statuses().buffer().clone())
    }
}

/// Counts successful flips and requests that can be retried from one batch.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Round {
    /// Number of edges successfully flipped.
    pub flipped: u32,
    /// Number of requests that lost a shared vertex to a selected flip. A
    /// later batch can retry them.
    pub retry: u32,
}
impl Round {
    pub fn from_statuses(words: &[u32]) -> Self {
        let mut round = Self::default();
        for &word in words {
            match FlipStatus::from_word(word) {
                Some(FlipStatus::Flipped) => round.flipped += 1,
                Some(FlipStatus::LostClaim) => round.retry += 1,
                _ => {}
            }
        }
        round
    }
}

/// Copies a COPY_SRC buffer to the CPU and returns its contents as `u32` words.
/// The buffer size must be a multiple of four bytes. Blocks for GPU completion
/// and panics if the copy cannot be mapped or the wait fails or times out.
pub fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Delaunay readback"),
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
