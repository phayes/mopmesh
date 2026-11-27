use super::pipeline::{self, workgroups};
use super::{GpuError, GpuTopology, GpuVertexAttribute, MeshAdjacency};
use wgpu::util::DeviceExt;

/// Settings for [`SmoothQuery`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Smoothing {
    /// How far each step moves a vertex toward the average of its
    /// neighbors: 0 leaves it in place, 1 moves it all the way. Values
    /// between 0 and 1 are typical; larger values can overshoot.
    pub step: f32,
    /// The number of steps. Each step reads the previous step's positions.
    pub iterations: u32,
}

/// The kernel's uniform parameters, laid out as in `smooth.wesl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    vertices: u32,
    step: f32,
    has_fixed: u32,
    pad: u32,
}

/// Smooths vertex positions with uniform Laplacian steps: each step moves
/// every free vertex part of the way toward the average position of the
/// vertices it shares an edge with. Cache one per device.
///
/// Repeated smoothing shrinks the mesh and flattens detail. Keep boundary
/// vertices in place by passing the `vertices` flags from
/// [`BoundaryQuery`](super::BoundaryQuery) as `fixed`; otherwise an open
/// mesh's boundary also shrinks. Neighbors are summed in an unspecified
/// order, so results may differ in the last bits between runs.
pub struct SmoothQuery {
    pipeline: wgpu::ComputePipeline,
}
impl SmoothQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh smoothing",
            include_str!(concat!(env!("OUT_DIR"), "/smooth.wgsl")),
        );
        Self {
            pipeline: pipeline::compute_pipeline(
                device,
                "MOPMesh smoothing",
                &module,
                "main",
                None,
            ),
        }
    }

    /// Records `settings.iterations` smoothing steps and returns the new
    /// positions, leaving `positions` unchanged.
    ///
    /// `adjacency` supplies each vertex's neighbors and must have been
    /// computed from `mesh` as it is now. `positions` holds `[x, y, z, w]` per
    /// vertex; `w` is copied unchanged. A vertex whose `fixed` flag is
    /// non-zero keeps its position; pass `None` to move every vertex. Vertices
    /// with no neighbors never move, and unused vertex IDs keep their input
    /// values. The result has no topology stamp, so it stays usable after
    /// later edits.
    ///
    /// Returns [`GpuError::MeshMismatch`] unless all inputs belong to `mesh`,
    /// [`GpuError::Stale`] if `adjacency` or a computed `fixed` attribute
    /// predates an edit of `mesh`, [`GpuError::InvalidBuffer`] if `positions`
    /// or `fixed` doesn't have one value per vertex ID, or if `iterations` is
    /// zero and `positions` lacks `COPY_SRC` usage (the result is then a
    /// copy), or [`GpuError::LimitsExceeded`] if a buffer or dispatch is too
    /// large. Errors occur before allocating or recording commands.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &impl GpuTopology,
        adjacency: &MeshAdjacency,
        positions: &GpuVertexAttribute<[f32; 4]>,
        fixed: Option<&GpuVertexAttribute<u32>>,
        settings: Smoothing,
    ) -> Result<GpuVertexAttribute<[f32; 4]>, GpuError> {
        // Validate the inputs against the mesh's current connectivity, then
        // usage and sizes, before allocating.
        const RESOURCE: &str = "smoothing";
        let current = mesh.topology();
        adjacency.topology().check(current)?;
        positions.check_for(current)?;
        if let Some(fixed) = fixed {
            fixed.check_for(current)?;
        }
        let vertices = mesh.vertex_capacity();
        if positions.len() != vertices || fixed.is_some_and(|f| f.len() != vertices) {
            return Err(GpuError::InvalidBuffer {
                resource: "vertex attribute",
            });
        }
        let mesh = current.mesh_id();
        if settings.iterations == 0
            && !positions
                .buffer()
                .usage()
                .contains(wgpu::BufferUsages::COPY_SRC)
        {
            return Err(GpuError::InvalidBuffer {
                resource: "vertex attribute",
            });
        }
        let limits = device.limits();
        let count = positions.len();
        GpuVertexAttribute::<[f32; 4]>::check_output(&limits, count, RESOURCE)?;
        let groups = workgroups(&limits, count, RESOURCE)?;
        let result = GpuVertexAttribute::output(device, mesh, count, "MOPMesh smoothed positions");
        if count == 0 {
            return Ok(result);
        }
        if settings.iterations == 0 {
            encoder.copy_buffer_to_buffer(
                positions.buffer(),
                0,
                result.buffer(),
                0,
                u64::from(count) * 16,
            );
            return Ok(result);
        }

        // Steps alternate between the result and a scratch buffer, arranged
        // so the last step writes the result.
        let scratch = GpuVertexAttribute::<[f32; 4]>::output(
            device,
            mesh,
            count,
            "MOPMesh smoothing scratch",
        );
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh smoothing parameters"),
            contents: bytemuck::bytes_of(&Params {
                vertices: count,
                step: settings.step,
                has_fixed: u32::from(fixed.is_some()),
                pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Without flags, bind the neighbor rows in their place; they are never read.
        let flags = fixed.map_or(adjacency.vertex_neighbors(), |f| f.buffer());
        let layout = self.pipeline.get_bind_group_layout(0);
        let mut source = positions.buffer();
        for i in 0..settings.iterations {
            let writes_result = (settings.iterations - 1 - i).is_multiple_of(2);
            let destination = if writes_result {
                result.buffer()
            } else {
                scratch.buffer()
            };
            let bindings = pipeline::bind_buffers(
                device,
                "MOPMesh smoothing bindings",
                &layout,
                &[
                    &params,
                    adjacency.vertex_neighbors(),
                    flags,
                    source,
                    destination,
                ],
            );
            pipeline::dispatch(
                encoder,
                "MOPMesh smoothing",
                &self.pipeline,
                &bindings,
                groups,
            );
            source = destination;
        }
        Ok(result)
    }
}
