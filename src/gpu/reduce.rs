use super::pipeline::{self, WORKGROUP_SIZE, check_binding};
use super::{GpuAttribute, GpuError, GpuValue};
use crate::{Element, MeshId};
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

/// Scalar attribute values that [`ReduceQuery`] can combine: `f32`, `u32`
/// and `i32`. This trait is sealed.
pub trait ReduceValue: GpuValue + sealed::Sealed {
    #[doc(hidden)]
    const TYPE: f64;
}
impl ReduceValue for f32 {
    const TYPE: f64 = 0.;
}
impl ReduceValue for u32 {
    const TYPE: f64 = 1.;
}
impl ReduceValue for i32 {
    const TYPE: f64 = 2.;
}

/// How [`ReduceQuery`] combines an attribute's values into one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    /// The sum of all values. Integer sums wrap on overflow. An empty
    /// attribute sums to zero.
    Sum,
    /// The smallest value. An empty attribute gives +infinity for `f32`,
    /// or the largest `u32`/`i32`.
    Min,
    /// The largest value. An empty attribute gives -infinity for `f32`,
    /// or zero for `u32` and the smallest `i32`.
    Max,
}

/// A single reduced value of type `T` on the GPU, as one `T` at byte 0 of
/// [`buffer`](Self::buffer). The buffer supports `STORAGE | COPY_SRC | COPY_DST`.
pub struct GpuReduction<T: ReduceValue> {
    buffer: wgpu::Buffer,
    mesh: MeshId,
    marker: PhantomData<T>,
}
impl<T: ReduceValue> GpuReduction<T> {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    /// Returns the ID of the mesh whose attribute was reduced.
    pub fn mesh_id(&self) -> MeshId {
        self.mesh
    }
}

/// One level of the reduction tree: `count` inputs, read from the attribute
/// (first level) or from `source` in scratch, combined into one partial per
/// workgroup at `destination` in scratch.
struct Level {
    count: u32,
    source: u32,
    destination: u32,
    grid: [u32; 2],
}

/// Plans the levels that reduce `count` values to one. Even an empty input
/// gets one level, which writes the identity value.
fn plan(
    count: u32,
    limits: &wgpu::Limits,
    resource: &'static str,
) -> Result<(Vec<Level>, u64), GpuError> {
    let mut levels = Vec::new();
    let (mut remaining, mut source, mut destination) = (count, 0u64, 0u64);
    loop {
        let blocks = remaining.div_ceil(WORKGROUP_SIZE).max(1);
        let grid = pipeline::workgroup_grid(limits, remaining.max(1), resource)?;
        levels.push(Level {
            count: remaining,
            source: source as u32,
            destination: destination as u32,
            grid,
        });
        source = destination;
        destination += u64::from(blocks);
        if destination > u64::from(u32::MAX) {
            return Err(GpuError::LimitsExceeded { resource });
        }
        if blocks == 1 {
            return Ok((levels, destination));
        }
        remaining = blocks;
    }
}

/// The compiled reduction kernels for one value type and operation.
struct Kernels {
    layout: wgpu::BindGroupLayout,
    first: wgpu::ComputePipeline,
    next: wgpu::ComputePipeline,
}
impl Kernels {
    /// `op` is the shader's operation code: 0 sum, 1 min, 2 max, 3 dot.
    fn new(device: &wgpu::Device, value_type: f64, op: f64) -> Self {
        let label = "MOPMesh reduce";
        let module = pipeline::shader_module(
            device,
            label,
            include_str!(concat!(env!("OUT_DIR"), "/reduce.wgsl")),
        );
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let (layout, pipeline_layout) = pipeline::buffer_layout(
            device,
            label,
            &[
                wgpu::BufferBindingType::Uniform,
                read,
                read,
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let constants = [("VALUE_TYPE", value_type), ("OP", op)];
        let create = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..pipeline::NO_ZERO_FILL
                },
                cache: None,
            })
        };
        Self {
            first: create("first"),
            next: create("next"),
            layout,
        }
    }

    /// Records every level, then copies the final value into a new result.
    #[allow(clippy::too_many_arguments)]
    fn encode<T: ReduceValue>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: MeshId,
        count: u32,
        input: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        resource: &'static str,
    ) -> Result<GpuReduction<T>, GpuError> {
        // Plan the levels and check the scratch size before allocating.
        let limits = device.limits();
        let (levels, words) = plan(count, &limits, resource)?;
        check_binding(&limits, words * 4, resource)?;
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh reduce scratch"),
            size: words * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Each level combines groups of 64 values: the first reads the input,
        // the rest read the previous level's partials.
        for (i, level) in levels.iter().enumerate() {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MOPMesh reduce level"),
                contents: bytemuck::cast_slice(&[level.count, level.source, level.destination, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bindings = pipeline::bind_buffers(
                device,
                "MOPMesh reduce bindings",
                &self.layout,
                &[&params, input, input_b, &scratch],
            );
            let kernel = if i == 0 { &self.first } else { &self.next };
            pipeline::dispatch_grid(encoder, "MOPMesh reduce", kernel, &bindings, level.grid);
        }

        // The last level wrote the single result as the last scratch word.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh reduce result"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&scratch, (words - 1) * 4, &buffer, 0, 4);
        Ok(GpuReduction {
            buffer,
            mesh,
            marker: PhantomData,
        })
    }
}

/// Reduces a scalar attribute of any element kind to one value with a
/// [`ReduceOp`]. Create one per device, value type and operation.
///
/// `f32` results depend on the order values are combined, which is fixed for
/// a given attribute length, so repeated runs on one device agree. NaN
/// values give unspecified results.
pub struct ReduceQuery<T: ReduceValue> {
    kernels: Kernels,
    marker: PhantomData<T>,
}
impl<T: ReduceValue> ReduceQuery<T> {
    pub fn new(device: &wgpu::Device, op: ReduceOp) -> Self {
        let op = match op {
            ReduceOp::Sum => 0.,
            ReduceOp::Min => 1.,
            ReduceOp::Max => 2.,
        };
        Self {
            kernels: Kernels::new(device, T::TYPE, op),
            marker: PhantomData,
        }
    }

    /// Records a reduction of every value of `attribute`. Returns
    /// [`GpuError::LimitsExceeded`] before allocating or recording commands if
    /// the scratch buffer or a dispatch is too large.
    pub fn encode<E: Element>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        attribute: &GpuAttribute<E, T>,
    ) -> Result<GpuReduction<T>, GpuError> {
        let buffer = attribute.buffer();
        self.kernels.encode(
            device,
            encoder,
            attribute.mesh_id(),
            attribute.len(),
            buffer,
            buffer,
            "reduction",
        )
    }
}

/// Computes the dot product of two `f32` attributes of the same mesh and
/// element kind: the sum of their element-wise products. Rounding behaves as
/// for [`ReduceQuery`] sums. Cache one per device.
pub struct DotQuery {
    kernels: Kernels,
}
impl DotQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            kernels: Kernels::new(device, f32::TYPE, 3.),
        }
    }

    /// Records the dot product of `a` and `b`. Returns
    /// [`GpuError::MeshMismatch`] if they belong to different meshes,
    /// [`GpuError::Stale`] if both were computed from connectivity but from
    /// different uploads or versions, or
    /// [`GpuError::LimitsExceeded`] if the scratch buffer or a dispatch is too
    /// large. Errors occur before allocating or recording commands. Empty
    /// attributes give zero.
    pub fn encode<E: Element>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuAttribute<E, f32>,
        b: &GpuAttribute<E, f32>,
    ) -> Result<GpuReduction<f32>, GpuError> {
        if a.mesh_id() != b.mesh_id() {
            return Err(GpuError::MeshMismatch);
        }
        if let (Some(ta), Some(tb)) = (a.topology(), b.topology()) {
            ta.check(tb)?;
        }
        self.kernels.encode(
            device,
            encoder,
            a.mesh_id(),
            a.len(),
            a.buffer(),
            b.buffer(),
            "dot product",
        )
    }
}
