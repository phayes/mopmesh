//! Helpers for creating query pipelines, checking sizes, and recording GPU work.
use super::GpuError;

/// Number of shader invocations in each built-in query workgroup.
/// Must match the shaders' `@workgroup_size` declarations.
pub(super) const WORKGROUP_SIZE: u32 = 64;

/// Returns the largest storage buffer size, in bytes, allowed by both the
/// allocation limit and the per-binding limit.
pub(super) fn max_binding_bytes(limits: &wgpu::Limits) -> u64 {
    limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size)
}

/// Checks a buffer size in bytes against the storage-binding, allocation, and
/// `u32` word-index limits. `resource` names the buffer in any returned error.
pub(super) fn check_binding(
    limits: &wgpu::Limits,
    bytes: u64,
    resource: &'static str,
) -> Result<(), GpuError> {
    if bytes / 4 > u64::from(u32::MAX) || bytes > max_binding_bytes(limits) {
        return Err(GpuError::LimitsExceeded { resource });
    }
    Ok(())
}

/// Returns enough workgroups to cover `count` invocations, rounding up.
/// Returns zero for an empty input, or a limit error labeled with `resource`
/// if the one-dimensional dispatch would exceed the device limit.
pub(super) fn workgroups(
    limits: &wgpu::Limits,
    count: u32,
    resource: &'static str,
) -> Result<u32, GpuError> {
    let groups = count.div_ceil(WORKGROUP_SIZE);
    if groups > limits.max_compute_workgroups_per_dimension {
        return Err(GpuError::LimitsExceeded { resource });
    }
    Ok(groups)
}

/// Returns a two-dimensional workgroup grid covering `count` invocations: as
/// many groups along x as the device allows, then further rows along y.
/// Shaders recover the flat index with `shaders::parallel::grid::flat_index`.
/// Returns `[0, 0]` for an empty input, or a limit error labeled with
/// `resource` if even a square grid would exceed the device limit.
pub(super) fn workgroup_grid(
    limits: &wgpu::Limits,
    count: u32,
    resource: &'static str,
) -> Result<[u32; 2], GpuError> {
    let groups = count.div_ceil(WORKGROUP_SIZE);
    let width = groups.min(limits.max_compute_workgroups_per_dimension);
    if groups == 0 {
        return Ok([0, 0]);
    }
    let rows = groups.div_ceil(width);
    if rows > limits.max_compute_workgroups_per_dimension {
        return Err(GpuError::LimitsExceeded { resource });
    }
    Ok([width, rows])
}

/// Like [`dispatch`], over a grid from [`workgroup_grid`].
pub(super) fn dispatch_grid(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    [x, y]: [u32; 2],
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

pub(super) fn shader_module(
    device: &wgpu::Device,
    label: &str,
    wgsl: &'static str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    })
}

/// Creates a compute pipeline for `entry_point` in `module`.
/// Pass `None` for `layout` to infer buffer bindings from the shader.
pub(super) fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layout: Option<&wgpu::PipelineLayout>,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout,
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Creates a buffer bind-group layout and a pipeline layout using it as group 0.
/// Slice positions become binding numbers, starting at zero; all bindings are
/// visible to compute shaders.
pub(super) fn buffer_layout(
    device: &wgpu::Device,
    label: &str,
    bindings: &[wgpu::BufferBindingType],
) -> (wgpu::BindGroupLayout, wgpu::PipelineLayout) {
    let entries: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(binding, &ty)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let group = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&group)],
        immediate_size: 0,
    });
    (group, pipeline)
}

/// Creates a bind group with each entire buffer bound at its slice index.
/// The buffers must match the types and order required by `layout`.
pub(super) fn bind_buffers(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// Records a compute pass using `bindings` as group 0 and dispatching
/// `workgroups` groups along the x axis. Later passes can read this pass's writes.
pub(super) fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &wgpu::ComputePipeline,
    bindings: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

/// Creates a zero-initialized buffer of `bytes` bytes with `STORAGE | COPY_SRC`
/// usage, suitable for shader results and copying to a readback buffer.
pub(super) fn output_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_checks_use_the_smaller_buffer_limit_and_u32_words() {
        let mut limits = wgpu::Limits {
            max_buffer_size: 1 << 40,
            max_storage_buffer_binding_size: 1 << 40,
            ..Default::default()
        };
        assert!(check_binding(&limits, 4 * u64::from(u32::MAX), "x").is_ok());
        assert_eq!(
            check_binding(&limits, 4 * u64::from(u32::MAX) + 4, "x"),
            Err(GpuError::LimitsExceeded { resource: "x" })
        );
        limits.max_storage_buffer_binding_size = 64;
        assert!(check_binding(&limits, 64, "x").is_ok());
        assert!(check_binding(&limits, 68, "x").is_err());
        limits.max_storage_buffer_binding_size = 1 << 40;
        limits.max_buffer_size = 32;
        assert!(check_binding(&limits, 32, "x").is_ok());
        assert!(check_binding(&limits, 36, "x").is_err());
    }

    #[test]
    fn workgroup_counts_round_up_and_respect_the_dispatch_limit() {
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 2,
            ..Default::default()
        };
        assert_eq!(workgroups(&limits, 0, "x"), Ok(0));
        assert_eq!(workgroups(&limits, 1, "x"), Ok(1));
        assert_eq!(workgroups(&limits, 128, "x"), Ok(2));
        assert_eq!(
            workgroups(&limits, 129, "x"),
            Err(GpuError::LimitsExceeded { resource: "x" })
        );
    }
}
