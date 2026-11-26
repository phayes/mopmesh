use super::GpuLimitError;
use wgpu::util::DeviceExt;

const WIDTH: u32 = 64;

struct Level {
    count: u32,
    offset: u32,
    next_offset: u32,
}

/// The first `count` words are input and become exclusive prefixes. Remaining
/// words store recursively scanned block sums, ending in one grand-total word.
pub(super) struct ScanPlan {
    levels: Vec<Level>,
    words: u64,
}
impl ScanPlan {
    pub(super) fn new(count: u32, limits: &wgpu::Limits) -> Result<Self, GpuLimitError> {
        let mut levels = Vec::new();
        let mut remaining = count;
        let mut offset = 0u64;
        let mut words = 1;
        while remaining != 0 {
            let blocks = remaining.div_ceil(WIDTH);
            let next_offset = offset + u64::from(remaining);
            words = next_offset + u64::from(blocks);
            if words > u64::from(u32::MAX) || blocks > limits.max_compute_workgroups_per_dimension {
                return Err(GpuLimitError);
            }
            levels.push(Level {
                count: remaining,
                offset: offset as u32,
                next_offset: next_offset as u32,
            });
            if blocks == 1 {
                break;
            }
            offset = next_offset;
            remaining = blocks;
        }
        if words * 4
            > limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size)
        {
            return Err(GpuLimitError);
        }
        Ok(Self { levels, words })
    }
    pub(super) fn byte_size(&self) -> u64 {
        self.words * 4
    }
}

pub(super) struct ScanPipeline {
    layout: wgpu::BindGroupLayout,
    scan: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
}
impl ScanPipeline {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh hierarchical scan"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/scan.wgsl")).into(),
            ),
        });
        let entries = [
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MOPMesh scan layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh scan pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            scan: pipeline("scan"),
            add: pipeline("add"),
            layout,
        }
    }

    pub(super) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        arena: &wgpu::Buffer,
        plan: &ScanPlan,
    ) {
        let bindings: Vec<_> = plan
            .levels
            .iter()
            .map(|level| {
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("MOPMesh scan level"),
                    contents: bytemuck::cast_slice(&[level.count, level.offset, level.next_offset]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MOPMesh scan bindings"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: arena.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        let dispatch =
            |encoder: &mut wgpu::CommandEncoder, index: usize, pipeline: &wgpu::ComputePipeline| {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MOPMesh scan level"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bindings[index], &[]);
                pass.dispatch_workgroups(plan.levels[index].count.div_ceil(WIDTH), 1, 1);
            };
        for i in 0..plan.levels.len() {
            dispatch(encoder, i, &self.scan);
        }
        // The top level already has global prefixes; propagate them downward.
        for i in (0..plan.levels.len().saturating_sub(1)).rev() {
            dispatch(encoder, i, &self.add);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_plan_checks_scratch_and_dispatch_limits() {
        let limits = wgpu::Limits::default();
        for (count, words, depth) in [
            (0, 1, 0),
            (64, 65, 1),
            (65, 68, 2),
            (4096, 4161, 2),
            (4097, 4165, 3),
        ] {
            let plan = ScanPlan::new(count, &limits).unwrap();
            assert_eq!(plan.words, words);
            assert_eq!(plan.levels.len(), depth);
        }
        let mut limits = limits;
        limits.max_compute_workgroups_per_dimension = 1;
        assert!(ScanPlan::new(64, &limits).is_ok());
        assert!(ScanPlan::new(65, &limits).is_err());
        limits.max_storage_buffer_binding_size = 64 * 4;
        assert!(ScanPlan::new(64, &limits).is_err()); // Includes the total word.
        limits.max_storage_buffer_binding_size = 65 * 4;
        assert!(ScanPlan::new(64, &limits).is_ok());
        limits.max_buffer_size = 64 * 4;
        assert!(ScanPlan::new(64, &limits).is_err());
        assert!(ScanPlan::new(u32::MAX, &wgpu::Limits::default()).is_err());
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn hierarchical_scan_matches_integer_prefixes() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let pipeline = ScanPipeline::new(&device);
        // Exact boundaries, partial blocks, and up to four recursive levels.
        for count in [0, 1, 63, 64, 65, 4095, 4096, 4097, 262145] {
            for mode in 0..3 {
                let plan = ScanPlan::new(count, &device.limits()).unwrap();
                let mut data = vec![0u32; plan.words as usize];
                for i in 0..count {
                    data[i as usize] = match mode {
                        0 => 0,
                        1 => i % 17,
                        _ => {
                            if i % 7 == 0 {
                                u32::MAX
                            } else {
                                i % 3
                            }
                        }
                    };
                }
                let mut sum = 0u32;
                let expected: Vec<_> = data[..count as usize]
                    .iter()
                    .map(|&value| {
                        let prefix = sum;
                        sum = sum.wrapping_add(value);
                        prefix
                    })
                    .collect();
                let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scan test data"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("scan test readback"),
                    size: plan.byte_size(),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                pipeline.encode(&device, &mut encoder, &arena, &plan);
                encoder.copy_buffer_to_buffer(&arena, 0, &readback, 0, plan.byte_size());
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
                {
                    let bytes = readback.slice(..).get_mapped_range().unwrap();
                    let actual: &[u32] = bytemuck::cast_slice(&bytes);
                    assert_eq!(
                        &actual[..count as usize],
                        expected,
                        "count {count}, mode {mode}"
                    );
                    assert_eq!(
                        *actual.last().unwrap(),
                        sum,
                        "grand total, count {count}, mode {mode}"
                    );
                }
                readback.unmap();
            }
        }
    }
}
