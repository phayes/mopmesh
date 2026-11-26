use super::GpuError;
use super::pipeline::{self, WORKGROUP_SIZE as WIDTH, check_binding};
use wgpu::util::DeviceExt;

const RESOURCE: &str = "prefix scan";

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
    pub(super) fn new(count: u32, limits: &wgpu::Limits) -> Result<Self, GpuError> {
        let mut levels = Vec::new();
        let mut remaining = count;
        let mut offset = 0u64;
        let mut words = 1;
        while remaining != 0 {
            let blocks = remaining.div_ceil(WIDTH);
            let next_offset = offset + u64::from(remaining);
            words = next_offset + u64::from(blocks);
            if words > u64::from(u32::MAX) || blocks > limits.max_compute_workgroups_per_dimension {
                return Err(GpuError::LimitsExceeded { resource: RESOURCE });
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
        check_binding(limits, words * 4, RESOURCE)?;
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
        let module = pipeline::shader_module(
            device,
            "MOPMesh hierarchical scan",
            include_str!(concat!(env!("OUT_DIR"), "/scan.wgsl")),
        );
        let (layout, pipeline_layout) = pipeline::buffer_layout(
            device,
            "MOPMesh scan layout",
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let build = |label, entry| {
            pipeline::compute_pipeline(device, label, &module, entry, Some(&pipeline_layout))
        };
        Self {
            scan: build("MOPMesh scan blocks", "scan"),
            add: build("MOPMesh scan propagate", "add"),
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
                pipeline::bind_buffers(
                    device,
                    "MOPMesh scan bindings",
                    &self.layout,
                    &[&params, arena],
                )
            })
            .collect();
        let levels: Vec<_> = plan
            .levels
            .iter()
            .map(|level| level.count.div_ceil(WIDTH))
            .zip(&bindings)
            .collect();
        for &(groups, bindings) in &levels {
            let label = "MOPMesh scan blocks";
            pipeline::dispatch(encoder, label, &self.scan, bindings, groups);
        }
        // The highest level already includes all preceding values. Work downward,
        // adding each block's starting offset to its locally computed prefixes.
        for &(groups, bindings) in levels.iter().rev().skip(1) {
            let label = "MOPMesh scan propagate";
            pipeline::dispatch(encoder, label, &self.add, bindings, groups);
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
