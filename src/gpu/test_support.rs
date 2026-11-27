//! Test helpers for comparing GPU-produced connectivity with the CPU reference.
use super::arena_contents;
use crate::{BuildError, Mesh, MeshBuilder};
use wgpu::util::DeviceExt;

#[path = "../../tests/support/fixtures.rs"]
pub mod fixtures;

/// Creates a device and queue on the default adapter with `limits`; pass
/// `wgpu::Limits::default()` to test against portable WebGPU limits. Panics if no
/// adapter is available, so an opt-in GPU test never passes without running.
pub fn device(limits: wgpu::Limits) -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("GPU adapter required for this test");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap()
}

/// Builds `fixture` on the CPU with `patch_size`, returning the mesh and its
/// arena words, or the error the CPU builder reports. Panics if the arena
/// exceeds `limits`.
pub fn reference(
    fixture: &fixtures::Fixture,
    patch_size: usize,
    limits: &wgpu::Limits,
) -> Result<(Mesh, Vec<u32>), BuildError> {
    let builder = MeshBuilder::new().patch_size(patch_size);
    reference_with(fixture, builder, limits)
}

/// Like [`reference`], with a builder that already has its patch settings.
pub fn reference_with(
    fixture: &fixtures::Fixture,
    builder: MeshBuilder,
    limits: &wgpu::Limits,
) -> Result<(Mesh, Vec<u32>), BuildError> {
    let mesh = builder
        .positions(&fixture.positions)
        .triangles(&fixture.triangles)
        .build()?;
    let words = arena_contents(&mesh, limits).unwrap();
    Ok((mesh, words))
}

/// Uploads `fixture` as GPU build inputs: `[f32; 4]` positions and `[u32; 4]`
/// triangles, each padded with a fourth lane that the build must ignore. Empty
/// lists become 16-byte buffers, since bindings cannot be empty.
pub fn input_buffers(
    device: &wgpu::Device,
    fixture: &fixtures::Fixture,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let mut positions: Vec<[f32; 4]> = fixture
        .positions
        .iter()
        .map(|&[x, y, z]| [x, y, z, f32::NAN])
        .collect();
    let mut triangles: Vec<[u32; 4]> = fixture
        .triangles
        .iter()
        .map(|&[a, b, c]| [a, b, c, u32::MAX])
        .collect();
    positions.resize(positions.len().max(1), [0.; 4]);
    triangles.resize(triangles.len().max(1), [0; 4]);
    let upload = |label, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    (
        upload("test positions", bytemuck::cast_slice(&positions)),
        upload("test triangles", bytemuck::cast_slice(&triangles)),
    )
}

/// Copies `buffer` (which needs `COPY_SRC` usage) to the host after submitting
/// `encoder`, and waits for the result.
pub fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
) -> Vec<u32> {
    // Copy into a mappable buffer and submit.
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
    queue.submit([encoder.finish()]);

    // Map and wait.
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
    let words = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
    readback.unmap();
    words
}

/// Panics unless `actual` equals `expected`, the arena of `mesh`. The message
/// names the first differing word and the arena region it falls in, for example
/// "patch 3 FE" or "edge tasks".
pub fn assert_arena_eq(context: &str, mesh: &Mesh, actual: &[u32], expected: &[u32]) {
    let Some(index) =
        (0..actual.len().max(expected.len())).find(|&i| actual.get(i) != expected.get(i))
    else {
        return;
    };
    panic!(
        "{context}: arena differs at word {index} ({}): GPU {:?}, CPU {:?}; lengths GPU {}, CPU {}",
        arena_region(mesh, expected, index),
        actual.get(index),
        expected.get(index),
        actual.len(),
        expected.len(),
    );
}

/// Names the region of `mesh`'s arena `words` that contains word `index`.
fn arena_region(mesh: &Mesh, words: &[u32], index: usize) -> String {
    if index < 4 {
        return "header".into();
    }

    // Walk the per-patch regions in arena order.
    let mut start = 4;
    for p in mesh.patches() {
        let sections = [
            ("vertex IDs", p.vertex_ids().len()),
            ("EV", 2 * p.edge_vertices().len()),
            ("FE", 6 * p.face_edges().len()),
            ("face IDs", p.face_ids().len()),
        ];
        for (name, len) in sections {
            if index < start + len {
                return format!("patch {} {name}, word {}", p.id(), index - start);
            }
            start += len;
        }
    }

    // Then the task lists, whose offsets the header records.
    let edge_tasks = words[3] as usize;
    if index < edge_tasks {
        format!("face task {}", (index - words[1] as usize) / 4)
    } else if index < words.len() {
        format!("edge task {}", (index - edge_tasks) / 7)
    } else {
        "past the end".into()
    }
}
