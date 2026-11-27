//! Test helpers for comparing GPU-produced connectivity with the CPU reference.
pub(crate) use super::ArenaWords;
use super::{arena_contents, patch_layout as layout};
use crate::{BuildError, Mesh, MeshBuilder};
use wgpu::util::DeviceExt;

#[path = "../../tests/support/fixtures.rs"]
pub mod fixtures;

/// Edge slots staged per chunk; the value of `STAGED_EDGES` in
/// `topology::patch_staging`. A patch with more edge slots than this is
/// staged in several chunks.
pub const STAGED_EDGES: usize = 512;

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

/// Builds a CPU reference mesh from `fixture` using the requested patch size.
/// Returns the mesh and its connectivity and mask words for comparison with
/// GPU results. Propagates mesh build errors; panics if storage exceeds `limits`.
pub fn reference(
    fixture: &fixtures::Fixture,
    patch_size: usize,
    limits: &wgpu::Limits,
) -> Result<(Mesh, ArenaWords), BuildError> {
    let builder = MeshBuilder::new().patch_size(patch_size);
    reference_with(fixture, builder, limits)
}

/// Builds the same reference data as [`reference`] using a configured builder.
/// Preserves its patch settings and supplies positions and triangles from `fixture`.
pub fn reference_with(
    fixture: &fixtures::Fixture,
    builder: MeshBuilder,
    limits: &wgpu::Limits,
) -> Result<(Mesh, ArenaWords), BuildError> {
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

/// Checks GPU connectivity and masks against CPU reference words for `mesh`.
/// Panics on a difference, naming the word and the patch region that contains it.
/// For example, "FE" in the message identifies face-edge data.
/// The arena must match exactly; masks may have extra trailing zero words.
pub fn assert_arena_eq(
    context: &str,
    mesh: &Mesh,
    actual_arena: &[u32],
    actual_masks: &[u32],
    expected: &ArenaWords,
) {
    // Find the first differing or missing connectivity word.
    let first_difference = |actual: &[u32], expected: &[u32]| {
        (0..actual.len().max(expected.len())).find(|&i| actual.get(i) != expected.get(i))
    };
    if let Some(index) = first_difference(actual_arena, &expected.arena) {
        panic!(
            "{context}: arena word {index} differs ({}): GPU {:?}, CPU {:?}; \
             lengths GPU {}, CPU {}",
            arena_region(mesh, index),
            actual_arena.get(index),
            expected.arena.get(index),
            actual_arena.len(),
            expected.arena.len(),
        );
    }

    // Compare all expected mask words and allow only zero padding after them.
    let len = expected.masks.len().min(actual_masks.len());
    let padded =
        actual_masks.len() >= expected.masks.len() && actual_masks[len..].iter().all(|&w| w == 0);
    if let Some(index) = first_difference(&actual_masks[..len], &expected.masks) {
        panic!(
            "{context}: mask word {index} differs ({}): GPU {:?}, CPU {:?}",
            mask_region(mesh, index),
            actual_masks.get(index),
            expected.masks.get(index),
        );
    }
    assert!(
        padded,
        "{context}: masks have {} words, CPU {}, or a nonzero word past the CPU length",
        actual_masks.len(),
        expected.masks.len()
    );
}

/// Describes where a word falls in `mesh`'s static connectivity buffer,
/// so a failed comparison identifies the affected patch and kind of data.
fn arena_region(mesh: &Mesh, index: usize) -> String {
    let records = layout::HEADER_WORDS + layout::RECORD_WORDS * mesh.patches().len();
    if index < layout::HEADER_WORDS {
        return "header".into();
    }
    if index < records {
        let at = index - layout::HEADER_WORDS;
        return format!(
            "patch {} record, word {}",
            at / layout::RECORD_WORDS,
            at % layout::RECORD_WORDS
        );
    }

    // Locate the word by following the stored order of each patch's data.
    let mut start = records;
    for p in mesh.patches() {
        let sections = [
            ("vertex IDs", p.vertex_ids().len()),
            ("EV", 2 * p.edge_vertices().len()),
            ("FE", 6 * p.face_edges().len()),
            ("edge IDs", p.edge_ids().len()),
            ("face IDs", p.face_ids().len()),
        ];
        for (name, len) in sections {
            if index < start + len {
                return format!("patch {} {name}, word {}", p.id(), index - start);
            }
            start += len;
        }
    }
    "past the end".into()
}

/// Describes where a word falls in `mesh`'s static mask buffer, including
/// the patch, element kind and whether it tracks active or owned slots.
fn mask_region(mesh: &Mesh, index: usize) -> String {
    let mut start = 0;
    for p in mesh.patches() {
        let words = [p.vertex_ids().len(), p.edge_ids().len(), p.face_ids().len()]
            .map(|n| layout::mask_words(n as u32) as usize);
        let mut sections = vec![("used counts".to_owned(), 3)];
        for mask in ["active", "owned"] {
            for (kind, len) in ["vertex", "edge", "face"].into_iter().zip(words) {
                sections.push((format!("{mask} {kind} mask"), len));
            }
        }
        for (name, len) in sections {
            if index < start + len {
                return format!("patch {} {name}, word {}", p.id(), index - start);
            }
            start += len;
        }
    }
    "past the end".into()
}
