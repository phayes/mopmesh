//! GPU checks of `topology::patch_map` and of the maps that edits keep
//! current: map operations match a host mirror of the same probing, and
//! flips made by downstream kernels and by `EdgeFlipQuery` keep every
//! editable-mesh invariant.
use super::editable::inspect::check_invariants;
use super::patch_layout as layout;
use super::test_support::{self, fixtures};
use super::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh, INVALID_ID};
use crate::MeshBuilder;
use wgpu::util::DeviceExt;

const FACE: usize = 2;

/// Compiles entry point `entry` of the test consumer shader `name`.
fn consumer(device: &wgpu::Device, name: &str, entry: &str) -> wgpu::ComputePipeline {
    let mut resolver = wesl::resolver::StandardResolver::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/shaders"
    ));
    resolver.add_package(&crate::mopmesh::PACKAGE);
    let wgsl = wesl::Compiler::default()
        .with_resolver(resolver)
        .compile_module(&format!("package::{name}").parse().unwrap())
        .unwrap()
        .to_string();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(name),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Binds `maps` at group 1, binding 7, as `topology::patch_map` expects.
fn maps_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    maps: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test maps"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 7,
            resource: maps.as_entire_binding(),
        }],
    })
}

/// Binds `buffers` at group 0, bindings 0, 1, ...
fn group0(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

fn storage(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let encoder = device.create_command_encoder(&Default::default());
    test_support::read_words(device, queue, encoder, buffer)
}

/// The same probing as `topology::patch_map`, on a host copy of the maps.
struct Mirror(Vec<u32>);
impl Mirror {
    fn table(&self, p: u32, kind: u32) -> (usize, u32) {
        let at = layout::MAP_HEADER_WORDS * p as usize + 2 * kind as usize;
        (self.0[at] as usize, self.0[at + 1])
    }
    fn entry(&self, p: u32, kind: u32, global: u32) -> Option<usize> {
        let (offset, entries) = self.table(p, kind);
        let mut at = layout::map_hash(global) & (entries - 1);
        for _ in 0..entries {
            let e = offset + 2 * at as usize;
            if self.0[e] == global {
                return Some(e);
            }
            if self.0[e] == layout::MAP_EMPTY {
                return None;
            }
            at = (at + 1) & (entries - 1);
        }
        None
    }
    fn find(&self, p: u32, kind: u32, global: u32) -> u32 {
        self.entry(p, kind, global)
            .map_or(INVALID_ID, |e| self.0[e + 1])
    }
    fn set(&mut self, p: u32, kind: u32, global: u32, slot: u32) -> bool {
        if let Some(e) = self.entry(p, kind, global) {
            self.0[e + 1] = slot;
            return true;
        }
        let (offset, entries) = self.table(p, kind);
        let mut at = layout::map_hash(global) & (entries - 1);
        for _ in 0..entries {
            let e = offset + 2 * at as usize;
            if self.0[e] >= layout::MAP_TOMBSTONE {
                self.0[e] = global;
                self.0[e + 1] = slot;
                return true;
            }
            at = (at + 1) & (entries - 1);
        }
        false
    }
    fn erase(&mut self, p: u32, kind: u32, global: u32) -> bool {
        let Some(e) = self.entry(p, kind, global) else {
            return false;
        };
        self.0[e] = layout::MAP_TOMBSTONE;
        self.0[e + 1] = INVALID_ID;
        true
    }
    /// The live keys of one map.
    fn keys(&self, p: u32, kind: u32) -> Vec<u32> {
        let (offset, entries) = self.table(p, kind);
        (0..entries as usize)
            .map(|i| self.0[offset + 2 * i])
            .filter(|&k| k < layout::MAP_TOMBSTONE)
            .collect()
    }
    fn first_probe(&self, p: u32, kind: u32, global: u32) -> u32 {
        layout::map_hash(global) & (self.table(p, kind).1 - 1)
    }
}

/// A multi-patch grid upload.
fn grid_upload(
    device: &wgpu::Device,
    side: u32,
    patch_size: usize,
) -> (crate::Mesh, GpuEditableMesh) {
    let (positions, triangles) = fixtures::grid(side);
    let mesh = MeshBuilder::new()
        .patch_size(patch_size)
        .positions(&positions)
        .triangles(&triangles)
        .build()
        .unwrap();
    let gpu = GpuEditableMesh::upload(device, &mesh, EditCapacity::default()).unwrap();
    (mesh, gpu)
}

/// IDs past `start` whose first probe in patch `p`'s `kind` map is `target`.
fn colliding(mirror: &Mirror, p: u32, kind: u32, target: u32, start: u32, n: usize) -> Vec<u32> {
    (start..)
        .filter(|&g| mirror.first_probe(p, kind, g) == target)
        .take(n)
        .collect()
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn patch_map_operations_match_a_host_mirror() {
    let (device, queue) = test_support::device(wgpu::Limits::default());
    let pipeline = consumer(&device, "patch_map_consumer", "sequential");
    let (mesh, gpu) = grid_upload(&device, 9, 16);
    let mut mirror = Mirror(read(&device, &queue, gpu.maps()));

    // Every held ID of every map is found where the mirror says.
    let (find, set, erase) = (0u32, 1u32, 2u32);
    let mut ops: Vec<[u32; 5]> = Vec::new();
    for p in 0..gpu.patch_count() {
        for kind in 0..3 {
            for g in mirror.keys(p, kind) {
                ops.push([find, p, kind, g, 0]);
            }
        }
    }

    // On one face map: misses, colliding sets, a tombstone inside a probe
    // chain, its reuse, an overwrite, and an erase of a missing ID.
    let (p, k) = (0, FACE as u32);
    let held = mirror.keys(p, k);
    let target = mirror.first_probe(p, k, held[0]);
    let c = colliding(&mirror, p, k, target, mesh.face_count() as u32, 3);
    ops.extend([
        [find, p, k, c[0], 0],
        [set, p, k, c[0], 500],
        [set, p, k, c[1], 501],
        [erase, p, k, c[0], 0],
        [find, p, k, c[1], 0],
        [set, p, k, c[2], 502],
        [find, p, k, c[2], 0],
        [find, p, k, c[0], 0],
        [set, p, k, held[0], 777],
        [find, p, k, held[0], 0],
        [erase, p, k, c[0], 0],
    ]);

    // Fill the map until a set fails, then look up a missing ID in it.
    let entries = mirror.table(p, k).1;
    for i in 0..entries {
        ops.push([set, p, k, 1_000_000 + i, i]);
    }
    ops.push([find, p, k, 999_999, 0]);

    // Expected results from the mirror, in order.
    let expected: Vec<u32> = ops
        .iter()
        .map(|&[code, p, kind, g, slot]| match code {
            0 => mirror.find(p, kind, g),
            1 => u32::from(mirror.set(p, kind, g, slot)),
            _ => u32::from(mirror.erase(p, kind, g)),
        })
        .collect();
    let tail = &expected[expected.len() - entries as usize - 12..];
    assert_eq!(
        tail[..11],
        [INVALID_ID, 1, 1, 1, 501, 1, 502, INVALID_ID, 1, 777, 0],
        "mirror semantics"
    );
    assert!(
        expected
            .iter()
            .rev()
            .skip(1)
            .take(entries as usize)
            .any(|&r| r == 0),
        "the map fills"
    );
    assert_eq!(*expected.last().unwrap(), INVALID_ID);

    // Run the same operations on the GPU.
    let ops_buffer = storage(&device, ops.as_flattened());
    let results = storage(&device, &vec![0xdead_beef; ops.len()]);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(
            0,
            &group0(&device, &pipeline, &[&ops_buffer, &results]),
            &[],
        );
        pass.set_bind_group(1, &maps_group(&device, &pipeline, gpu.maps()), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    assert_eq!(read(&device, &queue, &results), expected);
    assert_eq!(read(&device, &queue, gpu.maps()), mirror.0);
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn concurrent_sets_of_colliding_ids_all_land() {
    let (device, queue) = test_support::device(wgpu::Limits::default());
    let pipeline = consumer(&device, "patch_map_consumer", "concurrent");
    let (mesh, gpu) = grid_upload(&device, 9, 16);
    let mirror = Mirror(read(&device, &queue, gpu.maps()));

    // 64 new IDs that all start probing at the same entry.
    let (p, k) = (0, FACE as u32);
    let held = mirror.keys(p, k);
    let entries = mirror.table(p, k).1 as usize;
    assert!(entries - held.len() >= 64, "room for 64 more IDs");
    let target = mirror.first_probe(p, k, held[0]);
    let ids = colliding(&mirror, p, k, target, mesh.face_count() as u32, 64);
    let mut words = vec![p, k];
    words.extend(&ids);

    let ops_buffer = storage(&device, &words);
    let results = storage(&device, &[0xdead_beef; 128]);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(
            0,
            &group0(&device, &pipeline, &[&ops_buffer, &results]),
            &[],
        );
        pass.set_bind_group(1, &maps_group(&device, &pipeline, gpu.maps()), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);

    // Every set succeeded and is found; each ID has one entry; nothing else changed.
    let results = read(&device, &queue, &results);
    assert!(results[..64].iter().all(|&r| r == 1), "{results:?}");
    assert_eq!(results[64..], (1000..1064).collect::<Vec<u32>>());
    let after = Mirror(read(&device, &queue, gpu.maps()));
    let mut keys = after.keys(p, k);
    keys.sort_unstable();
    let mut want = held.clone();
    want.extend(&ids);
    want.sort_unstable();
    assert_eq!(keys, want);
    for &g in &held {
        assert_eq!(after.find(p, k, g), mirror.find(p, k, g));
    }
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn fresh_slot_and_in_place_flips_keep_every_invariant() {
    let (device, queue) = test_support::device(wgpu::Limits::default());
    let pipeline = consumer(&device, "flip_consumer", "main");
    for patch_size in [8, 32, 512] {
        let (positions, triangles) = fixtures::grid(12);
        let mesh = MeshBuilder::new()
            .patch_size(patch_size)
            .positions(&positions)
            .triangles(&triangles)
            .build()
            .unwrap();
        let mut gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
        let check = |gpu: &GpuEditableMesh, context: &str| {
            println!("checking invariants at patch size {patch_size}: {context}");
            let words = [gpu.arena(), gpu.masks(), gpu.maps()].map(|b| read(&device, &queue, b));
            check_invariants(&words[0], &words[1], &words[2]);
        };
        check(&gpu, "fresh upload");

        // Flip every third edge into free slots, one dispatch each.
        let vec4: Vec<[f32; 4]> = positions.iter().map(|&[x, y, z]| [x, y, z, 1.0]).collect();
        let positions = storage(&device, bytemuck::cast_slice(&vec4));
        let maps = maps_group(&device, &pipeline, gpu.maps());
        let mut requests = Vec::new();
        let mut encoder = device.create_command_encoder(&Default::default());
        for edge in (0..mesh.edge_count() as u32).step_by(3) {
            let request = storage(&device, &[edge, 0, 1]);
            let bindings = group0(
                &device,
                &pipeline,
                &[gpu.arena(), gpu.masks(), &positions, &request],
            );
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.set_bind_group(1, &maps, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);
            requests.push(request);
        }
        queue.submit([encoder.finish()]);
        gpu.mark_edited();
        let flipped = requests
            .iter()
            .filter(|r| {
                FlipStatus::from_word(read(&device, &queue, r)[1]) == Some(FlipStatus::Flipped)
            })
            .count();
        assert!(
            flipped > 0,
            "patch size {patch_size}: some fresh-slot flips"
        );
        check(&gpu, "after fresh-slot flips");

        // Then a batch of in-place flips over every edge.
        let query = EdgeFlipQuery::new(&device);
        let edges: Vec<u32> = (0..mesh.edge_count() as u32).collect();
        let mut encoder = device.create_command_encoder(&Default::default());
        query
            .encode(&device, &mut encoder, &mut gpu, &edges)
            .unwrap();
        queue.submit([encoder.finish()]);
        check(&gpu, "after an in-place batch");
    }
}
