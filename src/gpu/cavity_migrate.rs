//! Stages that carry out a cavity round's selected edits on an editable mesh,
//! using `topology::cavity_migrate`.
//! [`CavityRounds::encode_round`](super::cavity_rounds::CavityRounds::encode_round)
//! records them when asked to migrate.
use super::cavity_rounds::{CavityState, PLAN_PATCHES};
use super::pipeline;
use super::{GpuEditableMesh, GpuError};
use std::ops::Range;

/// Entry points in dispatch order. Each runs one workgroup per patch.
pub(crate) const STAGES: [&str; 8] = [
    "ring",
    "claim",
    "preflight",
    "fill_vertices",
    "fill_edges",
    "fill_faces",
    "transfer",
    "cleanup",
];
/// Pass labels for [`STAGES`], distinct from the scheduler's.
const LABELS: [&str; 8] = [
    "migrate ring",
    "migrate claim",
    "migrate preflight",
    "migrate fill vertices",
    "migrate fill edges",
    "migrate fill faces",
    "migrate transfer",
    "migrate cleanup",
];
/// The stages to record before the scheduler's `conclude` stage; the rest
/// come after it.
pub(crate) const BEFORE_CONCLUDE: Range<usize> = 0..3;
pub(crate) const AFTER_CONCLUDE: Range<usize> = 3..8;

const RESOURCE: &str = "cavity migration";

/// The migration pipelines and their bind group layouts. Create one per
/// device.
pub(crate) struct CavityMigrate {
    empty: wgpu::BindGroupLayout,
    bindings: wgpu::BindGroupLayout,
    stages: Vec<wgpu::ComputePipeline>,
}

impl CavityMigrate {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "MOPMesh cavity migration",
            include_str!(concat!(env!("OUT_DIR"), "/cavity_migrate.wgsl")),
        );

        // Group 0 is unused; group 1 holds the mesh, the round state and the
        // maps at binding 7.
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let (empty, _) = pipeline::buffer_layout(device, "MOPMesh cavity migration unused", &[]);
        let bindings = pipeline::buffer_layout_at(
            device,
            "MOPMesh cavity migration",
            &[0, 1, 2, 3, 4, 7].map(|binding| (binding, write)),
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh cavity migration"),
            bind_group_layouts: &[Some(&empty), Some(&bindings)],
            immediate_size: 0,
        });
        let stages = STAGES
            .iter()
            .map(|entry| {
                pipeline::compute_pipeline(
                    device,
                    "MOPMesh cavity migration stage",
                    &module,
                    entry,
                    Some(&layout),
                )
            })
            .collect();
        Self {
            empty,
            bindings,
            stages,
        }
    }

    /// Records `stages` (indices into [`STAGES`]) for the round last
    /// recorded on `state`. With `planned`, each stage runs with the round's
    /// planned sizes, so nothing runs once no request is pending; otherwise
    /// each runs one workgroup per patch.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuEditableMesh,
        state: &CavityState,
        stages: Range<usize>,
        planned: bool,
    ) -> Result<(), GpuError> {
        let patches = pipeline::group_grid(&device.limits(), mesh.patch_count(), RESOURCE)?;
        let empty = pipeline::bind_buffers(device, RESOURCE, &self.empty, &[]);
        let bindings = pipeline::bind_buffers_at(
            device,
            RESOURCE,
            &self.bindings,
            &[
                (0, mesh.arena()),
                (1, mesh.masks()),
                (2, &state.vertex_records),
                (3, &state.patch_words),
                (4, &state.scratch),
                (7, mesh.maps()),
            ],
        );
        let groups = [&empty, &bindings];
        for i in stages {
            let (pipeline, label) = (&self.stages[i], LABELS[i]);
            if planned {
                pipeline::dispatch_indirect_groups(
                    encoder,
                    label,
                    pipeline,
                    &groups,
                    &state.args,
                    PLAN_PATCHES,
                );
            } else {
                pipeline::dispatch_grid_groups(encoder, label, pipeline, &groups, patches);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::EditCapacity;
    use crate::gpu::bench_meshes;
    use crate::gpu::bench_migration::Sim;
    use crate::gpu::cavity_rounds::tests::predict_round;
    use crate::gpu::cavity_rounds::{
        CavityRounds, FAILED, INVOLVED_WORD, NONE, PATCH_WORDS, PENDING, REQUEST_WORDS,
    };
    use crate::gpu::editable::inspect::{
        PatchView, check_exact_ribbons, check_invariants, patch_views,
    };
    use crate::gpu::test_support;
    use std::collections::{BTreeMap, BTreeSet, HashSet};

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
        let encoder = device.create_command_encoder(&Default::default());
        test_support::read_words(device, queue, encoder, buffer)
    }

    fn rotate([a, b, c]: [u32; 3]) -> [u32; 3] {
        let m = a.min(b).min(c);
        if m == a {
            [a, b, c]
        } else if m == b {
            [b, c, a]
        } else {
            [c, a, b]
        }
    }

    /// Patch `p` as the model holds it, in the form `patch_views` reads.
    pub(in crate::gpu) fn model_view(sim: &Sim, p: u32) -> PatchView {
        let mut view = PatchView::default();
        for &v in sim.held(p, 0) {
            view.vertices.insert(v);
            if sim.owner(0, v) == p {
                view.owned_vertices.insert(v);
            }
            if sim.interior_to(v) == Some(p) {
                view.interior.insert(v);
            }
        }
        for &e in sim.held(p, 1) {
            let [a, b] = sim.edge_ends(e);
            view.edges.insert([a.min(b), a.max(b)]);
            if sim.owner(1, e) == p {
                view.owned_edges.insert([a.min(b), a.max(b)]);
            }
        }
        for &f in sim.held(p, 2) {
            view.faces.insert(rotate(sim.triangle(f)));
            if sim.owner(2, f) == p {
                view.owned_faces.insert(rotate(sim.triangle(f)));
            }
        }
        view
    }

    /// Sorts each `width`-word entry's words in `range`, which the queries
    /// leave in unspecified order.
    fn sort_within(words: &mut [u32], width: usize, range: std::ops::Range<usize>) {
        for entry in words.chunks_mut(width) {
            entry[range.clone()].sort();
        }
    }

    /// Sorts the rows of a CSR list of `rows` rows with `width`-word entries,
    /// whose row order is unspecified.
    fn sort_rows(words: &mut [u32], rows: usize, width: usize) {
        let (offsets, entries) = words.split_at_mut(rows + 1);
        for w in offsets.windows(2) {
            let row = &mut entries[width * w[0] as usize..width * w[1] as usize];
            let mut sorted: Vec<Vec<u32>> = row.chunks(width).map(<[u32]>::to_vec).collect();
            sorted.sort();
            row.copy_from_slice(&sorted.concat());
        }
    }

    /// Every connectivity query on `mesh`, read back with unspecified orders
    /// normalized.
    fn connectivity(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &GpuEditableMesh,
    ) -> Vec<Vec<u32>> {
        use crate::gpu::{
            AdjacencyQuery, BoundaryQuery, EdgeQuery, FaceVertexQuery, VertexFaceQuery,
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let e = &mut encoder;
        let edges = EdgeQuery::new(device).encode(device, e, mesh).unwrap();
        let faces = FaceVertexQuery::new(device)
            .encode(device, e, mesh)
            .unwrap();
        let vertex_faces = VertexFaceQuery::new(device)
            .encode(device, e, mesh)
            .unwrap();
        let adjacency = AdjacencyQuery::new(device).encode(device, e, mesh).unwrap();
        let boundary = BoundaryQuery::new(device).encode(device, e, mesh).unwrap();
        queue.submit([encoder.finish()]);

        let vertices = mesh.vertex_count() as usize;
        let words = |b: &wgpu::Buffer| read(device, queue, b);
        let mut edge_words = words(edges.buffer());
        sort_within(&mut edge_words, 4, 2..4);
        let mut vertex_face_words = words(vertex_faces.buffer());
        sort_rows(&mut vertex_face_words, vertices, 1);
        let mut neighbor_words = words(adjacency.vertex_neighbors());
        sort_rows(&mut neighbor_words, vertices, 2);
        let mut diamond_words = words(adjacency.edge_diamonds());
        sort_within(&mut diamond_words, 4, 2..4);
        vec![
            edge_words,
            words(faces.buffer()),
            vertex_face_words,
            neighbor_words,
            words(adjacency.face_neighbors()),
            diamond_words,
            words(boundary.edges.buffer()),
            words(boundary.vertices.buffer()),
        ]
    }

    /// Evidence that a run exercised the cases worth testing.
    #[derive(Default, Debug)]
    struct Seen {
        /// Acting patches that imported at least one copy.
        imports: usize,
        /// Acting patches whose selected requests' cavities shared a face.
        overlapping: usize,
        /// Faces handed back to a patch that owned them before.
        returns: usize,
        /// Acting patches that ran out of room.
        failures: usize,
        /// Rounds in which every acting patch failed, leaving arena and
        /// masks byte-identical.
        unchanged_rounds: usize,
    }

    /// Runs three batches of migration rounds over every edge, on each
    /// generated mesh at patch sizes 16 and 64, with `capacity` times the
    /// starting slots. After every round the GPU must match the CPU model
    /// exactly and keep every invariant.
    fn migrate_batches(capacity: f32) -> Seen {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let rounds = CavityRounds::new(&device);
        let mut seen = Seen::default();
        for bench in bench_meshes::generated(4096) {
            for patch_size in [16u32, 64] {
                let mesh = MeshBuilder::new()
                    .patch_size(patch_size as usize)
                    .positions(&bench.positions)
                    .triangles(&bench.triangles)
                    .build()
                    .unwrap();
                let factor = EditCapacity::new(capacity).unwrap();
                let mut gpu = GpuEditableMesh::upload(&device, &mesh, factor).unwrap();
                let mut sim = Sim::with_capacity(
                    &bench.positions,
                    &bench.triangles,
                    patch_size,
                    f64::from(capacity),
                );
                let mut past_owners: Vec<HashSet<u32>> = (0..bench.triangles.len() as u32)
                    .map(|f| HashSet::from([sim.owner(2, f)]))
                    .collect();

                for batch in 0..3 {
                    let seed = 7 + batch;
                    let context =
                        format!("{} at patch size {patch_size}, batch {batch}", bench.name);
                    let mut words = Vec::with_capacity(REQUEST_WORDS * sim.edge_count());
                    for e in 0..sim.edge_count() as u32 {
                        match sim.quad(e) {
                            Some(q) => {
                                words.extend([PENDING, 0]);
                                words.extend(q.corners);
                            }
                            None => words.extend([NONE; REQUEST_WORDS]),
                        }
                    }
                    let state = CavityState::new(&device, &gpu, &words, seed).unwrap();
                    let mut taken = HashSet::new();
                    let mut round = 0;
                    while words.chunks(REQUEST_WORDS).any(|w| w[0] == PENDING) {
                        assert!(round < 10_000, "{context}: rounds drain");
                        let context = format!("{context}, round {round}");
                        let before = [gpu.arena(), gpu.masks()].map(|b| read(&device, &queue, b));

                        // Model: group the selected requests by acting
                        // patch, migrate each, then clean every footprint.
                        let selected = predict_round(&sim, &mut words, &mut taken, seed, round);
                        let mut by_actor: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
                        for &r in &selected {
                            by_actor
                                .entry(words[REQUEST_WORDS * r + 1])
                                .or_default()
                                .push(r);
                        }
                        let mut dirty = BTreeSet::new();
                        let mut any_migrated = false;
                        for (&actor, requests) in &by_actor {
                            let mut faces = BTreeSet::new();
                            let mut overlap = false;
                            for &r in requests {
                                let at = REQUEST_WORDS * r + 2;
                                let cavity = sim.cavity_faces(&words[at..at + 4]);
                                overlap |= !faces.is_disjoint(&cavity);
                                faces.extend(cavity);
                            }
                            seen.overlapping += usize::from(overlap);
                            dirty.extend(sim.footprint(&faces));
                            let moving: Vec<u32> = faces
                                .iter()
                                .copied()
                                .filter(|&f| sim.owner(2, f) != actor)
                                .collect();
                            match sim.migrate(actor, &faces) {
                                Some(missing) => {
                                    any_migrated = true;
                                    seen.imports += usize::from(missing.iter().sum::<usize>() > 0);
                                    for f in moving {
                                        if !past_owners[f as usize].insert(actor) {
                                            seen.returns += 1;
                                        }
                                    }
                                }
                                None => {
                                    seen.failures += 1;
                                    for &r in requests {
                                        let at = REQUEST_WORDS * r;
                                        words[at] = FAILED;
                                        for v in &words[at + 2..at + 6] {
                                            taken.remove(v);
                                        }
                                    }
                                }
                            }
                        }
                        for q in dirty {
                            assert_eq!(sim.clean(q), 0, "{context}: model ribbon of {q}");
                        }

                        // GPU: the same round, with migration.
                        let mut encoder = device.create_command_encoder(&Default::default());
                        rounds
                            .encode_round(&device, &mut encoder, &gpu, &state, round, true)
                            .unwrap();
                        gpu.mark_edited();
                        queue.submit([encoder.finish()]);
                        let read_back = read(&device, &queue, state.requests());
                        assert_eq!(read_back, words, "{context}: requests");
                        let arena = read(&device, &queue, gpu.arena());
                        let masks = read(&device, &queue, gpu.masks());
                        check_invariants(&arena, &masks, &read(&device, &queue, gpu.maps()));
                        check_exact_ribbons(&arena, &masks);
                        for (p, view) in patch_views(&arena, &masks).into_iter().enumerate() {
                            assert_eq!(view, model_view(&sim, p as u32), "{context}: patch {p}");
                        }
                        if !selected.is_empty() && !any_migrated {
                            assert_eq!([arena, masks], before, "{context}: failed round");
                            seen.unchanged_rounds += 1;
                        }
                        round += 1;
                    }
                }

                // Migration doesn't change the mesh, so every query matches
                // a fresh upload.
                let fresh = GpuEditableMesh::upload(&device, &mesh, factor).unwrap();
                assert_eq!(
                    connectivity(&device, &queue, &gpu),
                    connectivity(&device, &queue, &fresh),
                    "{} at patch size {patch_size}: queries",
                    bench.name
                );
            }
        }
        println!("{seen:?}");
        seen
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn rounds_migrate_like_the_cpu_model() {
        let seen = migrate_batches(1.5);
        assert!(seen.imports > 0, "some acting patch imports copies");
        assert!(
            seen.overlapping > 0,
            "some acting patch has overlapping cavities"
        );
        assert!(seen.returns > 0, "some face returns to a former owner");
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn acting_patches_without_room_change_nothing() {
        let seen = migrate_batches(1.0);
        assert!(seen.failures > 0, "some acting patch runs out of room");
        assert!(seen.unchanged_rounds > 0, "some round fails entirely");
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn cleanup_leaves_fresh_uploads_unchanged() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let migrate = CavityMigrate::new(&device);
        for bench in bench_meshes::generated(4096) {
            for patch_size in [16usize, 64] {
                let context = format!("{} at patch size {patch_size}", bench.name);
                let mesh = MeshBuilder::new()
                    .patch_size(patch_size)
                    .positions(&bench.positions)
                    .triangles(&bench.triangles)
                    .build()
                    .unwrap();
                let gpu = GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();
                let arena = read(&device, &queue, gpu.arena());
                let masks = read(&device, &queue, gpu.masks());
                check_exact_ribbons(&arena, &masks);

                // Flag every patch as involved, so cleanup runs on all of them.
                let state = CavityState::new(&device, &gpu, &[], 0).unwrap();
                let mut words = vec![0u32; PATCH_WORDS * gpu.patch_count() as usize];
                words
                    .iter_mut()
                    .skip(INVOLVED_WORD)
                    .step_by(PATCH_WORDS)
                    .for_each(|w| *w = 1);
                queue.write_buffer(&state.patch_words, 0, bytemuck::cast_slice(&words));
                let mut encoder = device.create_command_encoder(&Default::default());
                migrate
                    .encode(&device, &mut encoder, &gpu, &state, 0..STAGES.len(), false)
                    .unwrap();
                queue.submit([encoder.finish()]);

                // The arena and masks are unchanged, and the maps, which
                // cleanup leaves alone when it drops nothing, still record
                // exactly the active slots.
                assert_eq!(
                    read(&device, &queue, gpu.arena()),
                    arena,
                    "{context}: arena"
                );
                assert_eq!(
                    read(&device, &queue, gpu.masks()),
                    masks,
                    "{context}: masks"
                );
                check_invariants(&arena, &masks, &read(&device, &queue, gpu.maps()));
            }
        }
    }
}
