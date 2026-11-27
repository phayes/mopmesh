//! Rounds that select edits which may proceed together across patch
//! borders, using `topology::cavity_rounds`. Each round records seven
//! dispatches: lead, bid, relay forward, relay back, check, claim and
//! resolve. When asked to migrate, it also records the stages of
//! `topology::cavity_migrate`, with the scheduler's conclude stage among
//! them.
use super::cavity_migrate::{AFTER_CONCLUDE, BEFORE_CONCLUDE, CavityMigrate};
use super::pipeline;
use super::{GpuEditableMesh, GpuError};
use wgpu::util::DeviceExt;

/// Words per request: stage, acting patch, then four corner vertex IDs.
pub(crate) const REQUEST_WORDS: usize = 6;
pub(crate) const NONE: u32 = 0;
pub(crate) const PENDING: u32 = 1;
pub(crate) const SELECTED: u32 = 3;
pub(crate) const LOST: u32 = 4;
pub(crate) const FAILED: u32 = 5;
/// Words per mesh vertex in a round's vertex records.
const VERTEX_WORDS: u64 = 14;
/// Words per patch in a round's patch words; the first is the patch's lock.
pub(crate) const PATCH_WORDS: usize = 5;

/// Entry points in dispatch order, and whether each runs one workgroup per
/// patch rather than one invocation per request.
const STAGES: [(&str, bool); 7] = [
    ("lead", true),
    ("bid", false),
    ("relay_forward", true),
    ("relay_back", true),
    ("check", false),
    ("claim", false),
    ("resolve", false),
];

const RESOURCE: &str = "cavity rounds";

/// The round pipelines and the bind group layouts they share. Create one per
/// device.
pub(crate) struct CavityRounds {
    empty: wgpu::BindGroupLayout,
    mesh: wgpu::BindGroupLayout,
    state: wgpu::BindGroupLayout,
    stages: Vec<wgpu::ComputePipeline>,
    conclude: wgpu::ComputePipeline,
    migrate: CavityMigrate,
}

/// One batch's requests and scratch words. Requests stay in `requests()`
/// between rounds; corners used by selected requests stay taken for the
/// whole batch.
pub(crate) struct CavityState {
    requests: wgpu::Buffer,
    pub(super) vertex_records: wgpu::Buffer,
    pub(super) patch_words: wgpu::Buffer,
    taken: wgpu::Buffer,
    /// As long as the mesh's masks; used by `topology::cavity_migrate`.
    pub(super) scratch: wgpu::Buffer,
    request_count: u32,
    seed: u32,
}

impl CavityRounds {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::BufferBindingType::{Storage, Uniform};
        let module = pipeline::shader_module(
            device,
            "MOPMesh cavity rounds",
            include_str!(concat!(env!("OUT_DIR"), "/cavity_rounds.wgsl")),
        );

        // Group 0 is unused, group 1 is `topology::patch_arena`, group 2 is
        // the round state.
        let (read, write) = (Storage { read_only: true }, Storage { read_only: false });
        let (empty, _) = pipeline::buffer_layout(device, "MOPMesh cavity rounds unused", &[]);
        let (mesh, _) =
            pipeline::buffer_layout(device, "MOPMesh cavity rounds mesh", &[read, read]);
        let (state, _) = pipeline::buffer_layout(
            device,
            "MOPMesh cavity rounds state",
            &[write, write, write, write, Uniform],
        );
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MOPMesh cavity rounds"),
            bind_group_layouts: &[Some(&empty), Some(&mesh), Some(&state)],
            immediate_size: 0,
        });
        let stage = |entry| {
            pipeline::compute_pipeline(
                device,
                "MOPMesh cavity round",
                &module,
                entry,
                Some(&layout),
            )
        };
        Self {
            empty,
            mesh,
            state,
            stages: STAGES.iter().map(|(entry, _)| stage(entry)).collect(),
            conclude: stage("conclude"),
            migrate: CavityMigrate::new(device),
        }
    }

    /// Records round `round` of `state`'s batch on `mesh`. Rounds must be
    /// recorded in order, starting at 0, with nothing else that changes
    /// `mesh` between a round's stages. With `migrate`, each selected
    /// request's acting patch also takes over the faces around its corners,
    /// or the request becomes FAILED; this counts as an edit of `mesh`.
    pub fn encode_round(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &mut GpuEditableMesh,
        state: &CavityState,
        round: u32,
        migrate: bool,
    ) -> Result<(), GpuError> {
        // Clear this round's vertex records, patch words and slot scratch.
        let limits = device.limits();
        let flat = pipeline::workgroup_grid(&limits, state.request_count, RESOURCE)?;
        let patches = pipeline::group_grid(&limits, mesh.patch_count(), RESOURCE)?;
        for buffer in [&state.vertex_records, &state.patch_words, &state.scratch] {
            encoder.clear_buffer(buffer, 0, None);
        }

        // Bind the mesh and this round's parameters.
        let round_seed = state.seed ^ (round + 1).wrapping_mul(0x9e37_79b9);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh cavity round parameters"),
            contents: bytemuck::cast_slice(&[state.seed, round_seed, state.request_count, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let empty = pipeline::bind_buffers(device, RESOURCE, &self.empty, &[]);
        let mesh_group =
            pipeline::bind_buffers(device, RESOURCE, &self.mesh, &[mesh.arena(), mesh.masks()]);
        let state_group = pipeline::bind_buffers(
            device,
            RESOURCE,
            &self.state,
            &[
                &state.requests,
                &state.vertex_records,
                &state.patch_words,
                &state.taken,
                &params,
            ],
        );

        // Run the stages in order.
        let groups = [&empty, &mesh_group, &state_group];
        for (pipeline, &(entry, per_patch)) in self.stages.iter().zip(&STAGES) {
            let grid = if per_patch { patches } else { flat };
            pipeline::dispatch_grid_groups(encoder, entry, pipeline, &groups, grid);
        }

        // Migrate, with failed requests concluded before anything is copied.
        if migrate {
            let (m, before, after) = (&self.migrate, BEFORE_CONCLUDE, AFTER_CONCLUDE);
            m.encode(device, encoder, mesh, state, before)?;
            pipeline::dispatch_grid_groups(encoder, "conclude", &self.conclude, &groups, flat);
            m.encode(device, encoder, mesh, state, after)?;
            mesh.mark_edited();
        }
        Ok(())
    }
}

impl CavityState {
    /// Uploads `requests`, REQUEST_WORDS words each, for a batch on `mesh`.
    /// `seed` orders requests that share an acting patch.
    pub fn new(
        device: &wgpu::Device,
        mesh: &GpuEditableMesh,
        requests: &[u32],
        seed: u32,
    ) -> Result<Self, GpuError> {
        // Check sizes before allocating.
        let limits = device.limits();
        let vertex_bytes = (u64::from(mesh.vertex_count()) * 4).max(4);
        let record_bytes = vertex_bytes * VERTEX_WORDS;
        let patch_bytes = (u64::from(mesh.patch_count()) * 4 * PATCH_WORDS as u64).max(4);
        let request_bytes = (requests.len() as u64 * 4).max(4);
        let scratch_bytes = mesh.masks().size();
        for bytes in [record_bytes, patch_bytes, request_bytes, scratch_bytes] {
            pipeline::check_binding(&limits, bytes, RESOURCE)?;
        }

        let scratch = |label, bytes| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mut words = requests.to_vec();
        if words.is_empty() {
            words.push(NONE);
        }
        Ok(Self {
            requests: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MOPMesh cavity requests"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            vertex_records: scratch("MOPMesh cavity vertex records", record_bytes),
            patch_words: scratch("MOPMesh cavity patch words", patch_bytes),
            taken: scratch("MOPMesh cavity taken vertices", vertex_bytes),
            scratch: scratch("MOPMesh cavity slot scratch", scratch_bytes),
            request_count: (requests.len() / REQUEST_WORDS) as u32,
            seed,
        })
    }

    pub fn requests(&self) -> &wgpu::Buffer {
        &self.requests
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::EditCapacity;
    use crate::gpu::bench_meshes;
    use crate::gpu::bench_migration::{Sim, priority};
    use crate::gpu::test_support;
    use std::collections::{BTreeSet, HashMap, HashSet};

    /// Predicts one round on the CPU: updates `words` (REQUEST_WORDS per
    /// request) and `taken` as the GPU should. Returns the selected requests.
    pub(in crate::gpu) fn predict_round(
        sim: &Sim,
        words: &mut [u32],
        taken: &mut HashSet<u32>,
        seed: u32,
        round: u32,
    ) -> Vec<usize> {
        let round_seed = seed ^ (round + 1).wrapping_mul(0x9e37_79b9);
        let bid = |p: u32| priority(p, round_seed | 0x8000_0000);
        let count = words.len() / REQUEST_WORDS;
        let corners = |words: &[u32], r: usize| -> [u32; 4] {
            let at = REQUEST_WORDS * r + 2;
            [words[at], words[at + 1], words[at + 2], words[at + 3]]
        };

        // Bid: lost requests, acting patches, and corner bids.
        let mut vertex_bid: HashMap<u32, u32> = HashMap::new();
        for r in 0..count {
            let at = REQUEST_WORDS * r;
            if words[at] != PENDING {
                continue;
            }
            let c = corners(words, r);
            if c.iter().any(|v| taken.contains(v)) {
                words[at] = LOST;
                continue;
            }
            let actor = sim.star_leaders(c);
            words[at + 1] = actor;
            for v in c {
                let b = vertex_bid.entry(v).or_default();
                *b = (*b).max(bid(actor));
            }
        }

        // Relay: lock words from each bid vertex's star, then lost vertices.
        let star: HashMap<u32, BTreeSet<u32>> = vertex_bid
            .keys()
            .map(|&v| (v, sim.footprint(&sim.cavity_faces(&[v]))))
            .collect();
        let mut lock: HashMap<u32, u32> = HashMap::new();
        for (&v, &b) in &vertex_bid {
            for &q in &star[&v] {
                let l = lock.entry(q).or_default();
                *l = (*l).max(b);
            }
        }
        let lost = |v: u32| star[&v].iter().any(|q| lock[q] != vertex_bid[&v]);

        // Check, claim and resolve.
        let mut won = Vec::new();
        for r in 0..count {
            let at = REQUEST_WORDS * r;
            let b = bid(words[at + 1]);
            if words[at] == PENDING
                && corners(words, r)
                    .iter()
                    .all(|&v| vertex_bid[&v] == b && !lost(v))
            {
                won.push(r);
            }
        }
        let mut claims: HashMap<u32, u32> = HashMap::new();
        for &r in &won {
            for v in corners(words, r) {
                let c = claims.entry(v).or_default();
                *c = (*c).max(priority(r as u32, seed));
            }
        }
        let mut selected = Vec::new();
        for r in won {
            let c = corners(words, r);
            if c.iter().all(|v| claims[v] == priority(r as u32, seed)) {
                words[REQUEST_WORDS * r] = SELECTED;
                taken.extend(c);
                selected.push(r);
            }
        }
        selected
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn rounds_select_exactly_the_predicted_requests() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let rounds = CavityRounds::new(&device);
        let seed = 7;
        for bench in bench_meshes::generated(4096) {
            for patch_size in [16u32, 64] {
                let context = format!("{} at patch size {patch_size}", bench.name);
                let sim = Sim::new(&bench.positions, &bench.triangles, patch_size);
                let mesh = MeshBuilder::new()
                    .patch_size(patch_size as usize)
                    .positions(&bench.positions)
                    .triangles(&bench.triangles)
                    .build()
                    .unwrap();
                let mut gpu =
                    GpuEditableMesh::upload(&device, &mesh, EditCapacity::default()).unwrap();

                // Request every edge with two faces.
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

                // Every round selects exactly the predicted requests, at
                // least one unless every pending request lost to a taken
                // corner, and acting patches' selected footprints never
                // overlap.
                let mut taken = HashSet::new();
                let mut round = 0;
                while words.chunks(REQUEST_WORDS).any(|w| w[0] == PENDING) {
                    assert!(round < 10_000, "{context}: rounds drain");
                    let selected = predict_round(&sim, &mut words, &mut taken, seed, round);
                    let still_pending = words.chunks(REQUEST_WORDS).any(|w| w[0] == PENDING);
                    assert!(
                        !selected.is_empty() || !still_pending,
                        "{context}: round {round} selects a request"
                    );
                    let mut by_actor: HashMap<u32, BTreeSet<u32>> = HashMap::new();
                    for &r in &selected {
                        let at = REQUEST_WORDS * r;
                        let faces = sim.cavity_faces(&words[at + 2..at + 6]);
                        by_actor
                            .entry(words[at + 1])
                            .or_default()
                            .extend(sim.footprint(&faces));
                    }
                    let mut claimed = BTreeSet::new();
                    for (actor, footprint) in by_actor {
                        assert!(footprint.contains(&actor), "{context}: actor in footprint");
                        assert!(
                            claimed.is_disjoint(&footprint),
                            "{context}: footprints overlap"
                        );
                        claimed.extend(footprint);
                    }

                    let mut encoder = device.create_command_encoder(&Default::default());
                    rounds
                        .encode_round(&device, &mut encoder, &mut gpu, &state, round, false)
                        .unwrap();
                    let read = test_support::read_words(&device, &queue, encoder, state.requests());
                    assert_eq!(read, words, "{context}: round {round}");
                    round += 1;
                }
                println!("{context}: {round} rounds");
            }
        }
    }
}
