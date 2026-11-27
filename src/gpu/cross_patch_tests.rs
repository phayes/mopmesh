//! GPU tests of edge flips across patch borders through `EdgeFlipQuery`,
//! against a CPU prediction built from the migration model.
use super::bench_meshes;
use super::bench_migration::{Sim, priority};
use super::cavity_migrate::tests::model_view;
use super::cavity_rounds::tests::predict_round;
use super::cavity_rounds::{DONE, FAILED, LOST, NONE, PENDING, REQUEST_WORDS};
use super::editable::inspect::{check_exact_ribbons, check_invariants, face_vertices, patch_views};
use super::test_support;
use super::{EdgeFlipQuery, EditCapacity, FlipStatus, GpuEditableMesh};
use crate::MeshBuilder;
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Rounds `EdgeFlipQuery` records for flips across patch borders.
const ROUNDS: u32 = 8;

/// Evidence that a run exercised the outcomes worth testing.
#[derive(Default, Debug)]
struct Seen {
    interior_flips: usize,
    /// Flips across patch borders that went ahead.
    cross_flips: usize,
    /// Flips across patch borders whose new edge already existed.
    cross_connected: usize,
    /// Flips across patch borders that lost a vertex to a flip inside a patch.
    lost_to_interior: usize,
    retries: usize,
    out_of_capacity: usize,
}

/// Predicts one `EdgeFlipQuery::encode` batch requesting `edges` with
/// priority seed `seed`, applying it to `sim`. Returns each edge's status,
/// `None` for edges not requested.
fn predict_batch(
    sim: &mut Sim,
    edges: &[u32],
    seed: u32,
    seen: &mut Seen,
) -> Vec<Option<FlipStatus>> {
    let mut status = vec![None; sim.edge_count()];
    let mut words = vec![NONE; REQUEST_WORDS * sim.edge_count()];
    let mut taken = HashSet::new();

    // Inside single patches: greedy by priority.
    let mut order: Vec<u32> = edges
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    order.sort_by_key(|&e| std::cmp::Reverse(priority(e, seed)));
    for &e in &order {
        let Some(q) = sim.quad(e) else {
            status[e as usize] = Some(FlipStatus::Boundary);
            continue;
        };
        let p = sim.owner(1, e);
        let [_, _, c, d] = q.corners;
        status[e as usize] = Some(
            if q.corners.iter().any(|&v| sim.interior_to(v) != Some(p)) {
                let at = REQUEST_WORDS * e as usize;
                words[at] = PENDING;
                words[at + 2..at + 6].copy_from_slice(&q.corners);
                FlipStatus::NotInterior
            } else if q.corners.iter().any(|v| taken.contains(v)) {
                FlipStatus::LostClaim
            } else if sim.edge_exists(c, d) {
                FlipStatus::Connected
            } else {
                taken.extend(q.corners);
                sim.flip(e, q.corners);
                seen.interior_flips += 1;
                FlipStatus::Flipped
            },
        );
    }
    let interior_taken = taken.clone();

    // Across patch borders: each round selects, migrates, then flips.
    for round in 0..ROUNDS {
        let selected = predict_round(sim, &mut words, &mut taken, seed, round);
        let mut by_actor: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &r in &selected {
            by_actor
                .entry(words[REQUEST_WORDS * r + 1])
                .or_default()
                .push(r);
        }
        let mut dirty = BTreeSet::new();
        let mut migrated = Vec::new();
        for (&actor, requests) in &by_actor {
            let mut faces = BTreeSet::new();
            for &r in requests {
                let at = REQUEST_WORDS * r + 2;
                faces.extend(sim.cavity_faces(&words[at..at + 4]));
            }
            dirty.extend(sim.footprint(&faces));
            if sim.migrate(actor, &faces).is_some() {
                migrated.extend(requests.iter().copied());
                continue;
            }
            for &r in requests {
                let at = REQUEST_WORDS * r;
                words[at] = FAILED;
                for v in &words[at + 2..at + 6] {
                    taken.remove(v);
                }
            }
        }
        for q in dirty {
            assert_eq!(sim.clean(q), 0, "model ribbon of patch {q}");
        }
        for r in migrated {
            let at = REQUEST_WORDS * r;
            let corners: [u32; 4] = words[at + 2..at + 6].try_into().unwrap();
            words[at] = DONE;
            status[r] = Some(if sim.edge_exists(corners[2], corners[3]) {
                for v in corners {
                    taken.remove(&v);
                }
                seen.cross_connected += 1;
                FlipStatus::Connected
            } else {
                sim.flip(r as u32, corners);
                seen.cross_flips += 1;
                FlipStatus::Flipped
            });
        }
    }

    // Edges still waiting report why.
    for (e, s) in status.iter_mut().enumerate() {
        if *s != Some(FlipStatus::NotInterior) {
            continue;
        }
        let at = REQUEST_WORDS * e;
        *s = Some(match words[at] {
            LOST => {
                let corners = &words[at + 2..at + 6];
                if corners.iter().any(|v| interior_taken.contains(v)) {
                    seen.lost_to_interior += 1;
                }
                FlipStatus::LostClaim
            }
            FAILED => {
                seen.out_of_capacity += 1;
                FlipStatus::OutOfCapacity
            }
            _ => {
                seen.retries += 1;
                FlipStatus::Retry
            }
        });
    }
    status
}

/// Runs batches requesting every edge on each generated mesh at patch sizes
/// 16 and 64, with `capacity` times the starting slots. Every batch's
/// statuses, patches and faces must match the prediction exactly, and the
/// arena must keep every invariant.
fn flip_batches(capacity: f32) -> Seen {
    let (device, queue) = test_support::device(wgpu::Limits::default());
    let query = EdgeFlipQuery::new(&device);
    let read = |buffer: &wgpu::Buffer| {
        let encoder = device.create_command_encoder(&Default::default());
        test_support::read_words(&device, &queue, encoder, buffer)
    };
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
            let edges: Vec<u32> = (0..gpu.edge_count()).collect();

            for batch in 0..4 {
                let context = format!("{} at patch size {patch_size}, batch {batch}", bench.name);
                let seed = gpu.version() as u32;
                let expected = predict_batch(&mut sim, &edges, seed, &mut seen);

                let mut encoder = device.create_command_encoder(&Default::default());
                let statuses = query
                    .encode(&device, &mut encoder, &mut gpu, &edges)
                    .unwrap();
                queue.submit([encoder.finish()]);
                let words = read(statuses.buffer());
                for (e, want) in expected.iter().enumerate() {
                    assert_eq!(
                        FlipStatus::from_word(words[e]),
                        *want,
                        "{context}: edge {e}"
                    );
                }

                let arena = read(gpu.arena());
                let masks = read(gpu.masks());
                check_invariants(&arena, &masks, &read(gpu.maps()));
                check_exact_ribbons(&arena, &masks);
                for (p, view) in patch_views(&arena, &masks).into_iter().enumerate() {
                    assert_eq!(view, model_view(&sim, p as u32), "{context}: patch {p}");
                }
                for f in 0..bench.triangles.len() as u32 {
                    assert_eq!(
                        face_vertices(&arena, f),
                        sim.triangle(f),
                        "{context}: face {f}"
                    );
                }
            }
        }
    }
    println!("{seen:?}");
    seen
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn cross_patch_flips_match_the_prediction() {
    let seen = flip_batches(2.0);
    assert!(seen.cross_flips > 0, "some flip crosses patch borders");
    assert!(seen.interior_flips > 0, "some flip stays inside a patch");
    assert!(
        seen.cross_connected > 0,
        "some cross-patch flip finds its edge"
    );
    assert!(
        seen.lost_to_interior > 0,
        "some cross-patch flip loses to an interior one"
    );
    assert!(seen.retries > 0, "some cross-patch flip runs out of rounds");
}

#[test]
#[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
fn cross_patch_flips_without_room_report_out_of_capacity() {
    let seen = flip_batches(1.0);
    assert!(
        seen.out_of_capacity > 0,
        "some acting patch runs out of room"
    );
}
