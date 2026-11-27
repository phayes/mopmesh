//! CPU simulation of cross-patch flips by ownership migration, to size the
//! Phase 5 design before any GPU work: how many flips need another patch,
//! how many patches each transaction must lock, how many copies it imports,
//! whether the spare slot capacity holds, and how many bid rounds drain a
//! batch.
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --lib migration_simulation -- --ignored --nocapture`.
//! Without `MOPMESH_BENCH` it returns at once. Settings, all optional:
//! - `MOPMESH_BENCH_FACES`: comma-separated target face counts for the
//!   all-edges batch, default `524288`.
//! - `MOPMESH_BENCH_PATCHES`: comma-separated patch sizes, default `128,512`.
//! - `MOPMESH_BENCH_FOOTPRINT`: the most patches one transaction may lock,
//!   default `8`.
//! - `MOPMESH_BENCH_SIDES`: comma-separated skewed-grid sides for the
//!   Delaunay settle, default `513`.
//! - `MOPMESH_BENCH_ACTOR`: which patch makes each boundary flip: `edge`
//!   for the edge's owner, `corners` for the most common owner of its four
//!   corner vertices, `endpoint` for whichever endpoint owner owns more
//!   corners, or by default the patch owning most faces around them.
//! - `MOPMESH_BENCH_BIDS`: `relay` for relayed per-flip bids with no
//!   footprint cap; by default bounded batches capped at
//!   `MOPMESH_BENCH_FOOTPRINT`.
//!
//! The model follows the Phase 5 plan:
//! - An interior pass flips, greedily by priority, the requested edges whose
//!   four corners are interior to the edge's owner.
//! - Then, each round:
//!   - Every patch picks the highest-priority remaining boundary flips of the
//!     edges it owns, until their combined footprint would exceed the cap.
//!     The footprint is the owners of every face at the four corners, and of
//!     those faces' vertices and edges.
//!   - Each such patch bids one priority on every patch in its footprint. A
//!     patch wins if it holds the highest bid on all of them.
//!   - Each winner takes ownership of every face at its flips' corners, and of
//!     those faces' vertices and edges. It imports the copies it newly
//!     needs, then flips.
//!   - Every patch in a winner's footprint is cleaned up, keeping exactly the
//!     copies its owned faces need.
use super::bench_meshes;
use crate::MeshBuilder;
use std::collections::{BTreeSet, HashMap, HashSet};

const VERTEX: usize = 0;
const EDGE: usize = 1;
const FACE: usize = 2;
/// Spare slot capacity per patch, as [`EditCapacity::default`](super::EditCapacity).
const CAPACITY: f64 = 1.5;
/// Rounds after which leftover flips are counted, as a candidate default.
const ENCODED_ROUNDS: usize = 8;
/// Face owners counted per vertex by the star-leader rule; a vertex with
/// more is left out of the vote.
pub(super) const STAR_LEADER_SLOTS: usize = 4;
/// Rounds after which the simulation gives up draining a batch.
const MAX_ROUNDS: usize = 10_000;

/// The same hash as `topology::flip::flip_priority`.
pub(super) fn priority(index: u32, seed: u32) -> u32 {
    let mut x = index ^ seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// What happened to one requested flip.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Status {
    Pending,
    Flipped,
    LostClaim,
    Connected,
    OutOfCapacity,
    NotAnEdge,
}

/// Measurements gathered over one or more batches.
#[derive(Default)]
struct Stats {
    requested: usize,
    interior: usize,
    boundary: usize,
    flipped_interior: usize,
    flipped_boundary: usize,
    lost: usize,
    connected: usize,
    out_of_capacity: usize,
    /// Boundary flips still pending after `ENCODED_ROUNDS` rounds, per batch.
    left_after_encoded: usize,
    cavity_footprints: Vec<usize>,
    batch_footprints: Vec<usize>,
    batch_cavities: Vec<usize>,
    imported_faces: Vec<usize>,
    winners: Vec<usize>,
    contenders: Vec<usize>,
    rounds: Vec<usize>,
    /// Faces that changed owner at least once, and at least twice.
    moved_once: usize,
    moved_twice: usize,
    /// Highest peak slot use over capacity, per element kind.
    peak_ratio: [f64; 3],
    /// Copies needed after cleanup that the patch didn't hold: must stay 0.
    ribbon_gaps: usize,
}

/// Mesh topology, ownership and per-patch copies.
pub(super) struct Sim {
    triangles: Vec<[u32; 3]>,
    positions: Vec<[f32; 3]>,
    /// Faces around each vertex.
    star: Vec<Vec<u32>>,
    edge_of: HashMap<(u32, u32), u32>,
    ends: Vec<[u32; 2]>,
    owner: [Vec<u32>; 3],
    owned_faces: Vec<HashSet<u32>>,
    /// Copies each patch holds, per element kind, including owned ones.
    held: Vec<[HashSet<u32>; 3]>,
    capacity: Vec<[usize; 3]>,
    moves: Vec<u32>,
    peak_ratio: [f64; 3],
    /// Which patch makes each boundary flip.
    actor: Actor,
    /// Whether rounds use relayed per-flip bids instead of bounded batches.
    relay: bool,
}

/// Which patch makes a boundary flip.
#[derive(Clone, Copy)]
enum Actor {
    /// The owner of the flipped edge, as in RXMesh.
    Edge,
    /// The patch owning the most faces around the four corners.
    Majority,
    /// The most common owner among the four corner vertices.
    Corners,
    /// Whichever of the edge's two endpoint owners owns more of the four
    /// corners, the lower ID on ties. It holds both faces of the edge.
    Endpoint,
    /// The most common owner among the quad's two faces, five edges and four
    /// vertices, the lower ID on ties.
    Local,
    /// The patch owning the most faces summed over the corners' stars,
    /// leaving out stars with many owners (see `Sim::star_leaders`).
    Stars,
}

/// A patch's flips for one round: the patch, the faces around their
/// corners, their footprint, and the flips.
type Plan = (u32, BTreeSet<u32>, BTreeSet<u32>, Vec<(u32, Quad)>);

fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// A flippable edge `ab`'s corners `[a, b, c, d]`, where `c` is opposite the
/// face running a -> b and `d` the face running b -> a.
pub(super) struct Quad {
    pub corners: [u32; 4],
}

impl Sim {
    pub(super) fn new(positions: &[[f32; 3]], triangles: &[[u32; 3]], patch_size: u32) -> Self {
        Self::with_capacity(positions, triangles, patch_size, CAPACITY)
    }

    /// Like `new`, with `capacity` times as many slots per patch as it
    /// starts with, rounded up, as `EditCapacity` gives.
    pub(super) fn with_capacity(
        positions: &[[f32; 3]],
        triangles: &[[u32; 3]],
        patch_size: u32,
        capacity_factor: f64,
    ) -> Self {
        let mesh = MeshBuilder::new()
            .patch_size(patch_size as usize)
            .positions(positions)
            .triangles(triangles)
            .build()
            .unwrap();
        let patches = mesh.patches().len();

        // Global connectivity and initial owners, read from the patches.
        let mut star = vec![Vec::new(); positions.len()];
        for (f, t) in triangles.iter().enumerate() {
            for &v in t {
                star[v as usize].push(f as u32);
            }
        }
        let mut ends = vec![[0; 2]; mesh.edge_count()];
        let mut owner = [
            vec![0; positions.len()],
            vec![0; mesh.edge_count()],
            vec![0; triangles.len()],
        ];
        let mut held = Vec::with_capacity(patches);
        let mut capacity = Vec::with_capacity(patches);
        let mut owned_faces = vec![HashSet::new(); patches];
        for p in mesh.patches() {
            for (&e, ev) in p.edge_ids().iter().zip(p.edge_vertices()) {
                ends[e as usize] = ev.map(|v| p.vertex_ids()[v as usize]);
            }
            for (&v, h) in p.vertex_ids().iter().zip(p.vertex_owners()) {
                owner[VERTEX][v as usize] = h.patch;
            }
            for (&e, h) in p.edge_ids().iter().zip(p.edge_owners()) {
                owner[EDGE][e as usize] = h.patch;
            }
            for (&f, h) in p.face_ids().iter().zip(p.face_owners()) {
                owner[FACE][f as usize] = h.patch;
                if h.patch == p.id() {
                    owned_faces[p.id() as usize].insert(f);
                }
            }
            let ids = [p.vertex_ids(), p.edge_ids(), p.face_ids()];
            held.push(ids.map(|ids| ids.iter().copied().collect::<HashSet<u32>>()));
            capacity.push(ids.map(|ids| (ids.len() as f64 * capacity_factor).ceil() as usize));
        }
        let edge_of = ends
            .iter()
            .enumerate()
            .map(|(e, &[a, b])| (key(a, b), e as u32))
            .collect();
        Self {
            triangles: triangles.to_vec(),
            positions: positions.to_vec(),
            star,
            edge_of,
            ends,
            owner,
            owned_faces,
            held,
            capacity,
            moves: vec![0; triangles.len()],
            peak_ratio: [0.0; 3],
            actor: match std::env::var("MOPMESH_BENCH_ACTOR").as_deref() {
                Ok("edge") => Actor::Edge,
                Ok("corners") => Actor::Corners,
                Ok("endpoint") => Actor::Endpoint,
                Ok("local") => Actor::Local,
                Ok("stars") => Actor::Stars,
                _ => Actor::Majority,
            },
            relay: std::env::var("MOPMESH_BENCH_BIDS").is_ok_and(|b| b == "relay"),
        }
    }

    pub(super) fn patch_count(&self) -> usize {
        self.held.len()
    }

    pub(super) fn edge_count(&self) -> usize {
        self.ends.len()
    }

    /// Returns edge `e`'s quad, or `None` for a boundary edge.
    pub(super) fn quad(&self, e: u32) -> Option<Quad> {
        let [a, b] = self.ends[e as usize];
        let runs = |f: u32, u: u32, v: u32| {
            let t = self.triangles[f as usize];
            (0..3).any(|k| t[k] == u && t[(k + 1) % 3] == v)
        };
        let first = *self.star[a as usize].iter().find(|&&f| runs(f, a, b))?;
        let second = *self.star[a as usize].iter().find(|&&f| runs(f, b, a))?;
        let opposite = |f: u32| {
            *self.triangles[f as usize]
                .iter()
                .find(|&&x| x != a && x != b)
                .unwrap()
        };
        Some(Quad {
            corners: [a, b, opposite(first), opposite(second)],
        })
    }

    /// Returns the patch owning every face at `v`, if one patch does.
    pub(super) fn interior_to(&self, v: u32) -> Option<u32> {
        let faces = &self.star[v as usize];
        let p = self.owner[FACE][faces[0] as usize];
        faces
            .iter()
            .all(|&f| self.owner[FACE][f as usize] == p)
            .then_some(p)
    }

    fn face_edges(&self, f: u32) -> [u32; 3] {
        let t = self.triangles[f as usize];
        [0, 1, 2].map(|k| self.edge_of[&key(t[k], t[(k + 1) % 3])])
    }

    /// Every face at any of `corners`.
    pub(super) fn cavity_faces(&self, corners: &[u32]) -> BTreeSet<u32> {
        corners
            .iter()
            .flat_map(|&v| self.star[v as usize].iter().copied())
            .collect()
    }

    /// The owners of `faces` and of their vertices and edges.
    pub(super) fn footprint(&self, faces: &BTreeSet<u32>) -> BTreeSet<u32> {
        let mut patches = BTreeSet::new();
        for &f in faces {
            patches.insert(self.owner[FACE][f as usize]);
            for v in self.triangles[f as usize] {
                patches.insert(self.owner[VERTEX][v as usize]);
            }
            for e in self.face_edges(f) {
                patches.insert(self.owner[EDGE][e as usize]);
            }
        }
        patches
    }

    /// The copies a patch owning `owned` must hold: every face at a vertex of
    /// an owned face, with its edges and vertices.
    fn required<'a>(&self, owned: impl Iterator<Item = &'a u32>) -> [HashSet<u32>; 3] {
        let mut need = [HashSet::new(), HashSet::new(), HashSet::new()];
        for &f in owned {
            for v in self.triangles[f as usize] {
                for &g in &self.star[v as usize] {
                    if need[FACE].insert(g) {
                        need[VERTEX].extend(self.triangles[g as usize]);
                        need[EDGE].extend(self.face_edges(g));
                    }
                }
            }
        }
        need
    }

    /// Bounded batches: each patch takes its highest-priority flips,
    /// vertex-disjoint, until their footprint would exceed `max_footprint`
    /// patches, then bids one priority on every patch of that footprint.
    /// Returns the batches that hold every bid.
    fn batch_winners(
        &self,
        by_patch: HashMap<u32, Vec<(u32, Quad)>>,
        max_footprint: usize,
        round_seed: u32,
        stats: &mut Stats,
    ) -> Vec<Plan> {
        // Prepare each patch's batch.
        let mut plans = Vec::new();
        for (p, cands) in by_patch {
            let mut used = HashSet::new();
            let (mut faces, mut fp, mut flips) = (BTreeSet::new(), BTreeSet::from([p]), Vec::new());
            for (e, q) in cands {
                if q.corners.iter().any(|v| used.contains(v)) {
                    continue;
                }
                let mine = self.cavity_faces(&q.corners);
                let union: BTreeSet<u32> = fp.union(&self.footprint(&mine)).copied().collect();
                if !flips.is_empty() && union.len() > max_footprint {
                    break;
                }
                used.extend(q.corners);
                faces.extend(mine);
                fp = union;
                flips.push((e, q));
            }
            stats.batch_footprints.push(fp.len());
            stats.batch_cavities.push(flips.len());
            plans.push((p, faces, fp, flips));
        }

        // Bid one priority per patch and round on every footprint patch.
        let mut best = vec![0u32; self.patch_count()];
        for (p, _, fp, _) in &plans {
            for &q in fp {
                best[q as usize] = best[q as usize].max(priority(*p, round_seed));
            }
        }
        let winners: Vec<_> = plans
            .into_iter()
            .filter(|(p, _, fp, _)| {
                fp.iter()
                    .all(|&q| best[q as usize] == priority(*p, round_seed))
            })
            .collect();
        stats
            .contenders
            .push(best.iter().filter(|&&b| b != 0).count());
        stats.winners.push(winners.len());
        winners
    }

    /// Relayed bids, with no stored footprints and no cap: each pending flip
    /// bids its patch's round priority on its four corners; each bid vertex
    /// forwards its highest bid to every patch of its star's footprint; a
    /// vertex is lost if any of those patches holds a higher bid. A flip
    /// wins if all its corners carry its patch's bid and none is lost. Among
    /// one patch's winners, one round of flip priorities on shared corners
    /// picks the flips to make; the rest stay pending.
    fn relay_winners(
        &self,
        by_patch: HashMap<u32, Vec<(u32, Quad)>>,
        seed: u32,
        round_seed: u32,
        stats: &mut Stats,
    ) -> Vec<Plan> {
        // The top bit keeps every bid nonzero and distinct per patch.
        let bid = |p: u32| priority(p, round_seed | 0x8000_0000);

        // Corner bids, then the forward relay to each star's footprint.
        let mut vbid: HashMap<u32, u32> = HashMap::new();
        for (&p, flips) in &by_patch {
            for (_, q) in flips {
                for &v in &q.corners {
                    let b = vbid.entry(v).or_default();
                    *b = (*b).max(bid(p));
                }
            }
        }
        let star_footprint: HashMap<u32, BTreeSet<u32>> = vbid
            .keys()
            .map(|&v| (v, self.footprint(&self.cavity_faces(&[v]))))
            .collect();
        let mut lock: HashMap<u32, u32> = HashMap::new();
        for (&v, &b) in &vbid {
            for &q in &star_footprint[&v] {
                let l = lock.entry(q).or_default();
                *l = (*l).max(b);
            }
        }

        // Back relay and win check, then one round among each patch's winners.
        let lost = |v: u32| star_footprint[&v].iter().any(|q| lock[q] != vbid[&v]);
        stats.contenders.push(by_patch.len());
        let mut plans = Vec::new();
        let mut claimed = BTreeSet::new();
        for (p, flips) in by_patch {
            let won: Vec<_> = flips
                .into_iter()
                .filter(|(_, q)| q.corners.iter().all(|&v| vbid[&v] == bid(p) && !lost(v)))
                .collect();
            if won.is_empty() {
                continue;
            }
            let mut best: HashMap<u32, u32> = HashMap::new();
            for (e, q) in &won {
                for &v in &q.corners {
                    let b = best.entry(v).or_default();
                    *b = (*b).max(priority(*e, seed));
                }
            }
            let flips: Vec<_> = won
                .into_iter()
                .filter(|(e, q)| q.corners.iter().all(|v| best[v] == priority(*e, seed)))
                .collect();
            let mut faces = BTreeSet::new();
            for (_, q) in &flips {
                faces.extend(self.cavity_faces(&q.corners));
            }
            let fp = self.footprint(&faces);
            assert!(fp.contains(&p), "a flip's patch is in its footprint");
            assert!(
                claimed.is_disjoint(&fp),
                "winning patches share a footprint patch"
            );
            claimed.extend(fp.iter().copied());
            stats.batch_footprints.push(fp.len());
            stats.batch_cavities.push(flips.len());
            plans.push((p, faces, fp, flips));
        }
        stats.winners.push(plans.len());
        plans
    }

    /// The most common owner among `corners`, the lowest ID on ties.
    fn corner_majority(&self, corners: [u32; 4]) -> u32 {
        let owners = corners.map(|v| self.owner[VERTEX][v as usize]);
        let count = |p: u32| owners.iter().filter(|&&o| o == p).count();
        *owners
            .iter()
            .max_by_key(|&&p| (count(p), std::cmp::Reverse(p)))
            .unwrap()
    }

    /// Whichever owner of corners `a` and `b` owns more of the four corners,
    /// the lower ID on ties.
    /// Sums, over the four corners, how many faces of each corner's star
    /// each patch owns, and returns the patch with the highest sum, the
    /// lower ID on ties. A corner whose star has more than
    /// `STAR_LEADER_SLOTS` face owners is left out; if every corner is,
    /// returns [`endpoint_owner`](Self::endpoint_owner). Matches
    /// `cavity_actor` in `topology::cavity_rounds`.
    pub(super) fn star_leaders(&self, corners: [u32; 4]) -> u32 {
        let mut score: HashMap<u32, usize> = HashMap::new();
        for v in corners {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for &f in &self.star[v as usize] {
                *counts.entry(self.owner[FACE][f as usize]).or_default() += 1;
            }
            if counts.len() > STAR_LEADER_SLOTS {
                continue;
            }
            for (p, n) in counts {
                *score.entry(p).or_default() += n;
            }
        }
        score
            .into_iter()
            .max_by_key(|&(p, n)| (n, std::cmp::Reverse(p)))
            .map_or_else(|| self.endpoint_owner(corners), |(p, _)| p)
    }

    /// The most common owner among the quad's two faces, five edges and four
    /// vertices, the lower ID on ties.
    pub(super) fn local_plurality(&self, [a, b, c, d]: [u32; 4]) -> u32 {
        let mut votes = Vec::with_capacity(11);
        for v in [a, b, c, d] {
            votes.push(self.owner[VERTEX][v as usize]);
        }
        for (u, w) in [(a, b), (a, c), (b, c), (a, d), (b, d)] {
            votes.push(self.owner[EDGE][self.edge_of[&key(u, w)] as usize]);
        }
        for f in self.cavity_faces(&[a]) {
            let t = self.triangles[f as usize];
            if t.contains(&b) {
                votes.push(self.owner[FACE][f as usize]);
            }
        }
        let count = |p: u32| votes.iter().filter(|&&o| o == p).count();
        *votes
            .iter()
            .max_by_key(|&&p| (count(p), std::cmp::Reverse(p)))
            .unwrap()
    }

    pub(super) fn endpoint_owner(&self, corners: [u32; 4]) -> u32 {
        let owners = corners.map(|v| self.owner[VERTEX][v as usize]);
        let count = |p: u32| owners.iter().filter(|&&o| o == p).count();
        *owners[..2]
            .iter()
            .max_by_key(|&&p| (count(p), std::cmp::Reverse(p)))
            .unwrap()
    }

    /// The patch owning the most of `faces`, the lowest ID on ties.
    fn majority_owner(&self, faces: &BTreeSet<u32>) -> u32 {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &f in faces {
            *counts.entry(self.owner[FACE][f as usize]).or_default() += 1;
        }
        counts
            .into_iter()
            .max_by_key(|&(p, n)| (n, std::cmp::Reverse(p)))
            .unwrap()
            .0
    }

    fn edge_exists(&self, a: u32, b: u32) -> bool {
        self.edge_of.contains_key(&key(a, b))
    }

    /// Flips edge `e` with quad `[a, b, c, d]`, keeping its ID and its faces' IDs.
    fn flip(&mut self, e: u32, [a, b, c, d]: [u32; 4]) {
        let runs = |t: &[u32; 3], u: u32, v: u32| (0..3).any(|k| t[k] == u && t[(k + 1) % 3] == v);
        let first = *self.star[a as usize]
            .iter()
            .find(|&&f| runs(&self.triangles[f as usize], a, b))
            .unwrap();
        let second = *self.star[a as usize]
            .iter()
            .find(|&&f| runs(&self.triangles[f as usize], b, a))
            .unwrap();
        self.triangles[first as usize] = [a, d, c];
        self.triangles[second as usize] = [d, b, c];
        self.star[a as usize].retain(|&f| f != second);
        self.star[b as usize].retain(|&f| f != first);
        self.star[c as usize].push(second);
        self.star[d as usize].push(first);
        self.edge_of.remove(&key(a, b));
        self.edge_of.insert(key(c, d), e);
        self.ends[e as usize] = [c.min(d), c.max(d)];
    }

    /// Whether the edge with quad `[a, b, c, d]` breaks the 2D Delaunay
    /// condition: its opposite angles sum to more than 180 degrees.
    fn not_delaunay(&self, [a, b, c, d]: [u32; 4]) -> bool {
        let p = |v: u32| {
            let [x, y, _] = self.positions[v as usize];
            [f64::from(x), f64::from(y)]
        };
        let angle = |at: u32| {
            let [o, u, v] = [p(at), p(a), p(b)];
            let (ux, uy, vx, vy) = (u[0] - o[0], u[1] - o[1], v[0] - o[0], v[1] - o[1]);
            (ux * vy - uy * vx).abs().atan2(ux * vx + uy * vy)
        };
        angle(c) + angle(d) > std::f64::consts::PI + 1e-6
    }

    /// Makes patch `p` own `faces` and their vertices and edges, importing
    /// the copies it then needs. Returns how many vertex, edge and face
    /// copies it imported, or `None`, changing nothing, if they don't fit
    /// in its capacity.
    pub(super) fn migrate(&mut self, p: u32, faces: &BTreeSet<u32>) -> Option<[usize; 3]> {
        let pu = p as usize;
        let mut owned: HashSet<u32> = self.owned_faces[pu].clone();
        owned.extend(faces.iter().copied());
        let need = self.required(owned.iter());
        let missing: [usize; 3] = [0, 1, 2].map(|k| need[k].difference(&self.held[pu][k]).count());
        if (0..3).any(|k| self.held[pu][k].len() + missing[k] > self.capacity[pu][k]) {
            return None;
        }

        // Take ownership of the faces and their vertices and edges.
        for &f in faces {
            let old = self.owner[FACE][f as usize];
            if old != p {
                self.owned_faces[old as usize].remove(&f);
                self.moves[f as usize] += 1;
                self.owner[FACE][f as usize] = p;
            }
            for v in self.triangles[f as usize] {
                self.owner[VERTEX][v as usize] = p;
            }
            for e in self.face_edges(f) {
                self.owner[EDGE][e as usize] = p;
            }
        }
        self.owned_faces[pu] = owned;
        for k in 0..3 {
            self.held[pu][k].extend(need[k].iter().copied());
        }
        self.record_peak(pu);
        Some(missing)
    }

    /// Makes patch `q` hold exactly the copies its owned faces need. Returns
    /// how many needed copies it lacked, which must be 0.
    pub(super) fn clean(&mut self, q: u32) -> usize {
        let need = self.required(self.owned_faces[q as usize].iter());
        let gaps = (0..3)
            .map(|k| need[k].difference(&self.held[q as usize][k]).count())
            .sum();
        self.held[q as usize] = need;
        gaps
    }

    /// The vertex, edge or face IDs patch `p` holds copies of.
    pub(super) fn held(&self, p: u32, kind: usize) -> &HashSet<u32> {
        &self.held[p as usize][kind]
    }

    /// The patch owning element `id` of `kind`.
    pub(super) fn owner(&self, kind: usize, id: u32) -> u32 {
        self.owner[kind][id as usize]
    }

    /// The endpoints of edge `e`.
    pub(super) fn edge_ends(&self, e: u32) -> [u32; 2] {
        self.ends[e as usize]
    }

    /// The vertices of face `f`.
    pub(super) fn triangle(&self, f: u32) -> [u32; 3] {
        self.triangles[f as usize]
    }

    fn record_peak(&mut self, p: usize) {
        for k in 0..3 {
            let ratio = self.held[p][k].len() as f64 / self.capacity[p][k].max(1) as f64;
            self.peak_ratio[k] = self.peak_ratio[k].max(ratio);
        }
    }

    /// Runs one batch over `requested` edges with priority seed `seed`, and
    /// returns how many flipped.
    fn batch(
        &mut self,
        requested: &[u32],
        seed: u32,
        max_footprint: usize,
        stats: &mut Stats,
    ) -> usize {
        let mut status: HashMap<u32, Status> = HashMap::new();
        let mut taken: HashSet<u32> = HashSet::new();
        let mut order = requested.to_vec();
        order.sort_by_key(|&e| std::cmp::Reverse(priority(e, seed)));
        stats.requested += requested.len();

        // Interior pass: greedy by priority over flips inside one patch.
        let mut boundary = Vec::new();
        for &e in &order {
            let Some(q) = self.quad(e) else {
                status.insert(e, Status::NotAnEdge);
                continue;
            };
            let p = self.owner[EDGE][e as usize];
            if q.corners.iter().any(|&v| self.interior_to(v) != Some(p)) {
                boundary.push(e);
                status.insert(e, Status::Pending);
                continue;
            }
            stats.interior += 1;
            let [_, _, c, d] = q.corners;
            let s = if q.corners.iter().any(|v| taken.contains(v)) {
                Status::LostClaim
            } else if self.edge_exists(c, d) {
                Status::Connected
            } else {
                taken.extend(q.corners);
                self.flip(e, q.corners);
                stats.flipped_interior += 1;
                Status::Flipped
            };
            status.insert(e, s);
        }
        stats.boundary += boundary.len();
        for &e in &boundary {
            let q = self.quad(e).unwrap();
            stats
                .cavity_footprints
                .push(self.footprint(&self.cavity_faces(&q.corners)).len());
        }

        // Boundary rounds until every boundary flip is decided.
        let mut round = 0;
        loop {
            // Flips sharing a corner with a flip already made have lost.
            let mut pending = Vec::new();
            for &e in &boundary {
                if status[&e] != Status::Pending {
                    continue;
                }
                let q = self.quad(e).unwrap();
                if q.corners.iter().any(|v| taken.contains(v)) {
                    status.insert(e, Status::LostClaim);
                } else {
                    pending.push((e, q));
                }
            }
            if round == ENCODED_ROUNDS {
                stats.left_after_encoded += pending.len();
            }
            if pending.is_empty() || round == MAX_ROUNDS {
                break;
            }

            // Group the pending flips by the patch that would make them.
            let mut by_patch: HashMap<u32, Vec<(u32, Quad)>> = HashMap::new();
            for (e, q) in pending {
                let actor = match self.actor {
                    Actor::Edge => self.owner[EDGE][e as usize],
                    Actor::Majority => self.majority_owner(&self.cavity_faces(&q.corners)),
                    Actor::Corners => self.corner_majority(q.corners),
                    Actor::Endpoint => self.endpoint_owner(q.corners),
                    Actor::Local => self.local_plurality(q.corners),
                    Actor::Stars => self.star_leaders(q.corners),
                };
                by_patch.entry(actor).or_default().push((e, q));
            }
            let round_seed = seed ^ (round as u32 + 1).wrapping_mul(0x9e37_79b9);
            let winners = if self.relay {
                self.relay_winners(by_patch, seed, round_seed, stats)
            } else {
                self.batch_winners(by_patch, max_footprint, round_seed, stats)
            };

            // Act, then clean up every footprint patch.
            let mut dirty = BTreeSet::new();
            for (p, faces, fp, flips) in winners {
                let Some(missing) = self.migrate(p, &faces) else {
                    for (e, _) in flips {
                        status.insert(e, Status::OutOfCapacity);
                    }
                    continue;
                };
                stats.imported_faces.push(missing[FACE]);

                // Flip the planned edges, in priority order.
                for (e, q) in flips {
                    let [_, _, c, d] = q.corners;
                    if self.edge_exists(c, d) {
                        status.insert(e, Status::Connected);
                    } else {
                        taken.extend(q.corners);
                        self.flip(e, q.corners);
                        stats.flipped_boundary += 1;
                        status.insert(e, Status::Flipped);
                    }
                }
                dirty.extend(fp);
            }
            for q in dirty {
                stats.ribbon_gaps += self.clean(q);
            }
            round += 1;
        }
        stats.rounds.push(round);

        // Tally the outcomes.
        let mut flipped = 0;
        for s in status.values() {
            match s {
                Status::Flipped => flipped += 1,
                Status::LostClaim => stats.lost += 1,
                Status::Connected => stats.connected += 1,
                Status::OutOfCapacity => stats.out_of_capacity += 1,
                Status::Pending | Status::NotAnEdge => {}
            }
        }
        flipped
    }

    fn finish(&self, stats: &mut Stats) {
        stats.moved_once = self.moves.iter().filter(|&&m| m >= 1).count();
        stats.moved_twice = self.moves.iter().filter(|&&m| m >= 2).count();
        stats.peak_ratio = self.peak_ratio;
    }
}

/// Median, 99th percentile and maximum of `values`.
fn spread(values: &mut [usize]) -> String {
    if values.is_empty() {
        return "-".into();
    }
    values.sort_unstable();
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    format!("{}/{}/{}", at(0.5), at(0.99), values[values.len() - 1])
}

fn print_header() {
    println!(
        "| mesh | patch size | requested | interior | boundary | flipped interior | flipped boundary \
         | lost | connected | out of capacity | left after {ENCODED_ROUNDS} rounds | rounds (med/p99/max) \
         | cavity footprint | batch footprint | batch flips | imported faces | winners per round \
         | contenders per round | faces moved (≥1, ≥2) | peak use V/E/F | ribbon gaps |"
    );
    println!(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|---:|"
    );
}

fn print_row(name: &str, patch_size: u32, mut s: Stats) {
    println!(
        "| {name} | {patch_size} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}, {} | {:.2}/{:.2}/{:.2} | {} |",
        s.requested,
        s.interior,
        s.boundary,
        s.flipped_interior,
        s.flipped_boundary,
        s.lost,
        s.connected,
        s.out_of_capacity,
        s.left_after_encoded,
        spread(&mut s.rounds),
        spread(&mut s.cavity_footprints),
        spread(&mut s.batch_footprints),
        spread(&mut s.batch_cavities),
        spread(&mut s.imported_faces),
        spread(&mut s.winners),
        spread(&mut s.contenders),
        s.moved_once,
        s.moved_twice,
        s.peak_ratio[0],
        s.peak_ratio[1],
        s.peak_ratio[2],
        s.ribbon_gaps,
    );
}

/// The Delaunay example's skewed grid with `n` vertices per side.
fn skewed_grid(n: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut positions = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let h = (x * 7919 + y * 104_729) % 1000;
            let jitter = (h as f32 / 1000.0 - 0.5) * 0.2;
            positions.push([x as f32 - 0.4 * y as f32 + jitter, y as f32 - jitter, 0.0]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let a = y * n + x;
            triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]);
        }
    }
    (positions, triangles)
}

#[test]
#[ignore = "simulation; run in release with --ignored --nocapture"]
fn migration_simulation() {
    if std::env::var_os("MOPMESH_BENCH").is_none() {
        println!("set MOPMESH_BENCH=1 to run the migration simulation");
        return;
    }
    let numbers = |name: &str, default: &str| -> Vec<u32> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .expect("a comma-separated list of integers")
            })
            .collect()
    };
    let faces = numbers("MOPMESH_BENCH_FACES", "524288");
    let patches = numbers("MOPMESH_BENCH_PATCHES", "128,512");
    let max_footprint = numbers("MOPMESH_BENCH_FOOTPRINT", "8")[0] as usize;
    let sides = numbers("MOPMESH_BENCH_SIDES", "513");
    println!("# MOPMesh migration simulation (CPU), footprint cap {max_footprint}");
    if cfg!(debug_assertions) {
        println!("WARNING: debug build; slow");
    }

    // One batch requesting every edge, as in the flip benchmark.
    println!("\n## One batch requesting every edge\n");
    print_header();
    for mesh in faces.into_iter().flat_map(bench_meshes::generated) {
        for &patch_size in &patches {
            let mut sim = Sim::new(&mesh.positions, &mesh.triangles, patch_size);
            let mut stats = Stats::default();
            let edges: Vec<u32> = (0..sim.ends.len() as u32).collect();
            sim.batch(&edges, 0, max_footprint, &mut stats);
            sim.finish(&mut stats);
            print_row(&mesh.name, patch_size, stats);
        }
    }

    // Delaunay settle: each batch requests every edge that breaks the
    // Delaunay condition, until none does or a batch flips nothing.
    println!("\n## Delaunay settle (rows sum over all batches)\n");
    print_header();
    for &n in &sides {
        let (positions, triangles) = skewed_grid(n);
        for &patch_size in &patches {
            let mut sim = Sim::new(&positions, &triangles, patch_size);
            let mut stats = Stats::default();
            let mut batches = 0;
            for seed in 0..100u32 {
                let requested: Vec<u32> = (0..sim.ends.len() as u32)
                    .filter(|&e| sim.quad(e).is_some_and(|q| sim.not_delaunay(q.corners)))
                    .collect();
                if requested.is_empty() {
                    break;
                }
                batches += 1;
                if sim.batch(&requested, seed, max_footprint, &mut stats) == 0 {
                    break;
                }
            }
            sim.finish(&mut stats);
            let left = (0..sim.ends.len() as u32)
                .filter(|&e| sim.quad(e).is_some_and(|q| sim.not_delaunay(q.corners)))
                .count();
            let name = format!("skewed {n}x{n}, {batches} batches, {left} edges left");
            print_row(&name, patch_size, stats);
        }
    }
}
