//! CPU simulation of edge splits and collapses, to size the Phase 6 design
//! before any GPU work: how uneven patch sizes are and what uniform patch
//! capacity would cost, whether a collapse may lock only the removed vertex
//! and its neighbours, how large collapse footprints get, and how often a
//! patch would run out of room and need slicing.
//!
//! Run: `MOPMESH_BENCH=1 cargo test --release --lib growth_simulation -- --ignored --nocapture`.
//! Without `MOPMESH_BENCH` it returns at once. Settings, all optional:
//! - `MOPMESH_BENCH_FACES`: comma-separated target face counts, default
//!   `32768`.
//! - `MOPMESH_BENCH_PATCHES`: comma-separated patch sizes, default
//!   `64,128,512`.
//! - `MOPMESH_BENCH_CAPACITY`: spare-slot factor per patch, default `2`.
//! - `MOPMESH_BENCH_BATCHES`: the most batches per run, default `12`; `0`
//!   prints only the patch-size table.
//! - `MOPMESH_BENCH_MESH`: runs only meshes whose name contains it.
//! - `MOPMESH_BENCH_LAYOUT_FACES`: comma-separated extra face counts for the
//!   patch-size table only.
//!
//! The model extends the migration simulation:
//! - A split of edge `ab` with quad `[a, b, c, d]` adds a vertex `m` at its
//!   middle, keeps the edge ID for `am` and each old face ID for one child,
//!   and adds edges `mb`, `mc`, `md` and two faces. Its corners are the quad.
//! - A collapse of edge `ab` keeps `a`, moves it to the middle, and removes
//!   `b`, edges `ab`, `bc`, `bd` and the edge's two faces; every other face
//!   and edge at `b` is renamed to `a` in place. It needs an interior edge
//!   whose endpoints aren't on the boundary, the link condition, and `c` and
//!   `d` with more than three faces.
//! - Each batch runs an interior pass, greedy by priority, over edits whose
//!   corners are all interior to the edge's owner, then relayed bid rounds
//!   for the rest, as the GPU's cross-patch flips do. A winner migrates the
//!   faces around its corners, checks room for what the edit adds, edits,
//!   and every patch in its footprint is cleaned.
use super::{
    EDGE, ENCODED_ROUNDS, FACE, MAX_ROUNDS, STAR_LEADER_SLOTS, Sim, VERTEX, bench_meshes, key,
    priority, skewed_grid, spread,
};
use crate::gpu::patch_layout;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Marks a removed element's owner and endpoints.
const DEAD: u32 = u32::MAX;
/// Patches reserved per initial patch for slicing, as planned.
const POOL_FACTOR: u64 = 2;
/// Words in a patch record, as `patch_layout::RECORD_WORDS`.
const RECORD_WORDS: u64 = 16;

/// A pending edit: the edge, its quad and the vertices it locks.
type Edit = (u32, [u32; 4], Vec<u32>);
/// A mesh's name, positions and triangles.
type Named = (String, Vec<[f32; 3]>, Vec<[u32; 3]>);

/// The edit a batch makes.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    Split,
    Collapse,
}

/// Which vertices a collapse of `ab` into `a` locks and migrates.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Corners {
    /// `b` and every vertex sharing a face with it.
    Removed,
    /// `a`, `b` and every vertex sharing a face with either.
    Both,
    /// Only the quad `[a, b, c, d]`, as a flip: too small, so the stale-copy
    /// and exactness checks must catch it.
    Quad,
}

/// What happened to one requested edit.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Status {
    Pending,
    Applied,
    LostClaim,
    Rejected,
    OutOfCapacity,
    NotAnEdge,
}

/// Measurements gathered over a run of batches.
#[derive(Default)]
struct Stats {
    batches: usize,
    requested: usize,
    interior: usize,
    applied: usize,
    lost: usize,
    rejected: usize,
    out_of_capacity: usize,
    /// Distinct patches with an edit out of room, summed over batches: how
    /// many slices the run would have asked for.
    patches_out_of_room: usize,
    left_after_encoded: usize,
    rounds: Vec<usize>,
    corners: Vec<usize>,
    footprints: Vec<usize>,
    winners: Vec<usize>,
    /// Copies of elements an edit changes held by a patch other than the
    /// editing one, at the time of the edit: must stay 0.
    stale: usize,
    /// Copies a cleaned patch needed but lacked: must stay 0.
    ribbon_gaps: usize,
    /// Patches whose copies differ from what their owned faces need, checked
    /// over every patch after each batch: must stay 0.
    inexact: usize,
    faces_before: usize,
    faces_after: usize,
}

impl Sim {
    /// Whether edge `e` is still in the mesh.
    fn edge_alive(&self, e: u32) -> bool {
        self.ends[e as usize][0] != DEAD
    }

    /// Every vertex sharing a face with `v`.
    fn link(&self, v: u32) -> BTreeSet<u32> {
        self.star[v as usize]
            .iter()
            .flat_map(|&f| self.triangles[f as usize])
            .filter(|&w| w != v)
            .collect()
    }

    /// Whether `v` has an edge with only one face.
    fn on_boundary(&self, v: u32) -> bool {
        let mut count: HashMap<u32, usize> = HashMap::new();
        for &f in &self.star[v as usize] {
            for w in self.triangles[f as usize] {
                if w != v {
                    *count.entry(w).or_default() += 1;
                }
            }
        }
        count.values().any(|&n| n == 1)
    }

    fn edge_length(&self, e: u32) -> f64 {
        let [a, b] = self.ends[e as usize];
        let (p, q) = (self.positions[a as usize], self.positions[b as usize]);
        (0..3)
            .map(|k| f64::from(p[k] - q[k]).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn mean_edge_length(&self) -> f64 {
        let alive: Vec<u32> = (0..self.ends.len() as u32)
            .filter(|&e| self.edge_alive(e))
            .collect();
        alive.iter().map(|&e| self.edge_length(e)).sum::<f64>() / alive.len() as f64
    }

    fn live_faces(&self) -> usize {
        self.owner[FACE].iter().filter(|&&o| o != DEAD).count()
    }

    /// Whether collapsing the edge with quad `[a, b, c, d]` keeps the mesh a
    /// manifold: neither endpoint on the boundary, `a` and `b` share only
    /// `c` and `d`, and `c` and `d` keep at least three faces.
    fn collapse_ok(&self, [a, b, c, d]: [u32; 4]) -> bool {
        if self.on_boundary(a) || self.on_boundary(b) {
            return false;
        }
        let shared: BTreeSet<u32> = self.link(a).intersection(&self.link(b)).copied().collect();
        shared == BTreeSet::from([c, d])
            && self.star[c as usize].len() > 3
            && self.star[d as usize].len() > 3
    }

    /// The vertices an edit of the edge with quad `quad` locks, sorted.
    fn edit_corners(&self, op: Op, quad: [u32; 4], corners: Corners) -> Vec<u32> {
        let [a, b, ..] = quad;
        let set: BTreeSet<u32> = match (op, corners) {
            (Op::Split, _) | (Op::Collapse, Corners::Quad) => quad.into_iter().collect(),
            (Op::Collapse, Corners::Removed) => {
                let mut s = self.link(b);
                s.insert(b);
                s
            }
            (Op::Collapse, Corners::Both) => {
                let mut s = self.link(a);
                s.extend(self.link(b));
                s.extend([a, b]);
                s
            }
        };
        set.into_iter().collect()
    }

    /// The star-leader rule over any number of corners, as `star_leaders`;
    /// falls back to the owner of `fallback`.
    fn leader(&self, corners: &[u32], fallback: u32) -> u32 {
        let mut score: BTreeMap<u32, usize> = BTreeMap::new();
        for &v in corners {
            let mut counts: BTreeMap<u32, usize> = BTreeMap::new();
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
            .map_or(self.owner[VERTEX][fallback as usize], |(p, _)| p)
    }

    /// Whether patch `p` has room to own `faces` as well as its own, and to
    /// add `new` vertices, edges and faces.
    fn room(&self, p: u32, faces: &BTreeSet<u32>, new: [usize; 3]) -> bool {
        let pu = p as usize;
        let mut owned = self.owned_faces[pu].clone();
        owned.extend(faces.iter().copied());
        let need = self.required(owned.iter());
        (0..3).all(|k| {
            let missing = need[k].difference(&self.held[pu][k]).count();
            self.held[pu][k].len() + missing + new[k] <= self.capacity[pu][k]
        })
    }

    /// The two faces of edge `ab`: the one running a -> b, then b -> a.
    fn edge_faces(&self, a: u32, b: u32) -> (u32, u32) {
        let runs = |f: u32, u: u32, v: u32| {
            let t = self.triangles[f as usize];
            (0..3).any(|k| t[k] == u && t[(k + 1) % 3] == v)
        };
        let star = &self.star[a as usize];
        (
            *star.iter().find(|&&f| runs(f, a, b)).unwrap(),
            *star.iter().find(|&&f| runs(f, b, a)).unwrap(),
        )
    }

    /// The vertices, edges and faces an edit rewrites or removes.
    fn changed(&self, op: Op, e: u32, [a, b, ..]: [u32; 4]) -> [Vec<u32>; 3] {
        match op {
            Op::Split => {
                let (f1, f2) = self.edge_faces(a, b);
                [vec![], vec![e], vec![f1, f2]]
            }
            Op::Collapse => {
                let edges = self
                    .link(b)
                    .into_iter()
                    .map(|w| self.edge_of[&key(b, w)])
                    .collect();
                [vec![b], edges, self.star[b as usize].clone()]
            }
        }
    }

    /// Counts copies of `changed` held by patches other than `p`.
    fn stale_copies(&self, p: u32, changed: &[Vec<u32>; 3]) -> usize {
        (0..self.patch_count() as u32)
            .filter(|&q| q != p)
            .map(|q| {
                (0..3)
                    .map(|k| {
                        changed[k]
                            .iter()
                            .filter(|id| self.held[q as usize][k].contains(id))
                            .count()
                    })
                    .sum::<usize>()
            })
            .sum()
    }

    /// Splits edge `e` with quad `[a, b, c, d]` in patch `p`, which must own
    /// every face at the corners. Returns the new vertex.
    fn split(&mut self, p: u32, e: u32, [a, b, c, d]: [u32; 4]) -> u32 {
        let (f1, f2) = self.edge_faces(a, b);
        let pu = p as usize;

        // The new vertex in the middle of the edge.
        let m = self.positions.len() as u32;
        let (pa, pb) = (self.positions[a as usize], self.positions[b as usize]);
        self.positions
            .push([0, 1, 2].map(|k| (pa[k] + pb[k]) / 2.0));
        self.owner[VERTEX].push(p);

        // Edge `e` becomes `am`; add `mb`, `mc` and `md`.
        self.edge_of.remove(&key(a, b));
        self.edge_of.insert(key(a, m), e);
        self.ends[e as usize] = [a.min(m), a.max(m)];
        let mut new_edges = Vec::new();
        for w in [b, c, d] {
            let id = self.ends.len() as u32;
            self.ends.push([w.min(m), w.max(m)]);
            self.edge_of.insert(key(w, m), id);
            self.owner[EDGE].push(p);
            new_edges.push(id);
        }

        // Each old face keeps the half at `a`; the halves at `b` are new.
        let g1 = self.triangles.len() as u32;
        let g2 = g1 + 1;
        self.triangles[f1 as usize] = [a, m, c];
        self.triangles[f2 as usize] = [m, a, d];
        self.triangles.extend([[m, b, c], [b, m, d]]);
        self.owner[FACE].extend([p, p]);
        self.moves.extend([0, 0]);
        self.owned_faces[pu].extend([g1, g2]);

        // Stars and the editing patch's copies.
        self.star[b as usize].retain(|&f| f != f1 && f != f2);
        self.star[b as usize].extend([g1, g2]);
        self.star[c as usize].push(g1);
        self.star[d as usize].push(g2);
        self.star.push(vec![f1, g1, f2, g2]);
        self.held[pu][VERTEX].insert(m);
        self.held[pu][EDGE].extend(new_edges);
        self.held[pu][FACE].extend([g1, g2]);
        self.record_peak(pu);
        m
    }

    /// Collapses edge `e` with quad `[a, b, c, d]` into `a`, which moves to
    /// the edge's middle. The editing patch must own every face at `b` and
    /// its neighbours.
    fn collapse(&mut self, e: u32, [a, b, c, d]: [u32; 4]) {
        let (f1, f2) = self.edge_faces(a, b);
        let link = self.link(b);

        // Rename `b` to `a` in the faces and edges that stay.
        for f in self.star[b as usize].clone() {
            if f == f1 || f == f2 {
                continue;
            }
            for v in &mut self.triangles[f as usize] {
                if *v == b {
                    *v = a;
                }
            }
            self.star[a as usize].push(f);
        }
        for &w in &link {
            if [a, c, d].contains(&w) {
                continue;
            }
            let id = self.edge_of.remove(&key(b, w)).unwrap();
            assert!(
                self.edge_of.insert(key(a, w), id).is_none(),
                "the link condition rules out duplicate edges"
            );
            self.ends[id as usize] = [a.min(w), a.max(w)];
        }

        // Remove `b`, edges `ab`, `bc`, `bd` and the edge's two faces.
        for w in [a, c, d] {
            let id = self.edge_of.remove(&key(b, w)).unwrap();
            self.ends[id as usize] = [DEAD; 2];
            self.owner[EDGE][id as usize] = DEAD;
        }
        debug_assert!(!self.edge_alive(e));
        for f in [f1, f2] {
            let o = self.owner[FACE][f as usize];
            self.owned_faces[o as usize].remove(&f);
            self.owner[FACE][f as usize] = DEAD;
            self.triangles[f as usize] = [DEAD; 3];
        }
        self.star[a as usize].retain(|&f| f != f1 && f != f2);
        self.star[c as usize].retain(|&f| f != f1);
        self.star[d as usize].retain(|&f| f != f2);
        self.star[b as usize].clear();
        self.owner[VERTEX][b as usize] = DEAD;
        let (pa, pb) = (self.positions[a as usize], self.positions[b as usize]);
        self.positions[a as usize] = [0, 1, 2].map(|k| (pa[k] + pb[k]) / 2.0);
    }

    /// Panics unless stars, edges and faces agree: every live face's edges
    /// exist, every star lists exactly the live faces at its vertex, and
    /// every edge entry matches its endpoints.
    fn check_mesh(&self) {
        let mut star: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); self.star.len()];
        for (f, t) in self.triangles.iter().enumerate() {
            if t[0] == DEAD {
                continue;
            }
            assert!(
                t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
                "face {f} {t:?} is degenerate"
            );
            for k in 0..3 {
                star[t[k] as usize].insert(f as u32);
                assert!(
                    self.edge_of.contains_key(&key(t[k], t[(k + 1) % 3])),
                    "face {f} {t:?} lacks an edge"
                );
            }
        }
        for (v, s) in self.star.iter().enumerate() {
            let listed: BTreeSet<u32> = s.iter().copied().collect();
            assert_eq!(listed.len(), s.len(), "vertex {v} lists a face twice");
            assert_eq!(listed, star[v], "vertex {v}'s star is wrong");
        }
        for (&(a, b), &e) in &self.edge_of {
            assert_eq!(
                self.ends[e as usize],
                [a, b],
                "edge {e} has stale endpoints"
            );
        }
    }

    /// Gives every patch the same capacity per kind: `factor` times the
    /// largest patch's copies, rounded up.
    fn uniform_capacity(&mut self, factor: f64) {
        let largest = [0, 1, 2].map(|k| self.held.iter().map(|h| h[k].len()).max().unwrap());
        let cap = largest.map(|n| (n as f64 * factor).ceil() as usize);
        self.capacity.fill(cap);
    }

    /// Relayed bids over requests with any number of corners, as
    /// `relay_winners`. Returns each winning patch with its edits.
    fn relay_edits(
        &self,
        by_patch: BTreeMap<u32, Vec<Edit>>,
        seed: u32,
        round_seed: u32,
        stats: &mut Stats,
    ) -> Vec<(u32, Vec<Edit>)> {
        let bid = |p: u32| priority(p, round_seed | 0x8000_0000);

        // Corner bids, then the forward relay to each star's footprint.
        let mut vbid: HashMap<u32, u32> = HashMap::new();
        for (&p, edits) in &by_patch {
            for (_, _, corners) in edits {
                for &v in corners {
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
        let mut plans = Vec::new();
        let mut claimed = BTreeSet::new();
        for (p, edits) in by_patch {
            let won: Vec<_> = edits
                .into_iter()
                .filter(|(_, _, cs)| cs.iter().all(|&v| vbid[&v] == bid(p) && !lost(v)))
                .collect();
            if won.is_empty() {
                continue;
            }
            let mut best: HashMap<u32, u32> = HashMap::new();
            for (e, _, cs) in &won {
                for &v in cs {
                    let b = best.entry(v).or_default();
                    *b = (*b).max(priority(*e, seed));
                }
            }
            let edits: Vec<_> = won
                .into_iter()
                .filter(|(e, _, cs)| cs.iter().all(|v| best[v] == priority(*e, seed)))
                .collect();
            let mut faces = BTreeSet::new();
            for (_, _, cs) in &edits {
                faces.extend(self.cavity_faces(cs));
            }
            let fp = self.footprint(&faces);
            assert!(
                claimed.is_disjoint(&fp),
                "winning patches share a footprint patch"
            );
            claimed.extend(fp.iter().copied());
            stats.footprints.push(fp.len());
            plans.push((p, edits));
        }
        stats.winners.push(plans.len());
        plans
    }

    /// Checks, decides and makes one edit in patch `p`, taking ownership of
    /// every face at `corners`. On success, adds `p` and the elements the
    /// edit rewrote or removed to `edited`, for a stale-copy check after
    /// cleanup. Returns its status.
    #[allow(clippy::too_many_arguments)]
    fn make_edit(
        &mut self,
        op: Op,
        p: u32,
        e: u32,
        quad: [u32; 4],
        corners: &[u32],
        taken: &mut HashSet<u32>,
        edited: &mut Vec<(u32, [Vec<u32>; 3])>,
    ) -> Status {
        let new = match op {
            Op::Split => [1, 3, 2],
            Op::Collapse => [0; 3],
        };
        if !self.room(p, &self.cavity_faces(corners), new) {
            return Status::OutOfCapacity;
        }
        self.migrate(p, &self.cavity_faces(corners))
            .expect("room covers migration");

        // Recheck after migration, as the GPU will: an edit whose corners
        // missed part of its cavity may have been changed by another.
        match self.quad(e) {
            Some(q) if q.corners == quad => {}
            _ => return Status::NotAnEdge,
        }
        if op == Op::Collapse && !self.collapse_ok(quad) {
            return Status::Rejected;
        }
        edited.push((p, self.changed(op, e, quad)));
        taken.extend(corners.iter().copied());
        match op {
            Op::Split => {
                let m = self.split(p, e, quad);
                taken.insert(m);
            }
            Op::Collapse => self.collapse(e, quad),
        }
        Status::Applied
    }

    /// The edge's quad and corners if it can be edited now, else its status.
    fn request(&self, op: Op, e: u32, mode: Corners) -> Result<([u32; 4], Vec<u32>), Status> {
        if !self.edge_alive(e) {
            return Err(Status::NotAnEdge);
        }
        let Some(q) = self.quad(e) else {
            return Err(Status::NotAnEdge);
        };
        if op == Op::Collapse && !self.collapse_ok(q.corners) {
            return Err(Status::Rejected);
        }
        Ok((q.corners, self.edit_corners(op, q.corners, mode)))
    }

    /// Runs one batch of `op` over `requested` edges with priority seed
    /// `seed`, and returns how many edits it made.
    fn growth_batch(
        &mut self,
        op: Op,
        requested: &[u32],
        seed: u32,
        mode: Corners,
        stats: &mut Stats,
    ) -> usize {
        let mut status: HashMap<u32, Status> = HashMap::new();
        let mut taken: HashSet<u32> = HashSet::new();
        let mut out_of_room: BTreeSet<u32> = BTreeSet::new();
        let mut order = requested.to_vec();
        order.sort_by_key(|&e| std::cmp::Reverse(priority(e, seed)));
        stats.requested += requested.len();
        stats.batches += 1;

        // Interior pass: greedy by priority over edits inside one patch.
        let mut boundary = Vec::new();
        for &e in &order {
            let (quad, corners) = match self.request(op, e, mode) {
                Ok(r) => r,
                Err(s) => {
                    status.insert(e, s);
                    continue;
                }
            };
            stats.corners.push(corners.len());
            let p = self.owner[EDGE][e as usize];
            if corners.iter().any(|&v| self.interior_to(v) != Some(p)) {
                boundary.push(e);
                status.insert(e, Status::Pending);
                continue;
            }
            stats.interior += 1;
            let mut edited = Vec::new();
            let s = if corners.iter().any(|v| taken.contains(v)) {
                Status::LostClaim
            } else {
                self.make_edit(op, p, e, quad, &corners, &mut taken, &mut edited)
            };
            if s == Status::OutOfCapacity {
                out_of_room.insert(p);
            }

            // The edit frees what it removed; no other patch may hold a copy.
            if s == Status::Applied {
                stats.ribbon_gaps += self.clean(p);
            }
            for (p, changed) in &edited {
                stats.stale += self.stale_copies(*p, changed);
            }
            status.insert(e, s);
        }

        // Boundary rounds until every boundary edit is decided.
        let mut round = 0;
        loop {
            let mut pending = Vec::new();
            for &e in &boundary {
                if status[&e] != Status::Pending {
                    continue;
                }
                match self.request(op, e, mode) {
                    Err(s) => {
                        status.insert(e, s);
                    }
                    Ok((_, cs)) if cs.iter().any(|v| taken.contains(v)) => {
                        status.insert(e, Status::LostClaim);
                    }
                    Ok((quad, cs)) => pending.push((e, quad, cs)),
                }
            }
            if round == ENCODED_ROUNDS {
                stats.left_after_encoded += pending.len();
            }
            if pending.is_empty() || round == MAX_ROUNDS {
                break;
            }

            // Group by acting patch, bid, then act and clean up.
            let mut by_patch: BTreeMap<u32, Vec<_>> = BTreeMap::new();
            for (e, quad, cs) in pending {
                let actor = self.leader(&cs, quad[0]);
                by_patch.entry(actor).or_default().push((e, quad, cs));
            }
            let round_seed = seed ^ (round as u32 + 1).wrapping_mul(0x9e37_79b9);
            let winners = self.relay_edits(by_patch, seed, round_seed, stats);
            let mut dirty = BTreeSet::new();
            let mut edited = Vec::new();
            for (p, edits) in winners {
                for (e, quad, cs) in edits {
                    dirty.extend(self.footprint(&self.cavity_faces(&cs)));
                    let s = self.make_edit(op, p, e, quad, &cs, &mut taken, &mut edited);
                    if s == Status::OutOfCapacity {
                        out_of_room.insert(p);
                    }
                    status.insert(e, s);
                }
            }
            for q in dirty {
                stats.ribbon_gaps += self.clean(q);
            }

            // After cleanup, no other patch may hold a copy of what an edit
            // rewrote or removed.
            for (p, changed) in &edited {
                stats.stale += self.stale_copies(*p, changed);
            }
            round += 1;
        }
        stats.rounds.push(round);

        // Every patch holds exactly what its owned faces need.
        self.check_mesh();
        stats.inexact += (0..self.patch_count())
            .filter(|&q| self.required(self.owned_faces[q].iter()) != self.held[q])
            .count();

        // Tally the outcomes.
        stats.patches_out_of_room += out_of_room.len();
        let mut applied = 0;
        for s in status.values() {
            match s {
                Status::Applied => applied += 1,
                Status::LostClaim => stats.lost += 1,
                Status::Rejected => stats.rejected += 1,
                Status::OutOfCapacity => stats.out_of_capacity += 1,
                Status::Pending | Status::NotAnEdge => {}
            }
        }
        stats.applied += applied;
        applied
    }

    /// Repeats batches of `op` over the edges `wanted` selects until none is
    /// wanted, a batch edits nothing, or `max_batches` have run.
    fn settle(
        &mut self,
        op: Op,
        mode: Corners,
        max_batches: usize,
        wanted: impl Fn(&Sim, u32) -> bool,
    ) -> Stats {
        let mut stats = Stats {
            faces_before: self.live_faces(),
            ..Stats::default()
        };
        for seed in 0..max_batches as u32 {
            let requested: Vec<u32> = (0..self.ends.len() as u32)
                .filter(|&e| self.edge_alive(e) && wanted(self, e))
                .collect();
            if requested.is_empty()
                || self.growth_batch(op, &requested, seed, mode, &mut stats) == 0
            {
                break;
            }
        }
        stats.faces_after = self.live_faces();
        stats
    }
}

/// Words a patch of capacities `caps` takes in the arena (record and
/// region), the masks and the maps.
fn patch_words(caps: [u32; 3]) -> [u64; 3] {
    [
        RECORD_WORDS + patch_layout::patch_region_words(caps),
        patch_layout::patch_mask_words(caps, true),
        patch_layout::maps_words(&[caps]),
    ]
}

fn print_layout_header() {
    println!(
        "| mesh | patch size | patches | copies V/E/F max | copies V/E/F mean | max/mean F \
         | today MiB | uniform, {POOL_FACTOR}x pool MiB | ratio | uniform arena/masks/maps MiB \
         | floor median MiB (ratio) | floor p90 MiB (ratio) | patches over 2x median |"
    );
    println!("|---|---:|---:|---|---|---:|---:|---:|---:|---|---|---|---:|");
}

/// The `q` quantile of each kind's count over `counts`.
fn quantile(counts: &[[usize; 3]], q: f64) -> [usize; 3] {
    [0, 1, 2].map(|k| {
        let mut v: Vec<usize> = counts.iter().map(|c| c[k]).collect();
        v.sort_unstable();
        v[((v.len() - 1) as f64 * q).round() as usize]
    })
}

/// Prints how uneven the patches are and the bytes per-patch and uniform
/// capacity take, at `factor` spare slots.
fn print_layout_row(name: &str, patch_size: u32, sim: &Sim, factor: f64) {
    let n = sim.patch_count();
    let counts: Vec<[usize; 3]> = sim
        .held
        .iter()
        .map(|h| [h[0].len(), h[1].len(), h[2].len()])
        .collect();
    let max = [0, 1, 2].map(|k| counts.iter().map(|c| c[k]).max().unwrap());
    let mean = [0, 1, 2].map(|k| counts.iter().map(|c| c[k]).sum::<usize>() as f64 / n as f64);
    let cap = |c: usize| (c as f64 * factor).ceil() as u32;
    let today: u64 = counts
        .iter()
        .map(|c| patch_words(c.map(cap)).iter().sum::<u64>())
        .sum();
    let parts = patch_words(max.map(cap)).map(|w| POOL_FACTOR * n as u64 * w);
    let uniform: u64 = parts.iter().sum();
    let mib = |w: u64| w as f64 * 4.0 / (1024.0 * 1024.0);

    // Floor policy: each live patch gets `factor` times the larger of its
    // own count and a typical count; spares get `factor` times the typical.
    let floor = |typical: [usize; 3]| -> u64 {
        let live: u64 = counts
            .iter()
            .map(|c| {
                let caps = [0, 1, 2].map(|k| cap(c[k].max(typical[k])));
                patch_words(caps).iter().sum::<u64>()
            })
            .sum();
        live + (POOL_FACTOR - 1) * n as u64 * patch_words(typical.map(cap)).iter().sum::<u64>()
    };
    let median = quantile(&counts, 0.5);
    let p90 = quantile(&counts, 0.9);
    let (floor_median, floor_p90) = (floor(median), floor(p90));
    let over = counts.iter().filter(|c| c[2] > 2 * median[2]).count();
    println!(
        "| {name} | {patch_size} | {n} | {}/{}/{} | {:.0}/{:.0}/{:.0} | {:.2} | {:.1} | {:.1} | {:.2} | {:.1}/{:.1}/{:.1} | {:.1} ({:.2}) | {:.1} ({:.2}) | {over} |",
        max[0],
        max[1],
        max[2],
        mean[0],
        mean[1],
        mean[2],
        max[2] as f64 / mean[2],
        mib(today),
        mib(uniform),
        uniform as f64 / today as f64,
        mib(parts[0]),
        mib(parts[1]),
        mib(parts[2]),
        mib(floor_median),
        floor_median as f64 / today as f64,
        mib(floor_p90),
        floor_p90 as f64 / today as f64,
    );
}

fn print_header() {
    println!(
        "| mesh | patch size | op | corners | batches | faces before → after | requested | interior \
         | applied | lost | rejected | out of room | patches out of room | left after {ENCODED_ROUNDS} rounds \
         | rounds (med/p99/max) | corners (med/p99/max) | footprint patches | winners per round \
         | peak use V/E/F | stale | ribbon gaps | inexact |"
    );
    println!(
        "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|"
    );
}

fn print_row(name: &str, patch_size: u32, op: Op, mode: Corners, sim: &Sim, mut s: Stats) {
    let corners = match (op, mode) {
        (Op::Split, _) => "quad",
        (Op::Collapse, Corners::Removed) => "b + link(b)",
        (Op::Collapse, Corners::Both) => "a, b + links",
        (Op::Collapse, Corners::Quad) => "quad (control)",
    };
    println!(
        "| {name} | {patch_size} | {op:?} | {corners} | {} | {} → {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2}/{:.2}/{:.2} | {} | {} | {} |",
        s.batches,
        s.faces_before,
        s.faces_after,
        s.requested,
        s.interior,
        s.applied,
        s.lost,
        s.rejected,
        s.out_of_capacity,
        s.patches_out_of_room,
        s.left_after_encoded,
        spread(&mut s.rounds),
        spread(&mut s.corners),
        spread(&mut s.footprints),
        spread(&mut s.winners),
        sim.peak_ratio[0],
        sim.peak_ratio[1],
        sim.peak_ratio[2],
        s.stale,
        s.ribbon_gaps,
        s.inexact,
    );
}

#[test]
#[ignore = "simulation; run in release with --ignored --nocapture"]
fn growth_simulation() {
    if std::env::var_os("MOPMESH_BENCH").is_none() {
        println!("set MOPMESH_BENCH=1 to run the growth simulation");
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
    let faces = numbers("MOPMESH_BENCH_FACES", "32768");
    let patches = numbers("MOPMESH_BENCH_PATCHES", "64,128,512");
    let max_batches = numbers("MOPMESH_BENCH_BATCHES", "12")[0] as usize;
    let factor: f64 = std::env::var("MOPMESH_BENCH_CAPACITY")
        .map(|c| c.parse().expect("a capacity factor"))
        .unwrap_or(super::CAPACITY);
    println!("# MOPMesh growth simulation (CPU), capacity {factor}, uniform per kind");
    if cfg!(debug_assertions) {
        println!("WARNING: debug build; slow");
    }
    let mut meshes: Vec<Named> = faces
        .into_iter()
        .flat_map(bench_meshes::generated)
        .map(|m| (m.name, m.positions, m.triangles))
        .collect();
    let (positions, triangles) = skewed_grid(129);
    meshes.push(("skewed 129x129".into(), positions, triangles));
    if let Ok(only) = std::env::var("MOPMESH_BENCH_MESH") {
        meshes.retain(|(name, ..)| name.contains(&only));
    }

    // How uneven patches are, and the bytes uniform capacity would take.
    println!("\n## Patch sizes and arena bytes (spare factor {factor})\n");
    print_layout_header();
    let layout_faces = std::env::var("MOPMESH_BENCH_LAYOUT_FACES")
        .map_or(Vec::new(), |_| numbers("MOPMESH_BENCH_LAYOUT_FACES", ""));
    let layout_meshes: Vec<_> = layout_faces
        .into_iter()
        .flat_map(bench_meshes::generated)
        .map(|m| (m.name, m.positions, m.triangles))
        .collect();
    for (name, positions, triangles) in meshes.iter().chain(&layout_meshes) {
        for &patch_size in &patches {
            let sim = Sim::with_capacity(positions, triangles, patch_size, factor);
            print_layout_row(name, patch_size, &sim, factor);
        }
    }
    if max_batches == 0 {
        return;
    }

    // Split every edge longer than 4/3 of half the mean length, and collapse
    // every edge shorter than 4/5 of twice the mean length, as remeshing to
    // half and to twice the edge length would.
    println!("\n## Split and collapse runs (rows sum over all batches)\n");
    print_header();
    for (name, positions, triangles) in &meshes {
        for &patch_size in &patches {
            let runs = [
                (Op::Split, Corners::Removed),
                (Op::Collapse, Corners::Removed),
                (Op::Collapse, Corners::Both),
                (Op::Collapse, Corners::Quad),
            ];
            for (op, mode) in runs {
                let mut sim = Sim::with_capacity(positions, triangles, patch_size, factor);
                sim.uniform_capacity(factor);
                let mean = sim.mean_edge_length();
                let stats = match op {
                    Op::Split => sim.settle(op, mode, max_batches, |s, e| {
                        s.edge_length(e) > 4.0 / 3.0 * 0.5 * mean
                    }),
                    Op::Collapse => sim.settle(op, mode, max_batches, |s, e| {
                        s.edge_length(e) < 4.0 / 5.0 * 2.0 * mean
                    }),
                };
                print_row(name, patch_size, op, mode, &sim, stats);
            }
        }
    }
}
