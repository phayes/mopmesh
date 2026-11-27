# Benchmark baseline (ROADMAP milestones 2 and 3)

Setup, memory and warm query costs of the patch arena, measured against an
unpartitioned GPU baseline on the same adapter. No speedup is assumed; the
point is to find where the patch layout costs or pays. The baseline was
first run for milestone 2 (26 November 2026), rerun after milestone 3's
workgroup staging (commit `746a47f`), and rerun on 27 November 2026 at
`ab98f19` and again at `e6f8663`, after the default patch size became 512
and the in-kernel local queries landed. This report gives the `e6f8663`
run, with earlier figures where they show what changed.

## Bottom line

- **On shuffled meshes the arena beats the flat layout for adjacency.**
  At 2.1 M faces and patch size 128, batched adjacency (VV, FF, diamonds and
  edge records) takes 9.7–10.1 ms on the arena against 13.1–13.3 ms flat
  (0.74–0.76x). Edge records are level (0.96–1.08x). In milestone 2 the
  arena was slower on both (1.15–1.2x and 1.8–2.0x).
- **In input order the arena is still slower**: edge records 2.5–3.2x (from
  5.8–6.9x in milestone 2) and batched adjacency 1.6–1.9x (from 2.9–3.2x).
  Edge records take 2.7–3.3 ms against 1.0–1.1 ms flat.
- **Patch size 512, now the default, is the fastest setting measured.** At
  2.1 M faces, edge records take 1.6–1.9 ms in input order (1.6–1.9x flat)
  and 2.0–2.2 ms shuffled (0.55–0.6x flat); batched adjacency takes 3.5–3.9
  and 8.9–9.2 ms (1.2–1.4x and 0.68–0.69x flat). Patch size 32 is the slow
  case: edge records 8.8–11.0 ms. Its cause is not profiled.
- **Memory is 65–75 bytes per face at patch size 512 and 83–103 at 128**,
  against 36 flat. Milestone 2 took 129–145 at 128.
- **Boundary flags no longer cost a second pass**: 2.4–3.0 ms in input order
  and 3.6–4.4 ms shuffled at patch size 128, from 6.5–7.4 ms.
- **A regression came and went.** At `ab98f19` batched adjacency was
  1.1–1.2 ms slower in input order, because its staged pass read all three
  of a face's edge endpoints through the new `local_query` helpers.
  `e6f8663` restored the single read, and the times are back within noise
  of `746a47f`'s.
- **Per-face queries on the arena are 9–27% slower than in milestone 2** in
  input order (face vertices, centroids, normals, areas), from the per-patch
  dispatch and the active and owned checks. Shuffled face normals and areas
  varied by up to 50% between runs. The flat path's GPU times are within 5%
  of milestone 2's, so this is not machine drift.
- **VV alone is 0.86–0.93 ms in input order and 2.5–2.7 ms shuffled**, about
  a quarter of shuffled adjacency. Staging VV was skipped for that reason.
- **The GPU build still degrades on high-valence vertices.** The 2.1 M-face
  UV sphere builds in 2.3 s at patch size 128, 6.7 s at 32 and 0.47 s at
  512, against 126 ms for a grid of the same size at 128. Not yet profiled.

## Running it

```sh
MOPMESH_BENCH=1 cargo test --release --lib gpu_baseline_benchmark -- --ignored --nocapture
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `MOPMESH_BENCH_FACES` | `32768,524288,2097152` | Target face counts for the generated meshes |
| `MOPMESH_BENCH_PATCHES` | `512` | Patch sizes |
| `MOPMESH_BENCH_OBJ` | none | OBJ files to add, each with a shuffled copy |
| `MOPMESH_BENCH_RUNS` | `11` | Timed runs per query, after 2 warm-up runs |

This report used `MOPMESH_BENCH_PATCHES=32,128,512` and the default sizes.

The code is crate-private and only compiled for tests:
`src/gpu/bench_baseline.rs` (the benchmark), `src/gpu/bench_meshes.rs` (the
meshes) and `src/gpu/baseline.rs` (the flat twins, with a test that their
results match the patched queries).

## Method

**Meshes.** Four generated surfaces at each size, each also with a shuffled copy:

- a flat grid (boundary, valence 6);
- a closed UV sphere (two poles of very high valence);
- a closed torus (every vertex valence 6);
- an irregular grid (random quad diagonals and jittered positions, valence 2–8).

The shuffled copy permutes vertex IDs and face order randomly and rotates each
triangle's corners, so neighboring elements no longer have nearby IDs. The
surface and winding are unchanged.

**Flat baseline.** The same connectivity the arena stores (edge endpoints, and
three directed edges per face) as global arrays, without patches or ribbons:
36 bytes per face. Face vertices are decoded the same way from global arrays.
Edge records take three passes: copy the endpoints, then each face writes its
ID into its edges' slots (the direction picks the slot), then move a boundary
edge's face to the first slot. FF and diamonds take one pass per edge over
the edge records and face vertices. VV, centroids and smoothing run the same
kernels on both paths, because they read only global results.

**Arena path.** Edge records, batched adjacency's edge, FF and diamond
pass, and boundary flags run one workgroup per patch, staging each edge's
faces in workgroup memory. Face vertices and the per-face queries also run
one workgroup per patch, over its active, owned faces.

**Timing.**

- Patched and flat runs alternate within each round, so machine drift affects
  both: 11 timed rounds after 2 warm-up rounds.
- *Wall time* covers encoding, including the buffers each `encode` call
  allocates, submitting, and waiting for the GPU.
- *GPU time* is the sum of each compute pass's own duration, from timestamp
  queries at pass boundaries. It leaves out gaps between passes. This
  adapter has no timestamps inside encoders.
- Setup rows are 3 timed runs after 1 warm-up.
- The whole benchmark ran three times. Slow stretches came in bursts that
  raised even the minimum of a row, so each figure here is the lowest of
  the three runs' medians. Of the 1,048 rows with a median of at least
  0.1 ms, 86% have at least two runs within 5% of that figure, and the three
  medians span under 10% in 78%. Treat differences under about 10% as noise.

**Counts.** Dispatches come from a test-only counter in the shared dispatch
helper. Buffers and bind groups come from wgpu's internal counters (the
`counters` feature, enabled for tests only).

**Machine.** Apple M1 Max, Metal, release build, 2026-11-27, commit
`e6f8663`. Storage binding limit 4 GiB, 65535 workgroups per dimension,
timestamp period 1 ns.

## Memory and setup

At 2.1 M faces and patch size 128 (see [Patch size](#patch-size) for 32 and
512). Both upload columns include laying out the data on the host from the
CPU mesh.

| Mesh | Arena B/face | Milestone 2 | Flat B/face | Faces held per face | CPU build | Upload arena | Upload flat | GPU build |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 83.3 | 128.7 | 36.0 | 1.636 | 476 ms | 89.3 ms | 33.4 ms | 126 ms |
| grid, shuffled | 91.8 | 135.5 | 36.0 | 1.786 | 2176 ms | 126.7 ms | 57.6 ms | 406 ms |
| sphere | 98.1 | 141.0 | 36.0 | 1.902 | 507 ms | 102.5 ms | 33.9 ms | 2277 ms |
| sphere, shuffled | 102.9 | 144.9 | 36.0 | 1.982 | 2240 ms | 138.5 ms | 59.3 ms | 753 ms |
| torus | 89.9 | 134.1 | 36.0 | 1.758 | 517 ms | 98.1 ms | 35.9 ms | 166 ms |
| torus, shuffled | 92.0 | 135.7 | 36.0 | 1.791 | 2200 ms | 127.5 ms | 57.9 ms | 452 ms |
| irregular | 87.6 | 132.3 | 36.0 | 1.713 | 634 ms | 100.9 ms | 40.4 ms | 157 ms |
| irregular, shuffled | 92.5 | 136.2 | 36.0 | 1.798 | 2500 ms | 126.8 ms | 58.9 ms | 421 ms |

- The arena holds the shared patch layout: an 8-word header, a 16-word
  record per patch, the patch regions, and a mask buffer of used counts,
  active masks and owned masks. The per-face task lists are gone.
- Both builds slow down on shuffled input: the CPU build by 3.9–4.6x and
  the GPU build by 2.7–3.2x (sphere aside). Patches stay almost as good
  (1.79 against 1.64 faces held per face).
- Host-side setup ran 10–24% slower than in milestone 2 for the CPU build
  and 2–17% for the flat upload, neither of which changed, so the machine
  was slower on the CPU. The arena upload took 8–52% longer, more than that
  accounts for; it now also lays out the mask buffer.
- The sphere's GPU build is 18x slower than the grid's at this size, and
  9x at 523 K faces (262 against 28 ms). Its shuffled copy builds faster
  (753 ms). This is not yet profiled.

## Queries with a flat twin

Median GPU time in ms, patched / flat, at 2.1 M faces and patch size 128.
The ratio is patched divided by flat; above 1 means the arena is slower.

| Query | Grid | Grid, shuffled | Sphere | Sphere, shuffled |
|---|---|---|---|---|
| Face vertices | 0.65 / 0.34 (1.9x) | 1.23 / 0.86 (1.4x) | 0.73 / 0.34 (2.1x) | 1.25 / 0.86 (1.5x) |
| Edge records | 2.67 / 1.05 (2.5x) | 3.67 / 3.66 (1.0x) | 3.26 / 1.02 (3.2x) | 3.93 / 3.65 (1.1x) |
| Face centroids | 0.88 / 0.59 (1.5x) | 1.95 / 1.58 (1.2x) | 0.95 / 0.59 (1.6x) | 1.96 / 1.58 (1.2x) |
| VV + FF + diamonds + edges | 4.55 / 2.83 (1.6x) | 9.83 / 13.12 (0.75x) | 5.37 / 2.84 (1.9x) | 10.10 / 13.24 (0.76x) |
| Adjacency + 10 smoothing steps | 9.22 / 7.50 (1.2x) | 18.93 / 22.22 (0.85x) | 13.62 / 11.09 (1.2x) | 25.14 / 28.29 (0.89x) |

| Query | Torus | Torus, shuffled | Irregular | Irregular, shuffled |
|---|---|---|---|---|
| Face vertices | 0.76 / 0.34 (2.2x) | 1.23 / 0.86 (1.4x) | 0.70 / 0.34 (2.1x) | 1.21 / 0.86 (1.4x) |
| Edge records | 3.21 / 1.03 (3.1x) | 3.67 / 3.66 (1.0x) | 2.90 / 1.03 (2.8x) | 3.54 / 3.67 (0.96x) |
| Face centroids | 0.98 / 0.59 (1.7x) | 1.95 / 1.58 (1.2x) | 0.92 / 0.59 (1.6x) | 1.92 / 1.58 (1.2x) |
| VV + FF + diamonds + edges | 5.27 / 2.81 (1.9x) | 9.94 / 13.26 (0.75x) | 5.05 / 2.92 (1.7x) | 9.71 / 13.16 (0.74x) |
| Adjacency + 10 smoothing steps | 9.97 / 7.53 (1.3x) | 19.03 / 22.33 (0.85x) | 10.37 / 8.23 (1.3x) | 18.70 / 22.14 (0.84x) |

Median wall time for the same runs, grid in input order: face vertices 1.80 /
1.48 ms, edge records 4.70 / 3.10 ms, centroids 3.28 / 2.97 ms, batched
adjacency 11.5 / 10.6 ms, and adjacency plus smoothing 17.9 / 17.0 ms.

Earlier runs, same queries, patched: edge records 6.29–7.22 ms in input
order and 6.42–7.49 ms shuffled in milestone 2, and the same as now at
`746a47f`. Batched adjacency took 8.22–9.29 and 15.09–16.28 ms in milestone
2, 4.47–5.24 and 9.67–10.07 ms at `746a47f`, and 5.53–6.35 and
10.13–10.58 ms at `ab98f19`. The flat path's GPU times are within 5% of
milestone 2's.

At 523 K faces the arena is slower on every twin query (edge records
3.5–5.4x, batched adjacency 1.4–2.2x), because the flat path's penalty for
shuffled input is smaller there: flat edge records take 0.27–0.29 ms
shuffled against 0.15–0.16 ms in input order. At 32 K faces every query
takes under 1 ms of GPU time and at most about 2 ms wall, and the fixed
per-call cost dominates.

What the columns show:

- **Edge records** cost O(1) per face: each face writes its local slot into
  its edges' entries in workgroup memory, then each owned edge reads its two
  entries. The arena's remaining input-order gap is the staged pass itself
  (patch records, masks, local-to-global lookups). Shuffling barely moves
  it, because its reads stay inside one patch.
- **Batched adjacency** is one staged pass for edge records, FF and
  diamonds, then the global VV count, scan and scatter. VV alone is
  0.86–0.93 ms in input order and 2.5–2.7 ms shuffled; the staged pass is
  most of the rest, and its global writes of FF and diamonds land scattered
  on shuffled meshes.
- **Face vertices and centroids** cost 1.2–2.2x on the arena from
  indirection (record, local IDs, global IDs) and the active and owned
  checks; they are 9–27% slower than in milestone 2.
- **Smoothing** reads the global VV rows, so it slows on shuffled meshes on
  both paths. On the sphere one invocation walks each pole's 1448 neighbors.

## Queries without a twin

Median GPU time (wall) in ms, patched, at 2.1 M faces and patch size 128.

| Query | Grid | Grid, shuffled | Sphere | Sphere, shuffled | Torus | Torus, shuffled | Irregular | Irregular, shuffled |
|---|---|---|---|---|---|---|---|---|
| Vertex faces | 1.32 (3.97) | 4.46 (7.22) | 1.42 (4.10) | 4.34 (7.03) | 1.41 (4.06) | 4.88 (7.78) | 1.45 (4.13) | 4.71 (7.55) |
| Face normals | 0.88 (3.26) | 2.42 (4.88) | 0.95 (3.33) | 1.96 (4.25) | 0.97 (3.35) | 2.67 (5.61) | 0.92 (3.30) | 2.53 (5.11) |
| Face areas | 0.81 (2.34) | 2.49 (4.02) | 0.89 (2.39) | 1.91 (3.35) | 0.91 (2.44) | 3.05 (4.71) | 0.85 (2.36) | 2.46 (4.10) |
| Vertex normals | 2.61 (7.00) | 12.89 (18.31) | 3.36 (7.79) | 11.78 (16.28) | 2.79 (7.07) | 13.77 (18.51) | 2.94 (7.18) | 13.78 (19.14) |
| Boundary flags | 2.40 (3.43) | 4.41 (5.56) | 2.95 (3.95) | 3.57 (4.58) | 2.91 (3.88) | 4.42 (5.48) | 2.60 (3.61) | 4.23 (5.29) |
| VV, edge records given | 0.86 (1.64) | 2.54 (3.38) | 0.90 (1.66) | 2.54 (3.31) | 0.87 (1.64) | 2.70 (3.44) | 0.93 (1.70) | 2.58 (3.33) |
| 10 smoothing steps, adjacency given | 5.70 (7.69) | 11.18 (13.64) | 8.34 (10.32) | 15.17 (17.23) | 4.86 (6.89) | 12.35 (14.55) | 5.43 (7.50) | 12.42 (14.67) |
| Sum of face areas | 0.18 (0.76) | 0.23 (0.91) | 0.14 (0.70) | 0.14 (0.73) | 0.14 (0.73) | 0.23 (0.93) | 0.18 (0.77) | 0.23 (0.89) |
| Dot of face areas | 0.18 (0.75) | 0.18 (0.88) | 0.14 (0.69) | 0.14 (0.71) | 0.14 (0.72) | 0.19 (0.83) | 0.14 (0.72) | 0.23 (0.90) |

- **Boundary flags** run one staged pass and write the flags directly, with
  no edge records: 6.5–7.4 ms in milestone 2.
- **Vertex normals** slow 3.5–4.9x on shuffled input. They gather over VF
  rows of scattered faces. On the shuffled grid they take 12.89 ms against
  10.42 ms in milestone 2 (24%); not investigated.
- **Smoothing on shuffled meshes depends on the adjacency it reads.** This
  row reuses one adjacency built at setup and took 11–14 ms on the shuffled
  grid, torus and irregular grid. "Adjacency + 10 smoothing steps", which
  rebuilds the adjacency every round, implies about 9 ms for the same
  smoothing in most runs, with occasional 12–14 ms rounds. The sphere is
  steady at about 15 ms. VV row order is unspecified and differs between
  builds, which is the likely factor; not verified. Giving the benchmark's
  VV row its own adjacency did not change this.
- Reductions read one attribute linearly and are unaffected by mesh shape.

## Patch size

Median GPU time in ms, patched / flat, for patch sizes 32, 128 and 512 in the
same runs. Edge records and batched adjacency have flat twins; boundary
flags and VV do not. The default patch size is 512.

At 2.1 M faces:

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | VV | GPU build |
|---|---:|---:|---:|---|---|---:|---:|---:|
| grid | 32 | 123.5 | 2.338 | 8.82 / 1.05 | 11.18 / 2.82 | 8.39 | 0.84 | 157 ms |
| grid | 128 | 83.3 | 1.636 | 2.67 / 1.05 | 4.55 / 2.83 | 2.40 | 0.86 | 126 ms |
| grid | 512 | 65.2 | 1.307 | 1.64 / 1.04 | 3.48 / 2.81 | 1.38 | 0.84 | 145 ms |
| grid, shuffled | 32 | 146.2 | 2.725 | 10.59 / 3.65 | 15.71 / 13.12 | 12.86 | 2.41 | 454 ms |
| grid, shuffled | 128 | 91.8 | 1.786 | 3.67 / 3.66 | 9.83 / 13.12 | 4.41 | 2.54 | 406 ms |
| grid, shuffled | 512 | 69.2 | 1.382 | 2.01 / 3.66 | 8.92 / 13.12 | 2.18 | 2.58 | 403 ms |
| sphere | 32 | 158.7 | 2.953 | 10.33 / 1.02 | 12.91 / 2.85 | 9.84 | 0.89 | 6666 ms |
| sphere | 128 | 98.1 | 1.902 | 3.26 / 1.02 | 5.37 / 2.84 | 2.95 | 0.90 | 2277 ms |
| sphere | 512 | 72.2 | 1.435 | 1.93 / 1.02 | 3.90 / 2.84 | 1.66 | 0.90 | 470 ms |
| sphere, shuffled | 32 | 168.6 | 3.118 | 11.02 / 3.66 | 16.14 / 13.23 | 10.51 | 2.55 | 1395 ms |
| sphere, shuffled | 128 | 102.9 | 1.982 | 3.93 / 3.65 | 10.10 / 13.24 | 3.57 | 2.54 | 753 ms |
| sphere, shuffled | 512 | 75.2 | 1.487 | 2.20 / 3.65 | 9.18 / 13.23 | 1.86 | 2.54 | 524 ms |
| torus | 32 | 136.7 | 2.575 | 9.61 / 1.02 | 12.08 / 2.83 | 9.17 | 0.87 | 197 ms |
| torus | 128 | 89.9 | 1.758 | 3.21 / 1.03 | 5.27 / 2.81 | 2.91 | 0.87 | 166 ms |
| torus | 512 | 67.5 | 1.354 | 1.84 / 1.02 | 3.78 / 2.81 | 1.56 | 0.86 | 182 ms |
| torus, shuffled | 32 | 146.7 | 2.733 | 10.64 / 3.67 | 15.84 / 13.24 | 12.10 | 2.52 | 536 ms |
| torus, shuffled | 128 | 92.0 | 1.791 | 3.67 / 3.66 | 9.94 / 13.26 | 4.42 | 2.70 | 452 ms |
| torus, shuffled | 512 | 69.5 | 1.387 | 2.02 / 3.65 | 9.01 / 13.22 | 2.23 | 2.71 | 438 ms |
| irregular | 32 | 132.6 | 2.500 | 9.14 / 1.05 | 11.70 / 2.92 | 8.73 | 0.93 | 193 ms |
| irregular | 128 | 87.6 | 1.713 | 2.90 / 1.03 | 5.05 / 2.92 | 2.60 | 0.93 | 157 ms |
| irregular | 512 | 67.4 | 1.347 | 1.84 / 1.04 | 3.88 / 2.92 | 1.56 | 0.94 | 161 ms |
| irregular, shuffled | 32 | 145.6 | 2.715 | 10.31 / 3.67 | 15.45 / 13.14 | 11.74 | 2.42 | 499 ms |
| irregular, shuffled | 128 | 92.5 | 1.798 | 3.54 / 3.67 | 9.71 / 13.16 | 4.23 | 2.58 | 421 ms |
| irregular, shuffled | 512 | 69.4 | 1.383 | 2.03 / 3.65 | 8.91 / 13.14 | 2.42 | 2.59 | 412 ms |

At 523 K faces, on the grid (the other meshes follow the same pattern):

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | VV | GPU build |
|---|---:|---:|---:|---|---|---:|---:|---:|
| grid | 32 | 123.5 | 2.337 | 1.78 / 0.16 | 2.46 / 0.64 | 1.66 | 0.25 | 33 ms |
| grid | 128 | 83.3 | 1.635 | 0.69 / 0.16 | 1.24 / 0.64 | 0.60 | 0.25 | 28 ms |
| grid | 512 | 65.4 | 1.311 | 0.45 / 0.15 | 1.00 / 0.65 | 0.36 | 0.25 | 32 ms |
| grid, shuffled | 32 | 146.1 | 2.722 | 3.10 / 0.34 | 3.71 / 1.33 | 2.49 | 0.55 | 53 ms |
| grid, shuffled | 128 | 91.5 | 1.780 | 1.17 / 0.29 | 1.88 / 1.34 | 0.82 | 0.55 | 46 ms |
| grid, shuffled | 512 | 69.2 | 1.381 | 0.51 / 0.27 | 1.48 / 1.32 | 0.41 | 0.55 | 43 ms |

- **Staged edge records get faster as patches grow.** In milestone 2 they
  grew linearly with patch size (0.61 ms at patch size 32 and 5.00 ms at
  512, grid at 523 K faces); now they fall (1.78 and 0.45 ms). Larger
  patches have fewer workgroups and fewer ribbon copies per face, which
  plausibly explains it; not profiled.
- **Patch size 512 is the best setting measured**, and the default since
  27 November 2026, as in RXMesh. At 2.1 M faces it gives the lowest time
  on every staged query and the least memory (65–75 B/face).
- **Patch size 32 is about 3x slower than milestone 2** on staged edge
  records: at 523 K faces, 1.78 ms on the grid and 2.38 ms on the sphere,
  against 0.61 and 0.72 ms then. At 2.1 M faces it takes 8.8–11.0 ms. It is
  the most memory-hungry setting too (124–169 B/face). The cause is not
  profiled.
- **Boundary flags track edge records** at every size, since they run the
  same staging.
- **VV does not depend on patch size**, because it is a global pass over
  the edge records.
- **The sphere's GPU build falls as patches grow** (6.7 s at 32, 0.47 s at
  512, 2.1 M faces). That points to small patches around the high-valence
  poles, but the cause is not yet profiled.

## Dispatches and allocations

Per call, at 2.1 M faces.

| Query | Dispatches (patched / flat) | Buffers (patched / flat) | Bind groups (patched / flat) |
|---|---|---|---|
| Face vertices | 1 / 1 | 1 / 1 | 2 / 1 |
| Edge records | 1 / 3 | 1 / 1 | 2 / 1 |
| Face centroids | 2 / 2 | 2 / 2 | 3 / 2 |
| VV + FF + diamonds + edges | 11 / 15 | 15 / 16 | 7 / 8 |
| Adjacency + 10 smoothing steps | 21 / 25 | 19 / 20 | 17 / 18 |
| Boundary flags | 1 | 2 | 2 |
| VV, edge records given | 10 | 11 | 5 |
| Vertex normals | 13 | 17 | 10 |
| Sum or dot of face areas | 4 | 10 | 4 |

Patch queries bind the arena and masks as a second bind group. The scan's
pass count grows with the vertex count, so the adjacency dispatch counts are
9 / 13 at 32 K faces.

## Limits of this baseline

- One adapter (Apple M1 Max, a unified-memory GPU). Discrete GPUs may weigh
  memory traffic and per-call overhead differently.
- Generated meshes only. Scanned meshes have irregular valence and density;
  pass them with `MOPMESH_BENCH_OBJ`.
- Summed pass time leaves out gaps between passes, so it understates GPU
  occupancy for queries with many short passes.
- Scratch bytes are not reported. wgpu's buffer-memory counter is not
  maintained on Metal, and only buffer counts are. On Vulkan or DX12 the
  same counters would give bytes.
- The flat baseline reuses this crate's kernels for VV, centroids and
  smoothing. It is not an independent GPU mesh library, and a different flat
  layout (for example one that also stores face vertices) would be faster
  still.
- Only static meshes are measured. Editable meshes use the same layout and
  kernels, plus spare slots; flips are timed separately by the flip
  benchmark.
