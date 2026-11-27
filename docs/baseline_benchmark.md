# Benchmark baseline (ROADMAP milestones 2 and 3)

Setup, memory and warm query costs of the patch arena, measured against an
unpartitioned GPU baseline on the same adapter. No speedup is assumed; the
point is to find where the patch layout costs or pays. The baseline was
first run for milestone 2 (26 November 2026), rerun after milestone 3's
workgroup staging (commit `746a47f`), and rerun on 27 November 2026 at
`ab98f19`, `e6f8663` and `2dcee78`. This report gives the `2dcee78` run, the
first after staged kernels moved to 128-lane workgroups, 512 staged edges
and no zero-filling of workgroup memory (see the
[staged pass profile](workgroup_staging_plan.md#staged-pass-profile)).
Earlier figures appear where they show what changed.

## Bottom line

- **The staged queries are much faster than at `e6f8663`.** The profile
  found that workgroup memory, not staging work, cost most of their time.
  Across meshes and patch sizes at 2.1 M faces, edge records are 1.4–3.2x
  faster, boundary flags 1.2–2.6x, and batched adjacency (VV, FF, diamonds
  and edge records) 1.4–2.5x in input order. Shuffled adjacency gained
  little: 5% faster at patch size 128 and 1–2% slower at 512.
- **On shuffled meshes the arena beats the flat layout clearly.** At patch
  size 128, edge records run at 0.43–0.50x of flat and batched adjacency
  at 0.71–0.73x. At patch size 512 they run at 0.37–0.43x and 0.69–0.71x.
- **In input order the arena is now level with flat for adjacency.**
  Batched adjacency is 0.91–1.08x of flat at patch size 128 and 0.85–0.96x
  at 512. Edge records are 1.05–1.42x at 128 and 0.92–1.16x at 512, where
  the grid, torus and irregular grid are within 8% of flat and the sphere
  is 16% slower. At `e6f8663` they were 1.6–1.9x and 2.5–3.2x at 128.
- **Patch size 32 recovered most of its loss**: edge records take 2.7–4.4 ms
  at 2.1 M faces, from 8.8–11.0 ms, and 0.69 ms on the grid at 523 K faces,
  against 0.61 ms in milestone 2. It is still the slowest and largest
  setting.
- **Memory is 65–75 bytes per face at patch size 512 and 83–103 at 128**,
  against 36 flat. Milestone 2 took 129–145 at 128. Unchanged by this run.
- **Per-face queries are unchanged** (face vertices, centroids, normals,
  areas), still 9–27% slower than in milestone 2 in input order. VV alone
  is 0.86–0.96 ms in input order and 2.5–2.9 ms shuffled, as before.
- **Smoothing time depends on the adjacency build.** It measured higher in
  this run than at `e6f8663`, but a side-by-side run of both builds found
  overlapping ranges; see [Queries without a twin](#queries-without-a-twin).
- **The GPU build still degrades on high-valence vertices.** The 2.1 M-face
  UV sphere builds in 1.6 s at patch size 128 (2.3 s at `e6f8663`), 6.6 s
  at 32 and 0.57 s at 512, against 123 ms for a grid of the same size at
  128. Not yet profiled.

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
pass, and boundary flags run one workgroup of 128 invocations per patch,
staging each edge's faces in 4 KB of workgroup memory, 512 edges at a time.
Face vertices and the per-face queries also run one workgroup per patch,
over its active, owned faces. No pipeline on either path zero-fills
workgroup memory.

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
  the three runs' medians. Of the 1,030 rows with a median of at least
  0.1 ms, 84% have at least two runs within 5% of that figure, and the three
  medians span under 10% in 81%. Treat differences under about 10% as noise.

**Counts.** Dispatches come from a test-only counter in the shared dispatch
helper. Buffers and bind groups come from wgpu's internal counters (the
`counters` feature, enabled for tests only).

**Machine.** Apple M1 Max, Metal, release build, 2026-11-27, commit
`2dcee78`. Storage binding limit 4 GiB, 65535 workgroups per dimension,
timestamp period 1 ns.

## Memory and setup

At 2.1 M faces and patch size 128 (see [Patch size](#patch-size) for 32 and
512). Both upload columns include laying out the data on the host from the
CPU mesh.

| Mesh | Arena B/face | Milestone 2 | Flat B/face | Faces held per face | CPU build | Upload arena | Upload flat | GPU build |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 83.3 | 128.7 | 36.0 | 1.636 | 478 ms | 89.6 ms | 33.2 ms | 123 ms |
| grid, shuffled | 91.8 | 135.5 | 36.0 | 1.786 | 2190 ms | 127.2 ms | 57.8 ms | 406 ms |
| sphere | 98.1 | 141.0 | 36.0 | 1.902 | 509 ms | 102.9 ms | 34.4 ms | 1599 ms |
| sphere, shuffled | 102.9 | 144.9 | 36.0 | 1.982 | 2266 ms | 138.2 ms | 59.4 ms | 758 ms |
| torus | 89.9 | 134.1 | 36.0 | 1.758 | 515 ms | 97.9 ms | 36.5 ms | 166 ms |
| torus, shuffled | 92.0 | 135.7 | 36.0 | 1.791 | 2195 ms | 127.7 ms | 58.0 ms | 453 ms |
| irregular | 87.6 | 132.3 | 36.0 | 1.713 | 636 ms | 101.5 ms | 40.7 ms | 155 ms |
| irregular, shuffled | 92.5 | 136.2 | 36.0 | 1.798 | 2525 ms | 126.5 ms | 59.0 ms | 424 ms |

- The arena holds the shared patch layout: an 8-word header, a 16-word
  record per patch, the patch regions, and a mask buffer of used counts,
  active masks and owned masks.
- Both builds slow down on shuffled input: the CPU build by 4.0–4.6x and
  the GPU build by 2.7–3.3x (sphere aside). Patches stay almost as good
  (1.79 against 1.64 faces held per face).
- Host-side setup is within a few percent of `e6f8663`. It runs 10–24%
  slower than in milestone 2 for the CPU build and the flat upload, neither
  of which changed, so the machine has been slower on the CPU since then.
  The arena upload takes 8–52% longer than in milestone 2; it now also lays
  out the mask buffer.
- The sphere's GPU build is 13x slower than the grid's at this size, and
  12x at 523 K faces (329 against 28 ms). It took 2.3 s at `e6f8663`. The
  build kernels now also skip zero-filling workgroup memory, but that is
  not shown to explain the drop. Its shuffled copy builds faster (758 ms).
  Not yet profiled.

## Queries with a flat twin

Median GPU time in ms, patched / flat, at 2.1 M faces and patch size 128.
The ratio is patched divided by flat; above 1 means the arena is slower.

| Query | Grid | Grid, shuffled | Sphere | Sphere, shuffled |
|---|---|---|---|---|
| Face vertices | 0.65 / 0.34 (1.9x) | 1.23 / 0.86 (1.4x) | 0.73 / 0.34 (2.1x) | 1.25 / 0.86 (1.5x) |
| Edge records | 1.09 / 1.04 (1.05x) | 1.64 / 3.66 (0.45x) | 1.45 / 1.02 (1.4x) | 1.83 / 3.64 (0.50x) |
| Face centroids | 0.88 / 0.59 (1.5x) | 1.95 / 1.58 (1.2x) | 0.95 / 0.59 (1.6x) | 1.96 / 1.57 (1.2x) |
| VV + FF + diamonds + edges | 2.56 / 2.82 (0.91x) | 9.31 / 13.11 (0.71x) | 3.04 / 2.82 (1.08x) | 9.63 / 13.20 (0.73x) |
| Adjacency + 10 smoothing steps | 7.23 / 7.49 (0.97x) | 18.39 / 22.18 (0.83x) | 11.29 / 11.07 (1.02x) | 24.65 / 28.26 (0.87x) |

| Query | Torus | Torus, shuffled | Irregular | Irregular, shuffled |
|---|---|---|---|---|
| Face vertices | 0.76 / 0.34 (2.2x) | 1.23 / 0.86 (1.4x) | 0.70 / 0.34 (2.1x) | 1.21 / 0.86 (1.4x) |
| Edge records | 1.27 / 1.02 (1.25x) | 1.65 / 3.66 (0.45x) | 1.16 / 1.04 (1.12x) | 1.58 / 3.66 (0.43x) |
| Face centroids | 0.97 / 0.58 (1.7x) | 1.95 / 1.58 (1.2x) | 0.92 / 0.59 (1.6x) | 1.92 / 1.58 (1.2x) |
| VV + FF + diamonds + edges | 2.85 / 2.82 (1.01x) | 9.43 / 13.24 (0.71x) | 2.79 / 2.91 (0.96x) | 9.27 / 13.13 (0.71x) |
| Adjacency + 10 smoothing steps | 7.55 / 7.52 (1.00x) | 18.50 / 22.32 (0.83x) | 8.10 / 8.23 (0.98x) | 18.26 / 22.14 (0.82x) |

Median wall time for the same runs, grid in input order: face vertices 1.80 /
1.50 ms, edge records 3.15 / 3.21 ms, centroids 3.27 / 3.00 ms, batched
adjacency 9.63 / 10.70 ms, and adjacency plus smoothing 16.0 / 17.1 ms.

Earlier runs, same queries, patched: edge records took 6.29–7.22 ms in
input order and 6.42–7.49 ms shuffled in milestone 2, and 2.67–3.26 and
3.54–3.93 ms at `e6f8663`. Batched adjacency took 8.22–9.29 and
15.09–16.28 ms in milestone 2, and 4.55–5.37 and 9.71–10.10 ms at
`e6f8663`. The flat path's GPU times are within 5% of milestone 2's.

At 523 K faces the arena is still slower on most twin queries, because the
flat path's penalty for shuffled input is smaller there: flat edge records
take 0.27–0.29 ms shuffled against 0.15 ms in input order. Edge records are
1.9–2.3x flat in input order and 1.5–1.8x shuffled, and batched adjacency
1.1–1.2x and 0.88–1.06x. At 32 K faces every query takes under 1 ms of GPU
time and at most about 2 ms wall, and the fixed per-call cost dominates.

What the columns show:

- **Edge records** cost O(1) per face: each face writes its local slot into
  its edges' entries in workgroup memory, then each owned edge reads its two
  entries. Staging is still most of the pass; the record writes take about
  0.3 ms at this size. Shuffling barely moves it, because its reads stay
  inside one patch.
- **Batched adjacency** is one staged pass for edge records, FF and
  diamonds, then the global VV count, scan and scatter. VV alone is
  0.86–0.96 ms in input order and 2.5–2.9 ms shuffled. On shuffled meshes
  the staged pass's global writes of FF and diamonds land scattered, and
  that cost did not improve.
- **Face vertices and centroids** cost 1.2–2.2x on the arena from
  indirection (record, local IDs, global IDs) and the active and owned
  checks; they are 9–27% slower than in milestone 2.
- **Smoothing** reads the global VV rows, so it slows on shuffled meshes on
  both paths. On the sphere one invocation walks each pole's 1448 neighbors.

## Queries without a twin

Median GPU time (wall) in ms, patched, at 2.1 M faces and patch size 128.

| Query | Grid | Grid, shuffled | Sphere | Sphere, shuffled | Torus | Torus, shuffled | Irregular | Irregular, shuffled |
|---|---|---|---|---|---|---|---|---|
| Vertex faces | 1.32 (4.03) | 4.32 (7.09) | 1.42 (4.12) | 4.34 (7.07) | 1.40 (4.12) | 4.54 (7.38) | 1.45 (4.13) | 4.63 (7.44) |
| Face normals | 0.88 (3.30) | 1.95 (4.37) | 0.95 (3.37) | 1.96 (4.33) | 0.97 (3.37) | 2.56 (5.09) | 0.92 (3.26) | 2.52 (5.10) |
| Face areas | 0.81 (2.33) | 1.90 (3.44) | 0.89 (2.40) | 1.91 (3.39) | 0.91 (2.42) | 2.49 (4.14) | 0.85 (2.41) | 2.46 (4.20) |
| Vertex normals | 2.59 (7.20) | 11.49 (16.06) | 3.34 (7.88) | 11.76 (16.43) | 2.79 (7.35) | 13.55 (18.42) | 2.94 (7.45) | 13.49 (18.63) |
| Boundary flags | 1.16 (2.08) | 1.70 (2.67) | 1.28 (2.20) | 1.50 (2.43) | 1.33 (2.32) | 2.13 (3.10) | 1.05 (1.99) | 2.01 (2.95) |
| VV, edge records given | 0.89 (1.66) | 2.57 (3.35) | 0.89 (1.64) | 2.54 (3.29) | 0.86 (1.62) | 2.86 (3.62) | 0.96 (1.70) | 2.59 (3.51) |
| 10 smoothing steps, adjacency given | 5.85 (7.92) | 13.32 (16.04) | 8.33 (10.36) | 15.12 (17.17) | 5.76 (7.76) | 14.48 (17.03) | 6.81 (8.78) | 13.85 (16.44) |
| Sum of face areas | 0.17 (0.70) | 0.25 (0.94) | 0.13 (0.71) | 0.13 (0.73) | 0.17 (0.80) | 0.25 (0.95) | 0.17 (0.80) | 0.22 (0.93) |
| Dot of face areas | 0.18 (0.67) | 0.22 (0.87) | 0.13 (0.72) | 0.13 (0.73) | 0.18 (0.75) | 0.22 (0.81) | 0.17 (0.78) | 0.22 (0.82) |

- **Boundary flags** run one staged pass and write the flags directly, with
  no edge records: 2.4–3.0 and 3.6–4.4 ms at `e6f8663`, and 6.5–7.4 ms in
  milestone 2.
- **Vertex normals** slow 3.5–4.9x on shuffled input. They gather over VF
  rows of scattered faces.
- **Smoothing depends on the adjacency it reads.** This row reuses one
  adjacency built at setup, so each benchmark run measures one build. At
  patch size 128 it measured 13.3–14.5 ms on the shuffled grid, torus and
  irregular grid, against 11.2–12.4 ms at `e6f8663`, and 5.8 against 4.9 ms
  on the torus. A side-by-side run at patch size 512, alternating builds
  of `e6f8663`'s code and this code, gave overlapping ranges (for example
  4.8 against 4.9 ms on the torus), so the difference is not attributed to
  this change. VV row order is unspecified and differs between builds,
  which is the likely factor; not verified.
- Reductions read one attribute linearly and are unaffected by mesh shape.

## Patch size

Median GPU time in ms, patched / flat, for patch sizes 32, 128 and 512 in the
same runs. Edge records and batched adjacency have flat twins; boundary
flags and VV do not. The default patch size is 512.

At 2.1 M faces:

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | VV | GPU build |
|---|---:|---:|---:|---|---|---:|---:|---:|
| grid | 32 | 123.5 | 2.338 | 2.72 / 1.04 | 4.47 / 2.81 | 3.55 | 0.90 | 144 ms |
| grid | 128 | 83.3 | 1.636 | 1.09 / 1.04 | 2.56 / 2.82 | 1.16 | 0.89 | 123 ms |
| grid | 512 | 65.2 | 1.307 | 0.95 / 1.03 | 2.38 / 2.80 | 0.82 | 0.83 | 146 ms |
| grid, shuffled | 32 | 146.2 | 2.725 | 4.10 / 3.66 | 10.63 / 13.10 | 5.20 | 2.56 | 458 ms |
| grid, shuffled | 128 | 91.8 | 1.786 | 1.64 / 3.66 | 9.31 / 13.11 | 1.70 | 2.57 | 406 ms |
| grid, shuffled | 512 | 69.2 | 1.382 | 1.38 / 3.68 | 9.05 / 13.12 | 1.75 | 2.72 | 403 ms |
| sphere | 32 | 158.7 | 2.953 | 3.95 / 1.02 | 5.93 / 2.84 | 4.42 | 0.93 | 6611 ms |
| sphere | 128 | 98.1 | 1.902 | 1.45 / 1.02 | 3.04 / 2.82 | 1.28 | 0.89 | 1599 ms |
| sphere | 512 | 72.2 | 1.435 | 1.18 / 1.02 | 2.72 / 2.84 | 1.05 | 0.89 | 566 ms |
| sphere, shuffled | 32 | 168.6 | 3.118 | 4.37 / 3.65 | 10.99 / 13.22 | 4.00 | 2.53 | 1394 ms |
| sphere, shuffled | 128 | 102.9 | 1.982 | 1.83 / 3.64 | 9.63 / 13.20 | 1.50 | 2.54 | 758 ms |
| sphere, shuffled | 512 | 75.2 | 1.487 | 1.58 / 3.65 | 9.37 / 13.20 | 1.27 | 2.53 | 530 ms |
| torus | 32 | 136.7 | 2.575 | 3.30 / 1.02 | 5.19 / 2.82 | 4.04 | 0.90 | 198 ms |
| torus | 128 | 89.9 | 1.758 | 1.27 / 1.02 | 2.85 / 2.82 | 1.33 | 0.86 | 166 ms |
| torus | 512 | 67.5 | 1.354 | 1.06 / 1.02 | 2.56 / 2.80 | 0.93 | 0.87 | 185 ms |
| torus, shuffled | 32 | 146.7 | 2.733 | 4.11 / 3.65 | 10.74 / 13.23 | 5.09 | 2.69 | 535 ms |
| torus, shuffled | 128 | 92.0 | 1.791 | 1.65 / 3.66 | 9.43 / 13.24 | 2.13 | 2.86 | 453 ms |
| torus, shuffled | 512 | 69.5 | 1.387 | 1.39 / 3.66 | 9.14 / 13.23 | 1.65 | 2.69 | 436 ms |
| irregular | 32 | 132.6 | 2.500 | 2.96 / 1.04 | 4.89 / 2.91 | 3.59 | 0.99 | 186 ms |
| irregular | 128 | 87.6 | 1.713 | 1.16 / 1.04 | 2.79 / 2.91 | 1.05 | 0.96 | 155 ms |
| irregular | 512 | 67.4 | 1.347 | 1.04 / 1.03 | 2.61 / 2.90 | 0.91 | 0.93 | 162 ms |
| irregular, shuffled | 32 | 145.6 | 2.715 | 3.85 / 3.65 | 10.30 / 13.11 | 5.13 | 2.58 | 501 ms |
| irregular, shuffled | 128 | 92.5 | 1.798 | 1.58 / 3.66 | 9.27 / 13.13 | 2.01 | 2.59 | 424 ms |
| irregular, shuffled | 512 | 69.4 | 1.383 | 1.38 / 3.64 | 9.02 / 13.12 | 1.77 | 2.59 | 414 ms |

At 523 K faces, on the grid (the other meshes follow the same pattern):

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | VV | GPU build |
|---|---:|---:|---:|---|---|---:|---:|---:|
| grid | 32 | 123.5 | 2.337 | 0.69 / 0.15 | 1.18 / 0.64 | 0.62 | 0.25 | 34 ms |
| grid | 128 | 83.3 | 1.635 | 0.29 / 0.15 | 0.72 / 0.64 | 0.24 | 0.25 | 28 ms |
| grid | 512 | 65.4 | 1.311 | 0.27 / 0.15 | 0.69 / 0.64 | 0.22 | 0.25 | 32 ms |
| grid, shuffled | 32 | 146.1 | 2.722 | 1.36 / 0.29 | 1.93 / 1.45 | 1.13 | 0.57 | 53 ms |
| grid, shuffled | 128 | 91.5 | 1.780 | 0.52 / 0.29 | 1.43 / 1.35 | 0.45 | 0.57 | 46 ms |
| grid, shuffled | 512 | 69.2 | 1.381 | 0.35 / 0.27 | 1.27 / 1.31 | 0.28 | 0.54 | 43 ms |

- **Staged queries get faster as patches grow**, with the largest gain
  between 32 and 128. Larger patches have fewer workgroups and fewer
  ribbon copies per face.
- **Patch size 512 is the best setting measured**, and the default since
  27 November 2026, as in RXMesh. At 2.1 M faces it gives the lowest time
  on every staged query except boundary flags on the shuffled grid (within
  noise of 128), and the least memory (65–75 B/face).
- **Patch size 32 is now close to milestone 2** on staged edge records: at
  523 K faces, 0.69 ms on the grid and 0.97 ms on the sphere, against 0.61
  and 0.72 ms then (1.78 and 2.38 ms at `e6f8663`). At 2.1 M faces it takes
  2.7–4.4 ms. It is the most memory-hungry setting too (124–169 B/face).
- **Boundary flags track edge records** at every size, since they run the
  same staging.
- **VV does not depend on patch size**, because it is a global pass over
  the edge records.
- **The sphere's GPU build falls as patches grow** (6.6 s at 32, 0.57 s at
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
  memory traffic and per-call overhead differently, and the best lane count
  and staging size may differ there.
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
