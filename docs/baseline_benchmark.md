# Benchmark baseline (ROADMAP milestones 2 and 3)

Setup, memory and warm query costs of the patch arena, measured against an
unpartitioned GPU baseline on the same adapter. No speedup is assumed; the
point is to find where the patch layout costs or pays. The baseline was
first run for milestone 2 (26 November 2026) and rerun after milestone 3's
workgroup staging (27 November 2026); this report gives the rerun, with
milestone 2 figures where they show what changed.

## Bottom line

- **On shuffled meshes the arena now beats the flat layout for adjacency.**
  At 2.1 M faces and patch size 128, batched adjacency (VV, FF, diamonds and
  edge records) takes 9.7–10.1 ms on the arena against 13.1–13.2 ms flat
  (0.75x). Edge records are level (0.97–1.08x). In milestone 2 the arena
  was slower on both (1.15–1.2x and 1.8–2.0x).
- **In input order the arena is still slower**, but by much less: edge
  records 2.6–3.2x (from 5.8–6.9x) and batched adjacency 1.6–1.9x (from
  2.9–3.2x). Edge records take 2.7–3.3 ms against 1.0–1.1 ms flat.
- **Bigger patches are now faster, not slower.** Each workgroup stages its
  patch's edge faces in O(1) per face, so there is no per-edge scan. At
  2.1 M faces and patch size 512, edge records take 1.6–1.9 ms in input
  order (1.5–1.9x flat) and 2.0–2.2 ms shuffled (0.55–0.6x flat); batched
  adjacency takes 3.4–3.8 and 8.9–9.2 ms. Patch size 32 is now the slow
  case: edge records 8.8–11.0 ms. Its cause is not profiled.
- **Memory is 83–103 bytes per face at patch size 128, from 129–145**,
  against 36 flat (2.3–2.9x). Patch size 512 takes 65–75 bytes per face.
- **Boundary flags no longer cost a second pass**: 2.4–3.0 ms in input order
  and 3.4–4.4 ms shuffled, from 6.5–7.4 ms.
- **Per-face queries on the arena got 9–27% slower** (face vertices,
  centroids, normals, areas; face areas on the shuffled grid 44%), from the
  per-patch dispatch and the active and owned checks. The flat path's GPU
  times are within 5% of milestone 2's, so this is not machine drift.
- **VV alone is 0.86–0.94 ms in input order and 2.5–2.6 ms shuffled**, a
  quarter of shuffled adjacency. Most of the rest is the staged pass that
  writes edge records, FF and diamonds. Staging VV was skipped for that
  reason.
- **The GPU build still degrades on high-valence vertices.** The 2.1 M-face
  UV sphere builds in 2.5 s at patch size 128 and 6.8 s at patch size 32,
  against 129 and 152 ms for a grid of the same size. Not yet profiled.

## Running it

```sh
MOPMESH_BENCH=1 cargo test --release --lib gpu_baseline_benchmark -- --ignored --nocapture
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `MOPMESH_BENCH_FACES` | `32768,524288,2097152` | Target face counts for the generated meshes |
| `MOPMESH_BENCH_PATCHES` | `512` | Patch sizes (`128` when this report was run) |
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
  the three runs' medians. Of the 1,059 rows with a median of at least
  0.1 ms, 79% have at least two runs within 5% of that figure, and the three
  medians span under 10% in 61%. Treat differences under about 10% as noise.

**Counts.** Dispatches come from a test-only counter in the shared dispatch
helper. Buffers and bind groups come from wgpu's internal counters (the
`counters` feature, enabled for tests only).

**Machine.** Apple M1 Max, Metal, release build, 2026-11-27, commit
`746a47f`. Storage binding limit 4 GiB, 65535 workgroups per dimension,
timestamp period 1 ns.

## Memory and setup

At 2.1 M faces and patch size 128. Both upload columns include laying out
the data on the host from the CPU mesh.

| Mesh | Arena B/face | Milestone 2 | Flat B/face | Faces held per face | CPU build | Upload arena | Upload flat | GPU build |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid | 83.3 | 128.7 | 36.0 | 1.636 | 470 ms | 89.1 ms | 33.5 ms | 129 ms |
| grid, shuffled | 91.8 | 135.5 | 36.0 | 1.786 | 2185 ms | 126.3 ms | 58.3 ms | 407 ms |
| sphere | 98.1 | 141.0 | 36.0 | 1.902 | 506 ms | 103.9 ms | 35.0 ms | 2543 ms |
| sphere, shuffled | 102.9 | 144.9 | 36.0 | 1.982 | 2301 ms | 138.8 ms | 60.5 ms | 754 ms |
| torus | 89.9 | 134.1 | 36.0 | 1.758 | 517 ms | 100.2 ms | 36.8 ms | 171 ms |
| torus, shuffled | 92.0 | 135.7 | 36.0 | 1.791 | 2187 ms | 126.8 ms | 57.7 ms | 454 ms |
| irregular | 87.6 | 132.3 | 36.0 | 1.713 | 635 ms | 103.6 ms | 41.5 ms | 155 ms |
| irregular, shuffled | 92.5 | 136.2 | 36.0 | 1.798 | 2602 ms | 123.8 ms | 59.5 ms | 422 ms |

- The arena now holds the shared patch layout: an 8-word header, a 16-word
  record per patch, the patch regions, and a mask buffer of used counts,
  active masks and owned masks. The per-face task lists are gone.
- Both builds slow down on shuffled input: the CPU build by 4.1–4.6x and
  the GPU build by 2.7–3.2x (sphere aside). Patches stay almost as good
  (1.79 against 1.64 faces held per face).
- Host-side setup ran 11–24% slower than in milestone 2 for the CPU build
  and 2–18% for the flat upload, neither of which changed, so the machine
  was slower on the CPU this time. The arena upload took 8–52% longer,
  more than that accounts for; it now also lays out the mask buffer.
- The sphere's GPU build is 20x slower than the grid's at this size, and
  12x at 523 K faces (349 against 28 ms). Its shuffled copy builds faster
  (754 ms). This is not yet profiled.

## Queries with a flat twin

Median GPU time in ms, patched / flat, at 2.1 M faces and patch size 128.
The ratio is patched divided by flat; above 1 means the arena is slower.

| Query | Grid | Grid, shuffled | Sphere | Sphere, shuffled |
|---|---|---|---|---|
| Face vertices | 0.65 / 0.34 (1.9x) | 1.23 / 0.86 (1.4x) | 0.73 / 0.34 (2.1x) | 1.25 / 0.86 (1.5x) |
| Edge records | 2.67 / 1.04 (2.6x) | 3.67 / 3.66 (1.0x) | 3.26 / 1.01 (3.2x) | 3.93 / 3.65 (1.1x) |
| Face centroids | 0.88 / 0.59 (1.5x) | 1.95 / 1.58 (1.2x) | 0.95 / 0.59 (1.6x) | 1.96 / 1.57 (1.2x) |
| VV + FF + diamonds + edges | 4.47 / 2.84 (1.6x) | 9.81 / 13.12 (0.75x) | 5.24 / 2.84 (1.8x) | 10.07 / 13.23 (0.76x) |
| Adjacency + 10 smoothing steps | 9.14 / 7.50 (1.2x) | 18.91 / 22.22 (0.85x) | 13.49 / 11.09 (1.2x) | 25.15 / 28.30 (0.89x) |

| Query | Torus | Torus, shuffled | Irregular | Irregular, shuffled |
|---|---|---|---|---|
| Face vertices | 0.76 / 0.34 (2.2x) | 1.23 / 0.86 (1.4x) | 0.70 / 0.34 (2.1x) | 1.21 / 0.86 (1.4x) |
| Edge records | 3.22 / 1.02 (3.2x) | 3.68 / 3.66 (1.0x) | 2.90 / 1.05 (2.8x) | 3.54 / 3.65 (0.97x) |
| Face centroids | 0.98 / 0.59 (1.7x) | 1.95 / 1.58 (1.2x) | 0.92 / 0.59 (1.6x) | 1.92 / 1.58 (1.2x) |
| VV + FF + diamonds + edges | 5.14 / 2.82 (1.8x) | 9.92 / 13.24 (0.75x) | 4.86 / 2.92 (1.7x) | 9.67 / 13.13 (0.74x) |
| Adjacency + 10 smoothing steps | 9.85 / 7.54 (1.3x) | 18.99 / 22.30 (0.85x) | 10.18 / 8.25 (1.2x) | 18.66 / 22.14 (0.84x) |

Median wall time for the same runs, grid in input order: face vertices 1.80 /
1.50 ms, edge records 4.79 / 3.23 ms, centroids 3.30 / 3.02 ms, batched
adjacency 11.8 / 11.0 ms, and adjacency plus smoothing 18.0 / 17.2 ms.

Milestone 2, same queries, patched: edge records 6.29–7.22 ms in input
order and 6.42–7.49 ms shuffled; batched adjacency 8.22–9.29 and
15.09–16.28 ms. The flat path's GPU times are within 5% of milestone 2's.

At 523 K faces the arena is slower on every twin query (edge records
3.1–5.4x, batched adjacency 1.4–2.2x), because the flat path's penalty for
shuffled input is smaller there: flat edge records take 0.27–0.29 ms
shuffled against 0.15–0.16 ms in input order. At 32 K faces every query
takes under 1 ms of GPU time and at most about 2 ms wall, and the fixed
per-call cost dominates.

What the columns show:

- **Edge records** now cost O(1) per face: each face writes its local slot
  into its edges' entries in workgroup memory, then each owned edge reads its
  two entries. The arena's remaining input-order gap is the staged pass
  itself (patch records, masks, local-to-global lookups). Shuffling barely
  moves it, because its reads stay inside one patch.
- **Batched adjacency** is one staged pass for edge records, FF and
  diamonds, then the global VV count, scan and scatter. VV alone is
  0.86–0.94 ms in input order and 2.5–2.6 ms shuffled; the staged pass is
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
| Vertex faces | 1.32 (4.06) | 4.39 (7.19) | 1.42 (4.15) | 4.33 (7.24) | 1.41 (4.13) | 4.50 (7.28) | 1.45 (4.25) | 4.64 (7.47) |
| Face normals | 0.88 (3.30) | 2.08 (4.63) | 0.95 (3.34) | 1.96 (4.36) | 0.97 (3.31) | 2.56 (5.01) | 0.92 (3.37) | 2.52 (5.12) |
| Face areas | 0.81 (2.35) | 2.46 (3.94) | 0.89 (2.42) | 1.92 (3.43) | 0.91 (2.40) | 2.49 (4.00) | 0.85 (2.39) | 2.46 (4.10) |
| Vertex normals | 2.58 (6.91) | 12.77 (17.63) | 3.33 (7.85) | 11.78 (16.59) | 2.79 (7.29) | 13.14 (18.45) | 2.93 (7.35) | 13.01 (18.29) |
| Boundary flags | 2.39 (3.39) | 4.41 (5.59) | 2.95 (3.97) | 3.57 (4.62) | 2.92 (3.96) | 3.36 (4.37) | 2.59 (3.63) | 4.27 (5.44) |
| VV, edge records given | 0.86 (1.65) | 2.57 (3.33) | 0.91 (1.67) | 2.54 (3.31) | 0.87 (1.64) | 2.53 (3.33) | 0.94 (1.78) | 2.55 (3.30) |
| 10 smoothing steps, adjacency given | 4.90 (7.12) | 13.08 (15.85) | 8.31 (10.35) | 15.14 (17.49) | 5.04 (7.02) | 9.15 (11.18) | 5.35 (7.66) | 12.74 (15.36) |
| Sum of face areas | 0.18 (0.70) | 0.23 (0.91) | 0.14 (0.67) | 0.14 (0.70) | 0.14 (0.71) | 0.14 (0.72) | 0.14 (0.68) | 0.14 (0.73) |
| Dot of face areas | 0.18 (0.70) | 0.23 (0.93) | 0.14 (0.66) | 0.14 (0.69) | 0.15 (0.73) | 0.14 (0.73) | 0.14 (0.68) | 0.14 (0.73) |

- **Boundary flags** run one staged pass and write the flags directly, with
  no edge records: 6.5–7.4 ms in milestone 2.
- **Vertex normals** slow 3.5–4.9x on shuffled input. They gather over VF
  rows of scattered faces. On the shuffled grid they take 12.77 ms against
  10.42 ms in milestone 2 (23%), more than face normals' own change
  explains; not investigated.
- **Smoothing with adjacency given** is unreliable on shuffled meshes in
  this run. On the shuffled grid it takes 13.1 ms here, but subtracting
  batched adjacency from adjacency plus smoothing gives 9.1 ms, as in
  milestone 2. This row runs right after the new VV row, which rewrites the
  same adjacency's VV buffer; that is the likely cause, not yet checked.
- Reductions read one attribute linearly and are unaffected by mesh shape.

## Patch size

Median GPU time in ms, patched / flat, for patch sizes 32, 128 and 512 in the
same runs. Edge records and batched adjacency have flat twins; boundary
flags and VV do not.

At 2.1 M faces:

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | VV | GPU build |
|---|---:|---:|---:|---|---|---:|---:|---:|
| grid | 32 | 123.5 | 2.338 | 8.80 / 1.07 | 11.04 / 2.84 | 8.37 | 0.85 | 152 ms |
| grid | 128 | 83.3 | 1.636 | 2.67 / 1.04 | 4.47 / 2.84 | 2.39 | 0.86 | 129 ms |
| grid | 512 | 65.2 | 1.307 | 1.64 / 1.08 | 3.43 / 2.85 | 1.38 | 0.87 | 160 ms |
| grid, shuffled | 32 | 146.2 | 2.725 | 10.61 / 3.67 | 15.61 / 13.12 | 12.67 | 2.41 | 460 ms |
| grid, shuffled | 128 | 91.8 | 1.786 | 3.67 / 3.66 | 9.81 / 13.12 | 4.41 | 2.57 | 407 ms |
| grid, shuffled | 512 | 69.2 | 1.382 | 2.01 / 3.65 | 8.91 / 13.09 | 2.41 | 2.53 | 407 ms |
| sphere | 32 | 158.7 | 2.953 | 10.33 / 1.02 | 12.72 / 2.84 | 9.86 | 0.91 | 6780 ms |
| sphere | 128 | 98.1 | 1.902 | 3.26 / 1.01 | 5.24 / 2.84 | 2.95 | 0.91 | 2543 ms |
| sphere | 512 | 72.2 | 1.435 | 1.94 / 1.01 | 3.81 / 2.84 | 1.66 | 0.92 | 469 ms |
| sphere, shuffled | 32 | 168.6 | 3.118 | 11.02 / 3.68 | 16.04 / 13.25 | 11.93 | 2.56 | 1413 ms |
| sphere, shuffled | 128 | 102.9 | 1.982 | 3.93 / 3.65 | 10.07 / 13.23 | 3.57 | 2.54 | 754 ms |
| sphere, shuffled | 512 | 75.2 | 1.487 | 2.19 / 3.64 | 9.17 / 13.23 | 2.91 | 2.66 | 538 ms |
| torus | 32 | 136.7 | 2.575 | 9.61 / 1.02 | 11.90 / 2.84 | 9.15 | 0.88 | 208 ms |
| torus | 128 | 89.9 | 1.758 | 3.22 / 1.02 | 5.14 / 2.82 | 2.92 | 0.87 | 171 ms |
| torus | 512 | 67.5 | 1.354 | 1.84 / 1.02 | 3.69 / 2.84 | 1.57 | 0.89 | 186 ms |
| torus, shuffled | 32 | 146.7 | 2.733 | 10.64 / 3.67 | 15.75 / 13.24 | 12.91 | 2.53 | 542 ms |
| torus, shuffled | 128 | 92.0 | 1.791 | 3.68 / 3.66 | 9.92 / 13.24 | 3.36 | 2.53 | 454 ms |
| torus, shuffled | 512 | 69.5 | 1.387 | 2.02 / 3.65 | 9.00 / 13.24 | 2.24 | 2.71 | 445 ms |
| irregular | 32 | 132.6 | 2.500 | 9.14 / 1.08 | 11.52 / 2.93 | 8.73 | 0.94 | 195 ms |
| irregular | 128 | 87.6 | 1.713 | 2.90 / 1.05 | 4.86 / 2.92 | 2.59 | 0.94 | 155 ms |
| irregular | 512 | 67.4 | 1.347 | 1.84 / 1.05 | 3.78 / 2.92 | 1.56 | 0.94 | 165 ms |
| irregular, shuffled | 32 | 145.6 | 2.715 | 10.28 / 3.66 | 15.34 / 13.13 | 12.58 | 2.42 | 503 ms |
| irregular, shuffled | 128 | 92.5 | 1.798 | 3.54 / 3.65 | 9.67 / 13.13 | 4.27 | 2.55 | 422 ms |
| irregular, shuffled | 512 | 69.4 | 1.383 | 2.03 / 3.66 | 8.90 / 13.12 | 2.17 | 2.55 | 415 ms |

At 523 K faces, on the grid (the other meshes follow the same pattern):

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | VV | GPU build |
|---|---:|---:|---:|---|---|---:|---:|---:|
| grid | 32 | 123.5 | 2.337 | 1.78 / 0.16 | 2.42 / 0.64 | 1.65 | 0.25 | 32 ms |
| grid | 128 | 83.3 | 1.635 | 0.69 / 0.16 | 1.21 / 0.64 | 0.60 | 0.25 | 28 ms |
| grid | 512 | 65.4 | 1.311 | 0.45 / 0.15 | 0.98 / 0.64 | 0.36 | 0.25 | 32 ms |
| grid, shuffled | 32 | 146.1 | 2.722 | 3.12 / 0.33 | 3.66 / 1.32 | 2.48 | 0.55 | 53 ms |
| grid, shuffled | 128 | 91.5 | 1.780 | 0.94 / 0.28 | 1.86 / 1.33 | 0.85 | 0.55 | 46 ms |
| grid, shuffled | 512 | 69.2 | 1.381 | 0.50 / 0.27 | 1.48 / 1.33 | 0.41 | 0.55 | 43 ms |

- **Staged edge records get faster as patches grow.** In milestone 2 they
  grew linearly with patch size (0.61 ms at patch size 32 and 5.00 ms at
  512, grid at 523 K faces); now they fall (1.78 and 0.45 ms). Larger
  patches have fewer workgroups and fewer ribbon copies per face, which
  plausibly explains it; not profiled.
- **Patch size 512 is the best setting measured.** At 2.1 M faces it gives
  the lowest time on every staged query and the least memory (65–75 B/face).
  Shuffled, its edge records run at 0.55–0.6x the flat path and batched
  adjacency at 0.68–0.69x; in input order they are 1.5–1.9x and 1.2–1.3x.
  The default patch size was 128 when this report was run; it is now 512
  (decided 27 November 2026), matching RXMesh's default.
- **Patch size 32 is about 3x slower than milestone 2** on staged edge
  records: at 523 K faces, 1.78 ms on the grid and 2.38 ms on the sphere,
  against 0.61 and 0.72 ms then. At 2.1 M faces it takes 8.8–11.0 ms. It is
  the most memory-hungry setting too (124–169 B/face). The cause is not
  profiled.
- **Boundary flags track edge records** at every size, since they run the
  same staging.
- **VV does not depend on patch size**, because it is a global pass over
  the edge records.
- **The sphere's GPU build falls as patches grow** (6.8 s at 32, 0.47 s at
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
  kernels, plus spare slots.
