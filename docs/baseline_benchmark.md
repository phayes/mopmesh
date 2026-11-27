# Benchmark baseline (ROADMAP milestone 2)

Setup, memory and warm query costs of the patch arena, measured against an
unpartitioned GPU baseline on the same adapter. No speedup is assumed; the
point is to find where the patch layout costs or pays, and to choose the
milestone 3 experiment.

## Bottom line

- **The patch arena is slower than the flat baseline on every query today.**
  At 2.1 M faces in input order, edge records take 6.3 ms on the arena against
  1.1 ms flat (5.8x), and the batched adjacency takes 8.2 against 2.9 ms (2.9x).
  Edge records are the main cost: each owned edge scans every face its owner
  patch holds.
- **The arena is insensitive to element order; the flat layout is not.** With
  vertex and face IDs shuffled, flat edge records slow from 1.1 to 3.7 ms,
  while the arena's stay at 6.4 ms. Batched adjacency becomes 15.1 against
  13.1 ms (1.15x).
- **The kernels after the edge records dominate on shuffled meshes, on both
  paths.** They read global arrays (the VV transpose, FF, diamonds), and on
  shuffled input they cost about 7.6 ms against 1.4 ms in input order. No
  query uses patch locality there yet.
- **The arena takes 3.6–4x the memory**: 129–145 bytes per face against 36.
  Patches hold 1.6–2.0 faces per face (ribbons included).
- **Per-call overhead is large.** Wall time exceeds summed GPU pass time by
  about 8 ms for batched adjacency at 2.1 M faces. That gap is host encoding,
  the 16 buffers each call allocates (review finding P3), and gaps between
  passes. At that size it is 73% of the flat path's wall time.
- **The GPU build degrades badly on high-valence vertices.** A 2.1 M-face UV
  sphere (two poles of valence 1448) builds in 3.1 s, against 109 ms for a
  grid of the same size. The CPU build is unaffected (418 against 404 ms).
- **Milestone 3 candidate: edge incidence**, as the roadmap suggested. It is
  the largest gap and is local by construction. The global VV/FF kernels are
  the second target, because only patch-local traversal can make them
  insensitive to element order.

## Running it

```sh
MOPMESH_BENCH=1 cargo test --release --lib gpu_baseline_benchmark -- --ignored --nocapture
```

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `MOPMESH_BENCH_FACES` | `32768,524288,2097152` | Target face counts for the generated meshes |
| `MOPMESH_BENCH_PATCHES` | `128` | Patch sizes |
| `MOPMESH_BENCH_OBJ` | none | OBJ files to add, each with a shuffled copy |
| `MOPMESH_BENCH_RUNS` | `11` | Timed runs per query, after 2 warm-up runs |

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
36 bytes per face. It has twins of the two kernels that read the arena. Face
vertices are decoded the same way from global arrays. Edge records take three
passes: copy the endpoints, then each face writes its ID into its edges'
slots (the direction picks the slot), then move a boundary edge's face to the
first slot. Every other query (VV, FF, diamonds, centroids, smoothing) runs
the same kernels on both paths, because they read only global results.

**Timing.**

- Patched and flat runs alternate within each round, so machine drift affects
  both: 11 timed rounds after 2 warm-up rounds.
- *Wall time* covers encoding, including the buffers each `encode` call
  allocates, submitting, and waiting for the GPU.
- *GPU time* is the sum of each compute pass's own duration, from timestamp
  queries at pass boundaries. It leaves out gaps between passes. This
  adapter has no timestamps inside encoders.
- Setup rows are 3 timed runs after 1 warm-up.
- Tables show medians. The GPU min-to-max spread is under 2% for nearly every
  row, so differences above a few percent are real.

**Counts.** Dispatches come from a test-only counter in the shared dispatch
helper. Buffers and bind groups come from wgpu's internal counters (the
`counters` feature, enabled for tests only).

**Machine.** Apple M1 Max, Metal, release build, 2026-11-26, patch size 128.
Storage binding limit 4 GiB, 65535 workgroups per dimension, timestamp
period 1 ns.

## Memory and setup

At 2.1 M faces (medians). "Upload flat" includes deriving the global arrays on
the host.

| Mesh | Arena B/face | Flat B/face | Faces held per face | CPU build | Upload arena | Upload flat | GPU build |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid | 128.7 | 36.0 | 1.636 | 404 ms | 82.5 ms | 32.9 ms | 109 ms |
| grid, shuffled | 135.5 | 36.0 | 1.786 | 1962 ms | 87.5 ms | 50.0 ms | 407 ms |
| sphere | 141.0 | 36.0 | 1.902 | 418 ms | 73.6 ms | 30.2 ms | 3069 ms |
| sphere, shuffled | 144.9 | 36.0 | 1.982 | 2026 ms | 91.1 ms | 51.1 ms | 775 ms |
| torus | 134.1 | 36.0 | 1.758 | 428 ms | 73.3 ms | 32.8 ms | 136 ms |
| torus, shuffled | 135.7 | 36.0 | 1.791 | 1962 ms | 88.1 ms | 50.5 ms | 453 ms |
| irregular | 132.3 | 36.0 | 1.713 | 512 ms | 72.9 ms | 35.1 ms | 121 ms |
| irregular, shuffled | 136.2 | 36.0 | 1.798 | 2263 ms | 88.6 ms | 50.4 ms | 424 ms |

- Both builds slow down on shuffled input: the CPU build by about 4.8x and the
  GPU build by about 3.5x (sphere aside). Patches stay almost as good
  (1.79 against 1.64 faces held per face).
- The sphere's GPU build is 28x slower than the grid's at this size, and 10x at
  522 K faces (327 against 32 ms). Its shuffled copy builds faster (775 ms),
  so the cost depends on how the poles' fans land in patches, not on the
  valence alone.
- The same pattern holds at 522 K faces at about a quarter of these times.

## Queries with a flat twin

Median GPU time, patched / flat, at 2.1 M faces. The ratio is patched divided
by flat; above 1 means the arena is slower.

| Query | Grid | Grid, shuffled | Sphere | Sphere, shuffled | Torus | Irregular |
|---|---|---|---|---|---|---|
| Face vertices | 0.58 / 0.35 (1.7x) | 1.04 / 0.86 (1.2x) | 0.59 / 0.34 (1.7x) | 1.06 / 0.87 (1.2x) | 0.60 / 0.35 (1.7x) | 0.60 / 0.34 (1.8x) |
| Edge records | 6.29 / 1.09 (5.8x) | 6.42 / 3.66 (1.8x) | 7.22 / 1.05 (6.9x) | 7.49 / 3.67 (2.0x) | 6.62 / 1.06 (6.2x) | 6.54 / 1.09 (6.0x) |
| Face centroids | 0.81 / 0.59 (1.4x) | 1.77 / 1.58 (1.1x) | 0.83 / 0.58 (1.4x) | 1.77 / 1.57 (1.1x) | 0.83 / 0.59 (1.4x) | 0.83 / 0.59 (1.4x) |
| VV + FF + diamonds + edges | 8.22 / 2.86 (2.9x) | 15.09 / 13.14 (1.15x) | 9.29 / 2.87 (3.2x) | 16.28 / 13.25 (1.2x) | 8.67 / 2.86 (3.0x) | 8.65 / 2.94 (2.9x) |
| Adjacency + 10 smoothing steps | 12.88 / 7.51 (1.7x) | 24.17 / 22.23 (1.1x) | 17.55 / 11.10 (1.6x) | 31.40 / 28.36 (1.1x) | 13.37 / 7.55 (1.8x) | 13.96 / 8.25 (1.7x) |

Median wall time for the same runs, grid in input order: face vertices 1.76 /
1.49 ms, edge records 8.28 / 3.22 ms, centroids 3.07 / 2.89 ms, batched
adjacency 16.1 / 10.7 ms, and adjacency plus smoothing 22.2 / 16.8 ms.

At 522 K faces the ratios are similar in input order (edge records about
10x: 1.59 / 0.16 ms on the grid; batched adjacency about 3x). Shuffled, the
flat path's penalty is smaller: edge records 1.64 / 0.32 ms, batched
adjacency 2.72 / 1.32 ms. At 32 K faces every query takes under 1 ms of GPU
time and under 2 ms wall, and the fixed per-call cost of about 0.3 ms
dominates.

What the columns show:

- **Edge records** are the gap. The arena kernel scans all the faces of the
  owner patch, about 210 with ribbons, for each owned edge. The flat kernel
  writes each face once. The arena's time barely moves when the input is
  shuffled, because its reads stay inside one patch.
- **Face vertices and centroids** cost 1.4–1.8x on the arena from indirection
  alone: arena offsets, then local IDs, then global IDs.
- **Batched adjacency**, after the edge records, runs the same global kernels
  on both paths (VV count, scan and scatter, then FF and diamonds). In input
  order they take about 1.4 ms; shuffled, about 7.6 ms. This part is
  sensitive to element order, and patches do not help it today.
- **Smoothing** reads the global VV rows, so it slows on shuffled meshes on
  both paths. On the sphere one invocation walks each pole's 1448 neighbors.

## Queries without a twin

Median GPU time (wall), patched, at 2.1 M faces.

| Query | Grid | Grid, shuffled | Sphere | Torus | Irregular |
|---|---|---|---|---|---|
| Vertex faces | 1.25 (3.74) | 4.15 (6.73) | 1.31 (3.76) | 1.28 (3.81) | 1.37 (3.87) |
| Face normals | 0.81 (3.02) | 1.76 (4.00) | 0.83 (3.02) | 0.83 (3.06) | 0.83 (3.06) |
| Face areas | 0.74 (2.14) | 1.71 (3.16) | 0.75 (2.16) | 0.76 (2.18) | 0.76 (2.19) |
| Vertex normals | 2.42 (6.41) | 10.42 (14.48) | 3.03 (6.96) | 2.46 (6.52) | 2.73 (6.67) |
| Boundary flags | 6.54 (10.54) | 6.59 (9.19) | 7.39 (9.94) | 6.78 (9.38) | 6.71 (9.28) |
| 10 smoothing steps (adjacency given) | 5.91 (8.12) | 9.12 (11.01) | 8.32 (10.24) | 4.72 (6.55) | 5.32 (7.15) |
| Sum of face areas | 0.14 (0.68) | 0.14 (0.63) | 0.14 (0.76) | 0.14 (0.62) | 0.14 (0.69) |
| Dot of face areas | 0.14 (0.66) | 0.14 (0.69) | 0.14 (0.74) | 0.14 (0.67) | 0.14 (0.69) |

- Boundary flags cost about as much as the edge records, so they would
  benefit from the same milestone 3 work.
- Vertex normals slow 4x on shuffled input. They gather over VF rows of
  scattered faces.
- Reductions read one attribute linearly and are unaffected by mesh shape.

## Patch size

A second run at 522–524 K faces with patch sizes 32, 64, 256 and 512; the
128 column is from the main run. Same machine, same session. Median GPU time
in ms, patched / flat.

| Mesh | Patch size | Arena B/face | Faces held per face | Edge records | Batched adjacency | Boundary flags | GPU build (wall) |
|---|---:|---:|---:|---|---|---:|---:|
| grid | 32 | 160.8 | 2.337 | 0.61 / 0.15 | 1.18 / 0.65 | 0.65 | 38 ms |
| grid | 64 | 147.3 | 2.046 | 0.99 / 0.15 | 1.80 / 0.78 | 1.03 | 31 ms |
| grid | 128 | 128.7 | 1.635 | 1.59 / 0.16 | 2.13 / 0.69 | 1.64 | 32 ms |
| grid | 256 | 121.4 | 1.476 | 2.83 / 0.16 | 3.38 / 0.65 | 2.87 | 30 ms |
| grid | 512 | 114.0 | 1.311 | 5.00 / 0.16 | 5.55 / 0.67 | 5.04 | 34 ms |
| grid, shuffled | 32 | 178.4 | 2.722 | 0.64 / 0.28 | 1.76 / 1.32 | 0.68 | 47 ms |
| grid, shuffled | 512 | 117.1 | 1.381 | 5.07 / 0.33 | 6.18 / 1.32 | 5.11 | 46 ms |
| sphere | 32 | 189.3 | 2.949 | 0.72 / 0.15 | 1.31 / 0.64 | 0.76 | 1170 ms |
| sphere | 128 | 141.2 | 1.905 | 1.83 / 0.16 | 2.37 / 0.73 | 1.88 | 327 ms |
| sphere | 512 | 119.6 | 1.432 | 5.53 / 0.16 | 8.28 / 0.74 | 5.49 | 100 ms |

The torus and irregular grid follow the grid. The sweep's raw rows hold two
outliers, irregular grid at patch size 64 (adjacency 2.60 / 0.94, boundary
3.37), where the flat path also slowed, so they are drift rather than layout.

- **Edge records grow linearly with patch size** on the arena, about
  0.01 ms per face of patch size at this mesh size, and stay flat on the flat
  path. That is the per-owned-edge scan of the owner patch's faces.
- **Smaller patches trade memory for that scan, without closing the gap.**
  At patch size 32 the arena's edge records are still 4.1–4.8x slower than
  the flat path in input order and 2.3–3.4x shuffled, while the arena takes
  161–198 bytes per face and patches hold 2.3–3.1 faces per face.
- **Boundary flags track edge records** at every size, so they share the
  same bottleneck.
- **The sphere's GPU build falls as patches grow** (1170 ms at 32, 100 ms at
  512), so its cost comes from small patches around the high-valence poles.

## Dispatches and allocations

Per call, at 2.1 M faces. Counts are the same on both paths except for the
two extra edge passes on the flat path.

| Query | Dispatches (patched / flat) | Buffers | Bind groups |
|---|---|---:|---:|
| Face vertices | 1 / 1 | 1 | 1 |
| Edge records | 1 / 3 | 1 | 1 |
| Face centroids | 2 / 2 | 2 | 2 |
| VV + FF + diamonds + edges | 13 / 15 | 16 | 8 |
| Adjacency + 10 smoothing steps | 23 / 25 | 20 | 18 |
| Vertex normals | 13 | 17 | 8 |
| Sum or dot of face areas | 4 | 10 | 4 |

The scan's pass count grows with the vertex count, so the adjacency counts
are 11 / 13 at 32 K faces.

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
- The flat baseline reuses this crate's kernels downstream of the edge
  records. It is not an independent GPU mesh library, and a different flat
  layout (for example one that also stores face vertices) would be faster
  still.
