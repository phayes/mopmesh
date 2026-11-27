# Approximate benchmark against Alum

A rough wall-clock comparison of MOPMesh's GPU queries against the same data
built by [Alum](https://crates.io/crates/alum) on one CPU thread. This is a
sanity check, not the full baseline in ROADMAP milestone 2: no real-world
meshes, no GPU timestamp queries, no allocation counts. That baseline is in
[the baseline report](baseline_benchmark.md).

```sh
cargo run --release --example bench_alum            # grids 128, 512, 1024
cargo run --release --example bench_alum 256 2048   # custom grid sizes
```

## Method

- Mesh: an n x n vertex grid with 2 (n - 1)^2 triangles, default patch size.
- Each figure is the median of 7 runs after 2 warm-up runs.
- GPU time is wall clock for encoding, submitting and waiting for one query,
  without readback. It includes the buffers each `encode` call allocates.
- Alum time is the equivalent traversal written into plain `Vec`s:
  `calc_face_centroid` per face, `vf_ccw_iter` per vertex as CSR, and
  `vv_ccw_iter` CSR plus `ff_ccw_iter` and edge endpoints for the adjacency row.
- "build" is host construction for both libraries: `MeshBuilder::build`
  against `add_vertex`/`add_face`.

## Results

Apple M1 Max, Metal, release build, 2026-11-26. The ratio is Alum time divided
by MOPMesh time; above 1 means MOPMesh is faster.

| Grid | Faces | Workload | MOPMesh | Alum | Ratio |
|---|---:|---|---:|---:|---:|
| 128² | 32 K | build (CPU) | 5.6 ms | 2.5 ms | 0.45x |
| | | upload mesh + positions | 1.8 ms | – | – |
| | | face centroids | 0.38 ms | 0.90 ms | 2.4x |
| | | vertex-face CSR | 0.78 ms | 0.60 ms | 0.8x |
| | | VV + FF + edges | 1.37 ms | 1.13 ms | 0.8x |
| 512² | 522 K | build (CPU) | 94 ms | 40.2 ms | 0.43x |
| | | upload mesh + positions | 22.9 ms | – | – |
| | | face centroids | 1.30 ms | 14.7 ms | 11.3x |
| | | vertex-face CSR | 1.98 ms | 10.7 ms | 5.4x |
| | | VV + FF + edges | 4.49 ms | 21.4 ms | 4.8x |
| 1024² | 2.1 M | build (CPU) | 387 ms | 159 ms | 0.41x |
| | | upload mesh + positions | 94.2 ms | – | – |
| | | face centroids | 4.41 ms | 57.1 ms | 12.9x |
| | | vertex-face CSR | 5.85 ms | 40.3 ms | 6.9x |
| | | VV + FF + edges | 15.6 ms | 80.8 ms | 5.2x |

## GPU build

The GPU build (`GpuMeshBuilder`; see the [strategy](gpu_build_strategy.md))
against the CPU build plus upload, on the same grids. `bench_alum` also
prints it as "build (GPU, + input)". The stage breakdown and split comparison
below come from the internal benchmark:

```sh
MOPMESH_BENCH=1 cargo test --release --lib gpu_build_benchmark -- --ignored --nocapture
```

Apple M1 Max, Metal, patch size 128, median of 7 runs, 2026-11-26. This run
was on a machine about 25% slower than the table above (Alum's build took
205 ms at 2.1 M faces instead of 159 ms), so compare within this table only.

After the row-sort and arena-header speed-ups (same session, same machine
state). "GPU" includes uploading the raw positions and triangles. Alum's
build is from `bench_alum`, run in the same session.

| Grid | Faces | CPU build + upload | GPU, Morton split | GPU, input order | Alum build |
|---|---:|---:|---:|---:|---:|
| 128² | 32 K | 8.2 ms | 12.2 ms | 6.8 ms | 3.2 ms |
| 512² | 522 K | 139 ms | 37.4 ms | 72.5 ms | 51.9 ms |
| 1024² | 2.1 M | 570 ms | 154 ms | 265 ms | 205 ms |

Before the speed-ups the Morton column was 24.0, 54.5 and 182 ms.

GPU build stages at 2.1 M faces (each submitted and waited for on its own,
so the sum exceeds the pipelined total): elements 9.9 ms, connectivity
16.6 ms, Morton split 14.8 ms, member counts 16.3 ms, summary readback
0.3 ms, patch contents 63.1 ms, arena 20.7 ms. Before: patch contents 65.9 ms
(16.5 ms at 32 K faces, now 2.0 ms) and arena 59.9 ms. The speed-ups:
patch rows are sorted by one 64-wide workgroup per row instead of one thread,
and the arena header is written by a kernel instead of mapping the whole
arena at creation, which cost about 35 ms at this size.

Patch quality and query times on each split, at 2.1 M faces:

| Split | Faces held per face | Centroids | Vertex-face | VV + FF + edges |
|---|---:|---:|---:|---:|
| Breadth-first (CPU) | 1.636 | 3.24 ms | 3.86 ms | 16.5 ms |
| Morton (GPU) | 1.664 | 3.42 ms | 4.00 ms | 16.5 ms |
| Input order (GPU) | 3.076 | 4.32 ms | 3.98 ms | 22.5 ms |

At 522 K faces Morton held 1.692 against 1.636. Its query times there were
within 5% in one run and up to 39% slower (vertex-face) in another, so
single-run differences at this size are mostly noise.

## Against Alum, end to end

Same session, 2.1 M faces. MOPMesh uses the GPU build with the Morton split;
Alum builds a half-edge mesh on one CPU thread (no patches or ribbons).

| Workload | MOPMesh | Alum | Alum / MOPMesh |
|---|---:|---:|---:|
| Build (MOPMesh: raw input upload + GPU build) | 154 ms | 205 ms | 1.3x |
| Face centroids | 4.3 ms | 73 ms | 17x |
| Vertex-face CSR | 4.0 ms | 53 ms | 13x |
| VV + FF + edges | 16.9 ms | 104 ms | 6.1x |
| Build + all three queries | 179 ms | 435 ms | 2.4x |

At 522 K faces: build 37.4 against 51.9 ms (1.4x), and build plus all three
queries 47 against 109 ms (2.3x). At 32 K faces Alum is ahead: its build
takes 3.2 ms against the GPU build's 12.2 ms (6.8 ms with input order).

The public `GpuMeshBuilder` in `bench_alum`, a later run the same day (same
machine state): 9.7 ms against Alum's 3.2 ms at 32 K faces, 39.6 against
51.6 ms (1.3x) at 522 K, and 153 against 210 ms (1.4x) at 2.1 M.

## Takeaways

- Warm GPU queries beat single-threaded Alum by about 5–13x on meshes with
  500 K faces or more. At 32 K faces, fixed submit and allocation costs make
  them roughly even.
- Host construction is about 2.4x slower than Alum's. It was about 16x slower
  (2724 ms at 2.1 M faces) until the builder switched from sorted maps and
  per-element lists to bucketed sorts and flat arrays. The remaining gap is
  mostly patch construction, which Alum does not do.
- Upload of a 2 M-face mesh costs about 20 warm adjacency queries, so the
  GPU path pays off only when several queries run on the same upload.
- The GPU build (Morton split) is about 3.7x faster than the CPU build plus
  upload at 522 K and 2.1 M faces, and 1.3–1.4x faster than Alum's
  single-threaded build while also producing patches and ribbons. At 32 K
  faces it is still slower than both (12 ms; the 8-pass Morton sort and
  fixed per-dispatch costs dominate). Patch contents is the largest stage
  at 2.1 M faces, with the time now outside the row sort.
- Morton patches hold 2–3% more faces than breadth-first ones, with query
  times within noise except face centroids (+19% at 2.1 M). Input-order
  patches hold nearly twice as many faces and slow the adjacency query by
  about 38%; they are only a stepping stone.
- These numbers flatter neither side precisely: Alum could be parallelised,
  and the GPU figures include per-call allocation (review finding P3) and
  repeated face-vertex passes (P2).
