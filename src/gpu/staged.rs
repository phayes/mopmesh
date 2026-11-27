//! Tests that the patch queries give exactly the results of the global
//! kernels they replace, and the same results for static and editable
//! uploads, including editable meshes with spare storage slots.
use super::{GpuEditableMesh, PatchBuffers, buffers};
use crate::Mesh;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::adjacency::AdjacencyPlan;
    use crate::gpu::baseline::FlatQueries;
    use crate::gpu::bench_meshes;
    use crate::gpu::editable::{editable_contents, inspect};
    use crate::gpu::pipeline::workgroups;
    use crate::gpu::test_support::{self, fixtures};
    use crate::gpu::{
        AdjacencyQuery, BoundaryQuery, EdgeQuery, EditCapacity, FaceVertexQuery, GpuMesh,
        INVALID_ID,
    };

    struct Queries {
        edges: EdgeQuery,
        face_vertices: FaceVertexQuery,
        adjacency: AdjacencyQuery,
        boundary: BoundaryQuery,
        flat: FlatQueries,
    }

    /// Every patch query's results for one upload, read back as words.
    /// Vertex neighbor rows are sorted, since their order is unspecified.
    #[derive(Debug, PartialEq)]
    struct Results {
        edges: Vec<u32>,
        face_vertices: Vec<u32>,
        adjacency_edges: Vec<u32>,
        face_neighbors: Vec<u32>,
        diamonds: Vec<u32>,
        vertex_neighbors: Vec<u32>,
        boundary_edges: Vec<u32>,
        boundary_vertices: Vec<u32>,
    }

    /// Submits `encoder`, then reads back each of `buffers`.
    fn read_all(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
        buffers: &[&wgpu::Buffer],
    ) -> Vec<Vec<u32>> {
        queue.submit([encoder.finish()]);
        buffers
            .iter()
            .map(|b| {
                let encoder = device.create_command_encoder(&Default::default());
                test_support::read_words(device, queue, encoder, b)
            })
            .collect()
    }

    /// Sorts each row of a vertex neighbor list of `vertex_count` rows.
    fn sorted_rows(mut words: Vec<u32>, vertex_count: u32) -> Vec<u32> {
        let (offsets, entries) = words.split_at_mut(vertex_count as usize + 1);
        for row in offsets.windows(2) {
            let range = 2 * row[0] as usize..2 * row[1] as usize;
            let mut pairs: Vec<[u32; 2]> = entries[range.clone()]
                .chunks(2)
                .map(|p| [p[0], p[1]])
                .collect();
            pairs.sort();
            entries[range].copy_from_slice(pairs.as_flattened());
        }
        words
    }

    /// Records every patch query on `buffers` and reads back the results.
    fn run(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &Queries,
        buffers: PatchBuffers,
    ) -> Results {
        let mut encoder = device.create_command_encoder(&Default::default());
        let e = &mut encoder;
        let edges = q.edges.encode_patches(device, e, buffers).unwrap();
        let faces = q.face_vertices.encode_patches(device, e, buffers).unwrap();
        let adjacency = q.adjacency.encode_patches(device, e, buffers).unwrap();
        let boundary = q.boundary.encode_patches(device, e, buffers).unwrap();
        let [
            edges,
            face_vertices,
            adjacency_edges,
            face_neighbors,
            diamonds,
            vertex_neighbors,
            boundary_edges,
            boundary_vertices,
        ] = read_all(
            device,
            queue,
            encoder,
            &[
                edges.buffer(),
                faces.buffer(),
                adjacency.edge_connectivity().buffer(),
                adjacency.face_neighbors(),
                adjacency.edge_diamonds(),
                adjacency.vertex_neighbors(),
                boundary.edges.buffer(),
                boundary.vertices.buffer(),
            ],
        )
        .try_into()
        .unwrap();
        Results {
            edges,
            face_vertices,
            adjacency_edges,
            face_neighbors,
            diamonds,
            vertex_neighbors: sorted_rows(vertex_neighbors, buffers.vertex_capacity),
            boundary_edges,
            boundary_vertices,
        }
    }

    /// Returns the face neighbors and diamonds that the global kernel
    /// derives from a static upload's edge records and face vertices.
    fn global_face_neighbors(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &Queries,
        gpu: &GpuMesh,
    ) -> (Vec<u32>, Vec<u32>) {
        let limits = device.limits();
        let buffers = buffers(gpu);
        let plan = AdjacencyPlan::new(
            &limits,
            buffers.vertex_capacity,
            buffers.face_capacity,
            buffers.edge_capacity,
        )
        .unwrap();
        let groups = workgroups(&limits, buffers.edge_capacity, "test").unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let edges = q.edges.encode(device, &mut encoder, gpu).unwrap();
        let faces = q.face_vertices.encode(device, &mut encoder, gpu).unwrap();
        let adjacency = AdjacencyQuery::outputs(device, &plan, edges);
        q.flat
            .face_neighbors(device, &mut encoder, &adjacency, &faces, groups);
        let [ff, diamonds] = read_all(
            device,
            queue,
            encoder,
            &[adjacency.face_neighbors(), adjacency.edge_diamonds()],
        )
        .try_into()
        .unwrap();
        (ff, diamonds)
    }

    /// Returns the boundary edge and vertex flags of `vertex_count` vertices,
    /// derived on the CPU from edge record words.
    fn cpu_boundary(edges: &[u32], edge_count: u32, vertex_count: u32) -> (Vec<u32>, Vec<u32>) {
        let mut edge_flags = vec![0; edge_count as usize];
        let mut vertex_flags = vec![0; vertex_count as usize];
        for (flag, record) in edge_flags.iter_mut().zip(edges.chunks(4)) {
            let faces = record[2..].iter().filter(|&&f| f != INVALID_ID).count();
            if faces == 1 {
                *flag = 1;
                vertex_flags[record[0] as usize] = 1;
                vertex_flags[record[1] as usize] = 1;
            }
        }
        (edge_flags, vertex_flags)
    }

    /// Checks a static upload of `mesh` against the global kernels word for
    /// word, then checks that editable uploads both without and with 50%
    /// extra capacity give the same results as the static upload.
    fn check(device: &wgpu::Device, queue: &wgpu::Queue, q: &Queries, mesh: &Mesh, context: &str) {
        // Staged adjacency and boundary flags match the kernels they replace.
        let gpu = GpuMesh::upload(device, mesh).unwrap();
        let expected = run(device, queue, q, buffers(&gpu));
        let (edge_count, vertex_count) = (gpu.edge_count(), gpu.vertex_count());
        if edge_count != 0 {
            assert_eq!(
                expected.adjacency_edges, expected.edges,
                "adjacency edges, {context}"
            );
            let (ff, diamonds) = global_face_neighbors(device, queue, q, &gpu);
            assert_eq!(expected.face_neighbors, ff, "face neighbors, {context}");
            assert_eq!(expected.diamonds, diamonds, "diamonds, {context}");
            let (edge_flags, vertex_flags) =
                cpu_boundary(&expected.edges, edge_count, vertex_count);
            assert_eq!(
                expected.boundary_edges, edge_flags,
                "boundary edges, {context}"
            );
            assert_eq!(
                expected.boundary_vertices, vertex_flags,
                "boundary vertices, {context}"
            );
        }

        // Editable uploads give the same words as the static upload: exactly
        // with no spare IDs, and for the used IDs with spare ones, whose
        // other IDs read as unused.
        // Normalize the static results the same way, which drops the
        // placeholders of empty results.
        let counts = [vertex_count, edge_count, gpu.face_count()];
        let expected = dense(expected, counts, counts, |i| i);
        for (slots, ids) in [(1.0, 1.0), (1.5, 2.0)] {
            let capacity = EditCapacity::new(slots).unwrap().with_ids(ids).unwrap();
            let editable = GpuEditableMesh::upload(device, mesh, capacity).unwrap();
            let actual = run(device, queue, q, buffers(&editable));
            let capacities = [
                editable.vertex_capacity(),
                editable.edge_capacity(),
                editable.face_capacity(),
            ];
            let actual = dense(actual, counts, capacities, |i| i);
            assert_eq!(actual, expected, "{context}, slots {slots}, IDs {ids}");
        }

        // Used IDs spread out, with unused ones between them, give the same
        // results once renumbered.
        let limits = device.limits();
        let mut words = editable_contents(mesh, EditCapacity::default(), &limits).unwrap();
        inspect::spread_ids(&mut words);
        if !mesh.patches().is_empty() {
            inspect::check_invariants(&words.arena, &words.masks, &words.maps);
        }
        let editable = GpuEditableMesh::from_words(device, mesh.id(), &words);
        let actual = run(device, queue, q, buffers(&editable));
        let capacities = inspect::id_capacities(&words.arena);
        let actual = dense(actual, counts, capacities, inspect::spread);
        assert_eq!(actual, expected, "{context}, spread IDs");
    }

    /// Renumbers the results of an editable upload whose used IDs are
    /// `id(i)` for each `i` below `counts` (vertex, edge and face) back to
    /// `i`, after checking that every other ID below `capacities` reads as
    /// unused: `INVALID_ID` rows, zero flags and empty neighbor rows.
    fn dense(
        r: Results,
        counts: [u32; 3],
        capacities: [u32; 3],
        id: impl Fn(u32) -> u32,
    ) -> Results {
        const V: usize = 0;
        const E: usize = 1;
        const F: usize = 2;
        let mut back = capacities.map(|c| vec![INVALID_ID; c as usize]);
        for k in 0..3 {
            for i in 0..counts[k] {
                back[k][id(i) as usize] = i;
            }
        }
        let used = |k: usize, new: usize| back[k][new] != INVALID_ID;
        let old = |k: usize, w: u32| {
            if w == INVALID_ID {
                w
            } else {
                back[k][w as usize]
            }
        };

        // Fixed-width rows, with each column's values renumbered by its kind.
        let rows = |words: &[u32], row: usize, columns: &[usize], dead: u32, name: &str| {
            let mut out = vec![0; counts[row] as usize * columns.len()];
            for (new, chunk) in words.chunks(columns.len()).enumerate() {
                if new >= capacities[row] as usize {
                    break;
                }
                if !used(row, new) {
                    assert!(chunk.iter().all(|&w| w == dead), "{name}: unused row {new}");
                    continue;
                }
                let at = back[row][new] as usize * columns.len();
                for (c, &w) in chunk.iter().enumerate() {
                    out[at + c] = old(columns[c], w);
                }
            }
            out
        };
        let flags = |words: &[u32], row: usize, name: &str| {
            let mut out = vec![0; counts[row] as usize];
            for (new, &w) in words.iter().enumerate().take(capacities[row] as usize) {
                if used(row, new) {
                    out[back[row][new] as usize] = w;
                } else {
                    assert_eq!(w, 0, "{name}: unused flag {new}");
                }
            }
            out
        };

        // Vertex neighbor rows, rebuilt in dense order.
        let v = capacities[V] as usize;
        let (offsets, pairs) = r.vertex_neighbors.split_at(v + 1);
        let mut neighbors = vec![0u32];
        let mut entries: Vec<u32> = Vec::new();
        for i in 0..counts[V] {
            let new = id(i) as usize;
            let row = 2 * offsets[new] as usize..2 * offsets[new + 1] as usize;
            let mut row: Vec<[u32; 2]> = pairs[row]
                .chunks(2)
                .map(|p| [old(E, p[0]), old(V, p[1])])
                .collect();
            row.sort();
            entries.extend(row.as_flattened());
            neighbors.push((entries.len() / 2) as u32);
        }
        for new in (0..v).filter(|&n| !used(V, n)) {
            assert_eq!(
                offsets[new],
                offsets[new + 1],
                "unused vertex {new} has neighbors"
            );
        }
        assert_eq!(
            offsets[v] as usize,
            entries.len() / 2,
            "last offset counts the pairs"
        );
        neighbors.extend(entries);

        Results {
            edges: rows(&r.edges, E, &[V, V, F, F], INVALID_ID, "edges"),
            face_vertices: rows(&r.face_vertices, F, &[V, V, V], INVALID_ID, "faces"),
            adjacency_edges: rows(&r.adjacency_edges, E, &[V, V, F, F], INVALID_ID, "records"),
            face_neighbors: rows(&r.face_neighbors, F, &[F, F, F], INVALID_ID, "FF"),
            diamonds: rows(&r.diamonds, E, &[V, V, V, V], INVALID_ID, "diamonds"),
            vertex_neighbors: neighbors,
            boundary_edges: flags(&r.boundary_edges, E, "boundary edges"),
            boundary_vertices: flags(&r.boundary_vertices, V, "boundary vertices"),
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn patch_queries_match_global_kernels_and_editable_meshes() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let q = Queries {
            edges: EdgeQuery::new(&device),
            face_vertices: FaceVertexQuery::new(&device),
            adjacency: AdjacencyQuery::new(&device),
            boundary: BoundaryQuery::new(&device),
            flat: FlatQueries::new(&device),
        };
        let limits = device.limits();

        // Cover every valid test mesh at each supported test patch size.
        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                let context = format!("{} at patch size {patch_size}", fixture.name);
                check(&device, &queue, &q, &mesh, &context);
            }
        }

        // Test generated meshes before and after shuffling their input order.
        // The large sphere forces a patch near a pole to process its edges
        // in multiple batches because they do not all fit in shared memory.
        let mut chunked = false;
        for (faces, patch_size) in [(2000, 32), (2000, 512), (131072, 32)] {
            for base in bench_meshes::generated(faces) {
                if faces > 2000 && !base.name.starts_with("sphere") {
                    continue;
                }
                let shuffled = bench_meshes::shuffled(&base);
                for m in [base, shuffled] {
                    let mesh = MeshBuilder::new()
                        .positions(&m.positions)
                        .triangles(&m.triangles)
                        .patch_size(patch_size)
                        .build()
                        .unwrap();
                    chunked |= mesh
                        .patches()
                        .iter()
                        .any(|p| p.edge_ids().len() > test_support::STAGED_EDGES);
                    let context = format!("{} at patch size {patch_size}", m.name);
                    check(&device, &queue, &q, &mesh, &context);
                }
            }
        }
        assert!(chunked, "no patch exercised chunked staging");
    }
}
