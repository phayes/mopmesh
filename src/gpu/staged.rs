//! Tests that the patch queries give exactly the results of the global
//! kernels they replace, and the same results for static and editable
//! uploads, including editable meshes with spare storage slots.
use super::{GpuEditableMesh, PatchBuffers};
use crate::Mesh;

/// Borrows an editable upload's patch buffers, checking that it belongs to `mesh`.
fn editable_buffers<'a>(mesh: &Mesh, gpu: &'a GpuEditableMesh) -> PatchBuffers<'a> {
    assert_eq!(mesh.id(), gpu.mesh_id());
    PatchBuffers {
        arena: gpu.arena(),
        masks: gpu.masks(),
        patch_count: mesh.patches().len() as u32,
        vertex_count: gpu.vertex_count(),
        face_count: gpu.face_count(),
        edge_count: gpu.edge_count(),
        mesh: gpu.mesh_id(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::adjacency::AdjacencyPlan;
    use crate::gpu::baseline::FlatQueries;
    use crate::gpu::bench_meshes;
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
            vertex_neighbors: sorted_rows(vertex_neighbors, buffers.vertex_count),
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
        let buffers = gpu.buffers();
        let plan = AdjacencyPlan::new(
            &limits,
            buffers.vertex_count,
            buffers.face_count,
            buffers.edge_count,
        )
        .unwrap();
        let groups = workgroups(&limits, buffers.edge_count, "test").unwrap();
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
        let expected = run(device, queue, q, gpu.buffers());
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

        // Editable uploads give the same words as the static upload.
        for factor in [1.0, 1.5] {
            let capacity = EditCapacity::new(factor).unwrap();
            let editable = GpuEditableMesh::upload(device, mesh, capacity).unwrap();
            let actual = run(device, queue, q, editable_buffers(mesh, &editable));
            assert_eq!(actual, expected, "{context}, capacity {factor}");
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
                    chunked |= mesh.patches().iter().any(|p| p.edge_ids().len() > 1024);
                    let context = format!("{} at patch size {patch_size}", m.name);
                    check(&device, &queue, &q, &mesh, &context);
                }
            }
        }
        assert!(chunked, "no patch exercised chunked staging");
    }
}
