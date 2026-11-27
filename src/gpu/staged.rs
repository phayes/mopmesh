//! Tests that edge and face queries return the same results for static and
//! editable uploads, including editable meshes with spare storage slots.
use super::{GpuEditableMesh, PatchBuffers};
use crate::Mesh;

/// Borrows an editable upload's patch buffers, checking that it belongs to `mesh`.
fn editable_buffers<'a>(mesh: &Mesh, gpu: &'a GpuEditableMesh) -> PatchBuffers<'a> {
    assert_eq!(mesh.id(), gpu.mesh_id());
    PatchBuffers {
        arena: gpu.arena(),
        masks: gpu.masks(),
        patch_count: mesh.patches().len() as u32,
        face_count: gpu.face_count(),
        edge_count: gpu.edge_count(),
        mesh: gpu.mesh_id(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::bench_meshes;
    use crate::gpu::test_support::{self, fixtures};
    use crate::gpu::{EdgeQuery, EditCapacity, FaceVertexQuery, GpuMesh};

    struct Queries {
        edges: EdgeQuery,
        face_vertices: FaceVertexQuery,
    }

    /// Compares edge and face-vertex results with a static upload, word for word.
    /// Tests editable uploads both without extra capacity and with 50% extra capacity.
    fn check(device: &wgpu::Device, queue: &wgpu::Queue, q: &Queries, mesh: &Mesh, context: &str) {
        let read = |record: &dyn Fn(&mut wgpu::CommandEncoder) -> wgpu::Buffer| {
            let mut encoder = device.create_command_encoder(&Default::default());
            let buffer = record(&mut encoder);
            test_support::read_words(device, queue, encoder, &buffer)
        };
        let run = |buffers: PatchBuffers| {
            let edges = read(&|e| {
                let result = q.edges.encode_patches(device, e, buffers).unwrap();
                result.buffer().clone()
            });
            let faces = read(&|e| {
                let result = q.face_vertices.encode_patches(device, e, buffers).unwrap();
                result.buffer().clone()
            });
            (edges, faces)
        };
        let gpu = GpuMesh::upload(device, mesh).unwrap();
        let expected = run(gpu.buffers());
        for factor in [1.0, 1.5] {
            let capacity = EditCapacity::new(factor).unwrap();
            let editable = GpuEditableMesh::upload(device, mesh, capacity).unwrap();
            let actual = run(editable_buffers(mesh, &editable));
            assert_eq!(actual.0, expected.0, "edges, {context}, capacity {factor}");
            assert_eq!(
                actual.1, expected.1,
                "face vertices, {context}, capacity {factor}"
            );
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn editable_meshes_give_the_same_patch_query_results() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let q = Queries {
            edges: EdgeQuery::new(&device),
            face_vertices: FaceVertexQuery::new(&device),
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
