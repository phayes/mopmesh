//! Unpartitioned twins of the queries that read the patch arena, for the
//! benchmark baseline. A [`FlatMesh`] holds the same connectivity as the
//! arena (edge endpoints and face edges) in global arrays, without patches
//! or ribbons. Queries that only read global results (VV, FF, diamonds,
//! centroids, smoothing) run the same kernels on either path.
use super::adjacency::AdjacencyPlan;
use super::pipeline::{self, workgroups};
use super::{
    AdjacencyQuery, EdgeConnectivity, FaceCentroidQuery, FaceVertices, GpuError, GpuFaceAttribute,
    GpuVertexAttribute, MeshAdjacency,
};
use crate::{Mesh, MeshId};
use wgpu::util::DeviceExt;

/// Global connectivity arrays for one mesh. `fe` holds three
/// (global edge ID, reversed 0/1) pairs per face; `ev` holds each edge's
/// global vertex IDs in ascending order.
pub(super) struct FlatMesh {
    fe: wgpu::Buffer,
    ev: wgpu::Buffer,
    vertex_count: u32,
    face_count: u32,
    edge_count: u32,
    mesh: MeshId,
}

impl FlatMesh {
    /// Uploads `mesh`'s connectivity as global arrays. Attributes uploaded
    /// for `mesh` work with both this and its [`GpuMesh`](super::GpuMesh).
    pub(super) fn upload(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let (fe, ev) = flat_contents(mesh);
        let upload = |label, words: &[u32]| {
            // Bindings cannot be empty; a placeholder covers empty meshes.
            let words = if words.is_empty() { &[0, 0][..] } else { words };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        Self {
            fe: upload("baseline FE", &fe),
            ev: upload("baseline EV", &ev),
            vertex_count: mesh.positions().len() as u32,
            face_count: mesh.face_count() as u32,
            edge_count: mesh.edge_count() as u32,
            mesh: mesh.id(),
        }
    }

    /// Bytes of connectivity stored on the GPU.
    pub(super) fn bytes(&self) -> u64 {
        self.fe.size() + self.ev.size()
    }
}

/// Returns the global FE and EV words for `mesh`, taken from the patch
/// that owns each face and edge.
fn flat_contents(mesh: &Mesh) -> (Vec<u32>, Vec<u32>) {
    let mut fe = vec![0; 6 * mesh.face_count()];
    let mut ev = vec![0; 2 * mesh.edge_count()];
    for p in mesh.patches() {
        let global_vertex = |local: u32| p.vertex_ids()[local as usize];
        for (local, owner) in p.edge_owners().iter().enumerate() {
            if owner.patch == p.id() {
                let edge = p.edge_ids()[local] as usize;
                let [a, b] = p.edge_vertices()[local];
                ev[2 * edge..2 * edge + 2].copy_from_slice(&[global_vertex(a), global_vertex(b)]);
            }
        }
        for (local, owner) in p.face_owners().iter().enumerate() {
            if owner.patch == p.id() {
                let face = p.face_ids()[local] as usize;
                for (i, edge) in p.face_edges()[local].iter().enumerate() {
                    let global = p.edge_ids()[edge.edge as usize];
                    fe[6 * face + 2 * i..6 * face + 2 * i + 2]
                        .copy_from_slice(&[global, u32::from(edge.reversed)]);
                }
            }
        }
    }
    (fe, ev)
}

/// Pipelines for the unpartitioned queries. Cache one per device.
pub(super) struct FlatQueries {
    face_vertices: wgpu::ComputePipeline,
    edges: [wgpu::ComputePipeline; 3],
    edge_layout: wgpu::BindGroupLayout,
    centroids: FaceCentroidQuery,
    adjacency: AdjacencyQuery,
}

impl FlatQueries {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let module = pipeline::shader_module(
            device,
            "baseline face vertices",
            include_str!(concat!(env!("OUT_DIR"), "/baseline_face_vertices.wgsl")),
        );
        let face_vertices =
            pipeline::compute_pipeline(device, "baseline face vertices", &module, "main", None);
        let module = pipeline::shader_module(
            device,
            "baseline edges",
            include_str!(concat!(env!("OUT_DIR"), "/baseline_edges.wgsl")),
        );
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let (edge_layout, layout) =
            pipeline::buffer_layout(device, "baseline edges", &[read, read, write]);
        let edges = ["initialize", "scatter", "compact"].map(|entry| {
            pipeline::compute_pipeline(device, "baseline edges", &module, entry, Some(&layout))
        });
        Self {
            face_vertices,
            edges,
            edge_layout,
            centroids: FaceCentroidQuery::new(device),
            adjacency: AdjacencyQuery::new(device),
        }
    }

    /// Records the face-vertex query: three global vertex IDs per face.
    pub(super) fn face_vertices(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &FlatMesh,
    ) -> Result<FaceVertices, GpuError> {
        const RESOURCE: &str = "baseline face vertices";
        let groups = workgroups(&device.limits(), mesh.face_count, RESOURCE)?;
        let bytes = (12 * u64::from(mesh.face_count)).max(4);
        let result = FaceVertices {
            buffer: pipeline::output_buffer(device, RESOURCE, bytes),
            face_count: mesh.face_count,
            mesh: mesh.mesh,
        };
        if mesh.face_count != 0 {
            let bindings = pipeline::bind_buffers(
                device,
                RESOURCE,
                &self.face_vertices.get_bind_group_layout(0),
                &[&mesh.fe, &mesh.ev, &result.buffer],
            );
            pipeline::dispatch(encoder, RESOURCE, &self.face_vertices, &bindings, groups);
        }
        Ok(result)
    }

    /// Records the edge query: one `EdgeRecord` per edge, as from
    /// [`EdgeQuery`](super::EdgeQuery).
    pub(super) fn edges(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &FlatMesh,
    ) -> Result<EdgeConnectivity, GpuError> {
        const RESOURCE: &str = "baseline edges";
        let limits = device.limits();
        let edge_groups = workgroups(&limits, mesh.edge_count, RESOURCE)?;
        let face_groups = workgroups(&limits, mesh.face_count, RESOURCE)?;
        let bytes = (16 * u64::from(mesh.edge_count)).max(16);
        let result = EdgeConnectivity {
            buffer: pipeline::output_buffer(device, RESOURCE, bytes),
            edge_count: mesh.edge_count,
            mesh: mesh.mesh,
        };
        if mesh.edge_count != 0 {
            let bindings = pipeline::bind_buffers(
                device,
                RESOURCE,
                &self.edge_layout,
                &[&mesh.fe, &mesh.ev, &result.buffer],
            );
            let [initialize, scatter, compact] = &self.edges;
            for (pipeline, groups) in [
                (initialize, edge_groups),
                (scatter, face_groups),
                (compact, edge_groups),
            ] {
                pipeline::dispatch(encoder, RESOURCE, pipeline, &bindings, groups);
            }
        }
        Ok(result)
    }

    /// Records face centroids, as from [`FaceCentroidQuery`].
    pub(super) fn centroids(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &FlatMesh,
        positions: &GpuVertexAttribute<[f32; 4]>,
    ) -> Result<GpuFaceAttribute<[f32; 4]>, GpuError> {
        let faces = self.face_vertices(device, encoder, mesh)?;
        self.centroids
            .encode_from(device, encoder, &faces, positions)
    }

    /// Records VV, FF, diamonds and edges, as from [`AdjacencyQuery`].
    pub(super) fn adjacency(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &FlatMesh,
    ) -> Result<MeshAdjacency, GpuError> {
        let plan = AdjacencyPlan::new(
            &device.limits(),
            mesh.vertex_count,
            mesh.face_count,
            mesh.edge_count,
        )?;
        let edges = self.edges(device, encoder, mesh)?;
        let faces = self.face_vertices(device, encoder, mesh)?;
        Ok(self.adjacency.record(device, encoder, &plan, edges, &faces))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuMesh;
    use crate::gpu::test_support::{self, fixtures};

    /// Reads `buffer` back after submitting `encoder`.
    fn read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
    ) -> Vec<u32> {
        test_support::read_words(device, queue, encoder, buffer)
    }

    /// Sorts each edge record's two faces, so records compare regardless of
    /// face order.
    fn sorted_edge_faces(words: &[u32]) -> Vec<u32> {
        let mut words = words.to_vec();
        for record in words.chunks_mut(4) {
            record[2..].sort();
        }
        words
    }

    /// Sorts each VV row, and pairs each diamond's opposite vertex with its
    /// face so diamonds compare regardless of face order.
    fn canonical_adjacency(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adjacency: &MeshAdjacency,
    ) -> (Vec<u32>, Vec<u32>, Vec<[u32; 4]>) {
        let read = |buffer| {
            read(
                device,
                queue,
                device.create_command_encoder(&Default::default()),
                buffer,
            )
        };
        let v = adjacency.vertex_count() as usize;
        let mut vv = read(adjacency.vertex_neighbors());
        let (offsets, entries) = vv.split_at_mut(v + 1);
        for row in offsets.windows(2) {
            let range = 2 * row[0] as usize..2 * row[1] as usize;
            let mut pairs: Vec<[u32; 2]> = entries[range.clone()]
                .chunks(2)
                .map(|p| [p[0], p[1]])
                .collect();
            pairs.sort();
            entries[range].copy_from_slice(pairs.as_flattened());
        }
        let ff = read(adjacency.face_neighbors());
        let edges = read(adjacency.edge_connectivity().buffer());
        let diamonds = read(adjacency.edge_diamonds());
        let diamonds = edges
            .chunks(4)
            .zip(diamonds.chunks(4))
            .map(|(edge, diamond)| {
                let mut sides = [[edge[2], diamond[2]], [edge[3], diamond[3]]];
                sides.sort();
                [sides[0][0], sides[0][1], sides[1][0], sides[1][1]]
            })
            .collect();
        (vv, ff, diamonds)
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn flat_queries_match_the_patched_queries() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let flat_queries = FlatQueries::new(&device);
        let face_vertices = super::super::FaceVertexQuery::new(&device);
        let edges = super::super::EdgeQuery::new(&device);
        let centroids = FaceCentroidQuery::new(&device);
        let adjacency = AdjacencyQuery::new(&device);
        let limits = device.limits();
        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                let context = format!("{} at patch size {patch_size}", fixture.name);
                let (mesh, _) = test_support::reference(&fixture, patch_size, &limits).unwrap();
                let gpu = GpuMesh::upload(&device, &mesh).unwrap();
                let flat = FlatMesh::upload(&device, &mesh);
                let positions = gpu
                    .upload_attribute(&device, &mesh.position_attribute())
                    .unwrap();

                // Records `encode` into a fresh encoder and reads `buffer` of its result.
                macro_rules! run {
                    ($encode:expr, $buffer:expr) => {{
                        let mut encoder = device.create_command_encoder(&Default::default());
                        let result = $encode(&mut encoder).unwrap();
                        read(&device, &queue, encoder, $buffer(&result))
                    }};
                }
                assert_eq!(
                    run!(
                        |e| flat_queries.face_vertices(&device, e, &flat),
                        FaceVertices::buffer
                    ),
                    run!(
                        |e| face_vertices.encode(&device, e, &gpu),
                        FaceVertices::buffer
                    ),
                    "face vertices, {context}"
                );
                assert_eq!(
                    sorted_edge_faces(&run!(
                        |e| flat_queries.edges(&device, e, &flat),
                        EdgeConnectivity::buffer
                    )),
                    sorted_edge_faces(&run!(
                        |e| edges.encode(&device, e, &gpu),
                        EdgeConnectivity::buffer
                    )),
                    "edges, {context}"
                );
                assert_eq!(
                    run!(
                        |e| flat_queries.centroids(&device, e, &flat, &positions),
                        GpuFaceAttribute::buffer
                    ),
                    run!(
                        |e| centroids.encode(&device, e, &gpu, &positions),
                        GpuFaceAttribute::buffer
                    ),
                    "centroids, {context}"
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                let flat_adjacency = flat_queries
                    .adjacency(&device, &mut encoder, &flat)
                    .unwrap();
                let patched_adjacency = adjacency.encode(&device, &mut encoder, &gpu).unwrap();
                queue.submit([encoder.finish()]);
                assert_eq!(
                    canonical_adjacency(&device, &queue, &flat_adjacency),
                    canonical_adjacency(&device, &queue, &patched_adjacency),
                    "adjacency, {context}"
                );
            }
        }
    }
}
