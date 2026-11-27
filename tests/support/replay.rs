//! Replays GPU edge flips on CPU triangles, so tests can check edited meshes
//! without using the GPU mesh storage as the reference.
use mopmesh::Mesh;

/// Returns each edge's mesh-wide endpoints in stored order, indexed by edge ID.
pub fn edge_endpoints(mesh: &Mesh) -> Vec<[u32; 2]> {
    let mut ends = vec![[0; 2]; mesh.edge_count()];
    for p in mesh.patches() {
        for (&e, ev) in p.edge_ids().iter().zip(p.edge_vertices()) {
            ends[e as usize] = ev.map(|v| p.vertex_ids()[v as usize]);
        }
    }
    ends
}

/// Applies one flip of edge `[a, b]` to triangles indexed by face ID, as the
/// library documents it: the face running a -> b -> c becomes a -> d -> c,
/// and the face running b -> a -> d becomes d -> b -> c. Returns
/// `[a, b, c, d]`. Panics unless one triangle runs each direction along a-b.
pub fn host_flip(triangles: &mut [[u32; 3]], [a, b]: [u32; 2]) -> [u32; 4] {
    // Match both directed sides of the edge to preserve triangle orientation.
    let runs = |t: &[u32; 3], u, v| (0..3).any(|k| t[k] == u && t[(k + 1) % 3] == v);
    let first = triangles.iter().position(|t| runs(t, a, b)).unwrap();
    let second = triangles.iter().position(|t| runs(t, b, a)).unwrap();
    let opposite = |t: &[u32; 3]| *t.iter().find(|&&x| x != a && x != b).unwrap();
    let c = opposite(&triangles[first]);
    let d = opposite(&triangles[second]);
    triangles[first] = [a, d, c];
    triangles[second] = [d, b, c];
    [a, b, c, d]
}

/// Replays a flip of edge `edge` on `triangles` and updates `endpoints` to
/// the flipped edge's new endpoints, in ascending order. Returns `[a, b, c, d]`.
pub fn replay(triangles: &mut [[u32; 3]], endpoints: &mut [[u32; 2]], edge: usize) -> [u32; 4] {
    let quad = host_flip(triangles, endpoints[edge]);
    endpoints[edge] = [quad[2].min(quad[3]), quad[2].max(quad[3])];
    quad
}
