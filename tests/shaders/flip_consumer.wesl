import mopmesh::topology::flip::{FaceEdges, flip_quad, flipped_faces, FLIP_DONE};

// This downstream shader keeps its own connectivity and bindings, and uses
// the library only to work out a flip.
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main() {
    // Vertices 0..4 of a unit square; edge 0 is the diagonal 0-2.
    // Edges: 0 = (0, 2), 1 = (0, 1), 2 = (1, 2), 3 = (2, 3), 4 = (0, 3).
    let ev = array(vec2(0u, 2u), vec2(0u, 1u), vec2(1u, 2u), vec2(2u, 3u), vec2(0u, 3u));
    // Faces 0 -> 1 -> 2 and 0 -> 2 -> 3.
    let first = FaceEdges(vec3(1u, 2u, 0u), vec3(0u, 0u, 1u));
    let second = FaceEdges(vec3(0u, 3u, 4u), vec3(0u, 0u, 1u));
    let quad = flip_quad(
        0u,
        first, array(ev[1], ev[2], ev[0]),
        second, array(ev[0], ev[3], ev[4]),
    );
    output[0] = quad.status;
    output[1] = select(0u, 1u, quad.status == FLIP_DONE);
    let faces = flipped_faces(quad, first, second, 5u);
    output[2] = faces[0].edges.x;
    output[3] = faces[1].reversed.z;
}
