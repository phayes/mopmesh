// Mark missing faces and vertices with 0xffffffff, matching Rust's mopmesh::gpu::INVALID_ID.
// These values are sentinels, not valid indices into mesh data.
public const INVALID_FACE: u32 = 0xffffffffu;
public const INVALID_VERTEX: u32 = 0xffffffffu;

// Storage-array stride: 16 bytes. Endpoint IDs are global vertex IDs; faces
// are global face IDs, with INVALID_FACE denoting the missing boundary face.
public struct EdgeRecord {
    vertices: vec2<u32>,
    faces: vec2<u32>,
}

public fn empty_edge_faces() -> vec2<u32> {
    return vec2(INVALID_FACE);
}

// Requires at most one existing face and a distinct, valid new face ID.
public fn append_edge_face(faces: vec2<u32>, face: u32) -> vec2<u32> {
    if (faces.x == INVALID_FACE) { return vec2(face, INVALID_FACE); }
    return vec2(faces.x, face);
}

public fn edge_face_count(faces: vec2<u32>) -> u32 {
    return select(0u, 1u, faces.x != INVALID_FACE) + select(0u, 1u, faces.y != INVALID_FACE);
}

public fn is_boundary_edge(faces: vec2<u32>) -> bool {
    return edge_face_count(faces) == 1u;
}

// Returns 0xffffffff for an absent opposite face or a non-incident input.
public fn other_edge_face(faces: vec2<u32>, face: u32) -> u32 {
    if (face == INVALID_FACE) { return INVALID_FACE; }
    if (faces.x == face) { return faces.y; }
    if (faces.y == face) { return faces.x; }
    return INVALID_FACE;
}

// Returns the other endpoint of the edge. If vertex is not one of the two
// endpoints in vertices, returns INVALID_VERTEX.
public fn other_edge_vertex(vertices: vec2<u32>, vertex: u32) -> u32 {
    if (vertices.x == vertex) { return vertices.y; }
    if (vertices.y == vertex) { return vertices.x; }
    return INVALID_VERTEX;
}

