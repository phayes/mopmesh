import super::directed_edge::directed_start;
import super::edge::INVALID_VERTEX;

// Marks an edge slot that was not found. Its value, 0xffffffff, is not a valid slot index.
public const INVALID_SLOT: u32 = 0xffffffffu;

public fn face_contains_edge(edges: vec3<u32>, edge: u32) -> bool {
    return any(edges == vec3(edge));
}

// Returns the edge's position in the triangle: 0 for (x, y), 1 for (y, z),
// or 2 for (z, x), using the vertex IDs in vertices. Either endpoint order
// is accepted. Returns INVALID_SLOT if edge is not part of the triangle.
public fn face_edge_slot(vertices: vec3<u32>, edge: vec2<u32>) -> u32 {
    for (var i = 0u; i < 3u; i++) {
        let a = vertices[i];
        let b = vertices[(i + 1u) % 3u];
        if ((a == edge.x && b == edge.y) || (a == edge.y && b == edge.x)) { return i; }
    }
    return INVALID_SLOT;
}

// Returns the triangle vertex that is not an endpoint of edge.
// Returns INVALID_VERTEX if edge is not part of the triangle.
public fn opposite_vertex(vertices: vec3<u32>, edge: vec2<u32>) -> u32 {
    let slot = face_edge_slot(vertices, edge);
    if (slot == INVALID_SLOT) { return INVALID_VERTEX; }
    return vertices[(slot + 2u) % 3u];
}

// Returns the position (0, 1 or 2) of `vertex` in the triangle, or
// INVALID_SLOT if the triangle does not contain it.
public fn face_vertex_slot(vertices: vec3<u32>, vertex: u32) -> u32 {
    for (var i = 0u; i < 3u; i++) {
        if (vertices[i] == vertex) { return i; }
    }
    return INVALID_SLOT;
}

// Supply the EV records of a face's three edges, in FE order, together with
// their orientation flags (0 = forward, 1 = reversed). Returns local vertex
// IDs in face winding order. No bindings, pointer extensions, or barriers.
public fn face_vertices(
    edges: array<vec2<u32>, 3>,
    reversed: vec3<u32>,
) -> vec3<u32> {
    var result: vec3<u32>;
    for (var i = 0u; i < 3u; i++) {
        result[i] = directed_start(edges[i], reversed[i] != 0u);
    }
    return result;
}
