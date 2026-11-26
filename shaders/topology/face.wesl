import super::directed_edge::directed_start;

public fn face_contains_edge(edges: vec3<u32>, edge: u32) -> bool {
    return any(edges == vec3(edge));
}

// Slot i corresponds to the triangle edge (vertices[i], vertices[(i+1)%3]).
// Endpoint order is ignored. A missing edge returns 0xffffffff.
public fn face_edge_slot(vertices: vec3<u32>, edge: vec2<u32>) -> u32 {
    for (var i = 0u; i < 3u; i++) {
        let a = vertices[i];
        let b = vertices[(i + 1u) % 3u];
        if ((a == edge.x && b == edge.y) || (a == edge.y && b == edge.x)) { return i; }
    }
    return 0xffffffffu;
}

public fn opposite_vertex(vertices: vec3<u32>, edge: vec2<u32>) -> u32 {
    let slot = face_edge_slot(vertices, edge);
    if (slot == 0xffffffffu) { return 0xffffffffu; }
    return vertices[(slot + 2u) % 3u];
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
