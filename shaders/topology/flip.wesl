// Edge flips on triangle meshes, as binding-free helpers.
//
// Flipping the edge shared by two triangles replaces it with the other
// diagonal of the four-sided region they form. If the two triangles are
// a -> b -> c and b -> a -> d, the flip removes edge a-b and adds edge c-d,
// giving a -> d -> c and d -> b -> c. Every vertex keeps its position, and
// both new triangles keep the winding of the old ones.
//
// The caller loads each face's edges and each edge's endpoints from its own
// storage and passes the values in. Nothing here reads or writes buffers.

// Outcome codes for an edge flip. Callers may store these as status words.
public const FLIP_NOT_REQUESTED: u32 = 0u;
public const FLIP_DONE: u32 = 1u;
// The edge has only one face, so there is no second diagonal.
public const FLIP_BOUNDARY: u32 = 2u;
// Some face around one of the four vertices belongs to another patch.
public const FLIP_NOT_INTERIOR: u32 = 3u;
// Vertices c and d are already joined by an edge; flipping would duplicate it.
public const FLIP_CONNECTED: u32 = 4u;
// The two faces don't form a proper four-sided region: c equals d, or the
// faces run the shared edge in the same direction.
public const FLIP_DEGENERATE: u32 = 5u;
// Another flip claimed one of the same vertices first.
public const FLIP_LOST_CLAIM: u32 = 6u;
// The patch had no free slot for the new edge or faces.
public const FLIP_OUT_OF_CAPACITY: u32 = 7u;
// The requested edge ID is not an edge of the mesh.
public const FLIP_NOT_AN_EDGE: u32 = 8u;

// A face's three edges in winding order: edge IDs, and for each a flag that
// is 1 when the face runs the edge against its stored vertex order.
public struct FaceEdges {
    edges: vec3<u32>,
    reversed: vec3<u32>,
}

// The four-sided region around an edge, found by `flip_quad`.
// `first` runs a -> b -> c and `second` runs b -> a -> d. `first_slot` and
// `second_slot` are the positions (0, 1 or 2) of the shared edge in each
// face's FaceEdges. `status` is FLIP_DONE when the region is valid, or
// FLIP_DEGENERATE.
public struct FlipQuad {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    first_slot: u32,
    second_slot: u32,
    status: u32,
}

// Finds the region that flipping `edge` would change. `first` and `second`
// are the two faces on `edge`; `first_vertices` and `second_vertices` are the
// stored endpoints of each face's three edges, in FaceEdges order. Returns
// FLIP_DEGENERATE in `status` if either face doesn't contain `edge`, if both
// run it in the same direction, or if c equals d.
public fn flip_quad(
    edge: u32,
    first: FaceEdges,
    first_vertices: array<vec2<u32>, 3>,
    second: FaceEdges,
    second_vertices: array<vec2<u32>, 3>,
) -> FlipQuad {
    var quad = FlipQuad(0u, 0u, 0u, 0u, 3u, 3u, FLIP_DEGENERATE);

    // Find the shared edge in each face.
    for (var i = 0u; i < 3u; i++) {
        if (first.edges[i] == edge) { quad.first_slot = i; }
        if (second.edges[i] == edge) { quad.second_slot = i; }
    }
    if (quad.first_slot == 3u || quad.second_slot == 3u) { return quad; }
    let i = quad.first_slot;
    let j = quad.second_slot;
    if (first.reversed[i] == second.reversed[j]) { return quad; }

    // In `first` the shared edge runs a -> b and the edge after next starts at c.
    // In `second` it runs b -> a and the edge after next starts at d.
    let ab = first_vertices[i];
    let forward = first.reversed[i] == 0u;
    quad.a = select(ab.y, ab.x, forward);
    quad.b = select(ab.x, ab.y, forward);
    let ca = first_vertices[(i + 2u) % 3u];
    quad.c = select(ca.y, ca.x, first.reversed[(i + 2u) % 3u] == 0u);
    let db = second_vertices[(j + 2u) % 3u];
    quad.d = select(db.y, db.x, second.reversed[(j + 2u) % 3u] == 0u);
    if (quad.c != quad.d) { quad.status = FLIP_DONE; }
    return quad;
}

// The two faces that replace `first` and `second` after a flip, for a valid
// `quad`. `new_edge` is the ID of the new edge, whose stored vertex order
// must be (quad.c, quad.d). Returns a -> d -> c, then d -> b -> c.
public fn flipped_faces(
    quad: FlipQuad,
    first: FaceEdges,
    second: FaceEdges,
    new_edge: u32,
) -> array<FaceEdges, 2> {
    let i = quad.first_slot;
    let j = quad.second_slot;
    let bc = (i + 1u) % 3u;
    let ca = (i + 2u) % 3u;
    let ad = (j + 1u) % 3u;
    let db = (j + 2u) % 3u;
    // a -> d (from second), d -> c (new edge, against its stored order), c -> a (from first).
    let adc = FaceEdges(
        vec3(second.edges[ad], new_edge, first.edges[ca]),
        vec3(second.reversed[ad], 1u, first.reversed[ca]),
    );
    // d -> b (from second), b -> c (from first), c -> d (new edge, in its stored order).
    let dbc = FaceEdges(
        vec3(second.edges[db], first.edges[bc], new_edge),
        vec3(second.reversed[db], first.reversed[bc], 0u),
    );
    return array(adc, dbc);
}
