//! Test meshes shared by the GPU tests: plain positions and triangles, with no
//! library types, so both integration tests and in-crate tests can include
//! this file with `#[path]`.
#![allow(dead_code)]

/// A named test input. It may be invalid; the CPU builder decides.
pub struct Fixture {
    pub name: &'static str,
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// Patch sizes that every fixture is built with: one face per patch, a size that
/// leaves partial patches, and the default.
pub const PATCH_SIZES: [usize; 3] = [1, 3, 128];

/// An `n` x `n` vertex grid in the z = 0 plane with 2 (n - 1)^2 triangles,
/// wound counter-clockwise.
pub fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut positions = Vec::new();
    for y in 0..n {
        for x in 0..n {
            positions.push([x as f32, y as f32, 0.]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..n.saturating_sub(1) {
        for x in 0..n - 1 {
            let a = y * n + x;
            triangles.extend([[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]);
        }
    }
    (positions, triangles)
}

/// Valid meshes: grids with and without a hole, a closed tetrahedron, an empty
/// mesh, two disconnected triangles, a collinear triangle, a 129-face vertex fan
/// and a 64 x 64 grid.
pub fn valid() -> Vec<Fixture> {
    let mut fixtures = Vec::new();

    // 128 faces: two workgroups, boundaries, and ribbon duplication.
    let (positions, triangles) = grid(9);
    let hole = triangles
        .iter()
        .enumerate()
        .filter(|(i, _)| *i / 2 != 27)
        .map(|(_, &face)| face)
        .collect();
    fixtures.push(Fixture {
        name: "grid 9",
        positions: positions.clone(),
        triangles,
    });
    fixtures.push(Fixture {
        name: "grid 9 with a hole",
        positions,
        triangles: hole,
    });

    // A partial final workgroup, a closed surface, and an empty dispatch.
    fixtures.push(Fixture {
        name: "tetrahedron",
        positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        triangles: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
    });
    fixtures.push(Fixture {
        name: "empty",
        positions: vec![],
        triangles: vec![],
    });
    fixtures.push(Fixture {
        name: "two components",
        positions: vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [3., 0., 0.],
            [4., 0., 0.],
            [3., 1., 0.],
        ],
        triangles: vec![[0, 1, 2], [3, 4, 5]],
    });
    // Three distinct but collinear corners are not degenerate.
    fixtures.push(Fixture {
        name: "collinear triangle",
        positions: vec![[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
        triangles: vec![[0, 1, 2]],
    });

    // A row larger than a workgroup; boundary vertices have only two neighbours.
    let n = 129u32;
    let mut positions = vec![[0., 0., 0.]];
    for i in 0..n {
        let angle = std::f32::consts::TAU * i as f32 / n as f32;
        positions.push([angle.cos(), angle.sin(), 0.]);
    }
    fixtures.push(Fixture {
        name: "129-face fan",
        positions,
        triangles: (0..n).map(|i| [0, i + 1, (i + 1) % n + 1]).collect(),
    });

    // Exactly 64 vertices: one full workgroup per vertex, plus one more row.
    let (positions, triangles) = grid(8);
    fixtures.push(Fixture {
        name: "grid 8",
        positions,
        triangles,
    });

    // 4,225 vertices need three scan levels, with partial blocks at two levels.
    let (positions, triangles) = grid(65);
    fixtures.push(Fixture {
        name: "grid 65",
        positions,
        triangles,
    });
    fixtures
}

/// Inputs the CPU builder rejects, one per error kind it can report from a
/// mesh, plus cases where two errors compete and precedence decides.
pub fn invalid() -> Vec<Fixture> {
    let square = vec![
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., -1., 0.],
    ];
    let mut fixtures = Vec::new();
    let mut add = |name, positions: &[[f32; 3]], triangles: &[[u32; 3]]| {
        fixtures.push(Fixture {
            name,
            positions: positions.to_vec(),
            triangles: triangles.to_vec(),
        })
    };

    // Per-face and per-vertex input errors.
    add("missing vertex", &square, &[[0, 1, 9]]);
    add(
        "missing vertex in the last corner",
        &square,
        &[[0, 1, 2], [2, 1, 7]],
    );
    add("repeated corner", &square, &[[0, 0, 1]]);
    add("non-finite position", &[[f32::NAN, 0., 0.]], &[]);
    add("isolated vertex", &square, &[[0, 1, 2]]);
    add("faces but no vertices", &[], &[[0, 1, 2]]);
    add("vertices but no faces", &square, &[]);

    // Distinct vertices at equal coordinates; vertex 5 sits on vertex 1, and
    // vertex 6 on vertex 0 with a negative zero.
    let mut coincident = square.clone();
    coincident.extend([[1., 0., 0.], [-0., 0., 0.]]);
    add("coincident corners", &coincident, &[[0, 1, 2], [1, 5, 2]]);
    add("negative zero corner", &coincident, &[[0, 1, 6]]);
    // A missing vertex in an earlier face wins over a later degenerate face,
    // and within one face a missing vertex wins over equal coordinates.
    add(
        "missing vertex before degenerate face",
        &coincident,
        &[[0, 1, 2], [0, 8, 2], [1, 5, 2]],
    );
    add(
        "missing vertex in a degenerate face",
        &coincident,
        &[[1, 1, 9]],
    );
    // A non-finite position wins over any face error.
    let mut infinite = coincident.clone();
    infinite[4] = [0., f32::INFINITY, 0.];
    add("infinite position", &infinite, &[[0, 9, 2], [1, 5, 2]]);
    infinite[2] = [0., f32::NEG_INFINITY, 0.];
    add("two non-finite positions", &infinite, &[[0, 1, 3]]);

    // Duplicate faces, winding and non-manifold edges.
    add("reversed duplicate", &square, &[[0, 1, 2], [2, 1, 0]]);
    add("same-direction duplicate", &square, &[[0, 1, 2], [0, 1, 2]]);
    add("inconsistent winding", &square, &[[0, 1, 2], [0, 1, 4]]);
    // An edge error in an earlier face wins over a later duplicate, and the
    // reverse; the second case also has three faces on edge (0, 1).
    add(
        "winding before duplicate",
        &square,
        &[[0, 1, 2], [0, 1, 4], [0, 1, 2]],
    );
    add(
        "duplicate before non-manifold edge",
        &square,
        &[[0, 1, 2], [2, 1, 0], [0, 1, 4]],
    );
    let wedge = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
    ];
    add(
        "three faces on one edge",
        &wedge,
        &[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
    );
    // A later per-face error still wins over an earlier edge error.
    add(
        "winding before degenerate face",
        &coincident,
        &[[0, 1, 2], [0, 1, 4], [1, 5, 3]],
    );

    // Non-manifold vertices: two triangles, and two tetrahedra, sharing a vertex.
    add("bow tie", &square, &[[0, 1, 2], [0, 3, 4]]);
    // The lower vertex wins between an isolated vertex and a bow tie.
    let mut bow_tie = square.clone();
    bow_tie.push([2., 0., 0.]);
    add(
        "bow tie before isolated vertex",
        &bow_tie,
        &[[0, 1, 2], [0, 3, 4]],
    );
    add(
        "isolated vertex before bow tie",
        &bow_tie,
        &[[3, 1, 2], [3, 4, 5]],
    );
    // Two open fans at vertex 0, the first in the middle of its strip, so the
    // walk must turn both ways before counting.
    add(
        "two open fans",
        &bow_tie,
        &[[0, 1, 2], [0, 5, 1], [0, 3, 4]],
    );
    let positions = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
    ];
    let tetrahedron = |[a, b, c, d]: [u32; 4]| [[a, c, b], [a, b, d], [b, c, d], [c, a, d]];
    let faces: Vec<_> = tetrahedron([0, 1, 2, 3])
        .into_iter()
        .chain(tetrahedron([0, 4, 5, 6]))
        .collect();
    add("two tetrahedra sharing a vertex", &positions, &faces);
    fixtures
}
