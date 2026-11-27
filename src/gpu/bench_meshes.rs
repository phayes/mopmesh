//! Meshes for the baseline benchmark: generated surfaces of a chosen size,
//! shuffled copies of them, and OBJ files loaded through Alum.
use super::test_support::fixtures;

/// A benchmark input: positions and consistently wound triangles.
pub(super) struct BenchMesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

/// A small deterministic random number generator (SplitMix64), so every run
/// generates the same meshes.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Returns a value in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Returns a value in `[-1, 1)`.
    fn signed_unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.
    }
}

/// The generated meshes with about `faces` triangles each: a flat grid, a
/// closed UV sphere (high-valence poles), a closed torus (all valence 6) and
/// an irregular grid (random diagonals and jittered positions, valence 2–8).
pub(super) fn generated(faces: u32) -> Vec<BenchMesh> {
    // A grid side with about `faces` triangles: 2 (n - 1)^2.
    let side = ((faces as f64 / 2.).sqrt().round() as u32 + 1).max(2);
    let (positions, triangles) = fixtures::grid(side);
    vec![
        BenchMesh {
            name: format!("grid {side}x{side}"),
            positions,
            triangles,
        },
        sphere(faces),
        torus(faces),
        irregular(side),
    ]
}

/// A UV sphere with `2 s (r - 1)` faces for `r` rings and `s = 2r` segments.
fn sphere(faces: u32) -> BenchMesh {
    let rings = ((faces as f64 / 4.).sqrt().round() as u32).max(2);
    let segments = 2 * rings;
    let mut positions = vec![[0., 0., 1.]];
    for ring in 1..rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32;
        for s in 0..segments {
            let phi = std::f32::consts::TAU * s as f32 / segments as f32;
            positions.push([
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]);
        }
    }
    let south = positions.len() as u32;
    positions.push([0., 0., -1.]);

    // Vertex `s` of ring `ring` (1-based), wrapping around the segments.
    let at = |ring: u32, s: u32| 1 + (ring - 1) * segments + s % segments;
    let mut triangles = Vec::new();
    for s in 0..segments {
        triangles.push([0, at(1, s), at(1, s + 1)]);
        triangles.push([south, at(rings - 1, s + 1), at(rings - 1, s)]);
    }
    for ring in 1..rings - 1 {
        for s in 0..segments {
            let (a, b) = (at(ring, s), at(ring, s + 1));
            let (c, d) = (at(ring + 1, s), at(ring + 1, s + 1));
            triangles.extend([[a, c, d], [a, d, b]]);
        }
    }
    BenchMesh {
        name: format!("sphere {rings}x{segments}"),
        positions,
        triangles,
    }
}

/// A torus with `2 m k` faces for `m = 2k` major and `k` minor segments.
fn torus(faces: u32) -> BenchMesh {
    let minor = ((faces as f64 / 4.).sqrt().round() as u32).max(3);
    let major = 2 * minor;
    let mut positions = Vec::new();
    for i in 0..major {
        let u = std::f32::consts::TAU * i as f32 / major as f32;
        for j in 0..minor {
            let v = std::f32::consts::TAU * j as f32 / minor as f32;
            let r = 1. + 0.3 * v.cos();
            positions.push([r * u.cos(), r * u.sin(), 0.3 * v.sin()]);
        }
    }
    let at = |i: u32, j: u32| (i % major) * minor + j % minor;
    let mut triangles = Vec::new();
    for i in 0..major {
        for j in 0..minor {
            let (a, b) = (at(i, j), at(i + 1, j));
            let (c, d) = (at(i, j + 1), at(i + 1, j + 1));
            triangles.extend([[a, b, d], [a, d, c]]);
        }
    }
    BenchMesh {
        name: format!("torus {major}x{minor}"),
        positions,
        triangles,
    }
}

/// An `n` x `n` vertex grid whose quads take a random diagonal and whose
/// interior vertices are jittered within their cell.
fn irregular(n: u32) -> BenchMesh {
    let mut rng = Rng(0x5eed);
    let mut positions = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let interior = x > 0 && y > 0 && x + 1 < n && y + 1 < n;
            let jitter = if interior { 0.3 } else { 0. };
            positions.push([
                x as f32 + jitter * rng.signed_unit(),
                y as f32 + jitter * rng.signed_unit(),
                0.,
            ]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let a = y * n + x;
            let (b, c, d) = (a + 1, a + n, a + n + 1);
            if rng.below(2) == 0 {
                triangles.extend([[a, b, c], [b, d, c]]);
            } else {
                triangles.extend([[a, b, d], [a, d, c]]);
            }
        }
    }
    BenchMesh {
        name: format!("irregular {n}x{n}"),
        positions,
        triangles,
    }
}

/// Returns `mesh` with its vertex IDs and face order randomly permuted, and
/// each triangle's corners rotated by a random amount (keeping its winding).
/// The surface is unchanged, but neighboring elements no longer have nearby IDs.
pub(super) fn shuffled(mesh: &BenchMesh) -> BenchMesh {
    let mut rng = Rng(0x5eed_f1e5_u64.wrapping_add(mesh.triangles.len() as u64));
    let mut permute = |len: usize| {
        let mut order: Vec<u32> = (0..len as u32).collect();
        for i in (1..len).rev() {
            order.swap(i, rng.below(i + 1));
        }
        order
    };
    let new_vertex = permute(mesh.positions.len());
    let face_order = permute(mesh.triangles.len());
    let mut positions = vec![[0.; 3]; mesh.positions.len()];
    for (old, &new) in new_vertex.iter().enumerate() {
        positions[new as usize] = mesh.positions[old];
    }
    let triangles = face_order
        .iter()
        .map(|&f| {
            let t = mesh.triangles[f as usize].map(|v| new_vertex[v as usize]);
            let r = rng.below(3);
            [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
        })
        .collect();
    BenchMesh {
        name: format!("{} shuffled", mesh.name),
        positions,
        triangles,
    }
}

/// Loads a triangle or polygon mesh from an OBJ file through Alum, splitting
/// polygons into triangle fans. Panics if the file cannot be read or Alum
/// rejects its topology.
pub(super) fn load_obj(path: &str) -> BenchMesh {
    use alum::{Handle, HasIterators, HasTopology};
    let mesh = alum::PolyMeshF32::load_obj(path).unwrap_or_else(|e| panic!("{path}: {e:?}"));
    let points = mesh.points();
    let points = points.try_borrow().unwrap();
    let positions = mesh
        .vertices()
        .map(|v| {
            let p = points[v];
            [p.0, p.1, p.2]
        })
        .collect();
    let mut triangles = Vec::new();
    for f in mesh.faces() {
        let corners: Vec<u32> = mesh.fv_ccw_iter(f).map(|v| v.index()).collect();
        for i in 1..corners.len() - 1 {
            triangles.push([corners[0], corners[i], corners[i + 1]]);
        }
    }
    let name = std::path::Path::new(path)
        .file_name()
        .map_or(path.into(), |n| n.to_string_lossy().into_owned());
    BenchMesh {
        name,
        positions,
        triangles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;

    #[test]
    fn generated_and_shuffled_meshes_build() {
        for mesh in generated(2000) {
            let shuffled = shuffled(&mesh);
            for mesh in [mesh, shuffled] {
                let built = MeshBuilder::new()
                    .positions(&mesh.positions)
                    .triangles(&mesh.triangles)
                    .build()
                    .unwrap_or_else(|e| panic!("{}: {e}", mesh.name));
                let faces = built.face_count();
                assert!(
                    (1500..2600).contains(&faces),
                    "{}: {faces} faces",
                    mesh.name
                );
            }
        }
    }
}
