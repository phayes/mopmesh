import mopmesh::core::handle::{VertexHandle, same_vertex};
import mopmesh::topology::directed_edge::{directed_start, directed_end};
import mopmesh::topology::face::face_vertices;
import mopmesh::topology::csr::{row_from_count, row_length};
import mopmesh::geometry::triangle::triangle_centroid;
import mopmesh::topology::edge::{other_edge_face, other_edge_vertex, INVALID_FACE, INVALID_VERTEX};
import mopmesh::topology::face::{face_edge_slot, face_vertex_slot, opposite_vertex, INVALID_SLOT};
import mopmesh::build::exact::{is_finite_position, has_coincident_corners};
import mopmesh::build::edge_group::{empty_edge_group, edge_group_add, edge_group_error_corner};
import mopmesh::build::morton::{morton_code_30, morton_cell};

// This downstream shader chooses its own bindings and entry point.
@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let a = VertexHandle(3u, 7u);
    output[0] = select(0u, 1u, same_vertex(a, VertexHandle(3u, 7u)));
    output[1] = directed_start(vec2<u32>(5u, 9u), true);
    output[2] = directed_end(vec2<u32>(5u, 9u), true);
    output[3] = face_vertices(array(vec2(0u, 1u), vec2(1u, 2u), vec2(0u, 2u)), vec3(0u, 0u, 1u)).x;
    output[4] = row_length(row_from_count(7u, 3u));
    output[5] = u32(triangle_centroid(vec3(0.0), vec3(3.0), vec3(6.0)).x);
    output[6] = other_edge_face(vec2(3u, 5u), 3u);
    output[7] = other_edge_vertex(vec2(2u, 7u), 7u);
    output[8] = face_edge_slot(vec3(2u, 3u, 7u), vec2(7u, 3u));
    output[9] = opposite_vertex(vec3(2u, 3u, 7u), vec2(7u, 3u));
    // Check that missing vertices, edge slots, and boundary faces return the named sentinels.
    output[10] = select(0u, 1u, other_edge_vertex(vec2(2u, 7u), 4u) == INVALID_VERTEX);
    output[11] = select(0u, 1u, face_edge_slot(vec3(2u, 3u, 7u), vec2(2u, 9u)) == INVALID_SLOT);
    output[12] = select(0u, 1u, other_edge_face(vec2(3u, INVALID_FACE), 3u) == INVALID_FACE);
    output[13] = face_vertex_slot(vec3(2u, 3u, 7u), 7u);
    // Build helpers: exact position checks, edge grouping and Morton codes.
    let zero = bitcast<vec3<u32>>(vec3(0.0));
    let negative_zero = bitcast<vec3<u32>>(vec3(-0.0));
    output[14] = select(0u, 1u, is_finite_position(zero));
    output[15] = select(0u, 1u, has_coincident_corners(zero, negative_zero, vec3(1u)));
    let group = edge_group_add(edge_group_add(empty_edge_group(), 6u, false), 4u, false);
    output[16] = edge_group_error_corner(group);
    output[17] = morton_code_30(morton_cell(vec3(1.0), vec3(0.0), vec3(1.0)));
}
