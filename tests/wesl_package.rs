#[test]
fn downstream_shader_can_link_packaged_library() {
    for name in [
        "consumer",
        "scan_consumer",
        "geometry_consumer",
        "patch_consumer",
        "flip_consumer",
    ] {
        validate(&compile_consumer(name));
    }
    for name in ["weighted_boundary", "neighborhood", "bending"] {
        let mut resolver = wesl::resolver::StandardResolver::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/shaders"
        ));
        resolver.add_package(&mopmesh::mopmesh::PACKAGE);
        let compiled = wesl::Compiler::default()
            .with_resolver(resolver)
            .compile_module(&format!("package::{name}").parse().unwrap())
            .unwrap();
        validate(&compiled.to_string());
    }
}

fn compile_consumer(name: &str) -> String {
    let mut resolver = wesl::resolver::StandardResolver::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/shaders"
    ));
    resolver.add_package(&mopmesh::mopmesh::PACKAGE);
    let compiled = wesl::Compiler::default()
        .with_resolver(resolver)
        .compile_module(&format!("package::{name}").parse().unwrap())
        .expect("downstream shader should link against packaged MOPMesh modules");
    let wgsl = compiled.to_string();
    assert!(wgsl.contains("@compute"));
    assert!(!wgsl.contains("import "));
    assert!(!wgsl.contains("public "));
    wgsl
}

fn validate(wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl).expect("valid WGSL syntax");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("portable WGSL validation");
}

#[test]
fn built_in_gpu_query_is_valid_portable_wgsl() {
    validate(include_str!(concat!(
        env!("OUT_DIR"),
        "/vertex_neighbors.wgsl"
    )));
    validate(include_str!(concat!(
        env!("OUT_DIR"),
        "/face_neighbors.wgsl"
    )));
    validate(include_str!(concat!(env!("OUT_DIR"), "/edges.wgsl")));
    validate(include_str!(concat!(env!("OUT_DIR"), "/scan.wgsl")));
    validate(include_str!(concat!(
        env!("OUT_DIR"),
        "/face_centroids.wgsl"
    )));
    validate(include_str!(concat!(
        env!("OUT_DIR"),
        "/face_vertices.wgsl"
    )));
    for wgsl in [
        include_str!(concat!(env!("OUT_DIR"), "/face_normals.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/face_areas.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/vertex_normals.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/boundary.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/reduce.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/smooth.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/edge_flip.wgsl")),
    ] {
        validate(wgsl);
    }
    validate(include_str!(concat!(env!("OUT_DIR"), "/vertex_faces.wgsl")));
    validate(include_str!(concat!(
        env!("OUT_DIR"),
        "/build_elements.wgsl"
    )));
    validate(include_str!(concat!(
        env!("OUT_DIR"),
        "/build_connectivity.wgsl"
    )));
    for wgsl in [
        include_str!(concat!(env!("OUT_DIR"), "/build_patch_members.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/build_patch_rows.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/build_patch_local.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/build_arena.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/build_split.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/radix_sort.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/baseline_face_vertices.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/baseline_edges.wgsl")),
        include_str!(concat!(env!("OUT_DIR"), "/staged_edges.wgsl")),
    ] {
        validate(wgsl);
    }
}
