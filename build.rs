fn main() {
    println!("cargo:rerun-if-changed=runtime_shaders");
    println!("cargo:rerun-if-changed=wesl.toml");
    println!("cargo:rerun-if-changed=shaders");
    wesl::package::PackageBuilder::new("mopmesh")
        .scan_toml(".")
        .expect("scan MOPMesh shader package")
        .validate()
        .expect("validate MOPMesh shader modules")
        .build_artifact()
        .expect("emit MOPMesh shader package");
    for name in [
        "face_vertices",
        "vertex_faces",
        "face_centroids",
        "face_normals",
        "face_areas",
        "vertex_normals",
        "boundary",
        "reduce",
        "smooth",
        "edge_flip",
        "cavity_rounds",
        "cavity_migrate",
        "scan",
        "edges",
        "vertex_neighbors",
        "build_elements",
        "build_connectivity",
        "build_patch_members",
        "build_patch_rows",
        "build_patch_local",
        "build_arena",
        "build_split",
        "radix_sort",
        "baseline_face_vertices",
        "baseline_edges",
        "baseline_face_neighbors",
    ] {
        let resolver = wesl::resolver::StandardResolver::new(".");
        let compiled = wesl::Compiler::default()
            .with_resolver(resolver)
            .compile_module(&format!("package::runtime_shaders::{name}").parse().unwrap())
            .expect("compile connectivity query");
        std::fs::write(
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
                .join(format!("{name}.wgsl")),
            compiled.to_string(),
        )
        .expect("write compiled connectivity query");
    }
}
