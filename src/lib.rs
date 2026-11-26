//! Static patch/ribbon storage for portable mesh processing.
//!
//! Alum is the independent test oracle. Static GPU connectivity queries are
//! available; dynamic topology is not implemented yet.
pub mod gpu;
mod handle;
mod mesh;
mod patch;

pub use handle::{EdgeHandle, FaceHandle, VertexHandle};
pub use mesh::{BuildError, Mesh, MeshBuilder, VertexAttribute};
pub use patch::{DirectedEdge, Patch};

// Packaged sources can be linked by downstream WESL compilers independently
// of the host mesh API. No GPU device or MOPMesh entry point is required.
wesl_core::wesl_pkg!(pub mopmesh);
