# Reference implementations

## RXMesh

- Source: https://github.com/owensgroup/RXMesh
- Local checkout: `prior_art/RXMesh`
- Recorded revision: `3990228e91df0d5d2b5fcbf91827b3e92256f85d`
- License: BSD-2-Clause; original notice retained in the checkout and `THIRD_PARTY_NOTICES`.
- Role: patch/ribbon architecture and dynamic cavity prior art. CUDA reference execution has not been configured.

## Alum

- Source: https://github.com/ranjeethmahankali/alum
- Cargo development dependency: `alum` 0.7.1, with default features disabled.
- Reproducibility: exact resolved version and checksum in `Cargo.lock`.
- License: BSD-3-Clause; notice retained in `THIRD_PARTY_NOTICES`.
- Role: primary CPU oracle for canonical connectivity (FV, EV/EF, VF, VE/VV, FF and edge diamonds), centroids, and downstream boundary/neighborhood geometry results. Match edges by endpoints rather than assuming identical edge numbering. Future edit tests should use Alum operations rather than a homegrown CPU reference implementation.

The production Rust constructor builds MOPMesh storage and validates supported
input topology; it is not used as the independent correctness oracle.

## OpenMesh

- Official source: https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git
- Local checkout: `prior_art/OpenMesh`
- Recorded revision: `ded6081c3eea2fea42970fb104ccc3bb824c0add` (reports version 11.0.0).
- CMake submodule: `3eada60fc5ecbc40e297783d43c07c69a57b2ae1`.
- License: BSD-3-Clause; original notice retained in the checkout and `THIRD_PARTY_NOTICES`.
- Role: secondary C++ reference and prior art. OpenMeshCore builds locally; Rust comparison tests currently use Alum, not an OpenMesh FFI bridge.

Reproduce the checkout in a fresh repository:

```sh
git clone https://gitlab.vci.rwth-aachen.de:9000/OpenMesh/OpenMesh.git prior_art/OpenMesh
git -C prior_art/OpenMesh checkout ded6081c3eea2fea42970fb104ccc3bb824c0add
git -C prior_art/OpenMesh submodule update --init --recursive
```

Build the optional reference without GUI applications:

```sh
cmake -S prior_art/OpenMesh -B target/openmesh -DBUILD_APPS=OFF -DOPENMESH_BUILD_UNIT_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build target/openmesh --target OpenMeshCore --parallel 4
```

The downloaded source trees remain independent Git checkouts; this project has
not converted them into top-level submodules. See the recorded revisions above
when reproducing the environment. Neither source tree is required for `cargo test`.
The source checkouts are ignored by the top-level repository; this reference
document is explicitly included. Alum is fetched through Cargo. A benchmark
dataset and CUDA reference execution remain open work in [the roadmap](../ROADMAP.md).
