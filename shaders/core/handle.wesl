// Mesh-relative patch/local addresses. Migration and slot reuse invalidate them.
public struct VertexHandle {
    patch_id: u32,
    local_id: u32,
}

public struct EdgeHandle {
    patch_id: u32,
    local_id: u32,
}

public struct FaceHandle {
    patch_id: u32,
    local_id: u32,
}

public fn same_vertex(a: VertexHandle, b: VertexHandle) -> bool {
    return a.patch_id == b.patch_id && a.local_id == b.local_id;
}
