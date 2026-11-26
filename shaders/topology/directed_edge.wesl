// Binding-free connectivity helpers. Callers supply EV records themselves.
// Record values are independent of the caller's storage layout.
public fn directed_start(vertices: vec2<u32>, reversed: bool) -> u32 {
    return select(vertices.x, vertices.y, reversed);
}

public fn directed_end(vertices: vec2<u32>, reversed: bool) -> u32 {
    return select(vertices.y, vertices.x, reversed);
}
