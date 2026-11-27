// Morton (Z-order) codes: interleave the bits of three grid coordinates so
// that points close in space tend to get close codes. Sorting by the code
// gives a space-filling order, useful for grouping nearby faces.

// Spreads the low 10 bits of `x` so two zero bits follow each one.
public fn morton_spread_10(x: u32) -> u32 {
    var v = x & 0x3ffu;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v << 8u)) & 0x0300f00fu;
    v = (v | (v << 4u)) & 0x030c30c3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

// The 30-bit Morton code of a cell in a 1024^3 grid. Each coordinate must be
// below 1024; higher bits are ignored.
public fn morton_code_30(cell: vec3<u32>) -> u32 {
    return morton_spread_10(cell.x) | (morton_spread_10(cell.y) << 1u)
        | (morton_spread_10(cell.z) << 2u);
}

// The grid cell of `point` in the box from `low` to `high`, 1024 cells per
// axis, clamped to the grid. A flat axis (low == high) maps to cell 0.
public fn morton_cell(point: vec3<f32>, low: vec3<f32>, high: vec3<f32>) -> vec3<u32> {
    let extent = max(high - low, vec3(1e-30));
    let scaled = clamp((point - low) / extent * 1024.0, vec3(0.0), vec3(1023.0));
    return vec3<u32>(scaled);
}
