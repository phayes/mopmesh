// The corners that share one undirected edge, as a small value. Corner `c` is
// slot `c % 3` of face `c / 3` and runs from its vertex to the next corner's.
// Add each corner on the edge once, in any order; the group keeps the total
// count and the three lowest corners with their direction.
public struct EdgeGroup {
    count: u32,
    corners: array<u32, 3>,
    // True if the corner runs from the edge's larger vertex to its smaller one.
    reversed: array<bool, 3>,
}

public fn empty_edge_group() -> EdgeGroup {
    let none = 0xffffffffu;
    return EdgeGroup(0u, array(none, none, none), array(false, false, false));
}

// Returns `group` with `corner` added.
public fn edge_group_add(group: EdgeGroup, corner: u32, reversed: bool) -> EdgeGroup {
    var result = group;
    var c = corner;
    var r = reversed;
    let kept = min(group.count, 3u);
    for (var i = 0u; i < kept; i++) {
        if (c < result.corners[i]) {
            let tc = result.corners[i];
            let tr = result.reversed[i];
            result.corners[i] = c;
            result.reversed[i] = r;
            c = tc;
            r = tr;
        }
    }
    if (kept < 3u) {
        result.corners[kept] = c;
        result.reversed[kept] = r;
    }
    result.count += 1u;
    return result;
}

// True if the two lowest corners run the same way along the edge, so the
// faces sharing it have opposite winding.
public fn edge_group_misoriented(group: EdgeGroup) -> bool {
    return group.count >= 2u && group.reversed[0] == group.reversed[1];
}

// The corner that first makes the edge invalid, or 0xffffffff if it is valid:
// the second corner if the winding is inconsistent, else the third corner if
// more than two faces share the edge. This is the corner MeshBuilder reports.
public fn edge_group_error_corner(group: EdgeGroup) -> u32 {
    if (edge_group_misoriented(group)) { return group.corners[1]; }
    if (group.count >= 3u) { return group.corners[2]; }
    return 0xffffffffu;
}
