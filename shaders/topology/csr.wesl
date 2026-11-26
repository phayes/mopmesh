// CSR ranges use half-open entry indices, independent of buffer bindings.
public struct CsrRow {
    start: u32,
    end: u32,
}

// Caller ensures start + count fits in u32 and the entry buffer.
public fn row_from_count(start: u32, count: u32) -> CsrRow {
    return CsrRow(start, start + count);
}

public fn row_length(row: CsrRow) -> u32 {
    return row.end - row.start;
}
