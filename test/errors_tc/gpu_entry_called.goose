// error: compute fn k is a GPU entry point: a pipeline runs it, nothing calls it
// A GPU entry point runs in a pipeline (docs/design/gpu.md §5.1).
compute fn k() { }
fn main() { k(); }
