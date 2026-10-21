// error: i64 does not run on the GPU yet
// An untyped integer is an i64, which the GPU does not compute in (§5.2).
compute fn k() { var n = 3; n++; }
fn main() { print(gpu_entry(k).len); }
