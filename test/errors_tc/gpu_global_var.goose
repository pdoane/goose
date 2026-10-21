// error: global count is not a constant, and the GPU reads no other global
var count: u32 = 0;
compute fn k() { let c = count; }
fn main() { print(gpu_entry(k).len); }
