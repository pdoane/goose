// error: slices do not run on the GPU, strings among them
fn name() -> const u8[:] { "k" }
compute fn k() { let s = name(); }
fn main() { print(gpu_entry(k).len); }
