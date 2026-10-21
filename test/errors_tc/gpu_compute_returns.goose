// error: compute fn k cannot return values
compute fn k() -> u32 { 1 }
fn main() { print(gpu_entry(k).len); }
