// error: compute fn k cannot be generic
compute fn k<T>(x: T) { }
fn main() { print(gpu_entry(k).len); }
