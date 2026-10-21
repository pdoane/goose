// error: f64 does not run on the GPU: MSL has no doubles
compute fn k() { var x: f64 = 1.0; x += 1.0; }
fn main() { print(gpu_entry(k).len); }
