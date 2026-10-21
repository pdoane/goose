// error: field on is a bool, which has no layout on the GPU: use u32
struct Push { on: bool, n: u32 }
compute fn k(p: Push) { let n = p.n; }
fn main() { print(gpu_entry(k).len); }
