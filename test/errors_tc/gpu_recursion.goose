// error: recursion does not run on the GPU: down calls itself
recursive fn down(n: u32) -> u32 { if n == 0 { 0 } else { down(n - 1) } }
compute fn k() { let d = down(3); }
fn main() { print(gpu_entry(k).len); }
