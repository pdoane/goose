// error: vertex fn vs and fragment fn fs take different push constants
import vec;
struct Empty { }
struct P { a: u32 }
struct Q { b: u32 }
vertex fn vs(p: P) -> float4 { float4_0 }
fragment fn fs(e: Empty, q: Q) -> float4 { float4_1 }
fn main() { print(gpu_entry(vs, fs).len); }
