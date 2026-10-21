// error: vertex fn vs returns the clip position, a float4, then its varyings, if any
import vec;
struct Empty { }
vertex fn vs() -> f32 { 1.0 }
fragment fn fs(e: Empty) -> float4 { float4_1 }
fn main() { print(gpu_entry(vs, fs).len); }
