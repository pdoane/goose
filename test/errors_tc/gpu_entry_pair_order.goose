// error: gpu_entry(vs, fs) takes a vertex fn and then a fragment fn
import vec;
struct Empty { }
vertex fn vs() -> float4 { float4_0 }
fragment fn fs(e: Empty) -> float4 { float4_1 }
fn main() { print(gpu_entry(fs, vs).len); }
