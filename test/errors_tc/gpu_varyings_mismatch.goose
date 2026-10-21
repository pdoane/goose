// error: fragment fn fs takes other varyings than vertex fn vs passes it
import vec;
struct A { uv: float2 }
struct B { uv: float2 }
vertex fn vs() -> float4, A { return float4_0, A { float2_0 }; }
fragment fn fs(b: B) -> float4 { float4_1 }
fn main() { print(gpu_entry(vs, fs).len); }
