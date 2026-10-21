// error: fragment fn fs takes its vertex stage's varyings first, an empty struct for none
import vec;
vertex fn vs() -> float4 { float4_0 }
fragment fn fs() -> float4 { float4_1 }
fn main() { print(gpu_entry(vs, fs).len); }
