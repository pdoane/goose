// error: group memory exists in a compute entry point only, and tile is reached from fragment fn fs
import gpu_shader;
var tile: gpu::GroupShared<f32[4]>;
struct V { uv: float2 }
vertex fn vs() -> float4, V { return float4_0, V { uv: float2_0 }; }
fragment fn fs(v: V) -> float4 { float4 { tile.value[0], 0.0, 0.0, 1.0 } }
fn main() { print(gpu_entry(vs, fs).len); }
