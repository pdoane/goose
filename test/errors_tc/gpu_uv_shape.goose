// error: a texture coordinate is a float2, not vec3<f32>
import gpu_shader;
struct Empty { }
struct P { t: gpu::Texture2D<float4> }
vertex fn vs(p: P) -> float4 { float4_0 }
fragment fn fs(e: Empty, p: P) -> float4 { gpu_sample<float4>(p.t.index, 0, float3_0) }
fn main() { print(gpu_entry(vs, fs).len); }
