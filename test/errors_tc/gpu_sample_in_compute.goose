// error: sample exists in a fragment entry point only, and this is reached from compute fn k
import gpu_shader;
struct P { t: gpu::Texture2D<float4>, out: gpu::RWStructuredBuffer<float4> }
compute fn k(p: P) { gpu::store(p.out, 0, gpu::sample(p.t, gpu::LINEAR_CLAMP, float2_0)); }
fn main() { print(gpu_entry(k).len); }
