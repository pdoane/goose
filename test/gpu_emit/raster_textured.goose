// A vertex and a fragment entry point: vertices pulled from a buffer, a
// uniform block whose packing agrees with Direct3D's (no flattening), an
// integer varying (flat), a sampled texture through a constant sampler,
// discard, and a struct-valued global read as its initializer.
import gpu_shader;

struct Vertex { pos: float3, uv: float2 }
struct Draw { vertices: gpu::StructuredBuffer<Vertex>, albedo: gpu::Texture2D<float4>, tint: float4 }
struct Scene { view_proj: gpu::mat4, exposure: f32, pad 12 }
struct Varyings { uv: float2, id: u32 }

vertex fn mesh_vs(p: Draw, scene: const Scene&) -> float4, Varyings {
    let v = gpu::load(p.vertices, gpu::vertex_index());
    let clip = gpu::mul(scene.view_proj, float4 { v.pos.x, v.pos.y, v.pos.z, 1.0 });
    return clip, Varyings { uv: v.uv, id: gpu::instance_index() };
}

fragment fn mesh_fs(f: Varyings, p: Draw, scene: const Scene&) -> float4 {
    if f.id > 100 { gpu::discard(); }
    let c = gpu::sample(p.albedo, gpu::LINEAR_CLAMP, f.uv);
    c * p.tint * scene.exposure
}

fn main() { print(gpu_entry(mesh_vs, mesh_fs).len > 0); }
