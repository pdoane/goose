// Inline ray queries (docs/design/gpu.md §7): gpu::trace and gpu::trace_any
// as RayQuery in HLSL and an intersector in MSL, each a helper emitted on
// first use, reaching the acceleration structure through the heap.
import gpu_shader;

struct Push { scene: gpu::Accel, out: gpu::RWStructuredBuffer<float4> }

compute(8, 8) fn shade(p: Push) {
    let id = gpu::dispatch_thread_id();
    let r = gpu::Ray { origin: float3 { id.x, id.y, 10.0 }, t_min: 0.001,
                       dir: float3 { 0.0, 0.0, -1.0 }, t_max: 1000.0 };
    let h = gpu::trace(p.scene, r, 255);
    var c = float4_0;
    if h.kind == gpu::HIT_TRIANGLE {
        let shadow = gpu::Ray { origin: r.origin + r.dir * h.t, t_min: 0.001,
                                dir: float3 { 0.0, 1.0, 0.0 }, t_max: 1000.0 };
        let lit: f32 = if gpu::trace_any(p.scene, shadow, 255) { 0.2 } else { 1.0 };
        c = float4 { h.barycentrics.x * lit, h.barycentrics.y * lit, h.t, 1.0 };
    }
    gpu::store(p.out, id.y * 8 + id.x, c);
}

fn main() { print(gpu_entry(shade).len > 0); }
