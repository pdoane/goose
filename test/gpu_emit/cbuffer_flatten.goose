// Push constants whose Direct3D packing would move a field (a float4 after
// a u32): DXIL reads them flattened, from their Goose offsets, while MSL
// and SPIR-V declare the struct as it is (docs/design/gpu.md §5.3).
import gpu_shader;

struct Push { count: u32, color: float4, out: gpu::RWStructuredBuffer<float4> }

compute(8, 8) fn fill(p: Push) {
    let id = gpu::dispatch_thread_id();
    let i = id.y * 8 + id.x;
    if i < p.count { gpu::store(p.out, i, p.color); }
}

fn main() { print(gpu_entry(fill).len > 0); }
