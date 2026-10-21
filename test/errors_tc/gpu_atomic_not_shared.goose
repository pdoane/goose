// error: gpu_atomic changes a value in group memory (gpu::GroupShared); a buffer's atomics are gpu::atomic_add and the rest
import gpu_shader;
struct P { out: gpu::RWStructuredBuffer<u32> }
compute fn k(p: P) {
    var n: u32 = 0;
    gpu::store(p.out, 0, gpu_atomic(gpu::ATOMIC_ADD, n, 1));
}
fn main() { print(gpu_entry(k).len); }
