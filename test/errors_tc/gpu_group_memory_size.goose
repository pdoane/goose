// error: group memory is at most 32 KiB, and b brings it to 36864 bytes
import gpu_shader;
var a: gpu::GroupShared<u32[4096]>;
var b: gpu::GroupShared<u32[5120]>;
struct P { out: gpu::RWStructuredBuffer<u32> }
compute fn k(p: P) {
    let i = gpu::group_index();
    a.value[i] = i;
    b.value[i] = i;
    gpu::group_barrier();
    gpu::store(p.out, i, a.value[i] + b.value[i]);
}
fn main() { print(gpu_entry(k).len); }
