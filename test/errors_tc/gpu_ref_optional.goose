// error: optional references do not run on the GPU yet (design/gpu_language.md §8): x
import gpu_shader;
fn first(x: u32?) -> u32 { if x { return x; } return 0; }
struct P { out: gpu::RWStructuredBuffer<u32> }
compute fn k(p: P) {
    var v: u32 = 3;
    gpu::store(p.out, 0, first(v));
}
fn main() { print(gpu_entry(k).len); }
