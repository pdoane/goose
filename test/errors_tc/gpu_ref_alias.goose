// error: two references to v in one call, at least one of them written, do not run on the GPU: HLSL passes them as copies, which an alias could tell apart
import gpu_shader;
struct V { a: u32, b: u32 }
fn add(x: u32&, y: u32&) { x += y; }
struct P { out: gpu::RWStructuredBuffer<u32> }
compute fn k(p: P) {
    var v = V { a: 1, b: 2 };
    add(v.a, v.b);
    gpu::store(p.out, 0, v.a);
}
fn main() { print(gpu_entry(k).len); }
