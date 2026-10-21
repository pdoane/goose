// error: push constants holding an ADT do not run on the GPU yet: read it from a buffer
import gpu_shader;
enum Light { Point { power: f32 }, Sun }
struct P { light: Light, out: gpu::RWStructuredBuffer<f32> }
compute fn k(p: P) {
    gpu::store(p.out, 0, match p.light { Point l => l.power, Sun => 1.0 });
}
fn main() { print(gpu_entry(k).len); }
