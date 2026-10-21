// error: storing an ADT or a limited array into a buffer does not run on the GPU yet: Goose packs what follows its tag or length unaligned, and a GPU writes whole words, which neighbouring elements would share (design/gpu_language.md §5)
import gpu_shader;
enum Light { Point { power: f32 }, Sun }
struct P { out: gpu::RWStructuredBuffer<Light> }
compute fn k(p: P) { gpu::store(p.out, 0, Light.Point { power: 2.0 }); }
fn main() { print(gpu_entry(k).len); }
