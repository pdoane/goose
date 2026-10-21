// error: vertex_index() exists in a vertex entry point only, and this is reached from compute fn k
import gpu_shader;
fn which() -> u32 { gpu::vertex_index() }
compute fn k() { let v = which(); }
fn main() { print(gpu_entry(k).len); }
