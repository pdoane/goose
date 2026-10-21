// A compute entry point over typed buffers, push constants in a struct of
// handles (which DXIL reads flattened), a helper the CPU runs too, and the
// Euclidean % and masked shifts Goose defines (docs/design/gpu.md §5.6).
import gpu_shader;

struct Push {
    input: gpu::StructuredBuffer<f32>,
    output: gpu::RWStructuredBuffer<f32>,
    n: u32,
    scale: f32,
}

fn scaled(x: f32, s: f32) -> f32 { x * s + 1.0 }

compute(64) fn scale_kernel(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    if i >= p.n { return; }
    let k = (i as i32) % 7 - 3;
    var bits = i << 3;
    bits >>= 1;
    gpu::store(p.output, i, scaled(gpu::load(p.input, i), p.scale) + k + bits);
}

fn main() { print(gpu_entry(scale_kernel).len > 0, " ", scaled(2.0, 3.0)); }
