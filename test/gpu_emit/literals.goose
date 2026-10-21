// Constants Goose leaves untyped, and C's rounding functions: literal
// arithmetic folded at i64 and written at the type it meets, a literal cast
// or negated at its own type, and floor, ceil, trunc and round on f32,
// round's halves away from zero in HLSL too.
import gpu_shader;
import math;

struct Push { out: gpu::RWStructuredBuffer<u32>, f: gpu::RWStructuredBuffer<f32> }

compute fn literals(p: Push) {
    let g = gpu::dispatch_thread_id().x;
    gpu::store(p.out, 0, g | (g << 8) | (255 << 24));
    gpu::store(p.out, 1, 0xffffffff as u32);
    var lo: i32 = -0x7fffffff;
    lo += 1 << 4;
    gpu::store(p.out, 2, lo as u32);
    let x = g as f32 + 0.5;
    gpu::store(p.f, 0, floor(x) + ceil(x) + trunc(x) + round(x));
}

fn main() { print(gpu_entry(literals).len > 0); }
