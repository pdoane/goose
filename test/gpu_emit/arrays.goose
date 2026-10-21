// Fixed arrays on the GPU: locals, literals of both forms, struct fields of
// arrays of vectors (which MSL stores packed), indexing to read and write,
// passing and returning one, .len, and for over one by value, with its
// index, and by reference; and global tables of literals, declared once.
import gpu_shader;

let WEIGHTS: u32[4] = [0, 21, 43, 64];
struct Step { at: float2, scale: f32 }
let STEPS: Step[2] = [Step { at: float2 { 0.5, -1.0 }, scale: 2.0 }, Step { at: float2_0, scale: 1.0 / 3.0 }];

struct Block { texels: float3[16], weights: u32[4] }
struct Push { blocks: gpu::RWStructuredBuffer<Block>, out: gpu::RWStructuredBuffer<float4> }

fn total(ws: u32[4]) -> u32 {
    var s: u32 = 0;
    for w in ws { s += w; }
    s
}

fn doubled(a: f32[4]) -> f32[4] {
    var r = a;
    for &x in r { x *= 2.0; }
    r
}

compute(16) fn arrays(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    var b = gpu::load(p.blocks, i);
    let mins: f32[4] = [0.25, 0.5, 0.75, 1.0];
    var acc: float3[4] = [float3_0; 4];
    for t, k: u32 in b.texels {
        acc[k & 3] += t * mins[k & 3];
    }
    b.texels[0] = acc[1];
    b.texels[1].y = 2.0;
    b.weights[2] = total(b.weights) + b.weights.len as u32 + WEIGHTS[i & 3] + total(WEIGHTS);
    b.texels[2].x = STEPS[i & 1].at.y * STEPS[1].scale;
    gpu::store(p.blocks, i, b);
    let d = doubled(mins);
    gpu::store(p.out, i, float4 { d[0], d[1], d[2], d[3] });
}

fn main() { print(gpu_entry(arrays).len > 0); }
