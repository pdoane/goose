// References as parameters and locals, and several results: HLSL's inout,
// MSL's thread references, a copy in and out where MSL stores a vector
// packed, and a struct of the results.
import gpu_shader;

struct Body { pos: float3, vel: float3, hits: u32 }

fn bump(n: u32&) { n += 1; }
fn nudge(v: float3&) { v.y += 1.0; }
fn speed(b: const Body&) -> f32 { length(b.vel) }

fn step(b: Body&, dt: f32) {
    b.pos = b.pos + b.vel * dt;
    nudge(b.vel);
    bump(b.hits);
}

fn divmod(a: u32, b: u32) -> u32, u32 { return a / b, a % b; }

struct Push { out: gpu::RWStructuredBuffer<Body>, s: gpu::RWStructuredBuffer<f32> }

compute fn references(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    var b = gpu::load(p.out, i);
    step(b, 0.5);
    let q, r = divmod(i, 3);
    let h .= b.hits;
    h += q * r;
    gpu::store(p.out, i, b);
    gpu::store(p.s, i, speed(b));
}

fn main() { print(gpu_entry(references).len > 0); }
