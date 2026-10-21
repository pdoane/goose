// ADTs: a tag and the payload as words, variant literals packed into it,
// match as an if-chain with payloads unpacked, case functions dispatching
// on the tag, an integer match, and a buffer of them read as Goose packs
// it, a word at a time across the tag's byte.
import gpu_shader;

enum Light {
    Point { pos: float3, power: f32 },
    Spot { pos: float3, dir: float3, cone: f32 },
    Sun { dir: float3 },
    Off,
}

fn power(p: Light.Point) -> f32 { p.power }
fn power(s: Light.Spot) -> f32 { s.cone }
fn power(s: Light.Sun) -> f32 { 1.0 }
fn power(o: Light.Off) -> f32 { 0.0 }

fn kind(l: Light) -> u32 {
    match l {
        Point p => if p.power > 1.0 { 1 } else { 2 },
        Spot, Sun => 3,
        Off => 0,
    }
}

struct Push { lights: gpu::StructuredBuffer<Light>, out: gpu::RWStructuredBuffer<f32> }

compute fn adts(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    let mine: Light = match i % 3 { 0 => Light.Sun { dir: float3_1 }, 1..2 => Light.Point { pos: float3_0, power: 2.0 },
                             _ => Light.Off };
    let stored = gpu::load(p.lights, i);
    gpu::store(p.out, i, power(stored) + power(mine) + kind(stored));
}

fn main() { print(gpu_entry(adts).len > 0); }
