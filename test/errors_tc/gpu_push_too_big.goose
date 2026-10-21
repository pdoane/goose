// error: push constants are at most 128 bytes, and Big is 132
import vec;
struct Big { a: float4, b: float4, c: float4, d: float4, e: float4, f: float4, g: float4,
             h: float4, i: f32 }
compute fn k(p: Big) { let x = p.i; }
fn main() { print(gpu_entry(k).len); }
