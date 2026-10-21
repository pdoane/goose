// error: gpu_entry(vs) makes a compute pipeline: a vertex entry point needs its other stage, gpu_entry(vs, fs)
import vec;
vertex fn vs() -> float4 { float4_0 }
fn main() { print(gpu_entry(vs).len); }
