// A texture both written and sampled by shaders (docs/design/gpu.md §3):
// one pass writes each of its mips through a storage view of that mip,
// and after a barrier another reads every mip back through the texture's
// sampled handle.
import gpu from "..";
import std;

struct Write { dst: gpu::RWTexture2D<float4>, size: u32, mip: u32 }

compute(8, 8) fn fill(p: Write) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.size || id.y >= p.size { return; }
    let v = float4 { p.mip / 8.0, id.x / 64.0, id.y / 64.0, 1.0 };
    gpu::store(p.dst, int2 { id.x as i32, id.y as i32 }, v);
}

struct Read { src: gpu::Texture2D<float4>, out: gpu::RWStructuredBuffer<float4> }

// Texel (1, 0) of each of the four mips, and of the 1x1 one its only one.
compute(1) fn probe(p: Read) {
    for m: i32 in 4 {
        let x: i32 = if m == 3 { 0 } else { 1 };
        gpu::store(p.out, m as u32, gpu::texel(p.src, int2 { x, 0 }, m));
    }
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let t = gpu::texture(gpu::TextureDesc { format: gpu::RGBA16F, usage: gpu::SAMPLED | gpu::STORAGE,
                                            width: 8, height: 8, mips: 0, .. });
    let out = gpu::buffer(gpu::BUFFER_WRITE, 4 * 16);
    gpu::barrier(t, gpu::ACCESS_NONE, gpu::ACCESS_SHADER_WRITE);
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(fill));
    for m in 4 {
        let size = 8 >> m;
        gpu::push_constants(Write { dst: gpu::storage_mip<float4>(t, m), size: size as u32, mip: m as u32 });
        gpu::dispatch(1, 1, 1);
    }
    gpu::end_compute();
    gpu::barrier(t, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_SHADER_READ);
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(probe));
    gpu::push_constants(Read { src: gpu::sampled<float4>(t), out: gpu::rw_structured<float4>(out) });
    gpu::dispatch(1, 1, 1);
    gpu::end_compute();
    let r = gpu::read_buffer(out, 0, 4 * 16);
    gpu::wait(gpu::submit());
    var got: float4[>..] = [];
    guard gpu::values(r, got) else { abort("gpu: a read back is not ready"); }
    var wrong = 0;
    for m in 4 {
        let want = float4 { m / 8.0, if m == 3 { 0.0 } else { 1.0 / 64.0 }, 0.0, 1.0 };
        let d = got[m] - want;
        if abs(d.x) > 0.001 || abs(d.y) > 0.001 || abs(d.z) > 0.001 || abs(d.w) > 0.001 {
            print("mip ", m, ": ", got[m].x, " ", got[m].y, " ", got[m].z, " ", got[m].w);
            wrong++;
        }
    }
    print(gpu::driver(), ": 4 mips written through views and sampled, ", wrong, " wrong");
    gpu::close();
}
