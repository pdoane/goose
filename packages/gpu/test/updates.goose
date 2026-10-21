// Updates to resources in use (docs/design/gpu.md §3): several to one
// buffer before a submission, overlapping, land in order; a buffer
// rewritten for each of several submissions in flight at once is seen by
// each as it was written for it; and a texture updated between
// submissions likewise.
import gpu from "..";
import std;

struct Copy {
    src: gpu::StructuredBuffer<f32>,
    tex: gpu::Texture2D<float4>,
    dst: gpu::RWStructuredBuffer<f32>,
}

// Each thread copies one float of the buffer, and the first four the
// texture's texel (0, 0).
compute(64) fn copy(p: Copy) {
    let i = gpu::dispatch_thread_id().x;
    if i < 4 {
        let t = gpu::texel(p.tex, int2_0, 0);
        let c = if i == 0 { t.x } else if i == 1 { t.y } else if i == 2 { t.z } else { t.w };
        gpu::store(p.dst, i, c);
    } else if i < 64 {
        gpu::store(p.dst, i, gpu::load(p.src, i));
    }
}

fn floats(n: i64, v: f32) -> f32[>..] {
    var out: f32[>..] = [];
    for i in n { out.push(v + i); }
    out
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let src = gpu::buffer(gpu::BUFFER_READ, 64 * 4);
    let tex = gpu::texture(gpu::TextureDesc { format: gpu::RGBA8, width: 1, height: 1, .. });
    let p = gpu::compute_pipeline(copy);
    let rounds = 6;
    var outs: gpu::Buffer[>..] = [];
    var reads: gpu::Readback[>..] = [];
    var fences: u64[>..] = [];
    for r in rounds {
        // Three updates before this submission: the whole buffer, then two
        // overlapping halves, the later one winning where they overlap.
        gpu::update_buffer(src, bytes_of(floats(64, 1000.0 * r)));
        gpu::update_buffer(src, 16 * 4, bytes_of(floats(32, 2000.0 * r)));
        gpu::update_buffer(src, 32 * 4, bytes_of(floats(16, 3000.0 * r)));
        let texel: u8[4] = [r as u8 * 10, r as u8 * 20, r as u8 * 30, 255];
        gpu::update_texture(tex, texel);
        let dst = gpu::buffer(gpu::BUFFER_WRITE, 64 * 4);
        outs.push(dst);
        gpu::begin_compute();
        gpu::bind(p);
        gpu::push_constants(Copy { src: gpu::structured<f32>(src), tex: gpu::sampled<float4>(tex),
                                   dst: gpu::rw_structured<f32>(dst) });
        gpu::dispatch(1, 1, 1);
        gpu::end_compute();
        reads.push(gpu::read_buffer(dst, 0, 64 * 4));
        // No waiting: every round's work is in flight with the next's updates.
        fences.push(gpu::submit());
    }
    gpu::wait(fences[fences.len - 1]);
    var wrong = 0;
    for r in rounds {
        var got: f32[>..] = [];
        guard gpu::values(reads[r], got) else { abort("gpu: a read back is not ready"); }
        for i in 64 {
            var want: f32 = 1000.0 * r + i;
            if i >= 16 && i < 48 { want = 2000.0 * r + (i - 16); }
            if i >= 32 && i < 48 { want = 3000.0 * r + (i - 32); }
            if i < 4 { want = (r * (10 * (i + 1)) % 256) / 255.0; }
            if i == 3 { want = 1.0; }
            if abs(got[i] - want) > 0.0001 {
                if wrong < 4 { print("round ", r, " float ", i, ": ", got[i], " not ", want); }
                wrong++;
            }
        }
    }
    print(gpu::driver(), ": ", rounds, " rounds, ", wrong, " wrong");
    gpu::close();
}
