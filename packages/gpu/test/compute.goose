// The gpu package's first test: a compute shader scales a buffer, and the
// result is read back and compared with the same function run on the CPU.
import gpu from "..";
import std;

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
    gpu::store(p.output, i, scaled(gpu::load(p.input, i), p.scale));
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let n: u32 = 1000;
    var input: f32[>..] = [];
    for i in n { input.push(i * 0.25); }
    let src = gpu::buffer(gpu::BUFFER_READ, bytes_of(input));
    let dst = gpu::buffer(gpu::BUFFER_WRITE, n as i64 * 4);
    let p = gpu::compute_pipeline(scale_kernel);
    guard p.id != 0 else { abort(str("gpu: ", gpu::error())); }

    gpu::begin_compute();
    gpu::bind(p);
    gpu::push_constants(Push { input: gpu::StructuredBuffer<f32> { src.index },
                               output: gpu::RWStructuredBuffer<f32> { dst.index },
                               n: n, scale: 3.0 });
    gpu::dispatch((n as i64 + 63) / 64, 1, 1);
    gpu::end_compute();
    let r = gpu::read_buffer(dst, 0, n as i64 * 4);
    gpu::wait(gpu::submit());

    var got: f32[>..] = [];
    guard gpu::values(r, got) else { abort("gpu: the read back is not ready"); }
    var wrong = 0;
    for i in n { if got[i] != scaled(input[i], 3.0) { wrong++; } }
    print(gpu::driver(), ": ", n, " values, ", wrong, " wrong");
    gpu::close();
}
