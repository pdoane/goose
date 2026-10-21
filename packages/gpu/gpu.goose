// The gpu package (docs/design/gpu.md): bindless GPU work on Metal and
// Vulkan, with shaders written as Goose functions. Names and rules follow
// gfx's wherever the two models agree. This is the first step: a device,
// buffers, compute pipelines, dispatches with push constants, fences and
// read backs. Raster, textures and Direct3D 12 come after (§10).

namespace gpu;

import gpu_shader;

// A buffer, by its index in the resource heap: the value a shader reads it
// through (gpu::StructuredBuffer and friends) is the same index.
struct Buffer { index: u32 }

// What a shader may do with a buffer.
let BUFFER_READ = 1;
let BUFFER_WRITE = 2;

struct ComputePipeline { id: u32 }

// A read back in flight: a copy of part of a buffer, ready once the
// submission carrying it completes (§3, Fences and read backs).
struct Readback { id: u32 }

extern "goose_gpu_open" fn open_native(flags: i64) -> bool;
extern "goose_gpu_close" fn close();
extern "goose_gpu_error" fn error_native(out: u8[>..]&);
extern "goose_gpu_driver" fn driver_native(out: u8[>..]&);
extern "goose_gpu_check" fn check();
extern "goose_gpu_buffer" fn buffer_native(usage: i64, data: const u8[:], size: i64) -> u32;
extern "goose_gpu_buffer_size" fn buffer_size_native(index: u32) -> i64;
extern "goose_gpu_release_buffer" fn release_buffer_native(index: u32);
extern "goose_gpu_compute_pipeline" fn compute_pipeline_native(library: const u8[:]) -> u32;
extern "goose_gpu_release_pipeline" fn release_pipeline_native(id: u32);
extern "goose_gpu_begin_compute" fn begin_compute() -> bool;
extern "goose_gpu_end_compute" fn end_compute();
extern "goose_gpu_bind_compute" fn bind_compute_native(id: u32);
extern "goose_gpu_push_constants" fn push_constants(data: const u8[:]);
extern "goose_gpu_dispatch" fn dispatch(x: i64, y: i64, z: i64);
extern "goose_gpu_submit" fn submit() -> u64;
extern "goose_gpu_fence_done" fn fence_done(f: u64) -> bool;
extern "goose_gpu_wait" fn wait(f: u64);
extern "goose_gpu_read_buffer" fn read_buffer_native(index: u32, offset: i64, size: i64) -> u32;
extern "goose_gpu_readback_ready" fn readback_ready_native(id: u32) -> bool;
extern "goose_gpu_readback_bytes" fn readback_bytes_native(id: u32, out: u8[>..]&) -> bool;
extern "goose_gpu_release_readback" fn release_readback_native(id: u32);

// A device with no window, for compute and tests. GOOSE_GPU_BACKEND=vulkan
// picks Vulkan over Metal on a Mac.
fn open_headless() -> bool { open_native(0) }

// Why the last failing call failed.
fn error() -> u8[>..] {
    var s: u8[>..] = [];
    error_native(s);
    s
}

// "metal" or "vulkan".
fn driver() -> u8[>..] {
    var s: u8[>..] = [];
    driver_native(s);
    s
}

// A buffer holding `data`, uploaded before the next submission runs; or one
// of `size` bytes of zeros.
fn buffer(usage: i64, data: const u8[:]) -> Buffer {
    Buffer { buffer_native(usage, data, data.len) }
}
fn buffer(usage: i64, size: i64) -> Buffer { Buffer { buffer_native(usage, "", size) } }
fn buffer_size(b: Buffer) -> i64 { buffer_size_native(b.index) }
fn release(b: Buffer) { release_buffer_native(b.index); }

// The pipeline running the compute entry point F (docs/design/gpu.md §5.1).
fn compute_pipeline<F>() -> ComputePipeline {
    ComputePipeline { compute_pipeline_native(gpu_entry(F)) }
}
fn release(p: ComputePipeline) { release_pipeline_native(p.id); }
fn bind(p: ComputePipeline) { bind_compute_native(p.id); }

// Push constants from a value, for the dispatches that follow.
fn push_constants<T>(value: T) {
    let one = [value];
    push_constants(bytes_of(one));
}

// A copy of `size` bytes of b from `offset`, taken in order with the work
// recorded around it and ready once the submission carrying it completes.
fn read_buffer(b: Buffer, offset: i64, size: i64) -> Readback {
    Readback { read_buffer_native(b.index, offset, size) }
}
fn ready(r: Readback) -> bool { readback_ready_native(r.id) }
// Appends the bytes to out, once ready; false before then.
fn bytes(r: Readback, out: u8[>..]&) -> bool { readback_bytes_native(r.id, out) }
fn release(r: Readback) { release_readback_native(r.id); }
// Appends the read back's bytes to out as elements of its flat T, once
// ready: gpu::values(r, floats) into an f32[>..]. False before then.
fn values<T>(r: Readback, out: T[>..]&) -> bool {
    var raw: u8[>..] = [];
    guard bytes(r, raw) else { return false; }
    // Through from_bytes' image format: a LEB128 byte count, then the
    // elements, as gfx reads its buffers back.
    var image: u8[>..] = [];
    var n = raw.len as u64;
    loop {
        let low = (n & 127) as u8;
        n >>= 7;
        if n == 0 {
            image.push(low);
            break;
        }
        image.push(low | 128);
    }
    image.append(raw);
    let got, ok = from_bytes<T[>..]>(image);
    guard ok else { return false; }
    for v in got { out.push(v); }
    true
}
