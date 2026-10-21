// The gpu package (docs/design/gpu.md): bindless GPU work on Metal 4 and
// Vulkan, with shaders written as Goose functions. Names, constants and
// descriptor structs follow gfx's wherever the two models agree (§3): a
// buffer, texture or sampler is a handle shaders read directly, there is
// nothing to bind but pipelines and index buffers, and the program places
// barriers between passes. So far: devices without a window, buffers,
// textures, compute and raster pipelines, passes, fences and read backs.

namespace gpu;

import gpu_shader;

// A buffer or texture, by its index in the resource heap: the value a
// shader reads it through (gpu::StructuredBuffer, gpu::Texture2D, ...) is
// the same index.
struct Buffer { index: u32 }
struct Texture { index: u32 }

// What a shader may do with a buffer.
let BUFFER_READ = 1;
let BUFFER_WRITE = 2;
let BUFFER_INDEX = 4;

// Textures, as gfx's.
let TEXTURE_2D: i32 = 0;
let SAMPLED: i32 = 1;
let COLOR_TARGET: i32 = 2;
let DEPTH_TARGET: i32 = 4;
let STORAGE: i32 = 8;

let RGBA8: i32 = 1;            let BGRA8: i32 = 2;           let RGBA8_SRGB: i32 = 3;
let R8: i32 = 4;               let RG8: i32 = 5;             let RGBA16F: i32 = 6;
let RGBA32F: i32 = 7;          let R16F: i32 = 8;            let RG16F: i32 = 9;
let R32F: i32 = 10;            let RG32F: i32 = 11;          let R32UI: i32 = 12;
let RGBA8UI: i32 = 13;         let RGB10A2: i32 = 14;        let RG11B10F: i32 = 15;
let DEPTH16: i32 = 16;         let DEPTH24: i32 = 17;        let DEPTH32F: i32 = 18;
let DEPTH24_STENCIL8: i32 = 19;
let DEPTH: i32 = 20;

struct TextureDesc {
    kind: i32 = 0,          // TEXTURE_2D
    format: i32 = 1,        // RGBA8
    usage: i32 = 1,         // SAMPLED
    width: i32,
    height: i32,
    depth: i32 = 1,
    mips: i32 = 1,
    samples: i32 = 1,
}

// Pipelines, as gfx's.
let TRIANGLES: i32 = 0;        let TRIANGLE_STRIP: i32 = 1;  let LINES: i32 = 2;
let LINE_STRIP: i32 = 3;       let POINTS: i32 = 4;
let CULL_NONE: i32 = 0;        let CULL_FRONT: i32 = 1;      let CULL_BACK: i32 = 2;
let BLEND_NONE: i32 = 0;       let BLEND_ALPHA: i32 = 1;     let BLEND_ADD: i32 = 2;
let BLEND_PREMULTIPLIED: i32 = 3;
let BLEND_MULTIPLY: i32 = 4;
let COMPARE_NEVER: i32 = 1;    let COMPARE_LESS: i32 = 2;    let COMPARE_EQUAL: i32 = 3;
let COMPARE_LESS_EQUAL: i32 = 4;
let COMPARE_GREATER: i32 = 5;  let COMPARE_NOT_EQUAL: i32 = 6;
let COMPARE_GREATER_EQUAL: i32 = 7;
let COMPARE_ALWAYS: i32 = 8;

struct PipelineDesc {
    primitive: i32 = 0,             // TRIANGLES
    cull: i32 = 0,                  // CULL_NONE
    clockwise: bool = false,        // front faces wind clockwise
    wireframe: bool = false,
    depth_test: bool = false,
    depth_write: bool = false,
    depth_compare: i32 = 2,         // COMPARE_LESS
    blend: i32 = 0,                 // BLEND_NONE
}

struct PassDesc {
    color: Texture[4] = [Texture { 0 }; 4],     // targets, up to the first zero handle
    depth: Texture = Texture { 0 },
    clear_color: bool = true,
    clear_depth: bool = true,
    color_value: float4 = float4 { 0.0, 0.0, 0.0, 1.0 },
    depth_value: f32 = 1.0,
}

// What the GPU last did with a texture, or does next: the two sides of a
// barrier (§3).
let ACCESS_NONE = 0;
let ACCESS_SHADER_READ = 1;
let ACCESS_SHADER_WRITE = 2;
let ACCESS_RENDER_TARGET = 4;
let ACCESS_DEPTH_WRITE = 8;
let ACCESS_DEPTH_READ = 16;
let ACCESS_COPY_SRC = 32;
let ACCESS_COPY_DST = 64;
// What an acceleration structure's build reads: the vertices, indices and
// instances it is built from, once the GPU has written them (§7).
let ACCESS_ACCEL_INPUT = 256;

struct Pipeline { id: u32 }
struct ComputePipeline { id: u32 }

// A read back in flight: a copy taken where it was asked for, ready once
// the submission carrying it completes (§3, Fences and read backs).
struct Readback { id: u32 }

extern "goose_gpu_open" fn open_native(flags: i64) -> bool;
extern "goose_gpu_open_window" fn open_window_native(title: const u8[:], width: i64, height: i64,
                                                     flags: i64) -> bool;
extern "goose_gpu_frame" fn frame() -> bool;
extern "goose_gpu_poll" fn poll();
extern "goose_gpu_quit" fn quit();
extern "goose_gpu_set_title" fn set_title(title: const u8[:]);
extern "goose_gpu_screen" fn screen_native() -> u32;
extern "goose_gpu_screen_depth" fn screen_depth_native() -> u32;
extern "goose_gpu_screen_size" fn screen_size_native(out: i32[:]);
extern "goose_gpu_begin_screen" fn begin_screen_native(r: f32, g: f32, b: f32, a: f32) -> bool;
extern "goose_gpu_time" fn time() -> f64;
extern "goose_gpu_delta_time" fn delta_time() -> f64;
extern "goose_gpu_frame_count" fn frame_count() -> i64;
extern "goose_gpu_key_down" fn key_down(name: const u8[:]) -> bool;
extern "goose_gpu_key_pressed" fn key_pressed(name: const u8[:]) -> bool;
extern "goose_gpu_mouse_position" fn mouse_position_native(out: f32[:]);
extern "goose_gpu_mouse_down" fn mouse_down(button: i64) -> bool;
extern "goose_gpu_close" fn close();
extern "goose_gpu_error" fn error_native(out: u8[>..]&);
extern "goose_gpu_driver" fn driver_native(out: u8[>..]&);
extern "goose_gpu_check" fn check();
extern "goose_gpu_buffer" fn buffer_native(usage: i64, data: const u8[:], size: i64) -> u32;
extern "goose_gpu_buffer_size" fn buffer_size_native(index: u32) -> i64;
extern "goose_gpu_texture" fn texture_native(desc: TextureDesc, data: const u8[:]) -> u32;
extern "goose_gpu_release" fn release_native(index: u32);
extern "goose_gpu_update_buffer" fn update_buffer_native(index: u32, offset: i64,
                                                         data: const u8[:]) -> bool;
extern "goose_gpu_update_texture" fn update_texture_native(index: u32, data: const u8[:]) -> bool;
extern "goose_gpu_compute_pipeline" fn compute_pipeline_native(library: const u8[:]) -> u32;
extern "goose_gpu_pipeline" fn pipeline_native(library: const u8[:], desc: PipelineDesc) -> u32;
extern "goose_gpu_release_pipeline" fn release_pipeline_native(id: u32);
extern "goose_gpu_begin_compute" fn begin_compute() -> bool;
extern "goose_gpu_end_compute" fn end_compute();
extern "goose_gpu_begin_pass" fn begin_pass(desc: PassDesc) -> bool;
extern "goose_gpu_end_pass" fn end_pass();
extern "goose_gpu_bind" fn bind_native(id: u32);
extern "goose_gpu_push_constants" fn push_constants(data: const u8[:]);
extern "goose_gpu_push_uniforms" fn push_uniforms(data: const u8[:]);
extern "goose_gpu_bind_index_buffer" fn bind_index_buffer_native(index: u32, size: i64);
extern "goose_gpu_viewport" fn viewport(x: f32, y: f32, w: f32, h: f32);
extern "goose_gpu_scissor" fn scissor(x: i64, y: i64, w: i64, h: i64);
extern "goose_gpu_draw" fn draw(vertices: i64, instances: i64, first_vertex: i64,
                                first_instance: i64);
extern "goose_gpu_draw_indexed" fn draw_indexed(indices: i64, instances: i64, first_index: i64,
                                                vertex_offset: i64, first_instance: i64);
extern "goose_gpu_dispatch" fn dispatch(x: i64, y: i64, z: i64);
extern "goose_gpu_barrier" fn barrier_native(index: u32, before: i64, after: i64);
extern "goose_gpu_ray_query" fn ray_query() -> bool;
extern "goose_gpu_primitive_accel" fn primitive_accel_native(desc: PrimitiveAccelDesc) -> u32;
extern "goose_gpu_instance_accel" fn instance_accel_native(desc: InstanceAccelDesc) -> u32;
extern "goose_gpu_build" fn build_native(index: u32, instances: i64);
extern "goose_gpu_accel_reference" fn reference_native(index: u32) -> u64;
extern "goose_gpu_submit" fn submit() -> u64;
extern "goose_gpu_fence_done" fn fence_done(f: u64) -> bool;
extern "goose_gpu_wait" fn wait(f: u64);
extern "goose_gpu_read_buffer" fn read_buffer_native(index: u32, offset: i64, size: i64) -> u32;
extern "goose_gpu_read_texture" fn read_texture_native(index: u32) -> u32;
extern "goose_gpu_readback_ready" fn readback_ready_native(id: u32) -> bool;
extern "goose_gpu_readback_bytes" fn readback_bytes_native(id: u32, out: u8[>..]&) -> bool;
extern "goose_gpu_release_readback" fn release_readback_native(id: u32);

// A device with no window, for compute, rendering to textures, and tests.
// GOOSE_GPU_BACKEND=vulkan picks Vulkan over Metal on a Mac.
fn open_headless() -> bool { open_native(0) }

// gfx's window flags.
let WINDOW_RESIZABLE = 1;
let WINDOW_HIDDEN = 2;
let WINDOW_FULLSCREEN = 4;
let NO_VSYNC = 16;

// A window and the device drawing into it, as gfx's (§3): frame() shows
// what was drawn into screen(), and says false once the window is asked to
// close. With GOOSE_GPU_HEADLESS=1 there is no window, and the screen is a
// render target of the size asked for.
fn open(title: const u8[:], width: i64, height: i64) -> bool {
    open_window_native(title, width, height, 0)
}
fn open(title: const u8[:], width: i64, height: i64, flags: i64) -> bool {
    open_window_native(title, width, height, flags)
}

// The screen this frame draws into, in ACCESS_NONE until a barrier or
// begin_screen moves it; frame() shows it from whatever access it is left
// in. Its depth target is the screen's size.
fn screen() -> Texture { Texture { screen_native() } }
fn screen_depth() -> Texture { Texture { screen_depth_native() } }
fn screen_size() -> int2 {
    var wh: i32[2] = [0, 0];
    screen_size_native(wh);
    int2 { wh[0], wh[1] }
}

// A pass over the screen and its depth, both cleared, the barriers that put
// them there placed for the program.
fn begin_screen(clear: float4) -> bool { begin_screen_native(clear.x, clear.y, clear.z, clear.w) }

// In the window's pixels from its top left.
fn mouse_position() -> float2 {
    var xy: f32[2] = [0.0, 0.0];
    mouse_position_native(xy);
    float2 { xy[0], xy[1] }
}

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

// A buffer holding `data`, uploaded before the next submission runs (§3);
// or one of `size` bytes of zeros. An index buffer is a buffer too.
fn buffer(usage: i64, data: const u8[:]) -> Buffer {
    Buffer { buffer_native(usage, data, data.len) }
}
fn buffer(usage: i64, size: i64) -> Buffer { Buffer { buffer_native(usage, "", size) } }
fn index_buffer(data: const u8[:]) -> Buffer { buffer(BUFFER_INDEX, data) }
fn buffer_size(b: Buffer) -> i64 { buffer_size_native(b.index) }
fn release(b: Buffer) { release_native(b.index); }

// A texture, empty, or holding `data` as its mip 0's rows, top first,
// tightly packed: uploaded before the next submission runs, and left in
// ACCESS_SHADER_READ. An empty one is in ACCESS_NONE.
fn texture(desc: TextureDesc) -> Texture { Texture { texture_native(desc, "") } }
fn texture(desc: TextureDesc, data: const u8[:]) -> Texture { Texture { texture_native(desc, data) } }
fn texture2d(width: i64, height: i64) -> Texture {
    texture(TextureDesc { width: width as i32, height: height as i32, .. })
}
fn render_target(width: i64, height: i64, format: i32) -> Texture {
    texture(TextureDesc { format: format, usage: COLOR_TARGET | SAMPLED,
                          width: width as i32, height: height as i32, .. })
}
fn depth_target(width: i64, height: i64) -> Texture {
    texture(TextureDesc { format: DEPTH, usage: DEPTH_TARGET, width: width as i32,
                          height: height as i32, .. })
}
fn release(t: Texture) { release_native(t.index); }

// Updates to a buffer or texture in use (§3): the bytes copied now, and on
// the GPU ahead of the next submission, after the work already submitted.
// So the whole next submission sees them, draws recorded before the call
// included, and one rewritten every frame needs no copies to rotate. A
// texture's update is its whole mip 0; an empty one is in
// ACCESS_SHADER_READ after it.
fn update_buffer(b: Buffer, data: const u8[:]) -> bool { update_buffer_native(b.index, 0, data) }
fn update_buffer(b: Buffer, offset: i64, data: const u8[:]) -> bool {
    update_buffer_native(b.index, offset, data)
}
fn update_texture(t: Texture, data: const u8[:]) -> bool { update_texture_native(t.index, data) }

// The handles a shader takes, from the host's.
fn structured<T>(b: Buffer) -> StructuredBuffer<T> { StructuredBuffer<T> { b.index } }
fn rw_structured<T>(b: Buffer) -> RWStructuredBuffer<T> { RWStructuredBuffer<T> { b.index } }
fn sampled<T>(t: Texture) -> Texture2D<T> { Texture2D<T> { t.index } }
fn storage<T>(t: Texture) -> RWTexture2D<T> { RWTexture2D<T> { t.index } }

// The pipeline running the compute entry point F (docs/design/gpu.md §5.1).
fn compute_pipeline<F>() -> ComputePipeline {
    ComputePipeline { compute_pipeline_native(gpu_entry(F)) }
}
// The pipeline drawing with the vertex entry point V and the fragment entry
// point F, whose interface the compiler has checked. It is made for each
// pass's target formats the first time it is bound in one, as gfx's.
fn pipeline<V, F>() -> Pipeline { Pipeline { pipeline_native(gpu_entry(V, F), PipelineDesc { .. }) } }
fn pipeline<V, F>(desc: PipelineDesc) -> Pipeline {
    Pipeline { pipeline_native(gpu_entry(V, F), desc) }
}
fn release(p: ComputePipeline) { release_pipeline_native(p.id); }
fn release(p: Pipeline) { release_pipeline_native(p.id); }
fn bind(p: ComputePipeline) { bind_native(p.id); }
fn bind(p: Pipeline) { bind_native(p.id); }

// A pass drawing into one color target and, unless it is Texture { 0 }, a
// depth target, both cleared.
fn begin_pass(target: Texture, depth: Texture, clear: float4) -> bool {
    var d = PassDesc { color_value: clear, depth: depth, .. };
    d.color[0] = target;
    begin_pass(d)
}

// Push constants and the uniform buffer, from a value, for the draws and
// dispatches that follow (§3); push_constants and push_uniforms take
// bytes.
fn push_constants<T>(value: T) {
    let one = [value];
    push_constants(bytes_of(one));
}
fn uniforms<T>(value: T) {
    let one = [value];
    push_uniforms(bytes_of(one));
}

fn bind_index_buffer(b: Buffer, index_size: i64) { bind_index_buffer_native(b.index, index_size); }
fn draw(vertices: i64) { draw(vertices, 1, 0, 0); }
fn draw_instanced(vertices: i64, instances: i64) { draw(vertices, instances, 0, 0); }
fn draw_indexed(indices: i64) { draw_indexed(indices, 1, 0, 0, 0); }
fn draw_indexed_instanced(indices: i64, instances: i64) { draw_indexed(indices, instances, 0, 0, 0); }

// The GPU's work with t in access `before` done and visible to its work in
// `after`, and t's layout changed to suit: between passes, never in one.
fn barrier(t: Texture, before: i64, after: i64) { barrier_native(t.index, before, after); }
fn barrier(b: Buffer, before: i64, after: i64) { barrier_native(b.index, before, after); }

// Ray tracing (docs/design/gpu.md §7), where ray_query() says the device has
// it. A primitive acceleration structure holds triangles, positions
// float3s `vertex_stride` bytes apart, indexed or not; an instance one holds
// up to `max` gpu::Instance records from a buffer, which the host or a
// shader writes. Each is built when build() says, in order with the work
// around it; a shader reaches an instance one as a gpu::Accel, through push
// constants or a buffer, and traces it with gpu::trace.
struct PrimitiveAccelDesc {
    vertices: Buffer,
    vertex_offset: i64 = 0,
    vertex_stride: i64 = 12,
    vertex_count: i64,
    indices: Buffer = Buffer { 0 },
    index_size: i64 = 4,
    index_offset: i64 = 0,
    triangle_count: i64,
}

struct InstanceAccelDesc { instances: Buffer, max: i64 }

fn primitive_accel(desc: PrimitiveAccelDesc) -> Accel { Accel { primitive_accel_native(desc) } }
fn instance_accel(desc: InstanceAccelDesc) -> Accel { Accel { instance_accel_native(desc) } }
fn build(a: Accel) { build_native(a.index, 0); }
fn build(a: Accel, instances: i64) { build_native(a.index, instances); }
// What a gpu::Instance names a primitive structure by, its low word first.
fn reference(a: Accel) -> uint2 {
    let r = reference_native(a.index);
    uint2 { (r & 0xffffffff) as u32, (r >> 32) as u32 }
}
fn release(a: Accel) { release_native(a.index); }

// An instance of `a` placed by m (gpu::instance_of, which shaders call).
fn instance(m: mat4, a: Accel, id: u32) -> Instance { instance_of(m, reference(a), id) }

// A copy of `size` bytes of b from `offset`, or of all of t's mip 0, taken
// in order with the work recorded around it, and ready once the submission
// carrying it completes.
fn read_buffer(b: Buffer, offset: i64, size: i64) -> Readback {
    Readback { read_buffer_native(b.index, offset, size) }
}
fn read_texture(t: Texture) -> Readback { Readback { read_texture_native(t.index) } }
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
