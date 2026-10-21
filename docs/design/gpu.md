# A modern GPU stack: the `gpu` package and GPU functions

<details>
<summary>Table of contents</summary>

- [1. Why](#1-why)
- [2. Decisions](#2-decisions)
- [3. The device model](#3-the-device-model)
- [4. The binding ABI](#4-the-binding-abi)
- [5. GPU functions](#5-gpu-functions)
- [6. Compiling shaders](#6-compiling-shaders)
- [7. Ray tracing](#7-ray-tracing)
- [8. Relationship to gfx](#8-relationship-to-gfx)
- [9. Testing](#9-testing)
- [10. The plan](#10-the-plan)
- [11. Open questions](#11-open-questions)

</details>

This proposal adds a second graphics package, `gpu`, built in the shape
modern engines have settled on: bindless resources, shader vertex fetch,
explicit barriers, monolithic pipelines and inline ray queries, on Metal 4,
Vulkan and Direct3D 12. Its shaders are not GLSL. They are Goose functions,
a restricted subset the compiler checks and emits as HLSL and MSL, then
compiles to DXIL, SPIR-V and `.metallib` with each platform's own compiler.
A struct is declared once and has one layout on the CPU and the GPU, and the
data a pipeline takes is a type the checker holds the host to. Nothing here
is implemented; this note is the contract the implementation is built
against, and §10 is the order it lands in.

The package is opinionated: one fixed binding ABI per backend, and no
root-signature or descriptor-set configuration exposed to anyone. What it
replaces, in an engine written in C++, is usually the same shaders kept by
hand in MSL and HLSL behind a header of shims, with the rules that keep the
C++, MSL and HLSL layouts of a shared struct equal written down as
convention and checked, if at all, by a script. A compiler that sees both
sides of a struct can make those rules unnecessary instead.

## 1. Why

`gfx` (`docs/stdlib.md` §gfx) is SDL_GPU with shaders in GLSL 450, compiled
by a vendored cute_spirv into SPIR-V, MSL and HLSL text. It is a good small
API and stays (§8). What it cannot become:

* **Binding is per slot.** Resources are bound into SDL_GPU's fixed sets
  before each draw (`bind_texture(stage, slot, t, s)`), so a shader reaches
  only what a call site bound, and a material is a binding sequence instead
  of data. GPU-driven rendering wants every resource reachable by index.
* **Vertex input layouts.** A pipeline's vertex format is fixed function,
  and the instance stream is a second binding with its own layout.
* **The host-shader contract is a size.** A uniform struct of the wrong
  size is a misuse found at run time; one of the right size and the wrong
  fields is not found at all. std140 pads where Goose packs, so the program
  pads by hand.
* **No ray tracing**, and none coming through SDL_GPU.
* **Text is the last stage.** HLSL and MSL leave the compiler as text, and
  each backend compiles them when a pipeline is made. On Windows the product
  should be DXIL, and on Apple a `.metallib`.
* **The shading language is someone else's subset.** cute_spirv's GLSL has
  no instance names on uniform blocks and one runtime array per storage
  buffer, and its MSL and HLSL are a transpiler's.

## 2. Decisions

* **A new package, `gpu`, beside `gfx`.** Existing programs keep working;
  `gfx` is not reshaped into something else.
* **Bindless.** Every texture, buffer, sampler and acceleration structure
  is a `u32` index into one resource heap or the sampler heap. Creating a
  resource makes it visible to every shader. There is no bind call, and no
  descriptor-set, root-signature or argument-buffer layout anywhere in the
  API: the package fixes one ABI per backend (§4).
* **Two data paths.** Push constants (at most 128 bytes) and one uniform
  buffer (at most 64 KiB), each set until it is set again. Everything else is
  reached through handles, including a per-frame stream buffer for data
  written each frame.
* **Shader vertex fetch.** No vertex input layouts. A vertex shader reads
  its vertices from buffers by `vertex_index()`. Index buffers are the one
  bound buffer, because index fetch drives the post-transform cache.
* **Monolithic pipelines**, with only viewport, scissor, blend color and
  stencil reference dynamic: the Direct3D 12 baseline, which maps onto every
  backend without emulation.
* **Explicit barriers in terms of access** (`ACCESS_SHADER_READ`,
  `ACCESS_RENDER_TARGET`, ...), never layouts or resource states, which the
  backend derives. A frame graph that derives the barriers is a library on
  top (§10, phase 5), not part of the device layer.
* **Managed uploads, fenced read backs.** Uploads are staged by the
  package and complete before the next command buffer submitted, with
  every barrier they need recorded by the package; per-frame data goes in
  the stream instead (§3). A program's barriers cover only what its own GPU
  work writes. Nothing waits for the GPU unless the program asks to: a
  submission returns a fence, and a read back is ready when its fence is.
* **Deferred destruction.** `release` is safe at once; the backend frees
  when in-flight work no longer uses the resource.
* **Inline ray queries only.** Acceleration structures and a `trace` call
  usable from any stage. No ray tracing pipelines, no shader binding tables,
  no callable shaders (§7): the portable case, with no API surface for the
  rest.
* **Shaders are Goose functions.** An entry point is a function declared
  `vertex fn`, `fragment fn` or `compute(x, y, z) fn`, and what it calls
  must stay in the GPU subset (§5). The compiler emits HLSL and MSL and
  compiles them to the final binaries when the program is compiled (§6).
* **Backends in order: Metal 4 and Vulkan, then Direct3D 12.** Both of the
  first run on one Mac, Vulkan through KosmicKrisp, so Vulkan's validation
  layers are a test oracle from the start. KosmicKrisp may lack ray queries;
  ray tracing is a capability, so that is a device without it, not a
  port without it (§7).
* **External compilers are the platform's.** DXC (DXIL and SPIR-V) on every
  platform, the Metal toolchain (`xcrun metal`) on Apple. The compiler finds
  them as it finds the C compiler (`goose toolchain`).

Not in scope: geometry and tessellation stages, mesh shaders and multiple
queues (later, each as its own capability), ray tracing pipelines, and
programs as data, interpreted by a VM on the CPU or the GPU.

## 3. The device model

`gpu` follows `gfx` wherever the two models agree: the same names, the same
descriptor structs, the same frame loop and the same error rules, so moving
a program from one to the other changes its binding and its shaders, not
its shape. Where `gfx` binds, `gpu` passes handles; where SDL_GPU tracks
hazards, `gpu` asks for barriers. Everything is in namespace `gpu`.
Signatures below are illustrative; the package's own documentation is
normative once it exists.

### Device, window and frames

As `gfx`'s, by name and meaning:

```goose
fn open(title: const u8[:], width: i64, height: i64) -> bool        // and (..., flags)
fn open_headless(width: i64, height: i64) -> bool
fn close()     fn frame() -> bool     fn poll()     fn quit()     fn flush()
fn error() -> u8[>..]     fn check()     fn set_title(title: const u8[:])
fn driver() -> u8[>..]     // "metal", "vulkan" or "direct3d12"
fn screen() -> Texture     fn screen_depth() -> Texture     fn screen_size() -> int2
fn time() -> f64           fn delta_time() -> f64           fn frame_count() -> i64
fn caps() -> Caps          // new: what the device offers beyond the baseline
```

`open` takes `gfx`'s flags, and `GOOSE_GPU_BACKEND=vulkan` picks Vulkan
over Metal on a Mac, as `GOOSE_GFX_HEADLESS=1` picks a headless device.
Input is `gfx`'s too: the same functions under the same names, from code the
two packages share in one repository, as `gfx`, `audio` and `ui` already do.
Clip space is `gfx`'s (y up, depth 0 to 1, texture coordinates from the top
left), and `mat4` and its helpers are the same type.

### Resources and handles

```goose
struct Buffer { index: u32 }      struct Texture { index: u32 }
struct Sampler { index: u32 }     struct Accel { index: u32 }

fn buffer(usage: i64, data: const u8[:]) -> Buffer       // and (usage, size): zeros
fn index_buffer(data: const u8[:]) -> Buffer
fn update_buffer(b: Buffer, data: const u8[:]) -> bool   // and (b, offset, data)
fn read_buffer<T>(b: Buffer, out: T[>..]&, count: i64) -> bool
fn buffer_size(b: Buffer) -> i64

fn texture(desc: TextureDesc) -> Texture                 // and (desc, data)
fn texture2d(width: i64, height: i64) -> Texture         fn render_target(...)   fn depth_target(...)
fn load_texture(path: const u8[:], flags: i64) -> Texture
fn update_texture(t: Texture, data: const u8[:]) -> bool // and (t, region, data)
fn read_texture<T>(t: Texture, out: T[>..]&) -> bool     fn read_pixels(t: Texture) -> u8[>..]
fn save_png(t: Texture, path: const u8[:]) -> bool       fn screenshot(path: const u8[:]) -> bool
fn generate_mips(t: Texture)                             fn texture_info(t: Texture) -> TextureDesc

fn sampler(desc: SamplerDesc) -> Sampler                 fn sampler(filter: i32, wrap: i32) -> Sampler
fn release(b: Buffer)                                    // and every other kind
```

`TextureDesc`, `SamplerDesc` and `Region` are `gfx`'s structs, with its
kinds, formats, filters and wraps. Buffer usage is what a shader may do:
`BUFFER_READ`, `BUFFER_WRITE`, `BUFFER_INDEX`, `BUFFER_INDIRECT` and
`BUFFER_ACCEL_INPUT` (§7). `gfx`'s `BUFFER_VERTEX` has no counterpart,
since vertices are a buffer a vertex shader reads.

A handle is its heap index, so the value the host holds is the value a
shader reads: a struct of handles written into a buffer is a material table
with nothing to translate. `release` is safe at once; the handle's slot and
memory are reused only after the frames in flight that might reach it.
Index 0 of the resource heap is the stream buffer, and the sampler heap
starts with a fixed set (`gpu::POINT`, `gpu::LINEAR`, `gpu::LINEAR_CLAMP`,
`gpu::ANISO`, `gpu::SHADOW`, ...), so most programs make no sampler.

Shaders see typed views of these handles (§5.4): `Texture2D<float4>` is a
`Texture` a shader may sample and get a `float4` from. The host converts with
a checked call (`gpu::view<Texture2D<float4>>(t)`) that tests the texture's
kind and format once, where the view is made, rather than each time a
shader uses it.

### Getting data to the GPU

Two paths, by how often the data changes.

**Uploads** (`buffer(usage, data)`, `texture(desc, data)`, `load_texture`,
`update_buffer`, `update_texture`) are managed. The package copies the data
into staging memory at the call and records the copy into an upload
command buffer of its own, which it submits ahead of the next command buffer
the program submits, so an upload is complete before any work submitted
after it runs, and work submitted before it stays ordered ahead of it. Both
come from the barriers the package records around the copy (below); neither
needs a mechanism of its own.

The package owns every barrier an upload needs, so none of them shows up
in the program's. A resource has a **steady access** implied by its usage
(`BUFFER_READ` and a sampled texture: shader reads; an index buffer: index
reads), and the package brackets each copy with the transitions from it
to copy destination and back. Several uploads to one resource before a
submission, overlapping or not, are ordered with a write-after-write
barrier between copies to the same destination, which a model that assumes
one upload per resource per frame misses. A resource that shaders write has
no steady access, so an update to it takes the accesses before and after
from the program, which tracks that resource's barriers anyway.

An update is visible to the whole of the next submission, including draws
recorded before the call but not yet submitted, where in `gfx` it lands
between the passes around it. That is deliberate, and it is what makes
uploads both easy and fast. The only promise is completion before the next
submission, so the package may start a batch as early as it likes, when
its staging memory fills or at once for a large upload, and it manages
staging memory for every upload in the program in one place. Recording
copies into the program's own command buffers, ordered with its passes,
would hand each program the staging buffers to manage, which most would
manage badly. A frame that needs a resource's old and new contents in one
submission submits between them, or puts the data in the stream.

**The stream** carries data that is written wholesale every frame:
`stream(value)` and `stream(values: const T[:])` write into the current
frame's slice of a host-visible ring and return the byte offset to pass in
push constants. There is no copy and no barrier. Running out of stream is a
misuse.

A dedicated copy queue, for large uploads that should not delay the frame
they are issued in, is deferred until a program needs it (§11).

### Fences and read backs

Nothing waits for the GPU unless the program asks it to:

```goose
fn submit() -> u64                   // the frame's work so far; and submit(cb)
fn fence_done(f: u64) -> bool        // never waits
fn wait(f: u64)                      // waits; for tools, tests and shutdown

struct Readback { id: u32 }
fn read_buffer(b: Buffer, offset: i64, size: i64) -> Readback
fn read_texture(t: Texture, region: Region) -> Readback
fn ready(r: Readback) -> bool        fn bytes(r: Readback) -> const u8[:]   // once ready
fn release(r: Readback)
```

Submitting returns a fence value, and the fences of one queue complete in
order. A read back records a copy into host-visible memory the package owns
and is ready once the submission carrying it completes; `bytes` before then
is a misuse. `save_png` and `screenshot` take the same route and write their
file when the pixels arrive. `gfx`'s read backs, which wait, have no
counterpart: a test that wants one writes `wait(submit())` before `bytes`.

### Passes, pipelines and drawing

```goose
fn pipeline(vs: vertex fn, fs: fragment fn) -> Pipeline<P, U>   // and (vs, fs, desc)
fn compute_pipeline(cs: compute fn) -> ComputePipeline<P, U>

fn begin_pass(desc: PassDesc) -> bool      fn begin_screen(clear: float4) -> bool
fn begin_pass(target: Texture, depth: Texture, clear: float4) -> bool
fn end_pass()
fn bind(p: Pipeline<P, U>)
fn uniforms<T>(value: T)                   // the uniform buffer, as gfx's
fn push_uniforms(data: const u8[:])        // the same, as bytes
fn push_constants<T>(value: T)             // push constants; and (data: const u8[:])
fn bind_index_buffer(b: Buffer, index_size: i64)
fn viewport(x: f32, y: f32, w: f32, h: f32)     fn scissor(x: i64, y: i64, w: i64, h: i64)
fn draw(vertices: i64)                     // and gfx's other draw forms
fn draw_indirect(args: Buffer, offset: i64)     fn draw_indexed_indirect(args: Buffer, offset: i64)

fn begin_compute() -> bool                 fn end_compute()
fn bind(p: ComputePipeline<P, U>)         // uniforms and push constants as above
fn dispatch(x: i64, y: i64, z: i64)        fn dispatch_indirect(args: Buffer, offset: i64)

fn barrier(b: Buffer, before: i64, after: i64)      // ACCESS_SHADER_READ, ACCESS_RENDER_TARGET, ...
fn barrier(t: Texture, before: i64, after: i64)     // and (t, mip, layer, before, after)
```

`PassDesc` and `PipelineDesc` are `gfx`'s, without the vertex-format
fields (`instance_location`, `formats`). A pipeline is made for each pass's
target formats the first time it is bound in one, as in `gfx`; giving
`PipelineDesc.targets` makes it ahead of time, where a frame cannot hitch
on it.

`bind`, `uniforms` and `push_uniforms` are `gfx`'s without the stage and
slot, since a pipeline has one uniform block shared by its stages: they set
the uniform buffer. `push_constants` is new and sets the push constants,
which `gfx` has no counterpart for. Both persist across draws until set
again, and a draw checks each against the bound pipeline's types (§5.1):
set from a value, the type must be that type; set as bytes, the size must
be its size. Either mismatch is a misuse. A compute
pipeline's group size comes from its shader, so `dispatch` takes group
counts and nothing restates the size. `begin_compute` takes no list of
written resources, as `gfx`'s does, since nothing is bound.

`gfx`'s `bind_vertex_buffer`, `bind_texture`, `bind_storage_texture` and
`bind_storage_buffer` have no counterpart.
`barrier` is new: SDL_GPU tracks hazards by watching what is bound, and with
nothing bound there is nothing for the package to watch. A barrier names a
resource the program's own GPU work writes, a render target or a shader-
written buffer, and the accesses before and after; uploads (above) need
none. The frame graph of §10 derives them for programs with many passes.

### Presentation

`open` makes the swap chain for its window, and `frame()` presents it, with
vsync unless `open` was given `NO_VSYNC`, as in `gfx`. `caps().hdr_output`
says whether the display takes HDR, and the screen's format and color space
are queries a tone mapper reads rather than a choice the program makes.

Presenting works from any thread on Metal, and on Vulkan and Direct3D 12
as long as the main thread keeps pumping the window's events, which DXGI
needs for mode changes. So the frame's thread is the program's choice
(below).

On macOS a live resize redraws smoothly only if the frame presents inside
the Core Animation transaction that resizes the window, which is
`CAMetalLayer.presentsWithTransaction`. Setting the property is easy, since
the Metal backend is the package's own and SDL hands it the layer
(`SDL_Metal_GetLayer`). Honoring it is not: the frame has to be drawn,
committed, scheduled and presented on the main thread, during AppKit's
resize, while SDL owns the view and the run loop. That ties the program's
draw loop to the window system's, which no other part of the design does.
It is deferred. A route to try first is SDL3's `SDL_EVENT_WINDOW_EXPOSED`,
which an event watch receives during a live resize, but whether that runs
inside the transaction is unverified (§11).

### Command buffers and threads

Only the window needs the main thread: SDL, and macOS beneath it, create
windows and deliver events there. So `open`, `close`, `set_title`, `poll`,
which pumps events, and the input functions are main-thread functions, and
the rest of `gpu` is not. A program may record, submit and present from any
thread.

The recording functions above record into the frame's own command buffer,
on whichever thread drives the frame. `frame()` submits it, presents and
advances; called on the main thread it also polls, so the single-threaded
loop is `gfx`'s unchanged. A program that renders on another thread calls
`poll()` in its main thread's loop and `frame()` on the render thread.

More threads record into command buffers of their own: `commands()`
returns one, every recording function has a form taking it first, so the
calls read `cb.begin_pass(desc)`, `cb.bind(p)`, `cb.draw(36)` by UFCS, and
`submit(cb)` queues it, ordered by submission and returning its fence. A
command buffer is recorded by one thread at a time.

### Errors

As in `gfx`: a call that fails for reasons outside the program returns
false or a zero handle and `error()` says why; a call the program should not
have made (a released handle, push constants of another type, a stream
overflow, `bytes` before a read back is ready) is printed, skipped, and
aborts the program at the next `frame()`, `check()`, `wait` or `close()`. A fault inside a shader is reported the
same way (§5.6).

### Capabilities

`caps()` answers what the device offers beyond the baseline: `ray_query`,
`float64`, `float16_ops`, `timestamps`, `hdr_output`, and so on. A pipeline
whose entry points need a capability the device lacks fails to be made,
with an error naming the capability and the line that needs it (§5.5).

## 4. The binding ABI

One ABI per backend, fixed by the package and emitted by the compiler, so
the two can never disagree and no program configures it.

| | Push constants (≤ 128 B) | Uniform buffer (≤ 64 KiB) | Resource heap | Sampler heap |
|---|---|---|---|---|
| Direct3D 12 | root constants, `b0` | root CBV, `b1` | SM 6.6 `ResourceDescriptorHeap` | `SamplerDescriptorHeap` |
| Vulkan | push constant range | push descriptor, set 1 | set 0 binding 0: one `VK_EXT_mutable_descriptor_type` array | set 0 binding 1 |
| Metal 4 | argument table buffer 0 (bytes) | argument table buffer 1 | an argument buffer of resource IDs, buffer 2 | an argument buffer, buffer 3 |

The Vulkan row asks for Vulkan 1.2 with
descriptor indexing, `VK_EXT_mutable_descriptor_type`,
`VK_KHR_push_descriptor`, `VK_KHR_dynamic_rendering`,
`VK_KHR_synchronization2`, buffer device address and
`VK_KHR_maintenance4/5`, plus `scalarBlockLayout` (§5.3). The emitted HLSL
declares one typed array per resource kind at the heap's binding, which
mutable descriptors allow, so the SPIR-V DXC produces indexes the same heap
the Direct3D 12 shader does.

A program never sees this table. It exists so the package and the emitters
have one thing to agree on, and so a backend's capture tools show what a
programmer expects.

## 5. GPU functions

### 5.1 Entry points

```goose
import gpu;

struct Vertex { pos: float3, normal: float3, uv: float2 }

struct DrawPush {                       // 16 bytes of push constants
    vertices: gpu::StructuredBuffer<Vertex>,
    albedo: gpu::Texture2D<float4>,
    object: u32,
    pad 4,
}

struct Scene { view_proj: mat4, sun: float3, exposure: f32 }

struct Varyings { normal: float3, uv: float2 }

vertex fn mesh_vs(p: DrawPush, scene: const Scene&) -> float4, Varyings {
    let v = p.vertices.load(gpu::vertex_index());
    let clip = mul(scene.view_proj, float4 { v.pos.x, v.pos.y, v.pos.z, 1.0 });
    return clip, Varyings { normal: v.normal, uv: v.uv };
}

fragment fn mesh_fs(in: Varyings, p: DrawPush, scene: const Scene&) -> float4 {
    let base = p.albedo.sample(gpu::LINEAR, in.uv);
    let lit = max(dot(normalize(in.normal), scene.sun), 0.0);
    return base * (lit * scene.exposure);
}

compute(8, 8, 1) fn blur(p: BlurPush) {
    let id = gpu::dispatch_thread_id();
    ...
}
```

`vertex`, `fragment` and `compute(x, y, z)` are function modifiers, keywords
only directly before `fn`, like `inline` (spec §7.13). The group size is
three positive integer constants, `compute(64)` meaning `compute(64, 1, 1)`.

An entry point's parameters, in order:

1. **The stage input**, for a fragment entry only: the type its vertex
   stage returns after the clip position. It is required, an empty struct
   when there is nothing to pass.
2. **Push constants**: a value of a struct type, at most 128 bytes.
3. **Uniform data**: a `const T&`, since it is read through memory and
   never written.

Each but the first is optional. A vertex entry returns the clip position
as a `float4`, then optionally its varyings struct; a fragment entry
returns one to eight color values, which may be different vector types.
A compute entry returns nothing. System values are calls rather than
parameters (§5.5).

The host makes pipelines from entry points as `thread_spawn` takes a
`thread_fn`: by name, a compile-time function value (spec §7.6).

```goose
let p = gpu::pipeline(mesh_vs, mesh_fs, PipelineDesc { depth_test: true, .. });
let b = gpu::compute_pipeline(blur);
```

`gpu::pipeline` checks at compile time that the fragment input is the vertex
output and that both stages take the same push and uniform types, or that
one takes none. Its result's type carries those types,
`gpu::Pipeline<DrawPush, Scene>`, which is what a draw checks the pushed
data against (§3). Naming an entry point in one of these calls is what compiles it: an
entry point no call names is checked but not compiled.

Compiling the stage interface: integer varyings are flat, float
ones perspective-correct, and the clip position is invariant (Metal
`[[position, invariant]]`, SPIR-V and HLSL `precise` or `Invariant`), so
that every pipeline computing a position with the same expression gets the
same bits and a depth prepass works. An opt-out is deferred until a program
needs one.

### 5.2 The subset

A GPU entry point and everything it calls are compiled as a separate
program, the way a `thread_fn` is (spec §11.2), and checked as one: a pass
over the entry point's call graph like `CheckThreadGlobals`, reporting the
first construct outside the subset at its line together with the call chain
from the entry point. Any ordinary function may be called from an entry
point; it is a GPU function for that use only if it stays inside:

* **Scalars**: `i32`, `u32`, `f32`, `f16` (§5.8) and `bool`. Integer
  literals adapt as everywhere else; `i64` arithmetic that literals and
  `.len` would produce is an error at the line that makes it, with the type
  to write instead. `f64` is a capability, not part of the baseline: MSL
  has no doubles, so an entry point that uses one is a compile error in a
  program built for an Apple target, and needs `caps().float64` elsewhere.
* **Vectors and matrices**: `vec2/3/4<T>` of those scalars (`float3`,
  `int2`, and new `uint2/3/4` aliases in `vec`), and `mat4`, `mat3`,
  `float3x4`, column-major as `gfx`'s `mat4` is. Swizzles are what `vec`
  offers (`xy`, `xyz`); a general swizzle syntax is an open question (§11).
* **Structs**, **fixed arrays** and **fixed-mode ADTs** of those. `match` on
  a fixed-mode ADT compiles to a `switch`.
* **Locals** (`let`, `var`), `if`, `match`, `for` over a range or a fixed
  array, `while`, `loop`, `break`, `continue`, `return`, and multiple
  returns.
* **Calls** to other GPU functions, including generic ones (monomorphized as
  everywhere), overloads, UFCS, and static function values passed to them.
* **Globals**: `let` and `const` globals with constant initializers, emitted
  as constants.

Outside it: references and slices other than an entry point's uniform
parameter; every array with a length that is not fixed; strings;
optionals; varints; relative references; `var` globals; recursion; `extern
fn`; and the runtime's services (allocation, printing). `abort` is allowed
and faults (§5.6), its message kept only when it is a constant string.
`i8`, `u8`, `i16` and `u16` follow `f16` (§5.8); `i64` and `u64` are
outside it for now.

### 5.3 Shared data and layout

Goose structs are packed (spec §3.2): `float3` is 12 bytes, and a struct is
its fields in order with nothing between them but `pad`. On the GPU this is
the **only** layout. Every buffer element, push struct and uniform struct is
read with the struct's Goose layout, so a value the host writes with
`bytes_of` is the value a shader loads, with no mirror header and no
layout check outside the compiler.

What makes that true per backend:

* **Buffers** are loaded at Goose offsets: `ByteAddressBuffer.Load<T>` in
  HLSL (scalar layout in DXC's SPIR-V, `-fvk-use-scalar-layout`), `packed_`
  vector types and byte offsets in MSL.
* **Push constants and uniform data** are declared as the struct itself wherever the
  backend reads packed layout: MSL, with the same `packed_` types as for
  buffers, and SPIR-V, compiled with scalar block layout
  (`scalarBlockLayout`). Direct3D 12 is the exception. Its constant buffers
  keep the old packing, in 16-byte registers that a vector may not
  straddle, where arrays and nested structs start a register. For DXIL
  only, and only for a struct whose packing would move a field from its
  Goose offset, the emitter **flattens**: it declares the constant buffer as
  an array of `uint4` registers and reads each field at its Goose offset,
  reassembling one that straddles registers, into a local of the struct's
  type at the start of the entry point. A struct whose packing already
  agrees, which most do, is declared as itself on every backend. Nothing is
  rejected, and no program pads for the GPU's sake.

`bool` is a 1-byte value on the CPU and has no GPU storage size: a `bool` is
fine in a local and rejected in a struct that crosses, as are the scalars
§5.2 excludes. A field of a handle type crosses as its `u32`.

### 5.4 Resources

Typed views are generic structs over a `u32`:

```goose
struct StructuredBuffer<T> { index: u32 }      // elements of T, in Goose layout
struct RWStructuredBuffer<T> { index: u32 }
struct Texture2D<T> { index: u32 }             // T: f32, float2, float4, int4, uint4, ...
struct RWTexture2D<T> { index: u32 }
// Texture2DArray, Texture3D, TextureCube, TextureCubeArray, DepthTexture2D, and RW forms
```

Operations are functions, called with UFCS:

```goose
let c = tex.sample(gpu::LINEAR, uv);              // implicit LOD: fragment only
let c = tex.sample_level(gpu::LINEAR, uv, 2.0);   // any stage
let s = shadow.sample_compare(gpu::SHADOW, uv, depth);
let t = tex.load(int2 { x, y }, 0);               // texel fetch, no sampler
out.store(int2 { x, y }, value);                  // RW only
let v = verts.load(i);  verts_rw.store(i, v);
let slot = counter.atomic_add(0, 1);              // RWStructuredBuffer<u32>; the old value
let w = tex.width();  let n = tex.mips();
```

`ByteBuffer.load<T>(offset)` and `RWByteBuffer.store(offset, value)` read and
write a record of any GPU struct at a byte offset, for heterogeneous streams.

A handle that differs across a wave is wrapped where it is used:
`gpu::nonuniform(material.albedo).sample(...)`. Without it an access is
uniform, which is the backends' rule, and getting it wrong is undefined on
every one of them. Push and uniform handles are uniform by construction,
so the common case needs nothing, and the checker rejects `nonuniform`
anywhere but around a handle that is then used.

### 5.5 Stage capabilities

Some calls exist only in some stages or on some devices, and the checker
holds each to its entry point's stage across the call graph:

| Call | Where |
|---|---|
| `vertex_index()`, `instance_index()` | vertex |
| `frag_coord()`, `front_facing()`, `discard()`, `ddx`/`ddy`/`fwidth`, `sample` | fragment |
| `set_depth(d)` | fragment; the pipeline then writes depth |
| `dispatch_thread_id()`, `group_id()`, `group_thread_id()`, `group_index()` | compute |
| `trace`, `trace_any` (§7) | any; needs `caps().ray_query` |

A function that uses a fragment-only call becomes fragment-only, and a call
to it from a compute entry is an error at the call, naming the function and
the line inside it that made it so. A capability call marks the entry
points that reach it in the shader library (§6), which is how pipeline
creation knows to refuse them on a device without it.

Group-shared memory, group barriers, wave operations and the rest of the
atomics arrive later as their own capabilities, compute-only where the
backends require it.

### 5.6 Checks that abort

Goose aborts on an integer division by zero, an out-of-range index, and, in
debug builds, signed overflow (spec §6.2). A shader cannot abort, so it
**faults**: it writes the entry point and source line to a small fault
buffer the package owns (the first fault per frame wins), and then stops
doing harm: a compute invocation returns, a fragment is discarded, a vertex
is moved outside the clip volume. The host reads the buffer at the frame
boundary and at fence waits, and aborts with `fault in blur
(blur.goose:41): index 9 out of range for a length of 8`, the way a misuse
aborts.

Which checks are emitted:

* **Division by zero** and **local fixed-array indexing**: always, as on
  the CPU. Bounds-check elimination (spec §10.5) runs on GPU functions as
  on any others, so most index checks disappear; `checkfree` works too.
* **Signed overflow**: debug builds only, as on the CPU. Release wraps.
* **Resource access out of range**: debug builds only, by querying the
  resource's size. Release relies on the backend: Direct3D 12 and Vulkan
  (`robustBufferAccess`) read zeros and drop writes, and Metal promises
  nothing. §11 keeps this open.

Shift counts are masked to the width, as Goose defines them; the emitters
mask explicitly where MSL and SPIR-V leave it undefined. `%` is Euclidean,
emitted as a helper.

### 5.7 Numerics

Integer results are bit-exact with the same function on the CPU. For floats
the default is **strict**: no contraction into fused
multiply-adds and no reassociation, which means `-fno-fast-math` for Metal,
whose compiler defaults to fast math, and `precise` where DXC would contract.
Basic operations then round as on the CPU, apart from denormals, which GPUs
may flush. Transcendentals (`sin`, `exp`, `pow`, ...) are each backend's
and differ in the last bits.

Fast math is an option: `goose build --gpu-math fast`, or `gpu_math =
"fast";` in the program's sidecar, lets the backends contract and
reassociate and turns on Metal's fast math, for every entry point. A
per-entry choice is the likely next step (§11). The differential tests
(§9) build strict.

### 5.8 f16

Goose has no `f16` today, and shaders need one: half the bandwidth and
registers for colors, normals and much of lighting, and native arithmetic on
most GPUs. It is a language change, not a GPU one, and lands in the spec
(§3.1) and the C backend before GPU functions use it:

* **The type.** IEEE binary16, usable everywhere `f32` is: fields, array
  elements, locals, parameters and returns. Every operation at `f16` rounds
  once to `f16`. `f16` widens to `f32` and `f64` implicitly, never the
  reverse (`x as f16` rounds to nearest even). Float literals adapt to it as
  to the other float types, and integers convert to it as to them, rounding,
  with values beyond its range becoming infinities.
* **On the CPU.** C has no portable half type (MSVC and TinyCC have no
  `_Float16`), so the C backend stores an `f16` as a `uint16_t` and computes
  each operation in `float`, rounding the result back. That is exact, not
  an approximation: `f32`'s 24-bit significand is at least 2 × 11 + 2 bits,
  so for `+`, `-`, `*`, `/` and `sqrt` rounding the `f32` result to `f16`
  gives the correctly rounded `f16` result. Conversions use F16C or ARMv8
  instructions where the C compiler allows, and a runtime routine
  otherwise; a C compiler with `_Float16` may use it directly.
* **On the GPU.** Where the device has native 16-bit arithmetic
  (`caps().float16_ops`: always on Metal; DXC's `-enable-16bit-types` on
  Shader Model 6.2 and later; `shaderFloat16` with 16-bit storage on
  Vulkan), `f16` is `half` and computes natively. Elsewhere the emitter
  computes in `f32` and rounds to `f16` after each operation, which gives
  the same bits more slowly, and reads 16-bit fields with 32-bit loads it
  unpacks. A function's `f16` results are then the same on the CPU and every
  GPU.

`i8`, `u8`, `i16` and `u16` take the same route afterwards: storage in
buffers and push constants first (colors as four `u8`s), arithmetic at their
width, masked where the hardware computes wider.

## 6. Compiling shaders

The checker collects the entry points that pipeline calls name, checks each
call graph against §5, and hands the emitters a checked, monomorphized body
per function, just as it hands the C backend one.

**Emitters.** Two new backends, `src/gpu_hlsl.h` and `src/gpu_msl.h`,
emit one translation unit per pipeline stage. Both keep the function
structure, the control flow and the names of the Goose source, so the shader
a GPU debugger shows (Xcode shows only MSL) reads like what was written, and
both put `#line` directives on every statement, so a message from DXC or the
Metal compiler, which is a Goose compiler bug when it happens, still points
at the Goose source.

**Compilers.** The HLSL goes to DXC twice, for DXIL (`-T *_6_6`) and for
SPIR-V (`-spirv -fspv-target-env=vulkan1.2 -fvk-use-scalar-layout`); the two
texts differ only where DXIL flattens a constant buffer (§5.3). The MSL
goes to `xcrun -sdk macosx metal` (or the iOS SDK) and then `metallib`.
Which outputs a program gets depends on its target: SPIR-V and a
`.metallib` on Apple, DXIL and SPIR-V on Windows, SPIR-V on Linux. A
missing compiler is an error naming it and saying how to install it,
reported only by a program that compiles shaders.

**The shader library.** One blob per program, emitted as static data the
way `embed_shader`'s blobs are: per entry point its name, stage, group
size, push and uniform sizes, capability bits and its code for each
format. The format is the package's compatibility surface with the
compiler, like `src/gfx_blob.h` is for `gfx`, with a version both check.

**Caching.** The JIT compiles a program every time it runs, and DXC and the
Metal compiler are slow, so each stage's binary is cached under
`~/.goose/build/shaders/` by a hash of the emitted text and the compiler's
version. An edit to a Goose function recompiles only the stages that call
it.

**Inspection.** `goose build --emit-shaders dir` writes the emitted HLSL and
MSL, and the compilers' outputs, for a program; `--emit-shaders` with no
pipeline calls is how the golden tests (§9) see them.

## 7. Ray tracing

Ray tracing is acceleration structures on the host and one query on the GPU.

```goose
let blas = gpu::primitive_accel(PrimitiveAccelDesc { vertices: vb, indices: ib, count: tris, .. });
let tlas = gpu::instance_accel(InstanceAccelDesc { instances: inst_buf, max: 4096, .. });
gpu::build(blas);
gpu::build(tlas, live_count);            // after writing gpu::Instance records to inst_buf
gpu::barrier(tlas, gpu::ACCESS_ACCEL_WRITE, gpu::ACCESS_ACCEL_READ);
```

Builds, refits (`refit`) and compaction (`gpu::compact(blas)`) are
command-buffer work like any other, ordered by barriers. A top-level
structure is a heap handle (`gpu::Accel`), so a shader reaches it like a
texture, through push constants or a buffer.

`gpu::Instance` is one portable record: a row-major 3x4 transform, then
options, a mask, an intersection table offset (unused), a user id and the
primitive structure's reference, its low word first (`gpu::reference`
gives it as a `uint2`), 72 bytes. That is Metal's indirect instance
descriptor read with a row-major transform, which Metal 4 builds from
directly. It holds no 64-bit integer, so it is in the GPU subset: a culling
pass writes instances as readily as the host does (`gpu::instance_of`).

Direct3D 12 and Vulkan take a 64-byte record of their own, so their
backends convert. Shaders never read the converted records: the build is
their only reader, and the build is a host call naming its instance buffer,
so the package knows exactly when the read happens and records all of it
itself. `build` records, in order:

1. A conversion dispatch reading the program's `gpu::Instance` buffer and
   writing a native-layout buffer the package owns. That buffer is not in
   the heap, so nothing else can reach it, and the reference is copied
   through: on those backends `gpu::reference` is the structure's device
   address.
2. A barrier from that compute write to acceleration-structure build input.
3. The build, reading the native buffer by device address.
4. A barrier from the build's write to acceleration-structure reads, which
   is what every later trace needs.

The native buffer's hazards are the package's: the next build's
conversion would overwrite records the previous build may still be
reading, so each conversion starts with a barrier from build reads to
compute writes (or the package rotates buffers by fence).

What the package cannot know is when the program's own GPU work has
finished writing the instances, or a primitive structure's vertices and
indices: bindless means it never sees those writes. That ordering is the
program's, as for any resource its GPU work writes (§3), through the access
`ACCESS_ACCEL_INPUT`: `barrier(instances, ACCESS_SHADER_WRITE,
ACCESS_ACCEL_INPUT)` before `build`. Vulkan and Direct3D 12 map it to
compute-shader reads and build reads, covering the conversion; Metal keeps
it by its encoders' barriers. Instances the host writes with
`update_buffer` need no barrier, since uploads complete before the
submission holding the build.

On the GPU:

```goose
let hit = gpu::trace(scene.tlas, Ray { origin: p, dir: d, t_min: 0.001, t_max: 1e30 }, 0xff);
if hit.kind == HitKind.TRIANGLE { shade(hit.instance_id, hit.primitive, hit.barycentrics, hit.t) }
let lit = !gpu::trace_any(scene.tlas, shadow_ray, 0xff);   // stops at the first hit
```

`trace` returns the closest hit; `trace_any` returns whether there is one
and ends the search at the first, for shadow and occlusion rays. Both are
inline queries: `RayQuery` on Direct3D 12 and Vulkan (`VK_KHR_ray_query`),
`intersector` on Metal. Geometry is opaque in this first form; alpha-tested
geometry needs the candidate loop that inline queries expose, which comes
later as `trace` taking a function value that accepts or rejects a
candidate.

Ray queries are a capability (§5.5). KosmicKrisp may not offer them yet,
in which case on a Mac ray tracing is developed against Metal, and on Windows against
Direct3D 12 and Vulkan; a Vulkan device that gains them needs nothing new.

## 8. Relationship to gfx

`gfx` stays as it is: SDL_GPU, `embed_shader` and cute_spirv, for programs
that want a small API and no external shader compiler. `gpu` is a separate
package with a separate shader path, and a program uses one or the other.
`ui` draws through `gfx` today; it gets a `gpu` path once `gpu` can draw
its geometry, and whether `gfx` is then retired is decided then, with the
samples ported, not now.

## 9. Testing

* **Checker tests** (`test/gpu/*.goose`, expected output in
  `test/expected/`): every construct outside the subset, every stage
  capability misuse, every field that cannot cross to the GPU (§5.3).
  These need no compilers.
* **Emitter golden tests**: the HLSL and MSL emitted for a set of programs,
  flattened constant buffers included, compared as text, so a change to the
  emitters is reviewed as a change to
  their output. These need no compilers either.
* **Compile tests**: the same programs through DXC and the Metal compiler,
  where those are installed, with warnings as errors.
* **GPU tests**: headless programs that dispatch or draw and read back,
  run on every backend the machine has (Metal and Vulkan on a Mac; Direct3D
  12 and Vulkan on Windows), with Vulkan's validation layers on.
* **CPU differential tests**: a GPU function with no resources is also an
  ordinary Goose function, so a test runs it over a range of inputs on the
  CPU and in a compute shader, and compares: integers exactly, floats to a
  stated ulp bound. This tests the emitters and the numerics of §5.7. It is
  testing only: GPU functions do not run as programs on the CPU.

## 10. The plan

Each phase lands as working commits with its tests.

1. **One compute shader, end to end, on Metal 4 and Vulkan.** The subset
   checker for scalars, vectors, structs and control flow; both emitters;
   DXC and Metal compiler invocation and the cache; the shader library; and a
   `gpu` package that can make a device, a buffer with uploaded contents
   and a compute pipeline, dispatch with push constants, and read back through a
   fence. The milestone is a compute shader
   that writes a buffer, read back on both backends and compared with the
   same function run on the CPU.
   *Where it stands:* built. `compute(x, y, z) fn` entry points, the subset
   checker and both emitters are `src/gpu.h`, the library format
   `src/gpu_library.h`, and the package `packages/gpu`, with Metal 4 and
   Vulkan backends; `packages/gpu/test/compute.goose` passes on Metal and on
   KosmicKrisp, under each one's validation. Simplified until phase 2:
   buffers are host-visible and an upload is a copy at creation; a read back
   copies when its submission completes, not at its place in the stream;
   the backends put a barrier after every dispatch, and wait for idle
   before releasing anything. The emitters read the unoptimized
   specializations and emit no checks yet, so faults (§5.6) come
   with them in phase 2.
2. **Raster and the rest of the device model.** Textures and samplers,
   render passes, pipelines, swap chains, barriers, updates and the stream,
   uniforms, threads and command buffers, faults (§5.6), timestamps. A port of
   `samples/27_gfx_cube.goose` is the milestone. Alongside it, `f16` in the
   language (§5.8): spec, checker, C backend and tests, then GPU functions.

   *Where it stands:* headless raster is built. Vertex and fragment entry
   points with varyings, uniform blocks, textures and samplers through the
   heaps (DXC's `-fvk-bind-resource-heap` maps them onto the Vulkan ABI of
   §4), and in the package textures with managed uploads, render passes,
   pipelines made per target formats, indexed draws, barriers checked
   against each texture's tracked access, and read backs copied where they
   are asked for. `packages/gpu/test/raster.goose` renders a textured quad
   exactly on both backends, under validation. Windows and presentation
   are built too, through SDL3 where it is installed: a `CAMetalLayer` with
   Metal 4's drawable protocol, a Vulkan swap chain made again when the
   window's size changes, `begin_screen` placing its own barriers, input
   and frame timing as gfx's, and `GOOSE_GPU_HEADLESS=1` for tests.
   `packages/gpu/examples/cube.goose`, gfx's cube with Goose shaders, runs
   in a window on Metal 4 and on KosmicKrisp. Updates to resources in use
   (`update_buffer`, `update_texture`) are built as §3 says, staged and
   copied ahead of the next submission, with a barrier between copies to
   one resource; `packages/gpu/test/updates.goose` rewrites a buffer three
   times, overlapping, before each of six submissions in flight at once,
   and is clean under Vulkan's synchronization validation. Left of phase
   2: mipmaps, and screenshots to PNG.
3. **Direct3D 12.** By this point the HLSL emitter and the ABI have been
   exercised through Vulkan, so this is mostly the backend.
4. **Ray tracing**, on Metal and Direct3D 12: acceleration structures, the
   instance conversion pass, `trace` and `trace_any`. A small path tracer
   is the test, with an offline reference renderer for its images.

   *Where it stands:* built on Metal 4. `gpu::trace` and `gpu::trace_any`
   emit inline `RayQuery` in HLSL (compiled to DXIL and SPIR-V) and an
   `intersector` in MSL, and mark the library as needing ray queries,
   which a device without them refuses to make a pipeline from. The
   package builds primitive and instance acceleration structures, in
   order with the work around them. `packages/gpu/test/rays.goose` casts
   64 rays at three instances of one square and checks every hit's
   instance, distance, triangle and side, on Metal under validation.
   The test's instances are written by a compute pass and ordered before
   the build by a barrier into `ACCESS_ACCEL_INPUT`, so the contract the
   conversion needs is exercised on Metal already. KosmicKrisp has no ray
   queries yet, so Vulkan reports none; its build path and the instance
   conversion wait for a driver that has them.
5. **A frame graph, written in Goose**: passes that declare what they read
   and write, barriers and transient memory derived from that, unused passes
   pruned. Whether Goose's provenance analysis can reject a stale resource
   version at compile time, rather than asserting at run time, is a question
   for then.

## 11. Open questions

* **UFCS across namespaces.** `tex.sample(s, uv)` in §5.4 assumes a
  method call finds `gpu::sample`, but UFCS resolves only in the caller's
  own namespace (docs/design/namespaces.md), so user code writes
  `gpu::sample(tex, s, uv)` today. Associated-namespace lookup for a
  receiver whose type is namespaced would fix it for every package.
* **Swizzles.** `vec` has `xy` and `xyz`; shaders want `v.zyx`, `v.xxy` and
  assignment to `v.xy`. A general swizzle is a language feature, not a
  library one, and should be designed for CPU code too.
* **Matrix operators.** `mul(a, b)` today; whether `*` on matrices is worth
  the ambiguity with elementwise `*` on vectors.
* **Release-build resource robustness on Metal** (§5.6): accept undefined
  behavior, as Metal does, or keep the size checks in release at a cost.
* **Fast math per entry point**, which compiles a function reached from
  both strict and fast entries once per mode, and how a test states its
  tolerance against it.
* **Live resize on macOS** (§3, Presentation): whether
  `presentsWithTransaction` can be honored through SDL, starting with
  whether an event watch for `SDL_EVENT_WINDOW_EXPOSED` runs inside the
  resize's Core Animation transaction, and then how a program rendering on
  another thread draws that frame on the main thread.
* **A copy queue** for large uploads that should not hold up the
  submission after them, with resources that become ready later
  (§3). Deferred until a program needs it.
* **Main-thread functions**: a package is `main_thread_only` as a whole
  (`docs/packages.md`), and only `gpu`'s window and input functions need
  the main thread. Either the flag works per function, or the window and
  input move to a package of their own that `gfx` and `gpu` share, which
  is `main_thread_only` while `gpu` is not.
* **`u64`** for GPU addresses and hashes, after `f16` and the narrow
  integers.
* **How a fragment entry with no stage input reads**: a required empty
  struct is simple and unambiguous with push constants in the second place, but
  it is noise.
* **Where the package lives**: its own repository like `gfx`, with the
  compiler's half (checker, emitters, library format) here.
