// The Goose standard library: `gpu_shader` (docs/design/gpu.md §5): what a
// GPU function reaches beyond the language. The system values are externs
// under gs_shader_ names, which the shader emitters (src/gpu.h) turn into
// their stage's inputs and which nothing on the CPU defines; buffers are
// handles, heap indices, read and written through the gpu_load and
// gpu_store builtins. The gpu package builds its host API on top, in the
// same namespace.

namespace gpu;

import vec;
import math;

// A compute entry's invocation: its index in the whole dispatch, its
// group's, and its own within the group (flat, and per axis).
extern "gs_shader_dispatch_thread_id" fn dispatch_thread_id() -> uint3;
extern "gs_shader_group_id" fn group_id() -> uint3;
extern "gs_shader_group_thread_id" fn group_thread_id() -> uint3;
extern "gs_shader_group_index" fn group_index() -> u32;

// A vertex entry's vertex and instance.
extern "gs_shader_vertex_index" fn vertex_index() -> u32;
extern "gs_shader_instance_index" fn instance_index() -> u32;

// A buffer of T, by its index in the resource heap: elements T's packed size
// apart, each in T's Goose layout (§5.3).
struct StructuredBuffer<T> { index: u32 }
struct RWStructuredBuffer<T> { index: u32 }

fn load<T>(b: StructuredBuffer<T>, i: u32) -> T { gpu_load<T>(b.index, i) }
fn load<T>(b: RWStructuredBuffer<T>, i: u32) -> T { gpu_load<T>(b.index, i) }
fn store<T>(b: RWStructuredBuffer<T>, i: u32, v: T) { gpu_store(b.index, i, v); }

// A fragment entry's pixel: its window position (x and y in pixels from the
// top left, z its depth), whether its triangle faces the camera, and
// discarding it.
extern "gs_shader_frag_coord" fn frag_coord() -> float4;
extern "gs_shader_front_facing" fn front_facing() -> bool;
extern "gs_shader_discard" fn discard();

// How a value changes to the next pixel across and down: fragment entries
// only.
extern "gs_shader_ddx" fn ddx(x: f32) -> f32;         extern "gs_shader_ddx" fn ddx(v: float2) -> float2;
extern "gs_shader_ddx" fn ddx(v: float3) -> float3;   extern "gs_shader_ddx" fn ddx(v: float4) -> float4;
extern "gs_shader_ddy" fn ddy(x: f32) -> f32;         extern "gs_shader_ddy" fn ddy(v: float2) -> float2;
extern "gs_shader_ddy" fn ddy(v: float3) -> float3;   extern "gs_shader_ddy" fn ddy(v: float4) -> float4;
extern "gs_shader_fwidth" fn fwidth(x: f32) -> f32;   extern "gs_shader_fwidth" fn fwidth(v: float2) -> float2;

// A texture a shader samples or reads, with texels of T (f32, float2,
// float4, i32, uint4, ...), and one it writes; a sampler. Each by its index
// in its heap.
struct Texture2D<T> { index: u32 }
struct RWTexture2D<T> { index: u32 }
struct Sampler { index: u32 }

// The fixed samplers, the sampler heap's first entries (docs/design/gpu.md
// §3): filtering, then what a coordinate outside 0..1 reads.
let POINT_CLAMP = Sampler { 0 };
let LINEAR_CLAMP = Sampler { 1 };
let POINT_REPEAT = Sampler { 2 };
let LINEAR_REPEAT = Sampler { 3 };

// At the level of detail the pixel's footprint picks: fragment entries only.
fn sample<T>(t: Texture2D<T>, s: Sampler, uv: float2) -> T { gpu_sample<T>(t.index, s.index, uv) }
fn sample_level<T>(t: Texture2D<T>, s: Sampler, uv: float2, lod: f32) -> T {
    gpu_sample_level<T>(t.index, s.index, uv, lod)
}
// A texel by its coordinate in pixels, unfiltered.
fn texel<T>(t: Texture2D<T>, xy: int2, mip: i32) -> T { gpu_texel<T>(t.index, xy, mip) }
fn store<T>(t: RWTexture2D<T>, xy: int2, v: T) { gpu_store_texel(t.index, xy, v); }

// A 4x4 matrix, column-major, as gfx's: m * v is mul(m, v).
struct mat4 { c0: float4, c1: float4, c2: float4, c3: float4 }

let mat4_identity = mat4 { float4 { 1.0, 0.0, 0.0, 0.0 }, float4 { 0.0, 1.0, 0.0, 0.0 },
                           float4 { 0.0, 0.0, 1.0, 0.0 }, float4 { 0.0, 0.0, 0.0, 1.0 } };

fn mul(m: mat4, v: float4) -> float4 { m.c0 * v.x + m.c1 * v.y + m.c2 * v.z + m.c3 * v.w }
fn mul(a: mat4, b: mat4) -> mat4 { mat4 { mul(a, b.c0), mul(a, b.c1), mul(a, b.c2), mul(a, b.c3) } }

// Transforms, as gfx's: right-handed, looking down -z, to a clip space with
// y up and depth from 0 at `near` to 1 at `far`. They run on the CPU and
// the GPU alike.
fn translate(t: float3) -> mat4 {
    var m = mat4_identity;
    m.c3 = float4 { t.x, t.y, t.z, 1.0 };
    m
}

fn scale(s: float3) -> mat4 {
    mat4 { c0: float4 { s.x, 0.0, 0.0, 0.0 }, c1: float4 { 0.0, s.y, 0.0, 0.0 },
           c2: float4 { 0.0, 0.0, s.z, 0.0 }, c3: float4 { 0.0, 0.0, 0.0, 1.0 } }
}

// A rotation of `angle` radians about `axis`, counterclockwise looking down
// the axis.
fn rotate(axis: float3, angle: f32) -> mat4 {
    let a = normalize(axis);
    let c = cos(angle);
    let s = sin(angle);
    let k = 1.0 - c;
    mat4 { c0: float4 { c + a.x * a.x * k, a.y * a.x * k + a.z * s, a.z * a.x * k - a.y * s, 0.0 },
           c1: float4 { a.x * a.y * k - a.z * s, c + a.y * a.y * k, a.z * a.y * k + a.x * s, 0.0 },
           c2: float4 { a.x * a.z * k + a.y * s, a.y * a.z * k - a.x * s, c + a.z * a.z * k, 0.0 },
           c3: float4 { 0.0, 0.0, 0.0, 1.0 } }
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> mat4 {
    let f = 1.0 / tan(fovy * 0.5);
    mat4 { c0: float4 { f / aspect, 0.0, 0.0, 0.0 }, c1: float4 { 0.0, f, 0.0, 0.0 },
           c2: float4 { 0.0, 0.0, far / (near - far), -1.0 },
           c3: float4 { 0.0, 0.0, near * far / (near - far), 0.0 } }
}

fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> mat4 {
    mat4 { c0: float4 { 2.0 / (right - left), 0.0, 0.0, 0.0 },
           c1: float4 { 0.0, 2.0 / (top - bottom), 0.0, 0.0 },
           c2: float4 { 0.0, 0.0, 1.0 / (near - far), 0.0 },
           c3: float4 { (left + right) / (left - right), (top + bottom) / (bottom - top),
                        near / (near - far), 1.0 } }
}

fn look_at(eye: float3, target: float3, up: float3) -> mat4 {
    let f = normalize(target - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    mat4 { c0: float4 { s.x, u.x, -f.x, 0.0 }, c1: float4 { s.y, u.y, -f.y, 0.0 },
           c2: float4 { s.z, u.z, -f.z, 0.0 },
           c3: float4 { -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0 } }
}

// Four bytes, low first, as numbers from 0 to 1: a color the GPU reads
// packed in a u32, as GPU functions compute in 32-bit numbers only (§5.2).
fn unpack_unorm4x8(c: u32) -> float4 {
    let scale: f32 = 1.0 / 255.0;
    float4 { c & 255, (c >> 8) & 255, (c >> 16) & 255, (c >> 24) & 255 } * scale
}
