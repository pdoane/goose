// The Goose standard library: `gpu_shader` (docs/design/gpu.md §5): what a
// GPU function reaches beyond the language. The system values are externs
// under gs_shader_ names, which the shader emitters (src/gpu.h) turn into
// their stage's inputs and which nothing on the CPU defines; buffers are
// handles, heap indices, read and written through the gpu_load and
// gpu_store builtins. The gpu package builds its host API on top, in the
// same namespace.

namespace gpu;

import vec;

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
