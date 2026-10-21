# Goose on the GPU: the language past the shader subset

<details>
<summary>Table of contents</summary>

- [1. Why the language fits](#1-why-the-language-fits)
- [2. Where things stand](#2-where-things-stand)
- [3. References as parameters](#3-references-as-parameters)
- [4. Multiple returns and `return … from`](#4-multiple-returns-and-return--from)
- [5. ADTs, `match` and case functions](#5-adts-match-and-case-functions)
- [6. Faults: checks that abort](#6-faults-checks-that-abort)
- [7. Limited arrays](#7-limited-arrays)
- [8. Data stacks per invocation](#8-data-stacks-per-invocation)
- [9. Self-relative structures shared with the CPU](#9-self-relative-structures-shared-with-the-cpu)
- [10. Global arrays and index types as buffers](#10-global-arrays-and-index-types-as-buffers)
- [11. `const for`](#11-const-for)
- [12. Recursion](#12-recursion)
- [13. Typed queues as GPU work queues](#13-typed-queues-as-gpu-work-queues)
- [14. Scalars](#14-scalars)
- [15. The order of work](#15-the-order-of-work)

</details>

The GPU functions of [design/gpu.md](gpu.md) §5 are a subset: scalars,
vectors, structs, fixed arrays, loops and calls. This note is about the rest
of the language, feature by feature: what each would mean in a shader, how
it lowers to HLSL and MSL, and what it is worth. Several of Goose's
defining choices turn out to be what a GPU wants, and a few let Goose do
what no shading language does, chiefly sharing pointer-linked data with the
CPU byte for byte (§9) and carrying bounds proofs from the CPU into shaders
(§10).

## 1. Why the language fits

A GPU has no heap, no unwinder, no function pointers and, in practice, no
recursion; every shading language is C with those parts removed. Goose
never had most of them:

* **No heap.** Allocation is bump allocation on data stacks whose number is
  computed at compile time ([spec](../goose_spec.md) §1.2, §10.3). A thread
  program already carries a pointer to a block of its stacks (§11.2); a GPU
  invocation is the same, with a block carved out of a buffer (§8).
* **No escaping references.** A reference never outlives its root (§9.1),
  so a reference parameter is never stored, and a reference into a local
  never leaves its function's dynamic extent: the lowering can be an
  `inout` parameter or a thread-local pointer (§3).
* **Static dispatch only.** Function values are compile-time (§7.6), and
  the only dynamic polymorphism is a closed ADT's tag (§3.5, §8.2): a
  `switch`, which is what GPUs want instead of indirect calls (§5).
* **Long-distance return without unwinding.** `return … from` is defined as
  a hidden discriminant every intermediate frame tests (§7.9), which a GPU
  runs as written (§4).
* **Recursion that keeps no data stack.** A recursive cycle may hold no
  nonfixed value across a call into itself (§7.8), so its activations are
  fixed-size frames, which an explicit frame stack can hold (§12).
* **Position-independent data.** Self-relative references are offsets, and
  a structure linked with them is its own serialization (§3.9), so it can be
  uploaded as bytes and followed in a shader (§9).

## 2. Where things stand

Implemented, beyond the subset of gpu.md §5.2: fixed arrays and global
tables of literals, group memory (`gpu::GroupShared<T>`), barriers,
atomics and wave operations. Rejected with an error naming the construct:
references other than an entry point's uniform block, multiple returns,
ADTs and `match`, function values and nested functions, `return … from`,
non-fixed arrays, slices, strings, optionals, varints, relative references,
recursion, globals other than constants and group memory, and integers
other than `i32` and `u32`.

## 3. References as parameters

`T&` and `const T&` parameters, and `for &x` over what a parameter refers
to. MSL has references: `thread T&`, `device T&` and `threadgroup T&`, by
where the argument lives. HLSL has `inout`, copy-in copy-out, which is the
same as a reference unless the callee can see the storage another way
while it runs. In a GPU function it can only through another reference
parameter bound to overlapping storage, which the call site can see: two
arguments rooted at one variable where one is written. Such a call is
rejected in the HLSL dialect with that reason; Goose's checker already
computes the roots (§9.2).

What a reference may point at, on the GPU: a local or parameter (thread
memory), group memory, or an element of a buffer. The last is the
interesting one: `fn bump(p: Particle&)` called with an element of a
`RWStructuredBuffer<Particle>` means the element itself. MSL passes
`device Particle&`; HLSL, which has no buffer references, loads the
element into a temporary, passes it `inout`, and stores it back after the
call, which is the same thing for a buffer element no other invocation
writes, the case a GPU program already has to arrange. A function reached
with references of more than one address space is emitted once per space
in MSL, as monomorphization already does per root class (§10.2).

Optional references, references in structs and rebinding (`.=`) need
pointers, and wait for §8.

## 4. Multiple returns and `return … from`

A function returning several values returns a struct of them, and the
caller binds its fields; HLSL and MSL return structs by value, and the GPU
compilers keep them in registers.

`return E from f` lowers as the spec defines it (§7.9): each function on a
path from `f` to the return gains a hidden discriminant, returned beside
its result, which every caller on the path tests, returning at once when
it says "for `f`". No unwinder is involved, so nothing about it is CPU
specific. It gives shaders structured early exit through several calls:
a traversal that finds its answer deep in helper functions returns it to
the kernel in one statement.

## 5. ADTs, `match` and case functions

A fixed-mode ADT (§3.5) is a tag and the largest payload. Neither HLSL nor
MSL lowers unions portably, so a value is a struct of the tag and the
payload as 32-bit words, `uint` in both dialects, and a variant's fields
are read and written with bit casts (`asfloat`, `as_type<float>`) at their
offsets. Goose's packed layout is the layout in buffers too: an array of
`Material` uploaded from the CPU is read as is.

`match` lowers to a `switch` on the tag; binders by value read the
payload's words into the variant's struct. Integer `match` lowers to a
`switch` or to range tests. Case functions (§8.2) are a `match` over the
overloads, and the closed world means the GPU compiler sees every case.

This is the feature that changes how shaders are written. A path tracer's
materials are an ADT, `enum Material { Lambert { albedo: float3 }, Metal {
… }, Glass { … }, Emissive { … } }`, and scattering is a set of overloads
`fn scatter(m: Material.Glass, …)`, each a short function of its own,
instead of one uber-shader branching on flag bits. Adding a material is
adding a variant and an overload, and the compiler reports the overload
missing. Variable-mode ADTs, whose sizes differ, need §8.

## 6. Faults: checks that abort

Goose checks array bounds, integer conversions, division by zero and its
own invariants, and aborts when one fails (§9.3). A shader cannot abort.
gpu.md §5.6 gives it a fault buffer instead: a failed check writes its
code, its source location and the invocation's index there, once (an
atomic compare-exchange on the first word), and the invocation goes on
with a safe value (a clamped index, a zero quotient). The host reads the
buffer at each fence it waits on and aborts with the CPU's diagnostic,
naming the shader and the line. A check the bounds-check pass removes
(§10.5) costs nothing, as on the CPU; `checkfree fn` means the same thing
on the GPU, and index types (§10) remove the checks no analysis can.

Every later section's checks go through this: limited arrays' capacity,
stack budgets, ADT tags read from buffers.

## 7. Limited arrays

`T[..k]` is fixed-size: `k` elements and a length (§3.3). On the GPU it is
a fixed array and a count, in registers or thread memory, with `push`,
`pop`, `len`, indexing and iteration checked against the length. It is
what every GPU traversal hand-writes: `var todo: u32[..32] = [];` as a BVH
traversal stack, a clipping polygon's vertex list, a small priority queue.
Exceeding the capacity faults (§6), so an overflow is a diagnostic rather
than corrupted memory.

## 8. Data stacks per invocation

Resizable and variable-size locals (`T[>..]`, `T[]`, strings, variable
ADTs) live on data stacks (§1.3). For GPU code the compiler computes N′,
the stacks a program needs, exactly as for a thread program (§11.2), and
each invocation's N′ stacks are a slice of a scratch buffer the package
binds: its base is the buffer's address plus the invocation's index times
a stride, and each stack's top starts at its share of the stride. `push`
is a bump, a scope's exit restores the watermark, and nothing else exists
at run time, as on the CPU.

What differs is capacity. The CPU reserves address space and has guard
pages (§10.4); the GPU has a budget per invocation, given with the
dispatch (`gpu::dispatch(x, y, z)` grows a form taking the bytes each
invocation may use, a default the program can raise). Exceeding it faults
(§6). Where the compiler can bound a stack's use, from fixed pushes in
loops of constant count, the budget is checked at compile time and the
storage can move to thread memory.

References into a data stack are offsets into the scratch buffer: one
representation in both dialects, rooted as on the CPU, so the lifetime
rules carry over unchanged. This is also what optional references,
references in structs and rebinding need (§3): pointers, as offsets into
the buffers a shader can reach.

## 9. Self-relative structures shared with the CPU

A structure linked by self-relative references (§3.9) has no addresses in
it: each link is an offset from the field holding it. `bytes_of` of the
array holding it is its whole image, and that image can be uploaded to a
buffer and followed by a shader with the same Goose code: loading a link is
the field's own offset in the buffer plus the stored offset. A compact
tree, built on the CPU with `Node&<u16>` links (A.2), is traversed on the
GPU without a serializer, a conversion pass or a second definition of the
node.

The lowering: a reference into a buffer is the buffer's heap index and a
byte offset; a self-relative load adds the stored offset to the field's;
`in pool` links (§3.9) add theirs to the pool's base, the pool being a
global bound as §10 binds one. Verification is the CPU's: `from_bytes`
has checked an image that arrives from outside the program (§9.4), and an
image the program built was proven by construction.

The showcase is a BVH: built in Goose on the CPU over a glTF scene,
linked with relative references, uploaded as bytes, and traversed by one
`traverse` function that the CPU runs for testing and a compute shader
runs for a software path tracer, which then runs where the device has no
ray queries (KosmicKrisp today), checked against the hardware tracer on
Metal.

## 10. Global arrays and index types as buffers

An index type `u32 in nodes` is below `nodes.len` wherever it can be
named (§3.11), and its motivating example is a mesh's index buffer. If a
global array is a GPU buffer, the proof carries over: the program binds the
global's current contents (`gpu::upload(nodes)` makes or updates the
buffer; a shader reads `nodes[i]` as the global), and a shader indexing
`nodes` with a `u32 in nodes` read from another uploaded array needs no
check, since the arrays it came from were checked when they were built and
uploaded together. The upload records each global's length, which is what
a check against an ordinary index compares with.

What this removes from GPU programs is the class of bug where an index
buffer and a vertex buffer disagree, found today as garbage on the screen
or a device loss.

## 11. `const for`

Expansion over a struct's fields or an ADT's variants happens in the
checker, before the emitter sees the function (§6.5), so it should run on
the GPU as soon as what it expands to does. It is the way to derive what
shader code otherwise writes by hand per struct: interpolating every field
of a varyings struct, packing a struct into fewer words, converting
between array-of-structs and struct-of-arrays layouts
([design/soa.md](soa.md)), or a `match` over every material.

## 12. Recursion

A `recursive fn` cycle holds no data stack across a call into itself
(§7.8), so an activation is its fixed-size locals and parameters. That is
the condition under which recursion converts mechanically to a loop over
an explicit stack of frames: each call into the cycle pushes a frame and
a resume point, each return pops one, and the function body is a state
machine over the resume points. The frame stack is a limited array (§7)
whose capacity is a budget like §8's, and exceeding it faults.

This is how a naively written recursive traversal (a BVH, a quadtree, a
recursive subdivision) runs in a shader. The conversion is mechanical but
not small; it comes after the rest.

## 13. Typed queues as GPU work queues

Goose threads share nothing and communicate through typed queues (§11.2).
On the GPU, `qput(v)` from a shader would append to a typed buffer with an
atomic count, and the host, or the next dispatch through an indirect
argument, consumes it: GPU-driven work generation (culling that emits
draws, a wavefront path tracer's queues per material) in the language's own
coordination model rather than a hand-built append buffer.

## 14. Scalars

`f16` (gpu.md §5.8), `u8`, `i8`, `u16` and `i16` as storage, and `i64` and
`u64`, which Metal has and Direct3D 12 and Vulkan have on most devices, for
addresses and hashes. Each is a capability where a device may lack it.

## 15. The order of work

1. References as parameters (§3), multiple returns and `return … from`
   (§4): ordinary Goose code runs unchanged.
2. ADTs, `match` and case functions (§5), and the path tracer's materials
   rewritten as an ADT.
3. Faults (§6), then limited arrays (§7).
4. Self-relative structures in buffers (§9) and the software BVH tracer.
5. Data stacks per invocation (§8), with optional references and
   references in structs.
6. Global arrays and index types as buffers (§10), and `const for` (§11).
7. Recursion (§12) and queues (§13).
