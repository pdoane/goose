/* The gpu package's C (docs/design/gpu.md §3): the layer the Goose externs
   call (gpu.c), over one backend at a time (gpu_metal.m, gpu_vulkan.c).
   The common layer owns everything the backends need not differ on: slots
   and what they hold, pipelines and their state, the recording state, the
   access each texture was last left in, read backs and errors. A backend
   owns the device objects. */

#ifndef GOOSE_GPU_INTERNAL_H
#define GOOSE_GPU_INTERNAL_H

#include <stdint.h>
#include <stddef.h>

#include "gpu_library.h"

/* A Goose `const u8[:]`, as an extern fn passes it (spec §7.10). */
typedef struct { const uint8_t *data; int64_t len; } goose_gpu_bytes;

/* The resource heap's size, in slots. Slot 0 is the stream's (§3), unused
   so far, which keeps 0 meaning "none". */
#define GOOSE_GPU_HEAP_SLOTS 16384
#define GOOSE_GPU_MAX_PIPELINES 1024
#define GOOSE_GPU_MAX_TARGETS 4
#define GOOSE_GPU_SAMPLERS 64

/* gpu.goose's constants, which are gfx's (docs/stdlib.md §gfx). */
enum {
    GOOSE_GPU_RGBA8 = 1, GOOSE_GPU_BGRA8, GOOSE_GPU_RGBA8_SRGB, GOOSE_GPU_R8, GOOSE_GPU_RG8,
    GOOSE_GPU_RGBA16F, GOOSE_GPU_RGBA32F, GOOSE_GPU_R16F, GOOSE_GPU_RG16F, GOOSE_GPU_R32F,
    GOOSE_GPU_RG32F, GOOSE_GPU_R32UI, GOOSE_GPU_RGBA8UI, GOOSE_GPU_RGB10A2, GOOSE_GPU_RG11B10F,
    GOOSE_GPU_DEPTH16, GOOSE_GPU_DEPTH24, GOOSE_GPU_DEPTH32F, GOOSE_GPU_DEPTH24_STENCIL8,
    GOOSE_GPU_DEPTH,
    /* Block-compressed, 4x4 texels to a block of 8 or 16 bytes. */
    GOOSE_GPU_BC1, GOOSE_GPU_BC1_SRGB, GOOSE_GPU_BC3, GOOSE_GPU_BC3_SRGB, GOOSE_GPU_BC4,
    GOOSE_GPU_BC5, GOOSE_GPU_BC7, GOOSE_GPU_BC7_SRGB,
    GOOSE_GPU_FORMAT_END,
};
enum { GOOSE_GPU_SAMPLED = 1, GOOSE_GPU_COLOR_TARGET = 2, GOOSE_GPU_DEPTH_TARGET = 4,
       GOOSE_GPU_STORAGE = 8 };
enum { GOOSE_GPU_TRIANGLES, GOOSE_GPU_TRIANGLE_STRIP, GOOSE_GPU_LINES, GOOSE_GPU_LINE_STRIP,
       GOOSE_GPU_POINTS };
enum { GOOSE_GPU_CULL_NONE, GOOSE_GPU_CULL_FRONT, GOOSE_GPU_CULL_BACK };
enum { GOOSE_GPU_BLEND_NONE, GOOSE_GPU_BLEND_ALPHA, GOOSE_GPU_BLEND_ADD,
       GOOSE_GPU_BLEND_PREMULTIPLIED, GOOSE_GPU_BLEND_MULTIPLY };
/* Comparisons, COMPARE_NEVER = 1 to COMPARE_ALWAYS = 8. */

/* What the GPU last did with a resource, or does next (§3, barriers). */
enum {
    GOOSE_GPU_ACCESS_NONE = 0,
    GOOSE_GPU_ACCESS_SHADER_READ = 1,
    GOOSE_GPU_ACCESS_SHADER_WRITE = 2,
    GOOSE_GPU_ACCESS_RENDER_TARGET = 4,
    GOOSE_GPU_ACCESS_DEPTH_WRITE = 8,
    GOOSE_GPU_ACCESS_DEPTH_READ = 16,
    GOOSE_GPU_ACCESS_COPY_SRC = 32,
    GOOSE_GPU_ACCESS_COPY_DST = 64,
    GOOSE_GPU_ACCESS_PRESENT = 128,     /* the screen's, when frame() shows it */
    /* What an acceleration structure's build reads (§7): where a backend
       converts instances first, its conversion's reads too. */
    GOOSE_GPU_ACCESS_ACCEL_INPUT = 256,
};

#pragma pack(push, 1)
/* gpu.goose's TextureDesc, PipelineDesc and PassDesc, as Goose lays them
   out (packed, spec C.2). */
typedef struct {
    int32_t kind, format, usage, width, height, depth, mips, samples;
} goose_gpu_texture_desc;

typedef struct {
    int32_t primitive, cull;
    uint8_t clockwise, wireframe, depth_test, depth_write;
    int32_t depth_compare, blend;
} goose_gpu_pipeline_desc;

/* gpu.goose's PrimitiveAccelDesc and InstanceAccelDesc. */
typedef struct {
    uint32_t vertices;
    int64_t vertex_offset, vertex_stride, vertex_count;
    uint32_t indices;
    int64_t index_size, index_offset, triangle_count;
} goose_gpu_primitive_accel_desc;

typedef struct {
    uint32_t instances;
    int64_t max;
} goose_gpu_instance_accel_desc;

/* gpu::Instance, which is Metal's indirect instance descriptor read with a
   row-major transform (docs/design/gpu.md §7). */
#define GOOSE_GPU_INSTANCE_SIZE 72

typedef struct {
    uint32_t color[GOOSE_GPU_MAX_TARGETS];
    uint32_t depth;
    uint8_t clear_color, clear_depth;
    float color_value[4];
    float depth_value;
} goose_gpu_pass_desc;
#pragma pack(pop)

/* A texture as the common layer knows it. */
typedef struct {
    goose_gpu_texture_desc desc;
    int access;             /* GOOSE_GPU_ACCESS_*, as the last barrier left it */
    int submitted;          /*   and as the work submitted so far leaves it */
} goose_gpu_tex;

/* A pipeline as the common layer knows it: what its library said, and for
   a raster one the native pipelines made for each set of target formats
   it has drawn into (§3: made where first bound, as gfx makes them). */
#define GOOSE_GPU_PIPELINE_VARIANTS 8
typedef struct {
    int32_t colors[GOOSE_GPU_MAX_TARGETS];
    int32_t depth;
} goose_gpu_targets;

typedef struct {
    int used;
    int compute;
    gs_gpu_lib_header header;
    gs_gpu_lib_entry entries[2];
    uint8_t *library;           /* a copy: the backend reads its code */
    int64_t librarylen;
    goose_gpu_pipeline_desc desc;
    void *native;               /* compute: the backend's pipeline */
    struct { goose_gpu_targets key; void *native; } variants[GOOSE_GPU_PIPELINE_VARIANTS];
    int nvariants;
} goose_gpu_pipe;

/* A pass being recorded, its targets resolved. */
typedef struct {
    goose_gpu_pass_desc desc;
    goose_gpu_targets formats;
    int ncolor;
    int width, height;
} goose_gpu_pass;

typedef struct goose_gpu_backend {
    const char *name;
    /* 0 on failure, with goose_gpu_fail called. `window` is the SDL window
       the screen is shown in, null for none. */
    int (*open)(void *window, int vsync);
    void (*close)(void);
    /* A buffer of `size` bytes in heap slot `slot`, holding `data` when that
       is not null, and visible to shaders at that index. */
    int (*buffer_create)(uint32_t slot, int64_t usage, const void *data, int64_t size);
    /* A texture in `slot`. With data (tightly packed rows of mip 0), its
       upload is complete before the next submission runs, and leaves it in
       its steady access, shader reads. */
    int (*texture_create)(uint32_t slot, const goose_gpu_texture_desc *d, const void *data,
                          int64_t len);
    /* `kind` is GOOSE_GPU_KIND_*. */
    void (*release)(uint32_t slot, int kind);
    /* An update (§3): `size` bytes staged now and copied ahead of the next
       submission, ordered after the work already submitted and before the
       work that will be, copies to one resource after each other. A
       texture's whole mip 0, which the work submitted left in `access`. */
    int (*update_buffer)(uint32_t slot, int64_t offset, const void *data, int64_t size);
    int (*update_texture)(uint32_t slot, int access, const void *data, int64_t size);
    /* Mips 1 and up filled from mip 0, which the work before it left the
       texture in `access`, and every mip left in shader reads. With
       `upload`, after the copies staged for the next submission (a loaded
       texture's); else where the program's work is being recorded, outside
       passes. */
    int (*generate_mips)(uint32_t slot, int access, int upload);
    /* A copy of a buffer's bytes from `offset` into a texture's mip 0, laid
       out as an upload's are, where the program's work is recorded: the
       buffer made visible to copies and the texture left in
       GOOSE_GPU_ACCESS_COPY_DST by the program's barriers. */
    void (*copy_to_texture)(uint32_t buffer, int64_t offset, uint32_t texture);
    /* Ray tracing (§7): whether the device has ray queries, and
       acceleration structures in heap slots, sized for their description
       (one of `prim` and `inst`) and built when asked, in order with the
       work around the build. A bottom-level structure's reference is what
       a gpu::Instance names it by. */
    int (*ray_query)(void);
    int (*accel_create)(uint32_t slot, const goose_gpu_primitive_accel_desc *prim,
                        const goose_gpu_instance_accel_desc *inst);
    int (*accel_build)(uint32_t slot, int64_t instances);
    uint64_t (*accel_reference)(uint32_t slot);
    int (*compute_create)(goose_gpu_pipe *p);
    void *(*raster_create)(goose_gpu_pipe *p, const goose_gpu_targets *targets);
    void (*pipeline_release)(goose_gpu_pipe *p);
    int (*begin_compute)(void);
    void (*end_compute)(void);
    int (*begin_pass)(const goose_gpu_pass *pass);
    void (*end_pass)(void);
    /* A draw or dispatch with its push constants and uniform buffer:
       `uniformgen` changes when the uniform bytes do. */
    void (*dispatch)(goose_gpu_pipe *p, const void *push, uint32_t pushsize,
                     const void *uniforms, uint32_t uniformsize, uint64_t uniformgen,
                     uint32_t x, uint32_t y, uint32_t z);
    void (*draw)(goose_gpu_pipe *p, void *native, const void *push, uint32_t pushsize,
                 const void *uniforms, uint32_t uniformsize, uint64_t uniformgen,
                 int indexed, uint32_t count, uint32_t instances, uint32_t first,
                 int32_t vertex_offset, uint32_t first_instance);
    void (*index_buffer)(uint32_t slot, int index_size);
    void (*viewport)(float x, float y, float w, float h);
    void (*scissor)(int32_t x, int32_t y, int32_t w, int32_t h);
    /* Outside passes: the GPU's work with the texture in `before` done and
       visible to its work in `after`. */
    void (*barrier)(uint32_t slot, int before, int after);
    /* A copy of part of a buffer, or of a texture's mip 0 left in
       `access`, into memory the CPU reads once the submission carrying it
       completes. Returns the backend's handle on it. */
    void *(*readback_buffer)(uint32_t slot, int64_t offset, int64_t size);
    void *(*readback_texture)(uint32_t slot, int access, int64_t size);
    void (*readback_read)(void *rb, void *out, int64_t size);
    void (*readback_release)(void *rb);
    /* The screen's next image, in `slot` from here until it is presented,
       in access GOOSE_GPU_ACCESS_NONE; its size in pixels. 0 when there is
       none to draw into (a minimized window). */
    int (*acquire)(uint32_t slot, int *width, int *height);
    /* Submits what was recorded, after the uploads made since the last
       submission, and returns its fence, from 1 up; with `present`, shows
       the screen's image, left in GOOSE_GPU_ACCESS_PRESENT. */
    uint64_t (*submit)(int present);
    /* The latest fence that has completed. */
    uint64_t (*completed)(void);
    void (*wait)(uint64_t fence);
} goose_gpu_backend;

enum { GOOSE_GPU_KIND_BUFFER, GOOSE_GPU_KIND_TEXTURE, GOOSE_GPU_KIND_ACCEL };

/* The texture in a slot, for the backends; null for none. */
const goose_gpu_tex *goose_gpu_texture_of(uint32_t slot);
/* Bytes per texel of a format; 0 for a depth format the CPU does not read,
   and for a compressed one. */
int goose_gpu_texel_size(int format);
/* Bytes per 4x4 block of a compressed format; 0 for any other. */
int goose_gpu_block_size(int format);
/* The bytes of a row of mip 0 as the CPU lays it out, a row of blocks for
   a compressed format, and the number of such rows; 0 for a format the CPU
   does not read or write. */
int64_t goose_gpu_row_bytes(int format, int width);
int goose_gpu_rows(int format, int height);
int64_t goose_gpu_mip0_bytes(const goose_gpu_texture_desc *d);
int goose_gpu_is_depth(int format);
/* Whether mips of the format can be made by filtering: color, not
   integer. */
int goose_gpu_filters(int format);

/* Records why the call in progress failed, for gpu::error(). */
void goose_gpu_fail(const char *fmt, ...);
/* Records a misuse: printed now, and the program aborts at the next
   check (§3, Errors). */
void goose_gpu_misuse(const char *fmt, ...);

#ifdef __APPLE__
extern const goose_gpu_backend goose_gpu_metal;
#endif
extern const goose_gpu_backend goose_gpu_vulkan;

#endif
