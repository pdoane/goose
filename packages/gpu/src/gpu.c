/* The gpu package's common layer (gpu_internal.h): what the Goose externs
   in gpu.goose call. One device at a time, as gfx has one. */

#include "gpu_internal.h"
#include "goose.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const goose_gpu_backend *backend;
static char lasterror[1024];
static int misused;

/* Heap slots: what each holds, and its size or texture. */
enum { SLOT_FREE, SLOT_BUFFER, SLOT_TEXTURE };
static uint8_t slotkind[GOOSE_GPU_HEAP_SLOTS];
static int64_t slotsize[GOOSE_GPU_HEAP_SLOTS];
static goose_gpu_tex textures[GOOSE_GPU_HEAP_SLOTS];

static goose_gpu_pipe pipelines[GOOSE_GPU_MAX_PIPELINES];

/* The recording state: a compute or render pass, what is bound, and the
   push constants and uniform bytes set for the work that follows. */
enum { PASS_NONE, PASS_COMPUTE, PASS_RENDER };
static int pass;
static goose_gpu_pass renderpass;
static goose_gpu_pipe *bound;
static void *boundnative;
static uint8_t push[128];
static uint32_t pushlen;
static uint8_t *uniforms;
static uint32_t uniformlen;
static uint64_t uniformgen;
static int indexsize;

/* Read backs, by id from 1: each a copy recorded where it was asked for,
   ready once the submission carrying it completes. */
typedef struct {
    int used;
    void *native;
    int64_t size;
    uint64_t fence;         /* 0 until the submission carrying it */
} readback;
#define MAX_READBACKS 256
static readback readbacks[MAX_READBACKS];

void goose_gpu_fail(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(lasterror, sizeof lasterror, fmt, args);
    va_end(args);
}

void goose_gpu_misuse(const char *fmt, ...) {
    char msg[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof msg, fmt, args);
    va_end(args);
    fprintf(stderr, "gpu: misuse: %s\n", msg);
    misused = 1;
}

void goose_gpu_check(void) {
    if (!misused) return;
    fprintf(stderr, "gpu: aborting after a misuse of the gpu package\n");
    exit(1);
}

const goose_gpu_tex *goose_gpu_texture_of(uint32_t slot) {
    return slot && slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_TEXTURE ? &textures[slot]
                                                                                  : NULL;
}

int goose_gpu_is_depth(int format) {
    return format >= GOOSE_GPU_DEPTH16 && format <= GOOSE_GPU_DEPTH;
}

int goose_gpu_texel_size(int format) {
    switch (format) {
        case GOOSE_GPU_RGBA8: case GOOSE_GPU_BGRA8: case GOOSE_GPU_RGBA8_SRGB:
        case GOOSE_GPU_RG16F: case GOOSE_GPU_R32F: case GOOSE_GPU_R32UI: case GOOSE_GPU_RGBA8UI:
        case GOOSE_GPU_RGB10A2: case GOOSE_GPU_RG11B10F:
            return 4;
        case GOOSE_GPU_R8: return 1;
        case GOOSE_GPU_RG8: case GOOSE_GPU_R16F: case GOOSE_GPU_DEPTH16: return 2;
        case GOOSE_GPU_RGBA16F: case GOOSE_GPU_RG32F: return 8;
        case GOOSE_GPU_RGBA32F: return 16;
        case GOOSE_GPU_DEPTH32F: case GOOSE_GPU_DEPTH: return 4;
        default: return 0;
    }
}

static int ready(const char *what) {
    if (backend) return 1;
    goose_gpu_misuse("%s before gpu::open_headless", what);
    return 0;
}

static int outside_pass(const char *what) {
    if (pass == PASS_NONE) return 1;
    goose_gpu_misuse("%s inside a %s pass", what, pass == PASS_RENDER ? "render" : "compute");
    return 0;
}

uint8_t goose_gpu_open(int64_t flags) {
    (void)flags;
    if (backend) {
        goose_gpu_fail("the device is already open");
        return 0;
    }
    const char *want = getenv("GOOSE_GPU_BACKEND");
#ifdef __APPLE__
    const goose_gpu_backend *pick = &goose_gpu_metal;
#else
    const goose_gpu_backend *pick = &goose_gpu_vulkan;
#endif
    if (want && *want) {
        if (!strcmp(want, "vulkan")) pick = &goose_gpu_vulkan;
#ifdef __APPLE__
        else if (!strcmp(want, "metal")) pick = &goose_gpu_metal;
#endif
        else {
            goose_gpu_fail("GOOSE_GPU_BACKEND names no backend this build has: %s", want);
            return 0;
        }
    }
    if (!pick->open()) return 0;
    backend = pick;
    return 1;
}

static void release_pipeline(goose_gpu_pipe *p) {
    backend->pipeline_release(p);
    free(p->library);
    memset(p, 0, sizeof *p);
}

void goose_gpu_close(void) {
    if (!backend) return;
    backend->wait(backend->submit());
    for (int i = 0; i < MAX_READBACKS; i++)
        if (readbacks[i].used) {
            backend->readback_release(readbacks[i].native);
            memset(&readbacks[i], 0, sizeof readbacks[i]);
        }
    for (int i = 0; i < GOOSE_GPU_MAX_PIPELINES; i++)
        if (pipelines[i].used) release_pipeline(&pipelines[i]);
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (slotkind[s]) {
            backend->release(s, slotkind[s] == SLOT_TEXTURE);
            slotkind[s] = SLOT_FREE;
        }
    backend->close();
    backend = NULL;
    pass = PASS_NONE;
    free(uniforms);
    uniforms = NULL;
    uniformlen = 0;
    goose_gpu_check();
}

void goose_gpu_error(gs_rref out) { gs_bld_append(out, lasterror, (int64_t)strlen(lasterror)); }

void goose_gpu_driver(gs_rref out) {
    const char *n = backend ? backend->name : "";
    gs_bld_append(out, n, (int64_t)strlen(n));
}

static uint32_t free_slot(void) {
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (!slotkind[s]) return s;
    goose_gpu_fail("the resource heap is full (%d slots)", GOOSE_GPU_HEAP_SLOTS);
    return 0;
}

uint32_t goose_gpu_buffer(int64_t usage, goose_gpu_bytes data, int64_t size) {
    if (!ready("gpu::buffer")) return 0;
    if (size <= 0 || data.len > size) {
        goose_gpu_misuse("gpu::buffer of %lld bytes", (long long)size);
        return 0;
    }
    uint32_t slot = free_slot();
    if (!slot) return 0;
    if (!backend->buffer_create(slot, usage, data.len ? data.data : NULL, size)) return 0;
    slotkind[slot] = SLOT_BUFFER;
    slotsize[slot] = size;
    return slot;
}

int64_t goose_gpu_buffer_size(uint32_t slot) {
    return slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_BUFFER ? slotsize[slot] : 0;
}

uint32_t goose_gpu_texture(goose_gpu_texture_desc d, goose_gpu_bytes data) {
    if (!ready("gpu::texture")) return 0;
    if (d.kind != 0 || d.width <= 0 || d.height <= 0 || d.format <= 0 ||
        d.format >= GOOSE_GPU_FORMAT_END) {
        goose_gpu_misuse("gpu::texture of kind %d, format %d, %dx%d: 2D textures only, so far",
                         d.kind, d.format, d.width, d.height);
        return 0;
    }
    if (d.mips < 1) d.mips = 1;
    if (d.samples < 1) d.samples = 1;
    if (d.depth < 1) d.depth = 1;
    if (data.len) {
        int64_t want = (int64_t)d.width * d.height * goose_gpu_texel_size(d.format);
        if (!want || data.len != want) {
            goose_gpu_misuse("gpu::texture with %lld bytes, where its mip 0 is %lld",
                             (long long)data.len, (long long)want);
            return 0;
        }
    }
    uint32_t slot = free_slot();
    if (!slot) return 0;
    if (!backend->texture_create(slot, &d, data.len ? data.data : NULL, data.len)) return 0;
    slotkind[slot] = SLOT_TEXTURE;
    textures[slot].desc = d;
    textures[slot].access = data.len ? GOOSE_GPU_ACCESS_SHADER_READ : GOOSE_GPU_ACCESS_NONE;
    return slot;
}

void goose_gpu_texture_info(uint32_t slot, goose_gpu_texture_desc *out) {
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    if (t) *out = t->desc;
    else memset(out, 0, sizeof *out);
}

void goose_gpu_release(uint32_t slot) {
    if (!backend || !slot) return;
    if (slot >= GOOSE_GPU_HEAP_SLOTS || !slotkind[slot]) {
        goose_gpu_misuse("gpu::release of %u, which is no buffer or texture", slot);
        return;
    }
    backend->release(slot, slotkind[slot] == SLOT_TEXTURE);
    slotkind[slot] = SLOT_FREE;
}

static uint32_t new_pipeline(goose_gpu_bytes lib, int compute) {
    gs_gpu_lib_header h;
    if (lib.len < (int64_t)sizeof h) {
        goose_gpu_fail("not a shader library: %lld bytes", (long long)lib.len);
        return 0;
    }
    memcpy(&h, lib.data, sizeof h);
    if (memcmp(h.magic, GS_GPU_LIB_MAGIC, 4) || h.version != GS_GPU_LIB_VERSION) {
        goose_gpu_fail("a shader library of another version: rebuild the program with this "
                       "package's compiler");
        return 0;
    }
    if ((int)h.entry_count != (compute ? 1 : 2) ||
        lib.len < (int64_t)(sizeof h + h.entry_count * sizeof(gs_gpu_lib_entry))) {
        goose_gpu_fail("%s takes %s", compute ? "gpu::compute_pipeline" : "gpu::pipeline",
                       compute ? "a compute entry point" : "a vertex and a fragment entry point");
        return 0;
    }
    uint32_t id = 0;
    for (uint32_t i = 1; i < GOOSE_GPU_MAX_PIPELINES; i++)
        if (!pipelines[i].used) {
            id = i;
            break;
        }
    if (!id) {
        goose_gpu_fail("too many pipelines (%d)", GOOSE_GPU_MAX_PIPELINES);
        return 0;
    }
    goose_gpu_pipe *p = &pipelines[id];
    memset(p, 0, sizeof *p);
    p->compute = compute;
    p->header = h;
    memcpy(p->entries, lib.data + sizeof h, h.entry_count * sizeof(gs_gpu_lib_entry));
    p->library = malloc((size_t)lib.len);
    memcpy(p->library, lib.data, (size_t)lib.len);
    p->librarylen = lib.len;
    return id;
}

uint32_t goose_gpu_compute_pipeline(goose_gpu_bytes lib) {
    if (!ready("gpu::compute_pipeline")) return 0;
    uint32_t id = new_pipeline(lib, 1);
    if (!id) return 0;
    goose_gpu_pipe *p = &pipelines[id];
    if (!backend->compute_create(p)) {
        free(p->library);
        memset(p, 0, sizeof *p);
        return 0;
    }
    p->used = 1;
    return id;
}

uint32_t goose_gpu_pipeline(goose_gpu_bytes lib, goose_gpu_pipeline_desc desc) {
    if (!ready("gpu::pipeline")) return 0;
    uint32_t id = new_pipeline(lib, 0);
    if (!id) return 0;
    pipelines[id].desc = desc;
    pipelines[id].used = 1;
    return id;
}

void goose_gpu_release_pipeline(uint32_t id) {
    if (!backend || !id) return;
    if (id >= GOOSE_GPU_MAX_PIPELINES || !pipelines[id].used) {
        goose_gpu_misuse("gpu::release of pipeline %u, which is not one", id);
        return;
    }
    if (bound == &pipelines[id]) bound = NULL;
    release_pipeline(&pipelines[id]);
}

uint8_t goose_gpu_begin_compute(void) {
    if (!ready("gpu::begin_compute") || !outside_pass("gpu::begin_compute")) return 0;
    if (!backend->begin_compute()) return 0;
    pass = PASS_COMPUTE;
    bound = NULL;
    return 1;
}

void goose_gpu_end_compute(void) {
    if (pass != PASS_COMPUTE) {
        goose_gpu_misuse("gpu::end_compute outside a compute pass");
        return;
    }
    backend->end_compute();
    pass = PASS_NONE;
}

/* A pass's targets: textures made for what they are used as, and left in
   that access by a barrier (§3). */
static int target(uint32_t slot, int depth, int *format, int *w, int *h) {
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    const char *what = depth ? "depth target" : "color target";
    if (!t) {
        goose_gpu_misuse("a pass's %s %u is not a texture", what, slot);
        return 0;
    }
    int usage = depth ? GOOSE_GPU_DEPTH_TARGET : GOOSE_GPU_COLOR_TARGET;
    int access = depth ? GOOSE_GPU_ACCESS_DEPTH_WRITE : GOOSE_GPU_ACCESS_RENDER_TARGET;
    if (!(t->desc.usage & usage)) {
        goose_gpu_misuse("texture %u is a pass's %s, and was not made as one", slot, what);
        return 0;
    }
    if (t->access != access) {
        goose_gpu_misuse("texture %u is a pass's %s, and was left in access %d: a gpu::barrier "
                         "to %s goes first", slot, what, t->access,
                         depth ? "ACCESS_DEPTH_WRITE" : "ACCESS_RENDER_TARGET");
        return 0;
    }
    if (*w && (t->desc.width != *w || t->desc.height != *h)) {
        goose_gpu_misuse("a pass's targets differ in size");
        return 0;
    }
    *w = t->desc.width;
    *h = t->desc.height;
    *format = t->desc.format;
    return 1;
}

uint8_t goose_gpu_begin_pass(goose_gpu_pass_desc d) {
    if (!ready("gpu::begin_pass") || !outside_pass("gpu::begin_pass")) return 0;
    memset(&renderpass, 0, sizeof renderpass);
    renderpass.desc = d;
    int w = 0, h = 0;
    for (int i = 0; i < GOOSE_GPU_MAX_TARGETS && d.color[i]; i++) {
        if (!target(d.color[i], 0, &renderpass.formats.colors[i], &w, &h)) return 0;
        renderpass.ncolor = i + 1;
    }
    if (d.depth && !target(d.depth, 1, &renderpass.formats.depth, &w, &h)) return 0;
    if (!w) {
        goose_gpu_misuse("gpu::begin_pass with no targets");
        return 0;
    }
    renderpass.width = w;
    renderpass.height = h;
    if (!backend->begin_pass(&renderpass)) return 0;
    pass = PASS_RENDER;
    bound = NULL;
    boundnative = NULL;
    indexsize = 0;
    return 1;
}

void goose_gpu_end_pass(void) {
    if (pass != PASS_RENDER) {
        goose_gpu_misuse("gpu::end_pass outside a render pass");
        return;
    }
    backend->end_pass();
    pass = PASS_NONE;
}

/* The native pipeline for the pass's target formats, made the first time
   the pipeline is bound in a pass with them. */
static void *variant(goose_gpu_pipe *p) {
    goose_gpu_targets *k = &renderpass.formats;
    for (int i = 0; i < p->nvariants; i++)
        if (!memcmp(&p->variants[i].key, k, sizeof *k)) return p->variants[i].native;
    if ((int)p->header.color_outputs != renderpass.ncolor) {
        goose_gpu_misuse("a pipeline writing %u colors bound in a pass with %d targets",
                         p->header.color_outputs, renderpass.ncolor);
        return NULL;
    }
    if (p->nvariants == GOOSE_GPU_PIPELINE_VARIANTS) {
        goose_gpu_misuse("a pipeline bound in passes with more than %d sets of target formats",
                         GOOSE_GPU_PIPELINE_VARIANTS);
        return NULL;
    }
    void *n = backend->raster_create(p, k);
    if (!n) {
        goose_gpu_misuse("the pipeline could not be made: %s", lasterror);
        return NULL;
    }
    p->variants[p->nvariants].key = *k;
    p->variants[p->nvariants].native = n;
    p->nvariants++;
    return n;
}

void goose_gpu_bind(uint32_t id) {
    if (pass == PASS_NONE) {
        goose_gpu_misuse("gpu::bind outside a pass");
        return;
    }
    if (!id || id >= GOOSE_GPU_MAX_PIPELINES || !pipelines[id].used) {
        goose_gpu_misuse("gpu::bind of pipeline %u, which is not one", id);
        return;
    }
    goose_gpu_pipe *p = &pipelines[id];
    if (p->compute != (pass == PASS_COMPUTE)) {
        goose_gpu_misuse("gpu::bind of a %s pipeline in a %s pass", p->compute ? "compute" : "render",
                         pass == PASS_COMPUTE ? "compute" : "render");
        return;
    }
    if (!p->compute) {
        boundnative = variant(p);
        if (!boundnative) return;
    }
    bound = p;
}

void goose_gpu_push_constants(goose_gpu_bytes data) {
    if (data.len > (int64_t)sizeof push) {
        goose_gpu_misuse("push constants of %lld bytes: at most 128", (long long)data.len);
        return;
    }
    memcpy(push, data.data, (size_t)data.len);
    pushlen = (uint32_t)data.len;
}

void goose_gpu_push_uniforms(goose_gpu_bytes data) {
    if (data.len > 65536) {
        goose_gpu_misuse("a uniform buffer of %lld bytes: at most 64 KiB", (long long)data.len);
        return;
    }
    uniforms = realloc(uniforms, data.len ? (size_t)data.len : 1);
    memcpy(uniforms, data.data, (size_t)data.len);
    uniformlen = (uint32_t)data.len;
    uniformgen++;
}

/* The bound pipeline takes the push constants and uniforms set (§3). */
static int bound_ok(const char *what) {
    if (!bound) {
        goose_gpu_misuse("%s with no pipeline bound", what);
        return 0;
    }
    if (pushlen != bound->header.push_size) {
        goose_gpu_misuse("%s with %u bytes of push constants, where the pipeline takes %u", what,
                         pushlen, bound->header.push_size);
        return 0;
    }
    if (bound->header.uniform_size && uniformlen != bound->header.uniform_size) {
        goose_gpu_misuse("%s with a uniform buffer of %u bytes, where the pipeline takes %u",
                         what, uniformlen, bound->header.uniform_size);
        return 0;
    }
    return 1;
}

void goose_gpu_dispatch(int64_t x, int64_t y, int64_t z) {
    if (pass != PASS_COMPUTE) {
        goose_gpu_misuse("gpu::dispatch outside a compute pass");
        return;
    }
    if (!bound_ok("gpu::dispatch")) return;
    if (x <= 0 || y <= 0 || z <= 0) return;
    backend->dispatch(bound, push, pushlen, uniforms, bound->header.uniform_size, uniformgen,
                      (uint32_t)x, (uint32_t)y, (uint32_t)z);
}

void goose_gpu_bind_index_buffer(uint32_t slot, int64_t size) {
    if (pass != PASS_RENDER) {
        goose_gpu_misuse("gpu::bind_index_buffer outside a render pass");
        return;
    }
    if (slot >= GOOSE_GPU_HEAP_SLOTS || slotkind[slot] != SLOT_BUFFER || (size != 2 && size != 4)) {
        goose_gpu_misuse("gpu::bind_index_buffer of %u with %lld-byte indices", slot,
                         (long long)size);
        return;
    }
    indexsize = (int)size;
    backend->index_buffer(slot, (int)size);
}

void goose_gpu_viewport(float x, float y, float w, float h) {
    if (pass == PASS_RENDER) backend->viewport(x, y, w, h);
}

void goose_gpu_scissor(int64_t x, int64_t y, int64_t w, int64_t h) {
    if (pass == PASS_RENDER) backend->scissor((int32_t)x, (int32_t)y, (int32_t)w, (int32_t)h);
}

static void draw(int indexed, int64_t count, int64_t instances, int64_t first, int64_t offset,
                 int64_t first_instance) {
    const char *what = indexed ? "gpu::draw_indexed" : "gpu::draw";
    if (pass != PASS_RENDER) {
        goose_gpu_misuse("%s outside a render pass", what);
        return;
    }
    if (!bound_ok(what)) return;
    if (indexed && !indexsize) {
        goose_gpu_misuse("gpu::draw_indexed with no index buffer bound");
        return;
    }
    if (count <= 0 || instances <= 0) return;
    backend->draw(bound, boundnative, push, pushlen, uniforms, bound->header.uniform_size,
                  uniformgen, indexed, (uint32_t)count, (uint32_t)instances, (uint32_t)first,
                  (int32_t)offset, (uint32_t)first_instance);
}

void goose_gpu_draw(int64_t vertices, int64_t instances, int64_t first, int64_t first_instance) {
    draw(0, vertices, instances, first, 0, first_instance);
}

void goose_gpu_draw_indexed(int64_t indices, int64_t instances, int64_t first, int64_t offset,
                            int64_t first_instance) {
    draw(1, indices, instances, first, offset, first_instance);
}

void goose_gpu_barrier(uint32_t slot, int64_t before, int64_t after) {
    if (!ready("gpu::barrier") || !outside_pass("gpu::barrier")) return;
    goose_gpu_tex *t = slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_TEXTURE
                               ? &textures[slot] : NULL;
    if (!t) {
        if (slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_BUFFER) {
            backend->barrier(slot, (int)before, (int)after);
            return;
        }
        goose_gpu_misuse("gpu::barrier of %u, which is no buffer or texture", slot);
        return;
    }
    if (t->access != (int)before) {
        goose_gpu_misuse("gpu::barrier of texture %u from access %lld, which it was not left in "
                         "(%d)", slot, (long long)before, t->access);
        return;
    }
    backend->barrier(slot, (int)before, (int)after);
    t->access = (int)after;
}

uint64_t goose_gpu_submit(void) {
    if (!ready("gpu::submit")) return 0;
    if (!outside_pass("gpu::submit")) return 0;
    uint64_t f = backend->submit();
    for (int i = 0; i < MAX_READBACKS; i++)
        if (readbacks[i].used && !readbacks[i].fence) readbacks[i].fence = f;
    return f;
}

uint8_t goose_gpu_fence_done(uint64_t f) { return backend && backend->completed() >= f; }

void goose_gpu_wait(uint64_t f) {
    if (backend) backend->wait(f);
    goose_gpu_check();
}

static uint32_t new_readback(void *native, int64_t size) {
    if (!native) return 0;
    for (uint32_t i = 1; i < MAX_READBACKS; i++)
        if (!readbacks[i].used) {
            readbacks[i] = (readback) { 1, native, size, 0 };
            return i;
        }
    backend->readback_release(native);
    goose_gpu_fail("too many read backs in flight (%d)", MAX_READBACKS);
    return 0;
}

uint32_t goose_gpu_read_buffer(uint32_t slot, int64_t offset, int64_t size) {
    if (!ready("gpu::read_buffer") || !outside_pass("gpu::read_buffer")) return 0;
    if (slot >= GOOSE_GPU_HEAP_SLOTS || slotkind[slot] != SLOT_BUFFER || offset < 0 || size <= 0 ||
        offset + size > slotsize[slot]) {
        goose_gpu_misuse("gpu::read_buffer of bytes %lld..%lld of buffer %u", (long long)offset,
                         (long long)(offset + size), slot);
        return 0;
    }
    return new_readback(backend->readback_buffer(slot, offset, size), size);
}

uint32_t goose_gpu_read_texture(uint32_t slot) {
    if (!ready("gpu::read_texture") || !outside_pass("gpu::read_texture")) return 0;
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    if (!t || !goose_gpu_texel_size(t->desc.format) || t->access == GOOSE_GPU_ACCESS_NONE) {
        goose_gpu_misuse("gpu::read_texture of %u: a texture with contents, of a format the "
                         "CPU reads", slot);
        return 0;
    }
    int64_t size = (int64_t)t->desc.width * t->desc.height * goose_gpu_texel_size(t->desc.format);
    return new_readback(backend->readback_texture(slot, t->access, size), size);
}

static readback *readback_of(uint32_t id) {
    if (!id || id >= MAX_READBACKS || !readbacks[id].used) {
        goose_gpu_misuse("read back %u, which is not one", id);
        return NULL;
    }
    return &readbacks[id];
}

uint8_t goose_gpu_readback_ready(uint32_t id) {
    readback *r = readback_of(id);
    return r && backend && r->fence && backend->completed() >= r->fence;
}

uint8_t goose_gpu_readback_bytes(uint32_t id, gs_rref out) {
    if (!goose_gpu_readback_ready(id)) return 0;
    readback *r = &readbacks[id];
    /* Straight into the builder: the stack top is where its bytes go. */
    backend->readback_read(r->native, out.stk->top, r->size);
    out.stk->top += r->size;
    out.hdr->len += r->size;
    return 1;
}

void goose_gpu_release_readback(uint32_t id) {
    readback *r = readback_of(id);
    if (!r) return;
    backend->readback_release(r->native);
    memset(r, 0, sizeof *r);
}
