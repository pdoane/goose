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

/* Heap slots: 1 marks one in use, and its size. */
static uint8_t slotused[GOOSE_GPU_HEAP_SLOTS];
static int64_t slotsize[GOOSE_GPU_HEAP_SLOTS];

static goose_gpu_pipeline pipelines[GOOSE_GPU_MAX_PIPELINES];

/* The recording state between begin_compute and end_compute. */
static int incompute;
static goose_gpu_pipeline *bound;
static uint8_t push[128];
static uint32_t pushlen;

/* Read backs, by id from 1: each a copy taken when its fence completes. */
typedef struct {
    int used;
    uint32_t slot;
    int64_t offset, size;
    uint64_t fence;         /* 0 until the submission carrying it */
    uint8_t *data;          /* once copied */
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

static int ready(const char *what) {
    if (backend) return 1;
    goose_gpu_misuse("%s before gpu::open_headless", what);
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

void goose_gpu_close(void) {
    if (!backend) return;
    for (int i = 0; i < MAX_READBACKS; i++) {
        free(readbacks[i].data);
        memset(&readbacks[i], 0, sizeof readbacks[i]);
    }
    for (int i = 0; i < GOOSE_GPU_MAX_PIPELINES; i++)
        if (pipelines[i].used) {
            backend->pipeline_release(&pipelines[i]);
            memset(&pipelines[i], 0, sizeof pipelines[i]);
        }
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (slotused[s]) {
            backend->buffer_release(s);
            slotused[s] = 0;
        }
    backend->close();
    backend = NULL;
    goose_gpu_check();
}

void goose_gpu_error(gs_rref out) { gs_bld_append(out, lasterror, (int64_t)strlen(lasterror)); }

void goose_gpu_driver(gs_rref out) {
    const char *n = backend ? backend->name : "";
    gs_bld_append(out, n, (int64_t)strlen(n));
}

uint32_t goose_gpu_buffer(int64_t usage, goose_gpu_bytes data, int64_t size) {
    if (!ready("gpu::buffer")) return 0;
    if (size <= 0 || data.len > size) {
        goose_gpu_misuse("gpu::buffer of %lld bytes", (long long)size);
        return 0;
    }
    uint32_t slot = 0;
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (!slotused[s]) {
            slot = s;
            break;
        }
    if (!slot) {
        goose_gpu_fail("the resource heap is full (%d slots)", GOOSE_GPU_HEAP_SLOTS);
        return 0;
    }
    if (!backend->buffer_create(slot, usage, data.len ? data.data : NULL, size)) return 0;
    slotused[slot] = 1;
    slotsize[slot] = size;
    return slot;
}

int64_t goose_gpu_buffer_size(uint32_t slot) {
    return slot < GOOSE_GPU_HEAP_SLOTS && slotused[slot] ? slotsize[slot] : 0;
}

void goose_gpu_release_buffer(uint32_t slot) {
    if (!backend || !slot) return;
    if (slot >= GOOSE_GPU_HEAP_SLOTS || !slotused[slot]) {
        goose_gpu_misuse("gpu::release of buffer %u, which is not one", slot);
        return;
    }
    backend->buffer_release(slot);
    slotused[slot] = 0;
}

uint32_t goose_gpu_compute_pipeline(goose_gpu_bytes lib) {
    if (!ready("gpu::compute_pipeline")) return 0;
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
    if (h.stage != GS_GPU_STAGE_COMPUTE) {
        goose_gpu_fail("gpu::compute_pipeline takes a compute entry point");
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
    goose_gpu_pipeline *p = &pipelines[id];
    memset(p, 0, sizeof *p);
    p->header = h;
    if (!backend->pipeline_create(p, lib.data, lib.len)) return 0;
    p->used = 1;
    return id;
}

void goose_gpu_release_pipeline(uint32_t id) {
    if (!backend || !id) return;
    if (id >= GOOSE_GPU_MAX_PIPELINES || !pipelines[id].used) {
        goose_gpu_misuse("gpu::release of pipeline %u, which is not one", id);
        return;
    }
    if (bound == &pipelines[id]) bound = NULL;
    backend->pipeline_release(&pipelines[id]);
    memset(&pipelines[id], 0, sizeof pipelines[id]);
}

uint8_t goose_gpu_begin_compute(void) {
    if (!ready("gpu::begin_compute")) return 0;
    if (incompute) {
        goose_gpu_misuse("gpu::begin_compute inside a compute pass");
        return 0;
    }
    if (!backend->begin_compute()) return 0;
    incompute = 1;
    bound = NULL;
    return 1;
}

void goose_gpu_end_compute(void) {
    if (!incompute) {
        goose_gpu_misuse("gpu::end_compute outside a compute pass");
        return;
    }
    backend->end_compute();
    incompute = 0;
}

void goose_gpu_bind_compute(uint32_t id) {
    if (!incompute) {
        goose_gpu_misuse("gpu::bind of a compute pipeline outside a compute pass");
        return;
    }
    if (!id || id >= GOOSE_GPU_MAX_PIPELINES || !pipelines[id].used) {
        goose_gpu_misuse("gpu::bind of pipeline %u, which is not one", id);
        return;
    }
    bound = &pipelines[id];
}

void goose_gpu_push_constants(goose_gpu_bytes data) {
    if (data.len > (int64_t)sizeof push) {
        goose_gpu_misuse("push constants of %lld bytes: at most 128", (long long)data.len);
        return;
    }
    memcpy(push, data.data, (size_t)data.len);
    pushlen = (uint32_t)data.len;
}

void goose_gpu_dispatch(int64_t x, int64_t y, int64_t z) {
    if (!incompute || !bound) {
        goose_gpu_misuse("gpu::dispatch with no compute pipeline bound");
        return;
    }
    if (pushlen != bound->header.push_size) {
        goose_gpu_misuse("gpu::dispatch with %u bytes of push constants, where the pipeline "
                         "takes %u", pushlen, bound->header.push_size);
        return;
    }
    if (x <= 0 || y <= 0 || z <= 0) return;
    backend->dispatch(bound, push, pushlen, (uint32_t)x, (uint32_t)y, (uint32_t)z);
}

uint64_t goose_gpu_submit(void) {
    if (!ready("gpu::submit")) return 0;
    if (incompute) {
        goose_gpu_misuse("gpu::submit inside a compute pass");
        return 0;
    }
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

uint32_t goose_gpu_read_buffer(uint32_t slot, int64_t offset, int64_t size) {
    if (!ready("gpu::read_buffer")) return 0;
    if (slot >= GOOSE_GPU_HEAP_SLOTS || !slotused[slot] || offset < 0 || size <= 0 ||
        offset + size > slotsize[slot]) {
        goose_gpu_misuse("gpu::read_buffer of bytes %lld..%lld of buffer %u", (long long)offset,
                         (long long)(offset + size), slot);
        return 0;
    }
    for (uint32_t i = 1; i < MAX_READBACKS; i++)
        if (!readbacks[i].used) {
            readbacks[i] = (readback) { 1, slot, offset, size, 0, NULL };
            return i;
        }
    goose_gpu_fail("too many read backs in flight (%d)", MAX_READBACKS);
    return 0;
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
    if (!r || !backend) return 0;
    if (r->data) return 1;
    if (!r->fence || backend->completed() < r->fence) return 0;
    r->data = (uint8_t *)malloc((size_t)r->size);
    backend->buffer_read(r->slot, r->offset, r->size, r->data);
    return 1;
}

uint8_t goose_gpu_readback_bytes(uint32_t id, gs_rref out) {
    if (!goose_gpu_readback_ready(id)) return 0;
    readback *r = &readbacks[id];
    gs_bld_append(out, r->data, r->size);
    return 1;
}

void goose_gpu_release_readback(uint32_t id) {
    readback *r = readback_of(id);
    if (!r) return;
    free(r->data);
    memset(r, 0, sizeof *r);
}
