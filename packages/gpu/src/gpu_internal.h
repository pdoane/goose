/* The gpu package's C (docs/design/gpu.md §3): the layer the Goose externs
   call (gpu.c), over one backend at a time (gpu_metal.m, gpu_vulkan.c).
   The common layer owns everything the backends need not differ on: slots
   and their handles, the recording state, read backs, errors. A backend
   owns the device objects. */

#ifndef GOOSE_GPU_INTERNAL_H
#define GOOSE_GPU_INTERNAL_H

#include <stdint.h>
#include <stddef.h>

#include "gpu_library.h"

/* A Goose `const u8[:]`, as an extern fn passes it (spec §7.10). */
typedef struct { const uint8_t *data; int64_t len; } goose_gpu_bytes;

/* The resource heap's size, in slots. Slot 0 is the stream's (§3), unused
   so far, which keeps 0 meaning "no buffer". */
#define GOOSE_GPU_HEAP_SLOTS 16384
#define GOOSE_GPU_MAX_PIPELINES 1024

/* A pipeline as the common layer knows it: what its library said. */
typedef struct {
    int used;
    gs_gpu_lib_header header;
    void *native;           /* the backend's */
} goose_gpu_pipeline;

typedef struct goose_gpu_backend {
    const char *name;
    /* 0 on failure, with goose_gpu_fail called. */
    int (*open)(void);
    void (*close)(void);
    /* A buffer of `size` bytes in heap slot `slot`, holding `data` when
       that is not null, and visible to shaders at that index. */
    int (*buffer_create)(uint32_t slot, int64_t usage, const void *data, int64_t size);
    void (*buffer_release)(uint32_t slot);
    /* Reads `size` bytes from `offset` of a buffer whose writes have
       completed, into `out`. */
    void (*buffer_read)(uint32_t slot, int64_t offset, int64_t size, void *out);
    int (*pipeline_create)(goose_gpu_pipeline *p, const uint8_t *library, int64_t len);
    void (*pipeline_release)(goose_gpu_pipeline *p);
    int (*begin_compute)(void);
    void (*end_compute)(void);
    void (*dispatch)(goose_gpu_pipeline *p, const void *push, uint32_t pushsize,
                     uint32_t x, uint32_t y, uint32_t z);
    /* Submits what was recorded, and returns its fence, from 1 up. */
    uint64_t (*submit)(void);
    /* The latest fence that has completed. */
    uint64_t (*completed)(void);
    void (*wait)(uint64_t fence);
} goose_gpu_backend;

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
