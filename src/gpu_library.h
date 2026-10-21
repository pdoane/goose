/* The shader library `gpu_entry(f)` produces (docs/design/gpu.md §6): one
   GPU entry point, compiled from Goose when the program is, into every
   format the target's backends take, with what creating a pipeline for it
   needs to know. The compiler writes it (src/gpu.h) and embeds it in the
   program as static data; the gpu package reads it back from its own copy
   of this header, which the version below keeps in step. Plain C, so both
   can read it.

   Layout, little-endian and unaligned (a library sits in a byte array):

       gs_gpu_lib_header
       the entry's name, then each format's code, at the offsets the header
       gives; a format the target has no backend for has size 0.

   Readers copy the header out with memcpy rather than casting a pointer to
   it: nothing aligns the array holding the library. */

#ifndef GS_GPU_LIBRARY_H
#define GS_GPU_LIBRARY_H

#include <stdint.h>

#define GS_GPU_LIB_MAGIC "GSGL"
#define GS_GPU_LIB_VERSION 1

enum {
    GS_GPU_STAGE_VERTEX = 0,
    GS_GPU_STAGE_FRAGMENT = 1,
    GS_GPU_STAGE_COMPUTE = 2,
};

/* The formats, by the backend that takes each. */
enum {
    GS_GPU_FORMAT_SPIRV = 0,        /* Vulkan */
    GS_GPU_FORMAT_METALLIB = 1,     /* Metal */
    GS_GPU_FORMAT_DXIL = 2,         /* Direct3D 12 */
    GS_GPU_FORMAT_COUNT = 3,
};

/* What a device must offer to run the entry point (gpu::caps()). */
enum {
    GS_GPU_CAP_RAY_QUERY = 1 << 0,
    GS_GPU_CAP_FLOAT64 = 1 << 1,
    GS_GPU_CAP_FLOAT16_OPS = 1 << 2,
};

#pragma pack(push, 1)
typedef struct {
    uint32_t offset, size;
} gs_gpu_lib_range;

typedef struct {
    char magic[4];
    uint16_t version;
    uint8_t stage;
    uint8_t reserved;
    uint32_t group_size[3];         /* compute only; 1 elsewhere */
    uint32_t push_size;             /* bytes of push constants, 0 for none */
    uint32_t uniform_size;          /* bytes of the uniform buffer, 0 for none */
    uint32_t caps;                  /* GS_GPU_CAP_* */
    gs_gpu_lib_range name;          /* the entry point's name in every format */
    gs_gpu_lib_range code[GS_GPU_FORMAT_COUNT];
} gs_gpu_lib_header;
#pragma pack(pop)

#endif
