/* The shader library `gpu_entry` produces (docs/design/gpu.md §6): the
   entry points of one pipeline, a compute entry alone or a vertex entry
   and a fragment entry, compiled from Goose when the program is, into every
   format the target's backends take, with what creating a pipeline for
   them needs to know. The compiler writes it (src/gpu.h) and embeds it in
   the program as static data; the gpu package reads it back from its own
   copy of this header, which the version below keeps in step. Plain C, so
   both can read it.

   Layout, little-endian and unaligned (a library sits in a byte array):

       gs_gpu_lib_header
       gs_gpu_lib_entry entries[entry_count]
       the entries' names and each format's code, at the offsets the
       entries give; a format the target has no backend for has size 0;
       and the fault sites, at the offset the header gives.

   A fault site is a check a shader makes that the CPU would abort on
   (docs/design/gpu_language.md §6): a line of text per site, its id, a
   tab and the source place, `12\tblur.goose:41`. Ids are unique across
   the program's libraries, so a fault names its site in whichever.

   Readers copy the structs out with memcpy rather than casting a pointer
   to them: nothing aligns the array holding the library. */

#ifndef GS_GPU_LIBRARY_H
#define GS_GPU_LIBRARY_H

#include <stdint.h>

#define GS_GPU_LIB_MAGIC "GSGL"
#define GS_GPU_LIB_VERSION 3

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
    uint16_t entry_count;           /* 1 for compute, 2 for vertex and fragment */
    uint32_t push_size;             /* bytes of push constants, 0 for none */
    uint32_t uniform_size;          /* bytes of the uniform buffer, 0 for none */
    uint32_t caps;                  /* GS_GPU_CAP_*, of every entry */
    uint32_t color_outputs;         /* a fragment entry's color targets */
    gs_gpu_lib_range sites;         /* the fault sites' text */
} gs_gpu_lib_header;

/* A fault, as a shader records it in the buffer at heap index 0: the
   first wins. */
enum {
    GS_GPU_FAULT_INDEX = 1,         /* a: the index, b: the length */
    GS_GPU_FAULT_DIVIDE = 2,        /* division or remainder by zero */
    GS_GPU_FAULT_TAG = 3,           /* a: an ADT's tag read from memory, b: its variants */
    GS_GPU_FAULT_FULL = 4,          /* a push past a limited array's capacity, b */
    GS_GPU_FAULT_EMPTY = 5,         /* a pop from an empty limited array */
};

typedef struct {
    uint32_t site;                  /* the site's id + 1; 0 while there is no fault */
    uint32_t kind;                  /* GS_GPU_FAULT_* */
    uint32_t a, b;
    uint32_t invocation[3];         /* the dispatch thread, vertex or pixel */
} gs_gpu_fault;

typedef struct {
    uint8_t stage;
    uint8_t reserved[3];
    uint32_t group_size[3];         /* compute only; 1 elsewhere */
    gs_gpu_lib_range name;          /* the entry point's name in every format */
    gs_gpu_lib_range code[GS_GPU_FORMAT_COUNT];
} gs_gpu_lib_entry;
#pragma pack(pop)

#endif
