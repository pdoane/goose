/* The gpu package's common layer (gpu_internal.h): what the Goose externs
   in gpu.goose call. One device at a time, as gfx has one. */

#include "gpu_internal.h"
#include "goose.h"

#ifdef GOOSE_GPU_SDL
#include <SDL3/SDL.h>
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const goose_gpu_backend *backend;
static char lasterror[1024];
static int misused;

/* Heap slots: what each holds, and its size or texture. */
enum { SLOT_FREE, SLOT_BUFFER, SLOT_TEXTURE, SLOT_ACCEL, SLOT_VIEW };
static uint8_t accelinstance[GOOSE_GPU_HEAP_SLOTS];    /* an instance structure */
static int64_t accelmax[GOOSE_GPU_HEAP_SLOTS];
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

/* The window and its screen (§3): the slot the backend puts each frame's
   image in, or with no window a render target of the size asked for; and
   a depth target the screen's size. */
static void *window;
static uint32_t screenslot, depthslot;
static int screenacquired;
static int screenw, screenh;
static int quitting;
static uint64_t framecount;
static double now, lastframe, delta;
#define MAX_KEYS 512
static uint8_t keys[MAX_KEYS], lastkeys[MAX_KEYS];
static float mousex, mousey;
static uint32_t mousebuttons;

/* Read backs, by id from 1: each a copy recorded where it was asked for,
   ready once the submission carrying it completes. */
typedef struct {
    int used;
    void *native;
    int64_t size;
    uint64_t fence;         /* 0 until the submission carrying it */
    /* save_png's: the file written once the pixels arrive, and how. */
    char *png;
    int width, height, bgra, opaque;
} readback;
#define MAX_READBACKS 256
static readback readbacks[MAX_READBACKS];
static void note_readbacks(uint64_t f);
static void drop_readback(readback *r);
static void write_pngs(void);

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

int goose_gpu_filters(int format) {
    return !goose_gpu_is_depth(format) && !goose_gpu_block_size(format) &&
           format != GOOSE_GPU_R32UI && format != GOOSE_GPU_RGBA8UI;
}

int goose_gpu_block_size(int format) {
    switch (format) {
        case GOOSE_GPU_BC1: case GOOSE_GPU_BC1_SRGB: case GOOSE_GPU_BC4: return 8;
        case GOOSE_GPU_BC3: case GOOSE_GPU_BC3_SRGB: case GOOSE_GPU_BC5: case GOOSE_GPU_BC7:
        case GOOSE_GPU_BC7_SRGB:
            return 16;
        default: return 0;
    }
}

int64_t goose_gpu_row_bytes(int format, int width) {
    int block = goose_gpu_block_size(format);
    if (block) return (int64_t)((width + 3) / 4) * block;
    return (int64_t)width * goose_gpu_texel_size(format);
}

int goose_gpu_rows(int format, int height) {
    return goose_gpu_block_size(format) ? (height + 3) / 4 : height;
}

int64_t goose_gpu_mip0_bytes(const goose_gpu_texture_desc *d) {
    return goose_gpu_row_bytes(d->format, d->width) * goose_gpu_rows(d->format, d->height);
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

static double clock_seconds(void) {
#ifdef GOOSE_GPU_SDL
    return (double)SDL_GetTicksNS() / 1e9;
#else
    return 0;
#endif
}

static const goose_gpu_backend *pick_backend(void) {
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
            return NULL;
        }
    }
    return pick;
}

uint32_t goose_gpu_texture(goose_gpu_texture_desc d, goose_gpu_bytes data);
void goose_gpu_release(uint32_t slot);

/* The screen's depth target, made again whenever the screen's size
   changes. */
static void make_depth(void) {
    if (depthslot) goose_gpu_release(depthslot);
    goose_gpu_texture_desc d = { 0, GOOSE_GPU_DEPTH, GOOSE_GPU_DEPTH_TARGET, screenw, screenh, 1, 1, 1 };
    goose_gpu_bytes none = { NULL, 0 };
    depthslot = goose_gpu_texture(d, none);
}

static void reset_frame_state(void) {
    framecount = 0;
    now = lastframe = clock_seconds();
    delta = 0;
    quitting = 0;
    memset(keys, 0, sizeof keys);
    memset(lastkeys, 0, sizeof lastkeys);
}

uint8_t goose_gpu_open(int64_t flags) {
    (void)flags;
    if (backend) {
        goose_gpu_fail("the device is already open");
        return 0;
    }
    const goose_gpu_backend *pick = pick_backend();
    if (!pick || !pick->open(NULL, 1)) return 0;
    backend = pick;
    reset_frame_state();
    return 1;
}

/* gfx's window flags. */
enum { WINDOW_RESIZABLE = 1, WINDOW_HIDDEN = 2, WINDOW_FULLSCREEN = 4, NO_VSYNC = 16 };

uint8_t goose_gpu_open_window(goose_gpu_bytes title, int64_t width, int64_t height,
                              int64_t flags) {
    if (backend) {
        goose_gpu_fail("the device is already open");
        return 0;
    }
    if (width <= 0 || height <= 0) {
        goose_gpu_fail("a window of %lldx%lld", (long long)width, (long long)height);
        return 0;
    }
    const goose_gpu_backend *pick = pick_backend();
    if (!pick) return 0;
    const char *headless = getenv("GOOSE_GPU_HEADLESS");
    if (headless && *headless && strcmp(headless, "0")) {
        /* No window: the screen is a render target of the size asked for. */
        if (!pick->open(NULL, 1)) return 0;
        backend = pick;
        screenw = (int)width;
        screenh = (int)height;
        goose_gpu_texture_desc d = { 0, GOOSE_GPU_BGRA8, GOOSE_GPU_COLOR_TARGET | GOOSE_GPU_SAMPLED,
                                     screenw, screenh, 1, 1, 1 };
        goose_gpu_bytes none = { NULL, 0 };
        screenslot = goose_gpu_texture(d, none);
        make_depth();
        reset_frame_state();
        return screenslot && depthslot;
    }
#ifdef GOOSE_GPU_SDL
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        goose_gpu_fail("SDL: %s", SDL_GetError());
        return 0;
    }
    char name[256];
    int64_t n = title.len < 255 ? title.len : 255;
    memcpy(name, title.data, (size_t)n);
    name[n] = 0;
    SDL_WindowFlags wf = SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (flags & WINDOW_RESIZABLE) wf |= SDL_WINDOW_RESIZABLE;
    if (flags & WINDOW_HIDDEN) wf |= SDL_WINDOW_HIDDEN;
    if (flags & WINDOW_FULLSCREEN) wf |= SDL_WINDOW_FULLSCREEN;
#ifdef __APPLE__
    wf |= pick == &goose_gpu_metal ? SDL_WINDOW_METAL : SDL_WINDOW_VULKAN;
#else
    wf |= SDL_WINDOW_VULKAN;
#endif
    if (wf & SDL_WINDOW_VULKAN) {
        /* SDL loads the Vulkan loader the backend does. */
        extern const char *goose_gpu_vulkan_loader_path(void);
        const char *loaderpath = goose_gpu_vulkan_loader_path();
        if (loaderpath) SDL_SetHint(SDL_HINT_VULKAN_LIBRARY, loaderpath);
    }
    SDL_Window *w = SDL_CreateWindow(name, (int)width, (int)height, wf);
    if (!w) {
        goose_gpu_fail("SDL: %s", SDL_GetError());
        SDL_Quit();
        return 0;
    }
    if (!pick->open(w, !(flags & NO_VSYNC))) {
        SDL_DestroyWindow(w);
        SDL_Quit();
        return 0;
    }
    window = w;
    backend = pick;
    SDL_GetWindowSizeInPixels(w, &screenw, &screenh);
    /* The screen's slot, which holds each frame's image once acquired. */
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS && !screenslot; s++)
        if (!slotkind[s]) screenslot = s;
    slotkind[screenslot] = SLOT_TEXTURE;
    textures[screenslot].desc = (goose_gpu_texture_desc) { 0, GOOSE_GPU_BGRA8,
                                                           GOOSE_GPU_COLOR_TARGET, screenw,
                                                           screenh, 1, 1, 1 };
    textures[screenslot].access = GOOSE_GPU_ACCESS_NONE;
    make_depth();
    reset_frame_state();
    return depthslot != 0;
#else
    (void)title; (void)flags;
    goose_gpu_fail("this build of the gpu package has no SDL3, so no windows: install SDL3, "
                   "or set GOOSE_GPU_HEADLESS=1");
    return 0;
#endif
}

static void release_pipeline(goose_gpu_pipe *p) {
    backend->pipeline_release(p);
    free(p->library);
    memset(p, 0, sizeof *p);
}

void goose_gpu_close(void) {
    if (!backend) return;
    uint64_t last = backend->submit(0);
    note_readbacks(last);
    backend->wait(last);
    write_pngs();
    screenslot = depthslot = 0;
    screenacquired = 0;
    for (int i = 0; i < MAX_READBACKS; i++)
        if (readbacks[i].used) drop_readback(&readbacks[i]);
    for (int i = 0; i < GOOSE_GPU_MAX_PIPELINES; i++)
        if (pipelines[i].used) release_pipeline(&pipelines[i]);
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (slotkind[s] == SLOT_VIEW) {
            backend->release(s, GOOSE_GPU_KIND_VIEW);
            slotkind[s] = SLOT_FREE;
        }
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (slotkind[s]) {
            backend->release(s, slotkind[s] - SLOT_BUFFER);
            slotkind[s] = SLOT_FREE;
        }
    backend->close();
    backend = NULL;
    pass = PASS_NONE;
#ifdef GOOSE_GPU_SDL
    if (window) {
        SDL_DestroyWindow((SDL_Window *)window);
        window = NULL;
        SDL_Quit();
    }
#endif
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
    /* Mips 0: the full chain, down to 1x1, as gfx's. */
    int chain = 1;
    while (((d.width > d.height ? d.width : d.height) >> chain) > 0) chain++;
    if (d.mips == 0) d.mips = chain;
    if (d.mips < 0 || d.mips > chain) {
        goose_gpu_misuse("gpu::texture of %dx%d with %d mips: at most %d", d.width, d.height,
                         d.mips, chain);
        return 0;
    }
    if (d.samples < 1) d.samples = 1;
    if (d.depth < 1) d.depth = 1;
    if (goose_gpu_block_size(d.format) &&
        (d.width % 4 || d.height % 4 || (d.usage & ~GOOSE_GPU_SAMPLED) || d.mips > 1)) {
        goose_gpu_misuse("gpu::texture of compressed format %d, %dx%d: a sampled texture of one "
                         "mip, its size a multiple of 4", d.format, d.width, d.height);
        return 0;
    }
    if (data.len) {
        int64_t want = goose_gpu_mip0_bytes(&d);
        if (!want || data.len != want) {
            goose_gpu_misuse("gpu::texture with %lld bytes, where its mip 0 is %lld",
                             (long long)data.len, (long long)want);
            return 0;
        }
    }
    uint32_t slot = free_slot();
    if (!slot) return 0;
    /* Known before the backend makes it: its upload reads the description. */
    slotkind[slot] = SLOT_TEXTURE;
    memset(&textures[slot], 0, sizeof textures[slot]);
    textures[slot].desc = d;
    if (!backend->texture_create(slot, &d, data.len ? data.data : NULL, data.len)) {
        slotkind[slot] = SLOT_FREE;
        return 0;
    }
    textures[slot].access = data.len ? GOOSE_GPU_ACCESS_SHADER_READ : GOOSE_GPU_ACCESS_NONE;
    textures[slot].submitted = textures[slot].access;
    return slot;
}

/* Updates (§3): complete before the next submission runs, so the whole of
   it sees them, draws recorded before the call included. */
uint8_t goose_gpu_update_buffer(uint32_t slot, int64_t offset, goose_gpu_bytes data) {
    if (!ready("gpu::update_buffer")) return 0;
    if (slot >= GOOSE_GPU_HEAP_SLOTS || slotkind[slot] != SLOT_BUFFER || offset < 0 ||
        offset + data.len > slotsize[slot]) {
        goose_gpu_misuse("gpu::update_buffer of bytes %lld..%lld of buffer %u", (long long)offset,
                         (long long)(offset + data.len), slot);
        return 0;
    }
    if (!data.len) return 1;
    return (uint8_t)backend->update_buffer(slot, offset, data.data, data.len);
}

uint8_t goose_gpu_update_texture(uint32_t slot, goose_gpu_bytes data) {
    if (!ready("gpu::update_texture")) return 0;
    goose_gpu_tex *t = slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_TEXTURE
                           ? &textures[slot] : NULL;
    int64_t want = t ? goose_gpu_mip0_bytes(&t->desc) : 0;
    if (!t || !want || data.len != want || slot == screenslot) {
        goose_gpu_misuse("gpu::update_texture of %u with %lld bytes: a texture's whole mip 0, "
                         "of a format the CPU writes", slot, (long long)data.len);
        return 0;
    }
    if (!backend->update_texture(slot, t->submitted, data.data, data.len)) return 0;
    /* A texture with no contents yet has them now, in shader reads, the
       steady access an update leaves an empty texture in. */
    if (t->submitted == GOOSE_GPU_ACCESS_NONE) {
        if (t->access == GOOSE_GPU_ACCESS_NONE) t->access = GOOSE_GPU_ACCESS_SHADER_READ;
        t->submitted = GOOSE_GPU_ACCESS_SHADER_READ;
    }
    return 1;
}

/* What a submission leaves each texture in, which the next one's updates
   start from. */
static void note_submitted(void) {
    for (uint32_t s = 1; s < GOOSE_GPU_HEAP_SLOTS; s++)
        if (slotkind[s] == SLOT_TEXTURE) textures[s].submitted = textures[s].access;
}

uint8_t goose_gpu_texture_info(uint32_t slot, goose_gpu_texture_desc *out) {
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    if (t) *out = t->desc;
    else memset(out, 0, sizeof *out);
    return t != NULL;
}

/* Mips 1 and up from mip 0 (§3), where the program's work is recorded:
   after it, every mip is in shader reads. */
static int mippable(const goose_gpu_tex *t) {
    return t && t->desc.mips > 1 && (t->desc.usage & GOOSE_GPU_SAMPLED) &&
           goose_gpu_filters(t->desc.format);
}

void goose_gpu_generate_mips(uint32_t slot) {
    if (!ready("gpu::generate_mips") || !outside_pass("gpu::generate_mips")) return;
    goose_gpu_tex *t = slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_TEXTURE
                           ? &textures[slot] : NULL;
    if (!mippable(t) || t->access == GOOSE_GPU_ACCESS_NONE) {
        goose_gpu_misuse("gpu::generate_mips of %u: a sampled texture with more than one mip, "
                         "of a format that filters, with contents in mip 0", slot);
        return;
    }
    if (backend->generate_mips(slot, t->access, 0)) t->access = GOOSE_GPU_ACCESS_SHADER_READ;
}

/* gfx's load_texture flags. */
enum { LOAD_MIPS = 1, LOAD_SRGB = 2 };

#ifdef GOOSE_GPU_SDL
/* An image SDL loaded as an RGBA8 texture, uploaded as texture(desc, data)
   is, with its mips filled in the same upload with LOAD_MIPS. */
static uint32_t surface_texture(SDL_Surface *loaded, int64_t flags, const char *what) {
    if (!loaded) {
        goose_gpu_fail("cannot load %s: %s", what, SDL_GetError());
        return 0;
    }
    SDL_Surface *rgba = SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(loaded);
    if (!rgba) {
        goose_gpu_fail("cannot convert %s: %s", what, SDL_GetError());
        return 0;
    }
    /* Rows as the surface has them, which may be padded. */
    size_t row = (size_t)rgba->w * 4;
    uint8_t *tight = malloc(row * (size_t)rgba->h);
    for (int y = 0; y < rgba->h; y++)
        memcpy(tight + row * (size_t)y, (uint8_t *)rgba->pixels + (size_t)rgba->pitch * (size_t)y,
               row);
    goose_gpu_texture_desc d = { 0, (flags & LOAD_SRGB) ? GOOSE_GPU_RGBA8_SRGB : GOOSE_GPU_RGBA8,
                                 GOOSE_GPU_SAMPLED, rgba->w, rgba->h, 1,
                                 (flags & LOAD_MIPS) ? 0 : 1, 1 };
    goose_gpu_bytes data = { tight, (int64_t)(row * (size_t)rgba->h) };
    uint32_t slot = goose_gpu_texture(d, data);
    free(tight);
    SDL_DestroySurface(rgba);
    if (slot && textures[slot].desc.mips > 1 &&
        !backend->generate_mips(slot, GOOSE_GPU_ACCESS_SHADER_READ, 1)) {
        goose_gpu_release(slot);
        return 0;
    }
    return slot;
}
#endif

/* A PNG or .bmp file as a texture. */
uint32_t goose_gpu_load_texture(goose_gpu_bytes path, int64_t flags) {
    if (!ready("gpu::load_texture")) return 0;
#ifdef GOOSE_GPU_SDL
    char buf[1024];
    size_t n = path.len < 0 ? 0 : (size_t)path.len;
    if (n >= sizeof buf) n = sizeof buf - 1;
    if (n) memcpy(buf, path.data, n);
    buf[n] = 0;
    int bmp = n >= 4 && SDL_strcasecmp(buf + n - 4, ".bmp") == 0;
    return surface_texture(bmp ? SDL_LoadBMP(buf) : SDL_LoadPNG(buf), flags, buf);
#else
    (void)path; (void)flags;
    goose_gpu_fail("this build of the gpu package has no SDL3, so no image files");
    return 0;
#endif
}

/* A PNG or .bmp file's bytes, from memory, as a texture. */
uint32_t goose_gpu_image_texture(goose_gpu_bytes data, int64_t flags) {
    if (!ready("gpu::image_texture")) return 0;
#ifdef GOOSE_GPU_SDL
    SDL_IOStream *io = SDL_IOFromConstMem(data.data, (size_t)data.len);
    int bmp = data.len >= 2 && data.data[0] == 'B' && data.data[1] == 'M';
    return surface_texture(bmp ? SDL_LoadBMP_IO(io, true) : SDL_LoadPNG_IO(io, true), flags,
                           "an image");
#else
    (void)data; (void)flags;
    goose_gpu_fail("this build of the gpu package has no SDL3, so no image files");
    return 0;
#endif
}

void goose_gpu_release(uint32_t slot) {
    if (!backend || !slot) return;
    if (slot >= GOOSE_GPU_HEAP_SLOTS || !slotkind[slot] || slotkind[slot] == SLOT_VIEW) {
        goose_gpu_misuse("gpu::release of %u, which is no buffer or texture", slot);
        return;
    }
    if (slotkind[slot] == SLOT_TEXTURE)
        for (int m = 0; m < 16; m++) {
            uint32_t v = textures[slot].views[m];
            if (v && v != slot) {
                backend->release(v, GOOSE_GPU_KIND_VIEW);
                slotkind[v] = SLOT_FREE;
            }
        }
    backend->release(slot, slotkind[slot] - SLOT_BUFFER);
    slotkind[slot] = SLOT_FREE;
}

/* The slot shaders write mip `mip` of a storage texture through (§3): the
   texture's own for mip 0 of one that is not sampled, else a view made
   the first time it is asked for, released with the texture. */
uint32_t goose_gpu_storage_view(uint32_t slot, int64_t mip) {
    if (!ready("gpu::storage")) return 0;
    goose_gpu_tex *t = slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_TEXTURE
                           ? &textures[slot] : NULL;
    if (!t || !(t->desc.usage & GOOSE_GPU_STORAGE) || mip < 0 || mip >= t->desc.mips) {
        goose_gpu_misuse("gpu::storage of texture %u, mip %lld: a texture made with STORAGE, "
                         "and one of its mips", slot, (long long)mip);
        return 0;
    }
    if (t->views[mip]) return t->views[mip];
    if (mip == 0 && !(t->desc.usage & GOOSE_GPU_SAMPLED) && t->desc.mips == 1) {
        t->views[0] = slot;
        return slot;
    }
    uint32_t v = free_slot();
    if (!v) return 0;
    slotkind[v] = SLOT_VIEW;
    if (!backend->view_create(v, slot, (int)mip)) {
        slotkind[v] = SLOT_FREE;
        return 0;
    }
    t->views[mip] = v;
    return v;
}

/* Ray tracing (§7). */
uint8_t goose_gpu_ray_query(void) { return backend && backend->ray_query(); }

static int buffer_slot(uint32_t slot) {
    return slot && slot < GOOSE_GPU_HEAP_SLOTS && slotkind[slot] == SLOT_BUFFER;
}

static uint32_t new_accel(const goose_gpu_primitive_accel_desc *p,
                          const goose_gpu_instance_accel_desc *i) {
    if (!backend->ray_query()) {
        goose_gpu_fail("the %s device has no ray queries", backend->name);
        return 0;
    }
    uint32_t slot = free_slot();
    if (!slot) return 0;
    slotkind[slot] = SLOT_ACCEL;
    if (!backend->accel_create(slot, p, i)) {
        slotkind[slot] = SLOT_FREE;
        return 0;
    }
    accelinstance[slot] = i != NULL;
    accelmax[slot] = i ? i->max : 0;
    return slot;
}

uint32_t goose_gpu_primitive_accel(goose_gpu_primitive_accel_desc d) {
    if (!ready("gpu::primitive_accel")) return 0;
    int64_t vend = d.vertex_offset + d.vertex_count * d.vertex_stride;
    int64_t iend = d.index_offset + d.triangle_count * 3 * d.index_size;
    if (!buffer_slot(d.vertices) || d.vertex_count <= 0 || d.vertex_stride < 12 ||
        d.triangle_count <= 0 || vend > slotsize[d.vertices] ||
        (d.indices && (!buffer_slot(d.indices) || (d.index_size != 2 && d.index_size != 4) ||
                       iend > slotsize[d.indices])) ||
        (!d.indices && d.triangle_count * 3 > d.vertex_count)) {
        goose_gpu_misuse("gpu::primitive_accel of %lld triangles over %lld vertices, which its "
                         "buffers do not hold", (long long)d.triangle_count,
                         (long long)d.vertex_count);
        return 0;
    }
    return new_accel(&d, NULL);
}

uint32_t goose_gpu_instance_accel(goose_gpu_instance_accel_desc d) {
    if (!ready("gpu::instance_accel")) return 0;
    if (!buffer_slot(d.instances) || d.max <= 0 ||
        d.max * GOOSE_GPU_INSTANCE_SIZE > slotsize[d.instances]) {
        goose_gpu_misuse("gpu::instance_accel of %lld instances, which its buffer does not hold",
                         (long long)d.max);
        return 0;
    }
    return new_accel(NULL, &d);
}

void goose_gpu_build(uint32_t slot, int64_t instances) {
    if (!ready("gpu::build") || !outside_pass("gpu::build")) return;
    if (!slot || slot >= GOOSE_GPU_HEAP_SLOTS || slotkind[slot] != SLOT_ACCEL ||
        (accelinstance[slot] && (instances < 0 || instances > accelmax[slot]))) {
        goose_gpu_misuse("gpu::build of %u with %lld instances", slot, (long long)instances);
        return;
    }
    backend->accel_build(slot, instances);
}

uint64_t goose_gpu_accel_reference(uint32_t slot) {
    if (!backend || !slot || slot >= GOOSE_GPU_HEAP_SLOTS || slotkind[slot] != SLOT_ACCEL ||
        accelinstance[slot]) {
        goose_gpu_misuse("gpu::reference of %u, which is no primitive acceleration structure",
                         slot);
        return 0;
    }
    return backend->accel_reference(slot);
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
    if ((h.caps & GS_GPU_CAP_RAY_QUERY) && !backend->ray_query()) {
        goose_gpu_fail("the pipeline's shaders trace rays, and the %s device has no ray queries",
                       backend->name);
        return 0;
    }
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

void goose_gpu_copy_to_texture(uint32_t buf, int64_t offset, uint32_t slot) {
    if (!ready("gpu::copy_to_texture") || !outside_pass("gpu::copy_to_texture")) return;
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    int64_t size = t ? goose_gpu_mip0_bytes(&t->desc) : 0;
    if (!buffer_slot(buf) || !size || offset < 0 || offset + size > slotsize[buf]) {
        goose_gpu_misuse("gpu::copy_to_texture of buffer %u from %lld into %u: a buffer holding "
                         "the texture's mip 0, of a format the CPU lays out", buf,
                         (long long)offset, slot);
        return;
    }
    if (t->access != GOOSE_GPU_ACCESS_COPY_DST) {
        goose_gpu_misuse("gpu::copy_to_texture into texture %u, left in access %d: a gpu::barrier "
                         "to ACCESS_COPY_DST goes first", slot, t->access);
        return;
    }
    backend->copy_to_texture(buf, offset, slot);
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

/* The read backs recorded since the last submission are carried by this
   one. */
static void note_readbacks(uint64_t f) {
    for (int i = 0; i < MAX_READBACKS; i++)
        if (readbacks[i].used && !readbacks[i].fence) readbacks[i].fence = f;
}

uint64_t goose_gpu_submit(void) {
    if (!ready("gpu::submit")) return 0;
    if (!outside_pass("gpu::submit")) return 0;
    uint64_t f = backend->submit(0);
    note_submitted();
    note_readbacks(f);
    write_pngs();
    return f;
}

/* The screen, its frame's image acquired the first time it is asked for. */
uint32_t goose_gpu_screen(void) {
    if (!ready("gpu::screen")) return 0;
    if (!screenslot) {
        goose_gpu_misuse("gpu::screen with no window: gpu::open makes one");
        return 0;
    }
    if (window && !screenacquired) {
        int w = 0, h = 0;
        if (!backend->acquire(screenslot, &w, &h)) return 0;
        screenacquired = 1;
        textures[screenslot].access = GOOSE_GPU_ACCESS_NONE;
        if (w != screenw || h != screenh) {
            screenw = w;
            screenh = h;
            textures[screenslot].desc.width = w;
            textures[screenslot].desc.height = h;
            make_depth();
        }
    }
    return screenslot;
}

uint32_t goose_gpu_screen_depth(void) { return depthslot; }

/* Goose slices of numbers, as an extern fn passes them (spec §7.10). */
typedef struct { int32_t *data; int64_t len; } i32_slice;
typedef struct { float *data; int64_t len; } f32_slice;

void goose_gpu_screen_size(i32_slice out) {
    if (out.len < 2) return;
    out.data[0] = screenw;
    out.data[1] = screenh;
}

/* A pass over the screen and its depth, cleared, with the barriers that
   put them there placed for the program, as gfx's begin_screen has none to
   place. */
uint8_t goose_gpu_begin_screen(float r, float g, float b, float a) {
    uint32_t s = goose_gpu_screen();
    if (!s || !outside_pass("gpu::begin_screen")) return 0;
    if (textures[s].access != GOOSE_GPU_ACCESS_RENDER_TARGET) {
        backend->barrier(s, textures[s].access, GOOSE_GPU_ACCESS_RENDER_TARGET);
        textures[s].access = GOOSE_GPU_ACCESS_RENDER_TARGET;
    }
    goose_gpu_tex *d = &textures[depthslot];
    if (d->access != GOOSE_GPU_ACCESS_DEPTH_WRITE) {
        backend->barrier(depthslot, d->access, GOOSE_GPU_ACCESS_DEPTH_WRITE);
        d->access = GOOSE_GPU_ACCESS_DEPTH_WRITE;
    }
    goose_gpu_pass_desc pd;
    memset(&pd, 0, sizeof pd);
    pd.color[0] = s;
    pd.depth = depthslot;
    pd.clear_color = pd.clear_depth = 1;
    pd.color_value[0] = r;
    pd.color_value[1] = g;
    pd.color_value[2] = b;
    pd.color_value[3] = a;
    pd.depth_value = 1.0f;
    return goose_gpu_begin_pass(pd);
}

void goose_gpu_quit(void) { quitting = 1; }

static void poll_events(void) {
#ifdef GOOSE_GPU_SDL
    if (!window) return;
    SDL_Event e;
    while (SDL_PollEvent(&e))
        if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) quitting = 1;
    memcpy(lastkeys, keys, sizeof keys);
    int n = 0;
    const bool *state = SDL_GetKeyboardState(&n);
    for (int i = 0; i < MAX_KEYS; i++) keys[i] = i < n && state[i];
    mousebuttons = SDL_GetMouseState(&mousex, &mousey);
    float scale = SDL_GetWindowPixelDensity((SDL_Window *)window);
    mousex *= scale;
    mousey *= scale;
#endif
}

void goose_gpu_poll(void) { poll_events(); }

/* Ends the frame (§3): shows the screen if it was drawn into, submits, and
   keeps at most two frames in flight; false once asked to close. */
uint8_t goose_gpu_frame(void) {
    if (!ready("gpu::frame") || !outside_pass("gpu::frame")) return 0;
    int present = window && screenacquired;
    if (present && textures[screenslot].access != GOOSE_GPU_ACCESS_PRESENT) {
        backend->barrier(screenslot, textures[screenslot].access, GOOSE_GPU_ACCESS_PRESENT);
        textures[screenslot].access = GOOSE_GPU_ACCESS_PRESENT;
    }
    uint64_t f = backend->submit(present);
    note_submitted();
    note_readbacks(f);
    screenacquired = 0;
    if (f > 2) backend->wait(f - 2);
    write_pngs();
    poll_events();
    framecount++;
    double t = clock_seconds();
    delta = t - lastframe;
    lastframe = t;
    now = t;
    goose_gpu_check();
    return !quitting;
}

double goose_gpu_time(void) { return now; }
double goose_gpu_delta_time(void) { return delta; }
int64_t goose_gpu_frame_count(void) { return (int64_t)framecount; }

static int scancode(goose_gpu_bytes name) {
#ifdef GOOSE_GPU_SDL
    char buf[64];
    int64_t n = name.len < 63 ? name.len : 63;
    memcpy(buf, name.data, (size_t)n);
    buf[n] = 0;
    int sc = (int)SDL_GetScancodeFromName(buf);
    return sc > 0 && sc < MAX_KEYS ? sc : -1;
#else
    (void)name;
    return -1;
#endif
}

uint8_t goose_gpu_key_down(goose_gpu_bytes name) {
    int sc = scancode(name);
    return sc >= 0 && keys[sc];
}

uint8_t goose_gpu_key_pressed(goose_gpu_bytes name) {
    int sc = scancode(name);
    return sc >= 0 && keys[sc] && !lastkeys[sc];
}

void goose_gpu_mouse_position(f32_slice out) {
    if (out.len < 2) return;
    out.data[0] = mousex;
    out.data[1] = mousey;
}

uint8_t goose_gpu_mouse_down(int64_t button) {
    return button >= 1 && button <= 5 && (mousebuttons & (1u << (button - 1)));
}

void goose_gpu_set_title(goose_gpu_bytes title) {
#ifdef GOOSE_GPU_SDL
    if (!window) return;
    char name[256];
    int64_t n = title.len < 255 ? title.len : 255;
    memcpy(name, title.data, (size_t)n);
    name[n] = 0;
    SDL_SetWindowTitle((SDL_Window *)window, name);
#else
    (void)title;
#endif
}

uint8_t goose_gpu_fence_done(uint64_t f) {
    write_pngs();
    return backend && backend->completed() >= f;
}

void goose_gpu_wait(uint64_t f) {
    if (backend) backend->wait(f);
    write_pngs();
    goose_gpu_check();
}

static uint32_t new_readback(void *native, int64_t size) {
    if (!native) return 0;
    for (uint32_t i = 1; i < MAX_READBACKS; i++)
        if (!readbacks[i].used) {
            readbacks[i] = (readback) { 1, native, size, 0, NULL, 0, 0, 0, 0 };
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
    if (!t || !goose_gpu_mip0_bytes(&t->desc) || t->access == GOOSE_GPU_ACCESS_NONE) {
        goose_gpu_misuse("gpu::read_texture of %u: a texture with contents, of a format the "
                         "CPU reads", slot);
        return 0;
    }
    int64_t size = goose_gpu_mip0_bytes(&t->desc);
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

static void drop_readback(readback *r) {
    backend->readback_release(r->native);
    free(r->png);
    memset(r, 0, sizeof *r);
}

void goose_gpu_release_readback(uint32_t id) {
    readback *r = readback_of(id);
    if (r) drop_readback(r);
}

/* save_png and screenshot (§3): a read back of mip 0, written to the file
   once the submission carrying it completes; the screen opaque, as it is
   shown. True when the read back was taken. */
uint8_t goose_gpu_save_png(uint32_t slot, goose_gpu_bytes path) {
    if (!ready("gpu::save_png") || !outside_pass("gpu::save_png")) return 0;
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    int f = t ? t->desc.format : 0;
    if (f != GOOSE_GPU_RGBA8 && f != GOOSE_GPU_RGBA8_SRGB && f != GOOSE_GPU_BGRA8) {
        goose_gpu_misuse("gpu::save_png of %u: only an RGBA8, RGBA8_SRGB or BGRA8 texture", slot);
        return 0;
    }
#ifdef GOOSE_GPU_SDL
    uint32_t id = goose_gpu_read_texture(slot);
    if (!id) return 0;
    readback *r = &readbacks[id];
    size_t n = path.len < 0 ? 0 : (size_t)path.len;
    r->png = malloc(n + 1);
    if (n) memcpy(r->png, path.data, n);
    r->png[n] = 0;
    r->width = t->desc.width;
    r->height = t->desc.height;
    r->bgra = f == GOOSE_GPU_BGRA8;
    r->opaque = slot == screenslot;
    return 1;
#else
    (void)path;
    goose_gpu_fail("this build of the gpu package has no SDL3, so no image files");
    return 0;
#endif
}

/* The PNGs whose pixels have arrived, written. */
static void write_pngs(void) {
    if (!backend) return;
    uint64_t done = backend->completed();
    for (int i = 0; i < MAX_READBACKS; i++) {
        readback *r = &readbacks[i];
        if (!r->used || !r->png || !r->fence || r->fence > done) continue;
#ifdef GOOSE_GPU_SDL
        uint8_t *px = malloc((size_t)r->size);
        backend->readback_read(r->native, px, r->size);
        if (r->opaque)
            for (int64_t a = 3; a < r->size; a += 4) px[a] = 255;
        SDL_Surface *surf = SDL_CreateSurfaceFrom(r->width, r->height,
                                                  r->bgra ? SDL_PIXELFORMAT_BGRA32
                                                          : SDL_PIXELFORMAT_RGBA32,
                                                  px, r->width * 4);
        if (!surf || !SDL_SavePNG(surf, r->png)) {
            /* Nobody is waiting on the result: said now, and kept for
               gpu::error(). */
            goose_gpu_fail("gpu::save_png of %s: %s", r->png, SDL_GetError());
            fprintf(stderr, "gpu: %s\n", lasterror);
        }
        SDL_DestroySurface(surf);
        free(px);
#endif
        drop_readback(r);
    }
}
