/* The Metal backend (docs/design/gpu.md §4), on Metal 4. One argument
   table serves every encoder: push constants at buffer index 0, the
   uniform buffer at 1, the resource heap at 2, the sampler heap at 3. The
   resource heap holds one 64-bit word per slot, a buffer's GPU address or a
   texture's resource ID; the sampler heap the samplers' resource IDs.
   Metal 4 has no setBytes, so push constants and uniforms go into a ring of
   shared memory and the table points at them; it snapshots the table at
   each draw and dispatch. Everything is in one residency set the queue
   holds. Fences are a shared event's values, signaled after each
   submission.

   Resources are untracked, so every encoder starts with a barrier on all
   the work queued before it: the program's barriers (§3) are kept by
   construction, and its uploads are complete before what follows. Buffers
   are shared with the CPU, which Apple silicon gives the GPU at full
   speed, so a buffer's contents are a copy at creation. A texture is
   private, and its contents are staged and copied in an upload command
   buffer committed ahead of the next submission. */

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#ifdef GOOSE_GPU_SDL
#include <SDL3/SDL.h>
#endif

#include "gpu_internal.h"

#include <stdlib.h>
#include <string.h>

#define RING_SIZE (4 << 20)
#define RING_ALIGN 256
#define MAX_ALLOCATORS 16
#define MAX_PENDING 1024

static id<MTLDevice> device;
static id<MTL4CommandQueue> queue;
static id<MTL4Compiler> compiler;
static id<MTL4ArgumentTable> table;
static id<MTLResidencySet> residency;
static int residencydirty;
static id<MTLSharedEvent> event;
static id<MTLBuffer> heap;                       /* uint64_t per slot */
static id<MTLBuffer> samplerheap;
static id<MTLSamplerState> samplers[4];
static id<MTLBuffer> ring;                       /* push constants and uniforms */
static uint64_t ringhead, ringcmdstart;
static uint64_t uniformgen = UINT64_MAX;
static id<MTLBuffer> buffers[GOOSE_GPU_HEAP_SLOTS];
static id<MTLTexture> textures[GOOSE_GPU_HEAP_SLOTS];
static uint64_t submitted;

/* The window's layer, and the drawable the screen holds this frame. */
static void *metalview, *sdlwindow;
static CAMetalLayer *layer;
static id<CAMetalDrawable> drawable;
static int drawablewaited;

/* Command allocators, each free again once its submission's fence has
   passed. */
static struct { id<MTL4CommandAllocator> alloc; uint64_t fence; } allocators[MAX_ALLOCATORS];
static int curalloc = -1, uploadalloc = -1;
static id<MTL4CommandBuffer> cmd, uploads;
static id<MTL4ComputeCommandEncoder> compute, uploadenc;
static id<MTL4RenderCommandEncoder> render;
static id<MTLBuffer> pending[MAX_PENDING];       /* upload staging, until the next submission */
static int npending;
/* Staging kept until the fence of the submission that used it. */
#define MAX_RETIRED 2048
static struct { id<MTLBuffer> buffer; uint64_t fence; } retired[MAX_RETIRED];

/* The bound index buffer, and the pass's size for its viewport. */
static uint32_t indexslot;
static int indexsize;

static MTLPixelFormat mtl_format(int f) {
    switch (f) {
        case GOOSE_GPU_RGBA8: return MTLPixelFormatRGBA8Unorm;
        case GOOSE_GPU_BGRA8: return MTLPixelFormatBGRA8Unorm;
        case GOOSE_GPU_RGBA8_SRGB: return MTLPixelFormatRGBA8Unorm_sRGB;
        case GOOSE_GPU_R8: return MTLPixelFormatR8Unorm;
        case GOOSE_GPU_RG8: return MTLPixelFormatRG8Unorm;
        case GOOSE_GPU_RGBA16F: return MTLPixelFormatRGBA16Float;
        case GOOSE_GPU_RGBA32F: return MTLPixelFormatRGBA32Float;
        case GOOSE_GPU_R16F: return MTLPixelFormatR16Float;
        case GOOSE_GPU_RG16F: return MTLPixelFormatRG16Float;
        case GOOSE_GPU_R32F: return MTLPixelFormatR32Float;
        case GOOSE_GPU_RG32F: return MTLPixelFormatRG32Float;
        case GOOSE_GPU_R32UI: return MTLPixelFormatR32Uint;
        case GOOSE_GPU_RGBA8UI: return MTLPixelFormatRGBA8Uint;
        case GOOSE_GPU_RGB10A2: return MTLPixelFormatRGB10A2Unorm;
        case GOOSE_GPU_RG11B10F: return MTLPixelFormatRG11B10Float;
        case GOOSE_GPU_DEPTH16: return MTLPixelFormatDepth16Unorm;
        case GOOSE_GPU_DEPTH24: case GOOSE_GPU_DEPTH32F: case GOOSE_GPU_DEPTH:
            return MTLPixelFormatDepth32Float;
        case GOOSE_GPU_DEPTH24_STENCIL8: return MTLPixelFormatDepth32Float_Stencil8;
        default: return MTLPixelFormatInvalid;
    }
}

static uint64_t mtl_completed(void) { return event ? event.signaledValue : 0; }

static void reclaim(void) {
    uint64_t done = mtl_completed();
    for (int i = 0; i < MAX_RETIRED; i++)
        if (retired[i].buffer && retired[i].fence <= done) {
            [residency removeAllocation:retired[i].buffer];
            residencydirty = 1;
            retired[i].buffer = nil;
        }
}

static void mtl_wait(uint64_t fence) {
    if (fence > submitted) return;
    [event waitUntilSignaledValue:fence timeoutMS:UINT64_MAX];
    reclaim();
}

static void retire(id<MTLBuffer> b, uint64_t fence) {
    for (int tries = 0; tries < 2; tries++) {
        for (int i = 0; i < MAX_RETIRED; i++)
            if (!retired[i].buffer) {
                retired[i].buffer = b;
                retired[i].fence = fence;
                return;
            }
        mtl_wait(mtl_completed() + 1);
    }
}

static int make_samplers(void) {
    samplerheap = [device newBufferWithLength:GOOSE_GPU_SAMPLERS * sizeof(uint64_t)
                                      options:MTLResourceStorageModeShared];
    memset(samplerheap.contents, 0, GOOSE_GPU_SAMPLERS * sizeof(uint64_t));
    for (int i = 0; i < 4; i++) {
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
        MTLSamplerMinMagFilter f = i & 1 ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        MTLSamplerAddressMode m = i & 2 ? MTLSamplerAddressModeRepeat : MTLSamplerAddressModeClampToEdge;
        sd.minFilter = sd.magFilter = f;
        sd.mipFilter = i & 1 ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
        sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = m;
        sd.supportArgumentBuffers = YES;
        samplers[i] = [device newSamplerStateWithDescriptor:sd];
        if (!samplers[i]) {
            goose_gpu_fail("Metal made no sampler");
            return 0;
        }
        ((uint64_t *)samplerheap.contents)[i] = samplers[i].gpuResourceID._impl;
    }
    return 1;
}

static int mtl_open(void *window, int vsync) {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        goose_gpu_fail("no Metal device");
        return 0;
    }
    if (![device supportsFamily:MTLGPUFamilyMetal4]) {
        goose_gpu_fail("the Metal device (%s) does not support Metal 4", device.name.UTF8String);
        return 0;
    }
    NSError *err = nil;
    queue = [device newMTL4CommandQueue];
    compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&err];
    if (!queue || !compiler) {
        goose_gpu_fail("Metal made no queue or compiler: %s",
                       err ? err.localizedDescription.UTF8String : "");
        return 0;
    }
    MTL4ArgumentTableDescriptor *td = [MTL4ArgumentTableDescriptor new];
    td.maxBufferBindCount = 4;
    table = [device newArgumentTableWithDescriptor:td error:&err];
    heap = [device newBufferWithLength:GOOSE_GPU_HEAP_SLOTS * sizeof(uint64_t)
                               options:MTLResourceStorageModeShared];
    ring = [device newBufferWithLength:RING_SIZE options:MTLResourceStorageModeShared];
    if (!table || !heap || !ring || !make_samplers()) {
        goose_gpu_fail("Metal made no argument table or heaps");
        return 0;
    }
    memset(heap.contents, 0, GOOSE_GPU_HEAP_SLOTS * sizeof(uint64_t));
    [table setAddress:heap.gpuAddress atIndex:2];
    [table setAddress:samplerheap.gpuAddress atIndex:3];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = 64;
    residency = [device newResidencySetWithDescriptor:rd error:&err];
    if (!residency) {
        goose_gpu_fail("Metal made no residency set: %s", err.localizedDescription.UTF8String);
        return 0;
    }
    [residency addAllocation:heap];
    [residency addAllocation:samplerheap];
    [residency addAllocation:ring];
    [residency commit];
    [queue addResidencySet:residency];
    event = [device newSharedEvent];
    event.signaledValue = 0;
    submitted = 0;
    ringhead = ringcmdstart = 0;
    uniformgen = UINT64_MAX;
    if (!window) return 1;
#ifdef GOOSE_GPU_SDL
    sdlwindow = window;
    metalview = SDL_Metal_CreateView((SDL_Window *)window);
    layer = (__bridge CAMetalLayer *)SDL_Metal_GetLayer(metalview);
    if (!layer) {
        goose_gpu_fail("SDL made no Metal layer: %s", SDL_GetError());
        return 0;
    }
    layer.device = device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;      /* a screenshot reads it back */
    layer.displaySyncEnabled = vsync ? YES : NO;
    [queue addResidencySet:layer.residencySet];
    return 1;
#else
    (void)vsync;
    goose_gpu_fail("no SDL: no window");
    return 0;
#endif
}

static int mtl_acquire(uint32_t slot, int *width, int *height) {
#ifdef GOOSE_GPU_SDL
    /* The layer follows the window's size in pixels. */
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels((SDL_Window *)sdlwindow, &w, &h);
    if (w <= 0 || h <= 0) return 0;
    if (layer.drawableSize.width != w || layer.drawableSize.height != h)
        layer.drawableSize = CGSizeMake(w, h);
#endif
    drawable = [layer nextDrawable];
    if (!drawable) return 0;
    drawablewaited = 0;
    textures[slot] = drawable.texture;
    ((uint64_t *)heap.contents)[slot] = drawable.texture.gpuResourceID._impl;
    *width = (int)drawable.texture.width;
    *height = (int)drawable.texture.height;
    return 1;
}

static void mtl_close(void) {
    mtl_wait(submitted);
    drawable = nil;
    if (layer && queue) [queue removeResidencySet:layer.residencySet];
    layer = nil;
#ifdef GOOSE_GPU_SDL
    if (metalview) SDL_Metal_DestroyView(metalview);
#endif
    metalview = NULL;
    render = nil;
    compute = nil;
    uploadenc = nil;
    cmd = uploads = nil;
    for (int i = 0; i < MAX_ALLOCATORS; i++) allocators[i].alloc = nil;
    curalloc = uploadalloc = -1;
    for (int i = 0; i < npending; i++) pending[i] = nil;
    npending = 0;
    for (int i = 0; i < MAX_RETIRED; i++) retired[i].buffer = nil;
    for (int i = 0; i < 4; i++) samplers[i] = nil;
    if (queue && residency) [queue removeResidencySet:residency];
    residency = nil;
    event = nil;
    ring = nil;
    samplerheap = nil;
    heap = nil;
    table = nil;
    compiler = nil;
    queue = nil;
    device = nil;
}

/* An allocator whose last submission has completed. */
static int take_allocator(void) {
    uint64_t done = mtl_completed();
    for (int i = 0; i < MAX_ALLOCATORS; i++)
        if (i != curalloc && i != uploadalloc && allocators[i].alloc && allocators[i].fence <= done) {
            [allocators[i].alloc reset];
            return i;
        }
    for (int i = 0; i < MAX_ALLOCATORS; i++)
        if (!allocators[i].alloc) {
            allocators[i].alloc = [device newCommandAllocator];
            return i;
        }
    int oldest = -1;
    for (int i = 0; i < MAX_ALLOCATORS; i++)
        if (i != curalloc && i != uploadalloc &&
            (oldest < 0 || allocators[i].fence < allocators[oldest].fence))
            oldest = i;
    mtl_wait(allocators[oldest].fence);
    [allocators[oldest].alloc reset];
    return oldest;
}

static id<MTL4CommandBuffer> new_cmd(int *alloc) {
    *alloc = take_allocator();
    id<MTL4CommandBuffer> c = [device newCommandBuffer];
    [c beginCommandBufferWithAllocator:allocators[*alloc].alloc];
    return c;
}

static int begin(void) {
    if (cmd) return 1;
    cmd = new_cmd(&curalloc);
    ringcmdstart = ringhead;
    return 1;
}

/* The barrier every encoder starts with: all the work queued before it. */
static const MTLStages allstages = MTLStageVertex | MTLStageFragment | MTLStageDispatch |
                                   MTLStageBlit;

static id<MTL4ComputeCommandEncoder> open_compute(void) {
    id<MTL4ComputeCommandEncoder> e = [cmd computeCommandEncoder];
    [e barrierAfterQueueStages:allstages
                  beforeStages:MTLStageDispatch | MTLStageBlit
             visibilityOptions:MTL4VisibilityOptionDevice];
    [e setArgumentTable:table];
    return e;
}

static int mtl_buffer_create(uint32_t slot, int64_t usage, const void *data, int64_t size) {
    (void)usage;
    id<MTLBuffer> b = [device newBufferWithLength:(NSUInteger)size
                                          options:MTLResourceStorageModeShared];
    if (!b) {
        goose_gpu_fail("cannot make a buffer of %lld bytes", (long long)size);
        return 0;
    }
    if (data) memcpy(b.contents, data, (size_t)size);
    else memset(b.contents, 0, (size_t)size);
    buffers[slot] = b;
    ((uint64_t *)heap.contents)[slot] = b.gpuAddress;
    [residency addAllocation:b];
    residencydirty = 1;
    return 1;
}

static int mtl_update_texture(uint32_t slot, int access, const void *data, int64_t len);

static int mtl_texture_create(uint32_t slot, const goose_gpu_texture_desc *d, const void *data,
                              int64_t len) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = mtl_format(d->format);
    td.width = (NSUInteger)d->width;
    td.height = (NSUInteger)d->height;
    td.mipmapLevelCount = (NSUInteger)d->mips;
    td.storageMode = MTLStorageModePrivate;
    td.usage = 0;
    if (d->usage & GOOSE_GPU_SAMPLED) td.usage |= MTLTextureUsageShaderRead;
    if (d->usage & (GOOSE_GPU_COLOR_TARGET | GOOSE_GPU_DEPTH_TARGET))
        td.usage |= MTLTextureUsageRenderTarget;
    if (d->usage & GOOSE_GPU_STORAGE) td.usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> t = [device newTextureWithDescriptor:td];
    if (!t) {
        goose_gpu_fail("cannot make a %dx%d texture of format %d", d->width, d->height, d->format);
        return 0;
    }
    textures[slot] = t;
    ((uint64_t *)heap.contents)[slot] = t.gpuResourceID._impl;
    [residency addAllocation:t];
    residencydirty = 1;
    if (!data) return 1;
    return mtl_update_texture(slot, 0, data, len);
}

/* Uploads (§3): staged now, and copied in a command buffer committed ahead
   of the next submission, whose encoder waits on all the work queued
   before it and orders its copies one after another; every encoder after
   it waits on it in turn. */
static id<MTLBuffer> stage(const void *data, int64_t len) {
    if (npending == MAX_PENDING) {
        goose_gpu_fail("more than %d uploads before a submission", MAX_PENDING);
        return nil;
    }
    id<MTLBuffer> staging = [device newBufferWithBytes:data length:(NSUInteger)len
                                               options:MTLResourceStorageModeShared];
    if (!staging) {
        goose_gpu_fail("cannot stage %lld bytes", (long long)len);
        return nil;
    }
    pending[npending++] = staging;
    [residency addAllocation:staging];
    residencydirty = 1;
    if (!uploads) {
        uploads = new_cmd(&uploadalloc);
        uploadenc = [uploads computeCommandEncoder];
        [uploadenc barrierAfterQueueStages:allstages
                              beforeStages:MTLStageBlit
                         visibilityOptions:MTL4VisibilityOptionDevice];
    } else {
        [uploadenc barrierAfterEncoderStages:MTLStageBlit
                         beforeEncoderStages:MTLStageBlit
                           visibilityOptions:MTL4VisibilityOptionDevice];
    }
    return staging;
}

static int mtl_update_buffer(uint32_t slot, int64_t offset, const void *data, int64_t size) {
    id<MTLBuffer> staging = stage(data, size);
    if (!staging) return 0;
    [uploadenc copyFromBuffer:staging sourceOffset:0 toBuffer:buffers[slot]
            destinationOffset:(NSUInteger)offset size:(NSUInteger)size];
    return 1;
}

static int mtl_update_texture(uint32_t slot, int access, const void *data, int64_t len) {
    (void)access;
    id<MTLBuffer> staging = stage(data, len);
    if (!staging) return 0;
    id<MTLTexture> t = textures[slot];
    const goose_gpu_tex *gt = goose_gpu_texture_of(slot);
    const goose_gpu_texture_desc *d = &gt->desc;
    NSUInteger row = (NSUInteger)d->width * (NSUInteger)goose_gpu_texel_size(d->format);
    [uploadenc copyFromBuffer:staging
                 sourceOffset:0
            sourceBytesPerRow:row
          sourceBytesPerImage:row * (NSUInteger)d->height
                   sourceSize:MTLSizeMake((NSUInteger)d->width, (NSUInteger)d->height, 1)
                    toTexture:t
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
    return 1;
}

static void mtl_release(uint32_t slot, int texture) {
    /* Released once the work submitted so far is done (§2, deferred
       destruction); this backend waits for it. */
    mtl_wait(submitted);
    if (texture) {
        [residency removeAllocation:textures[slot]];
        textures[slot] = nil;
    } else {
        [residency removeAllocation:buffers[slot]];
        buffers[slot] = nil;
    }
    residencydirty = 1;
    ((uint64_t *)heap.contents)[slot] = 0;
}

static MTL4LibraryFunctionDescriptor *function_of(goose_gpu_pipe *p, int entry) {
    const gs_gpu_lib_entry *e = &p->entries[entry];
    gs_gpu_lib_range code = e->code[GS_GPU_FORMAT_METALLIB];
    if (!code.size || (int64_t)code.offset + code.size > p->librarylen) {
        goose_gpu_fail("the shader library has no metallib: build the program on a Mac");
        return nil;
    }
    dispatch_data_t dd = dispatch_data_create(p->library + code.offset, code.size, NULL,
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *err = nil;
    id<MTLLibrary> library = [device newLibraryWithData:dd error:&err];
    if (!library) {
        goose_gpu_fail("Metal took no library: %s", err.localizedDescription.UTF8String);
        return nil;
    }
    MTL4LibraryFunctionDescriptor *fd = [MTL4LibraryFunctionDescriptor new];
    fd.name = [[NSString alloc] initWithBytes:p->library + e->name.offset
                                       length:e->name.size
                                     encoding:NSUTF8StringEncoding];
    fd.library = library;
    return fd;
}

static int mtl_compute_create(goose_gpu_pipe *p) {
    MTL4LibraryFunctionDescriptor *fd = function_of(p, 0);
    if (!fd) return 0;
    MTL4ComputePipelineDescriptor *pd = [MTL4ComputePipelineDescriptor new];
    pd.computeFunctionDescriptor = fd;
    NSError *err = nil;
    id<MTLComputePipelineState> ps = [compiler newComputePipelineStateWithDescriptor:pd
                                                                 compilerTaskOptions:nil
                                                                               error:&err];
    if (!ps) {
        goose_gpu_fail("Metal made no pipeline for %s: %s", fd.name.UTF8String,
                       err.localizedDescription.UTF8String);
        return 0;
    }
    p->native = (__bridge_retained void *)ps;
    return 1;
}

/* A raster pipeline as Metal holds it: its state, and its depth test,
   which Metal keeps apart. */
@interface GooseRaster : NSObject
@property (strong) id<MTLRenderPipelineState> state;
@property (strong) id<MTLDepthStencilState> depth;
@end
@implementation GooseRaster
@end

static MTLCompareFunction compare_of(int c) {
    static const MTLCompareFunction fns[] = { MTLCompareFunctionLess, MTLCompareFunctionNever,
                                              MTLCompareFunctionLess, MTLCompareFunctionEqual,
                                              MTLCompareFunctionLessEqual, MTLCompareFunctionGreater,
                                              MTLCompareFunctionNotEqual,
                                              MTLCompareFunctionGreaterEqual,
                                              MTLCompareFunctionAlways };
    return c >= 1 && c <= 8 ? fns[c] : MTLCompareFunctionLess;
}

static void *mtl_raster_create(goose_gpu_pipe *p, const goose_gpu_targets *t) {
    MTL4LibraryFunctionDescriptor *vs = function_of(p, 0);
    MTL4LibraryFunctionDescriptor *fs = vs ? function_of(p, 1) : nil;
    if (!fs) return NULL;
    const goose_gpu_pipeline_desc *d = &p->desc;
    MTL4RenderPipelineDescriptor *pd = [MTL4RenderPipelineDescriptor new];
    pd.vertexFunctionDescriptor = vs;
    pd.fragmentFunctionDescriptor = fs;
    pd.rasterSampleCount = 1;
    pd.inputPrimitiveTopology = d->primitive == GOOSE_GPU_POINTS ? MTLPrimitiveTopologyClassPoint
                              : d->primitive == GOOSE_GPU_LINES || d->primitive == GOOSE_GPU_LINE_STRIP
                                    ? MTLPrimitiveTopologyClassLine
                                    : MTLPrimitiveTopologyClassTriangle;
    for (int i = 0; i < GOOSE_GPU_MAX_TARGETS && t->colors[i]; i++) {
        MTL4RenderPipelineColorAttachmentDescriptor *a = pd.colorAttachments[i];
        a.pixelFormat = mtl_format(t->colors[i]);
        a.rgbBlendOperation = a.alphaBlendOperation = MTLBlendOperationAdd;
        switch (d->blend) {
            case GOOSE_GPU_BLEND_ALPHA:
                a.blendingState = MTL4BlendStateEnabled;
                a.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
                a.sourceAlphaBlendFactor = MTLBlendFactorOne;
                a.destinationRGBBlendFactor = a.destinationAlphaBlendFactor =
                    MTLBlendFactorOneMinusSourceAlpha;
                break;
            case GOOSE_GPU_BLEND_ADD:
                a.blendingState = MTL4BlendStateEnabled;
                a.sourceRGBBlendFactor = a.sourceAlphaBlendFactor = MTLBlendFactorOne;
                a.destinationRGBBlendFactor = a.destinationAlphaBlendFactor = MTLBlendFactorOne;
                break;
            case GOOSE_GPU_BLEND_PREMULTIPLIED:
                a.blendingState = MTL4BlendStateEnabled;
                a.sourceRGBBlendFactor = a.sourceAlphaBlendFactor = MTLBlendFactorOne;
                a.destinationRGBBlendFactor = a.destinationAlphaBlendFactor =
                    MTLBlendFactorOneMinusSourceAlpha;
                break;
            case GOOSE_GPU_BLEND_MULTIPLY:
                a.blendingState = MTL4BlendStateEnabled;
                a.sourceRGBBlendFactor = a.sourceAlphaBlendFactor = MTLBlendFactorDestinationColor;
                a.destinationRGBBlendFactor = a.destinationAlphaBlendFactor = MTLBlendFactorZero;
                break;
            default:
                a.blendingState = MTL4BlendStateDisabled;
                break;
        }
    }
    NSError *err = nil;
    id<MTLRenderPipelineState> ps = [compiler newRenderPipelineStateWithDescriptor:pd
                                                               compilerTaskOptions:nil
                                                                             error:&err];
    if (!ps) {
        goose_gpu_fail("Metal made no render pipeline: %s", err.localizedDescription.UTF8String);
        return NULL;
    }
    MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = d->depth_test ? compare_of(d->depth_compare) : MTLCompareFunctionAlways;
    dd.depthWriteEnabled = d->depth_write && t->depth;
    GooseRaster *r = [GooseRaster new];
    r.state = ps;
    r.depth = [device newDepthStencilStateWithDescriptor:dd];
    return (__bridge_retained void *)r;
}

static void mtl_pipeline_release(goose_gpu_pipe *p) {
    mtl_wait(submitted);
    if (p->native) CFRelease(p->native);
    for (int i = 0; i < p->nvariants; i++) CFRelease(p->variants[i].native);
    p->native = NULL;
    p->nvariants = 0;
}

static int mtl_begin_compute(void) {
    begin();
    compute = open_compute();
    uniformgen = UINT64_MAX;
    return 1;
}

static void mtl_end_compute(void) {
    [compute endEncoding];
    compute = nil;
}

static int mtl_begin_pass(const goose_gpu_pass *ps) {
    begin();
    MTL4RenderPassDescriptor *rp = [MTL4RenderPassDescriptor new];
    for (int i = 0; i < ps->ncolor; i++) {
        MTLRenderPassColorAttachmentDescriptor *a = rp.colorAttachments[i];
        a.texture = textures[ps->desc.color[i]];
        a.loadAction = ps->desc.clear_color ? MTLLoadActionClear : MTLLoadActionLoad;
        a.storeAction = MTLStoreActionStore;
        const float *c = ps->desc.color_value;
        a.clearColor = MTLClearColorMake(c[0], c[1], c[2], c[3]);
    }
    if (ps->desc.depth) {
        rp.depthAttachment.texture = textures[ps->desc.depth];
        rp.depthAttachment.loadAction = ps->desc.clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.depthAttachment.clearDepth = ps->desc.depth_value;
    }
    render = [cmd renderCommandEncoderWithDescriptor:rp];
    if (!render) {
        goose_gpu_fail("Metal began no render pass");
        return 0;
    }
    [render barrierAfterQueueStages:allstages
                       beforeStages:MTLStageVertex | MTLStageFragment
                  visibilityOptions:MTL4VisibilityOptionDevice];
    [render setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
    [render setViewport:(MTLViewport) { 0, 0, (double)ps->width, (double)ps->height, 0, 1 }];
    uniformgen = UINT64_MAX;
    indexsize = 0;
    return 1;
}

static void mtl_end_pass(void) {
    [render endEncoding];
    render = nil;
}

static void mtl_viewport(float x, float y, float w, float h) {
    [render setViewport:(MTLViewport) { x, y, w, h, 0, 1 }];
}

static void mtl_scissor(int32_t x, int32_t y, int32_t w, int32_t h) {
    [render setScissorRect:(MTLScissorRect) { (NSUInteger)x, (NSUInteger)y, (NSUInteger)w,
                                              (NSUInteger)h }];
}

static void mtl_index_buffer(uint32_t slot, int size) {
    indexslot = slot;
    indexsize = size;
}

/* Room in the ring, which wraps once the submissions using its start have
   completed. */
static uint64_t ring_space(uint32_t size) {
    uint64_t at = (ringhead + RING_ALIGN - 1) / RING_ALIGN * RING_ALIGN;
    if (at + size > RING_SIZE) {
        if (ringcmdstart < size) {
            goose_gpu_misuse("more than %d bytes of push constants and uniforms in one "
                             "submission", RING_SIZE);
            return UINT64_MAX;
        }
        mtl_wait(submitted);
        at = 0;
    }
    ringhead = at + size;
    return at;
}

static void set_blocks(const void *push, uint32_t pushsize, const void *uniforms,
                       uint32_t uniformsize, uint64_t gen) {
    if (pushsize) {
        uint64_t at = ring_space(pushsize);
        if (at == UINT64_MAX) return;
        memcpy((uint8_t *)ring.contents + at, push, pushsize);
        [table setAddress:ring.gpuAddress + at atIndex:0];
    }
    if (uniformsize && gen != uniformgen) {
        uint64_t at = ring_space(uniformsize);
        if (at == UINT64_MAX) return;
        memcpy((uint8_t *)ring.contents + at, uniforms, uniformsize);
        [table setAddress:ring.gpuAddress + at atIndex:1];
        uniformgen = gen;
    }
}

static void mtl_dispatch(goose_gpu_pipe *p, const void *push, uint32_t pushsize,
                         const void *uniforms, uint32_t uniformsize, uint64_t gen,
                         uint32_t x, uint32_t y, uint32_t z) {
    set_blocks(push, pushsize, uniforms, uniformsize, gen);
    [compute setComputePipelineState:(__bridge id<MTLComputePipelineState>)p->native];
    const uint32_t *g = p->entries[0].group_size;
    [compute dispatchThreadgroups:MTLSizeMake(x, y, z)
            threadsPerThreadgroup:MTLSizeMake(g[0], g[1], g[2])];
    /* Each dispatch sees the ones before it in its pass. */
    [compute barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
}

static MTLPrimitiveType primitive_of(int p) {
    switch (p) {
        case GOOSE_GPU_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
        case GOOSE_GPU_LINES: return MTLPrimitiveTypeLine;
        case GOOSE_GPU_LINE_STRIP: return MTLPrimitiveTypeLineStrip;
        case GOOSE_GPU_POINTS: return MTLPrimitiveTypePoint;
        default: return MTLPrimitiveTypeTriangle;
    }
}

static void mtl_draw(goose_gpu_pipe *p, void *native, const void *push, uint32_t pushsize,
                     const void *uniforms, uint32_t uniformsize, uint64_t gen, int indexed,
                     uint32_t count, uint32_t instances, uint32_t first, int32_t offset,
                     uint32_t first_instance) {
    GooseRaster *r = (__bridge GooseRaster *)native;
    const goose_gpu_pipeline_desc *d = &p->desc;
    set_blocks(push, pushsize, uniforms, uniformsize, gen);
    [render setRenderPipelineState:r.state];
    [render setDepthStencilState:r.depth];
    [render setCullMode:d->cull == GOOSE_GPU_CULL_FRONT ? MTLCullModeFront
                      : d->cull == GOOSE_GPU_CULL_BACK ? MTLCullModeBack : MTLCullModeNone];
    [render setFrontFacingWinding:d->clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise];
    [render setTriangleFillMode:d->wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
    MTLPrimitiveType prim = primitive_of(d->primitive);
    if (!indexed) {
        [render drawPrimitives:prim vertexStart:first vertexCount:count instanceCount:instances
                  baseInstance:first_instance];
        return;
    }
    id<MTLBuffer> ib = buffers[indexslot];
    NSUInteger at = (NSUInteger)first * (NSUInteger)indexsize;
    [render drawIndexedPrimitives:prim
                       indexCount:count
                        indexType:indexsize == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                      indexBuffer:ib.gpuAddress + at
                indexBufferLength:ib.length - at
                    instanceCount:instances
                       baseVertex:offset
                     baseInstance:first_instance];
}

static void mtl_barrier(uint32_t slot, int before, int after) {
    /* Kept by construction: every encoder waits on what was queued before
       it (the file comment). */
    (void)slot; (void)before; (void)after;
}

typedef struct { void *buffer; } mtlreadback;

static void *readback_into(void (^copy)(id<MTL4ComputeCommandEncoder>, id<MTLBuffer>),
                           int64_t size) {
    begin();
    id<MTLBuffer> staging = [device newBufferWithLength:(NSUInteger)size
                                                options:MTLResourceStorageModeShared];
    if (!staging) {
        goose_gpu_fail("cannot make a read back of %lld bytes", (long long)size);
        return NULL;
    }
    [residency addAllocation:staging];
    residencydirty = 1;
    id<MTL4ComputeCommandEncoder> e = open_compute();
    copy(e, staging);
    [e endEncoding];
    mtlreadback *rb = calloc(1, sizeof *rb);
    rb->buffer = (__bridge_retained void *)staging;
    return rb;
}

static void *mtl_readback_buffer(uint32_t slot, int64_t offset, int64_t size) {
    id<MTLBuffer> src = buffers[slot];
    return readback_into(^(id<MTL4ComputeCommandEncoder> e, id<MTLBuffer> dst) {
        [e copyFromBuffer:src sourceOffset:(NSUInteger)offset toBuffer:dst destinationOffset:0
                     size:(NSUInteger)size];
    }, size);
}

static void *mtl_readback_texture(uint32_t slot, int access, int64_t size) {
    (void)access;
    id<MTLTexture> src = textures[slot];
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    NSUInteger row = (NSUInteger)t->desc.width * (NSUInteger)goose_gpu_texel_size(t->desc.format);
    return readback_into(^(id<MTL4ComputeCommandEncoder> e, id<MTLBuffer> dst) {
        [e copyFromTexture:src
               sourceSlice:0
               sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(src.width, src.height, 1)
                  toBuffer:dst
         destinationOffset:0
    destinationBytesPerRow:row
  destinationBytesPerImage:row * src.height];
    }, size);
}

static void mtl_readback_read(void *rb, void *out, int64_t size) {
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)((mtlreadback *)rb)->buffer;
    memcpy(out, b.contents, (size_t)size);
}

static void mtl_readback_release(void *rb) {
    if (!rb) return;
    id<MTLBuffer> b = (__bridge_transfer id<MTLBuffer>)((mtlreadback *)rb)->buffer;
    retire(b, submitted + 1);
    free(rb);
}

static uint64_t mtl_submit(int present) {
    begin();
    [cmd endCommandBuffer];
    id<MTL4CommandBuffer> list[2];
    NSUInteger n = 0;
    if (uploads) {
        [uploadenc endEncoding];
        uploadenc = nil;
        [uploads endCommandBuffer];
        list[n++] = uploads;
    }
    list[n++] = cmd;
    if (residencydirty) {
        [residency commit];
        residencydirty = 0;
    }
    uint64_t fence = submitted + 1;
    /* The display done with the drawable before anything drawing into it. */
    if (drawable && !drawablewaited) {
        [queue waitForDrawable:drawable];
        drawablewaited = 1;
    }
    [queue commit:list count:n];
    [queue signalEvent:event value:fence];
    if (present && drawable) {
        [queue signalDrawable:drawable];
        [drawable present];
        drawable = nil;
    }
    allocators[curalloc].fence = fence;
    if (uploadalloc >= 0) allocators[uploadalloc].fence = fence;
    for (int i = 0; i < npending; i++) {
        retire(pending[i], fence);
        pending[i] = nil;
    }
    npending = 0;
    submitted = fence;
    cmd = uploads = nil;
    curalloc = uploadalloc = -1;
    reclaim();
    return fence;
}

const goose_gpu_backend goose_gpu_metal = {
    "metal",
    mtl_open, mtl_close,
    mtl_buffer_create, mtl_texture_create, mtl_release, mtl_update_buffer, mtl_update_texture,
    mtl_compute_create, mtl_raster_create, mtl_pipeline_release,
    mtl_begin_compute, mtl_end_compute, mtl_begin_pass, mtl_end_pass,
    mtl_dispatch, mtl_draw, mtl_index_buffer, mtl_viewport, mtl_scissor,
    mtl_barrier,
    mtl_readback_buffer, mtl_readback_texture, mtl_readback_read, mtl_readback_release,
    mtl_acquire, mtl_submit, mtl_completed, mtl_wait,
};
