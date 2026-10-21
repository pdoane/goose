/* The Metal backend (docs/design/gpu.md §4), on Metal 4. One argument
   table serves every encoder: push constants at buffer index 0, the
   resource heap at 2. The heap is a buffer of GPU addresses, one per slot.
   Metal 4 has no setBytes, so push constants go into a ring of shared
   memory and the table points at them; it snapshots the table at each
   dispatch. Every buffer is in one residency set the queue holds. Fences
   are a shared event's values, signaled after each submission.

   Buffers are shared with the CPU, which Apple silicon gives the GPU at
   full speed, so an upload is a copy at creation and a read back a copy
   once the fence has passed. */

#import <Metal/Metal.h>

#include "gpu_internal.h"

#include <string.h>

#define RING_SIZE (1 << 20)
#define PUSH_ALIGN 256
#define MAX_ALLOCATORS 16

static id<MTLDevice> device;
static id<MTL4CommandQueue> queue;
static id<MTL4Compiler> compiler;
static id<MTL4ArgumentTable> table;
static id<MTLResidencySet> residency;
static int residencydirty;
static id<MTLSharedEvent> event;
static id<MTLBuffer> heap;                       /* uint64_t GPU addresses, by slot */
static id<MTLBuffer> ring;                       /* push constants */
static uint64_t ringhead, ringcmdstart;
static id<MTLBuffer> buffers[GOOSE_GPU_HEAP_SLOTS];
static uint64_t submitted;

/* Command allocators, each free again once its submission's fence has
   passed. */
static struct { id<MTL4CommandAllocator> alloc; uint64_t fence; } allocators[MAX_ALLOCATORS];
static int curalloc = -1;
static id<MTL4CommandBuffer> cmd;
static id<MTL4ComputeCommandEncoder> encoder;

static uint64_t mtl_completed(void) { return event ? event.signaledValue : 0; }

static void mtl_wait(uint64_t fence) {
    if (fence > submitted) return;
    [event waitUntilSignaledValue:fence timeoutMS:UINT64_MAX];
}

static int mtl_open(void) {
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
    if (!table || !heap || !ring) {
        goose_gpu_fail("Metal made no argument table or heap");
        return 0;
    }
    memset(heap.contents, 0, GOOSE_GPU_HEAP_SLOTS * sizeof(uint64_t));
    [table setAddress:heap.gpuAddress atIndex:2];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = 64;
    residency = [device newResidencySetWithDescriptor:rd error:&err];
    if (!residency) {
        goose_gpu_fail("Metal made no residency set: %s", err.localizedDescription.UTF8String);
        return 0;
    }
    [residency addAllocation:heap];
    [residency addAllocation:ring];
    [residency commit];
    [queue addResidencySet:residency];
    event = [device newSharedEvent];
    event.signaledValue = 0;
    submitted = 0;
    ringhead = ringcmdstart = 0;
    return 1;
}

static void mtl_close(void) {
    mtl_wait(submitted);
    encoder = nil;
    cmd = nil;
    for (int i = 0; i < MAX_ALLOCATORS; i++) allocators[i].alloc = nil;
    curalloc = -1;
    if (queue && residency) [queue removeResidencySet:residency];
    residency = nil;
    event = nil;
    ring = nil;
    heap = nil;
    table = nil;
    compiler = nil;
    queue = nil;
    device = nil;
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

static void mtl_buffer_release(uint32_t slot) {
    /* Released once the work submitted so far is done (§2, deferred
       destruction); this first backend waits for it. */
    mtl_wait(submitted);
    [residency removeAllocation:buffers[slot]];
    residencydirty = 1;
    buffers[slot] = nil;
    ((uint64_t *)heap.contents)[slot] = 0;
}

static void mtl_buffer_read(uint32_t slot, int64_t offset, int64_t size, void *out) {
    memcpy(out, (const uint8_t *)buffers[slot].contents + offset, (size_t)size);
}

static int mtl_pipeline_create(goose_gpu_pipeline *p, const uint8_t *lib, int64_t len) {
    const gs_gpu_lib_header *h = &p->header;
    gs_gpu_lib_range code = h->code[GS_GPU_FORMAT_METALLIB];
    if (!code.size || (int64_t)code.offset + code.size > len) {
        goose_gpu_fail("the shader library has no metallib: build the program on a Mac");
        return 0;
    }
    dispatch_data_t dd = dispatch_data_create(lib + code.offset, code.size, NULL,
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *err = nil;
    id<MTLLibrary> library = [device newLibraryWithData:dd error:&err];
    if (!library) {
        goose_gpu_fail("Metal took no library: %s", err.localizedDescription.UTF8String);
        return 0;
    }
    MTL4LibraryFunctionDescriptor *fd = [MTL4LibraryFunctionDescriptor new];
    fd.name = [[NSString alloc] initWithBytes:lib + h->name.offset
                                       length:h->name.size
                                     encoding:NSUTF8StringEncoding];
    fd.library = library;
    MTL4ComputePipelineDescriptor *pd = [MTL4ComputePipelineDescriptor new];
    pd.computeFunctionDescriptor = fd;
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

static void mtl_pipeline_release(goose_gpu_pipeline *p) {
    mtl_wait(submitted);
    if (p->native) CFRelease(p->native);
    p->native = NULL;
}

/* The command buffer being recorded, begun on an allocator whose last
   submission has completed. */
static int begin(void) {
    if (cmd) return 1;
    uint64_t done = mtl_completed();
    curalloc = -1;
    for (int i = 0; i < MAX_ALLOCATORS && curalloc < 0; i++)
        if (allocators[i].alloc && allocators[i].fence <= done) {
            [allocators[i].alloc reset];
            curalloc = i;
        }
    for (int i = 0; i < MAX_ALLOCATORS && curalloc < 0; i++)
        if (!allocators[i].alloc) {
            allocators[i].alloc = [device newCommandAllocator];
            curalloc = i;
        }
    if (curalloc < 0) {
        /* Every allocator is in flight: wait for the oldest. */
        int oldest = 0;
        for (int i = 1; i < MAX_ALLOCATORS; i++)
            if (allocators[i].fence < allocators[oldest].fence) oldest = i;
        mtl_wait(allocators[oldest].fence);
        [allocators[oldest].alloc reset];
        curalloc = oldest;
    }
    cmd = [device newCommandBuffer];
    [cmd beginCommandBufferWithAllocator:allocators[curalloc].alloc];
    ringcmdstart = ringhead;
    return 1;
}

static int mtl_begin_compute(void) {
    if (!begin()) return 0;
    encoder = [cmd computeCommandEncoder];
    [encoder setArgumentTable:table];
    return 1;
}

static void mtl_end_compute(void) {
    [encoder endEncoding];
    encoder = nil;
}

/* Room for `size` bytes of push constants in the ring, which wraps once
   the submissions using its start have completed. */
static uint64_t push_space(uint32_t size) {
    uint64_t at = (ringhead + PUSH_ALIGN - 1) / PUSH_ALIGN * PUSH_ALIGN;
    if (at + size > RING_SIZE) {
        if (ringcmdstart < size) {
            goose_gpu_misuse("more than %d bytes of push constants in one submission",
                             RING_SIZE);
            return UINT64_MAX;
        }
        mtl_wait(submitted);
        at = 0;
    }
    ringhead = at + size;
    return at;
}

static void mtl_dispatch(goose_gpu_pipeline *p, const void *push, uint32_t pushsize,
                         uint32_t x, uint32_t y, uint32_t z) {
    id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)p->native;
    if (pushsize) {
        uint64_t at = push_space(pushsize);
        if (at == UINT64_MAX) return;
        memcpy((uint8_t *)ring.contents + at, push, pushsize);
        [table setAddress:ring.gpuAddress + at atIndex:0];
    }
    [encoder setComputePipelineState:ps];
    const uint32_t *g = p->header.group_size;
    [encoder dispatchThreadgroups:MTLSizeMake(x, y, z)
            threadsPerThreadgroup:MTLSizeMake(g[0], g[1], g[2])];
    /* Each dispatch sees the ones before it, until barriers are the
       program's (§3). */
    [encoder barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
}

static uint64_t mtl_submit(void) {
    if (!begin()) return submitted;
    [cmd endCommandBuffer];
    if (residencydirty) {
        [residency commit];
        residencydirty = 0;
    }
    uint64_t fence = submitted + 1;
    id<MTL4CommandBuffer> list[1] = { cmd };
    [queue commit:list count:1];
    [queue signalEvent:event value:fence];
    allocators[curalloc].fence = fence;
    submitted = fence;
    cmd = nil;
    curalloc = -1;
    return fence;
}

const goose_gpu_backend goose_gpu_metal = {
    "metal",
    mtl_open, mtl_close,
    mtl_buffer_create, mtl_buffer_release, mtl_buffer_read,
    mtl_pipeline_create, mtl_pipeline_release,
    mtl_begin_compute, mtl_end_compute, mtl_dispatch,
    mtl_submit, mtl_completed, mtl_wait,
};
