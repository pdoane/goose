/* The Vulkan backend (docs/design/gpu.md §4): Vulkan 1.2 with descriptor
   indexing. The resource heap is one descriptor set: binding 0 an array of
   storage buffers, updated after bind, indexed by slot. Push constants are
   one 128-byte range. Buffers are host-visible and mapped for good, so an
   upload is a copy at creation and a read back a copy once the fence has
   passed. Fences are one timeline semaphore's values.

   The loader is opened at run time, so a program that never asks for
   Vulkan needs none installed. Portability drivers (MoltenVK) are not
   enumerated: on a Mac this is KosmicKrisp. GOOSE_GPU_VALIDATION=1 turns on
   the Khronos validation layer where it is installed. */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "gpu_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define LOADER "vulkan-1.dll"
static void *open_loader(void) { return (void *)LoadLibraryA(LOADER); }
static void *loader_sym(void *l, const char *n) { return (void *)GetProcAddress((HMODULE)l, n); }
#else
#include <dlfcn.h>
static void *open_loader(void) {
#ifdef __APPLE__
    static const char *names[] = { "libvulkan.1.dylib", "libvulkan.dylib",
                                   "/usr/local/lib/libvulkan.1.dylib", NULL };
#else
    static const char *names[] = { "libvulkan.so.1", "libvulkan.so", NULL };
#endif
    for (int i = 0; names[i]; i++) {
        void *l = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (l) return l;
    }
    return NULL;
}
static void *loader_sym(void *l, const char *n) { return dlsym(l, n); }
#endif

#define VK_GLOBAL(F) F(vkCreateInstance) F(vkEnumerateInstanceLayerProperties)
#define VK_INSTANCE(F) \
    F(vkDestroyInstance) F(vkEnumeratePhysicalDevices) F(vkGetPhysicalDeviceProperties2) \
    F(vkGetPhysicalDeviceQueueFamilyProperties) F(vkGetPhysicalDeviceMemoryProperties) \
    F(vkGetPhysicalDeviceFeatures2) F(vkCreateDevice) F(vkGetDeviceProcAddr)
#define VK_DEVICE(F) \
    F(vkDestroyDevice) F(vkGetDeviceQueue) F(vkCreateBuffer) F(vkDestroyBuffer) \
    F(vkGetBufferMemoryRequirements) F(vkAllocateMemory) F(vkFreeMemory) F(vkBindBufferMemory) \
    F(vkMapMemory) F(vkCreateDescriptorSetLayout) F(vkDestroyDescriptorSetLayout) \
    F(vkCreateDescriptorPool) F(vkDestroyDescriptorPool) F(vkAllocateDescriptorSets) \
    F(vkUpdateDescriptorSets) F(vkCreatePipelineLayout) F(vkDestroyPipelineLayout) \
    F(vkCreateShaderModule) F(vkDestroyShaderModule) F(vkCreateComputePipelines) \
    F(vkDestroyPipeline) F(vkCreateCommandPool) F(vkDestroyCommandPool) \
    F(vkAllocateCommandBuffers) F(vkFreeCommandBuffers) F(vkBeginCommandBuffer) \
    F(vkEndCommandBuffer) F(vkCmdBindPipeline) F(vkCmdBindDescriptorSets) \
    F(vkCmdPushConstants) F(vkCmdDispatch) F(vkCmdPipelineBarrier) F(vkQueueSubmit) \
    F(vkCreateSemaphore) F(vkDestroySemaphore) F(vkGetSemaphoreCounterValue) \
    F(vkWaitSemaphores) F(vkDeviceWaitIdle)

#define DECLARE(f) static PFN_##f f;
VK_GLOBAL(DECLARE)
VK_INSTANCE(DECLARE)
VK_DEVICE(DECLARE)
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

static void *loader;
static VkInstance instance;
static VkPhysicalDevice physical;
static VkDevice device;
static VkQueue queue;
static uint32_t queuefamily;
static VkPhysicalDeviceMemoryProperties memprops;
static char drivername[VK_MAX_DRIVER_NAME_SIZE];

static VkDescriptorSetLayout setlayout;
static VkDescriptorPool pool;
static VkDescriptorSet heapset;
static VkPipelineLayout pipelinelayout;
static VkCommandPool cmdpool;
static VkSemaphore timeline;
static uint64_t submitted;

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
} vkbuffer;
static vkbuffer buffers[GOOSE_GPU_HEAP_SLOTS];

/* The command buffer being recorded, and the ones in flight with the fence
   that frees each. */
static VkCommandBuffer cmd;
#define MAX_INFLIGHT 64
static struct { VkCommandBuffer cmd; uint64_t fence; } inflight[MAX_INFLIGHT];

static int check(VkResult r, const char *what) {
    if (r == VK_SUCCESS) return 1;
    goose_gpu_fail("Vulkan: %s failed (VkResult %d)", what, (int)r);
    return 0;
}

static int load_global(void) {
    if (!loader) loader = open_loader();
    if (!loader) {
        goose_gpu_fail("no Vulkan loader: install the Vulkan SDK (vulkan.lunarg.com)");
        return 0;
    }
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)loader_sym(loader, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) {
        goose_gpu_fail("the Vulkan loader has no vkGetInstanceProcAddr");
        return 0;
    }
#define LOAD(f) f = (PFN_##f)vkGetInstanceProcAddr(NULL, #f);
    VK_GLOBAL(LOAD)
#undef LOAD
    return 1;
}

static int has_layer(const char *name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, NULL);
    VkLayerProperties *props = calloc(n ? n : 1, sizeof *props);
    vkEnumerateInstanceLayerProperties(&n, props);
    int found = 0;
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(props[i].layerName, name)) found = 1;
    free(props);
    return found;
}

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags also) {
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < memprops.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = memprops.memoryTypes[i].propertyFlags;
            if (!(bits & (1u << i)) || (f & want) != want) continue;
            if (pass == 0 && (f & also) != also) continue;
            return i;
        }
    return UINT32_MAX;
}

static int vk_open(void) {
    if (!load_global()) return 0;
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "goose";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    const char *validation = "VK_LAYER_KHRONOS_validation";
    const char *v = getenv("GOOSE_GPU_VALIDATION");
    if (v && *v && strcmp(v, "0") && has_layer(validation)) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &validation;
    }
    if (!check(vkCreateInstance(&ici, NULL, &instance), "vkCreateInstance")) return 0;
#define LOAD(f) f = (PFN_##f)vkGetInstanceProcAddr(instance, #f);
    VK_INSTANCE(LOAD)
#undef LOAD

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, NULL);
    if (!n) {
        goose_gpu_fail("no Vulkan device: on a Mac, install KosmicKrisp (the Vulkan SDK's "
                       "conformant driver); MoltenVK is not one");
        return 0;
    }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    vkEnumeratePhysicalDevices(instance, &n, devs);
    physical = devs[0];
    VkPhysicalDeviceDriverProperties driver = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
    VkPhysicalDeviceVulkan12Properties p12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
    driver.pNext = &p12;
    VkPhysicalDeviceProperties2 props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    props.pNext = &driver;
    vkGetPhysicalDeviceProperties2(physical, &props);
    snprintf(drivername, sizeof drivername, "%s", driver.driverName);
    if (props.properties.apiVersion < VK_API_VERSION_1_2) {
        goose_gpu_fail("the Vulkan device (%s) is not Vulkan 1.2", props.properties.deviceName);
        return 0;
    }
    if (p12.maxPerStageDescriptorUpdateAfterBindStorageBuffers < GOOSE_GPU_HEAP_SLOTS) {
        goose_gpu_fail("the Vulkan device (%s) binds %u storage buffers per stage, and the heap "
                       "needs %d", props.properties.deviceName,
                       p12.maxPerStageDescriptorUpdateAfterBindStorageBuffers,
                       GOOSE_GPU_HEAP_SLOTS);
        return 0;
    }
    vkGetPhysicalDeviceMemoryProperties(physical, &memprops);

    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(physical, &f2);
    if (!f12.descriptorIndexing || !f12.runtimeDescriptorArray ||
        !f12.descriptorBindingPartiallyBound ||
        !f12.descriptorBindingStorageBufferUpdateAfterBind || !f12.scalarBlockLayout ||
        !f12.timelineSemaphore) {
        goose_gpu_fail("the Vulkan device (%s) lacks descriptor indexing, scalar block layout "
                       "or timeline semaphores", props.properties.deviceName);
        return 0;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, NULL);
    VkQueueFamilyProperties qprops[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, qprops);
    queuefamily = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queuefamily = i;
            break;
        }
    if (queuefamily == UINT32_MAX) {
        goose_gpu_fail("the Vulkan device has no compute queue");
        return 0;
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = queuefamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features want = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    want.descriptorIndexing = VK_TRUE;
    want.runtimeDescriptorArray = VK_TRUE;
    want.descriptorBindingPartiallyBound = VK_TRUE;
    want.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    want.shaderStorageBufferArrayNonUniformIndexing = f12.shaderStorageBufferArrayNonUniformIndexing;
    want.scalarBlockLayout = VK_TRUE;
    want.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &want;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (!check(vkCreateDevice(physical, &dci, NULL, &device), "vkCreateDevice")) return 0;
#define LOAD(f) f = (PFN_##f)vkGetDeviceProcAddr(device, #f);
    VK_DEVICE(LOAD)
#undef LOAD
    vkGetDeviceQueue(device, queuefamily, 0, &queue);

    /* The heap: set 0, binding 0, every slot a storage buffer. */
    VkDescriptorSetLayoutBinding b = { 0 };
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = GOOSE_GPU_HEAP_SLOTS;
    b.stageFlags = VK_SHADER_STAGE_ALL;
    VkDescriptorBindingFlags bflags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi =
        { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bfi.bindingCount = 1;
    bfi.pBindingFlags = &bflags;
    VkDescriptorSetLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.pNext = &bfi;
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    lci.bindingCount = 1;
    lci.pBindings = &b;
    if (!check(vkCreateDescriptorSetLayout(device, &lci, NULL, &setlayout), "descriptor layout"))
        return 0;
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, GOOSE_GPU_HEAP_SLOTS };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (!check(vkCreateDescriptorPool(device, &pci, NULL, &pool), "descriptor pool")) return 0;
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &setlayout;
    if (!check(vkAllocateDescriptorSets(device, &dai, &heapset), "descriptor set")) return 0;

    VkPushConstantRange range = { VK_SHADER_STAGE_ALL, 0, 128 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &setlayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    if (!check(vkCreatePipelineLayout(device, &plci, NULL, &pipelinelayout), "pipeline layout"))
        return 0;

    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.queueFamilyIndex = queuefamily;
    if (!check(vkCreateCommandPool(device, &cpi, NULL, &cmdpool), "command pool")) return 0;

    VkSemaphoreTypeCreateInfo sti = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &sti;
    if (!check(vkCreateSemaphore(device, &sci, NULL, &timeline), "timeline semaphore")) return 0;
    submitted = 0;
    return 1;
}

static uint64_t vk_completed(void) {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(device, timeline, &v);
    return v;
}

static void reclaim(void) {
    uint64_t done = vk_completed();
    for (int i = 0; i < MAX_INFLIGHT; i++)
        if (inflight[i].cmd && inflight[i].fence <= done) {
            vkFreeCommandBuffers(device, cmdpool, 1, &inflight[i].cmd);
            inflight[i].cmd = VK_NULL_HANDLE;
        }
}

static void vk_buffer_release(uint32_t slot);

static void vk_close(void) {
    if (!device) return;
    vkDeviceWaitIdle(device);
    reclaim();
    if (cmd) vkFreeCommandBuffers(device, cmdpool, 1, &cmd);
    cmd = VK_NULL_HANDLE;
    vkDestroySemaphore(device, timeline, NULL);
    vkDestroyCommandPool(device, cmdpool, NULL);
    vkDestroyPipelineLayout(device, pipelinelayout, NULL);
    vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyDescriptorSetLayout(device, setlayout, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}

static int vk_buffer_create(uint32_t slot, int64_t usage, const void *data, int64_t size) {
    (void)usage;
    vkbuffer *vb = &buffers[slot];
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = (VkDeviceSize)size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!check(vkCreateBuffer(device, &bci, NULL, &vb->buffer), "vkCreateBuffer")) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, vb->buffer, &req);
    uint32_t type = memory_type(req.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) {
        goose_gpu_fail("Vulkan: no host-visible memory for a buffer");
        return 0;
    }
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (!check(vkAllocateMemory(device, &mai, NULL, &vb->memory), "vkAllocateMemory")) return 0;
    vkBindBufferMemory(device, vb->buffer, vb->memory, 0);
    if (!check(vkMapMemory(device, vb->memory, 0, VK_WHOLE_SIZE, 0, &vb->mapped), "vkMapMemory"))
        return 0;
    if (data) memcpy(vb->mapped, data, (size_t)size);
    else memset(vb->mapped, 0, (size_t)size);
    VkDescriptorBufferInfo info = { vb->buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = heapset;
    w.dstBinding = 0;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
    return 1;
}

static void vk_buffer_release(uint32_t slot) {
    /* Released once the work submitted so far is done (§2, deferred
       destruction); this first backend waits for it. */
    vkDeviceWaitIdle(device);
    vkbuffer *vb = &buffers[slot];
    vkDestroyBuffer(device, vb->buffer, NULL);
    vkFreeMemory(device, vb->memory, NULL);
    memset(vb, 0, sizeof *vb);
}

static void vk_buffer_read(uint32_t slot, int64_t offset, int64_t size, void *out) {
    memcpy(out, (const uint8_t *)buffers[slot].mapped + offset, (size_t)size);
}

static int vk_pipeline_create(goose_gpu_pipeline *p, const uint8_t *lib, int64_t len) {
    const gs_gpu_lib_header *h = &p->header;
    gs_gpu_lib_range code = h->code[GS_GPU_FORMAT_SPIRV];
    if (!code.size || (int64_t)code.offset + code.size > len || code.size % 4) {
        goose_gpu_fail("the shader library has no SPIR-V");
        return 0;
    }
    /* The words, aligned, which the library's place in static data is not. */
    uint32_t *words = malloc(code.size);
    memcpy(words, lib + code.offset, code.size);
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = code.size;
    smi.pCode = words;
    VkShaderModule module;
    VkResult r = vkCreateShaderModule(device, &smi, NULL, &module);
    free(words);
    if (!check(r, "vkCreateShaderModule")) return 0;
    char name[256];
    uint32_t n = h->name.size < sizeof name - 1 ? h->name.size : (uint32_t)sizeof name - 1;
    memcpy(name, lib + h->name.offset, n);
    name[n] = 0;
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = name;
    cpi.layout = pipelinelayout;
    VkPipeline pipe;
    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
    vkDestroyShaderModule(device, module, NULL);
    if (!check(r, "vkCreateComputePipelines")) return 0;
    p->native = (void *)pipe;
    return 1;
}

static void vk_pipeline_release(goose_gpu_pipeline *p) {
    vkDeviceWaitIdle(device);
    vkDestroyPipeline(device, (VkPipeline)p->native, NULL);
    p->native = NULL;
}

static int begin(void) {
    if (cmd) return 1;
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = cmdpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (!check(vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers"))
        return 0;
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return 1;
}

static void barrier(VkPipelineStageFlags src, VkAccessFlags srca, VkPipelineStageFlags dst,
                    VkAccessFlags dsta) {
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = srca;
    mb.dstAccessMask = dsta;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, NULL, 0, NULL);
}

static int vk_begin_compute(void) {
    if (!begin()) return 0;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelinelayout, 0, 1, &heapset,
                            0, NULL);
    return 1;
}

static void vk_end_compute(void) {}

static void vk_dispatch(goose_gpu_pipeline *p, const void *push, uint32_t pushsize,
                        uint32_t x, uint32_t y, uint32_t z) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)p->native);
    if (pushsize) vkCmdPushConstants(cmd, pipelinelayout, VK_SHADER_STAGE_ALL, 0, pushsize, push);
    vkCmdDispatch(cmd, x, y, z);
    /* Each dispatch sees the ones before it, until barriers are the
       program's (§3). */
    barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

static uint64_t vk_submit(void) {
    reclaim();
    if (!begin()) return submitted;
    /* What the GPU wrote is the host's to read once the fence passes. */
    barrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    vkEndCommandBuffer(cmd);
    uint64_t fence = submitted + 1;
    VkTimelineSemaphoreSubmitInfo tsi = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    tsi.signalSemaphoreValueCount = 1;
    tsi.pSignalSemaphoreValues = &fence;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.pNext = &tsi;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &timeline;
    if (!check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit")) {
        goose_gpu_misuse("%s", "the submission failed");
        return submitted;
    }
    submitted = fence;
    int placed = 0;
    for (int i = 0; i < MAX_INFLIGHT && !placed; i++)
        if (!inflight[i].cmd) {
            inflight[i].cmd = cmd;
            inflight[i].fence = fence;
            placed = 1;
        }
    if (!placed) {
        uint64_t oldest = vk_completed() + 1;
        VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timeline;
        wi.pValues = &oldest;
        vkWaitSemaphores(device, &wi, UINT64_MAX);
        reclaim();
        for (int i = 0; i < MAX_INFLIGHT && !placed; i++)
            if (!inflight[i].cmd) {
                inflight[i].cmd = cmd;
                inflight[i].fence = fence;
                placed = 1;
            }
    }
    cmd = VK_NULL_HANDLE;
    return fence;
}

static void vk_wait(uint64_t fence) {
    if (fence > submitted) return;
    VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &fence;
    vkWaitSemaphores(device, &wi, UINT64_MAX);
    reclaim();
}

const goose_gpu_backend goose_gpu_vulkan = {
    "vulkan",
    vk_open, vk_close,
    vk_buffer_create, vk_buffer_release, vk_buffer_read,
    vk_pipeline_create, vk_pipeline_release,
    vk_begin_compute, vk_end_compute, vk_dispatch,
    vk_submit, vk_completed, vk_wait,
};
