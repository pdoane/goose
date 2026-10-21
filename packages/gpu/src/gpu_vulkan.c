/* The Vulkan backend (docs/design/gpu.md §4): Vulkan 1.3 with descriptor
   indexing, mutable descriptors and push descriptors. Set 0 is the heaps:
   binding 0 every resource slot, a mutable descriptor that is a storage
   buffer, a sampled image or a storage image; binding 1 the samplers. Set
   1 is the uniform buffer, pushed with each draw from a ring. Push
   constants are one 128-byte range. Fences are one timeline semaphore's
   values.

   Buffers are host-visible and mapped for good, so a buffer's contents are
   a copy at creation. A texture's are staged and copied in an upload
   command buffer submitted ahead of the next submission (§3), which leaves
   it in shader reads. A read back is a copy into host-visible memory,
   recorded where it was asked for.

   The loader is opened at run time, so a program that never asks for
   Vulkan needs none installed. Portability drivers (MoltenVK) are not
   enumerated: on a Mac this is KosmicKrisp. GOOSE_GPU_VALIDATION=1 turns on
   the Khronos validation layer where it is installed. */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "gpu_internal.h"

#ifdef GOOSE_GPU_SDL
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static void *open_loader(void) { return (void *)LoadLibraryA("vulkan-1.dll"); }
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
    F(vkGetPhysicalDeviceFeatures2) F(vkCreateDevice) F(vkGetDeviceProcAddr) \
    F(vkEnumerateDeviceExtensionProperties) F(vkGetPhysicalDeviceFormatProperties) \
    F(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) F(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    F(vkGetPhysicalDeviceSurfacePresentModesKHR) F(vkGetPhysicalDeviceSurfaceSupportKHR) \
    F(vkDestroySurfaceKHR)
#define VK_DEVICE(F) \
    F(vkDestroyDevice) F(vkGetDeviceQueue) F(vkCreateBuffer) F(vkDestroyBuffer) \
    F(vkGetBufferMemoryRequirements) F(vkAllocateMemory) F(vkFreeMemory) F(vkBindBufferMemory) \
    F(vkMapMemory) F(vkCreateDescriptorSetLayout) F(vkDestroyDescriptorSetLayout) \
    F(vkCreateDescriptorPool) F(vkDestroyDescriptorPool) F(vkAllocateDescriptorSets) \
    F(vkUpdateDescriptorSets) F(vkCreatePipelineLayout) F(vkDestroyPipelineLayout) \
    F(vkCreateShaderModule) F(vkDestroyShaderModule) F(vkCreateComputePipelines) \
    F(vkCreateGraphicsPipelines) F(vkDestroyPipeline) F(vkCreateCommandPool) \
    F(vkDestroyCommandPool) F(vkAllocateCommandBuffers) F(vkFreeCommandBuffers) \
    F(vkBeginCommandBuffer) F(vkEndCommandBuffer) F(vkCmdBindPipeline) \
    F(vkCmdBindDescriptorSets) F(vkCmdPushConstants) F(vkCmdDispatch) F(vkCmdPipelineBarrier2) \
    F(vkQueueSubmit) F(vkCreateSemaphore) F(vkDestroySemaphore) F(vkGetSemaphoreCounterValue) \
    F(vkWaitSemaphores) F(vkDeviceWaitIdle) F(vkCreateImage) F(vkDestroyImage) \
    F(vkGetImageMemoryRequirements) F(vkBindImageMemory) F(vkCreateImageView) \
    F(vkDestroyImageView) F(vkCreateSampler) F(vkDestroySampler) F(vkCmdCopyBufferToImage) \
    F(vkCmdCopyImageToBuffer) F(vkCmdCopyBuffer) F(vkCmdBlitImage) F(vkCmdBeginRendering) F(vkCmdEndRendering) \
    F(vkCmdSetViewport) F(vkCmdSetScissor) F(vkCmdDraw) F(vkCmdDrawIndexed) \
    F(vkCmdBindIndexBuffer) F(vkCmdPushDescriptorSetKHR) F(vkCreateSwapchainKHR) \
    F(vkDestroySwapchainKHR) F(vkGetSwapchainImagesKHR) F(vkAcquireNextImageKHR) \
    F(vkQueuePresentKHR)

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
static VkDeviceSize uniformalign;

static VkDescriptorSetLayout heaplayout, uniformlayout;
static VkDescriptorPool pool;
static VkDescriptorSet heapset;
static VkPipelineLayout pipelinelayout;
static VkCommandPool cmdpool;
static VkSemaphore timeline;
static uint64_t submitted;
static VkSampler samplers[4];

/* Memory a buffer or a staging copy holds. */
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
} vkbuffer;

typedef struct {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    VkFormat format;
    VkImageAspectFlags aspect;
} vkimage;

static vkbuffer buffers[GOOSE_GPU_HEAP_SLOTS];
static vkimage images[GOOSE_GPU_HEAP_SLOTS];

/* The command buffer being recorded, the uploads waiting for the next
   submission, and both kinds in flight with the fence that frees each,
   staging memory included. */
static VkCommandBuffer cmd, uploads;
#define MAX_INFLIGHT 128
static struct { VkCommandBuffer cmd; vkbuffer staging; uint64_t fence; } inflight[MAX_INFLIGHT];
#define MAX_PENDING 1024
static vkbuffer pendingstaging[MAX_PENDING];
static int npending;

/* The uniform ring, host-visible: each draw's uniform bytes, pushed as the
   descriptor of set 1. */
#define RING_SIZE (4 << 20)
static vkbuffer ring;
static uint64_t ringhead, ringcmdstart;
static uint64_t ringgen = UINT64_MAX;
static VkDeviceSize ringat;

/* The window's swap chain: its images, a render-done semaphore each, and a
   ring of acquire semaphores, each free again once the submission that
   waited on it has completed. */
#define MAX_SWAP 8
#define ACQUIRES 4
static VkSurfaceKHR surface;
static VkSwapchainKHR swapchain;
static VkFormat swapformat;
static VkExtent2D swapextent;
static uint32_t nswap;
static VkImage swapimages[MAX_SWAP];
static VkImageView swapviews[MAX_SWAP];
static VkSemaphore renderdone[MAX_SWAP];
static struct { VkSemaphore sem; uint64_t fence; } acquires[ACQUIRES];
static int acquirenext;
static int vsyncwanted;
static void *sdlwindow;
static int acquired = -1;           /* the image this frame draws into */
static int acquireslot = -1;        /* its acquire semaphore, until a submission waits */

static int inrender;
static VkPipelineBindPoint bindpoint;

static int check(VkResult r, const char *what) {
    if (r == VK_SUCCESS) return 1;
    goose_gpu_fail("Vulkan: %s failed (VkResult %d)", what, (int)r);
    return 0;
}

static VkFormat vk_format(int f) {
    switch (f) {
        case GOOSE_GPU_RGBA8: return VK_FORMAT_R8G8B8A8_UNORM;
        case GOOSE_GPU_BGRA8: return VK_FORMAT_B8G8R8A8_UNORM;
        case GOOSE_GPU_RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case GOOSE_GPU_R8: return VK_FORMAT_R8_UNORM;
        case GOOSE_GPU_RG8: return VK_FORMAT_R8G8_UNORM;
        case GOOSE_GPU_RGBA16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case GOOSE_GPU_RGBA32F: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case GOOSE_GPU_R16F: return VK_FORMAT_R16_SFLOAT;
        case GOOSE_GPU_RG16F: return VK_FORMAT_R16G16_SFLOAT;
        case GOOSE_GPU_R32F: return VK_FORMAT_R32_SFLOAT;
        case GOOSE_GPU_RG32F: return VK_FORMAT_R32G32_SFLOAT;
        case GOOSE_GPU_R32UI: return VK_FORMAT_R32_UINT;
        case GOOSE_GPU_RGBA8UI: return VK_FORMAT_R8G8B8A8_UINT;
        case GOOSE_GPU_RGB10A2: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case GOOSE_GPU_RG11B10F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case GOOSE_GPU_DEPTH16: return VK_FORMAT_D16_UNORM;
        case GOOSE_GPU_DEPTH24: case GOOSE_GPU_DEPTH32F: case GOOSE_GPU_DEPTH:
            return VK_FORMAT_D32_SFLOAT;
        case GOOSE_GPU_DEPTH24_STENCIL8: return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case GOOSE_GPU_BC1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case GOOSE_GPU_BC1_SRGB: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
        case GOOSE_GPU_BC3: return VK_FORMAT_BC3_UNORM_BLOCK;
        case GOOSE_GPU_BC3_SRGB: return VK_FORMAT_BC3_SRGB_BLOCK;
        case GOOSE_GPU_BC4: return VK_FORMAT_BC4_UNORM_BLOCK;
        case GOOSE_GPU_BC5: return VK_FORMAT_BC5_UNORM_BLOCK;
        case GOOSE_GPU_BC7: return VK_FORMAT_BC7_UNORM_BLOCK;
        case GOOSE_GPU_BC7_SRGB: return VK_FORMAT_BC7_SRGB_BLOCK;
        default: return VK_FORMAT_UNDEFINED;
    }
}

/* Where an access waits and what it does, and the layout it needs. */
typedef struct {
    VkPipelineStageFlags2 stage;
    VkAccessFlags2 access;
    VkImageLayout layout;
} vkaccess;

static vkaccess access_of(int a) {
    switch (a) {
        case GOOSE_GPU_ACCESS_SHADER_READ:
            return (vkaccess) { VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                                VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
        case GOOSE_GPU_ACCESS_SHADER_WRITE:
            return (vkaccess) { VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_GENERAL };
        case GOOSE_GPU_ACCESS_RENDER_TARGET:
            return (vkaccess) { VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL };
        case GOOSE_GPU_ACCESS_DEPTH_WRITE:
            return (vkaccess) { VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL };
        case GOOSE_GPU_ACCESS_DEPTH_READ:
            return (vkaccess) { VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_SHADER_READ_BIT,
                                VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
        case GOOSE_GPU_ACCESS_COPY_SRC:
            return (vkaccess) { VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL };
        case GOOSE_GPU_ACCESS_COPY_DST:
            return (vkaccess) { VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL };
        case GOOSE_GPU_ACCESS_ACCEL_INPUT:
            /* The build's reads, and the instance conversion's before it. */
            return (vkaccess) { VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                VK_ACCESS_2_SHADER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED };
        case GOOSE_GPU_ACCESS_PRESENT:
            return (vkaccess) { VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR };
        default:
            return (vkaccess) { VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                                VK_IMAGE_LAYOUT_UNDEFINED };
    }
}

/* Mips `base` to `base + count` of an image. */
static void mip_barrier(VkCommandBuffer cb, vkimage *im, uint32_t base, uint32_t count,
                        vkaccess from, vkaccess to) {
    VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = from.stage ? from.stage : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    b.srcAccessMask = from.access;
    b.dstStageMask = to.stage ? to.stage : VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    b.dstAccessMask = to.access;
    b.oldLayout = from.layout;
    b.newLayout = to.layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im->image;
    b.subresourceRange = (VkImageSubresourceRange) { im->aspect, base, count, 0,
                                                     VK_REMAINING_ARRAY_LAYERS };
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &di);
}

static void image_barrier(VkCommandBuffer cb, vkimage *im, vkaccess from, vkaccess to) {
    mip_barrier(cb, im, 0, VK_REMAINING_MIP_LEVELS, from, to);
}

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 srca,
                           VkPipelineStageFlags2 dst, VkAccessFlags2 dsta) {
    VkMemoryBarrier2 mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mb.srcStageMask = src;
    mb.srcAccessMask = srca;
    mb.dstStageMask = dst;
    mb.dstAccessMask = dsta;
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
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

/* The loader's path, for SDL to load the same one: its own search on a Mac
   looks for MoltenVK. Null where there is none. */
const char *goose_gpu_vulkan_loader_path(void) {
    if (!load_global()) return NULL;
#ifdef _WIN32
    static char path[1024];
    if (!GetModuleFileNameA((HMODULE)loader, path, sizeof path)) return NULL;
    return path;
#else
    Dl_info info;
    if (!dladdr((void *)vkGetInstanceProcAddr, &info) || !info.dli_fname) return NULL;
    return info.dli_fname;
#endif
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

static int has_extension(const char *name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(physical, NULL, &n, NULL);
    VkExtensionProperties *props = calloc(n ? n : 1, sizeof *props);
    vkEnumerateDeviceExtensionProperties(physical, NULL, &n, props);
    int found = 0;
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(props[i].extensionName, name)) found = 1;
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

/* A host-visible buffer, mapped. */
static int host_buffer(vkbuffer *b, VkDeviceSize size, VkBufferUsageFlags usage) {
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = usage;
    if (!check(vkCreateBuffer(device, &bci, NULL, &b->buffer), "vkCreateBuffer")) return 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b->buffer, &req);
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
    if (!check(vkAllocateMemory(device, &mai, NULL, &b->memory), "vkAllocateMemory")) return 0;
    vkBindBufferMemory(device, b->buffer, b->memory, 0);
    return check(vkMapMemory(device, b->memory, 0, VK_WHOLE_SIZE, 0, &b->mapped), "vkMapMemory");
}

static void free_buffer(vkbuffer *b) {
    if (b->buffer) vkDestroyBuffer(device, b->buffer, NULL);
    if (b->memory) vkFreeMemory(device, b->memory, NULL);
    memset(b, 0, sizeof *b);
}

static VkDescriptorType mutabletypes[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                           VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };

static int make_samplers(void) {
    for (int i = 0; i < 4; i++) {
        VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        VkFilter f = i & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        VkSamplerAddressMode m = i & 2 ? VK_SAMPLER_ADDRESS_MODE_REPEAT
                                       : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.magFilter = sci.minFilter = f;
        sci.mipmapMode = i & 1 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = m;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        if (!check(vkCreateSampler(device, &sci, NULL, &samplers[i]), "vkCreateSampler")) return 0;
        VkDescriptorImageInfo info = { samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
        VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = heapset;
        w.dstBinding = 1;
        w.dstArrayElement = (uint32_t)i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
    }
    return 1;
}

static int make_swapchain(void);
static void vk_wait(uint64_t fence);

static int vk_open(void *window, int vsync) {
    if (!load_global()) return 0;
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "goose";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    sdlwindow = window;
    vsyncwanted = vsync;
    int portability = 0;
#ifdef GOOSE_GPU_SDL
    /* The surface extensions SDL needs for the window. */
    if (window) {
        Uint32 next = 0;
        const char *const *wext = SDL_Vulkan_GetInstanceExtensions(&next);
        ici.enabledExtensionCount = next;
        ici.ppEnabledExtensionNames = wext;
        for (Uint32 i = 0; i < next; i++)
            if (!strcmp(wext[i], VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) portability = 1;
        if (portability) ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
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
    physical = VK_NULL_HANDLE;
    /* A portability driver (MoltenVK) only where nothing else is offered. */
    for (uint32_t i = 0; i < n && !physical; i++) {
        VkPhysicalDeviceDriverProperties drv = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        p2.pNext = &drv;
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        if (drv.driverID != VK_DRIVER_ID_MOLTENVK) physical = devs[i];
    }
    if (!physical) {
        goose_gpu_fail("no Vulkan device but MoltenVK, which the gpu package does not use: on a "
                       "Mac, install KosmicKrisp (the Vulkan SDK's conformant driver)");
        return 0;
    }
    VkPhysicalDeviceProperties2 props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    vkGetPhysicalDeviceProperties2(physical, &props);
    const char *dn = props.properties.deviceName;
    if (props.properties.apiVersion < VK_API_VERSION_1_3) {
        goose_gpu_fail("the Vulkan device (%s) is not Vulkan 1.3", dn);
        return 0;
    }
    uniformalign = props.properties.limits.minUniformBufferOffsetAlignment;
    if (uniformalign < 256) uniformalign = 256;
    vkGetPhysicalDeviceMemoryProperties(physical, &memprops);
    if (!has_extension(VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME) ||
        !has_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        goose_gpu_fail("the Vulkan device (%s) lacks VK_EXT_mutable_descriptor_type or "
                       "VK_KHR_push_descriptor", dn);
        return 0;
    }

    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT fm =
        { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT };
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    f13.pNext = &fm;
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(physical, &f2);
    if (!f12.descriptorIndexing || !f12.runtimeDescriptorArray ||
        !f12.descriptorBindingPartiallyBound ||
        !f12.descriptorBindingStorageBufferUpdateAfterBind ||
        !f12.descriptorBindingSampledImageUpdateAfterBind || !f12.scalarBlockLayout ||
        !f12.timelineSemaphore || !f13.dynamicRendering || !f13.synchronization2 ||
        !fm.mutableDescriptorType) {
        goose_gpu_fail("the Vulkan device (%s) lacks descriptor indexing, mutable descriptors, "
                       "scalar block layout, timeline semaphores, dynamic rendering or "
                       "synchronization2", dn);
        return 0;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, NULL);
    VkQueueFamilyProperties qprops[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, qprops);
    queuefamily = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if ((qprops[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
            queuefamily = i;
            break;
        }
    if (queuefamily == UINT32_MAX) {
        goose_gpu_fail("the Vulkan device has no graphics and compute queue");
        return 0;
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = queuefamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT wm =
        { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT };
    wm.mutableDescriptorType = VK_TRUE;
    VkPhysicalDeviceVulkan13Features w13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    w13.pNext = &wm;
    w13.dynamicRendering = VK_TRUE;
    w13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features w12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    w12.pNext = &w13;
    w12.descriptorIndexing = VK_TRUE;
    w12.runtimeDescriptorArray = VK_TRUE;
    w12.descriptorBindingPartiallyBound = VK_TRUE;
    w12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    w12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    w12.descriptorBindingStorageImageUpdateAfterBind = f12.descriptorBindingStorageImageUpdateAfterBind;
    w12.shaderStorageBufferArrayNonUniformIndexing = f12.shaderStorageBufferArrayNonUniformIndexing;
    w12.shaderSampledImageArrayNonUniformIndexing = f12.shaderSampledImageArrayNonUniformIndexing;
    w12.scalarBlockLayout = VK_TRUE;
    w12.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceFeatures2 w2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    w2.pNext = &w12;
    w2.features.fillModeNonSolid = f2.features.fillModeNonSolid;
    w2.features.vertexPipelineStoresAndAtomics = f2.features.vertexPipelineStoresAndAtomics;
    w2.features.fragmentStoresAndAtomics = f2.features.fragmentStoresAndAtomics;
    w2.features.samplerAnisotropy = f2.features.samplerAnisotropy;
    w2.features.textureCompressionBC = f2.features.textureCompressionBC;
    w2.features.shaderStorageImageReadWithoutFormat = f2.features.shaderStorageImageReadWithoutFormat;
    w2.features.shaderStorageImageWriteWithoutFormat = f2.features.shaderStorageImageWriteWithoutFormat;
    const char *exts[] = { VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME,
                           VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &w2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = window ? 3 : 2;
    dci.ppEnabledExtensionNames = exts;
    if (!check(vkCreateDevice(physical, &dci, NULL, &device), "vkCreateDevice")) return 0;
#define LOAD(f) f = (PFN_##f)vkGetDeviceProcAddr(device, #f);
    VK_DEVICE(LOAD)
#undef LOAD
    vkGetDeviceQueue(device, queuefamily, 0, &queue);

    /* Set 0, the heaps. */
    VkMutableDescriptorTypeListEXT list = { 3, mutabletypes };
    VkMutableDescriptorTypeCreateInfoEXT mci = { VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT };
    mci.mutableDescriptorTypeListCount = 1;
    mci.pMutableDescriptorTypeLists = &list;
    VkDescriptorSetLayoutBinding b[2] = { { 0 }, { 0 } };
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_MUTABLE_EXT;
    b[0].descriptorCount = GOOSE_GPU_HEAP_SLOTS;
    b[0].stageFlags = VK_SHADER_STAGE_ALL;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    b[1].descriptorCount = GOOSE_GPU_SAMPLERS;
    b[1].stageFlags = VK_SHADER_STAGE_ALL;
    VkDescriptorBindingFlags bflags[2] = {
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi =
        { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bfi.pNext = &mci;
    bfi.bindingCount = 2;
    bfi.pBindingFlags = bflags;
    VkDescriptorSetLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.pNext = &bfi;
    lci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    lci.bindingCount = 2;
    lci.pBindings = b;
    if (!check(vkCreateDescriptorSetLayout(device, &lci, NULL, &heaplayout), "heap layout"))
        return 0;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_MUTABLE_EXT, GOOSE_GPU_HEAP_SLOTS },
                                   { VK_DESCRIPTOR_TYPE_SAMPLER, GOOSE_GPU_SAMPLERS } };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.pNext = &mci;
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pci.maxSets = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = ps;
    if (!check(vkCreateDescriptorPool(device, &pci, NULL, &pool), "descriptor pool")) return 0;
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &heaplayout;
    if (!check(vkAllocateDescriptorSets(device, &dai, &heapset), "descriptor set")) return 0;

    /* Set 1, the uniform buffer, pushed. */
    VkDescriptorSetLayoutBinding ub = { 0 };
    ub.binding = 0;
    ub.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ub.descriptorCount = 1;
    ub.stageFlags = VK_SHADER_STAGE_ALL;
    VkDescriptorSetLayoutCreateInfo uci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    uci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    uci.bindingCount = 1;
    uci.pBindings = &ub;
    if (!check(vkCreateDescriptorSetLayout(device, &uci, NULL, &uniformlayout), "uniform layout"))
        return 0;

    VkPushConstantRange range = { VK_SHADER_STAGE_ALL, 0, 128 };
    VkDescriptorSetLayout sets[2] = { heaplayout, uniformlayout };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 2;
    plci.pSetLayouts = sets;
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
    if (!host_buffer(&ring, RING_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) return 0;
    submitted = 0;
    ringhead = ringcmdstart = 0;
    ringgen = UINT64_MAX;
    if (!make_samplers()) return 0;
    if (!window) return 1;
#ifdef GOOSE_GPU_SDL
    if (!SDL_Vulkan_CreateSurface((SDL_Window *)window, instance, NULL, &surface)) {
        goose_gpu_fail("SDL made no Vulkan surface: %s", SDL_GetError());
        return 0;
    }
    VkBool32 ok = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical, queuefamily, surface, &ok);
    if (!ok) {
        goose_gpu_fail("the Vulkan queue cannot present to the window");
        return 0;
    }
    for (int i = 0; i < ACQUIRES; i++) {
        VkSemaphoreCreateInfo bsi = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (!check(vkCreateSemaphore(device, &bsi, NULL, &acquires[i].sem), "vkCreateSemaphore"))
            return 0;
        acquires[i].fence = 0;
    }
    return make_swapchain();
#else
    goose_gpu_fail("no SDL: no window");
    return 0;
#endif
}

static void drop_swapchain(void) {
    for (uint32_t i = 0; i < nswap; i++) {
        vkDestroyImageView(device, swapviews[i], NULL);
        vkDestroySemaphore(device, renderdone[i], NULL);
    }
    nswap = 0;
    if (swapchain) vkDestroySwapchainKHR(device, swapchain, NULL);
    swapchain = VK_NULL_HANDLE;
}

/* The swap chain, made again when the window's size changes. */
static int make_swapchain(void) {
    vkDeviceWaitIdle(device);
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        int w = 0, h = 0;
#ifdef GOOSE_GPU_SDL
        SDL_GetWindowSizeInPixels((SDL_Window *)sdlwindow, &w, &h);
#endif
        extent = (VkExtent2D) { (uint32_t)w, (uint32_t)h };
    }
    if (!extent.width || !extent.height) return 1;     /* minimized: none for now */
    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &nf, NULL);
    VkSurfaceFormatKHR formats[64];
    if (nf > 64) nf = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &nf, formats);
    swapformat = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    for (uint32_t i = 0; i < nf; i++)
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM) {
            swapformat = formats[i].format;
            space = formats[i].colorSpace;
        }
    if (swapformat == VK_FORMAT_UNDEFINED) {
        goose_gpu_fail("the window takes no BGRA8 swap chain");
        return 0;
    }
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsyncwanted) {
        uint32_t nm = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &nm, NULL);
        VkPresentModeKHR modes[16];
        if (nm > 16) nm = 16;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &nm, modes);
        for (uint32_t i = 0; i < nm; i++)
            if (modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR || modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
                mode = modes[i];
    }
    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > MAX_SWAP) count = MAX_SWAP;
    VkSwapchainKHR old = swapchain;
    VkSwapchainCreateInfoKHR sci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sci.surface = surface;
    sci.minImageCount = count;
    sci.imageFormat = swapformat;
    sci.imageColorSpace = space;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = old;
    VkSwapchainKHR made;
    if (!check(vkCreateSwapchainKHR(device, &sci, NULL, &made), "vkCreateSwapchainKHR")) return 0;
    swapchain = old;
    drop_swapchain();
    swapchain = made;
    swapextent = extent;
    nswap = MAX_SWAP;
    vkGetSwapchainImagesKHR(device, swapchain, &nswap, swapimages);
    for (uint32_t i = 0; i < nswap; i++) {
        VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = swapimages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = swapformat;
        vci.subresourceRange = (VkImageSubresourceRange) { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (!check(vkCreateImageView(device, &vci, NULL, &swapviews[i]), "vkCreateImageView"))
            return 0;
        VkSemaphoreCreateInfo bsi = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (!check(vkCreateSemaphore(device, &bsi, NULL, &renderdone[i]), "vkCreateSemaphore"))
            return 0;
    }
    return 1;
}

static int vk_acquire(uint32_t slot, int *width, int *height) {
    if (!surface) return 0;
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps);
    if (!swapchain || (caps.currentExtent.width != UINT32_MAX &&
                       (caps.currentExtent.width != swapextent.width ||
                        caps.currentExtent.height != swapextent.height)))
        if (!make_swapchain() || !swapchain) return 0;
    for (int tries = 0; tries < 2; tries++) {
        int a = acquirenext;
        vk_wait(acquires[a].fence);
        uint32_t index = 0;
        VkResult r = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquires[a].sem,
                                           VK_NULL_HANDLE, &index);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            if (!make_swapchain() || !swapchain) return 0;
            continue;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            check(r, "vkAcquireNextImageKHR");
            return 0;
        }
        acquirenext = (acquirenext + 1) % ACQUIRES;
        acquireslot = a;
        acquired = (int)index;
        vkimage *im = &images[slot];
        im->image = swapimages[index];
        im->view = swapviews[index];
        im->memory = VK_NULL_HANDLE;
        im->format = swapformat;
        im->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        *width = (int)swapextent.width;
        *height = (int)swapextent.height;
        return 1;
    }
    return 0;
}

static uint64_t vk_completed(void) {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(device, timeline, &v);
    return v;
}

static void vk_wait(uint64_t fence);

static void reclaim(void) {
    uint64_t done = vk_completed();
    for (int i = 0; i < MAX_INFLIGHT; i++)
        if (inflight[i].fence && inflight[i].fence <= done) {
            if (inflight[i].cmd) vkFreeCommandBuffers(device, cmdpool, 1, &inflight[i].cmd);
            free_buffer(&inflight[i].staging);
            memset(&inflight[i], 0, sizeof inflight[i]);
        }
}

/* Holds a command buffer or staging memory until `fence` passes. */
static void retire(VkCommandBuffer cb, vkbuffer *staging, uint64_t fence) {
    for (int tries = 0; tries < 2; tries++) {
        for (int i = 0; i < MAX_INFLIGHT; i++)
            if (!inflight[i].fence) {
                inflight[i].cmd = cb;
                if (staging) inflight[i].staging = *staging;
                inflight[i].fence = fence;
                return;
            }
        vk_wait(vk_completed() + 1);
    }
}

static void vk_close(void) {
    if (!device) return;
    vkDeviceWaitIdle(device);
    if (surface) {
        drop_swapchain();
        for (int i = 0; i < ACQUIRES; i++) vkDestroySemaphore(device, acquires[i].sem, NULL);
        vkDestroySurfaceKHR(instance, surface, NULL);
        surface = VK_NULL_HANDLE;
    }
    acquired = acquireslot = -1;
    reclaim();
    if (cmd) vkFreeCommandBuffers(device, cmdpool, 1, &cmd);
    if (uploads) vkFreeCommandBuffers(device, cmdpool, 1, &uploads);
    cmd = uploads = VK_NULL_HANDLE;
    for (int i = 0; i < npending; i++) free_buffer(&pendingstaging[i]);
    npending = 0;
    free_buffer(&ring);
    for (int i = 0; i < 4; i++) vkDestroySampler(device, samplers[i], NULL);
    vkDestroySemaphore(device, timeline, NULL);
    vkDestroyCommandPool(device, cmdpool, NULL);
    vkDestroyPipelineLayout(device, pipelinelayout, NULL);
    vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyDescriptorSetLayout(device, uniformlayout, NULL);
    vkDestroyDescriptorSetLayout(device, heaplayout, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    device = VK_NULL_HANDLE;
    instance = VK_NULL_HANDLE;
}

static VkCommandBuffer begin_cb(void) {
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = cmdpool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (!check(vkAllocateCommandBuffers(device, &cai, &cb), "vkAllocateCommandBuffers"))
        return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

static int begin(void) {
    if (cmd) return 1;
    reclaim();
    cmd = begin_cb();
    ringcmdstart = ringhead;
    return cmd != VK_NULL_HANDLE;
}

static void write_heap(uint32_t slot, VkDescriptorType type, VkDescriptorBufferInfo *bi,
                       VkDescriptorImageInfo *ii) {
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = heapset;
    w.dstBinding = 0;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = type;
    w.pBufferInfo = bi;
    w.pImageInfo = ii;
    vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
}

static int vk_buffer_create(uint32_t slot, int64_t usage, const void *data, int64_t size) {
    (void)usage;
    vkbuffer *vb = &buffers[slot];
    if (!host_buffer(vb, (VkDeviceSize)size,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT))
        return 0;
    if (data) memcpy(vb->mapped, data, (size_t)size);
    else memset(vb->mapped, 0, (size_t)size);
    VkDescriptorBufferInfo info = { vb->buffer, 0, VK_WHOLE_SIZE };
    write_heap(slot, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &info, NULL);
    return 1;
}

/* Uploads (§3): staged now, and copied in a command buffer submitted ahead
   of the next submission. It opens with a barrier on all the work submitted
   before it, orders its copies one after another, and closes with one
   before all the work after it (vk_submit). */
static vkbuffer *stage(const void *data, int64_t len) {
    if (npending == MAX_PENDING) {
        goose_gpu_fail("more than %d uploads before a submission", MAX_PENDING);
        return NULL;
    }
    vkbuffer *st = &pendingstaging[npending];
    if (!host_buffer(st, (VkDeviceSize)len, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return NULL;
    npending++;
    memcpy(st->mapped, data, (size_t)len);
    if (!uploads) {
        uploads = begin_cb();
        memory_barrier(uploads, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    } else {
        /* After the copies before it: two to one resource write in order. */
        memory_barrier(uploads, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    return st;
}

static int vk_update_buffer(uint32_t slot, int64_t offset, const void *data, int64_t size) {
    vkbuffer *st = stage(data, size);
    if (!st) return 0;
    VkBufferCopy region = { 0, (VkDeviceSize)offset, (VkDeviceSize)size };
    vkCmdCopyBuffer(uploads, st->buffer, buffers[slot].buffer, 1, &region);
    return 1;
}

/* A texture's mip 0, left by the work submitted in `access`, and by the
   copy in that access again, or in shader reads when it had none. */
static int vk_update_texture_at(uint32_t slot, int access, const void *data, int64_t size) {
    vkbuffer *st = stage(data, size);
    if (!st) return 0;
    vkimage *im = &images[slot];
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    int after = access == GOOSE_GPU_ACCESS_NONE ? GOOSE_GPU_ACCESS_SHADER_READ : access;
    image_barrier(uploads, im, access_of(access), access_of(GOOSE_GPU_ACCESS_COPY_DST));
    VkBufferImageCopy region = { 0 };
    region.imageSubresource = (VkImageSubresourceLayers) { im->aspect, 0, 0, 1 };
    region.imageExtent = (VkExtent3D) { (uint32_t)t->desc.width, (uint32_t)t->desc.height, 1 };
    vkCmdCopyBufferToImage(uploads, st->buffer, im->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);
    image_barrier(uploads, im, access_of(GOOSE_GPU_ACCESS_COPY_DST), access_of(after));
    return 1;
}

static int vk_update_texture(uint32_t slot, int access, const void *data, int64_t size) {
    return vk_update_texture_at(slot, access, data, size);
}

static int vk_texture_create(uint32_t slot, const goose_gpu_texture_desc *d, const void *data,
                             int64_t len) {
    vkimage *im = &images[slot];
    im->format = vk_format(d->format);
    int depth = goose_gpu_is_depth(d->format);
    im->aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = im->format;
    ici.extent = (VkExtent3D) { (uint32_t)d->width, (uint32_t)d->height, 1 };
    ici.mipLevels = (uint32_t)d->mips;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (d->usage & GOOSE_GPU_SAMPLED) ici.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (d->usage & GOOSE_GPU_COLOR_TARGET) ici.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (d->usage & GOOSE_GPU_DEPTH_TARGET) ici.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (d->usage & GOOSE_GPU_STORAGE) ici.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!check(vkCreateImage(device, &ici, NULL, &im->image), "vkCreateImage")) return 0;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, im->image, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = memory_type(req.memoryTypeBits, 0, 0);
    if (!check(vkAllocateMemory(device, &mai, NULL, &im->memory), "vkAllocateMemory")) return 0;
    vkBindImageMemory(device, im->image, im->memory, 0);
    VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = im->image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = im->format;
    vci.subresourceRange = (VkImageSubresourceRange) { im->aspect, 0, (uint32_t)d->mips, 0, 1 };
    if (!check(vkCreateImageView(device, &vci, NULL, &im->view), "vkCreateImageView")) return 0;
    /* A slot holds one kind of descriptor: what shaders sample, else what
       they write. */
    if (d->usage & GOOSE_GPU_SAMPLED) {
        VkDescriptorImageInfo info = { VK_NULL_HANDLE, im->view, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
        write_heap(slot, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, NULL, &info);
    } else if (d->usage & GOOSE_GPU_STORAGE) {
        VkDescriptorImageInfo info = { VK_NULL_HANDLE, im->view, VK_IMAGE_LAYOUT_GENERAL };
        write_heap(slot, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, NULL, &info);
    }
    if (!data) return 1;
    return vk_update_texture_at(slot, GOOSE_GPU_ACCESS_NONE, data, len);
}

/* Mips (§3): each blitted from the one before it, which a barrier first
   makes a copy source, then every mip left in shader reads. */
static int vk_generate_mips(uint32_t slot, int access, int upload) {
    VkCommandBuffer cb = upload ? uploads : (begin() ? cmd : VK_NULL_HANDLE);
    if (!cb) return 0;
    vkimage *im = &images[slot];
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(physical, im->format, &fp);
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if ((fp.optimalTilingFeatures & need) != need) {
        goose_gpu_fail("the device cannot filter format %d into mips", t->desc.format);
        return 0;
    }
    uint32_t mips = (uint32_t)t->desc.mips;
    vkaccess src = access_of(GOOSE_GPU_ACCESS_COPY_SRC), dst = access_of(GOOSE_GPU_ACCESS_COPY_DST);
    mip_barrier(cb, im, 0, 1, access_of(access), src);
    mip_barrier(cb, im, 1, mips - 1, access_of(access), dst);
    int32_t w = t->desc.width, h = t->desc.height;
    for (uint32_t i = 1; i < mips; i++) {
        int32_t mw = w > 1 ? w / 2 : 1, mh = h > 1 ? h / 2 : 1;
        VkImageBlit blit = { 0 };
        blit.srcSubresource = (VkImageSubresourceLayers) { im->aspect, i - 1, 0, 1 };
        blit.srcOffsets[1] = (VkOffset3D) { w, h, 1 };
        blit.dstSubresource = (VkImageSubresourceLayers) { im->aspect, i, 0, 1 };
        blit.dstOffsets[1] = (VkOffset3D) { mw, mh, 1 };
        vkCmdBlitImage(cb, im->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im->image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        mip_barrier(cb, im, i, 1, dst, src);
        w = mw;
        h = mh;
    }
    mip_barrier(cb, im, 0, mips, src, access_of(GOOSE_GPU_ACCESS_SHADER_READ));
    return 1;
}

static void vk_copy_to_texture(uint32_t buf, int64_t offset, uint32_t slot) {
    if (!begin()) return;
    vkimage *im = &images[slot];
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    VkBufferImageCopy region = { 0 };
    region.bufferOffset = (VkDeviceSize)offset;
    region.imageSubresource = (VkImageSubresourceLayers) { im->aspect, 0, 0, 1 };
    region.imageExtent = (VkExtent3D) { (uint32_t)t->desc.width, (uint32_t)t->desc.height, 1 };
    vkCmdCopyBufferToImage(cmd, buffers[buf].buffer, im->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &region);
}

/* KosmicKrisp has no ray queries yet; VK_KHR_ray_query comes with a device
   that does (docs/design/gpu.md §7). */
static int vk_ray_query(void) { return 0; }
static int vk_accel_create(uint32_t slot, const goose_gpu_primitive_accel_desc *p,
                           const goose_gpu_instance_accel_desc *i) {
    (void)slot; (void)p; (void)i;
    goose_gpu_fail("Vulkan ray queries are not implemented");
    return 0;
}
static int vk_accel_build(uint32_t slot, int64_t n) { (void)slot; (void)n; return 0; }
static uint64_t vk_accel_reference(uint32_t slot) { (void)slot; return 0; }

/* A storage view: an image view of one mip, written as a storage image in
   general layout. */
static int vk_view_create(uint32_t view, uint32_t texture, int mip) {
    vkimage *im = &images[texture];
    vkimage *v = &images[view];
    memset(v, 0, sizeof *v);
    v->image = im->image;
    v->format = im->format;
    v->aspect = im->aspect;
    VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = im->image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = im->format;
    vci.subresourceRange = (VkImageSubresourceRange) { im->aspect, (uint32_t)mip, 1, 0, 1 };
    if (!check(vkCreateImageView(device, &vci, NULL, &v->view), "vkCreateImageView")) return 0;
    VkDescriptorImageInfo info = { VK_NULL_HANDLE, v->view, VK_IMAGE_LAYOUT_GENERAL };
    write_heap(view, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, NULL, &info);
    return 1;
}

static void vk_release(uint32_t slot, int kind) {
    int texture = kind == GOOSE_GPU_KIND_TEXTURE;
    if (kind == GOOSE_GPU_KIND_ACCEL) return;
    /* Released once the work submitted so far is done (§2, deferred
       destruction); this backend waits for it. */
    vkDeviceWaitIdle(device);
    if (kind == GOOSE_GPU_KIND_VIEW) {
        vkDestroyImageView(device, images[slot].view, NULL);
        memset(&images[slot], 0, sizeof images[slot]);
        return;
    }
    if (texture) {
        vkimage *im = &images[slot];
        if (!im->memory) {          /* a swap chain image, the swap chain's to destroy */
            memset(im, 0, sizeof *im);
            return;
        }
        vkDestroyImageView(device, im->view, NULL);
        vkDestroyImage(device, im->image, NULL);
        vkFreeMemory(device, im->memory, NULL);
        memset(im, 0, sizeof *im);
    } else {
        free_buffer(&buffers[slot]);
    }
}

static VkShaderModule module_of(goose_gpu_pipe *p, int entry, char *name, size_t namesize) {
    const gs_gpu_lib_entry *e = &p->entries[entry];
    gs_gpu_lib_range code = e->code[GS_GPU_FORMAT_SPIRV];
    if (!code.size || (int64_t)code.offset + code.size > p->librarylen || code.size % 4) {
        goose_gpu_fail("the shader library has no SPIR-V");
        return VK_NULL_HANDLE;
    }
    uint32_t *words = malloc(code.size);
    memcpy(words, p->library + code.offset, code.size);
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = code.size;
    smi.pCode = words;
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(device, &smi, NULL, &module);
    free(words);
    if (!check(r, "vkCreateShaderModule")) return VK_NULL_HANDLE;
    size_t n = e->name.size < namesize - 1 ? e->name.size : namesize - 1;
    memcpy(name, p->library + e->name.offset, n);
    name[n] = 0;
    return module;
}

static int vk_compute_create(goose_gpu_pipe *p) {
    char name[256];
    VkShaderModule module = module_of(p, 0, name, sizeof name);
    if (!module) return 0;
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = module;
    cpi.stage.pName = name;
    cpi.layout = pipelinelayout;
    VkPipeline pipe;
    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe);
    vkDestroyShaderModule(device, module, NULL);
    if (!check(r, "vkCreateComputePipelines")) return 0;
    p->native = (void *)pipe;
    return 1;
}

static VkCompareOp compare_of(int c) {
    static const VkCompareOp ops[] = { VK_COMPARE_OP_LESS, VK_COMPARE_OP_NEVER, VK_COMPARE_OP_LESS,
                                       VK_COMPARE_OP_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL,
                                       VK_COMPARE_OP_GREATER, VK_COMPARE_OP_NOT_EQUAL,
                                       VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_ALWAYS };
    return c >= 1 && c <= 8 ? ops[c] : VK_COMPARE_OP_LESS;
}

static void *vk_raster_create(goose_gpu_pipe *p, const goose_gpu_targets *t) {
    char vname[256], fname[256];
    VkShaderModule vs = module_of(p, 0, vname, sizeof vname);
    VkShaderModule fs = vs ? module_of(p, 1, fname, sizeof fname) : VK_NULL_HANDLE;
    if (!fs) {
        if (vs) vkDestroyShaderModule(device, vs, NULL);
        return NULL;
    }
    const goose_gpu_pipeline_desc *d = &p->desc;
    VkPipelineShaderStageCreateInfo stages[2] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT,
          vs, vname, NULL },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fs, fname, NULL },
    };
    VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    static const VkPrimitiveTopology topo[] = { VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                                                VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
                                                VK_PRIMITIVE_TOPOLOGY_POINT_LIST };
    ia.topology = d->primitive >= 0 && d->primitive <= 4 ? topo[d->primitive] : topo[0];
    VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1;
    vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = d->wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = d->cull == GOOSE_GPU_CULL_FRONT ? VK_CULL_MODE_FRONT_BIT
                : d->cull == GOOSE_GPU_CULL_BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    /* The vertex shaders invert y (-fvk-invert-y, src/gpu.h), which turns
       the winding the framebuffer sees around. */
    rs.frontFace = d->clockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = d->depth_test;
    ds.depthWriteEnable = d->depth_write;
    ds.depthCompareOp = compare_of(d->depth_compare);
    VkPipelineColorBlendAttachmentState blends[GOOSE_GPU_MAX_TARGETS];
    VkFormat colors[GOOSE_GPU_MAX_TARGETS];
    uint32_t ncolor = 0;
    for (; ncolor < GOOSE_GPU_MAX_TARGETS && t->colors[ncolor]; ncolor++) {
        VkPipelineColorBlendAttachmentState *b = &blends[ncolor];
        memset(b, 0, sizeof *b);
        b->colorWriteMask = 0xf;
        b->colorBlendOp = b->alphaBlendOp = VK_BLEND_OP_ADD;
        b->srcColorBlendFactor = b->srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        b->dstColorBlendFactor = b->dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        switch (d->blend) {
            case GOOSE_GPU_BLEND_ALPHA:
                b->blendEnable = VK_TRUE;
                b->srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                b->dstColorBlendFactor = b->dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                break;
            case GOOSE_GPU_BLEND_ADD:
                b->blendEnable = VK_TRUE;
                b->dstColorBlendFactor = b->dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                break;
            case GOOSE_GPU_BLEND_PREMULTIPLIED:
                b->blendEnable = VK_TRUE;
                b->dstColorBlendFactor = b->dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                break;
            case GOOSE_GPU_BLEND_MULTIPLY:
                b->blendEnable = VK_TRUE;
                b->srcColorBlendFactor = b->srcAlphaBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
                break;
            default: break;
        }
        colors[ncolor] = vk_format(t->colors[ncolor]);
    }
    VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = ncolor;
    cb.pAttachments = blends;
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    ri.colorAttachmentCount = ncolor;
    ri.pColorAttachmentFormats = colors;
    ri.depthAttachmentFormat = t->depth ? vk_format(t->depth) : VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo gpi = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gpi.pNext = &ri;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pDepthStencilState = &ds;
    gpi.pColorBlendState = &cb;
    gpi.pDynamicState = &dy;
    gpi.layout = pipelinelayout;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, NULL, &pipe);
    vkDestroyShaderModule(device, vs, NULL);
    vkDestroyShaderModule(device, fs, NULL);
    if (!check(r, "vkCreateGraphicsPipelines")) return NULL;
    return (void *)pipe;
}

static void vk_pipeline_release(goose_gpu_pipe *p) {
    vkDeviceWaitIdle(device);
    if (p->native) vkDestroyPipeline(device, (VkPipeline)p->native, NULL);
    for (int i = 0; i < p->nvariants; i++)
        vkDestroyPipeline(device, (VkPipeline)p->variants[i].native, NULL);
    p->native = NULL;
    p->nvariants = 0;
}

static void bind_heaps(VkPipelineBindPoint bp) {
    bindpoint = bp;
    ringgen = UINT64_MAX;
    vkCmdBindDescriptorSets(cmd, bp, pipelinelayout, 0, 1, &heapset, 0, NULL);
}

static int vk_begin_compute(void) {
    if (!begin()) return 0;
    bind_heaps(VK_PIPELINE_BIND_POINT_COMPUTE);
    return 1;
}

static void vk_end_compute(void) {}

static int vk_begin_pass(const goose_gpu_pass *ps) {
    if (!begin()) return 0;
    VkRenderingAttachmentInfo colors[GOOSE_GPU_MAX_TARGETS];
    for (int i = 0; i < ps->ncolor; i++) {
        VkRenderingAttachmentInfo *a = &colors[i];
        memset(a, 0, sizeof *a);
        a->sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        a->imageView = images[ps->desc.color[i]].view;
        a->imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        a->loadOp = ps->desc.clear_color ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        a->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        memcpy(a->clearValue.color.float32, ps->desc.color_value, sizeof(float) * 4);
    }
    VkRenderingAttachmentInfo depth = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    if (ps->desc.depth) {
        depth.imageView = images[ps->desc.depth].view;
        depth.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;
        depth.loadOp = ps->desc.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth.clearValue.depthStencil.depth = ps->desc.depth_value;
    }
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea.extent = (VkExtent2D) { (uint32_t)ps->width, (uint32_t)ps->height };
    ri.layerCount = 1;
    ri.colorAttachmentCount = (uint32_t)ps->ncolor;
    ri.pColorAttachments = colors;
    ri.pDepthAttachment = ps->desc.depth ? &depth : NULL;
    vkCmdBeginRendering(cmd, &ri);
    inrender = 1;
    bind_heaps(VK_PIPELINE_BIND_POINT_GRAPHICS);
    VkViewport v = { 0, 0, (float)ps->width, (float)ps->height, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &v);
    VkRect2D sc = { { 0, 0 }, { (uint32_t)ps->width, (uint32_t)ps->height } };
    vkCmdSetScissor(cmd, 0, 1, &sc);
    return 1;
}

static void vk_end_pass(void) {
    vkCmdEndRendering(cmd);
    inrender = 0;
}

static void vk_viewport(float x, float y, float w, float h) {
    VkViewport v = { x, y, w, h, 0, 1 };
    vkCmdSetViewport(cmd, 0, 1, &v);
}

static void vk_scissor(int32_t x, int32_t y, int32_t w, int32_t h) {
    VkRect2D sc = { { x, y }, { (uint32_t)w, (uint32_t)h } };
    vkCmdSetScissor(cmd, 0, 1, &sc);
}

static void vk_index_buffer(uint32_t slot, int size) {
    vkCmdBindIndexBuffer(cmd, buffers[slot].buffer, 0,
                         size == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
}

/* Push constants, and the uniform bytes when they changed: into the ring,
   whose start the submissions still using it hold. */
static void set_blocks(const void *push, uint32_t pushsize, const void *uniforms,
                       uint32_t uniformsize, uint64_t gen) {
    if (pushsize) vkCmdPushConstants(cmd, pipelinelayout, VK_SHADER_STAGE_ALL, 0, pushsize, push);
    if (!uniformsize || gen == ringgen) return;
    uint64_t at = (ringhead + uniformalign - 1) / uniformalign * uniformalign;
    if (at + uniformsize > RING_SIZE) {
        if (ringcmdstart < uniformsize) {
            goose_gpu_misuse("more than %d bytes of uniforms in one submission", RING_SIZE);
            return;
        }
        vk_wait(submitted);
        at = 0;
    }
    ringhead = at + uniformsize;
    memcpy((uint8_t *)ring.mapped + at, uniforms, uniformsize);
    VkDescriptorBufferInfo info = { ring.buffer, at, uniformsize };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w.pBufferInfo = &info;
    vkCmdPushDescriptorSetKHR(cmd, bindpoint, pipelinelayout, 1, 1, &w);
    ringgen = gen;
}

static void vk_dispatch(goose_gpu_pipe *p, const void *push, uint32_t pushsize,
                        const void *uniforms, uint32_t uniformsize, uint64_t gen,
                        uint32_t x, uint32_t y, uint32_t z) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)p->native);
    set_blocks(push, pushsize, uniforms, uniformsize, gen);
    vkCmdDispatch(cmd, x, y, z);
    /* Each dispatch sees the ones before it in its pass (§3: barriers are
       the program's between passes). */
    memory_barrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
}

static void vk_draw(goose_gpu_pipe *p, void *native, const void *push, uint32_t pushsize,
                    const void *uniforms, uint32_t uniformsize, uint64_t gen, int indexed,
                    uint32_t count, uint32_t instances, uint32_t first, int32_t offset,
                    uint32_t first_instance) {
    (void)p;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipeline)native);
    set_blocks(push, pushsize, uniforms, uniformsize, gen);
    if (indexed) vkCmdDrawIndexed(cmd, count, instances, first, offset, first_instance);
    else vkCmdDraw(cmd, count, instances, first, first_instance);
}

static void vk_barrier(uint32_t slot, int before, int after) {
    if (!begin()) return;
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    if (!t) {
        vkaccess a = access_of(before), b = access_of(after);
        memory_barrier(cmd, a.stage ? a.stage : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, a.access,
                       b.stage, b.access);
        return;
    }
    image_barrier(cmd, &images[slot], access_of(before), access_of(after));
}

typedef struct { vkbuffer staging; } vkreadback;

static void *vk_readback_buffer(uint32_t slot, int64_t offset, int64_t size) {
    if (!begin()) return NULL;
    vkreadback *rb = calloc(1, sizeof *rb);
    if (!host_buffer(&rb->staging, (VkDeviceSize)size, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        free(rb);
        return NULL;
    }
    memory_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy region = { (VkDeviceSize)offset, 0, (VkDeviceSize)size };
    vkCmdCopyBuffer(cmd, buffers[slot].buffer, rb->staging.buffer, 1, &region);
    return rb;
}

static void *vk_readback_texture(uint32_t slot, int access, int64_t size) {
    if (!begin()) return NULL;
    vkreadback *rb = calloc(1, sizeof *rb);
    if (!host_buffer(&rb->staging, (VkDeviceSize)size, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        free(rb);
        return NULL;
    }
    const goose_gpu_tex *t = goose_gpu_texture_of(slot);
    vkimage *im = &images[slot];
    image_barrier(cmd, im, access_of(access), access_of(GOOSE_GPU_ACCESS_COPY_SRC));
    VkBufferImageCopy region = { 0 };
    region.imageSubresource = (VkImageSubresourceLayers) { im->aspect, 0, 0, 1 };
    region.imageExtent = (VkExtent3D) { (uint32_t)t->desc.width, (uint32_t)t->desc.height, 1 };
    vkCmdCopyImageToBuffer(cmd, im->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           rb->staging.buffer, 1, &region);
    image_barrier(cmd, im, access_of(GOOSE_GPU_ACCESS_COPY_SRC), access_of(access));
    return rb;
}

static void vk_readback_read(void *rb, void *out, int64_t size) {
    memcpy(out, ((vkreadback *)rb)->staging.mapped, (size_t)size);
}

static void vk_readback_release(void *rb) {
    if (!rb) return;
    free_buffer(&((vkreadback *)rb)->staging);
    free(rb);
}

static uint64_t vk_submit(int present) {
    if (!begin()) return submitted;
    /* What the GPU wrote is the host's to read once the fence passes. */
    memory_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    vkEndCommandBuffer(cmd);
    VkCommandBuffer list[2];
    uint32_t n = 0;
    if (uploads) {
        memory_barrier(uploads, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        vkEndCommandBuffer(uploads);
        list[n++] = uploads;
    }
    list[n++] = cmd;
    uint64_t fence = submitted + 1;
    /* The screen's image: waited for by the first submission after it was
       acquired, and shown after the one that presents it. */
    VkSemaphore waits[1], signals[2] = { timeline, VK_NULL_HANDLE };
    VkPipelineStageFlags waitstages[1] = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
    uint64_t waitvalues[1] = { 0 }, signalvalues[2] = { fence, 0 };
    uint32_t nwait = 0, nsignal = 1;
    if (acquireslot >= 0) {
        waits[nwait++] = acquires[acquireslot].sem;
        acquires[acquireslot].fence = fence;
        acquireslot = -1;
    }
    int presenting = present && acquired >= 0;
    if (presenting) signals[nsignal++] = renderdone[acquired];
    VkTimelineSemaphoreSubmitInfo tsi = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    tsi.waitSemaphoreValueCount = nwait;
    tsi.pWaitSemaphoreValues = waitvalues;
    tsi.signalSemaphoreValueCount = nsignal;
    tsi.pSignalSemaphoreValues = signalvalues;
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.pNext = &tsi;
    si.waitSemaphoreCount = nwait;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = waitstages;
    si.commandBufferCount = n;
    si.pCommandBuffers = list;
    si.signalSemaphoreCount = nsignal;
    si.pSignalSemaphores = signals;
    if (!check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit")) {
        goose_gpu_misuse("%s", "the submission failed");
        return submitted;
    }
    submitted = fence;
    if (presenting) {
        uint32_t index = (uint32_t)acquired;
        VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &renderdone[index];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &index;
        VkResult r = vkQueuePresentKHR(queue, &pi);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) swapextent = (VkExtent2D) { 0, 0 };
        acquired = -1;
    }
    retire(cmd, NULL, fence);
    if (uploads) retire(uploads, NULL, fence);
    for (int i = 0; i < npending; i++) retire(VK_NULL_HANDLE, &pendingstaging[i], fence);
    npending = 0;
    cmd = uploads = VK_NULL_HANDLE;
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
    vk_buffer_create, vk_texture_create, vk_release, vk_view_create, vk_update_buffer,
    vk_update_texture,
    vk_generate_mips, vk_copy_to_texture,
    vk_ray_query, vk_accel_create, vk_accel_build, vk_accel_reference,
    vk_compute_create, vk_raster_create, vk_pipeline_release,
    vk_begin_compute, vk_end_compute, vk_begin_pass, vk_end_pass,
    vk_dispatch, vk_draw, vk_index_buffer, vk_viewport, vk_scissor,
    vk_barrier,
    vk_readback_buffer, vk_readback_texture, vk_readback_read, vk_readback_release,
    vk_acquire, vk_submit, vk_completed, vk_wait,
};
