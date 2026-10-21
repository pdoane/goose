# The gpu package's native part (docs/design/gpu.md §3): a static library
# the compiler links into programs that import gpu (gpu.build). Metal on
# Apple platforms, Vulkan everywhere; the Vulkan loader is opened at run
# time, so only its headers are needed here.
cmake_minimum_required(VERSION 3.20)
project(goose_gpu C)

set(GOOSE_GPU_SOURCES src/gpu.c src/gpu_vulkan.c)
if(APPLE)
    enable_language(OBJC)
    list(APPEND GOOSE_GPU_SOURCES src/gpu_metal.m)
endif()

add_library(goose_gpu STATIC ${GOOSE_GPU_SOURCES})
target_include_directories(goose_gpu PRIVATE src)

# The Vulkan headers: the SDK's, or the system's.
find_path(GOOSE_GPU_VULKAN_INCLUDE vulkan/vulkan.h
          HINTS $ENV{VULKAN_SDK}/include
                $ENV{HOME}/VulkanSDK/1.4.350.0/macOS/include)
if(NOT GOOSE_GPU_VULKAN_INCLUDE)
    message(FATAL_ERROR "the gpu package needs the Vulkan headers: install the Vulkan SDK "
                        "(vulkan.lunarg.com) and set VULKAN_SDK")
endif()
target_include_directories(goose_gpu PRIVATE ${GOOSE_GPU_VULKAN_INCLUDE})

if(APPLE)
    set_source_files_properties(src/gpu_metal.m PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(goose_gpu PUBLIC "-framework Metal" "-framework Foundation")
elseif(UNIX)
    target_link_libraries(goose_gpu PUBLIC ${CMAKE_DL_LIBS})
endif()
