# gpu

A bindless GPU package for Goose, on Metal 4 and Vulkan, with shaders
written as Goose functions: the design and its plan are
[docs/design/gpu.md](../../docs/design/gpu.md), the shader side
[stdlib/gpu_shader.goose](../../stdlib/gpu_shader.goose).

```goose
import gpu from "path/to/packages/gpu";

compute(64) fn double(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    gpu::store(p.out, i, gpu::load(p.input, i) * 2.0);
}
```

The compiler compiles every entry point a pipeline names when the program
is compiled, which needs DXC (the Vulkan SDK has it) and, on a Mac, Xcode's
Metal toolchain. The package's C needs the Vulkan headers, and SDL3 for
windows; without SDL3 it builds with devices that have no window.

    goose test/compute.goose          a compute shader, checked against the CPU
    goose test/raster.goose           a textured quad, checked pixel by pixel
    goose test/updates.goose          updates to resources in use, in flight
    goose test/rays.goose             inline ray queries, where the device has them
    goose examples/cube.goose         gfx's cube, in a window

`GOOSE_GPU_BACKEND=vulkan` picks Vulkan over Metal on a Mac, where the
driver is KosmicKrisp; `GOOSE_GPU_VALIDATION=1` turns on Vulkan's
validation layer where it is installed (`VK_LAYER_PATH` names the SDK's
layers), with `VK_LAYER_VALIDATE_SYNC=1` for synchronization, and
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` Metal's. Leave Metal's shader
validation off under Vulkan: KosmicKrisp runs on Metal, and its texture
reads come back zero with it on.
`GOOSE_GPU_HEADLESS=1` opens no window.
