// A ray-traced scene in a window: a ring of spinning cubes over a floor,
// lit by the sun with ray-traced shadows and ambient occlusion. Each frame
// rewrites the instances (update_buffer), rebuilds the instance
// acceleration structure, traces every pixel in a compute shader into a
// storage texture, and draws that over the screen with a full-screen
// triangle. Every shader is a Goose function.
//
// Needs a device with ray queries: Metal 4 on Apple silicon. Escape or
// closing the window quits; `-- --frames N` stops after N frames and says
// whether the middle of the screen is lit.
//
//   goose examples/raytrace.goose

import gpu from "..";
import std;
import vec;
import math;
import os;

let CUBES = 8;

// What a pixel's shader reaches: the scene, the mesh every instance shares
// (positions and triangles), each instance's model matrix and color, and
// the camera.
struct Trace {
    scene: gpu::Accel,
    positions: gpu::StructuredBuffer<float3>,
    triangles: gpu::StructuredBuffer<uint3>,
    models: gpu::StructuredBuffer<gpu::mat4>,
    colors: gpu::StructuredBuffer<float4>,
    target: gpu::RWTexture2D<float4>,
    width: u32,
    height: u32,
    frame: u32,
    pad 4,
}

struct Camera { eye: float3, pad 4, forward: float3, pad 4, right: float3, pad 4, up: float3, pad 4 }

// A hash of the pixel and the frame, for the occlusion rays' directions.
fn pcg(x: u32) -> u32 {
    var h = x * 747796405 + 2891336453;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737;
    (h >> 22) ^ h
}

fn unit(h: u32) -> f32 { (h & 16777215) * (1.0 / 16777216.0) }

// The hit's normal in world space, from its triangle's corners. The floor's
// structure numbers its triangles from 0, after the cube's 12 in the mesh.
fn normal_at(p: Trace, h: gpu::Hit) -> float3 {
    let first: u32 = if h.id == CUBES { 12 } else { 0 };
    let tri = gpu::load(p.triangles, h.primitive + first);
    let a = gpu::load(p.positions, tri.x);
    let b = gpu::load(p.positions, tri.y);
    let c = gpu::load(p.positions, tri.z);
    let n = cross(b - a, c - a);
    let m = gpu::load(p.models, h.instance);
    normalize(xyz(gpu::mul(m, float4 { n.x, n.y, n.z, 0.0 })))
}

compute(8, 8) fn trace_pixels(p: Trace, cam: const Camera&) {
    let xy = gpu::dispatch_thread_id();
    if xy.x >= p.width || xy.y >= p.height { return; }
    let aspect: f32 = p.width / (p.height * 1.0);
    let u: f32 = ((xy.x + 0.5) / p.width * 2.0 - 1.0) * aspect;
    let v: f32 = 1.0 - (xy.y + 0.5) / p.height * 2.0;
    let dir = normalize(cam.forward * 1.6 + cam.right * u + cam.up * v);
    let h = gpu::trace(p.scene, gpu::Ray { origin: cam.eye, t_min: 0.001, dir: dir, t_max: 200.0 }, 255);
    var color = float3 { 0.45, 0.6, 0.85 } + float3 { 0.3, 0.25, 0.1 } * (1.0 - max(dir.y, 0.0));
    if h.kind == gpu::HIT_TRIANGLE {
        let at = cam.eye + dir * h.t;
        var n = normal_at(p, h);
        if dot(n, dir) > 0.0 { n = -n; }
        let base = xyz(gpu::load(p.colors, h.instance));
        let sun = normalize(float3 { 0.5, 0.9, 0.3 });
        let origin = at + n * 0.002;
        var light: f32 = max(dot(n, sun), 0.0);
        if light > 0.0 && gpu::trace_any(p.scene, gpu::Ray { origin: origin, t_min: 0.001, dir: sun,
                                                              t_max: 200.0 }, 255) {
            light = 0.0;
        }
        // Ambient occlusion: a few rays into the hemisphere, half a unit long.
        var open: f32 = 0.0;
        var seed = pcg(xy.x * 1973 + xy.y * 9277 + p.frame * 26699);
        let samples: u32 = 6;
        for k in samples {
            seed = pcg(seed);
            let a = unit(seed) * 6.2831853;
            seed = pcg(seed);
            let z = unit(seed);
            let r = sqrt(1.0 - z * z);
            var d = float3 { cos(a) * r, sin(a) * r, z };
            if dot(d, n) < 0.0 { d = -d; }
            if !gpu::trace_any(p.scene, gpu::Ray { origin: origin, t_min: 0.001, dir: d, t_max: 0.5 }, 255) {
                open += 1.0;
            }
        }
        let ambient: f32 = 0.35 * open / samples;
        color = base * (light * 0.9 + ambient);
    }
    // A plain gamma to the screen.
    let c = float3 { sqrt(color.x), sqrt(color.y), sqrt(color.z) };
    gpu::store(p.target, int2 { xy.x as i32, xy.y as i32 }, float4 { c.x, c.y, c.z, 1.0 });
}

// The traced image over the screen: one triangle covering it.
struct Blit { image: gpu::Texture2D<float4> }
struct Pixel { uv: float2 }

vertex fn fullscreen_vs(b: Blit) -> float4, Pixel {
    let i = gpu::vertex_index();
    let uv = float2 { (i << 1) & 2, i & 2 };
    return float4 { uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0 }, Pixel { uv: uv };
}

fragment fn fullscreen_fs(px: Pixel, b: Blit) -> float4 { gpu::sample(b.image, gpu::LINEAR_CLAMP, px.uv) }

// A cube's 12 triangles, (-0.5..0.5)^3, and the floor's two, the last.
fn meshes(positions: float3[>..]&, triangles: uint3[>..]&) -> u32 {
    for z in 2 { for y in 2 { for x in 2 {
        positions.push(float3 { x - 0.5, y - 0.5, z - 0.5 });
    } } }
    let faces: u32[24] = [0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5];
    for f in 6 {
        let q = f * 4;
        triangles.push(uint3 { faces[q], faces[q + 1], faces[q + 2] });
        triangles.push(uint3 { faces[q], faces[q + 2], faces[q + 3] });
    }
    let floor = positions.len as u32;
    positions.append([float3 { -1.0, 0.0, -1.0 }, float3 { 1.0, 0.0, -1.0 },
                      float3 { 1.0, 0.0, 1.0 }, float3 { -1.0, 0.0, 1.0 }]);
    triangles.push(uint3 { floor, floor + 2, floor + 1 });
    triangles.push(uint3 { floor, floor + 3, floor + 2 });
    12
}

fn main() {
    var frames = -1;
    let argv = args();
    if argv.len == 3 && argv[1] == "--frames" {
        let n, ok = parse_int(argv[2]);
        if ok { frames = n; }
    }
    guard gpu::open("Goose: ray tracing", 960, 600, gpu::WINDOW_RESIZABLE) else {
        write_stderr(str("gpu: ", gpu::error(), "\n"));
        return;
    }
    guard gpu::ray_query() else {
        write_stderr(str("gpu: the ", gpu::driver(), " device has no ray queries\n"));
        gpu::close();
        return;
    }
    var positions: float3[>..] = [];
    var triangles: uint3[>..] = [];
    let cube_tris = meshes(positions, triangles);
    let pb = gpu::buffer(gpu::BUFFER_READ, bytes_of(positions));
    let tb = gpu::buffer(gpu::BUFFER_READ, bytes_of(triangles));
    // Two primitive structures over the one mesh: the cube's triangles, and
    // the floor's after them.
    let cube = gpu::primitive_accel(gpu::PrimitiveAccelDesc { vertices: pb, vertex_count: 8,
                                                              indices: tb, triangle_count: cube_tris, .. });
    let floor = gpu::primitive_accel(gpu::PrimitiveAccelDesc { vertices: pb, vertex_count: positions.len,
                                                               indices: tb, index_offset: cube_tris * 12,
                                                               triangle_count: 2, .. });
    gpu::build(cube);
    gpu::build(floor);

    let count = CUBES + 1;
    let instances = gpu::buffer(gpu::BUFFER_READ, count * 72);
    let models = gpu::buffer(gpu::BUFFER_READ, count * 64);
    var colors: float4[>..] = [];
    for i in CUBES {
        let hue: f32 = i * (6.2831853 / CUBES);
        colors.push(float4 { 0.55 + 0.4 * cos(hue), 0.55 + 0.4 * cos(hue - 2.1), 0.55 + 0.4 * cos(hue + 2.1), 1.0 });
    }
    colors.push(float4 { 0.8, 0.8, 0.78, 1.0 });
    let cb = gpu::buffer(gpu::BUFFER_READ, bytes_of(colors));
    let scene = gpu::instance_accel(gpu::InstanceAccelDesc { instances: instances, max: count });

    let tracer = gpu::compute_pipeline(trace_pixels);
    let blit = gpu::pipeline(fullscreen_vs, fullscreen_fs);
    var image = gpu::Texture { 0 };
    var image_size = int2_0;
    var drawn = 0;
    var middle: u8[>..] = [];
    while gpu::frame() {
        if gpu::key_pressed("Escape") { break; }
        let size = gpu::screen_size();
        if size.x == 0 { continue; }
        // The image the tracer writes, at the screen's size.
        if size.x != image_size.x || size.y != image_size.y {
            if image.index != 0 { gpu::release(image); }
            image = gpu::texture(gpu::TextureDesc { format: gpu::RGBA8, usage: gpu::STORAGE | gpu::SAMPLED,
                                                    width: size.x, height: size.y, .. });
            gpu::barrier(image, gpu::ACCESS_NONE, gpu::ACCESS_SHADER_WRITE);
            image_size = size;
        } else {
            gpu::barrier(image, gpu::ACCESS_SHADER_READ, gpu::ACCESS_SHADER_WRITE);
        }
        // The cubes on a ring, turning, and the floor under them.
        let t = gpu::time() as f32;
        var insts: gpu::Instance[>..] = [];
        var mats: gpu::mat4[>..] = [];
        for i in CUBES {
            let a = i * (6.2831853 / CUBES) + t * 0.3;
            let m = gpu::mul(gpu::translate(float3 { cos(a) * 2.2, 0.5 + 0.25 * sin(t * 1.3 + i), sin(a) * 2.2 }),
                             gpu::mul(gpu::rotate(float3 { 0.3, 1.0, 0.2 }, t + i),
                                      gpu::scale(float3 { 0.8, 0.8, 0.8 })));
            insts.push(gpu::instance(m, cube, i as u32));
            mats.push(m);
        }
        let fm = gpu::mul(gpu::translate(float3 { 0.0, -0.3, 0.0 }), gpu::scale(float3 { 6.0, 1.0, 6.0 }));
        insts.push(gpu::instance(fm, floor, CUBES));
        mats.push(fm);
        gpu::update_buffer(instances, bytes_of(insts));
        gpu::update_buffer(models, bytes_of(mats));
        gpu::build(scene, count);

        let eye = float3 { cos(t * 0.1) * 6.0, 2.6, sin(t * 0.1) * 6.0 };
        let forward = normalize(float3_0 - eye);
        let right = normalize(cross(forward, float3 { 0.0, 1.0, 0.0 }));
        let up = cross(right, forward);
        gpu::begin_compute();
        gpu::bind(tracer);
        gpu::uniforms(Camera { eye: eye, forward: forward, right: right, up: up });
        gpu::push_constants(Trace { scene: scene, positions: gpu::structured<float3>(pb),
                                    triangles: gpu::structured<uint3>(tb),
                                    models: gpu::structured<gpu::mat4>(models),
                                    colors: gpu::structured<float4>(cb),
                                    target: gpu::storage<float4>(image), width: size.x as u32,
                                    height: size.y as u32, frame: drawn as u32 });
        gpu::dispatch((size.x + 7) / 8, (size.y + 7) / 8, 1);
        gpu::end_compute();
        gpu::barrier(image, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_SHADER_READ);

        guard gpu::begin_screen(float4 { 0.0, 0.0, 0.0, 1.0 }) else { continue; }
        gpu::bind(blit);
        gpu::push_constants(Blit { gpu::sampled<float4>(image) });
        gpu::draw(3);
        gpu::end_pass();
        drawn++;
        if drawn == frames {
            gpu::barrier(gpu::screen(), gpu::ACCESS_RENDER_TARGET, gpu::ACCESS_COPY_SRC);
            let r = gpu::read_texture(gpu::screen());
            gpu::wait(gpu::submit());
            guard gpu::bytes(r, middle) else { abort("gpu: the read back is not ready"); }
            break;
        }
    }
    if frames > 0 && middle.len > 0 {
        let size = gpu::screen_size();
        let mid = ((size.y as i64 / 2) * size.x + size.x as i64 / 2) * 4;
        var lit = false;
        for i in 3 { lit = lit || middle[mid + i] > 20; }
        print(gpu::driver(), ": traced ", drawn, " frames; middle lit: ", lit);
    }
    gpu::close();
}
