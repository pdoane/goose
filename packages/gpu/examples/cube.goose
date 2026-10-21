// A window with a spinning, lit cube, drawn with the gpu package: gfx's
// cube (samples/27_gfx_cube.goose) with its shaders written as Goose
// functions. The vertex shader pulls its vertices from a buffer by index,
// the transform is a uniform block, and the face colors are four bytes
// packed in a u32.
//
// The arrow keys turn the cube, and Escape or closing the window quits.
// `-- --frames N` stops after N frames and says whether the cube was drawn
// in the middle of the screen. GOOSE_GPU_BACKEND=vulkan draws with Vulkan
// on a Mac; GOOSE_GPU_HEADLESS=1 opens no window.
//
//   goose examples/cube.goose

import gpu from "..";
import std;
import vec;
import math;
import os;

struct Vertex { pos: float3, normal: float3, color: u32 }
struct Draw { vertices: gpu::StructuredBuffer<Vertex> }
struct Transform { model: gpu::mat4, view_proj: gpu::mat4 }
struct Varyings { normal: float3, color: float4 }

vertex fn cube_vs(d: Draw, t: const Transform&) -> float4, Varyings {
    let v = gpu::load(d.vertices, gpu::vertex_index());
    let n = gpu::mul(t.model, float4 { v.normal.x, v.normal.y, v.normal.z, 0.0 });
    let world = gpu::mul(t.model, float4 { v.pos.x, v.pos.y, v.pos.z, 1.0 });
    return gpu::mul(t.view_proj, world), Varyings { normal: xyz(n), color: gpu::unpack_unorm4x8(v.color) };
}

// The face color, lit from one direction, with some light everywhere.
fragment fn cube_fs(f: Varyings, d: Draw, t: const Transform&) -> float4 {
    let light = max(dot(normalize(f.normal), normalize(float3 { 0.4, 0.8, 0.6 })), 0.0);
    let c = f.color * (0.25 + 0.75 * light);
    float4 { c.x, c.y, c.z, 1.0 }
}

// Four corners and two triangles per face, counterclockwise seen from
// outside, which is what culling back faces keeps.
fn cube(verts: Vertex[>..]&, indices: u16[>..]&) {
    let axes: float3[3] = [float3 { 1.0, 0.0, 0.0 }, float3 { 0.0, 1.0, 0.0 },
                           float3 { 0.0, 0.0, 1.0 }];
    let colors: u32[6] = [0xff3c50e6, 0xff32aaf0, 0xff5ac85a, 0xffe6a046, 0xffdc64a0, 0xffe6e6e6];
    for f in 6 {
        let sign: f32 = if f % 2 == 0 { 1.0 } else { -1.0 };
        let n = axes[f / 2] * sign;
        let u = axes[(f / 2 + 1) % 3] * sign;
        let v = cross(n, u);
        let base = verts.len as u16;
        let corners: float2[4] = [float2 { -1.0, -1.0 }, float2 { 1.0, -1.0 },
                                  float2 { 1.0, 1.0 }, float2 { -1.0, 1.0 }];
        for c in corners {
            let p = n + u * c.x + v * c.y;
            verts.push(Vertex { pos: p * 0.5, normal: n, color: colors[f] });
        }
        indices.append([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn main() {
    var frames = -1;
    let argv = args();
    if argv.len == 3 && argv[1] == "--frames" {
        let n, ok = parse_int(argv[2]);
        if ok { frames = n; }
    }
    guard gpu::open("Goose: a gpu cube", 640, 480, gpu::WINDOW_RESIZABLE) else {
        write_stderr(str("gpu: ", gpu::error(), "\n"));
        return;
    }
    var verts: Vertex[>..] = [];
    var indices: u16[>..] = [];
    cube(verts, indices);
    let vb = gpu::buffer(gpu::BUFFER_READ, bytes_of(verts));
    let ib = gpu::index_buffer(bytes_of(indices));
    let pipeline = gpu::pipeline(gpu::PipelineDesc { cull: gpu::CULL_BACK, depth_test: true,
                                                     depth_write: true, .. }, cube_vs, cube_fs);

    var yaw: f32 = 0.6;
    var pitch: f32 = 0.5;
    var drawn = 0;
    var middle: u8[>..] = [];
    while gpu::frame() {
        if gpu::key_pressed("Escape") { break; }
        if gpu::key_down("Left") { yaw -= 0.04; }
        if gpu::key_down("Right") { yaw += 0.04; }
        if gpu::key_down("Up") { pitch -= 0.04; }
        if gpu::key_down("Down") { pitch += 0.04; }
        yaw += 0.01;

        let size = gpu::screen_size();
        let aspect = size.x as f32 / size.y;
        let model = gpu::mul(gpu::rotate(float3 { 1.0, 0.0, 0.0 }, pitch),
                             gpu::rotate(float3 { 0.0, 1.0, 0.0 }, yaw));
        let view = gpu::look_at(float3 { 0.0, 0.4, 3.0 }, float3_0, float3 { 0.0, 1.0, 0.0 });
        let projection = gpu::perspective(0.8, aspect, 0.1, 20.0);

        guard gpu::begin_screen(float4 { 0.1, 0.12, 0.16, 1.0 }) else { continue; }
        gpu::bind(pipeline);
        gpu::push_constants(Draw { gpu::structured<Vertex>(vb) });
        gpu::uniforms(Transform { model: model, view_proj: gpu::mul(projection, view) });
        gpu::bind_index_buffer(ib, 2);
        gpu::draw_indexed(indices.len);
        gpu::end_pass();
        drawn++;
        if drawn == frames {
            // The last frame: read back, to see the cube in its middle.
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
        // BGRA: the background is 41, 31, 26 in that order.
        let background: u8[3] = [41, 31, 26];
        var cube_there = false;
        for i in 3 { cube_there = cube_there || abs(middle[mid + i] as i64 - background[i]) > 8; }
        print(gpu::driver(), ": drew ", drawn, " frames; cube in the middle: ", cube_there);
    }
    gpu::close();
}
