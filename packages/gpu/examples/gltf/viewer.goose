// A glTF viewer: Khronos's metallic-roughness materials, rasterized and
// lit by the environment (ibl.goose) and a sun. Vertices are pulled from
// one buffer by index, every material's textures are handles in a table,
// and each draw pushes no more than which instance and material it is.
//
// Drag with the mouse, or use the arrow keys, to turn around the model;
// W and S move in and out; P saves a screenshot; Escape quits.
//
//   goose viewer.goose -- [model.glb] [--hdr sky.hdr] [--yaw r --pitch r]
//                          [--frames N --out shot.png]
//
// Without a model it shows DamagedHelmet from ~/.goose/assets, which
//   curl -L -o ~/.goose/assets/DamagedHelmet.glb https://raw.githubusercontent.com/
//       KhronosGroup/glTF-Sample-Assets/main/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb
// fetches. GOOSE_GPU_HEADLESS=1 opens no window.

import gpu from "../..";
import .gltf;
import .scene;
import .pbr;
import .ibl;
import std;
import os;
import math;

struct Push { refs: scene::Refs, instance: u32, material: u32 }

struct Frame {
    view_proj: gpu::mat4,
    camera: float3,
    exposure: f32,
    sun: float3,
    sun_strength: f32,
    sun_color: float3,
    spec_mips: f32,
    irradiance: gpu::Texture2D<float4>,
    specular: gpu::Texture2D<float4>,
    lut: gpu::Texture2D<float4>,
    env: gpu::Texture2D<float4>,
}

struct Varyings { pos: float3, normal: float3, tangent: float4, uv: float2 }

vertex fn mesh_vs(p: Push, f: const Frame&) -> float4, Varyings {
    let v = gpu::load(p.refs.verts, gpu::vertex_index());
    let inst = gpu::load(p.refs.instances, p.instance);
    let world = gpu::mul(inst.world, float4 { v.pos.x, v.pos.y, v.pos.z, 1.0 });
    let n = xyz(gpu::mul(inst.normal, float4 { v.normal.x, v.normal.y, v.normal.z, 0.0 }));
    let t = xyz(gpu::mul(inst.world, float4 { v.tangent.x, v.tangent.y, v.tangent.z, 0.0 }));
    return gpu::mul(f.view_proj, world),
           Varyings { pos: xyz(world), normal: n, tangent: float4 { t.x, t.y, t.z, v.tangent.w }, uv: v.uv };
}

// The shading normal: the geometric one turned toward the viewer for a
// back face, bent by the normal map in the tangent frame the file gives
// or, without one, the frame screen-space derivatives of position and
// texture coordinates make (Schüler, "Normal Mapping without Precomputed
// Tangents").
fn shading_normal(v: Varyings, m: gltf::Material, flags: u32) -> float3 {
    var n = normalize(v.normal);
    if !gpu::front_facing() { n = n * -1.0; }
    let s = xyz(gpu::sample(m.normal_tex, gpu::LINEAR_REPEAT, v.uv)) * 2.0 - float3_1;
    let tn = normalize(float3 { s.x * m.normal_scale, s.y * m.normal_scale, s.z });
    var t = float3_0;
    var b = float3_0;
    if flags & gltf::HAS_TANGENTS != 0 {
        let vt = xyz(v.tangent);
        t = normalize(vt - n * dot(n, vt));
        b = cross(n, t) * v.tangent.w;
    } else {
        let dp1 = gpu::ddx(v.pos);
        let dp2 = gpu::ddy(v.pos);
        let duv1 = gpu::ddx(v.uv);
        let duv2 = gpu::ddy(v.uv);
        let dp2perp = cross(dp2, n);
        let dp1perp = cross(n, dp1);
        t = dp2perp * duv1.x + dp1perp * duv2.x;
        b = dp2perp * duv1.y + dp1perp * duv2.y;
        let inv = 1.0 / sqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
        t = t * inv;
        b = b * inv;
        // uv's v runs down the image, glTF's tangent space up it.
        b = b * -1.0;
    }
    normalize(t * tn.x + b * tn.y + n * tn.z)
}

fn env_lookup(t: gpu::Texture2D<float4>, d: float3, lod: f32) -> float3 {
    xyz(gpu::sample_level(t, gpu::LINEAR_REPEAT, pbr::equirect_uv(d), lod))
}

fragment fn mesh_fs(v: Varyings, p: Push, f: const Frame&) -> float4 {
    let m = gpu::load(p.refs.materials, p.material);
    let base = m.base * gpu::sample(m.base_tex, gpu::LINEAR_REPEAT, v.uv);
    if m.flags & gltf::ALPHA_MASK != 0 && base.w < m.alpha_cutoff { gpu::discard(); }
    let mr = gpu::sample(m.mr_tex, gpu::LINEAR_REPEAT, v.uv);
    let s = pbr::Surface { base: xyz(base), metallic: clamp(m.metallic * mr.z, 0.0, 1.0),
                           roughness: clamp(m.roughness * mr.y, 0.04, 1.0) };
    let prim = gpu::load(p.refs.prims, gpu::load(p.refs.instances, p.instance).prim);
    let n = shading_normal(v, m, prim.flags);
    let view = normalize(f.camera - v.pos);
    let nv = max(dot(n, view), 0.0001);

    // The sun, then the environment's two halves of the split sum.
    var c = pbr::brdf(s, n, view, f.sun) * f.sun_color * f.sun_strength;
    let f0 = lerp(float3 { 0.04, 0.04, 0.04 }, s.base, s.metallic);
    let ab = xyz(gpu::sample_level(f.lut, gpu::LINEAR_CLAMP, float2 { nv, s.roughness }, 0.0));
    let r = n * (2.0 * dot(n, view)) - view;
    let spec = env_lookup(f.specular, r, s.roughness * (f.spec_mips - 1.0)) * (f0 * ab.x + float3_1 * ab.y);
    let diffuse = env_lookup(f.irradiance, n, 0.0) * s.base * (1.0 - s.metallic);
    let ao = 1.0 + m.occlusion * (gpu::sample(m.occlusion_tex, gpu::LINEAR_REPEAT, v.uv).x - 1.0);
    c = c + (diffuse + spec) * ao;
    c = c + m.emissive * xyz(gpu::sample(m.emissive_tex, gpu::LINEAR_REPEAT, v.uv));
    let out = pbr::display(c * f.exposure);
    float4 { out.x, out.y, out.z, base.w }
}

// The environment behind the model: a triangle over the screen at the far
// plane, each pixel the direction through it from the camera's basis.
struct Sky { right: float3, tan_x: f32, up: float3, tan_y: f32, forward: float3, exposure: f32,
             env: gpu::Texture2D<float4> }
struct SkyVaryings { ndc: float2 }

vertex fn sky_vs(s: Sky) -> float4, SkyVaryings {
    let i = gpu::vertex_index();
    let xy = float2 { if i == 1 { 3.0 } else { -1.0 }, if i == 2 { 3.0 } else { -1.0 } };
    return float4 { xy.x, xy.y, 1.0, 1.0 }, SkyVaryings { ndc: xy };
}

fragment fn sky_fs(v: SkyVaryings, s: Sky) -> float4 {
    let d = normalize(s.forward + s.right * (v.ndc.x * s.tan_x) + s.up * (v.ndc.y * s.tan_y));
    pbr::display(env_lookup(s.env, d, 0.0) * s.exposure)
}

fn main() {
    let argv = args();
    var home: u8[>..] = [];
    env("HOME", home);
    var model = str(home, "/.goose/assets/DamagedHelmet.glb");
    var hdr: u8[>..] = [];
    var out: u8[>..] = [];
    var frames: i64 = -1;
    var yaw: f32 = 0.6;
    var pitch: f32 = 0.15;
    var i: i64 = 1;
    while i < argv.len {
        if argv[i] == "--hdr" && i + 1 < argv.len { hdr = str(argv[i + 1]); i += 2; continue; }
        if argv[i] == "--out" && i + 1 < argv.len { out = str(argv[i + 1]); i += 2; continue; }
        if (argv[i] == "--yaw" || argv[i] == "--pitch") && i + 1 < argv.len {
            let v, ok = parse_flt(argv[i + 1]);
            if ok && argv[i] == "--yaw" { yaw = v as f32; }
            if ok && argv[i] == "--pitch" { pitch = v as f32; }
            i += 2;
            continue;
        }
        if argv[i] == "--frames" && i + 1 < argv.len {
            let n, ok = parse_int(argv[i + 1]);
            if ok { frames = n; }
            i += 2;
            continue;
        }
        model = str(argv[i]);
        i++;
    }
    guard gpu::open("Goose: glTF", 1280, 800, gpu::WINDOW_RESIZABLE) else {
        write_stderr(str("gpu: ", gpu::error(), "\n"));
        return;
    }
    let t0 = clock();
    var m = scene::Model { refs: scene::Refs { verts: gpu::StructuredBuffer<gltf::Vertex> { 0 },
                                               indices: gpu::StructuredBuffer<u32> { 0 },
                                               materials: gpu::StructuredBuffer<gltf::Material> { 0 },
                                               prims: gpu::StructuredBuffer<gltf::Prim> { 0 },
                                               instances: gpu::StructuredBuffer<gltf::Instance> { 0 } },
                           vertex_buffer: gpu::Buffer { 0 }, index_buffer: gpu::Buffer { 0 }, index_count: 0,
                           triangles: 0, lo: float3_0, hi: float3_0, draws: [] };
    let e = scene::load(model, m);
    guard e.len == 0 else {
        write_stderr(str(e, "\n"));
        gpu::close();
        return;
    }
    let sun = normalize(float3 { 0.5, 0.8, 0.35 });
    var ierr: u8[>..] = [];
    let maps = ibl::make(hdr, sun, ierr);
    if ierr.len > 0 { write_stderr(str(ierr, "\n")); }
    gpu::wait(gpu::submit());
    print("loaded ", model, ": ", m.triangles, " triangles, ", m.draws.len, " draws, in ",
          (clock() - t0) * 1000.0, " ms");

    var pipes: gpu::Pipeline[4] = [gpu::Pipeline { 0 }; 4];
    for k in 4 {
        let blend = k >= 2;
        let desc = gpu::PipelineDesc { cull: if k % 2 == 0 { gpu::CULL_BACK } else { gpu::CULL_NONE },
                                       depth_test: true, depth_write: !blend,
                                       blend: if blend { gpu::BLEND_ALPHA } else { gpu::BLEND_NONE }, .. };
        pipes[k] = gpu::pipeline(desc, mesh_vs, mesh_fs);
    }
    let skydesc = gpu::PipelineDesc { depth_test: true, depth_write: false,
                                      depth_compare: gpu::COMPARE_LESS_EQUAL, .. };
    let skyp = gpu::pipeline(skydesc, sky_vs, sky_fs);

    let center = (m.lo + m.hi) * 0.5;
    let radius = length(m.hi - m.lo) * 0.5;
    var dist = radius * 2.6;
    var last = gpu::mouse_position();
    var shots = 0;
    var n: i64 = 0;
    while gpu::frame() {
        if gpu::key_down("Escape") { gpu::quit(); }
        let dt: f32 = min(gpu::delta_time(), 0.1) as f32;
        let mouse = gpu::mouse_position();
        if gpu::mouse_down(1) {
            yaw += (mouse.x - last.x) * 0.006;
            pitch += (mouse.y - last.y) * 0.006;
        }
        last = mouse;
        if gpu::key_down("Left") { yaw -= dt * 1.5; }
        if gpu::key_down("Right") { yaw += dt * 1.5; }
        if gpu::key_down("Up") { pitch += dt; }
        if gpu::key_down("Down") { pitch -= dt; }
        if gpu::key_down("W") { dist = max(dist * (1.0 - dt), radius * 0.3); }
        if gpu::key_down("S") { dist = dist * (1.0 + dt); }
        pitch = clamp(pitch, -1.5, 1.5);

        let size = gpu::screen_size();
        let eye = center + float3 { cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw) } * dist;
        let view = gpu::look_at(eye, center, float3 { 0.0, 1.0, 0.0 });
        let fov: f32 = 0.8;
        let width: f32 = size.x;
        let aspect = width / max(size.y, 1);
        let proj = gpu::perspective(fov, aspect, radius * 0.01, radius * 20.0);
        let vp = gpu::mul(proj, view);
        let frame = Frame { view_proj: vp, camera: eye, exposure: 1.0, sun: sun, sun_strength: 3.0,
                            sun_color: float3 { 1.0, 0.95, 0.85 }, spec_mips: ibl::SPEC_MIPS,
                            irradiance: gpu::sampled<float4>(maps.irradiance),
                            specular: gpu::sampled<float4>(maps.specular), lut: gpu::sampled<float4>(maps.lut),
                            env: gpu::sampled<float4>(maps.env) };

        // What is blended, farthest first, after what is not.
        var order: i64[>..] = [];
        for d in m.draws.len { if m.draws[d].flags & gltf::ALPHA_BLEND == 0 { order.push(d); } }
        var blended: i64[>..] = [];
        for d in m.draws.len { if m.draws[d].flags & gltf::ALPHA_BLEND != 0 { blended.push(d); } }
        sort(blended) { a, b => length(m.draws[a].center - eye) > length(m.draws[b].center - eye) };
        order.append(blended);

        guard gpu::begin_screen(float4 { 0.0, 0.0, 0.0, 1.0 }) else { continue; }
        gpu::uniforms(frame);
        gpu::bind_index_buffer(m.index_buffer, 4);
        for d in order {
            let dr = m.draws[d];
            let blend = dr.flags & gltf::ALPHA_BLEND != 0;
            let sided = dr.flags & gltf::DOUBLE_SIDED != 0;
            gpu::bind(pipes[(if blend { 2 } else { 0 }) + (if sided { 1 } else { 0 })]);
            gpu::push_constants(Push { refs: m.refs, instance: dr.instance, material: dr.material });
            gpu::draw_indexed(dr.index_count, 1, dr.first_index, 0, 0);
        }
        gpu::bind(skyp);
        let fwd = normalize(center - eye);
        let right = normalize(cross(fwd, float3 { 0.0, 1.0, 0.0 }));
        let th = tan(fov * 0.5);
        gpu::push_constants(Sky { right: right, tan_x: th * aspect, up: cross(right, fwd), tan_y: th,
                                  forward: fwd, exposure: 1.0, env: gpu::sampled<float4>(maps.env) });
        gpu::draw(3);
        gpu::end_pass();

        n++;
        if gpu::key_pressed("P") {
            let path = str("gltf_", shots, ".png");
            gpu::screenshot(path);
            print("saved ", path);
            shots++;
        }
        if frames >= 0 && n >= frames {
            if out.len > 0 { gpu::screenshot(out); }
            gpu::quit();
        }
    }
    gpu::close();
}
