// Image-based lighting, computed on the GPU: an environment as an
// equirectangular map (a procedural sky, or a Radiance .hdr file's
// pixels), its irradiance for diffuse light, its radiance prefiltered by
// GGX lobes of growing roughness, a mip each, for specular light, and the
// split-sum table of how much of that a surface reflects (Karis, "Real
// Shading in Unreal Engine 4"). Each map's integral samples the
// environment's mips by the solid angle a sample stands for, which keeps
// a few hundred samples free of noise.

namespace ibl;

import gpu from "../..";
import .pbr;
import std;
import os;
import math;

let ENV_W: i64 = 1024;
let ENV_H: i64 = 512;
let IRR_W: i64 = 64;
let IRR_H: i64 = 32;
let SPEC_W: i64 = 512;
let SPEC_H: i64 = 256;
let SPEC_MIPS: i64 = 6;
let LUT: i64 = 128;

struct Maps {
    env: gpu::Texture,
    irradiance: gpu::Texture,
    specular: gpu::Texture,
    lut: gpu::Texture,
}

struct Sky { dst: gpu::RWTexture2D<float4>, sun: float3, width: u32, height: u32 }

compute(8, 8) fn sky_kernel(p: Sky) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let d = pbr::equirect_dir(pbr::texel_uv(id, p.width, p.height));
    let c = pbr::sky(d, p.sun);
    gpu::store(p.dst, int2 { id.x as i32, id.y as i32 }, float4 { c.x, c.y, c.z, 1.0 });
}

// A Radiance file's pixels, RGBE packed in u32s, as radiance.
struct Rgbe { src: gpu::StructuredBuffer<u32>, dst: gpu::RWTexture2D<float4>, width: u32, height: u32 }

compute(8, 8) fn rgbe_kernel(p: Rgbe) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let v = gpu::load(p.src, id.y * p.width + id.x);
    let e = (v >> 24) & 255;
    var c = float3_0;
    if e != 0 {
        let two: f32 = 2.0;
        let f = pow(two, e as f32 - 136.0);
        c = float3 { (v & 255) * f, ((v >> 8) & 255) * f, ((v >> 16) & 255) * f };
    }
    gpu::store(p.dst, int2 { id.x as i32, id.y as i32 }, float4 { c.x, c.y, c.z, 1.0 });
}

struct Integrate {
    env: gpu::Texture2D<float4>,
    dst: gpu::RWTexture2D<float4>,
    width: u32,
    height: u32,
    roughness: f32,
    samples: u32,
}

// The mip of the environment whose texels stand for the solid angle a
// sample of density `pdf` does, of `n`.
fn env_lod(pdf: f32, n: u32) -> f32 {
    let texel = 4.0 * pbr::PI / (ENV_W * ENV_H);
    let sample = 1.0 / (n * pdf + 0.0001);
    max(0.5 * log2(sample / texel) + 1.0, 0.0)
}

fn env_at(env: gpu::Texture2D<float4>, d: float3, lod: f32) -> float3 {
    xyz(gpu::sample_level(env, gpu::LINEAR_REPEAT, pbr::equirect_uv(d), lod))
}

// Cosine-weighted radiance about each texel's direction: irradiance over
// pi, what a white Lambertian surface facing that way reflects.
compute(8, 8) fn irradiance_kernel(p: Integrate) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let n = pbr::equirect_dir(pbr::texel_uv(id, p.width, p.height));
    var sum = float3_0;
    for i: u32 in p.samples {
        let xi = pbr::hammersley(i, p.samples);
        let l = pbr::cosine_dir(xi, n);
        let pdf = max(dot(n, l), 0.0) / pbr::PI;
        sum += env_at(p.env, l, env_lod(pdf, p.samples));
    }
    let c = sum / (p.samples as f32);
    gpu::store(p.dst, int2 { id.x as i32, id.y as i32 }, float4 { c.x, c.y, c.z, 1.0 });
}

// Radiance through a GGX lobe of the mip's roughness about each texel's
// direction, seen from straight on (the split sum's assumption).
compute(8, 8) fn specular_kernel(p: Integrate) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let n = pbr::equirect_dir(pbr::texel_uv(id, p.width, p.height));
    var c = float3_0;
    if p.roughness == 0.0 {
        c = env_at(p.env, n, 0.0);
    } else {
        let alpha = p.roughness * p.roughness;
        var sum = float3_0;
        var weight: f32 = 0.0;
        for i: u32 in p.samples {
            let h = pbr::ggx_half(pbr::hammersley(i, p.samples), n, alpha);
            let l = h * (2.0 * dot(n, h)) - n;
            let nl = dot(n, l);
            if nl > 0.0 {
                let nh = max(dot(n, h), 0.0);
                let pdf = pbr::ggx_d(nh, alpha) / 4.0;
                sum += env_at(p.env, l, env_lod(pdf, p.samples)) * nl;
                weight += nl;
            }
        }
        c = sum / max(weight, 0.0001);
    }
    gpu::store(p.dst, int2 { id.x as i32, id.y as i32 }, float4 { c.x, c.y, c.z, 1.0 });
}

// The scale and bias on F0 of the light a GGX surface reflects, by N.V
// across and roughness down.
compute(8, 8) fn lut_kernel(p: Integrate) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let uv = pbr::texel_uv(id, p.width, p.height);
    let nv = max(uv.x, 0.001);
    let rough = uv.y;
    let alpha = rough * rough;
    let v = float3 { sqrt(1.0 - nv * nv), 0.0, nv };
    let n = float3 { 0.0, 0.0, 1.0 };
    var a: f32 = 0.0;
    var b: f32 = 0.0;
    for i: u32 in p.samples {
        let h = pbr::ggx_half(pbr::hammersley(i, p.samples), n, alpha);
        let l = h * (2.0 * dot(v, h)) - v;
        let nl = max(l.z, 0.0);
        let nh = max(h.z, 0.0);
        let vh = max(dot(v, h), 0.0);
        if nl > 0.0 {
            // V's GGX visibility times the sampling's weight, 4 V.H / N.H.
            let g = pbr::smith_v(nl, nv, alpha) * 4.0 * nl * vh / max(nh, 0.0001);
            let five: f32 = 5.0;
            let fc = pow(1.0 - vh, five);
            a += (1.0 - fc) * g;
            b += fc * g;
        }
    }
    gpu::store(p.dst, int2 { id.x as i32, id.y as i32 },
               float4 { a / p.samples, b / p.samples, 0.0, 1.0 });
}

// A Radiance .hdr file's pixels, RGBE, and its size; "" or why not. The
// new run-length scan lines, and flat ones.
fn read_hdr(path: const u8[:], px: u32[>..]&, w: i64&, h: i64&) -> u8[>..] {
    var d: u8[>..] = [];
    guard read_file(path, d) else { return str("cannot read ", path); }
    // The header ends at an empty line; the size line follows.
    var p: i64 = 0;
    while p + 1 < d.len && !(d[p] == '\n' && d[p + 1] == '\n') { p++; }
    p += 2;
    var line: u8[>..] = [];
    while p < d.len && d[p] != '\n' { line.push(d[p]); p++; }
    p++;
    // "-Y h +X w"
    var nums: i64[2] = [0, 0];
    var k = 0;
    var i: i64 = 0;
    while i < line.len && k < 2 {
        if line[i] >= '0' && line[i] <= '9' {
            var v: i64 = 0;
            while i < line.len && line[i] >= '0' && line[i] <= '9' { v = v * 10 + (line[i] - '0'); i++; }
            nums[k] = v;
            k++;
        } else {
            i++;
        }
    }
    h = nums[0];
    w = nums[1];
    guard w > 0 && h > 0 else { return str(path, ": no image size"); }
    var row: u8[>..] = [];
    row.resize(w * 4, 0);
    for y in h {
        if p + 4 <= d.len && d[p] == 2 && d[p + 1] == 2 && (d[p + 2] as i64 << 8 | d[p + 3]) == w {
            // Each of the four channels as runs and literals.
            p += 4;
            for c in 4 {
                var x: i64 = 0;
                while x < w && p < d.len {
                    var n: i64 = d[p];
                    p++;
                    if n > 128 {
                        n -= 128;
                        let v = d[p];
                        p++;
                        for j in n { if x + j < w { row[(x + j) * 4 + c] = v; } }
                    } else {
                        for j in n { if x + j < w && p < d.len { row[(x + j) * 4 + c] = d[p]; } p++; }
                    }
                    x += n;
                }
            }
        } else {
            for x in w * 4 { row[x] = if p < d.len { d[p] } else { 0 }; p++; }
        }
        for x in w {
            px.push(row[x * 4] | (row[x * 4 + 1] as u32 << 8) | (row[x * 4 + 2] as u32 << 16) |
                    (row[x * 4 + 3] as u32 << 24));
        }
    }
    str("")
}

fn texture(w: i64, h: i64, mips: i32) -> gpu::Texture {
    gpu::texture(gpu::TextureDesc { format: gpu::RGBA16F, usage: gpu::SAMPLED | gpu::STORAGE,
                                    width: w as i32, height: h as i32, mips: mips, .. })
}

fn dispatch<F>(p: Integrate, w: i64, h: i64) {
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(F));
    gpu::push_constants(p);
    gpu::dispatch((w + 7) / 8, (h + 7) / 8, 1);
    gpu::end_compute();
}

// The maps, from the .hdr file at `hdr` or, with "", the sky lit by a sun
// toward `sun`; recorded, and complete once submitted.
fn make(hdr: const u8[:], sun: float3, err: u8[>..]&) -> Maps {
    var w = ENV_W;
    var h = ENV_H;
    var px: u32[>..] = [];
    if hdr.len > 0 {
        let e = read_hdr(hdr, px, w, h);
        if e.len > 0 { err.append(e); }
    }
    let env = texture(w, h, 0);
    gpu::barrier(env, gpu::ACCESS_NONE, gpu::ACCESS_SHADER_WRITE);
    gpu::begin_compute();
    if px.len > 0 {
        let src = gpu::buffer(gpu::BUFFER_READ, bytes_of(px));
        gpu::bind(gpu::compute_pipeline(rgbe_kernel));
        gpu::push_constants(Rgbe { src: gpu::structured<u32>(src), dst: gpu::storage<float4>(env),
                                   width: w as u32, height: h as u32 });
        gpu::dispatch((w + 7) / 8, (h + 7) / 8, 1);
        gpu::release(src);
    } else {
        gpu::bind(gpu::compute_pipeline(sky_kernel));
        gpu::push_constants(Sky { dst: gpu::storage<float4>(env), sun: normalize(sun), width: w as u32,
                                  height: h as u32 });
        gpu::dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
    gpu::end_compute();
    gpu::barrier(env, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_SHADER_READ);
    gpu::generate_mips(env);

    let irr = texture(IRR_W, IRR_H, 1);
    let spec = texture(SPEC_W, SPEC_H, SPEC_MIPS as i32);
    let lut = texture(LUT, LUT, 1);
    for t in [irr, spec, lut] { gpu::barrier(t, gpu::ACCESS_NONE, gpu::ACCESS_SHADER_WRITE); }
    let src = gpu::sampled<float4>(env);
    let pi = Integrate { env: src, dst: gpu::storage<float4>(irr), width: IRR_W as u32, height: IRR_H as u32,
                         roughness: 0.0, samples: 512 };
    dispatch(pi, IRR_W, IRR_H, irradiance_kernel);
    for m in SPEC_MIPS {
        let mw = SPEC_W >> m;
        let mh = SPEC_H >> m;
        let ps = Integrate { env: src, dst: gpu::storage_mip<float4>(spec, m), width: mw as u32,
                             height: mh as u32, roughness: m / (SPEC_MIPS - 1) as f32, samples: 256 };
        dispatch(ps, mw, mh, specular_kernel);
    }
    let pl = Integrate { env: src, dst: gpu::storage<float4>(lut), width: LUT as u32, height: LUT as u32,
                         roughness: 0.0, samples: 512 };
    dispatch(pl, LUT, LUT, lut_kernel);
    for t in [irr, spec, lut] { gpu::barrier(t, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_SHADER_READ); }
    Maps { env: env, irradiance: irr, specular: spec, lut: lut }
}
