// Physically based shading as glTF's metallic-roughness model defines it
// (the glTF 2.0 specification, appendix B), and what image-based lighting
// needs around it: directions to and from an equirectangular map,
// low-discrepancy samples, GGX importance sampling, a procedural sky, and
// tone mapping. GPU functions, which the CPU can run as well.

namespace pbr;

import gpu_shader;
import vec;
import math;
import std;

let PI: f32 = 3.14159265;

// A direction's place in an equirectangular map, u around from -x through
// +z, v down from +y; and back.
fn equirect_uv(d: float3) -> float2 {
    float2 { atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI }
}

// The middle of texel `id` of a w x h map, from 0 to 1.
fn texel_uv(id: uint3, w: u32, h: u32) -> float2 {
    let half: f32 = 0.5;
    float2 { (id.x + half) / w, (id.y + half) / h }
}

fn equirect_dir(uv: float2) -> float3 {
    let phi = (uv.x - 0.5) * 2.0 * PI;
    let theta = uv.y * PI;
    float3 { sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi) }
}

// The i-th of n points of the Hammersley set.
fn hammersley(i: u32, n: u32) -> float2 {
    var b = i;
    b = (b << 16) | (b >> 16);
    b = ((b & 0x55555555) << 1) | ((b & 0xaaaaaaaa) >> 1);
    b = ((b & 0x33333333) << 2) | ((b & 0xcccccccc) >> 2);
    b = ((b & 0x0f0f0f0f) << 4) | ((b & 0xf0f0f0f0) >> 4);
    b = ((b & 0x00ff00ff) << 8) | ((b & 0xff00ff00) >> 8);
    let scale: f32 = 2.3283064e-10;
    float2 { i / (n as f32), b * scale }
}

// Two axes that make a right-handed basis with n.
struct Basis { t: float3, b: float3 }

fn basis(n: float3) -> Basis {
    let up = if abs(n.y) < 0.999 { float3 { 0.0, 1.0, 0.0 } } else { float3 { 1.0, 0.0, 0.0 } };
    let t = normalize(cross(up, n));
    Basis { t: t, b: cross(n, t) }
}

fn to_world(v: float3, n: float3) -> float3 {
    let bs = basis(n);
    bs.t * v.x + bs.b * v.y + n * v.z
}

// A half vector from GGX's distribution of roughness `alpha` (squared
// perceptual roughness) about n.
fn ggx_half(xi: float2, n: float3, alpha: f32) -> float3 {
    let phi = 2.0 * PI * xi.x;
    let cos_t = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
    let sin_t = sqrt(1.0 - cos_t * cos_t);
    to_world(float3 { sin_t * cos(phi), sin_t * sin(phi), cos_t }, n)
}

// A cosine-weighted direction about n.
fn cosine_dir(xi: float2, n: float3) -> float3 {
    let phi = 2.0 * PI * xi.x;
    let r = sqrt(xi.y);
    to_world(float3 { r * cos(phi), r * sin(phi), sqrt(max(1.0 - xi.y, 0.0)) }, n)
}

// GGX's normal distribution, and Smith's height-correlated visibility
// (the spec's V, G over 4 N.L N.V), at alpha.
fn ggx_d(nh: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = nh * nh * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

fn smith_v(nl: f32, nv: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    let gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    let s = gv + gl;
    if s > 0.0 { 0.5 / s } else { 0.0 }
}

fn schlick(f0: float3, vh: f32) -> float3 {
    let five: f32 = 5.0;
    let k = pow(clamp(1.0 - vh, 0.0, 1.0), five);
    f0 + (float3_1 - f0) * k
}

// A surface's response to light from l, seen from v: the spec's mix of a
// dielectric's diffuse and specular and a metal's specular, times N.L.
struct Surface { base: float3, metallic: f32, roughness: f32 }

fn brdf(s: Surface, n: float3, v: float3, l: float3) -> float3 {
    let nl = dot(n, l);
    let nv = dot(n, v);
    if nl <= 0.0 || nv <= 0.0 { return float3_0; }
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0);
    let alpha = max(s.roughness * s.roughness, 0.002);
    let f0 = lerp(float3 { 0.04, 0.04, 0.04 }, s.base, s.metallic);
    let f = schlick(f0, vh);
    let spec = f * (ggx_d(nh, alpha) * smith_v(nl, nv, alpha));
    let diffuse = (float3_1 - f) * s.base * ((1.0 - s.metallic) / PI);
    (diffuse + spec) * nl
}

// The sky: blue overhead, paler at the horizon, warm ground below, and a
// glow around the sun, in linear radiance.
fn sky(d: float3, sun: float3) -> float3 {
    let up = d.y;
    var c = if up > 0.0 {
        let k: f32 = 0.6;
        lerp(float3 { 0.85, 0.9, 1.0 }, float3 { 0.25, 0.45, 0.9 }, pow(up, k))
    } else {
        lerp(float3 { 0.6, 0.55, 0.5 }, float3 { 0.18, 0.15, 0.12 }, min(-up * 3.0, 1.0))
    };
    let s = max(dot(d, sun), 0.0);
    let tight: f32 = 64.0;
    let wide: f32 = 8.0;
    c = c + float3 { 1.0, 0.85, 0.6 } * (pow(s, tight) * 4.0 + pow(s, wide) * 0.3);
    c * 1.2
}

// The colour shown for a radiance: Khronos's PBR Neutral tone mapper, then
// sRGB's encoding.
fn neutral(c: float3) -> float3 {
    let start: f32 = 0.8 - 0.04;
    let desat: f32 = 0.15;
    let x = min(c.x, min(c.y, c.z));
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    var col = c - float3 { offset, offset, offset };
    let peak = max(col.x, max(col.y, col.z));
    if peak < start { return col; }
    let d = 1.0 - start;
    let np = 1.0 - d * d / (peak + d - start);
    col = col * (np / peak);
    let g = 1.0 - 1.0 / (desat * (peak - np) + 1.0);
    lerp(col, float3 { np, np, np }, g)
}

fn srgb1(x: f32) -> f32 {
    let c = clamp(x, 0.0, 1.0);
    let inv: f32 = 1.0 / 2.4;
    if c <= 0.0031308 { c * 12.92 } else { 1.055 * pow(c, inv) - 0.055 }
}

fn display(c: float3) -> float4 {
    let t = neutral(c);
    float4 { srgb1(t.x), srgb1(t.y), srgb1(t.z), 1.0 }
}
