// A glTF path tracer: the viewer's scene and materials (scene.goose,
// pbr.goose), lit by the same sky and sun, traced with inline ray queries
// against an acceleration structure of the model's instances. A sample a
// pixel a frame, averaged while the camera stays still.
//
// Each bounce reads the triangle a ray hit through the scene's buffers
// (the hit's instance names it; its primitive, material and vertices
// follow by index) and its material's textures through the bindless
// table. The sun is sampled directly with a shadow ray; the next direction
// is drawn from GGX's lobe or the diffuse one and weighted by the density
// of both together. Alpha-masked texels, and blended ones with their
// opacity's probability, pass the ray on. Metal 4 only, so far: Vulkan's
// ray queries wait for a driver with them.
//
//   goose pathtrace.goose -- [model.glb] [--hdr sky.hdr] [--yaw r --pitch r --zoom k]
//                            [--frames N --out shot.png]

import gpu from "../..";
import .gltf;
import .scene;
import .pbr;
import .ibl;
import std;
import os;
import math;

struct Camera {
    eye: float3,
    tan_x: f32,
    right: float3,
    tan_y: f32,
    up: float3,
    exposure: f32,
    forward: float3,
    sun_strength: f32,
    sun: float3,
    bounces: u32,
    sun_color: float3,
    frame: u32,
}

struct Trace {
    refs: scene::Refs,
    scene: gpu::Accel,
    accum: gpu::RWStructuredBuffer<float4>,     // the sum of the samples so far, and their count
    env: gpu::Texture2D<float4>,
    width: u32,
    height: u32,
    alpha: u32,         // whether any material masks or blends
}

// A hash of the pixel, the frame and the bounce, as random numbers.
fn pcg(x: u32) -> u32 {
    let s = x * 747796405 + 2891336453;
    let w = ((s >> ((s >> 28) + 4)) ^ s) * 277803737;
    (w >> 22) ^ w
}

struct Rng { state: u32 }

fn next(r: Rng) -> Rng { Rng { state: pcg(r.state) } }
fn unit(r: Rng) -> f32 {
    let scale: f32 = 2.3283064e-10;
    r.state * scale
}

// A hit's surface: where, its shading and geometric normals facing the
// ray, and its material's values there.
struct Surface {
    pos: float3,
    n: float3,
    geo: float3,
    base: float4,
    metallic: f32,
    roughness: f32,
    emissive: float3,
    flags: u32,
    cutoff: f32,
    transmission: f32,
    ior: f32,
    absorption: float3,
    entering: u32,      // the ray meets the surface's outside, against its winding's normal
}

fn surface_at(refs: scene::Refs, h: gpu::Hit, ray: gpu::Ray) -> Surface {
    let inst = gpu::load(refs.instances, h.id);
    let prim = gpu::load(refs.prims, inst.prim);
    let m = gpu::load(refs.materials, prim.material);
    let at = prim.first_index + 3 * h.primitive;
    let a = gpu::load(refs.verts, gpu::load(refs.indices, at));
    let b = gpu::load(refs.verts, gpu::load(refs.indices, at + 1));
    let c = gpu::load(refs.verts, gpu::load(refs.indices, at + 2));
    let w1 = h.barycentrics.x;
    let w2 = h.barycentrics.y;
    let w0 = 1.0 - w1 - w2;
    let uv = a.uv * w0 + b.uv * w1 + c.uv * w2;
    let local_n = a.normal * w0 + b.normal * w1 + c.normal * w2;
    var n = normalize(xyz(gpu::mul(inst.normal, float4 { local_n.x, local_n.y, local_n.z, 0.0 })));
    let pa = xyz(gpu::mul(inst.world, float4 { a.pos.x, a.pos.y, a.pos.z, 1.0 }));
    let pb = xyz(gpu::mul(inst.world, float4 { b.pos.x, b.pos.y, b.pos.z, 1.0 }));
    let pc = xyz(gpu::mul(inst.world, float4 { c.pos.x, c.pos.y, c.pos.z, 1.0 }));
    var geo = normalize(cross(pb - pa, pc - pa));
    let entering: u32 = if dot(geo, ray.dir) < 0.0 { 1 } else { 0 };
    if dot(geo, ray.dir) > 0.0 { geo = geo * -1.0; }
    if dot(n, geo) < 0.0 { n = n * -1.0; }

    // The normal map, in the file's tangent frame or the triangle's own.
    var t = float3_0;
    var sign: f32 = 1.0;
    if prim.flags & gltf::HAS_TANGENTS != 0 {
        let lt = a.tangent * w0 + b.tangent * w1 + c.tangent * w2;
        t = xyz(gpu::mul(inst.world, float4 { lt.x, lt.y, lt.z, 0.0 }));
        sign = if lt.w < 0.0 { -1.0 } else { 1.0 };
    } else {
        let e1 = pb - pa;
        let e2 = pc - pa;
        let d1 = b.uv - a.uv;
        let d2 = c.uv - a.uv;
        let det = d1.x * d2.y - d2.x * d1.y;
        let r = if abs(det) > 1e-12 { 1.0 / det } else { 0.0 };
        t = (e1 * d2.y - e2 * d1.y) * r;
        let bt = (e2 * d1.x - e1 * d2.x) * r;
        sign = if dot(cross(n, t), bt) < 0.0 { -1.0 } else { 1.0 };
    }
    t = t - n * dot(n, t);
    if dot(t, t) > 1e-12 {
        t = normalize(t);
        let bt = cross(n, t) * sign;
        let s = xyz(gpu::sample_level(m.normal_tex, gpu::LINEAR_REPEAT, uv, 0.0)) * 2.0 - float3_1;
        let tn = normalize(float3 { s.x * m.normal_scale, s.y * m.normal_scale, s.z });
        n = normalize(t * tn.x + bt * tn.y + n * tn.z);
    }

    let base = m.base * gpu::sample_level(m.base_tex, gpu::LINEAR_REPEAT, uv, 0.0);
    let mr = gpu::sample_level(m.mr_tex, gpu::LINEAR_REPEAT, uv, 0.0);
    let em = m.emissive * xyz(gpu::sample_level(m.emissive_tex, gpu::LINEAR_REPEAT, uv, 0.0));
    let pos = ray.origin + ray.dir * h.t;
    Surface { pos: pos, n: n, geo: geo, base: base, metallic: clamp(m.metallic * mr.z, 0.0, 1.0),
              roughness: clamp(m.roughness * mr.y, 0.04, 1.0), emissive: em, flags: m.flags,
              cutoff: m.alpha_cutoff, transmission: m.transmission, ior: m.ior, absorption: m.absorption,
              entering: entering }
}

fn env_at(env: gpu::Texture2D<float4>, d: float3) -> float3 {
    xyz(gpu::sample_level(env, gpu::LINEAR_REPEAT, pbr::equirect_uv(d), 0.0))
}

fn lum(c: float3) -> f32 { c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722 }

// What a surface does with light, a kind each: glTF's metallic-roughness
// model, or a smooth dielectric that light passes through
// (KHR_materials_transmission), absorbing as it goes inside
// (KHR_materials_volume). Each kind's functions follow; the path loop calls
// them by the kind, as case functions (spec §8.2), and a new kind is a
// variant and its overloads, which the compiler asks for.
enum Bsdf {
    Opaque { s: pbr::Surface },
    Glass { tint: float3, ior: f32, absorption: float3 },
}

fn bsdf_of(s: Surface) -> Bsdf {
    if s.transmission > 0.5 {
        return Bsdf.Glass { tint: xyz(s.base), ior: s.ior, absorption: s.absorption };
    }
    Bsdf.Opaque { s: pbr::Surface { base: xyz(s.base), metallic: s.metallic, roughness: s.roughness } }
}

// Where a path goes from a surface: the direction, what its throughput is
// multiplied by (zero to end it), and whether it passed through.
struct Bounce { dir: float3, weight: float3, through: u32 }

// What a bounce needs: the shading normal and the geometric one, both
// facing the ray; the direction back along it; whether the ray comes from
// outside; and three random numbers.
struct At { n: float3, geo: float3, v: float3, entering: u32, u: float3 }

// Light from direction l reflected toward v, which a delta lobe has none of.
fn direct(b: Bsdf.Opaque, at: At, l: float3) -> float3 { pbr::brdf(b.s, at.n, at.v, l) }
fn direct(g: Bsdf.Glass, at: At, l: float3) -> float3 { float3_0 }

// What the medium on a surface's inside absorbs per unit of distance.
fn absorption(b: Bsdf.Opaque) -> float3 { float3_0 }
fn absorption(g: Bsdf.Glass) -> float3 { g.absorption }

// The density of drawing l from the mix of the two lobes.
fn lobe_pdf(s: pbr::Surface, n: float3, v: float3, l: float3, spec: f32) -> f32 {
    let nl = dot(n, l);
    if nl <= 0.0 { return 0.0; }
    let h = normalize(v + l);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(v, h), 0.0001);
    let alpha = max(s.roughness * s.roughness, 0.002);
    let ps = pbr::ggx_d(nh, alpha) * nh / (4.0 * vh);
    spec * ps + (1.0 - spec) * nl / pbr::PI
}

// GGX's lobe or the diffuse one, by how much each reflects, weighted by
// the density of both together.
fn scatter(b: Bsdf.Opaque, at: At) -> Bounce {
    let s = b.s;
    let f0 = lerp(float3 { 0.04, 0.04, 0.04 }, s.base, s.metallic);
    let spec = clamp(lum(f0) / (lum(f0) + lum(s.base) * (1.0 - s.metallic) + 0.0001), 0.15, 0.95);
    var l = float3_0;
    if at.u.z < spec {
        let alpha = max(s.roughness * s.roughness, 0.002);
        let h = pbr::ggx_half(float2 { at.u.x, at.u.y }, at.n, alpha);
        l = h * (2.0 * dot(at.v, h)) - at.v;
    } else {
        l = pbr::cosine_dir(float2 { at.u.x, at.u.y }, at.n);
    }
    let pdf = lobe_pdf(s, at.n, at.v, l, spec);
    if pdf <= 0.0 || dot(l, at.geo) <= 0.0 { return Bounce { dir: l, weight: float3_0, through: 0 }; }
    Bounce { dir: l, weight: pbr::brdf(s, at.n, at.v, l) / pdf, through: 0 }
}

// Reflected or refracted by Fresnel's odds, all of the light either way,
// refraction tinted by the base colour.
fn scatter(g: Bsdf.Glass, at: At) -> Bounce {
    let cosi = clamp(dot(at.n, at.v), 0.0, 1.0);
    let eta = if at.entering != 0 { 1.0 / g.ior } else { g.ior };
    let sin2t = eta * eta * (1.0 - cosi * cosi);
    let reflect = at.n * (2.0 * cosi) - at.v;
    if sin2t >= 1.0 { return Bounce { dir: reflect, weight: float3_1, through: 0 }; }
    let cost = sqrt(1.0 - sin2t);
    let rs = (eta * cosi - cost) / (eta * cosi + cost);
    let rp = (cosi - eta * cost) / (cosi + eta * cost);
    let fresnel = 0.5 * (rs * rs + rp * rp);
    if at.u.z < fresnel { return Bounce { dir: reflect, weight: float3_1, through: 0 }; }
    let refract = at.v * -eta + at.n * (eta * cosi - cost);
    Bounce { dir: normalize(refract), weight: g.tint, through: 1 }
}

fn offset(p: float3, n: float3) -> float3 { p + n * 0.0005 }

// Whether a ray reaches the sky: past cut-out texels, and blended ones with
// the chance of their transparency. Opaque surfaces end it at once, which
// trace_any finds soonest when no surface in the model is anything else.
fn sees_sky(p: Trace, ray: gpu::Ray, rng: Rng, alpha: bool) -> bool {
    if !alpha { return !gpu::trace_any(p.scene, ray, 255); }
    var r = ray;
    var g = rng;
    for k: u32 in 8 {
        let h = gpu::trace(p.scene, r, 255);
        if h.kind == gpu::HIT_NONE { return true; }
        let s = surface_at(p.refs, h, r);
        g = next(g);
        let skip = (s.flags & gltf::ALPHA_MASK != 0 && s.base.w < s.cutoff) ||
                   (s.flags & gltf::ALPHA_BLEND != 0 && unit(g) > s.base.w);
        if !skip { return false; }
        r = gpu::Ray { origin: s.pos + r.dir * 0.001, t_min: 0.0, dir: r.dir, t_max: 1e30 };
    }
    false
}

compute(8, 8) fn trace_kernel(p: Trace, cam: const Camera&) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    var rng = Rng { state: pcg(id.x + pcg(id.y + pcg(cam.frame))) };
    rng = next(rng);
    let jx = unit(rng);
    rng = next(rng);
    let jy = unit(rng);
    let ndc = float2 { ((id.x + jx) / p.width) * 2.0 - 1.0, 1.0 - ((id.y + jy) / p.height) * 2.0 };
    var ray = gpu::Ray { origin: cam.eye, t_min: 0.0,
                         dir: normalize(cam.forward + cam.right * (ndc.x * cam.tan_x) + cam.up * (ndc.y * cam.tan_y)),
                         t_max: 1e30 };
    var radiance = float3_0;
    var through = float3_1;
    var bounce: u32 = 0;
    var passes: u32 = 0;
    var medium = float3_0;      // what the space the ray travels through absorbs
    loop {
        let h = gpu::trace(p.scene, ray, 255);
        if h.kind == gpu::HIT_NONE {
            radiance = radiance + through * env_at(p.env, ray.dir);
            break;
        }
        // What the medium the ray crossed absorbed on the way (Beer-Lambert).
        through = through * float3 { exp(-medium.x * h.t), exp(-medium.y * h.t), exp(-medium.z * h.t) };
        let s = surface_at(p.refs, h, ray);
        // Cut-out and see-through texels pass the ray on, a few at most.
        rng = next(rng);
        let skip = (s.flags & gltf::ALPHA_MASK != 0 && s.base.w < s.cutoff) ||
                   (s.flags & gltf::ALPHA_BLEND != 0 && unit(rng) > s.base.w);
        if skip && passes < 8 {
            passes += 1;
            ray = gpu::Ray { origin: s.pos + ray.dir * 0.001, t_min: 0.0, dir: ray.dir, t_max: 1e30 };
            continue;
        }
        radiance = radiance + through * s.emissive;
        let b = bsdf_of(s);
        rng = next(rng);
        let u1 = unit(rng);
        rng = next(rng);
        let u2 = unit(rng);
        rng = next(rng);
        let at = At { n: s.n, geo: s.geo, v: ray.dir * -1.0, entering: s.entering, u: float3 { u1, u2, unit(rng) } };

        // The sun, if nothing is in the way and the surface reflects it.
        let shadow = gpu::Ray { origin: offset(s.pos, s.geo), t_min: 0.0, dir: cam.sun, t_max: 1e30 };
        rng = next(rng);
        if dot(s.n, cam.sun) > 0.0 {
            let lit = direct(b, at, cam.sun);
            if lum(lit) > 0.0 && sees_sky(p, shadow, rng, p.alpha != 0) {
                radiance = radiance + through * lit * cam.sun_color * cam.sun_strength;
            }
        }
        if bounce >= cam.bounces { break; }

        let next_bounce = scatter(b, at);
        if lum(next_bounce.weight) <= 0.0 { break; }
        through = through * next_bounce.weight;
        // Through a surface: into its medium from outside, out of it from inside.
        if next_bounce.through != 0 { medium = if s.entering != 0 { absorption(b) } else { float3_0 }; }
        // Past three bounces, paths end with the chance they carry little.
        if bounce >= 3 {
            let q = clamp(max(through.x, max(through.y, through.z)), 0.05, 1.0);
            rng = next(rng);
            if unit(rng) > q { break; }
            through = through / q;
        }
        let side = if next_bounce.through != 0 { s.geo * -1.0 } else { s.geo };
        ray = gpu::Ray { origin: offset(s.pos, side), t_min: 0.0, dir: next_bounce.dir, t_max: 1e30 };
        bounce += 1;
    }
    // A sample that went wrong (NaN fails every comparison) counts as
    // black, and one too bright to average away is limited.
    if !(lum(radiance) >= 0.0) { radiance = float3_0; }
    let peak = max(radiance.x, max(radiance.y, radiance.z));
    if peak > 64.0 { radiance = radiance * (64.0 / peak); }
    let at = id.y * p.width + id.x;
    let sum = if cam.frame == 0 { float4_0 } else { gpu::load(p.accum, at) };
    gpu::store(p.accum, at, sum + float4 { radiance.x, radiance.y, radiance.z, 1.0 });
}

struct Show { accum: gpu::StructuredBuffer<float4>, width: u32, exposure: f32 }
struct ShowVaryings { ndc: float2 }

vertex fn show_vs(s: Show) -> float4, ShowVaryings {
    let i = gpu::vertex_index();
    let xy = float2 { if i == 1 { 3.0 } else { -1.0 }, if i == 2 { 3.0 } else { -1.0 } };
    return float4 { xy.x, xy.y, 0.5, 1.0 }, ShowVaryings { ndc: xy };
}

// The average of the samples, tone mapped.
fragment fn show_fs(v: ShowVaryings, s: Show) -> float4 {
    let px = gpu::frag_coord();
    let sum = gpu::load(s.accum, (px.y as u32) * s.width + px.x as u32);
    pbr::display(xyz(sum) * (s.exposure / max(sum.w, 1.0)))
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
    var zoom: f32 = 1.0;
    var i: i64 = 1;
    while i < argv.len {
        if argv[i] == "--hdr" && i + 1 < argv.len { hdr = str(argv[i + 1]); i += 2; continue; }
        if argv[i] == "--out" && i + 1 < argv.len { out = str(argv[i + 1]); i += 2; continue; }
        if (argv[i] == "--yaw" || argv[i] == "--pitch" || argv[i] == "--zoom") && i + 1 < argv.len {
            let v, ok = parse_flt(argv[i + 1]);
            if ok && argv[i] == "--yaw" { yaw = v as f32; }
            if ok && argv[i] == "--pitch" { pitch = v as f32; }
            if ok && argv[i] == "--zoom" && v > 0.0 { zoom = v as f32; }
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
    guard gpu::open("Goose: glTF, path traced", 1280, 800, gpu::WINDOW_RESIZABLE) else {
        write_stderr(str("gpu: ", gpu::error(), "\n"));
        return;
    }
    guard gpu::ray_query() else {
        write_stderr(str("gpu: the ", gpu::driver(), " device has no ray queries\n"));
        gpu::close();
        return;
    }
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

    // A structure for each instance's primitive, and the instances over them,
    // each naming its instance by its id.
    let vcount = gpu::buffer_size(m.vertex_buffer) / 48;
    var accels: gpu::Accel[>..] = [];
    var records: gpu::Instance[>..] = [];
    for d in m.draws {
        let a = gpu::primitive_accel(gpu::PrimitiveAccelDesc {
            vertices: m.vertex_buffer, vertex_stride: 48, vertex_count: vcount, indices: m.index_buffer,
            index_size: 4, index_offset: d.first_index * 4, triangle_count: d.index_count / 3, .. });
        guard a.index != 0 else { abort(str("gpu: ", gpu::error())); }
        accels.push(a);
    }
    for d in m.draws.len { records.push(gpu::instance(m.draws[d].world, accels[d], m.draws[d].instance)); }
    let ibuf = gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_ACCEL_INPUT, bytes_of(records));
    let tlas = gpu::instance_accel(gpu::InstanceAccelDesc { instances: ibuf, max: records.len });
    for a in accels { gpu::build(a); }
    gpu::build(tlas, records.len);
    gpu::wait(gpu::submit());

    var alpha = false;
    for d in m.draws { if d.flags & (gltf::ALPHA_MASK | gltf::ALPHA_BLEND) != 0 { alpha = true; } }
    let tracer = gpu::compute_pipeline(trace_kernel);
    let shower = gpu::pipeline(show_vs, show_fs);
    let center = (m.lo + m.hi) * 0.5;
    let radius = length(m.hi - m.lo) * 0.5;
    var dist = radius * 2.6 / zoom;
    var last = gpu::mouse_position();
    var accum = gpu::Buffer { 0 };
    var aw: i64 = 0;
    var ah: i64 = 0;
    var sample: u32 = 0;
    var shots = 0;
    var n: i64 = 0;
    var t0 = clock();
    while gpu::frame() {
        if gpu::key_down("Escape") { gpu::quit(); }
        let dt: f32 = min(gpu::delta_time(), 0.1) as f32;
        let mouse = gpu::mouse_position();
        var moved = false;
        if gpu::mouse_down(1) && (mouse.x != last.x || mouse.y != last.y) {
            yaw += (mouse.x - last.x) * 0.006;
            pitch += (mouse.y - last.y) * 0.006;
            moved = true;
        }
        last = mouse;
        if gpu::key_down("Left") { yaw -= dt * 1.5; moved = true; }
        if gpu::key_down("Right") { yaw += dt * 1.5; moved = true; }
        if gpu::key_down("Up") { pitch += dt; moved = true; }
        if gpu::key_down("Down") { pitch -= dt; moved = true; }
        if gpu::key_down("W") { dist = max(dist * (1.0 - dt), radius * 0.3); moved = true; }
        if gpu::key_down("S") { dist = dist * (1.0 + dt); moved = true; }
        pitch = clamp(pitch, -1.5, 1.5);

        let size = gpu::screen_size();
        if size.x != aw || size.y != ah {
            if accum.index != 0 { gpu::release(accum); }
            aw = size.x;
            ah = size.y;
            accum = gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_WRITE, aw * ah * 16);
            moved = true;
        }
        if moved { sample = 0; t0 = clock(); }
        let eye = center + float3 { cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw) } * dist;
        let fwd = normalize(center - eye);
        let right = normalize(cross(fwd, float3 { 0.0, 1.0, 0.0 }));
        let half_fov: f32 = 0.4;
        let th = tan(half_fov);
        let width: f32 = aw;
        let cam = Camera { eye: eye, tan_x: th * width / max(ah, 1), right: right, tan_y: th,
                           up: cross(right, fwd), exposure: 1.0, forward: fwd, sun_strength: 3.0, sun: sun,
                           bounces: 6, sun_color: float3 { 1.0, 0.95, 0.85 }, frame: sample };

        gpu::begin_compute();
        gpu::bind(tracer);
        gpu::push_constants(Trace { refs: m.refs, scene: tlas, accum: gpu::rw_structured<float4>(accum),
                                    env: gpu::sampled<float4>(maps.env), width: aw as u32, height: ah as u32,
                                    alpha: if alpha { 1 } else { 0 } });
        gpu::uniforms(cam);
        gpu::dispatch((aw + 7) / 8, (ah + 7) / 8, 1);
        gpu::end_compute();
        gpu::barrier(accum, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_SHADER_READ);
        guard gpu::begin_screen(float4 { 0.0, 0.0, 0.0, 1.0 }) else { continue; }
        gpu::bind(shower);
        gpu::push_constants(Show { accum: gpu::structured<float4>(accum), width: aw as u32, exposure: 1.0 });
        gpu::draw(3);
        gpu::end_pass();
        sample += 1;
        n++;
        if gpu::key_pressed("P") {
            let path = str("pathtrace_", shots, ".png");
            gpu::screenshot(path);
            print("saved ", path, " at ", sample, " samples");
            shots++;
        }
        if frames >= 0 && n >= frames {
            if out.len > 0 { gpu::screenshot(out); }
            gpu::wait(gpu::submit());
            print(sample, " samples a pixel in ", (clock() - t0) * 1000.0, " ms");
            gpu::quit();
        }
    }
    gpu::close();
}
