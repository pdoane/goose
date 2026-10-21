// A software ray tracer: a bounding volume hierarchy over a glTF model's
// triangles, built on the CPU in a global array whose nodes link their
// children by self-relative references (Node&<u32>, spec §3.9), uploaded
// byte for byte, and walked by one `trace` function that the CPU and the
// GPU both run (docs/design/gpu_language.md §9, §10). Nothing converts the
// tree for the GPU: a link is the distance from the field holding it to the
// node it names, on either side, which is why bytes_of writes it out at
// all. The traversal's stack holds links `in nodes`, measured from the
// array's start, which never leave the program.
//
// It renders the model shaded by its normals and an ambient occlusion
// term, then checks every pixel's hit against the CPU's and, where the
// device has ray queries, against the hardware's own traversal of the same
// triangles. It needs no ray queries itself, so it runs on Vulkan too.
//
//   goose bvh.goose -- [model.glb] [--out bvh.png]

import gpu from "../..";
import .gltf;
import std;
import os;
import math;

struct Tri { a: float3, b: float3, c: float3 }

// A node: its bounds, and either two children or a run of triangles.
struct Node {
    lo: float3,
    hi: float3,
    left: Node&<u32>?,
    right: Node&<u32>?,
    first: u32,
    count: u32,
}

var nodes: Node[>..] = [];
var tris: Tri[>..] = [];

let NONE: u32 = 0xffffffff;

struct Hit { t: f32, tri: u32, u: f32, v: f32 }

// Whether the ray (origin o, inverse direction inv) meets the box before tmax.
fn hit_box(o: float3, inv: float3, lo: float3, hi: float3, tmax: f32) -> bool {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let tnear = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let tfar = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    tnear <= tfar && tfar >= 0.0 && tnear < tmax
}

// Möller and Trumbore's test, keeping the nearer hit.
fn hit_tri(o: float3, d: float3, t: Tri, best: Hit, id: u32) -> Hit {
    let e1 = t.b - t.a;
    let e2 = t.c - t.a;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if abs(det) < 1e-12 { return best; }
    let inv = 1.0 / det;
    let s = o - t.a;
    let u = dot(s, p) * inv;
    if u < 0.0 || u > 1.0 { return best; }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 { return best; }
    let dist = dot(e2, q) * inv;
    if dist <= 1e-4 || dist >= best.t { return best; }
    Hit { t: dist, tri: id, u: u, v: v }
}

// The nearest hit along o + t d, walking the tree from `root` with a stack
// of links.
fn trace(o: float3, d: float3, root: const Node&) -> Hit {
    let inv = float3 { 1.0 / d.x, 1.0 / d.y, 1.0 / d.z };
    var best = Hit { t: 1e30, tri: NONE, u: 0.0, v: 0.0 };
    var todo: (const Node&<u32 in nodes>)[..64] = [];
    todo.push(root);
    while todo.len > 0 {
        let n .= todo.pop();
        if !hit_box(o, inv, n.lo, n.hi, best.t) { continue; }
        if n.count > 0 {
            for k: u32 in n.count { best = hit_tri(o, d, tris[n.first + k], best, n.first + k); }
        } else {
            let a .= n.left;
            let b .= n.right;
            if a { todo.push(a); }
            if b { todo.push(b); }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Building: nodes split at the median centroid along their longest axis,
// down to four triangles a leaf; the triangles reordered so a leaf's are a
// run.

struct Box { lo: float3, hi: float3 }

fn grow(b: Box, p: float3) -> Box {
    Box { lo: float3 { min(b.lo.x, p.x), min(b.lo.y, p.y), min(b.lo.z, p.z) },
          hi: float3 { max(b.hi.x, p.x), max(b.hi.y, p.y), max(b.hi.z, p.z) } }
}

fn empty() -> Box { Box { lo: float3 { 1e30, 1e30, 1e30 }, hi: float3 { -1e30, -1e30, -1e30 } } }

fn axis_of(v: float3, a: i64) -> f32 { if a == 0 { v.x } else if a == 1 { v.y } else { v.z } }

struct Span { node: i64, lo: i64, hi: i64 }

fn build(src: Tri[:]) {
    var order: i64[>..] = [];
    var centre: float3[>..] = [];
    for i in src.len {
        order.push(i);
        centre.push((src[i].a + src[i].b + src[i].c) * (1.0 / 3.0));
    }
    nodes.push(Node { lo: float3_0, hi: float3_0, left: null, right: null, first: 0, count: 0 });
    var work: Span[>..<] = [Span { node: 0, lo: 0, hi: src.len }];
    var leaves: Span[>..] = [];
    while work.len > 0 {
        let s = work.pop();
        var b = empty();
        var cb = empty();
        for k in s.lo..s.hi {
            let t = src[order[k]];
            b = grow(grow(grow(b, t.a), t.b), t.c);
            cb = grow(cb, centre[order[k]]);
        }
        nodes[s.node].lo = b.lo;
        nodes[s.node].hi = b.hi;
        if s.hi - s.lo <= 4 {
            leaves.push(s);
            continue;
        }
        let ext = cb.hi - cb.lo;
        let a = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
        // The median along the axis, by sorting the span's triangles.
        sort(order[s.lo..s.hi]) { x, y => axis_of(centre[x], a) < axis_of(centre[y], a) };
        let mid = (s.lo + s.hi) / 2;
        let l = nodes.len;
        nodes.push(Node { lo: float3_0, hi: float3_0, left: null, right: null, first: 0, count: 0 });
        nodes.push(Node { lo: float3_0, hi: float3_0, left: null, right: null, first: 0, count: 0 });
        nodes[s.node].left .= &nodes[l];
        nodes[s.node].right .= &nodes[l + 1];
        work.push(Span { node: l, lo: s.lo, hi: mid });
        work.push(Span { node: l + 1, lo: mid, hi: s.hi });
    }
    // The triangles in leaf order, each leaf a run of them.
    for s in leaves {
        nodes[s.node].first = tris.len as u32;
        nodes[s.node].count = (s.hi - s.lo) as u32;
        for k in s.lo..s.hi { tris.push(src[order[k]]); }
    }
}

// ---------------------------------------------------------------------------
// Rendering: a ray a pixel, shaded by the hit's normal and how much of the
// hemisphere above it a few short rays find open.

struct Camera { eye: float3, right: float3, up: float3, forward: float3, tan_x: f32, tan_y: f32, reach: f32 }

fn ray_dir(c: Camera, x: u32, y: u32, w: u32, h: u32) -> float3 {
    let half: f32 = 0.5;
    let nx = ((x + half) / w) * 2.0 - 1.0;
    let ny = 1.0 - ((y + half) / h) * 2.0;
    normalize(c.forward + c.right * (nx * c.tan_x) + c.up * (ny * c.tan_y))
}

fn shade(c: Camera, d: float3, h: Hit) -> u32 {
    if h.tri == NONE { return 0xff806040; }
    let t = tris[h.tri];
    var n = normalize(cross(t.b - t.a, t.c - t.a));
    if dot(n, d) > 0.0 { n = n * -1.0; }
    // Occlusion from eight fixed directions about the normal.
    let p = c.eye + d * h.t + n * (c.reach * 0.001);
    var open: f32 = 0.0;
    for k: u32 in 8 {
        let a: f32 = k * 0.785398;
        let side = normalize(cross(n, if abs(n.y) < 0.9 { float3 { 0.0, 1.0, 0.0 } } else { float3 { 1.0, 0.0, 0.0 } }));
        let up = cross(n, side);
        let dir = normalize(n + side * cos(a) * 0.8 + up * sin(a) * 0.8);
        let o = trace(p, dir, nodes[0]);
        if o.tri == NONE || o.t > c.reach * 0.25 { open += 0.125; }
    }
    let col = (n * 0.5 + float3 { 0.5, 0.5, 0.5 }) * (0.25 + 0.75 * open);
    let r = clamp(col.x, 0.0, 1.0) * 255.0;
    let g = clamp(col.y, 0.0, 1.0) * 255.0;
    let b = clamp(col.z, 0.0, 1.0) * 255.0;
    (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | 0xff000000
}

struct Render { cam: Camera, out: gpu::RWStructuredBuffer<u32>, hits: gpu::RWStructuredBuffer<Hit>, width: u32, height: u32 }

compute(8, 8) fn render(p: Render) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let d = ray_dir(p.cam, id.x, id.y, p.width, p.height);
    let h = trace(p.cam.eye, d, nodes[0]);
    let at = id.y * p.width + id.x;
    gpu::store(p.hits, at, h);
    gpu::store(p.out, at, shade(p.cam, d, h));
}

fn main() {
    let argv = args();
    var home: u8[>..] = [];
    env("HOME", home);
    var model = str(home, "/.goose/assets/DamagedHelmet.glb");
    var out = str("bvh.png");
    var i: i64 = 1;
    while i < argv.len {
        if argv[i] == "--out" && i + 1 < argv.len { out = str(argv[i + 1]); i += 2; continue; }
        model = str(argv[i]);
        i++;
    }
    // The model's triangles in world space.
    var blob: u8[>..] = [];
    var verts: gltf::Vertex[>..] = [];
    var indices: u32[>..] = [];
    var prims: gltf::Prim[>..] = [];
    var mats: gltf::MaterialDesc[>..] = [];
    var images: gltf::ImageRef[>..] = [];
    var textures: i64[>..] = [];
    var insts: gltf::Instance[>..] = [];
    let e = gltf::load(model, blob, verts, indices, prims, mats, images, textures, insts);
    guard e.len == 0 else { abort(e); }
    var src: Tri[>..] = [];
    for inst in insts {
        let pr = prims[inst.prim];
        var k: u32 = 0;
        while k + 2 < pr.index_count {
            var w: float3[3] = [float3_0; 3];
            for j in 3 {
                let v = verts[indices[pr.first_index + k + j as u32]].pos;
                w[j] = xyz(gpu::mul(inst.world, float4 { v.x, v.y, v.z, 1.0 }));
            }
            src.push(Tri { a: w[0], b: w[1], c: w[2] });
            k += 3;
        }
    }
    let t0 = clock();
    build(src);
    let bms = (clock() - t0) * 1000.0;

    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    gpu::upload(gpu_global_id(nodes), nodes);
    gpu::upload(gpu_global_id(tris), tris);
    let w: u32 = 960;
    let h: u32 = 720;
    let centre = (nodes[0].lo + nodes[0].hi) * 0.5;
    let radius = length(nodes[0].hi - nodes[0].lo) * 0.5;
    let eye = centre + normalize(float3 { 0.6, 0.25, 0.75 }) * (radius * 2.2);
    let fwd = normalize(centre - eye);
    let right = normalize(cross(fwd, float3 { 0.0, 1.0, 0.0 }));
    let th: f32 = 0.42;
    let cam = Camera { eye: eye, right: right, up: cross(right, fwd), forward: fwd, tan_x: th * w / h, tan_y: th,
                       reach: radius };
    let pixels = gpu::buffer(gpu::BUFFER_WRITE, w * h * 4);
    let hits = gpu::buffer(gpu::BUFFER_WRITE, w * h * 16);
    let p = gpu::compute_pipeline(render);
    var gms = 0.0;
    for run in 2 {
        let t1 = clock();
        gpu::begin_compute();
        gpu::bind(p);
        gpu::push_constants(Render { cam: cam, out: gpu::rw_structured<u32>(pixels),
                                     hits: gpu::rw_structured<Hit>(hits), width: w, height: h });
        gpu::dispatch((w + 7) / 8, (h + 7) / 8, 1);
        gpu::end_compute();
        gpu::wait(gpu::submit());
        gms = (clock() - t1) * 1000.0;
    }
    let image = gpu::texture(gpu::TextureDesc { width: w as i32, height: h as i32, .. });
    gpu::barrier(pixels, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_COPY_SRC);
    gpu::barrier(image, gpu::ACCESS_NONE, gpu::ACCESS_COPY_DST);
    gpu::copy_to_texture(pixels, 0, image);
    gpu::barrier(image, gpu::ACCESS_COPY_DST, gpu::ACCESS_SHADER_READ);
    gpu::save_png(image, out);
    let rh = gpu::read_buffer(hits, 0, w * h * 16);
    gpu::wait(gpu::submit());
    var got: Hit[>..] = [];
    guard gpu::values(rh, got) else { abort("a read back is not ready"); }

    // The CPU's trace of a sample of the pixels.
    var checked = 0;
    var differ = 0;
    var k: u32 = 0;
    while k < w * h {
        let x = k % w;
        let y = k / w;
        let mine = trace(cam.eye, ray_dir(cam, x, y, w, h), nodes[0]);
        if mine.tri != got[k].tri || abs(mine.t - got[k].t) > 1e-3 * radius {
            if differ < 3 { print("  pixel ", x, ",", y, ": CPU ", mine.tri, " at ", mine.t, ", GPU ", got[k].tri, " at ", got[k].t); }
            differ++;
        }
        checked++;
        k += 97;
    }
    var report = str(gpu::driver(), ": ", src.len, " triangles, ", nodes.len, " nodes built in ", fixed(bms, 1),
                     " ms; ", w, "x", h, " rays and 8 occlusion rays each in ", fixed(gms, 2), " ms on the GPU; ",
                     differ, " of ", checked, " sampled pixels differ from the CPU");

    // The hardware's own traversal of the same triangles, where there is one.
    if gpu::ray_query() {
        let vb = gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_ACCEL_INPUT, bytes_of(tris));
        let blas = gpu::primitive_accel(gpu::PrimitiveAccelDesc { vertices: vb, vertex_count: tris.len * 3,
                                                                  triangle_count: tris.len, .. });
        let one = [gpu::instance(gpu::mat4_identity, blas, 0)];
        let ib = gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_ACCEL_INPUT, bytes_of(one));
        let tlas = gpu::instance_accel(gpu::InstanceAccelDesc { instances: ib, max: 1 });
        gpu::build(blas);
        gpu::build(tlas, 1);
        let hw = gpu::buffer(gpu::BUFFER_WRITE, w * h * 16);
        gpu::begin_compute();
        gpu::bind(gpu::compute_pipeline(hardware));
        gpu::push_constants(Hardware { cam: cam, scene: tlas, hits: gpu::rw_structured<Hit>(hw), width: w,
                                       height: h });
        gpu::dispatch((w + 7) / 8, (h + 7) / 8, 1);
        gpu::end_compute();
        let rhw = gpu::read_buffer(hw, 0, w * h * 16);
        gpu::wait(gpu::submit());
        var theirs: Hit[>..] = [];
        gpu::values(rhw, theirs);
        var agree = 0;
        for j in got.len { if got[j].tri == theirs[j].tri || abs(got[j].t - theirs[j].t) < 1e-4 * radius { agree++; } }
        report.append(str("; the hardware's traversal agrees on ", agree, " of ", got.len, " pixels"));
    }
    print(report);
    print("image in ", out);
    gpu::close();
}

fn fixed(v: f64, decimals: i64) -> u8[>..] {
    var o: u8[>..] = [];
    format_flt(o, v, decimals);
    o
}

struct Hardware { cam: Camera, scene: gpu::Accel, hits: gpu::RWStructuredBuffer<Hit>, width: u32, height: u32 }

compute(8, 8) fn hardware(p: Hardware) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let d = ray_dir(p.cam, id.x, id.y, p.width, p.height);
    let r = gpu::trace(p.scene, gpu::Ray { origin: p.cam.eye, t_min: 1e-4, dir: d, t_max: 1e30 }, 255);
    let h = if r.kind == gpu::HIT_NONE { Hit { t: 1e30, tri: NONE, u: 0.0, v: 0.0 } }
            else { Hit { t: r.t, tri: r.primitive, u: r.barycentrics.x, v: r.barycentrics.y } };
    gpu::store(p.hits, id.y * p.width + id.x, h);
}
