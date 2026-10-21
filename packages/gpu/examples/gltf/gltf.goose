// glTF 2.0 loading: a .glb, or a .gltf with its buffers and images beside
// it or in data URIs, flattened for drawing. Every primitive's vertices go
// into one array and its indices into another, made absolute so that no
// draw needs a base vertex; each node that holds a mesh becomes an
// instance of each of the mesh's primitives, with its world transform.
// Triangles only; no skins, morph targets, cameras or animation.

namespace gltf;

import gpu from "../..";
import std;
import os;
import json;
import base64;
import math;

// What shaders read: vertices, materials, primitives and instances, in
// Goose's packed layout.
struct Vertex { pos: float3, normal: float3, tangent: float4, uv: float2 }

let DOUBLE_SIDED: u32 = 1;
let ALPHA_MASK: u32 = 2;
let ALPHA_BLEND: u32 = 4;
let HAS_TANGENTS: u32 = 8;

struct Material {
    base: float4,
    emissive: float3,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion: f32,
    alpha_cutoff: f32,
    flags: u32,
    // KHR_materials_transmission, _ior and _volume: how much light passes
    // through, the index of refraction, and the medium's absorption per
    // unit of distance (-ln(attenuation colour) / attenuation distance).
    transmission: f32,
    ior: f32,
    absorption: float3,
    base_tex: gpu::Texture2D<float4>,
    mr_tex: gpu::Texture2D<float4>,
    normal_tex: gpu::Texture2D<float4>,
    occlusion_tex: gpu::Texture2D<float4>,
    emissive_tex: gpu::Texture2D<float4>,
}

struct Prim { first_index: u32, index_count: u32, material: u32, flags: u32 }

struct Instance { world: gpu::mat4, normal: gpu::mat4, prim: u32, center: float3 }

// A material as the file gives it: texture indices into the textures the
// caller makes, -1 for none.
struct MaterialDesc {
    base: float4,
    emissive: float3,
    metallic: f32,
    roughness: f32,
    normal_scale: f32,
    occlusion: f32,
    alpha_cutoff: f32,
    flags: u32,
    transmission: f32,
    ior: f32,
    absorption: float3,
    tex: i64[5],        // base colour, metallic-roughness, normal, occlusion, emissive
}

// An image's bytes in the blob, and whether a material reads it as colour
// (sRGB) or as data, or both.
struct ImageRef { at: i64, len: i64, color: bool, data: bool }

// The model's bounds, from its instances' vertices.
struct Bounds { lo: float3, hi: float3 }

fn nth(n: json::Node?, i: i64) -> json::Node? {
    guard n else { return null; }
    var c = n.first;
    var k: i64 = 0;
    while c {
        if k == i { return c; }
        c .= c.next;
        k++;
    }
    null
}

fn count(n: json::Node?) -> i64 {
    guard n else { return 0; }
    match n.val {
        Arr a => { return a.count; },
        Obj o => { return o.count; },
        _ => { return 0; },
    }
}

fn int_of(n: json::Node?, fallback: i64) -> i64 {
    let v, ok = json::as_i64(n);
    if ok { v } else { fallback }
}

fn num_of(n: json::Node?, fallback: f64) -> f32 {
    let v, ok = json::as_f64(n);
    (if ok { v } else { fallback }) as f32
}

fn nums(n: json::Node?, out: f32[:]) -> bool {
    guard n else { return false; }
    for i in out.len { out[i] = num_of(nth(n, i), 0.0); }
    true
}

// f32s from little-endian bytes, through the image format from_bytes
// reads: a LEB128 byte count, then the elements.
fn floats(bytes: const u8[:]) -> f32[>..] {
    var image: u8[>..] = [];
    var n = bytes.len as u64;
    loop {
        let low = (n & 127) as u8;
        n >>= 7;
        if n == 0 {
            image.push(low);
            break;
        }
        image.push(low | 128);
    }
    image.append(bytes);
    let got, ok = from_bytes<f32[>..]>(image);
    if ok { got } else { [] }
}

// An accessor's elements as floats, `comps` of each (normalized integers
// as 0..1), or its indices.
struct View { at: i64, stride: i64, count: i64, comps: i64, ctype: i64, normalized: bool }

fn comps_of(t: const u8[:]) -> i64 {
    if t == "SCALAR" { return 1; }
    if t == "VEC2" { return 2; }
    if t == "VEC3" { return 3; }
    if t == "VEC4" { return 4; }
    if t == "MAT4" { return 16; }
    0
}

fn csize(ctype: i64) -> i64 {
    if ctype == 5120 || ctype == 5121 { return 1; }
    if ctype == 5122 || ctype == 5123 { return 2; }
    4
}

fn view_of(root: json::Node?, bufat: i64[:], acc: i64) -> View {
    let a = nth(json::member(root, "accessors"), acc);
    let bv = nth(json::member(root, "bufferViews"), int_of(json::member(a, "bufferView"), -1));
    let ctype = int_of(json::member(a, "componentType"), 5126);
    let tname, _ = json::as_string(json::member(a, "type"));
    let comps = comps_of(tname);
    let norm, _ = json::as_bool(json::member(a, "normalized"));
    let buffer = int_of(json::member(bv, "buffer"), 0);
    let at = bufat[buffer] + int_of(json::member(bv, "byteOffset"), 0) + int_of(json::member(a, "byteOffset"), 0);
    let stride = int_of(json::member(bv, "byteStride"), comps * csize(ctype));
    View { at: at, stride: stride, count: int_of(json::member(a, "count"), 0), comps: comps, ctype: ctype,
           normalized: norm }
}

// Component c of element i, as a number.
fn read(blob: const u8[:], v: View, i: i64, c: i64) -> f32 {
    let p = v.at + i * v.stride + c * csize(v.ctype);
    if v.ctype == 5126 {
        let f = floats(blob[p..p + 4]);
        return f[0];
    }
    let u: u32 = if v.ctype == 5121 || v.ctype == 5120 { blob[p] }
                 else if v.ctype == 5123 || v.ctype == 5122 { blob[p] | ((blob[p + 1] as u32) << 8) }
                 else { blob[p] | ((blob[p + 1] as u32) << 8) | ((blob[p + 2] as u32) << 16) | ((blob[p + 3] as u32) << 24) };
    if !v.normalized { return u; }
    if v.ctype == 5121 { return u / 255.0; }
    if v.ctype == 5123 { return u / 65535.0; }
    u
}

// Every element's components at once, for float data packed or not.
fn read_all(blob: const u8[:], v: View, out: f32[>..]&) {
    if v.ctype == 5126 && v.stride == v.comps * 4 {
        out.append(floats(blob[v.at..v.at + v.count * v.stride]));
        return;
    }
    if v.ctype == 5126 {
        var packed: u8[>..] = [];
        for i in v.count { packed.append(blob[v.at + i * v.stride..v.at + i * v.stride + v.comps * 4]); }
        out.append(floats(packed));
        return;
    }
    for i in v.count { for c in v.comps { out.push(read(blob, v, i, c)); } }
}

fn quat_mat(q: float4) -> gpu::mat4 {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    gpu::mat4 { c0: float4 { 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0 },
                c1: float4 { 2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0 },
                c2: float4 { 2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0 },
                c3: float4 { 0.0, 0.0, 0.0, 1.0 } }
}

// The inverse transpose of a transform's upper 3x3, for normals.
fn normal_matrix(m: gpu::mat4) -> gpu::mat4 {
    let a = xyz(m.c0); let b = xyz(m.c1); let c = xyz(m.c2);
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let det = dot(a, r0);
    let k = if abs(det) > 1e-20 { 1.0 / det } else { 1.0 };
    gpu::mat4 { c0: float4 { r0.x * k, r0.y * k, r0.z * k, 0.0 }, c1: float4 { r1.x * k, r1.y * k, r1.z * k, 0.0 },
                c2: float4 { r2.x * k, r2.y * k, r2.z * k, 0.0 }, c3: float4 { 0.0, 0.0, 0.0, 1.0 } }
}

fn local_matrix(node: json::Node?) -> gpu::mat4 {
    var m: f32[16] = [0.0; 16];
    if nums(json::member(node, "matrix"), m) {
        return gpu::mat4 { c0: float4 { m[0], m[1], m[2], m[3] }, c1: float4 { m[4], m[5], m[6], m[7] },
                           c2: float4 { m[8], m[9], m[10], m[11] }, c3: float4 { m[12], m[13], m[14], m[15] } };
    }
    var t: f32[3] = [0.0; 3];
    var r: f32[4] = [0.0, 0.0, 0.0, 1.0];
    var s: f32[3] = [1.0; 3];
    nums(json::member(node, "translation"), t);
    nums(json::member(node, "rotation"), r);
    nums(json::member(node, "scale"), s);
    gpu::mul(gpu::translate(float3 { t[0], t[1], t[2] }),
             gpu::mul(quat_mat(float4 { r[0], r[1], r[2], r[3] }), gpu::scale(float3 { s[0], s[1], s[2] })))
}

// The file's bytes: the GLB's JSON and binary chunks, or the .gltf's text
// with its buffers read after it.
fn dir_of(path: const u8[:]) -> u8[>..] {
    var cut: i64 = 0;
    for i in path.len { if path[i] == '/' { cut = i + 1; } }
    str(path[0..cut])
}

fn fetch(uri: const u8[:], dir: const u8[:], out: u8[>..]&) -> bool {
    if uri.len > 5 && uri[0..5] == "data:" {
        var comma: i64 = 0;
        for i in uri.len { if uri[i] == ',' && comma == 0 { comma = i; } }
        return base64::decode(out, uri[comma + 1..uri.len]);
    }
    read_file(str(dir, uri), out)
}

// A model, loaded into the arrays given; "" or why it could not be.
fn load(path: const u8[:], blob: u8[>..]&, verts: Vertex[>..]&, indices: u32[>..]&, prims: Prim[>..]&,
        materials: MaterialDesc[>..]&, images: ImageRef[>..]&, textures: i64[>..]&,
        instances: Instance[>..]&) -> u8[>..] {
    var file: u8[>..] = [];
    guard read_file(path, file) else { return str("cannot read ", path); }
    let dir = dir_of(path);
    var text: u8[>..] = [];
    var bufat: i64[>..] = [];
    if file.len >= 12 && file[0..4] == "glTF" {
        let jlen = (file[12] | (file[13] as u32 << 8) | (file[14] as u32 << 16) | (file[15] as u32 << 24)) as i64;
        text.append(file[20..20 + jlen]);
        let b = 20 + jlen;
        if b + 8 <= file.len {
            bufat.push(blob.len);
            let blen = (file[b] | (file[b + 1] as u32 << 8) | (file[b + 2] as u32 << 16) | (file[b + 3] as u32 << 24)) as i64;
            blob.append(file[b + 8..b + 8 + blen]);
        }
    } else {
        text.append(file);
    }
    var doc: json::Node[>..] = [];
    var decoded: u8[>..] = [];
    let root, err = json::parse(text, doc, decoded);
    guard root else { return err; }

    // A .gltf's buffers, and a .glb's beyond its own.
    let buffers = json::member(root, "buffers");
    for i in bufat.len..count(buffers) {
        let u, ok = json::as_string(json::member(nth(buffers, i), "uri"));
        let uri = str(u);       // its own, as reading grows the blob
        bufat.push(blob.len);
        if ok && !fetch(uri, dir, blob) { return str("cannot read buffer ", uri); }
    }

    // Images, their bytes after the buffers'.
    let jimages = json::member(root, "images");
    for i in count(jimages) {
        let im = nth(jimages, i);
        let at = blob.len;
        let bvi = int_of(json::member(im, "bufferView"), -1);
        if bvi >= 0 {
            let bv = nth(json::member(root, "bufferViews"), bvi);
            let s = bufat[int_of(json::member(bv, "buffer"), 0)] + int_of(json::member(bv, "byteOffset"), 0);
            let n = int_of(json::member(bv, "byteLength"), 0);
            var copy: u8[>..] = [];
            copy.append(blob[s..s + n]);
            blob.append(copy);
        } else {
            let u, _ = json::as_string(json::member(im, "uri"));
            let uri = str(u);
            if !fetch(uri, dir, blob) { return str("cannot read image ", uri); }
        }
        images.push(ImageRef { at: at, len: blob.len - at, color: false, data: false });
    }
    let jtextures = json::member(root, "textures");
    for i in count(jtextures) { textures.push(int_of(json::member(nth(jtextures, i), "source"), -1)); }

    let jmaterials = json::member(root, "materials");
    for i in count(jmaterials) {
        let m = nth(jmaterials, i);
        let pbr = json::member(m, "pbrMetallicRoughness");
        var d = MaterialDesc { base: float4_1, emissive: float3_0, metallic: num_of(json::member(pbr, "metallicFactor"), 1.0),
                               roughness: num_of(json::member(pbr, "roughnessFactor"), 1.0),
                               normal_scale: num_of(json::member(json::member(m, "normalTexture"), "scale"), 1.0),
                               occlusion: num_of(json::member(json::member(m, "occlusionTexture"), "strength"), 1.0),
                               alpha_cutoff: num_of(json::member(m, "alphaCutoff"), 0.5), flags: 0,
                               transmission: 0.0, ior: 1.5, absorption: float3_0, tex: [-1; 5] };
        let ext = json::member(m, "extensions");
        d.transmission = num_of(json::member(json::member(ext, "KHR_materials_transmission"), "transmissionFactor"),
                                0.0);
        d.ior = num_of(json::member(json::member(ext, "KHR_materials_ior"), "ior"), 1.5);
        let vol = json::member(ext, "KHR_materials_volume");
        var ac: f32[3] = [1.0; 3];
        let dist = num_of(json::member(vol, "attenuationDistance"), 0.0);
        if nums(json::member(vol, "attenuationColor"), ac) && dist > 0.0 {
            d.absorption = float3 { -log(max(ac[0], 1e-4)) / dist, -log(max(ac[1], 1e-4)) / dist,
                                    -log(max(ac[2], 1e-4)) / dist };
        }
        var c: f32[4] = [1.0; 4];
        if nums(json::member(pbr, "baseColorFactor"), c) { d.base = float4 { c[0], c[1], c[2], c[3] }; }
        var e: f32[3] = [0.0; 3];
        if nums(json::member(m, "emissiveFactor"), e) { d.emissive = float3 { e[0], e[1], e[2] }; }
        let sided, _ = json::as_bool(json::member(m, "doubleSided"));
        if sided { d.flags |= DOUBLE_SIDED; }
        let mode, _ = json::as_string(json::member(m, "alphaMode"));
        if mode == "MASK" { d.flags |= ALPHA_MASK; }
        if mode == "BLEND" { d.flags |= ALPHA_BLEND; }
        let refs = [json::member(pbr, "baseColorTexture"), json::member(pbr, "metallicRoughnessTexture"),
                    json::member(m, "normalTexture"), json::member(m, "occlusionTexture"),
                    json::member(m, "emissiveTexture")];
        for k in 5 {
            let t = int_of(json::member(refs[k], "index"), -1);
            d.tex[k] = t;
            if t >= 0 && t < textures.len && textures[t] >= 0 {
                let color = k == 0 || k == 4;
                if color { images[textures[t]].color = true; } else { images[textures[t]].data = true; }
            }
        }
        materials.push(d);
    }
    if materials.len == 0 {
        materials.push(MaterialDesc { base: float4_1, emissive: float3_0, metallic: 1.0, roughness: 1.0,
                                      normal_scale: 1.0, occlusion: 1.0, alpha_cutoff: 0.5, flags: 0,
                                      transmission: 0.0, ior: 1.5, absorption: float3_0, tex: [-1; 5] });
    }

    // Meshes: their primitives' first entries in `prims`.
    let jmeshes = json::member(root, "meshes");
    var meshfirst: i64[>..] = [];
    var meshcount: i64[>..] = [];
    for mi in count(jmeshes) {
        let jprims = json::member(nth(jmeshes, mi), "primitives");
        meshfirst.push(prims.len);
        var made: i64 = 0;
        for pi in count(jprims) {
            let p = nth(jprims, pi);
            if int_of(json::member(p, "mode"), 4) != 4 { continue; }
            let attrs = json::member(p, "attributes");
            let pos = int_of(json::member(attrs, "POSITION"), -1);
            if pos < 0 { continue; }
            let base = verts.len;
            var pv: f32[>..] = [];
            let vp = view_of(root, bufat, pos);
            read_all(blob, vp, pv);
            var nv: f32[>..] = [];
            let ni = int_of(json::member(attrs, "NORMAL"), -1);
            if ni >= 0 { read_all(blob, view_of(root, bufat, ni), nv); }
            var tv: f32[>..] = [];
            let ti = int_of(json::member(attrs, "TANGENT"), -1);
            if ti >= 0 { read_all(blob, view_of(root, bufat, ti), tv); }
            var uv: f32[>..] = [];
            let ui = int_of(json::member(attrs, "TEXCOORD_0"), -1);
            if ui >= 0 { read_all(blob, view_of(root, bufat, ui), uv); }
            for i in vp.count {
                verts.push(Vertex {
                    pos: float3 { pv[3 * i], pv[3 * i + 1], pv[3 * i + 2] },
                    normal: if nv.len > 0 { float3 { nv[3 * i], nv[3 * i + 1], nv[3 * i + 2] } } else { float3_0 },
                    tangent: if tv.len > 0 { float4 { tv[4 * i], tv[4 * i + 1], tv[4 * i + 2], tv[4 * i + 3] } }
                             else { float4_0 },
                    uv: if uv.len > 0 { float2 { uv[2 * i], uv[2 * i + 1] } } else { float2_0 } });
            }
            let first = indices.len;
            let ii = int_of(json::member(p, "indices"), -1);
            if ii >= 0 {
                let iv = view_of(root, bufat, ii);
                for i in iv.count { indices.push(base as u32 + read(blob, iv, i, 0) as u32); }
            } else {
                for i in vp.count { indices.push((base + i) as u32); }
            }
            // Flat normals where the file has none.
            if nv.len == 0 {
                var t = first;
                while t + 2 < indices.len {
                    let a = verts[indices[t]].pos;
                    let b = verts[indices[t + 1]].pos;
                    let c = verts[indices[t + 2]].pos;
                    let n = normalize(cross(b - a, c - a));
                    for k in 3 { verts[indices[t + k]].normal = n; }
                    t += 3;
                }
            }
            let mat = clamp(int_of(json::member(p, "material"), 0), 0, materials.len - 1);
            var flags = materials[mat].flags;
            if tv.len > 0 { flags |= HAS_TANGENTS; }
            prims.push(Prim { first_index: first as u32, index_count: (indices.len - first) as u32,
                              material: mat as u32, flags: flags });
            made++;
        }
        meshcount.push(made);
    }

    // The default scene's nodes, depth first, with their world transforms.
    let jnodes = json::member(root, "nodes");
    let scene = nth(json::member(root, "scenes"), int_of(json::member(root, "scene"), 0));
    var stack: i64[>..] = [];
    var mats: gpu::mat4[>..] = [];
    let roots = json::member(scene, "nodes");
    if roots {
        for i in count(roots) {
            stack.push(int_of(nth(roots, i), 0));
            mats.push(gpu::mat4_identity);
        }
    } else {
        for i in count(jnodes) { stack.push(i); mats.push(gpu::mat4_identity); }
    }
    while stack.len > 0 {
        let ni = stack[stack.len - 1];
        let parent = mats[mats.len - 1];
        stack.pop();
        mats.pop();
        let node = nth(jnodes, ni);
        let world = gpu::mul(parent, local_matrix(node));
        let mesh = int_of(json::member(node, "mesh"), -1);
        if mesh >= 0 && mesh < meshfirst.len {
            for k in meshcount[mesh] {
                let pr = meshfirst[mesh] + k;
                instances.push(Instance { world: world, normal: normal_matrix(world), prim: pr as u32,
                                          center: float3_0 });
            }
        }
        let children = json::member(node, "children");
        for c in count(children) {
            stack.push(int_of(nth(children, c), 0));
            mats.push(world);
        }
    }
    str("")
}

// The bounds of the instances' vertices in world space, and each
// instance's centre, which sorting what is blended needs.
fn bounds(verts: Vertex[:], indices: u32[:], prims: Prim[:], instances: Instance[>..]&) -> Bounds {
    var lo = float3 { 1e30, 1e30, 1e30 };
    var hi = float3 { -1e30, -1e30, -1e30 };
    for &inst in instances {
        let p = prims[inst.prim];
        var ilo = float3 { 1e30, 1e30, 1e30 };
        var ihi = float3 { -1e30, -1e30, -1e30 };
        for k in p.index_count {
            let v = verts[indices[p.first_index + k]].pos;
            let w = xyz(gpu::mul(inst.world, float4 { v.x, v.y, v.z, 1.0 }));
            ilo = float3 { min(ilo.x, w.x), min(ilo.y, w.y), min(ilo.z, w.z) };
            ihi = float3 { max(ihi.x, w.x), max(ihi.y, w.y), max(ihi.z, w.z) };
        }
        inst.center = (ilo + ihi) * 0.5;
        lo = float3 { min(lo.x, ilo.x), min(lo.y, ilo.y), min(lo.z, ilo.z) };
        hi = float3 { max(hi.x, ihi.x), max(hi.y, ihi.y), max(hi.z, ihi.z) };
    }
    Bounds { lo: lo, hi: hi }
}
