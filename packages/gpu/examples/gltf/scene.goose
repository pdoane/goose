// A glTF model on the GPU: its vertices, indices, materials and instances
// in buffers that shaders read by handle, its images as textures, and what
// the CPU needs to draw it. Materials are a table of factors and texture
// handles, a bindless one: a draw names its material by index, and a
// texture a material lacks is a 1x1 default that leaves the factor as it
// is.

namespace scene;

import gpu from "../..";
import .gltf;
import .images;
import std;

// The buffers shaders read the model through.
struct Refs {
    verts: gpu::StructuredBuffer<gltf::Vertex>,
    indices: gpu::StructuredBuffer<u32>,
    materials: gpu::StructuredBuffer<gltf::Material>,
    prims: gpu::StructuredBuffer<gltf::Prim>,
    instances: gpu::StructuredBuffer<gltf::Instance>,
}

// A draw: an instance of a primitive, with what choosing its pipeline and
// order needs.
struct Draw {
    instance: u32,
    first_index: u32,
    index_count: u32,
    material: u32,
    flags: u32,
    center: float3,
    world: gpu::mat4,
}

struct Model {
    refs: Refs,
    vertex_buffer: gpu::Buffer,
    index_buffer: gpu::Buffer,
    index_count: i64,
    triangles: i64,
    lo: float3,
    hi: float3,
    draws: Draw[>..],
}

fn solid(r: u8, g: u8, b: u8, srgb: bool) -> gpu::Texture {
    let px: u8[4] = [r, g, b, 255];
    gpu::texture(gpu::TextureDesc { format: if srgb { gpu::RGBA8_SRGB } else { gpu::RGBA8 }, width: 1,
                                    height: 1, .. }, px)
}

// The model at `path`, uploaded; "" or why it could not be.
fn load(path: const u8[:], out: Model&) -> u8[>..] {
    var blob: u8[>..] = [];
    var verts: gltf::Vertex[>..] = [];
    var indices: u32[>..] = [];
    var prims: gltf::Prim[>..] = [];
    var descs: gltf::MaterialDesc[>..] = [];
    var imgs: gltf::ImageRef[>..] = [];
    var textures: i64[>..] = [];
    var insts: gltf::Instance[>..] = [];
    let e = gltf::load(path, blob, verts, indices, prims, descs, imgs, textures, insts);
    if e.len > 0 { return e; }
    guard insts.len > 0 else { return str(path, ": nothing to draw"); }
    let b = gltf::bounds(verts, indices, prims, insts);
    out.lo = b.lo;
    out.hi = b.hi;

    // Each image as the materials read it: colour in sRGB, data linearly.
    var color: gpu::Texture[>..] = [];
    var data: gpu::Texture[>..] = [];
    for im in imgs {
        var err: u8[>..] = [];
        let bytes = blob[im.at..im.at + im.len];
        color.push(if im.color { images::texture(bytes, true, err) } else { gpu::Texture { 0 } });
        data.push(if im.data { images::texture(bytes, false, err) } else { gpu::Texture { 0 } });
        if err.len > 0 { return str(path, ": an image: ", err); }
    }
    let white_srgb = solid(255, 255, 255, true);
    let white = solid(255, 255, 255, false);
    let flat = solid(128, 128, 255, false);
    var mats: gltf::Material[>..] = [];
    for d in descs {
        let defaults = [white_srgb, white, flat, white, white_srgb];
        var t: gpu::Texture[5] = defaults;
        for k in 5 {
            let ti = d.tex[k];
            if ti >= 0 && ti < textures.len && textures[ti] >= 0 && textures[ti] < imgs.len {
                let img = textures[ti];
                let made = if k == 0 || k == 4 { color[img] } else { data[img] };
                if made.index != 0 { t[k] = made; }
            }
        }
        mats.push(gltf::Material { base: d.base, emissive: d.emissive, metallic: d.metallic,
                                   roughness: d.roughness, normal_scale: d.normal_scale, occlusion: d.occlusion,
                                   alpha_cutoff: d.alpha_cutoff, flags: d.flags, transmission: d.transmission,
                                   ior: d.ior, absorption: d.absorption,
                                   base_tex: gpu::sampled<float4>(t[0]), mr_tex: gpu::sampled<float4>(t[1]),
                                   normal_tex: gpu::sampled<float4>(t[2]),
                                   occlusion_tex: gpu::sampled<float4>(t[3]),
                                   emissive_tex: gpu::sampled<float4>(t[4]) });
    }

    let usage = gpu::BUFFER_READ | gpu::BUFFER_ACCEL_INPUT;
    out.vertex_buffer = gpu::buffer(usage, bytes_of(verts));
    out.index_buffer = gpu::buffer(usage | gpu::BUFFER_INDEX, bytes_of(indices));
    out.index_count = indices.len;
    out.triangles = indices.len / 3;
    out.refs = Refs { verts: gpu::structured<gltf::Vertex>(out.vertex_buffer),
                      indices: gpu::structured<u32>(out.index_buffer),
                      materials: gpu::structured<gltf::Material>(gpu::buffer(gpu::BUFFER_READ, bytes_of(mats))),
                      prims: gpu::structured<gltf::Prim>(gpu::buffer(gpu::BUFFER_READ, bytes_of(prims))),
                      instances: gpu::structured<gltf::Instance>(gpu::buffer(gpu::BUFFER_READ, bytes_of(insts))) };
    for i in insts.len {
        let p = prims[insts[i].prim];
        out.draws.push(Draw { instance: i as u32, first_index: p.first_index, index_count: p.index_count,
                              material: p.material, flags: p.flags, center: insts[i].center,
                              world: insts[i].world });
    }
    str("")
}
