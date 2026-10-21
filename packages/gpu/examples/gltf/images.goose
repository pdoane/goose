// Images as textures: a JPEG's entropy decoded on the CPU (jpeg.goose) and
// the rest on the GPU, a thread a pixel, into a buffer copied into the
// texture, whose mips are then generated; a PNG through the gpu package.
// The copy is what lets an sRGB texture take the pixels, which no shader
// may write.

namespace images;

import gpu from "../..";
import .jpeg;
import std;

struct Pixels {
    coefs: gpu::StructuredBuffer<i32>,
    out: gpu::RWStructuredBuffer<u32>,
    layout: jpeg::Layout,
}

compute(8, 8) fn jpeg_pixels(p: Pixels) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.layout.width || id.y >= p.layout.height { return; }
    gpu::store(p.out, id.y * p.layout.width + id.x, jpeg::pixel(p.coefs, p.layout, id.x, id.y));
}

// A decoded JPEG's pixels, in a buffer of RGBA8 the image's size: work
// recorded in a compute pass of its own, the buffer readable by copies
// after it. Its coefficients' buffer is released once the work is done.
struct Decoded { pixels: gpu::Buffer, coefs: gpu::Buffer, width: i64, height: i64 }

fn jpeg_pixels_on_gpu(img: jpeg::Image&) -> Decoded {
    let coefs = gpu::buffer(gpu::BUFFER_READ, bytes_of(img.coefs));
    let out = gpu::buffer(gpu::BUFFER_WRITE, img.width * img.height * 4);
    let l = jpeg::layout(img);
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(jpeg_pixels));
    gpu::push_constants(Pixels { coefs: gpu::structured<i32>(coefs), out: gpu::rw_structured<u32>(out),
                                 layout: l });
    gpu::dispatch((img.width + 7) / 8, (img.height + 7) / 8, 1);
    gpu::end_compute();
    gpu::barrier(out, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_COPY_SRC);
    Decoded { pixels: out, coefs: coefs, width: img.width, height: img.height }
}

// An image file's bytes (JPEG or PNG) as a texture with its full chain of
// mips, sRGB where its texels are colours; Texture { 0 } and the reason
// when it cannot be read.
fn texture(data: const u8[:], srgb: bool, err: u8[>..]&) -> gpu::Texture {
    if data.len >= 2 && data[0] == 0xff && data[1] == 0xd8 {
        var img = jpeg::image();
        let e = jpeg::decode_file(data, img);
        if e.len > 0 {
            err.append(e);
            return gpu::Texture { 0 };
        }
        let d = jpeg_pixels_on_gpu(img);
        let t = gpu::texture(gpu::TextureDesc { format: if srgb { gpu::RGBA8_SRGB } else { gpu::RGBA8 },
                                                width: d.width as i32, height: d.height as i32, mips: 0, .. });
        gpu::barrier(t, gpu::ACCESS_NONE, gpu::ACCESS_COPY_DST);
        gpu::copy_to_texture(d.pixels, 0, t);
        gpu::barrier(t, gpu::ACCESS_COPY_DST, gpu::ACCESS_SHADER_READ);
        gpu::generate_mips(t);
        gpu::release(d.pixels);
        gpu::release(d.coefs);
        return t;
    }
    let t = gpu::image_texture(data, gpu::LOAD_MIPS | (if srgb { gpu::LOAD_SRGB } else { 0 }));
    if t.index == 0 { err.append(gpu::error()); }
    t
}
