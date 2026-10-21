// Checks the JPEG decoder: a file decoded on the GPU (images.goose), its
// pixels compared with the CPU's running the same function on a sample of
// them, and with a reference decoding of it as a PNG when one is given.
//
//     goose jpegcheck.goose -- image.jpg [reference.png] [out.png]

import gpu from "../..";
import .jpeg;
import .images;
import std;
import os;
import math;

fn main() {
    guard arg_count() > 1 else { abort("usage: jpegcheck.goose -- image.jpg [reference.png] [out.png]"); }
    var path: u8[>..] = [];
    arg(1, path);
    var data: u8[>..] = [];
    guard read_file(path, data) else { abort(str("cannot read ", path)); }
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }

    let t0 = clock();
    var img = jpeg::image();
    let e = jpeg::decode_file(data, img);
    guard e.len == 0 else { abort(str(path, ": ", e)); }
    let cpums = (clock() - t0) * 1000.0;
    let t1 = clock();
    let d = images::jpeg_pixels_on_gpu(img);
    let r = gpu::read_buffer(d.pixels, 0, img.width * img.height * 4);
    gpu::wait(gpu::submit());
    let gpums = (clock() - t1) * 1000.0;
    var px: u32[>..] = [];
    guard gpu::values(r, px) else { abort("a read back is not ready"); }

    // The CPU's pixels, a sample spread over the image.
    let l = jpeg::layout(img);
    var maxcpu: i32 = 0;
    let step = max(img.width * img.height / 4099, 1);
    var k: i64 = 0;
    while k < img.width * img.height {
        let mine = jpeg::pixel(img.coefs, l, (k % img.width) as u32, (k / img.width) as u32);
        for s in 3 {
            let a = ((mine >> (8 * s)) & 255) as i32;
            let b = ((px[k] >> (8 * s)) & 255) as i32;
            maxcpu = max(maxcpu, abs(a - b));
        }
        k += step;
    }
    var report = str(path, ": ", img.width, "x", img.height, ", ", img.ncomp, " components, ",
                     img.hmax, "x", img.vmax, " sampling; entropy decode ", cpums, " ms, GPU ", gpums,
                     " ms; CPU pixels within ", maxcpu);

    if arg_count() > 2 {
        var refpath: u8[>..] = [];
        arg(2, refpath);
        let rt = gpu::load_texture(refpath, 0);
        guard rt.index != 0 else { abort(str("cannot load ", refpath, ": ", gpu::error())); }
        let rr = gpu::read_texture(rt);
        gpu::wait(gpu::submit());
        var want: u32[>..] = [];
        gpu::values(rr, want);
        var sum = 0.0;
        var worst: i32 = 0;
        var at: i64 = 0;
        var over: i64 = 0;
        for i in px.len {
            for s in 3 {
                let a = ((want[i] >> (8 * s)) & 255) as i32;
                let b = ((px[i] >> (8 * s)) & 255) as i32;
                sum += (a - b) * (a - b);
                if abs(a - b) > worst { worst = abs(a - b); at = i; }
                if abs(a - b) > 8 { over++; }
            }
        }
        let psnr = 10.0 * log10(255.0 * 255.0 / max(sum / (px.len * 3), 1e-9));
        report.append(str("; against the reference ", psnr, " dB, worst ", worst, " at ", at % img.width,
                          ",", at / img.width, " (", px[at] & 0xffffff, " not ", want[at] & 0xffffff,
                          "), ", over, " channels off by more than 8"));
    }
    if arg_count() > 3 {
        var outpath: u8[>..] = [];
        arg(3, outpath);
        let t = gpu::texture(gpu::TextureDesc { width: img.width as i32, height: img.height as i32, .. });
        gpu::barrier(t, gpu::ACCESS_NONE, gpu::ACCESS_COPY_DST);
        gpu::copy_to_texture(d.pixels, 0, t);
        gpu::barrier(t, gpu::ACCESS_COPY_DST, gpu::ACCESS_SHADER_READ);
        gpu::save_png(t, outpath);
    }
    print(report);
    gpu::close();
}
