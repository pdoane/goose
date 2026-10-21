// Mips and image files (docs/design/gpu.md §3): a texture made with its
// full chain has its mips filled from mip 0 by generate_mips; a texture
// saved as a PNG, once its submission completes, loads back with the same
// pixels, and LOAD_MIPS fills the loaded one's mips in its upload.
import gpu from "..";
import std;
import os;

struct Mips {
    made: gpu::Texture2D<float4>,
    loaded: gpu::Texture2D<float4>,
    dst: gpu::RWStructuredBuffer<float4>,
}

// Texels of each texture's mips: the made one's mip 1 at its left and
// right edges and its 1x1 mip 3, then the loaded one's 1x1 mip 2.
compute(1) fn probe(p: Mips) {
    gpu::store(p.dst, 0, gpu::texel(p.made, int2 { 0, 0 }, 1));
    gpu::store(p.dst, 1, gpu::texel(p.made, int2 { 3, 0 }, 1));
    gpu::store(p.dst, 2, gpu::texel(p.made, int2_0, 3));
    gpu::store(p.dst, 3, gpu::texel(p.loaded, int2_0, 2));
}

fn near(a: float4, b: float4) -> bool {
    let d = 0.02;
    abs(a.x - b.x) < d && abs(a.y - b.y) < d && abs(a.z - b.z) < d && abs(a.w - b.w) < d
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    var wrong = 0;

    // 8x8, its left half red and its right half blue.
    var px: u8[>..] = [];
    for y in 8 {
        for x in 8 {
            if x < 4 { px.append([255, 0, 0, 255]); } else { px.append([0, 0, 255, 255]); }
        }
    }
    let made = gpu::texture(gpu::TextureDesc { width: 8, height: 8, mips: 0, .. }, px);
    if gpu::texture_info(made).mips != 4 {
        print("an 8x8 texture's full chain is ", gpu::texture_info(made).mips, " mips, not 4");
        wrong++;
    }
    gpu::generate_mips(made);

    // 4x2: a row of red over a row of green, its alpha varying, through a
    // PNG file and back.
    var img: u8[>..] = [];
    for x in 4 { img.append([255, 0, 0, (60 * x + 15) as u8]); }
    for x in 4 { img.append([0, 255, 0, 255]); }
    let small = gpu::texture(gpu::TextureDesc { width: 4, height: 2, .. }, img);
    let path = "gpu_mips_test.png";
    guard gpu::save_png(small, path) else { abort(str("gpu: save_png: ", gpu::error())); }
    gpu::wait(gpu::submit());
    guard file_exists(path) else { abort("gpu: save_png wrote no file once its fence passed"); }
    let loaded = gpu::load_texture(path, gpu::LOAD_MIPS);
    var file: u8[>..] = [];
    read_file(path, file);
    let frommem = gpu::image_texture(file, 0);
    delete_file(path);
    guard loaded.index != 0 else { abort(str("gpu: load_texture: ", gpu::error())); }
    if gpu::texture_info(loaded).mips != 3 {
        print("a loaded 4x2 texture's full chain is ", gpu::texture_info(loaded).mips,
              " mips, not 3");
        wrong++;
    }

    let dst = gpu::buffer(gpu::BUFFER_WRITE, 4 * 16);
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(probe));
    gpu::push_constants(Mips { made: gpu::sampled<float4>(made), loaded: gpu::sampled<float4>(loaded),
                               dst: gpu::rw_structured<float4>(dst) });
    gpu::dispatch(1, 1, 1);
    gpu::end_compute();
    let texels = gpu::read_buffer(dst, 0, 4 * 16);
    let back = gpu::read_texture(loaded);
    let backmem = gpu::read_texture(frommem);
    gpu::wait(gpu::submit());

    var got: float4[>..] = [];
    guard gpu::values(texels, got) else { abort("gpu: a read back is not ready"); }
    let want = [float4 { 1.0, 0.0, 0.0, 1.0 }, float4 { 0.0, 0.0, 1.0, 1.0 },
                float4 { 0.5, 0.0, 0.5, 1.0 },
                // The loaded image's average: alpha 15, 75, 135, 195 over
                // four opaque texels.
                float4 { 0.5, 0.5, 0.0, (15.0 + 75.0 + 135.0 + 195.0 + 4.0 * 255.0) / 8.0 / 255.0 }];
    for i in 4 {
        if !near(got[i], want[i]) {
            print("texel ", i, ": ", got[i].x, " ", got[i].y, " ", got[i].z, " ", got[i].w,
                  " not ", want[i].x, " ", want[i].y, " ", want[i].z, " ", want[i].w);
            wrong++;
        }
    }
    var bytes: u8[>..] = [];
    guard gpu::bytes(back, bytes) else { abort("gpu: a read back is not ready"); }
    var memory: u8[>..] = [];
    guard gpu::bytes(backmem, memory) else { abort("gpu: a read back is not ready"); }
    for i in img.len {
        if memory[i] != img[i] {
            if wrong < 8 { print("byte ", i, " from memory: ", memory[i], " not ", img[i]); }
            wrong++;
        }
        if bytes[i] != img[i] {
            if wrong < 8 { print("loaded byte ", i, ": ", bytes[i], " not ", img[i]); }
            wrong++;
        }
    }
    print(gpu::driver(), ": mips and a PNG round trip, ", wrong, " wrong");
    gpu::close();
}
