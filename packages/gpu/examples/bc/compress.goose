// GPU texture compression: an image encoded to BC1, BC3, BC4, BC5 and BC7
// by a compute shader, a thread a block, with the encoders of bcn.goose,
// then copied into a compressed texture that the GPU's own decoder reads
// back. For each format the program reports the quality (PSNR against the
// original), the time, and two checks: the CPU, running the same Goose
// functions, encodes the same bits, and bcn.goose's decoder agrees with
// the GPU's.
//
//     goose compress.goose                 a test image of its own
//     goose compress.goose photo.png       a PNG, cropped to a multiple of 4
//     goose compress.goose -- --out dir    where the decoded PNGs go (bc_out)
//     goose compress.goose -- --cpu N      blocks the CPU checks (all: -1)

import gpu from "../..";
import .bcn;
import std;
import os;
import math;

struct Encode {
    pixels: gpu::StructuredBuffer<u32>,
    blocks: gpu::RWStructuredBuffer<uint2>,
    width: u32,
    height: u32,
    format: u32,
}

// A block a thread: its texels, encoded, stored as one or two uint2s.
compute(8, 8) fn encode(p: Encode) {
    let id = gpu::dispatch_thread_id();
    let bw = p.width / 4;
    if id.x >= bw || id.y >= p.height / 4 { return; }
    var px: int4[16] = [int4_0; 16];
    for i: u32 in 16 {
        let x = id.x * 4 + (i & 3);
        let y = id.y * 4 + (i >> 2);
        px[i] = bc::unpack(gpu::load(p.pixels, y * p.width + x));
    }
    let b = bc::encode(p.format, px);
    let at = id.y * bw + id.x;
    if bc::block_bytes(p.format) == 8 {
        gpu::store(p.blocks, at, uint2 { b.x, b.y });
    } else {
        gpu::store(p.blocks, 2 * at, uint2 { b.x, b.y });
        gpu::store(p.blocks, 2 * at + 1, uint2 { b.z, b.w });
    }
}

struct Expand { src: gpu::Texture2D<float4>, dst: gpu::RWTexture2D<float4>, width: u32, height: u32 }

// The compressed texture as the GPU decodes it, into an RGBA8 one.
compute(8, 8) fn expand(p: Expand) {
    let id = gpu::dispatch_thread_id();
    if id.x >= p.width || id.y >= p.height { return; }
    let xy = int2 { id.x as i32, id.y as i32 };
    gpu::store(p.dst, xy, gpu::texel(p.src, xy, 0));
}

// ---------------------------------------------------------------------------
// The test image: smooth gradients, hard edges, fine detail, noise, and an
// alpha that varies, a quadrant of each and some of each mixed.

fn pcg(x: u32) -> u32 {
    var h = x * 747796405 + 2891336453;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737;
    (h >> 22) ^ h
}

fn test_image(w: i64, h: i64) -> u32[>..] {
    var out: u32[>..] = [];
    for y in h {
        for x in w {
            let u = x as f64 / w as f64;
            let v = y as f64 / h as f64;
            var c = double4 { 0.0, 0.0, 0.0, 1.0 };
            if u < 0.5 && v < 0.5 {
                // A colour wheel's hues, darkening outward.
                let a = atan2(v - 0.25, u - 0.25);
                let r = sqrt((u - 0.25) * (u - 0.25) + (v - 0.25) * (v - 0.25)) * 4.0;
                c = double4 { 0.5 + 0.5 * cos(a), 0.5 + 0.5 * cos(a - 2.094), 0.5 + 0.5 * cos(a + 2.094),
                              1.0 } * (1.0 - min(r, 1.0) * 0.7);
                c.w = 1.0;
            } else if v < 0.5 {
                // Stripes at an angle and a checkerboard: hard edges.
                let s = sin((u * 3.0 + v * 2.0) * 60.0);
                let check = ((x / 16) + (y / 16)) % 2 == 0;
                c = if s > 0.0 { double4 { 0.9, 0.8, 0.1, 1.0 } } else { double4 { 0.1, 0.2, 0.7, 1.0 } };
                if check { c = c * 0.6; c.w = 1.0; }
            } else if u < 0.5 {
                // Noise over a gradient, and a soft alpha.
                let n = (pcg((y * w + x) as u32) & 255) / 255.0;
                c = double4 { u * 2.0, 0.5 * n + 0.25, 1.0 - u * 2.0, 1.0 };
                c.w = clamp(1.0 - sqrt((u - 0.25) * (u - 0.25) + (v - 0.75) * (v - 0.75)) * 4.0, 0.0,
                            1.0);
            } else {
                // A normal map's ripples, for BC5, with alpha cut out.
                let nx = sin(u * 40.0) * 0.5;
                let ny = cos(v * 30.0) * 0.5;
                c = double4 { nx * 0.5 + 0.5, ny * 0.5 + 0.5, 1.0, if (x / 8 + y / 8) % 3 == 0 { 0.0 } else { 1.0 } };
            }
            let q = clamp255(int4 { (c.x * 255.0 + 0.5) as! i32, (c.y * 255.0 + 0.5) as! i32,
                                    (c.z * 255.0 + 0.5) as! i32, (c.w * 255.0 + 0.5) as! i32 });
            out.push(bc::pack(q));
        }
    }
    out
}

fn clamp255(c: int4) -> int4 { bc::clamp255(c) }

// ---------------------------------------------------------------------------

struct Format { name: const u8[:], id: u32, gpu: i32, channels: i64 }

// PSNR of `got` against `want` over the channels a format keeps.
fn psnr(want: u32[:], got: u32[:], channels: i64) -> f64 {
    var sum = 0.0;
    for i in want.len {
        let a = bc::unpack(want[i]);
        let b = bc::unpack(got[i]);
        let d = a - b;
        sum += d.x * d.x;
        if channels > 1 { sum += d.y * d.y; }
        if channels > 2 { sum += d.z * d.z; }
        if channels > 3 { sum += d.w * d.w; }
    }
    let mse = sum / (want.len * channels);
    if mse == 0.0 { return 99.0; }
    10.0 * log10(255.0 * 255.0 / mse)
}

// The image's texels, read as the decoders give them: BC4 keeps red alone,
// BC5 red and green, BC1 and mode-1 BC7 opaque colour.
fn kept(c: int4, channels: i64) -> int4 {
    if channels == 1 { return int4 { c.x, 0, 0, 255 }; }
    if channels == 2 { return int4 { c.x, c.y, 0, 255 }; }
    if channels == 3 { return int4 { c.x, c.y, c.z, 255 }; }
    c
}

fn block_texels(pixels: u32[:], w: i64, bx: i64, by: i64) -> int4[16] {
    var px: int4[16] = [int4_0; 16];
    for i in 16 { px[i] = bc::unpack(pixels[(by * 4 + i / 4) * w + bx * 4 + i % 4]); }
    px
}

fn arg_value(name: const u8[:], fallback: const u8[:]) -> u8[>..] {
    for i in 1..arg_count() {
        var a: u8[>..] = [];
        arg(i, a);
        if a == name && i + 1 < arg_count() {
            var v: u8[>..] = [];
            arg(i + 1, v);
            return v;
        }
    }
    str(fallback)
}

// `s` in a column `width` wide.
fn col(s: const u8[:], width: i64) -> u8[>..] {
    var out = str(s);
    while out.len < width { out.push(' '); }
    out
}

fn fixed(v: f64, decimals: i64) -> u8[>..] {
    var out: u8[>..] = [];
    format_flt(out, v, decimals);
    out
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let outdir = arg_value("--out", "bc_out");
    let cpublocks, _ = parse_int(arg_value("--cpu", "4096"));
    make_dir(outdir);

    // The image: a PNG named on the command line, or the test image.
    var w: i64 = 512;
    var h: i64 = 512;
    var pixels: u32[>..] = [];
    var path: u8[>..] = [];
    if arg_count() > 1 { arg(1, path); }
    if path.len > 0 && path[0] != '-' {
        let t = gpu::load_texture(path, 0);
        guard t.index != 0 else { abort(str("cannot load ", path, ": ", gpu::error())); }
        let info = gpu::texture_info(t);
        let r = gpu::read_texture(t);
        gpu::wait(gpu::submit());
        var bytes: u8[>..] = [];
        gpu::bytes(r, bytes);
        w = info.width / 4 * 4;
        h = info.height / 4 * 4;
        for y in h {
            for x in w {
                let o = (y * info.width + x) * 4;
                pixels.push(bytes[o] | ((bytes[o + 1] as u32) << 8) |
                            ((bytes[o + 2] as u32) << 16) | ((bytes[o + 3] as u32) << 24));
            }
        }
        gpu::release(t);
    } else {
        pixels = test_image(w, h);
    }
    let bw = w / 4;
    let bh = h / 4;
    print(gpu::driver(), ": ", w, "x", h, ", ", bw * bh, " blocks");

    let original = gpu::texture(gpu::TextureDesc { width: w as i32, height: h as i32, .. },
                                bytes_of(pixels));
    gpu::save_png(original, str(outdir, "/original.png"));
    let src = gpu::buffer(gpu::BUFFER_READ, bytes_of(pixels));
    let enc = gpu::compute_pipeline(encode);
    let exp = gpu::compute_pipeline(expand);

    let formats = [Format { name: "BC1", id: bc::FORMAT_BC1, gpu: gpu::BC1, channels: 3 },
                   Format { name: "BC3", id: bc::FORMAT_BC3, gpu: gpu::BC3, channels: 4 },
                   Format { name: "BC4", id: bc::FORMAT_BC4, gpu: gpu::BC4, channels: 1 },
                   Format { name: "BC5", id: bc::FORMAT_BC5, gpu: gpu::BC5, channels: 2 },
                   Format { name: "BC7", id: bc::FORMAT_BC7, gpu: gpu::BC7, channels: 4 }];
    var failed = false;
    print(col("format", 8), col("PSNR dB", 10), col("GPU ms", 9), col("CPU us/block", 14),
          col("CPU = GPU blocks", 18), "decoder max diff");
    for f in formats {
        let bytes = bw * bh * bc::block_bytes(f.id) as i64;
        let blocks = gpu::buffer(gpu::BUFFER_WRITE, bytes);
        let push = Encode { pixels: gpu::structured<u32>(src), blocks: gpu::rw_structured<uint2>(blocks),
                            width: w as u32, height: h as u32, format: f.id };
        // Twice, the second timed: the first pays for the pipeline's
        // first use.
        var gpums = 0.0;
        for run in 2 {
            let t0 = clock();
            gpu::begin_compute();
            gpu::bind(enc);
            gpu::push_constants(push);
            gpu::dispatch((bw + 7) / 8, (bh + 7) / 8, 1);
            gpu::end_compute();
            gpu::wait(gpu::submit());
            gpums = (clock() - t0) * 1000.0;
        }

        // Into a compressed texture, and out through the GPU's decoder.
        let tex = gpu::texture(gpu::TextureDesc { format: f.gpu, width: w as i32, height: h as i32, .. });
        guard tex.index != 0 else { abort(str(f.name, ": ", gpu::error())); }
        let dst = gpu::texture(gpu::TextureDesc { usage: gpu::STORAGE, width: w as i32, height: h as i32, .. });
        gpu::barrier(blocks, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_COPY_SRC);
        gpu::barrier(tex, gpu::ACCESS_NONE, gpu::ACCESS_COPY_DST);
        gpu::copy_to_texture(blocks, 0, tex);
        gpu::barrier(tex, gpu::ACCESS_COPY_DST, gpu::ACCESS_SHADER_READ);
        gpu::barrier(dst, gpu::ACCESS_NONE, gpu::ACCESS_SHADER_WRITE);
        gpu::begin_compute();
        gpu::bind(exp);
        gpu::push_constants(Expand { src: gpu::sampled<float4>(tex), dst: gpu::storage<float4>(dst),
                                     width: w as u32, height: h as u32 });
        gpu::dispatch((w + 7) / 8, (h + 7) / 8, 1);
        gpu::end_compute();
        let rblocks = gpu::read_buffer(blocks, 0, bytes);
        let rdecoded = gpu::read_texture(dst);
        gpu::save_png(dst, str(outdir, "/", f.name, ".png"));
        gpu::wait(gpu::submit());
        var words: u32[>..] = [];
        guard gpu::values(rblocks, words) else { abort("a read back is not ready"); }
        var decoded: u32[>..] = [];
        guard gpu::values(rdecoded, decoded) else { abort("a read back is not ready"); }

        // The GPU's decode against the original, over the channels kept.
        var want: u32[>..] = [];
        for p in pixels { want.push(bc::pack(kept(bc::unpack(p), f.channels))); }
        let quality = psnr(want, decoded, f.channels);

        // bcn.goose's decoder against the GPU's, every block.
        let wpb = bc::block_bytes(f.id) as i64 / 4;
        var maxdiff: i32 = 0;
        for by in bh {
            for bx in bw {
                let at = (by * bw + bx) * wpb;
                let blk = if wpb == 2 { uint4 { words[at], words[at + 1], 0, 0 } }
                          else { uint4 { words[at], words[at + 1], words[at + 2], words[at + 3] } };
                let mine = bc::decode(f.id, blk);
                for i in 16 {
                    let theirs = kept(bc::unpack(decoded[(by * 4 + i / 4) * w + bx * 4 + i % 4]), f.channels);
                    let d = kept(mine[i], f.channels) - theirs;
                    maxdiff = max(maxdiff, max(max(abs(d.x), abs(d.y)), max(abs(d.z), abs(d.w))));
                }
            }
        }

        // The CPU's encoding of the same blocks, spread over the image.
        let total = bw * bh;
        let n = if cpublocks < 0 || cpublocks > total { total } else { cpublocks };
        var same: i64 = 0;
        let t0 = clock();
        for k in n {
            let b = k * total / n;
            let bx = b % bw;
            let by = b / bw;
            let mine = bc::encode(f.id, block_texels(pixels, w, bx, by));
            let at = b * wpb;
            var eq = mine.x == words[at] && mine.y == words[at + 1];
            if wpb == 4 { eq = eq && mine.z == words[at + 2] && mine.w == words[at + 3]; }
            if eq { same++; }
            else if k - same < 3 {
                print("  ", f.name, " block ", bx, ",", by, " differs: CPU ", mine.x, " ", mine.y, " ",
                      mine.z, " ", mine.w, ", GPU ", words[at], " ", words[at + 1], " ", words[at + 2], " ",
                      words[at + 3]);
            }
        }
        let cpuus = (clock() - t0) * 1e6 / n as f64;
        print(col(f.name, 8), col(fixed(quality, 2), 10), col(fixed(gpums, 2), 9),
              col(fixed(cpuus, 1), 14), col(str(same, "/", n), 18), maxdiff);
        if same != n { failed = true; }
        gpu::release(blocks);
        gpu::release(tex);
        gpu::release(dst);
    }
    print("decoded images in ", outdir);
    gpu::close();
    if failed { abort("the CPU and the GPU encoded some blocks differently"); }
}
