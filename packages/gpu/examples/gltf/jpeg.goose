// JPEG decoding, baseline and progressive: the entropy decoding on the
// CPU, into each component's coefficients, dequantized; and the rest, the
// inverse DCT, chroma upsampling and colour conversion, a function of one
// pixel that the CPU runs to check and a GPU kernel runs for the image
// (gltf.goose). Greyscale and YCbCr, any sampling factors, restart
// intervals; not arithmetic coding, 12-bit samples or CMYK.

namespace jpeg;

import std;
import gpu_shader;

// Natural order of the zig-zag sequence's k-th coefficient.
let ZIGZAG: i32[64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// A component: its sampling factors, quantization table, its grid of
// blocks (whole MCUs' worth), and where its coefficients start in the
// image's, 64 a block, block by block in rows.
struct Component {
    id: i64,
    h: i64,
    v: i64,
    tq: i64,
    bw: i64,
    bh: i64,
    at: i64,
    dc: i32,        // the DC prediction while a scan decodes
}

struct Image {
    width: i64,
    height: i64,
    hmax: i64,
    vmax: i64,
    ncomp: i64,
    comps: Component[3],
    coefs: i32[>..],
}

fn image() -> Image {
    let none = Component { id: 0, h: 1, v: 1, tq: 0, bw: 0, bh: 0, at: 0, dc: 0 };
    Image { width: 0, height: 0, hmax: 1, vmax: 1, ncomp: 0, comps: [none; 3], coefs: [] }
}

// A Huffman table: a 9-bit lookup for short codes (length << 8 | symbol,
// 0 for none), and the canonical decoding of the rest.
struct Huffman {
    fast: i32[512],
    maxcode: i32[18],
    valptr: i32[17],
    mincode: i32[17],
    vals: u8[256],
}

fn huffman(counts: u8[:], vals: u8[:]) -> Huffman {
    var t = Huffman { fast: [0; 512], maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], vals: [0; 256] };
    for i in vals.len { t.vals[i] = vals[i]; }
    var code: i32 = 0;
    var k: i32 = 0;
    for l in 1..17 {
        let n = counts[l - 1] as i32;
        t.valptr[l] = k;
        t.mincode[l] = code;
        for j in n {
            if l <= 9 {
                // Every 9-bit value this code starts.
                let c = code + j;
                let shift = 9 - l;
                for f in (c << shift)..((c + 1) << shift) { t.fast[f] = ((l as i32) << 8) | vals[k + j]; }
            }
        }
        code += n;
        k += n;
        t.maxcode[l] = if n > 0 { code - 1 } else { -1 };
        code <<= 1;
    }
    t.maxcode[17] = 0x7fffffff;
    t
}

// The entropy-coded bits: bytes with 0xFF 0x00 standing for 0xFF, ended by
// a marker, past which the reader gives zeros.
struct Reader {
    pos: i64,
    buf: u32,
    bits: i32,
    marker: bool,
}

fn fill(r: Reader&, d: u8[:]) {
    while r.bits <= 24 {
        var b: u32 = 0;
        if !r.marker && r.pos < d.len {
            b = d[r.pos];
            if b == 0xff {
                let next = if r.pos + 1 < d.len { d[r.pos + 1] } else { 0xd9 };
                if next == 0 { r.pos += 2; } else { r.marker = true; b = 0; }
            } else {
                r.pos++;
            }
        }
        r.buf |= b << (24 - r.bits);
        r.bits += 8;
    }
}

fn bits(r: Reader&, d: u8[:], n: i32) -> i32 {
    if n == 0 { return 0; }
    fill(r, d);
    let v = (r.buf >> (32 - n)) as i32;
    r.buf <<= n;
    r.bits -= n;
    v
}

fn bit(r: Reader&, d: u8[:]) -> i32 { bits(r, d, 1) }

// A value of `s` bits, signed as JPEG codes it.
fn receive(r: Reader&, d: u8[:], s: i32) -> i32 {
    if s == 0 { return 0; }
    let v = bits(r, d, s);
    if v < (1 << (s - 1)) { v - (1 << s) + 1 } else { v }
}

fn decode(r: Reader&, d: u8[:], t: Huffman&) -> i32 {
    fill(r, d);
    let f = t.fast[r.buf >> 23];
    if f != 0 {
        let n = f >> 8;
        r.buf <<= n;
        r.bits -= n;
        return f & 255;
    }
    var code: i32 = 0;
    for l in 1..17 {
        code = (code << 1) | bit(r, d);
        if code <= t.maxcode[l] { return t.vals[t.valptr[l] + code - t.mincode[l]]; }
    }
    -1
}

// After a restart marker: the reader at the next byte, with nothing kept.
fn restart(r: Reader&, d: u8[:]) {
    var p = r.pos;
    while p + 1 < d.len && !(d[p] == 0xff && d[p + 1] >= 0xd0 && d[p + 1] <= 0xd7) { p++; }
    r.pos = p + 2;
    r.buf = 0;
    r.bits = 0;
    r.marker = false;
}

fn be16(d: u8[:], at: i64) -> i64 { (d[at] as i64 << 8) | d[at + 1] }

// The scan state a progressive image keeps across blocks.
struct Scan {
    ss: i32,
    se: i32,
    ah: i32,
    al: i32,
    eobrun: i32,
}

// One block of a scan, its coefficients at `at` in coefs.
fn decode_block(img: Image&, r: Reader&, d: u8[:], c: i64, at: i64, dc: Huffman&, ac: Huffman&, sc: Scan&,
         progressive: bool) -> bool {
    if !progressive {
        let t = decode(r, d, dc);
        if t < 0 { return false; }
        img.comps[c].dc += receive(r, d, t);
        img.coefs[at] = img.comps[c].dc;
        var k: i32 = 1;
        while k < 64 {
            let rs = decode(r, d, ac);
            if rs < 0 { return false; }
            let run = rs >> 4;
            let s = rs & 15;
            if s == 0 {
                if run != 15 { break; }
                k += 16;
            } else {
                k += run;
                if k > 63 { return false; }
                img.coefs[at + ZIGZAG[k]] = receive(r, d, s);
                k++;
            }
        }
        return true;
    }
    if sc.ss == 0 {
        // DC: its first scan, then a bit at a time.
        if sc.ah == 0 {
            let t = decode(r, d, dc);
            if t < 0 { return false; }
            img.comps[c].dc += receive(r, d, t);
            img.coefs[at] = img.comps[c].dc << sc.al;
        } else if bit(r, d) != 0 {
            img.coefs[at] |= 1 << sc.al;
        }
        return true;
    }
    if sc.ah == 0 {
        // AC, a first scan of a band.
        if sc.eobrun > 0 {
            sc.eobrun -= 1;
            return true;
        }
        var k = sc.ss;
        while k <= sc.se {
            let rs = decode(r, d, ac);
            if rs < 0 { return false; }
            let run = rs >> 4;
            let s = rs & 15;
            if s == 0 {
                if run < 15 {
                    sc.eobrun = (1 << run) - 1;
                    if run > 0 { sc.eobrun += bits(r, d, run); }
                    break;
                }
                k += 16;
            } else {
                k += run;
                if k > 63 { return false; }
                img.coefs[at + ZIGZAG[k]] = receive(r, d, s) * (1 << sc.al);
                k++;
            }
        }
        return true;
    }
    // AC, refining a band a bit at a time.
    let p1: i32 = 1 << sc.al;
    var k = sc.ss;
    if sc.eobrun > 0 {
        sc.eobrun -= 1;
        while k <= sc.se {
            let z = at + ZIGZAG[k];
            if img.coefs[z] != 0 && bit(r, d) != 0 && (img.coefs[z] & p1) == 0 {
                img.coefs[z] += if img.coefs[z] > 0 { p1 } else { -p1 };
            }
            k++;
        }
        return true;
    }
    while k <= sc.se {
        let rs = decode(r, d, ac);
        if rs < 0 { return false; }
        var run = rs >> 4;
        var s = rs & 15;
        var val: i32 = 0;
        if s == 0 {
            if run < 15 {
                sc.eobrun = (1 << run) - 1;
                if run > 0 { sc.eobrun += bits(r, d, run); }
                run = 64;
            }
        } else {
            if s != 1 { return false; }
            val = if bit(r, d) != 0 { p1 } else { -p1 };
        }
        while k <= sc.se {
            let z = at + ZIGZAG[k];
            k++;
            if img.coefs[z] != 0 {
                if bit(r, d) != 0 && (img.coefs[z] & p1) == 0 {
                    img.coefs[z] += if img.coefs[z] > 0 { p1 } else { -p1 };
                }
            } else {
                if run == 0 {
                    img.coefs[z] = val;
                    break;
                }
                run -= 1;
            }
        }
    }
    true
}

// The image's coefficients, dequantized, or an error.
fn decode_file(d: u8[:], out: Image&) -> u8[>..] {
    if d.len < 4 || d[0] != 0xff || d[1] != 0xd8 { return str("not a JPEG"); }
    var qt: i32[256] = [1; 256];
    var dcs: Huffman[4] = [huffman([0; 16], []); 4];
    var acs: Huffman[4] = [huffman([0; 16], []); 4];
    var interval: i64 = 0;
    var progressive = false;
    var p: i64 = 2;
    while p + 4 <= d.len {
        if d[p] != 0xff { p++; continue; }
        let m = d[p + 1];
        if m == 0xff { p++; continue; }
        if m == 0xd8 || (m >= 0xd0 && m <= 0xd7) || m == 0x01 { p += 2; continue; }
        if m == 0xd9 { break; }
        let len = be16(d, p + 2);
        let seg = p + 4;
        let end = p + 2 + len;
        if end > d.len { return str("a segment runs past the end"); }
        if m == 0xdb {
            var q = seg;
            while q < end {
                let pq = d[q] >> 4;
                let tq = (d[q] & 15) as i64;
                for k in 64 {
                    let v = if pq == 0 { d[q + 1 + k] as i32 } else { be16(d, q + 1 + 2 * k) as i32 };
                    qt[tq * 64 + ZIGZAG[k]] = v;
                }
                q += 1 + 64 * (if pq == 0 { 1 } else { 2 });
            }
        } else if m == 0xc4 {
            var q = seg;
            while q < end {
                let tc = d[q] >> 4;
                let th = (d[q] & 3) as i64;
                var n: i64 = 0;
                for k in 16 { n += d[q + 1 + k]; }
                let t = huffman(d[q + 1..q + 17], d[q + 17..q + 17 + n]);
                if tc == 0 { dcs[th] = t; } else { acs[th] = t; }
                q += 17 + n;
            }
        } else if m == 0xdd {
            interval = be16(d, seg);
        } else if m == 0xc0 || m == 0xc1 || m == 0xc2 {
            progressive = m == 0xc2;
            if d[seg] != 8 { return str("12-bit samples"); }
            out.height = be16(d, seg + 1);
            out.width = be16(d, seg + 3);
            let n = d[seg + 5] as i64;
            if n != 1 && n != 3 { return str(n, " components"); }
            out.ncomp = n;
            out.hmax = 1;
            out.vmax = 1;
            for c in n {
                let at = seg + 6 + 3 * c;
                let comp = Component { id: d[at], h: d[at + 1] >> 4, v: d[at + 1] & 15,
                                       tq: d[at + 2] & 3, bw: 0, bh: 0, at: 0, dc: 0 };
                out.hmax = max(out.hmax, comp.h);
                out.vmax = max(out.vmax, comp.v);
                out.comps[c] = comp;
            }
            let mcux = (out.width + 8 * out.hmax - 1) / (8 * out.hmax);
            let mcuy = (out.height + 8 * out.vmax - 1) / (8 * out.vmax);
            var total: i64 = 0;
            for c in n {
                out.comps[c].bw = mcux * out.comps[c].h;
                out.comps[c].bh = mcuy * out.comps[c].v;
                out.comps[c].at = total;
                total += out.comps[c].bw * out.comps[c].bh * 64;
            }
            out.coefs.resize(total, 0);
        } else if m >= 0xc3 && m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc {
            return str("an unsupported JPEG process (marker ", m, ")");
        } else if m == 0xda {
            let ns: i64 = d[seg];
            var cs: i64[4] = [0; 4];
            var td: i64[4] = [0; 4];
            var ta: i64[4] = [0; 4];
            for k in ns {
                let id: i64 = d[seg + 1 + 2 * k];
                for c in out.ncomp { if out.comps[c].id == id { cs[k] = c; } }
                td[k] = d[seg + 2 + 2 * k] >> 4;
                ta[k] = d[seg + 2 + 2 * k] & 15;
            }
            let h = seg + 1 + 2 * ns;
            var sc = Scan { ss: d[h], se: d[h + 1], ah: d[h + 2] >> 4, al: d[h + 2] & 15,
                            eobrun: 0 };
            var r = Reader { pos: end, buf: 0, bits: 0, marker: false };
            for c in ns { out.comps[cs[c]].dc = 0; }
            var count: i64 = 0;
            if ns == 1 {
                // One component, its own blocks in rows, as many as cover it.
                let c = cs[0];
                let cw = (out.width * out.comps[c].h + out.hmax - 1) / out.hmax;
                let ch = (out.height * out.comps[c].v + out.vmax - 1) / out.vmax;
                for by in (ch + 7) / 8 {
                    for bx in (cw + 7) / 8 {
                        if interval > 0 && count > 0 && count % interval == 0 {
                            restart(r, d);
                            out.comps[c].dc = 0;
                            sc.eobrun = 0;
                        }
                        let at = out.comps[c].at + (by * out.comps[c].bw + bx) * 64;
                        if !decode_block(out, r, d, c, at, dcs[td[0]], acs[ta[0]], sc, progressive) {
                            return str("corrupt entropy-coded data");
                        }
                        count++;
                    }
                }
            } else {
                // Interleaved: MCU by MCU, each component's blocks in it.
                let mcux = (out.width + 8 * out.hmax - 1) / (8 * out.hmax);
                let mcuy = (out.height + 8 * out.vmax - 1) / (8 * out.vmax);
                for my in mcuy {
                    for mx in mcux {
                        if interval > 0 && count > 0 && count % interval == 0 {
                            restart(r, d);
                            for k in ns { out.comps[cs[k]].dc = 0; }
                            sc.eobrun = 0;
                        }
                        for k in ns {
                            let c = cs[k];
                            let ch = out.comps[c].h;
                            let cv = out.comps[c].v;
                            for y in cv {
                                for x in ch {
                                    let bx = mx * ch + x;
                                    let by = my * cv + y;
                                    let at = out.comps[c].at + (by * out.comps[c].bw + bx) * 64;
                                    if !decode_block(out, r, d, c, at, dcs[td[k]], acs[ta[k]], sc, progressive) {
                                        return str("corrupt entropy-coded data");
                                    }
                                }
                            }
                        }
                        count++;
                    }
                }
            }
            // On to the marker after the scan's data.
            var q = r.pos;
            while q + 1 < d.len && !(d[q] == 0xff && d[q + 1] != 0 && !(d[q + 1] >= 0xd0 && d[q + 1] <= 0xd7)) {
                q++;
            }
            p = q;
            continue;
        }
        p = end;
    }
    if out.ncomp == 0 { return str("no frame"); }
    // Dequantized, so that what remains needs no table.
    for ci in out.ncomp {
        let c = out.comps[ci];
        for b in c.bw * c.bh {
            let at = c.at + b * 64;
            for k in 64 { out.coefs[at + k] *= qt[c.tq * 64 + k]; }
        }
    }
    str("")
}

// ---------------------------------------------------------------------------
// From coefficients to pixels: a function of one pixel, for the CPU and the
// GPU alike, given a way to read a block's coefficients.

// cos((2x + 1) u pi / 16), with u = 0's scaled by 1/sqrt(2), for x and u.
let COS: f32[64] = [
    0.70710678, 0.98078528, 0.92387953, 0.83146961, 0.70710678, 0.55557023, 0.38268343, 0.19509032,
    0.70710678, 0.83146961, 0.38268343, -0.19509032, -0.70710678, -0.98078528, -0.92387953, -0.55557023,
    0.70710678, 0.55557023, -0.38268343, -0.98078528, -0.70710678, 0.19509032, 0.92387953, 0.83146961,
    0.70710678, 0.19509032, -0.92387953, -0.55557023, 0.70710678, 0.83146961, -0.38268343, -0.98078528,
    0.70710678, -0.19509032, -0.92387953, 0.55557023, 0.70710678, -0.83146961, -0.38268343, 0.98078528,
    0.70710678, -0.55557023, -0.38268343, 0.98078528, -0.70710678, -0.19509032, 0.92387953, -0.83146961,
    0.70710678, -0.83146961, 0.38268343, 0.19509032, -0.70710678, 0.98078528, -0.92387953, 0.55557023,
    0.70710678, -0.98078528, 0.92387953, -0.83146961, 0.70710678, -0.55557023, 0.38268343, -0.19509032,
];

// The sample at (x, y) of a block, 0..255 before clamping.
fn idct(c: i32[64], x: u32, y: u32) -> f32 {
    var s: f32 = 0.0;
    for v: u32 in 8 {
        var row: f32 = 0.0;
        for u: u32 in 8 { row += c[v * 8 + u] * COS[x * 8 + u]; }
        s += row * COS[y * 8 + v];
    }
    s / 4.0 + 128.0
}

// What the pixel kernel needs of an image, a push constant's worth.
struct Layout {
    width: u32,
    height: u32,
    ncomp: u32,
    hmax: u32,
    vmax: u32,
    at: u32[3],     // each component's first coefficient
    bw: u32[3],     // and its blocks across
    bh: u32[3],     // and down
    h: u32[3],
    v: u32[3],
}

fn layout(img: Image&) -> Layout {
    var l = Layout { width: img.width as u32, height: img.height as u32, ncomp: img.ncomp as u32,
                     hmax: img.hmax as u32, vmax: img.vmax as u32, at: [0; 3], bw: [0; 3], bh: [0; 3],
                     h: [0; 3], v: [0; 3] };
    for c in img.ncomp {
        l.at[c] = img.comps[c].at as u32;
        l.bw[c] = img.comps[c].bw as u32;
        l.bh[c] = img.comps[c].bh as u32;
        l.h[c] = img.comps[c].h as u32;
        l.v[c] = img.comps[c].v as u32;
    }
    l
}

fn clamp8(x: f32) -> u32 {
    if x <= 0.0 { return 0; }
    if x >= 255.0 { return 255; }
    (x + 0.5) as u32
}

// YCbCr (or Y alone) to RGBA8, packed with red in the low byte.
fn rgba(y: f32, cb: f32, cr: f32, grey: bool) -> u32 {
    if grey {
        let g = clamp8(y);
        return g | (g << 8) | (g << 16) | (255 << 24);
    }
    let r = clamp8(y + 1.402 * (cr - 128.0));
    let g = clamp8(y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0));
    let b = clamp8(y + 1.772 * (cb - 128.0));
    r | (g << 8) | (b << 16) | (255 << 24)
}

// A coefficient, from the CPU's array or the GPU's buffer.
fn coef(c: const i32[:], i: u32) -> i32 { c[i] }
fn coef(b: gpu::StructuredBuffer<i32>, i: u32) -> i32 { gpu::load(b, i) }

// Component c's sample at (sx, sy) of its own grid, clamped to the part
// the image covers.
fn comp_sample<S>(src: S, l: Layout, c: u32, sx: i32, sy: i32) -> f32 {
    let cw = (l.width * l.h[c] + l.hmax - 1) / l.hmax;
    let ch = (l.height * l.v[c] + l.vmax - 1) / l.vmax;
    let x = clamp(sx, 0, cw as i32 - 1) as u32;
    let y = clamp(sy, 0, ch as i32 - 1) as u32;
    let at = l.at[c] + ((y / 8) * l.bw[c] + x / 8) * 64;
    var b: i32[64] = [0; 64];
    for k: u32 in 64 { b[k] = coef(src, at + k); }
    // An 8-bit sample, as the decoders keep each component's before they
    // upsample or convert it: ringing past 0 or 255 does not reach the
    // other components' pixels.
    clamp8(idct(b, x & 7, y & 7)) as f32
}

// Component c at pixel (x, y): a sample of its own where it is sampled as
// finely as the image, else the four around the pixel's centre blended,
// as libjpeg's fancy upsampling does.
fn comp_at<S>(src: S, l: Layout, c: u32, x: u32, y: u32) -> f32 {
    if l.h[c] == l.hmax && l.v[c] == l.vmax { return comp_sample(src, l, c, x as i32, y as i32); }
    let fx: f32 = (x as f32 + 0.5) * (l.h[c] as f32) / (l.hmax as f32) - 0.5;
    let fy: f32 = (y as f32 + 0.5) * (l.v[c] as f32) / (l.vmax as f32) - 0.5;
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let ix = x0 as i32;
    let iy = y0 as i32;
    let a = comp_sample(src, l, c, ix, iy) * (1.0 - tx) + comp_sample(src, l, c, ix + 1, iy) * tx;
    let b = comp_sample(src, l, c, ix, iy + 1) * (1.0 - tx) + comp_sample(src, l, c, ix + 1, iy + 1) * tx;
    a * (1.0 - ty) + b * ty
}

// The pixel at (x, y), RGBA8 packed with red in the low byte.
fn pixel<S>(src: S, l: Layout, x: u32, y: u32) -> u32 {
    let lum = comp_at(src, l, 0, x, y);
    if l.ncomp == 1 { return rgba(lum, 0.0, 0.0, true); }
    rgba(lum, comp_at(src, l, 1, x, y), comp_at(src, l, 2, x, y), false)
}
