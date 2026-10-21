// Block compression: encoders for BC1, BC3, BC4, BC5 and BC7, and decoders
// for what they make, written once and run both on the CPU and as GPU
// functions. Every decision is made in integers (i32 and u32, with no
// product that overflows and no division by zero), which the CPU and every
// GPU compute alike, so a block encodes to the same bits on each: the
// program compares them.
//
// A block is 4x4 texels, row by row, each an int4 of RGBA in 0..255.

namespace bc;

import std;
import vec;

// ---------------------------------------------------------------------------
// Shared pieces.

// The texels' squared distance in RGB, or in RGBA.
fn dist3(a: int4, b: int4) -> i32 {
    let d = a - b;
    d.x * d.x + d.y * d.y + d.z * d.z
}

fn dist4(a: int4, b: int4) -> i32 {
    let d = a - b;
    d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w
}

fn dot3(a: int4, b: int4) -> i32 { a.x * b.x + a.y * b.y + a.z * b.z }
fn dot4(a: int4, b: int4) -> i32 { a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w }

fn clamp255(c: int4) -> int4 {
    int4 { clamp(c.x, 0, 255), clamp(c.y, 0, 255), clamp(c.z, 0, 255), clamp(c.w, 0, 255) }
}

// RGBA8 packed in a u32, R in the low byte, as an RGBA8 texture's texel.
fn unpack(p: u32) -> int4 {
    int4 { (p & 255) as i32, ((p >> 8) & 255) as i32, ((p >> 16) & 255) as i32, (p >> 24) as i32 }
}

fn pack(c: int4) -> u32 {
    (c.x as u32) | ((c.y as u32) << 8) | ((c.z as u32) << 16) | ((c.w as u32) << 24)
}

// n/d rounded to nearest, for d > 0, either sign of n.
fn round_div(n: i32, d: i32) -> i32 {
    if n >= 0 { (n + d / 2) / d } else { -((d / 2 - n) / d) }
}

// The 128 bits of a block, written `n` bits at a time from bit `pos`.
fn put(b: u32[4], pos: u32, n: u32, v: u32) -> u32[4] {
    var r = b;
    let w = pos >> 5;
    let s = pos & 31;
    let m = if n == 32 { 0xffffffff as u32 } else { ((1 as u32) << n) - 1 };
    r[w] |= (v & m) << s;
    if s + n > 32 { r[w + 1] |= (v & m) >> (32 - s); }
    r
}

fn get(b: u32[4], pos: u32, n: u32) -> u32 {
    let w = pos >> 5;
    let s = pos & 31;
    var v = b[w] >> s;
    if s + n > 32 { v |= b[w + 1] << (32 - s); }
    if n == 32 { v } else { v & (((1 as u32) << n) - 1) }
}

// The principal axis of the texels `mask` selects (bit i, texel i), in RGB
// or with alpha too: power iteration on their covariance, kept in i32 by
// scaling it down and renormalizing the axis to under 512 each step.
fn axis(px: int4[16], mask: u32, alpha: bool) -> int4 {
    var sum = int4_0;
    var n: i32 = 0;
    var lo = int4 { 255, 255, 255, 255 };
    var hi = int4_0;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            sum += px[i];
            n += 1;
            lo = int4 { min(lo.x, px[i].x), min(lo.y, px[i].y), min(lo.z, px[i].z), min(lo.w, px[i].w) };
            hi = int4 { max(hi.x, px[i].x), max(hi.y, px[i].y), max(hi.z, px[i].z), max(hi.w, px[i].w) };
        }
    }
    if n == 0 { return int4_0; }
    // The covariance of n times each texel less the sum, which keeps it
    // integer: each product at most (16 * 255)^2 * 16, scaled by 2^-8.
    var cxx: i32 = 0; var cxy: i32 = 0; var cxz: i32 = 0; var cxw: i32 = 0;
    var cyy: i32 = 0; var cyz: i32 = 0; var cyw: i32 = 0;
    var czz: i32 = 0; var czw: i32 = 0; var cww: i32 = 0;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            let d = px[i] * n - sum;
            cxx += (d.x * d.x) >> 8; cxy += (d.x * d.y) >> 8; cxz += (d.x * d.z) >> 8;
            cyy += (d.y * d.y) >> 8; cyz += (d.y * d.z) >> 8; czz += (d.z * d.z) >> 8;
            if alpha {
                cxw += (d.x * d.w) >> 8; cyw += (d.y * d.w) >> 8; czw += (d.z * d.w) >> 8;
                cww += (d.w * d.w) >> 8;
            }
        }
    }
    // Each product is at most (16 * 255)^2 >> 8 = 65025, sixteen of them a
    // 2^20; scaled once more, times an axis under 512, four of them sum
    // under 2^31.
    cxx >>= 4; cxy >>= 4; cxz >>= 4; cxw >>= 4; cyy >>= 4; cyz >>= 4; cyw >>= 4;
    czz >>= 4; czw >>= 4; cww >>= 4;
    var v = hi - lo;
    if !alpha { v.w = 0; }
    for it: u32 in 6 {
        let x = cxx * v.x + cxy * v.y + cxz * v.z + cxw * v.w;
        let y = cxy * v.x + cyy * v.y + cyz * v.z + cyw * v.w;
        let z = cxz * v.x + cyz * v.y + czz * v.z + czw * v.w;
        let w = cxw * v.x + cyw * v.y + czw * v.z + cww * v.w;
        var m = max(max(abs(x), abs(y)), max(abs(z), abs(w)));
        if m == 0 { break; }
        var s: u32 = 0;
        while m >= 512 {
            m >>= 1;
            s += 1;
        }
        v = int4 { x >> s, y >> s, z >> s, w >> s };
    }
    if v.x == 0 && v.y == 0 && v.z == 0 && v.w == 0 {
        // No spread: luminance, which orders the endpoints as well as any.
        v = int4 { 299, 587, 114, if alpha { 256 } else { 0 } };
    }
    v
}

// The texels `mask` selects with the least and the greatest projection on
// the axis: a first guess at the endpoints.
struct Ends { lo: int4, hi: int4 }

fn extremes(px: int4[16], mask: u32, v: int4) -> Ends {
    var lo = int4_0;
    var hi = int4_0;
    var dlo: i32 = 0x7fffffff;
    var dhi: i32 = -0x7fffffff;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            let d = dot4(px[i], v);
            if d < dlo { dlo = d; lo = px[i]; }
            if d > dhi { dhi = d; hi = px[i]; }
        }
    }
    Ends { lo: lo, hi: hi }
}

// Least squares (§ "refine" below): the endpoints whose blend, with the
// weights the texels were given, is nearest to them. `ws` holds each
// texel's weight of the second endpoint out of `full`; texels outside
// `mask` take no part. Small weights keep every sum within i32: the caller
// passes them out of at most 16.
fn solve(px: int4[16], mask: u32, ws: i32[16], full: i32, old: Ends) -> Ends {
    var aa: i32 = 0; var ab: i32 = 0; var bb: i32 = 0;
    var ax = int4_0;
    var bx = int4_0;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            let b = ws[i];
            let a = full - b;
            aa += a * a; ab += a * b; bb += b * b;
            ax += px[i] * a;
            bx += px[i] * b;
        }
    }
    let det = aa * bb - ab * ab;
    if det <= 0 { return old; }
    // lo = full * (ax bb - bx ab) / det, hi likewise, in two steps so that
    // nothing overflows: the quotient, then the remainder's share.
    var lo = int4_0;
    var hi = int4_0;
    let nl = ax * bb - bx * ab;
    let nh = bx * aa - ax * ab;
    lo = int4 { scaled(nl.x, full, det), scaled(nl.y, full, det), scaled(nl.z, full, det),
                scaled(nl.w, full, det) };
    hi = int4 { scaled(nh.x, full, det), scaled(nh.y, full, det), scaled(nh.z, full, det),
                scaled(nh.w, full, det) };
    Ends { lo: clamp255(lo), hi: clamp255(hi) }
}

// n * k / d, rounded, for d > 0 and k at most 16, without forming n * k.
fn scaled(n: i32, k: i32, d: i32) -> i32 {
    let q = n / d;
    let r = n - q * d;
    q * k + round_div(r * k, d)
}

// ---------------------------------------------------------------------------
// BC1: two RGB565 endpoints and 2-bit indices, the palette the endpoints
// and their blends a third and two thirds of the way. 64 bits, as a uint2.

fn mul8bit(a: i32, b: i32) -> i32 {
    let t = a * b + 128;
    (t + (t >> 8)) >> 8
}

fn to565(c: int4) -> u32 {
    ((mul8bit(c.x, 31) << 11) | (mul8bit(c.y, 63) << 5) | mul8bit(c.z, 31)) as u32
}

fn from565(v: u32) -> int4 {
    let r = ((v >> 11) & 31) as i32;
    let g = ((v >> 5) & 63) as i32;
    let b = (v & 31) as i32;
    int4 { (r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2), 255 }
}

// The 4-colour palette of two 565 endpoints, the first greater.
fn palette4(c0: u32, c1: u32) -> int4[4] {
    let a = from565(c0);
    let b = from565(c1);
    [a, b, (a * 2 + b) / 3, (a + b * 2) / 3]
}

struct Fit1 { c0: u32, c1: u32, idx: u32, err: i32 }

// Each texel's nearest palette entry, and the error of the lot.
fn bc1_indices(px: int4[16], c0: u32, c1: u32) -> Fit1 {
    let pal = palette4(c0, c1);
    var idx: u32 = 0;
    var err: i32 = 0;
    for i: u32 in 16 {
        var best: u32 = 0;
        var be = dist3(px[i], pal[0]);
        for k: u32 in 1..4 {
            let e = dist3(px[i], pal[k]);
            if e < be { be = e; best = k; }
        }
        idx |= best << (2 * i);
        err += be;
    }
    Fit1 { c0: c0, c1: c1, idx: idx, err: err }
}

// The weight of the second endpoint, in thirds, each index gives.
let BC1_WEIGHT: i32[4] = [0, 3, 1, 2];

// A block of one colour: the endpoints whose blend two thirds of the way
// comes nearest to it, channel by channel, searched outright.
fn bc1_solid(c: int4) -> Fit1 {
    var e0 = int4_0;
    var e1 = int4_0;
    for ch: u32 in 3 {
        let bits: i32 = if ch == 1 { 6 } else { 5 };
        let top: i32 = (1 << bits) - 1;
        let want = if ch == 0 { c.x } else if ch == 1 { c.y } else { c.z };
        var best: i32 = 0x7fffffff;
        var b0: i32 = 0;
        var b1: i32 = 0;
        for a: i32 in 0..top + 1 {
            let ea = if bits == 5 { (a << 3) | (a >> 2) } else { (a << 2) | (a >> 4) };
            for b: i32 in 0..top + 1 {
                let eb = if bits == 5 { (b << 3) | (b >> 2) } else { (b << 2) | (b >> 4) };
                let e = abs((ea * 2 + eb) / 3 - want);
                if e < best { best = e; b0 = a; b1 = b; }
            }
        }
        if ch == 0 { e0.x = b0; e1.x = b1; } else if ch == 1 { e0.y = b0; e1.y = b1; }
        else { e0.z = b0; e1.z = b1; }
    }
    let c0 = ((e0.x << 11) | (e0.y << 5) | e0.z) as u32;
    let c1 = ((e1.x << 11) | (e1.y << 5) | e1.z) as u32;
    // Index 2, the blend, for every texel; ordered as the other blocks are.
    var px: int4[16] = [c; 16];
    var f = bc1_indices(px, c0, c1);
    f
}

// The 4-colour mode needs the first endpoint the greater: swapping them
// swaps index 0 with 1 and 2 with 3.
fn bc1_order(f: Fit1) -> uint2 {
    if f.c0 > f.c1 { return uint2 { f.c0 | (f.c1 << 16), f.idx }; }
    if f.c0 == f.c1 { return uint2 { f.c0 | (f.c1 << 16), 0 }; }
    uint2 { f.c1 | (f.c0 << 16), f.idx ^ 0x55555555 }
}

fn encode_bc1(px: int4[16]) -> uint2 {
    var solid = true;
    for i: u32 in 1..16 { if dist3(px[i], px[0]) != 0 { solid = false; } }
    if solid { return bc1_order(bc1_solid(px[0])); }
    let all: u32 = 0xffff;
    let e = extremes(px, all, axis(px, all, false));
    var best = bc1_indices(px, to565(e.hi), to565(e.lo));
    // Refine twice: least squares on the indices chosen, kept where better.
    for it: u32 in 2 {
        var ws: i32[16] = [0; 16];
        for i: u32 in 16 { ws[i] = BC1_WEIGHT[(best.idx >> (2 * i)) & 3]; }
        let r = solve(px, all, ws, 3, Ends { lo: from565(best.c0), hi: from565(best.c1) });
        let f = bc1_indices(px, to565(r.lo), to565(r.hi));
        if f.err < best.err { best = f; }
    }
    bc1_order(best)
}

fn decode_bc1(b: uint2) -> int4[16] {
    let c0 = b.x & 0xffff;
    let c1 = b.x >> 16;
    let a = from565(c0);
    let c = from565(c1);
    var pal: int4[4] = [a, c, (a * 2 + c) / 3, (a + c * 2) / 3];
    if c0 <= c1 { pal = [a, c, (a + c) / 2, int4_0]; }
    var out: int4[16] = [int4_0; 16];
    for i: u32 in 16 { out[i] = pal[(b.y >> (2 * i)) & 3]; }
    out
}

// ---------------------------------------------------------------------------
// BC4: one channel, two 8-bit endpoints and 3-bit indices: eight values
// between them when the first is greater, else six and then 0 and 255.
// 64 bits, as a uint2; BC5 is two of them, BC3's alpha is one.

fn palette8(e0: i32, e1: i32) -> i32[8] {
    if e0 > e1 {
        return [e0, e1, (6 * e0 + e1) / 7, (5 * e0 + 2 * e1) / 7, (4 * e0 + 3 * e1) / 7,
                (3 * e0 + 4 * e1) / 7, (2 * e0 + 5 * e1) / 7, (e0 + 6 * e1) / 7];
    }
    [e0, e1, (4 * e0 + e1) / 5, (3 * e0 + 2 * e1) / 5, (2 * e0 + 3 * e1) / 5, (e0 + 4 * e1) / 5, 0,
     255]
}

struct Fit4 { e0: i32, e1: i32, idx: u32[2], err: i32 }

fn bc4_fit(v: i32[16], e0: i32, e1: i32) -> Fit4 {
    let pal = palette8(e0, e1);
    var bits: u32[4] = [0; 4];
    var err: i32 = 0;
    for i: u32 in 16 {
        var best: u32 = 0;
        var be = abs(v[i] - pal[0]);
        for k: u32 in 1..8 {
            let e = abs(v[i] - pal[k]);
            if e < be { be = e; best = k; }
        }
        bits = put(bits, 3 * i, 3, best);
        err += be * be;
    }
    Fit4 { e0: e0, e1: e1, idx: [bits[0], bits[1]], err: err }
}

fn encode_bc4(v: i32[16]) -> uint2 {
    var lo: i32 = 255; var hi: i32 = 0;
    // Without the extremes, which the six-value mode has for free.
    var lo6: i32 = 255; var hi6: i32 = 0;
    for x in v {
        lo = min(lo, x); hi = max(hi, x);
        if x != 0 && x != 255 { lo6 = min(lo6, x); hi6 = max(hi6, x); }
    }
    var best = bc4_fit(v, hi, lo);
    if hi == lo { best = bc4_fit(v, lo, lo); }
    if lo6 <= hi6 {
        let f = bc4_fit(v, lo6, hi6);
        if f.err < best.err { best = f; }
    }
    // A step of each endpoint, which the rounded blends often reward.
    for d: u32 in 4 {
        let e0 = best.e0 + (if d == 0 { 1 } else if d == 1 { -1 } else { 0 });
        let e1 = best.e1 + (if d == 2 { 1 } else if d == 3 { -1 } else { 0 });
        if e0 >= 0 && e0 <= 255 && e1 >= 0 && e1 <= 255 && (e0 > e1) == (best.e0 > best.e1) {
            let f = bc4_fit(v, e0, e1);
            if f.err < best.err { best = f; }
        }
    }
    uint2 { (best.e0 as u32) | ((best.e1 as u32) << 8) | (best.idx[0] << 16),
            (best.idx[0] >> 16) | (best.idx[1] << 16) }
}

fn decode_bc4(b: uint2) -> i32[16] {
    let bits: u32[4] = [b.x, b.y, 0, 0];
    let pal = palette8((b.x & 255) as i32, ((b.x >> 8) & 255) as i32);
    var out: i32[16] = [0; 16];
    for i: u32 in 16 { out[i] = pal[get(bits, 16 + 3 * i, 3)]; }
    out
}

fn channel(px: int4[16], c: u32) -> i32[16] {
    var v: i32[16] = [0; 16];
    for i: u32 in 16 {
        v[i] = if c == 0 { px[i].x } else if c == 1 { px[i].y } else if c == 2 { px[i].z } else { px[i].w };
    }
    v
}

fn encode_bc5(px: int4[16]) -> uint4 {
    let r = encode_bc4(channel(px, 0));
    let g = encode_bc4(channel(px, 1));
    uint4 { r.x, r.y, g.x, g.y }
}

fn decode_bc5(b: uint4) -> int4[16] {
    let r = decode_bc4(uint2 { b.x, b.y });
    let g = decode_bc4(uint2 { b.z, b.w });
    var out: int4[16] = [int4_0; 16];
    for i: u32 in 16 { out[i] = int4 { r[i], g[i], 0, 255 }; }
    out
}

fn encode_bc3(px: int4[16]) -> uint4 {
    let a = encode_bc4(channel(px, 3));
    let c = encode_bc1(px);
    uint4 { a.x, a.y, c.x, c.y }
}

fn decode_bc3(b: uint4) -> int4[16] {
    let a = decode_bc4(uint2 { b.x, b.y });
    // BC3's colour is always the 4-colour palette.
    let pal = palette4(b.z & 0xffff, b.z >> 16);
    var out: int4[16] = [int4_0; 16];
    for i: u32 in 16 {
        let c = pal[(b.w >> (2 * i)) & 3];
        out[i] = int4 { c.x, c.y, c.z, a[i] };
    }
    out
}

// ---------------------------------------------------------------------------
// BC7, in two of its eight modes. Mode 6: one subset of RGBA endpoints of
// 7 bits and a p-bit each, and 4-bit indices. Mode 1: two subsets, picked
// from 64 partitions, of RGB endpoints of 6 bits and a p-bit each subset
// shares, and 3-bit indices; alpha is 255. 128 bits, as a uint4.

let W3: i32[8] = [0, 9, 18, 27, 37, 46, 55, 64];
let W4: i32[16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

// Which texels each two-subset partition puts in subset 1: bit i, texel i.
let PARTITION2: u32[64] = [
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
];

// The texel of subset 1 whose index drops its top bit, per partition.
let ANCHOR2: u32[64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
];

fn blend(a: int4, b: int4, w: i32) -> int4 { (a * (64 - w) + b * w + int4 { 32, 32, 32, 32 }) / 64 }

// An endpoint channel of `bits` bits and its p-bit, as 8 bits.
fn expand(q: i32, p: i32, bits: u32) -> i32 {
    let v = (q << 1) | p;
    let n = bits + 1;
    (v << (8 - n)) | (v >> (2 * n - 8))
}

// The `bits`-bit value whose expansion with p-bit p is nearest to x.
fn quantize(x: i32, p: i32, bits: u32) -> i32 {
    let top: i32 = (1 << bits) - 1;
    let guess = x >> (8 - bits);
    var best = guess;
    var be: i32 = 0x7fffffff;
    for d: i32 in -1..2 {
        let q = clamp(guess + d, 0, top);
        let e = abs(expand(q, p, bits) - x);
        if e < be { be = e; best = q; }
    }
    best
}

fn quantize4(c: int4, p: i32, bits: u32) -> int4 {
    int4 { quantize(c.x, p, bits), quantize(c.y, p, bits), quantize(c.z, p, bits),
           quantize(c.w, p, bits) }
}

fn expand4(q: int4, p: i32, bits: u32) -> int4 {
    int4 { expand(q.x, p, bits), expand(q.y, p, bits), expand(q.z, p, bits), expand(q.w, p, bits) }
}

// A subset's endpoints, quantized, with their p-bits and each texel's
// index (16 entries, the subset's own meaningful), and its error.
struct Fit7 { q0: int4, q1: int4, p0: i32, p1: i32, idx: u32[16], err: i32 }

// The texels of `mask` against the palette of quantized endpoints: `ib`
// index bits, RGBA when `alpha`.
fn fit7(px: int4[16], mask: u32, q0: int4, q1: int4, p0: i32, p1: i32, bits: u32, ib: u32,
        alpha: bool) -> Fit7 {
    let e0 = expand4(q0, p0, bits);
    let e1 = expand4(q1, p1, bits);
    let n: u32 = (1 as u32) << ib;
    var pal: int4[16] = [int4_0; 16];
    for k: u32 in n { pal[k] = blend(e0, e1, if ib == 3 { W3[k] } else { W4[k] }); }
    var idx: u32[16] = [0; 16];
    var err: i32 = 0;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            var best: u32 = 0;
            var be: i32 = 0x7fffffff;
            for k: u32 in n {
                let e = if alpha { dist4(px[i], pal[k]) } else { dist3(px[i], pal[k]) };
                if e < be { be = e; best = k; }
            }
            idx[i] = best;
            err += be;
        }
    }
    Fit7 { q0: q0, q1: q1, p0: p0, p1: p1, idx: idx, err: err }
}

// Endpoints for the texels of `mask`, at `bits` bits with a p-bit each
// (`shared`: one for both), refined once by least squares.
fn fit_subset(px: int4[16], mask: u32, bits: u32, ib: u32, alpha: bool, shared: bool) -> Fit7 {
    let e = extremes(px, mask, axis(px, mask, alpha));
    var best = Fit7 { q0: int4_0, q1: int4_0, p0: 0, p1: 0, idx: [0; 16], err: 0x7fffffff };
    var lo = e.lo;
    var hi = e.hi;
    if !alpha { lo.w = 255; hi.w = 255; }
    for pass: u32 in 2 {
        for pb: i32 in 0..4 {
            let p0 = pb & 1;
            let p1 = if shared { p0 } else { pb >> 1 };
            if shared && pb > 1 { continue; }
            let f = fit7(px, mask, quantize4(lo, p0, bits), quantize4(hi, p1, bits), p0, p1, bits,
                         ib, alpha);
            if f.err < best.err { best = f; }
        }
        if pass == 0 {
            // Least squares on the indices found, weights in sixteenths.
            var ws: i32[16] = [0; 16];
            for i: u32 in 16 {
                let w = if ib == 3 { W3[best.idx[i]] } else { W4[best.idx[i]] };
                ws[i] = (w + 2) >> 2;
            }
            let r = solve(px, mask, ws, 16, Ends { lo: lo, hi: hi });
            lo = r.lo;
            hi = r.hi;
            if !alpha { lo.w = 255; hi.w = 255; }
        }
    }
    best
}

// The error of the texels of `mask` against their bounding box's diagonal
// in eight steps: a quick measure of how well a subset fits a line, to
// rank partitions by.
fn box_error(px: int4[16], mask: u32) -> i32 {
    var lo = int4 { 255, 255, 255, 0 };
    var hi = int4_0;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            lo = int4 { min(lo.x, px[i].x), min(lo.y, px[i].y), min(lo.z, px[i].z), 0 };
            hi = int4 { max(hi.x, px[i].x), max(hi.y, px[i].y), max(hi.z, px[i].z), 0 };
        }
    }
    let d = hi - lo;
    let len = dot3(d, d);
    if len == 0 { return 0; }
    var err: i32 = 0;
    for i: u32 in 16 {
        if (mask >> i) & 1 != 0 {
            let t = dot3(px[i] - lo, d);
            let step = clamp((t * 7 + len / 2) / len, 0, 7);
            err += dist3(px[i], lo + d * step / 7);
        }
    }
    err
}

struct Choice { bits: uint4, err: i32 }

fn bc7_mode6(px: int4[16]) -> Choice {
    var f = fit_subset(px, 0xffff, 7, 4, true, false);
    // Texel 0's index drops its top bit: when it is set, swap the ends.
    if f.idx[0] >= 8 {
        f = Fit7 { q0: f.q1, q1: f.q0, p0: f.p1, p1: f.p0, idx: f.idx, err: f.err };
        for i: u32 in 16 { f.idx[i] = 15 - f.idx[i]; }
    }
    var b: u32[4] = [0; 4];
    b = put(b, 0, 7, 64);
    var pos: u32 = 7;
    for ch: u32 in 4 {
        let a = if ch == 0 { f.q0.x } else if ch == 1 { f.q0.y } else if ch == 2 { f.q0.z } else { f.q0.w };
        let c = if ch == 0 { f.q1.x } else if ch == 1 { f.q1.y } else if ch == 2 { f.q1.z } else { f.q1.w };
        b = put(b, pos, 7, a as u32);
        b = put(b, pos + 7, 7, c as u32);
        pos += 14;
    }
    b = put(b, pos, 1, f.p0 as u32);
    b = put(b, pos + 1, 1, f.p1 as u32);
    pos += 2;
    for i: u32 in 16 {
        let n: u32 = if i == 0 { 3 } else { 4 };
        b = put(b, pos, n, f.idx[i]);
        pos += n;
    }
    Choice { bits: uint4 { b[0], b[1], b[2], b[3] }, err: f.err }
}

fn bc7_mode1(px: int4[16]) -> Choice {
    // The partition whose subsets each fit a line best.
    var part: u32 = 0;
    var pe: i32 = 0x7fffffff;
    for p: u32 in 64 {
        let m = PARTITION2[p];
        let e = box_error(px, (~m) & 0xffff) + box_error(px, m);
        if e < pe { pe = e; part = p; }
    }
    let m1 = PARTITION2[part];
    var f0 = fit_subset(px, (~m1) & 0xffff, 6, 3, false, true);
    var f1 = fit_subset(px, m1, 6, 3, false, true);
    // Each subset's anchor drops its index's top bit.
    if f0.idx[0] >= 4 {
        f0 = Fit7 { q0: f0.q1, q1: f0.q0, p0: f0.p0, p1: f0.p1, idx: f0.idx, err: f0.err };
        for i: u32 in 16 { f0.idx[i] = 7 - f0.idx[i]; }
    }
    let anchor = ANCHOR2[part];
    if f1.idx[anchor] >= 4 {
        f1 = Fit7 { q0: f1.q1, q1: f1.q0, p0: f1.p0, p1: f1.p1, idx: f1.idx, err: f1.err };
        for i: u32 in 16 { f1.idx[i] = 7 - f1.idx[i]; }
    }
    var b: u32[4] = [0; 4];
    b = put(b, 0, 2, 2);
    b = put(b, 2, 6, part);
    var pos: u32 = 8;
    for ch: u32 in 3 {
        for e: u32 in 4 {
            let q = if e == 0 { f0.q0 } else if e == 1 { f0.q1 } else if e == 2 { f1.q0 } else { f1.q1 };
            let v = if ch == 0 { q.x } else if ch == 1 { q.y } else { q.z };
            b = put(b, pos, 6, v as u32);
            pos += 6;
        }
    }
    b = put(b, pos, 1, f0.p0 as u32);
    b = put(b, pos + 1, 1, f1.p0 as u32);
    pos += 2;
    for i: u32 in 16 {
        let in1 = (m1 >> i) & 1 != 0;
        let n: u32 = if i == 0 || i == anchor { 2 } else { 3 };
        b = put(b, pos, n, if in1 { f1.idx[i] } else { f0.idx[i] });
        pos += n;
    }
    Choice { bits: uint4 { b[0], b[1], b[2], b[3] }, err: f0.err + f1.err }
}

fn encode_bc7(px: int4[16]) -> uint4 {
    let six = bc7_mode6(px);
    var opaque = true;
    for c in px { if c.w != 255 { opaque = false; } }
    if !opaque { return six.bits; }
    let one = bc7_mode1(px);
    if one.err < six.err { one.bits } else { six.bits }
}

// Decodes modes 1 and 6, the ones encode_bc7 makes; any other block
// decodes to magenta, which a test notices.
fn decode_bc7(blk: uint4) -> int4[16] {
    let b: u32[4] = [blk.x, blk.y, blk.z, blk.w];
    var out: int4[16] = [int4 { 255, 0, 255, 255 }; 16];
    if b[0] & 127 == 64 {
        var q0 = int4_0; var q1 = int4_0;
        var pos: u32 = 7;
        for ch: u32 in 4 {
            let a = get(b, pos, 7) as i32;
            let c = get(b, pos + 7, 7) as i32;
            if ch == 0 { q0.x = a; q1.x = c; } else if ch == 1 { q0.y = a; q1.y = c; }
            else if ch == 2 { q0.z = a; q1.z = c; } else { q0.w = a; q1.w = c; }
            pos += 14;
        }
        let e0 = expand4(q0, get(b, pos, 1) as i32, 7);
        let e1 = expand4(q1, get(b, pos + 1, 1) as i32, 7);
        pos += 2;
        for i: u32 in 16 {
            let n: u32 = if i == 0 { 3 } else { 4 };
            out[i] = blend(e0, e1, W4[get(b, pos, n)]);
            pos += n;
        }
    } else if b[0] & 3 == 2 {
        let part = get(b, 2, 6);
        var q: int4[4] = [int4 { 0, 0, 0, 0 }; 4];
        var pos: u32 = 8;
        for ch: u32 in 3 {
            for e: u32 in 4 {
                let v = get(b, pos, 6) as i32;
                if ch == 0 { q[e].x = v; } else if ch == 1 { q[e].y = v; } else { q[e].z = v; }
                pos += 6;
            }
        }
        let pa = get(b, pos, 1) as i32;
        let pb = get(b, pos + 1, 1) as i32;
        pos += 2;
        var e: int4[4] = [int4_0; 4];
        for k: u32 in 4 {
            e[k] = expand4(q[k], if k < 2 { pa } else { pb }, 6);
            e[k].w = 255;
        }
        let m1 = PARTITION2[part];
        let anchor = ANCHOR2[part];
        for i: u32 in 16 {
            let n: u32 = if i == 0 || i == anchor { 2 } else { 3 };
            let s: u32 = (m1 >> i) & 1;
            out[i] = blend(e[2 * s], e[2 * s + 1], W3[get(b, pos, n)]);
            pos += n;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The formats, by number, as the program and its shaders name them.

let FORMAT_BC1: u32 = 0;
let FORMAT_BC3: u32 = 1;
let FORMAT_BC4: u32 = 2;
let FORMAT_BC5: u32 = 3;
let FORMAT_BC7: u32 = 4;

// A block of `format`, as four words (BC1 and BC4 use the first two).
fn encode(format: u32, px: int4[16]) -> uint4 {
    if format == FORMAT_BC1 {
        let b = encode_bc1(px);
        return uint4 { b.x, b.y, 0, 0 };
    }
    if format == FORMAT_BC4 {
        let b = encode_bc4(channel(px, 0));
        return uint4 { b.x, b.y, 0, 0 };
    }
    if format == FORMAT_BC3 { return encode_bc3(px); }
    if format == FORMAT_BC5 { return encode_bc5(px); }
    encode_bc7(px)
}

fn decode(format: u32, b: uint4) -> int4[16] {
    if format == FORMAT_BC1 { return decode_bc1(uint2 { b.x, b.y }); }
    if format == FORMAT_BC4 {
        let v = decode_bc4(uint2 { b.x, b.y });
        var out: int4[16] = [int4_0; 16];
        for i: u32 in 16 { out[i] = int4 { v[i], 0, 0, 255 }; }
        return out;
    }
    if format == FORMAT_BC3 { return decode_bc3(b); }
    if format == FORMAT_BC5 { return decode_bc5(b); }
    decode_bc7(b)
}

// The bytes of a block of `format`.
fn block_bytes(format: u32) -> u32 { if format == FORMAT_BC1 || format == FORMAT_BC4 { 8 } else { 16 } }
