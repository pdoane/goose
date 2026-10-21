// A GPU radix sort: key-value pairs of u32 sorted by key, stably, in four
// passes of 8 bits each from the lowest. A pass is three dispatches:
//
//   histogram  each tile of 4096 keys counts its digits, with atomics in
//              group memory
//   scan       one group turns every tile's counts into where each tile's
//              keys of each digit go, a prefix sum in digit-major order
//   scatter    each tile sorts 256 keys at a time by digit in group memory,
//              eight one-bit splits whose positions are group-wide prefix
//              sums built from wave prefix sums, and writes each key to its
//              place, after its digit's keys from earlier tiles and chunks
//
// The result is checked against a CPU radix sort of the same pairs, which
// is stable too, so the two agree value for value.
//
//     goose radix.goose -- [count]       (1048576 by default)

import gpu from "../..";
import std;
import os;

let THREADS: u32 = 256;
let CHUNKS: u32 = 16;          // a tile: CHUNKS of THREADS keys

struct Pass {
    keys: gpu::StructuredBuffer<u32>,
    vals: gpu::StructuredBuffer<u32>,
    keys_out: gpu::RWStructuredBuffer<u32>,
    vals_out: gpu::RWStructuredBuffer<u32>,
    hist: gpu::RWStructuredBuffer<u32>,     // count of digit d in tile t at d * tiles + t
    n: u32,
    shift: u32,
    tiles: u32,
}

fn digit(key: u32, shift: u32) -> u32 { (key >> shift) & 255 }

var counts: gpu::GroupShared<u32[256]>;
var waves: gpu::GroupShared<u32[64]>;

// A prefix sum across the group of 256: the sum of x before this
// invocation, and over all of them.
struct Scan { before: u32, total: u32 }

fn group_scan(x: u32, tid: u32) -> Scan {
    let lanes = gpu::wave_size();
    let before = gpu::wave_prefix_sum(x);
    if gpu::wave_lane() == lanes - 1 { waves.value[tid / lanes] = before + x; }
    gpu::group_barrier();
    var base: u32 = 0;
    var total: u32 = 0;
    let mine = tid / lanes;
    for w: u32 in 0..THREADS / lanes {
        if w < mine { base += waves.value[w]; }
        total += waves.value[w];
    }
    gpu::group_barrier();
    Scan { before: base + before, total: total }
}

compute(256) fn histogram(p: Pass) {
    let tid = gpu::group_index();
    let tile = gpu::group_id().x;
    counts.value[tid] = 0;
    gpu::group_barrier();
    for c: u32 in CHUNKS {
        let i = (tile * CHUNKS + c) * THREADS + tid;
        if i < p.n { gpu_atomic(gpu::ATOMIC_ADD, counts.value[digit(gpu::load(p.keys, i), p.shift)], 1); }
    }
    gpu::group_barrier();
    gpu::store(p.hist, tid * p.tiles + tile, counts.value[tid]);
}

// One group: each invocation sums a run of the counts, the runs are
// scanned across the group, and each invocation writes its run's prefix.
compute(256) fn scan(p: Pass) {
    let tid = gpu::group_index();
    let m = 256 * p.tiles;
    let run = (m + THREADS - 1) / THREADS;
    let lo = min(tid * run, m);
    let hi = min(lo + run, m);
    var sum: u32 = 0;
    for i: u32 in lo..hi { sum += gpu::load(p.hist, i); }
    var at = group_scan(sum, tid).before;
    for i: u32 in lo..hi {
        let c = gpu::load(p.hist, i);
        gpu::store(p.hist, i, at);
        at += c;
    }
}

var keys: gpu::GroupShared<u32[256]>;
var vals: gpu::GroupShared<u32[256]>;
var valid: gpu::GroupShared<u32[256]>;
var starts: gpu::GroupShared<u32[256]>;

compute(256) fn scatter(p: Pass) {
    let tid = gpu::group_index();
    let tile = gpu::group_id().x;
    // Where this tile's keys of digit tid go next.
    counts.value[tid] = gpu::load(p.hist, tid * p.tiles + tile);
    gpu::group_barrier();
    for c: u32 in CHUNKS {
        let first = (tile * CHUNKS + c) * THREADS;
        if first >= p.n { break; }
        let i = first + tid;
        var key: u32 = 0xffffffff;
        var val: u32 = 0;
        var ok: u32 = 0;
        if i < p.n {
            key = gpu::load(p.keys, i);
            val = gpu::load(p.vals, i);
            ok = 1;
        }
        // Stably by digit: a split on each of its bits, the zeros first.
        for bit: u32 in 8 {
            let b = (digit(key, p.shift) >> bit) & 1;
            let s = group_scan(1 - b, tid);
            let to = if b == 0 { s.before } else { s.total + tid - s.before };
            keys.value[to] = key;
            vals.value[to] = val;
            valid.value[to] = ok;
            gpu::group_barrier();
            key = keys.value[tid];
            val = vals.value[tid];
            ok = valid.value[tid];
            gpu::group_barrier();
        }
        // Each digit's first key in the sorted chunk, and every key's place
        // after its digit's from before.
        let d = digit(key, p.shift);
        let starts_run = tid == 0 || digit(keys.value[tid - 1], p.shift) != d;
        if starts_run { starts.value[d] = tid; }
        gpu::group_barrier();
        let rank = tid - starts.value[d];
        if ok != 0 {
            let to = counts.value[d] + rank;
            gpu::store(p.keys_out, to, key);
            gpu::store(p.vals_out, to, val);
        }
        gpu::group_barrier();
        let ends_run = tid == THREADS - 1 || digit(keys.value[tid + 1], p.shift) != d;
        if ends_run { counts.value[d] += rank + 1; }
        gpu::group_barrier();
    }
}

fn pcg(x: u32) -> u32 {
    var h = x * 747796405 + 2891336453;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737;
    (h >> 22) ^ h
}

// The CPU's sort: the same four stable passes, as counting sorts.
fn cpu_sort(keys: u32[>..]&, vals: u32[>..]&) {
    var k2: u32[>..] = [];
    var v2: u32[>..] = [];
    k2.resize(keys.len, 0);
    v2.resize(keys.len, 0);
    for pass in 4 {
        let shift = (pass * 8) as u32;
        var count: i64[256] = [0; 256];
        for k in keys { count[digit(k, shift)] += 1; }
        var at: i64 = 0;
        for d in 256 {
            let c = count[d];
            count[d] = at;
            at += c;
        }
        for i in keys.len {
            let d = digit(keys[i], shift);
            k2[count[d]] = keys[i];
            v2[count[d]] = vals[i];
            count[d] += 1;
        }
        for i in keys.len {
            keys[i] = k2[i];
            vals[i] = v2[i];
        }
    }
}

fn main() {
    var n: i64 = 1 << 20;
    if arg_count() > 1 {
        var a: u8[>..] = [];
        arg(1, a);
        let v, _ = parse_int(a);
        n = v;
    }
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    // Keys with many repeats, so that stability shows: the values are the
    // keys' first places.
    var keys: u32[>..] = [];
    var vals: u32[>..] = [];
    for i in n {
        keys.push(if i % 2 == 0 { pcg(i as u32) } else { pcg(i as u32) & 0xffff });
        vals.push(i as u32);
    }
    let tile: i64 = THREADS * CHUNKS;
    let tiles = ((n + tile - 1) / tile) as u32;
    var kb = [gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_WRITE, bytes_of(keys)),
              gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_WRITE, n * 4)];
    var vb = [gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_WRITE, bytes_of(vals)),
              gpu::buffer(gpu::BUFFER_READ | gpu::BUFFER_WRITE, n * 4)];
    let hist = gpu::buffer(gpu::BUFFER_WRITE, (256 * tiles) as i64 * 4);
    let ph = gpu::compute_pipeline(histogram);
    let ps = gpu::compute_pipeline(scan);
    let pc = gpu::compute_pipeline(scatter);

    // Sorted twice, the second timed: the first pays for each pipeline's
    // first use. Four passes leave the result where it started.
    var gpums = 0.0;
    for run in 2 {
        if run == 1 {
            gpu::update_buffer(kb[0], bytes_of(keys));
            gpu::update_buffer(vb[0], bytes_of(vals));
        }
        let t0 = clock();
        gpu::begin_compute();
        for pass in 4 {
            let a = pass % 2;
            let push = Pass { keys: gpu::structured<u32>(kb[a]), vals: gpu::structured<u32>(vb[a]),
                              keys_out: gpu::rw_structured<u32>(kb[1 - a]),
                              vals_out: gpu::rw_structured<u32>(vb[1 - a]),
                              hist: gpu::rw_structured<u32>(hist), n: n as u32,
                              shift: (pass * 8) as u32, tiles: tiles };
            gpu::bind(ph);
            gpu::push_constants(push);
            gpu::dispatch(tiles, 1, 1);
            gpu::bind(ps);
            gpu::dispatch(1, 1, 1);
            gpu::bind(pc);
            gpu::dispatch(tiles, 1, 1);
        }
        gpu::end_compute();
        gpu::wait(gpu::submit());
        gpums = (clock() - t0) * 1000.0;
    }
    let rk = gpu::read_buffer(kb[0], 0, n * 4);
    let rv = gpu::read_buffer(vb[0], 0, n * 4);
    gpu::wait(gpu::submit());
    var gk: u32[>..] = [];
    var gv: u32[>..] = [];
    guard gpu::values(rk, gk) && gpu::values(rv, gv) else { abort("gpu: a read back is not ready"); }

    let t0 = clock();
    cpu_sort(keys, vals);
    let cpums = (clock() - t0) * 1000.0;
    var wrong: i64 = 0;
    for i in n {
        if gk[i] != keys[i] || gv[i] != vals[i] {
            if wrong < 4 { print("pair ", i, ": ", gk[i], " ", gv[i], " not ", keys[i], " ", vals[i]); }
            wrong++;
        }
    }
    // Throughput of each, in millions of pairs a second.
    var grate: u8[>..] = [];
    format_flt(grate, n / gpums / 1000.0, 0);
    var crate: u8[>..] = [];
    format_flt(crate, n / cpums / 1000.0, 0);
    print(gpu::driver(), ": ", n, " pairs in ", tiles, " tiles; GPU ", grate, " M pairs/s, CPU ",
          crate, " M pairs/s; ", wrong, " differ");
    gpu::close();
    if wrong != 0 { abort("the GPU's sort differs from the CPU's"); }
}
