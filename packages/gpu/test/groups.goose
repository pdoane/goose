// Group memory, barriers, atomics and waves (docs/design/gpu.md §5.4),
// checked against the CPU: each group of 256 counts its keys' low four
// bits with atomics in group memory and adds them into a buffer with
// buffer atomics, computes each key's exclusive prefix sum across the
// group from wave prefix sums and each wave's total, and takes the min and
// max of the keys; and every invocation races to compare-exchange one
// word, which exactly one wins.
import gpu from "..";
import std;

var counts: gpu::GroupShared<u32[16]>;
var totals: gpu::GroupShared<u32[64]>;

struct Push {
    keys: gpu::StructuredBuffer<u32>,
    prefix: gpu::RWStructuredBuffer<u32>,
    stats: gpu::RWStructuredBuffer<u32>,     // 16 counts, min, max, the race, its winners, wave size
}

// The sum of x over the invocations before this one in its group.
fn group_prefix(x: u32, tid: u32) -> u32 {
    let lanes = gpu::wave_size();
    let before = gpu::wave_prefix_sum(x);
    if gpu::wave_lane() == lanes - 1 { totals.value[tid / lanes] = before + x; }
    gpu::group_barrier();
    var base: u32 = 0;
    for w: u32 in 0..tid / lanes { base += totals.value[w]; }
    gpu::group_barrier();
    base + before
}

compute(256) fn groups(p: Push) {
    let tid = gpu::group_index();
    let i = gpu::dispatch_thread_id().x;
    if tid < 16 { counts.value[tid] = 0; }
    gpu::group_barrier();
    let key = gpu::load(p.keys, i);
    gpu_atomic(gpu::ATOMIC_ADD, counts.value[key & 15], 1);
    gpu::group_barrier();
    if tid < 16 { gpu::atomic_add(p.stats, tid, counts.value[tid]); }
    gpu::store(p.prefix, i, group_prefix(key & 255, tid));
    gpu::atomic_min(p.stats, 16, key);
    gpu::atomic_max(p.stats, 17, key);
    if gpu::compare_exchange(p.stats, 18, 0, i + 1) == 0 { gpu::atomic_add(p.stats, 19, 1); }
    if i == 0 { gpu::store(p.stats, 20, gpu::wave_size()); }
}

fn pcg(x: u32) -> u32 {
    var h = x * 747796405 + 2891336453;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737;
    (h >> 22) ^ h
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let n = 16384;
    var keys: u32[>..] = [];
    for k in n { keys.push(pcg(k as u32) % 100000 + 7); }
    let kb = gpu::buffer(gpu::BUFFER_READ, bytes_of(keys));
    let prefix = gpu::buffer(gpu::BUFFER_WRITE, n * 4);
    var init: u32[>..] = [];
    for k in 21 { init.push(if k == 16 { 0xffffffff } else { 0 }); }
    let stats = gpu::buffer(gpu::BUFFER_WRITE, bytes_of(init));
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(groups));
    gpu::push_constants(Push { keys: gpu::structured<u32>(kb), prefix: gpu::rw_structured<u32>(prefix),
                               stats: gpu::rw_structured<u32>(stats) });
    gpu::dispatch(n / 256, 1, 1);
    gpu::end_compute();
    let rp = gpu::read_buffer(prefix, 0, n * 4);
    let rs = gpu::read_buffer(stats, 0, 21 * 4);
    gpu::wait(gpu::submit());
    var got: u32[>..] = [];
    var st: u32[>..] = [];
    guard gpu::values(rp, got) && gpu::values(rs, st) else { abort("gpu: a read back is not ready"); }

    var wrong = 0;
    var hist: u32[16] = [0; 16];
    var lo: u32 = 0xffffffff;
    var hi: u32 = 0;
    var run: u32 = 0;
    for k in n {
        if k % 256 == 0 { run = 0; }
        if got[k] != run {
            if wrong < 4 { print("prefix ", k, ": ", got[k], " not ", run); }
            wrong++;
        }
        run += keys[k] & 255;
        hist[keys[k] & 15] += 1;
        lo = min(lo, keys[k]);
        hi = max(hi, keys[k]);
    }
    for b in 16 {
        if st[b] != hist[b] {
            print("count ", b, ": ", st[b], " not ", hist[b]);
            wrong++;
        }
    }
    if st[16] != lo || st[17] != hi {
        print("min, max: ", st[16], ", ", st[17], " not ", lo, ", ", hi);
        wrong++;
    }
    if st[18] == 0 || st[18] > n || st[19] != 1 {
        print("the race: ", st[18], " won by ", st[19]);
        wrong++;
    }
    print(gpu::driver(), ": ", n, " keys, waves of ", st[20], ", ", wrong, " wrong");
    gpu::close();
}
