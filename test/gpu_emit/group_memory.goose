// Group memory, barriers, atomics and waves: a group's histogram counted
// with atomics in group memory, a prefix sum across the group from wave
// prefix sums and each wave's total, and buffer atomics on the results.
import gpu_shader;

var counts: gpu::GroupShared<u32[16]>;
var totals: gpu::GroupShared<u32[8]>;

struct Push { keys: gpu::StructuredBuffer<u32>, out: gpu::RWStructuredBuffer<u32> }

// The sum of x over the lanes before this one in the whole group.
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

compute(256) fn histogram(p: Push) {
    let tid = gpu::group_index();
    if tid < 16 { counts.value[tid] = 0; }
    gpu::group_barrier();
    let key = gpu::load(p.keys, gpu::dispatch_thread_id().x);
    let old = gpu_atomic(gpu::ATOMIC_ADD, counts.value[key & 15], 1);
    gpu_atomic(gpu::ATOMIC_MAX, counts.value[15], old);
    gpu_atomic_cas(counts.value[0], 0, 1);
    gpu::group_barrier();
    let at = group_prefix(key & 1, tid);
    if tid < 16 { gpu::atomic_add(p.out, tid, counts.value[tid]); }
    gpu::atomic_max(p.out, 16, at);
    gpu::compare_exchange(p.out, 17, 0, gpu::wave_count(key > 7));
    if gpu::wave_all(key < 100) && gpu::wave_any(key == 3) {
        gpu::store(p.out, 18 + gpu::wave_prefix_count(true), gpu::wave_read_lane(key, 0) +
                   gpu::wave_read_first(key) + gpu::wave_min(key) + gpu::wave_max(key) + gpu::wave_sum(key));
    }
}

fn main() { print(gpu_entry(histogram).len > 0); }
