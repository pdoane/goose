// `return … from`: a flag naming the target, which every call that can
// carry the return tests, and a channel per target for its values, here
// two of them.
import gpu_shader;

fn check(x: u32, want: u32, k: u32) {
    if x == want { return k, true from find; }
}

fn scan(xs: u32[4], want: u32) {
    for k: u32 in 4 { check(xs[k], want, k); }
}

fn find(xs: u32[4], want: u32) -> u32, bool {
    scan(xs, want);
    return 0, false;
}

struct Push { out: gpu::RWStructuredBuffer<u32> }

compute fn return_from(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    let xs: u32[4] = [7, i, 3, 9];
    let at, ok = find(xs, 3);
    gpu::store(p.out, i, if ok { at } else { 99 });
}

fn main() { print(gpu_entry(return_from).len > 0); }
