// Limited arrays: a length and the elements, push and pop within the
// capacity, indexing checked against the length, a for over the elements
// there are, and a buffer of them read as Goose packs them, the elements
// after the length's byte.
import gpu_shader;

struct Push { lists: gpu::StructuredBuffer<u32[..4]>, out: gpu::RWStructuredBuffer<u32> }

compute fn limited(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    var stack: u32[..8] = [i];
    var sum: u32 = 0;
    while stack.len > 0 {
        let n = stack.pop();
        sum += n;
        if n < 20 { stack.push(n * 2 + 1); }
    }
    let list = gpu::load(p.lists, i);
    for x in list { sum += x; }
    if list.len > 1 { sum += list[i % 2] + list[list.len as u32 - 1]; }
    gpu::store(p.out, i, sum + list.cap as u32);
}

fn main() { print(gpu_entry(limited).len > 0); }
