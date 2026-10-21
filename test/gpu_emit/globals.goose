// Global arrays read through the table of uploaded globals, and references
// into them: elements bound by reference, a self-relative link loaded as
// its field's offset plus what it stores, an `in pool` link as it is, a
// limited array of links as a stack, and a function given a place.
import gpu_shader;

struct Node { w: f32, next: Node&<u32>?, jump: Node&<u32 in nodes>?, k: u32 }
var nodes: Node[>..] = [];
var weights: f32[>..] = [];

fn weigh(n: const Node&) -> f32 { n.w * weights[n.k % weights.len as u32] }

fn walk(first: const Node&) -> f32 {
    var sum: f32 = 0.0;
    var todo: (const Node&<u32 in nodes>)[..8] = [];
    todo.push(first);
    while todo.len > 0 {
        let n .= todo.pop();
        sum += weigh(n);
        let a .= n.next;
        if a { todo.push(a); }
        let b .= n.jump;
        if b { todo.push(b); }
    }
    sum
}

struct Push { out: gpu::RWStructuredBuffer<f32> }

compute fn globals(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    var s: f32 = 0.0;
    for &n in nodes { s += n.w; }
    gpu::store(p.out, i, s + walk(nodes[i % nodes.len as u32]));
}

fn main() { print(gpu_entry(globals).len > 0); }
