// Global arrays on the GPU, and references into them (docs/design/
// gpu_language.md §9, §10): a pool of nodes linked by self-relative
// references, built on the CPU, uploaded byte for byte, and walked by the
// same function on the GPU and the CPU, with a stack of `in pool` links,
// whose sums must agree.
import gpu from "..";
import std;

struct Node { w: f32, next: Node&<u32>?, jump: Node&<u32>?, k: u32 }
var nodes: Node[>..] = [];
var weights: f32[>..] = [];

fn weigh(n: const Node&) -> f32 { n.w * weights[n.k % weights.len as u32] }

// Every node reachable from `first` by next and jump, depth first, a
// visit a node per path to it, up to a budget.
fn walk(first: const Node&) -> f32 {
    var sum: f32 = 0.0;
    var todo: (const Node&<u32 in nodes>)[..16] = [];
    todo.push(first);
    var visits: u32 = 0;
    while todo.len > 0 && visits < 40 {
        let n .= todo.pop();
        visits += 1;
        sum += weigh(n);
        let a .= n.next;
        if a && todo.len < 15 { todo.push(a); }
        let b .= n.jump;
        if b && todo.len < 15 { todo.push(b); }
    }
    sum
}

fn total() -> f32 {
    var s: f32 = 0.0;
    for &n in nodes { s += n.w; }
    s
}

struct Push { out: gpu::RWStructuredBuffer<f32> }

compute(64) fn run(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    if i < nodes.len as u32 { gpu::store(p.out, i, total() + walk(nodes[i])); }
}

fn pcg(x: u32) -> u32 {
    var h = x * 747796405 + 2891336453;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737;
    (h >> 22) ^ h
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let n = 200;
    for i in n {
        nodes.push(Node { w: (pcg(i as u32) % 100) * 0.25, next: null, jump: null, k: pcg(i as u32 + 7) % 13 });
    }
    // A chain through every third node, and jumps anywhere.
    for i in n {
        if i + 3 < n { nodes[i].next .= &nodes[i + 3]; }
        if pcg(i as u32 + 99) % 4 == 0 { nodes[i].jump .= &nodes[pcg(i as u32 + 5) % n as u32]; }
    }
    for k in 13 { weights.push(1.0 + k * 0.5); }
    gpu::upload(gpu_global_id(nodes), nodes);
    gpu::upload(gpu_global_id(weights), weights);
    let out = gpu::buffer(gpu::BUFFER_WRITE, n * 4);
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(run));
    gpu::push_constants(Push { out: gpu::rw_structured<f32>(out) });
    gpu::dispatch((n + 63) / 64, 1, 1);
    gpu::end_compute();
    let r = gpu::read_buffer(out, 0, n * 4);
    gpu::wait(gpu::submit());
    var got: f32[>..] = [];
    guard gpu::values(r, got) else { abort("gpu: a read back is not ready"); }
    var wrong = 0;
    for i in n {
        let want = total() + walk(nodes[i]);
        if abs(got[i] - want) > 1e-2 {
            if wrong < 4 { print("node ", i, ": ", got[i], " not ", want); }
            wrong++;
        }
    }
    print(gpu::driver(), ": ", n, " walks of a linked pool, ", wrong, " differ from the CPU");
    gpu::close();
}
