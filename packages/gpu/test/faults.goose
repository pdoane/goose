// Checks that abort, on the GPU (docs/design/gpu_language.md §6): a shader
// whose check fails records the fault, and the host ends the program with
// the place and the values, as the CPU would.
//
//     goose test/faults.goose                  no fault: checks pass
//     goose test/faults.goose -- index         an index out of range
//     goose test/faults.goose -- divide        a division by zero
//     goose test/faults.goose -- tag           an ADT's tag that is no variant
//     goose test/faults.goose -- full          a push onto a full limited array
//     goose test/faults.goose -- empty         a pop from an empty one
import gpu from "..";
import std;
import os;

enum Op { Add { k: u32 }, Scale { k: u32 } }

struct Push {
    ops: gpu::StructuredBuffer<Op>,
    out: gpu::RWStructuredBuffer<u32>,
    at: u32,
    by: u32,
    pushes: u32,
}

compute(64) fn run(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    let table: u32[4] = [10, 20, 30, 40];
    // Invocation 37 reads at `at`, and divides by `by`.
    let k = if i == 37 { p.at } else { i % 4 };
    let v = table[k] / (if i == 37 { p.by } else { 1 });
    var r = match gpu::load(p.ops, i % 2) { Add a => v + a.k, Scale s => v * s.k };
    // Invocation 37 pushes `pushes` onto a stack of 4, then pops 3.
    var stack: u32[..4] = [];
    for k: u32 in (if i == 37 { p.pushes } else { 3 }) { stack.push(k); }
    for k: u32 in 3 { r += stack.pop(); }
    gpu::store(p.out, i, r);
}

fn main() {
    let argv = args();
    let mode = if argv.len > 1 { str(argv[1]) } else { str("") };
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    var ops: Op[>..] = [Op.Add { k: 1 }, Op.Scale { k: 3 }];
    var bytes = str(bytes_of(ops));
    // An Op's tag is its first byte: 7 is no variant.
    if mode == "tag" { bytes[0] = 7; }
    let ob = gpu::buffer(gpu::BUFFER_READ, bytes);
    let out = gpu::buffer(gpu::BUFFER_WRITE, 64 * 4);
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(run));
    gpu::push_constants(Push { ops: gpu::structured<Op>(ob), out: gpu::rw_structured<u32>(out),
                               at: if mode == "index" { 9 } else { 2 },
                               by: if mode == "divide" { 0 } else { 2 },
                               pushes: if mode == "full" { 5 } else if mode == "empty" { 2 } else { 3 } });
    gpu::dispatch(1, 1, 1);
    gpu::end_compute();
    let r = gpu::read_buffer(out, 0, 64 * 4);
    gpu::wait(gpu::submit());
    var got: u32[>..] = [];
    gpu::values(r, got);
    print(gpu::driver(), ": no fault, invocation 37 gave ", got[37]);
    gpu::close();
}
