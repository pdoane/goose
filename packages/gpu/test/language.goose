// The language past the shader subset (docs/design/gpu_language.md), run
// on the GPU and on the CPU alike: one function, `work`, computes a record
// per invocation, and the two runs must agree exactly. Each feature that
// lands adds to it.
import gpu from "..";
import std;

struct Pair { a: i32, b: u32 }
struct Body { pos: float3, vel: float3, hits: u32 }

// References as parameters: written through, read only, passed on, and a
// field's and an array element's storage.
fn step(b: Body&, dt: f32) {
    b.vel = b.vel + float3 { 0.0, -9.8, 0.0 } * dt;
    b.pos = b.pos + b.vel * dt;
    if b.pos.y < 0.0 {
        b.pos.y = -b.pos.y;
        b.vel.y = -b.vel.y * 0.5;
        bump(b.hits);
    }
}

fn bump(n: u32&) { n += 1; }

fn energy(b: const Body&) -> f32 { dot(b.vel, b.vel) * 0.5 + 9.8 * b.pos.y }

fn swap_in(xs: u32[4]&, i: u32, j: u32) {
    let t = xs[i];
    xs[i] = xs[j];
    xs[j] = t;
}

// Several results.
fn divmod(a: u32, b: u32) -> u32, u32 { return a / b, a % b; }

fn minmax(xs: u32[4]) -> u32, u32 {
    var lo = xs[0];
    var hi = xs[0];
    for x in xs {
        lo = min(lo, x);
        hi = max(hi, x);
    }
    return lo, hi;
}

fn order(p: Pair) -> Pair, bool {
    if p.a < 0 { return Pair { a: -p.a, b: p.b }, true; }
    return p, false;
}

// A long-distance return: the search's answer leaves check, two calls
// down, for find.
fn check(x: u32, want: u32, k: u32) {
    if x == want { return k as i32 from find; }
}

fn scan(xs: u32[8], want: u32) {
    for k: u32 in 8 { check(xs[k], want, k); }
}

fn find(xs: u32[8], want: u32) -> i32 {
    scan(xs, want);
    -1
}

// ADTs: built, matched, dispatched on by case functions, and read from a
// buffer the CPU wrote, in Goose's packed layout.
enum Shape {
    Circle { r: f32 },
    Rect { w: f32, h: f32 },
    Tri { a: float2, b: float2, c: float2 },
    Point,
}

fn area(c: Shape.Circle) -> f32 { 3.0 * c.r * c.r }
fn area(r: Shape.Rect) -> f32 { r.w * r.h }
fn area(t: Shape.Tri) -> f32 {
    let u = t.b - t.a;
    let v = t.c - t.a;
    abs(u.x * v.y - u.y * v.x) * 0.5
}
fn area(p: Shape.Point) -> f32 { 0.0 }

fn shape_of(i: u32) -> Shape {
    let f: f32 = i % 9;
    match i % 4 {
        0 => Shape.Circle { r: f },
        1 => Shape.Rect { w: f, h: 2.0 },
        2 => Shape.Tri { a: float2_0, b: float2 { f, 0.0 }, c: float2 { 0.0, 3.0 } },
        _ => Shape.Point,
    }
}

fn corners(s: Shape) -> u32 {
    match s {
        Circle c => if c.r > 4.0 { 100 } else { 0 },
        Rect, Tri => 4,
        Point => 1,
    }
}

// An integer match, and a break out of a loop from inside one.
fn bucket(x: u32) -> u32 {
    match x % 12 { 0 => 10, 1..4 => 20, 5, 7 => 30, _ => 40 }
}

fn first_big(xs: u32[8]) -> u32 {
    var at: u32 = 99;
    for k: u32 in 8 {
        match xs[k] {
            0..5 => {},
            _ => { at = k; break; },
        }
    }
    at
}

struct Result {
    pos: float3,
    hits: u32,
    energy: f32,
    q: u32,
    r: u32,
    lo: u32,
    hi: u32,
    sorted: u32[4],
    flipped: u32,
    a: i32,
    found: i32,
    area: f32,
    stored: f32,
    corners: u32,
    bucket: u32,
    big: u32,
}

fn work(i: u32, stored: Shape) -> Result {
    var b = Body { pos: float3 { 0.0, 2.0 + i * 0.25, 0.0 }, vel: float3 { 1.0, 0.0, 0.0 }, hits: 0 };
    for k: u32 in 40 { step(b, 0.05); }
    let q, r = divmod(i * 7 + 3, 5);
    var xs: u32[4] = [i % 7, (i * 3) % 5, 9 - i % 4, i % 2];
    // A bubble sort by references to elements.
    for pass: u32 in 3 {
        for k: u32 in 3 - pass { if xs[k] > xs[k + 1] { swap_in(xs, k, k + 1); } }
    }
    let lo, hi = minmax(xs);
    var p = Pair { a: 5 - i as i32, b: i };
    let o, flipped = order(p);
    // A reference local names what it is bound to.
    let table: u32[8] = [3, 1, 4, 1, 5, 9, 2, 6];
    let found = find(table, i % 11);
    let first .= xs[0];
    first += 100;
    Result { pos: b.pos, hits: b.hits, energy: energy(b), q: q, r: r, lo: lo, hi: hi, sorted: xs,
             flipped: if flipped { 1 } else { 0 }, a: o.a, found: found, area: area(shape_of(i)),
             stored: area(stored), corners: corners(shape_of(i)) + corners(stored), bucket: bucket(i),
             big: first_big(table) }
}

struct Push { out: gpu::RWStructuredBuffer<Result>, shapes: gpu::StructuredBuffer<Shape>, n: u32 }

compute(64) fn run(p: Push) {
    let i = gpu::dispatch_thread_id().x;
    if i < p.n { gpu::store(p.out, i, work(i, gpu::load(p.shapes, i))); }
}

// The shapes the CPU uploads: every kind, in an order of their own.
fn stored(i: u32) -> Shape { shape_of(i * 7 + 1) }

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    let n: u32 = 256;
    let one = [work(0, stored(0))];
    let size = n * bytes_of(one).len;
    let out = gpu::buffer(gpu::BUFFER_WRITE, size);
    var shapes: Shape[>..] = [];
    for i in n { shapes.push(stored(i)); }
    let sb = gpu::buffer(gpu::BUFFER_READ, bytes_of(shapes));
    gpu::begin_compute();
    gpu::bind(gpu::compute_pipeline(run));
    gpu::push_constants(Push { out: gpu::rw_structured<Result>(out), shapes: gpu::structured<Shape>(sb), n: n });
    gpu::dispatch(n / 64, 1, 1);
    gpu::end_compute();
    let r = gpu::read_buffer(out, 0, size);
    gpu::wait(gpu::submit());
    var got: Result[>..] = [];
    guard gpu::values(r, got) else { abort("gpu: a read back is not ready"); }
    var wrong = 0;
    for i in n {
        let want = work(i, stored(i));
        let g = got[i];
        // Float results within a rounding or two of the CPU's: the GPU
        // compilers may fuse a multiply and an add.
        let close = abs(g.pos.x - want.pos.x) < 1e-3 && abs(g.pos.y - want.pos.y) < 1e-3 &&
                    abs(g.energy - want.energy) < 1e-2;
        if !close || g.hits != want.hits || g.q != want.q || g.r != want.r || g.lo != want.lo ||
           g.hi != want.hi || g.sorted != want.sorted || g.flipped != want.flipped || g.a != want.a ||
           g.found != want.found || abs(g.area - want.area) > 1e-3 || abs(g.stored - want.stored) > 1e-3 ||
           g.corners != want.corners || g.bucket != want.bucket || g.big != want.big {
            if wrong < 4 {
                print("invocation ", i, ": hits ", g.hits, "/", want.hits, " q ", g.q, "/", want.q, " lo ", g.lo,
                      "/", want.lo, " sorted ", g.sorted[0], ",", g.sorted[1], ",", g.sorted[2], ",", g.sorted[3],
                      " / ", want.sorted[0], ",", want.sorted[1], ",", want.sorted[2], ",", want.sorted[3],
                      " a ", g.a, "/", want.a, " pos.y ", g.pos.y, "/", want.pos.y, " area ", g.area, "/",
                      want.area, " stored ", g.stored, "/", want.stored, " corners ", g.corners, "/",
                      want.corners, " bucket ", g.bucket, "/", want.bucket, " big ", g.big, "/", want.big);
            }
            wrong++;
        }
    }
    print(gpu::driver(), ": ", n, " invocations of the language test, ", wrong, " differ from the CPU");
    gpu::close();
}
