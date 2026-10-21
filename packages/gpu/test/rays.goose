// Ray tracing (docs/design/gpu.md §7): one primitive acceleration structure,
// a unit square of two triangles, placed three times by an instance
// structure, and a compute shader casting a grid of rays down -z with
// gpu::trace and gpu::trace_any. Every hit is checked against where the
// squares are: the instance's id, the distance, the triangle, the side.
// The instances are GPU data: a compute pass writes them, and a barrier
// into ACCESS_ACCEL_INPUT orders the build after it.
import gpu from "..";
import std;

struct Cast { scene: gpu::Accel, out: gpu::RWStructuredBuffer<Result>, n: u32 }
struct Result { kind: u32, t: f32, id: u32, primitive: u32, front: u32, any: u32 }

let GRID = 8;

compute(8, 8) fn cast(p: Cast) {
    let xy = gpu::dispatch_thread_id();
    if xy.x >= p.n || xy.y >= p.n { return; }
    // Cell centers over -1.5..1.5, never on a square's edge or diagonal.
    let step: f32 = 3.0 / p.n;
    let x = -1.5 + (xy.x + 0.37) * step;
    let y = -1.5 + (xy.y + 0.61) * step;
    let r = gpu::Ray { origin: float3 { x, y, 5.0 }, t_min: 0.0, dir: float3 { 0.0, 0.0, -1.0 },
                       t_max: 100.0 };
    let h = gpu::trace(p.scene, r, 255);
    let any: u32 = if gpu::trace_any(p.scene, r, 255) { 1 } else { 0 };
    gpu::store(p.out, xy.y * p.n + xy.x, Result { kind: h.kind, t: h.t, id: h.id,
                                                  primitive: h.primitive, front: h.front_face,
                                                  any: any });
}

struct Square { x: f32, y: f32, z: f32, id: u32 }

struct Place {
    squares: gpu::StructuredBuffer<Square>,
    out: gpu::RWStructuredBuffer<gpu::Instance>,
    square: uint2,
    count: u32,
}

// One instance record per square, written on the GPU.
compute(64) fn place(p: Place) {
    let i = gpu::dispatch_thread_id().x;
    if i >= p.count { return; }
    let s = gpu::load(p.squares, i);
    gpu::store(p.out, i, gpu::instance_of(gpu::translate(float3 { s.x, s.y, s.z }), p.square, s.id));
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    guard gpu::ray_query() else {
        print(gpu::driver(), ": no ray queries on this device");
        gpu::close();
        return;
    }
    // The square: (0,0) to (1,1) at z 0, counterclockwise seen from +z.
    let verts: float3[4] = [float3 { 0.0, 0.0, 0.0 }, float3 { 1.0, 0.0, 0.0 },
                            float3 { 1.0, 1.0, 0.0 }, float3 { 0.0, 1.0, 0.0 }];
    let tris: u32[6] = [0, 1, 2, 0, 2, 3];
    let vb = gpu::buffer(gpu::BUFFER_READ, bytes_of(verts));
    let ib = gpu::buffer(gpu::BUFFER_READ, bytes_of(tris));
    let square = gpu::primitive_accel(gpu::PrimitiveAccelDesc { vertices: vb, vertex_count: 4,
                                                                indices: ib, triangle_count: 2, .. });
    // Three squares, the second in front of where it overlaps the third.
    let squares: Square[3] = [Square { x: 0.0, y: 0.0, z: 0.0, id: 10 },
                              Square { x: -1.0, y: -1.0, z: 1.0, id: 20 },
                              Square { x: -1.5, y: -0.5, z: -1.0, id: 30 }];
    let sbuf = gpu::buffer(gpu::BUFFER_READ, bytes_of(squares));
    let ibuf = gpu::buffer(gpu::BUFFER_WRITE, 3 * 72);
    let placer = gpu::compute_pipeline(place);
    gpu::begin_compute();
    gpu::bind(placer);
    gpu::push_constants(Place { squares: gpu::structured<Square>(sbuf),
                                out: gpu::rw_structured<gpu::Instance>(ibuf),
                                square: gpu::reference(square), count: 3 });
    gpu::dispatch(1, 1, 1);
    gpu::end_compute();
    // The build reads what the pass wrote: the program's barrier says so.
    gpu::barrier(ibuf, gpu::ACCESS_SHADER_WRITE, gpu::ACCESS_ACCEL_INPUT);
    let scene = gpu::instance_accel(gpu::InstanceAccelDesc { instances: ibuf, max: 3 });
    gpu::build(square);
    gpu::build(scene, 3);

    let out = gpu::buffer(gpu::BUFFER_WRITE, GRID * GRID * 24);
    let p = gpu::compute_pipeline(cast);
    guard p.id != 0 else { abort(str("gpu: ", gpu::error())); }
    gpu::begin_compute();
    gpu::bind(p);
    gpu::push_constants(Cast { scene: scene, out: gpu::rw_structured<Result>(out), n: GRID });
    gpu::dispatch(1, 1, 1);
    gpu::end_compute();
    let r = gpu::read_buffer(out, 0, GRID * GRID * 24);
    gpu::wait(gpu::submit());
    var got: Result[>..] = [];
    guard gpu::values(r, got) else { abort("gpu: the read back is not ready"); }

    var wrong = 0;
    var hits = 0;
    let step: f32 = 3.0 / GRID;
    for j in GRID {
        for i in GRID {
            let x = -1.5 + (i + 0.37) * step;
            let y = -1.5 + (j + 0.61) * step;
            // The nearest square under the ray: the highest z.
            var want = Result { kind: 0, t: 0.0, id: 0, primitive: 0, front: 0, any: 0 };
            var best: f32 = -100.0;
            for s in squares {
                let u = x - s.x;
                let v = y - s.y;
                if u > 0.0 && u < 1.0 && v > 0.0 && v < 1.0 && s.z > best {
                    best = s.z;
                    let tri: u32 = if v < u { 0 } else { 1 };
                    want = Result { kind: gpu::HIT_TRIANGLE, t: 5.0 - s.z, id: s.id,
                                    primitive: tri, front: 1, any: 1 };
                }
            }
            let g = got[j * GRID + i];
            let ok = g.kind == want.kind && g.any == want.any &&
                     (want.kind == 0 || (abs(g.t - want.t) < 0.0001 && g.id == want.id &&
                                         g.primitive == want.primitive && g.front == want.front));
            if want.kind != 0 { hits++; }
            if !ok {
                if wrong < 4 {
                    print("ray ", i, ",", j, ": kind ", g.kind, " t ", g.t, " id ", g.id, " tri ",
                          g.primitive, " front ", g.front, " any ", g.any, ", not id ", want.id,
                          " tri ", want.primitive);
                }
                wrong++;
            }
        }
    }
    print(gpu::driver(), ": ", GRID * GRID, " rays, ", hits, " hits, ", wrong, " wrong");
    gpu::close();
}
