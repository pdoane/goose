// Draws a textured quad over a 64x64 target, through an index buffer, with
// point sampling, push constants and a uniform block, and reads the target
// back: every pixel must be the texel of the 4x4 texture under it, which
// also pins down orientation, y up in clip space and texture coordinates
// from the top left, on every backend.
import gpu from "..";
import std;

struct Vertex { pos: float2, uv: float2 }
struct Draw { vertices: gpu::StructuredBuffer<Vertex>, tex: gpu::Texture2D<float4>, tint: float4 }
struct Scene { transform: gpu::mat4, scale: f32, pad 12 }
struct Varyings { uv: float2 }

vertex fn quad_vs(p: Draw, scene: const Scene&) -> float4, Varyings {
    let v = gpu::load(p.vertices, gpu::vertex_index());
    let clip = gpu::mul(scene.transform, float4 { v.pos.x, v.pos.y, 0.5, 1.0 });
    return clip, Varyings { uv: v.uv };
}

fragment fn quad_fs(f: Varyings, p: Draw, scene: const Scene&) -> float4 {
    gpu::sample(p.tex, gpu::POINT_CLAMP, f.uv) * p.tint * scene.scale
}

let SIZE = 64;

fn texel(x: i64, y: i64) -> u8[4] {
    [(x * 60 + 10) as u8, (y * 60 + 20) as u8, ((x + y) * 25 + 5) as u8, 255]
}

fn main() {
    guard gpu::open_headless() else { abort(str("gpu: ", gpu::error())); }
    var pixels: u8[>..] = [];
    for y in 4 { for x in 4 { pixels.append(texel(x, y)); } }
    let tex = gpu::texture(gpu::TextureDesc { width: 4, height: 4, .. }, pixels);
    let target = gpu::render_target(SIZE, SIZE, gpu::RGBA8);
    let quad: Vertex[4] = [
        Vertex { pos: float2 { -1.0, 1.0 }, uv: float2 { 0.0, 0.0 } },
        Vertex { pos: float2 { 1.0, 1.0 }, uv: float2 { 1.0, 0.0 } },
        Vertex { pos: float2 { -1.0, -1.0 }, uv: float2 { 0.0, 1.0 } },
        Vertex { pos: float2 { 1.0, -1.0 }, uv: float2 { 1.0, 1.0 } },
    ];
    let vb = gpu::buffer(gpu::BUFFER_READ, bytes_of(quad));
    let indices: u16[6] = [0, 1, 2, 2, 1, 3];
    let ib = gpu::index_buffer(bytes_of(indices));
    let p = gpu::pipeline(quad_vs, quad_fs);

    gpu::barrier(target, gpu::ACCESS_NONE, gpu::ACCESS_RENDER_TARGET);
    guard gpu::begin_pass(target, gpu::Texture { 0 }, float4 { 0.0, 0.0, 0.0, 1.0 }) else {
        abort(str("gpu: ", gpu::error()));
    }
    gpu::bind(p);
    gpu::push_constants(Draw { vertices: gpu::structured<Vertex>(vb),
                               tex: gpu::sampled<float4>(tex), tint: float4_1 });
    gpu::uniforms(Scene { transform: gpu::mat4_identity, scale: 1.0 });
    gpu::bind_index_buffer(ib, 2);
    gpu::draw_indexed(6);
    gpu::end_pass();
    gpu::barrier(target, gpu::ACCESS_RENDER_TARGET, gpu::ACCESS_COPY_SRC);
    let r = gpu::read_texture(target);
    gpu::wait(gpu::submit());

    var got: u8[>..] = [];
    guard gpu::bytes(r, got) else { abort("gpu: the read back is not ready"); }
    var wrong = 0;
    for y in SIZE {
        for x in SIZE {
            let want = texel(x * 4 / SIZE, y * 4 / SIZE);
            for c in 4 {
                if got[(y * SIZE + x) * 4 + c] != want[c] {
                    if wrong < 4 { print("pixel ", x, ",", y, ": ", got[(y * SIZE + x) * 4 + c], " not ", want[c]); }
                    wrong++;
                }
            }
        }
    }
    print(gpu::driver(), ": ", SIZE * SIZE, " pixels, ", wrong, " wrong channels");
    gpu::close();
}
