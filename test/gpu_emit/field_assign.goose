// Assignment to a struct's vector field, whole and by component, which MSL
// stores packed: the target is the stored field, while a read of one
// unpacks it.
import gpu_shader;

struct Pose { at: float3, scale: f32, color: float4 }
struct Push { out: gpu::RWStructuredBuffer<Pose> }

compute fn edit(p: Push) {
    var pose = Pose { at: float3_0, scale: 1.0, color: float4_1 };
    pose.at = float3 { 1.0, 2.0, 3.0 };
    pose.at.y += 0.5;
    pose.color.w = pose.at.x;
    pose.scale *= 2.0;
    gpu::store(p.out, 0, pose);
}

fn main() { print(gpu_entry(edit).len > 0); }
