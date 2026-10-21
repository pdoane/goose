// error: extern fn clock_ms runs on the CPU only
extern "gs_os_clock_ms" fn clock_ms() -> i64;
compute fn k() { let t = clock_ms() as u32; }
fn main() { print(gpu_entry(k).len); }
