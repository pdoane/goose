import std;

extern fn walk(n: i32) -> i32;
extern fn crash() -> i32;

export "goose_twice" fn twice(x: i32) -> i32 {
    let doubled = x * 2;
    return doubled;
}

fn main() {
    let base: i32 = 21;
    print(walk(base));
    print(crash());
}
