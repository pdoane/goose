import std;

recursive fn scan(xs: i32[:], i: i64, want: i32) -> i32 {
    if xs[i] == want {
        let hit = i as i32;
        print(hit);
        return hit from lookup;
    }
    let deeper = i + 1;
    return scan(xs, deeper, want);
}

fn lookup(xs: i32[:], want: i32) -> i32 {
    let at = scan(xs, 0, want);
    return at;
}

fn main() {
    var nums: i32[>..] = [];
    nums.push(5);
    nums.push(6);
    nums.push(7);
    let found = lookup(nums, 7);
    print(found);
}
