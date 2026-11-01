import std;

enum Mood {
    Calm,
    Hungry { snacks: i32 },
    Furious { decibels: i32 },
}

struct Point { x: f64, y: f64 }

struct Goose {
    name: u8[..8],
    wingspan: i32,
    at: Point,
    mood: Mood,
}

// Case functions: the call dispatches on the variant.
fn volume(m: Mood.Calm) -> i32 { 40 }
fn volume(m: Mood.Hungry) -> i32 { 70 + m.snacks }
fn volume(m: Mood.Furious) -> i32 { m.decibels }

// An optional reference into the slice.
fn widest_over(flock: Goose[:], min: i32) -> Goose? {
    for &g in flock {
        if g.wingspan > min { return g; }
    }
    return null;
}

fn loudest(flock: Goose[:]) -> i32 {
    var peak: i32 = 0;
    var i = 0;
    while i <= flock.len {
        let honk = volume(flock[i].mood);
        if honk > peak { peak = honk; }
        i++;
    }
    return peak;
}

fn main() {
    var flock: Goose[>..] = [];
    flock.push(Goose { "Wren", 142, Point { 1.5, 2.0 }, Mood.Calm });
    flock.push(Goose { "Gander", 178, Point { 4.0, 0.5 }, Mood.Furious { 118 } });
    flock.push(Goose { "Maple", 155, Point { 2.5, 3.0 }, Mood.Hungry { 3 } });

    let heights: i32[..4] = [142, 178, 155];
    let grid: i32[3][2] = [[1, 2, 3], [4, 5, 6]];

    var call: u8[>..] = [];
    format(call, flock[1].name, " leads");

    let found = widest_over(flock, 170);
    let lost = widest_over(flock, 500);

    let peak = loudest(flock);
    print(call, " at ", peak, " dB ", heights.len, grid[1][2], found != null, lost == null);
}
