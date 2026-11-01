import std;

struct Leg { miles: i32 }
struct Report { minutes: i32 }
struct Note { n: i32 }

thread_fn scout(speed: i32) {
    loop {
        let leg = qget<Leg>();
        guard leg.miles >= 0 else { break; }
        qput(Report { leg.miles * 60 / speed });
    }
}

fn main() {
    let fast = thread_spawn(scout, 40);
    let slow = thread_spawn(scout, 25);
    qput(Leg { 120 });
    qput(Leg { 75 });
    var total: i32 = 0;
    for i in 2 { total += qget<Report>().minutes; }
    qput(Note { 9 });
    print("all reported ", total);
    qput(Leg { -1 });
    qput(Leg { -1 });
    thread_wait(fast);
    thread_wait(slow);
    print(qget<Note>().n);
}
