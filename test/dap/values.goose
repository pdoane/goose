import std;

struct Flags { alive: bool, id: i32 }
struct Bag { id: i32, items: i32[>..] }
struct Person { name: u8[], age: i32 }
enum Note {
    Plain { text: u8[] },
    Count { n: varint },
    Empty,
}

fn main() {
    let yes = true;
    let f = Flags { false, 7 };
    let word: u8[] = "honk";
    var bag = Bag { 3, [] };
    bag.items.push(10);
    bag.items.push(20);
    var people: Person[>..] = [];
    people.push(Person { "Wren", 4 });
    people.push(Person { "Gander", 9 });
    var notes: Note..[>..] = [];
    notes.push(Note.Plain { "hi" });
    notes.push(Note.Count { 300 });
    notes.push(Note.Empty);
    var nums: i32[>..] = [];
    nums.push(5);
    nums.push(6);
    for k in 2 {
        let gone = k;
        print(gone);
    }
    var i = 1;
    print(yes, f.id, word.len, bag.items.len, people.len, notes.len, nums[i]);
}
