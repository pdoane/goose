"""The debug adapter, `goose dap`, driven over its own protocol: a client starts
it on a pipe and debugs test/dap/flock.goose. A breakpoint stops on a Goose line
with Goose names and values, stepping in reaches the callee without stopping in
generated code, and a failed --debug check stops as an exception on the line
that failed, with the program's own message."""
import json
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Client:
    def __init__(self, exe):
        self.p = subprocess.Popen([str(exe), "dap"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.seq = 0
        self.events = []
        self.responses = {}

    def send(self, command, args=None):
        self.seq += 1
        body = json.dumps({"seq": self.seq, "type": "request", "command": command,
                           "arguments": args or {}}).encode()
        self.p.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.p.stdin.flush()
        return self.seq

    def read(self):
        length = 0
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise AssertionError("the adapter ended")
            line = line.strip()
            if not line:
                break
            if line.lower().startswith(b"content-length"):
                length = int(line.split(b":")[1])
        return json.loads(self.p.stdout.read(length))

    def wait_response(self, seq):
        while seq not in self.responses:
            self.take(self.read())
        return self.responses.pop(seq)

    def take(self, m):
        if m["type"] == "response":
            self.responses[m["request_seq"]] = m
        elif m["type"] == "event":
            self.events.append(m)

    def request(self, command, args=None):
        return self.wait_response(self.send(command, args))

    def wait_event(self, name):
        while True:
            for i, m in enumerate(self.events):
                if m.get("event") == name:
                    return self.events.pop(i)
            self.take(self.read())

    def close(self):
        self.p.kill()
        self.p.wait()


def locals_of(c, frame):
    scopes = c.request("scopes", {"frameId": frame})["body"]["scopes"]
    assert scopes[0]["name"] == "Locals", scopes
    return {v["name"]: v for v in
            c.request("variables", {"variablesReference": scopes[0]["variablesReference"]})["body"]["variables"]}


def top(c, thread):
    frames = c.request("stackTrace", {"threadId": thread})["body"]["stackFrames"]
    return frames


def check_dap_protocol(exe, workdir):
    src = Path(workdir) / "dap_flock.goose"
    src.write_text((HERE / "dap" / "flock.goose").read_text())
    lines = src.read_text().split("\n")
    brk = next(i for i, l in enumerate(lines) if "let peak = loudest" in l) + 1
    fail = next(i for i, l in enumerate(lines) if "flock[i].mood" in l) + 1

    c = Client(exe)
    # Ending the adapter makes the client's next read fail, wherever it waits.
    watchdog = threading.Timer(90, c.close)
    watchdog.start()
    try:
        c.request("initialize", {"adapterID": "goose", "linesStartAt1": True, "pathFormat": "path"})
        launch = c.send("launch", {"program": str(src), "cwd": str(workdir)})
        c.wait_event("initialized")
        r = c.request("setBreakpoints", {"source": {"path": str(src)}, "breakpoints": [{"line": brk}]})
        assert r["body"]["breakpoints"][0]["verified"], r
        c.request("configurationDone")
        c.wait_response(launch)

        stop = c.wait_event("stopped")["body"]
        assert stop["reason"] == "breakpoint", stop
        thread = stop["threadId"]
        frames = top(c, thread)
        assert [(f["name"], f["line"]) for f in frames] == [("main", brk)], frames
        vs = locals_of(c, frames[0]["id"])
        flock = vs["flock"]
        assert flock["type"] == "Goose[>..]", flock
        assert '{name: "Wren", wingspan: 142, at: {x: 1.5, y: 2}, mood: Mood.Calm}' in flock["value"], flock
        kids = c.request("variables", {"variablesReference": flock["variablesReference"]})["body"]["variables"]
        assert "mood: Mood.Furious {decibels: 118}" in kids[1]["value"], kids
        assert vs["heights"]["type"] == "i32[..4]" and vs["heights"]["value"] == "[142, 178, 155]", vs
        assert vs["grid"]["value"] == "[[1, 2, 3], [4, 5, 6]]", vs
        assert vs["call"]["value"] == '"Gander leads"', vs
        assert vs["found"]["type"] == "Goose?" and '"Gander"' in vs["found"]["value"], vs
        assert vs["lost"]["value"] == "null", vs
        assert not [n for n in vs if n.endswith("_g") or n.startswith("t") and n[1:].isdigit()], vs
        assert "peak" not in vs, "a local shows before its declaration"
        kids = c.request("variables", {"variablesReference": flock["variablesReference"]})["body"]["variables"]
        assert [k["name"] for k in kids] == ["[0]", "[1]", "[2]"], kids

        c.request("stepIn", {"threadId": thread})
        stop = c.wait_event("stopped")["body"]
        assert stop["reason"] == "step", stop
        frames = top(c, thread)
        assert [f["name"] for f in frames] == ["loudest", "main"], frames

        case = next(i for i, l in enumerate(lines) if "m: Mood.Furious" in l) + 1
        c.request("setBreakpoints", {"source": {"path": str(src)}, "breakpoints": [{"line": case}]})
        c.request("continue", {"threadId": thread})
        c.wait_event("stopped")
        frames = top(c, thread)
        assert [f["name"] for f in frames] == ["volume", "loudest", "main"], frames
        m = locals_of(c, frames[0]["id"])["m"]
        assert m["type"] == "Mood.Furious" and m["value"] == "{decibels: 118}", m
        c.request("setBreakpoints", {"source": {"path": str(src)}, "breakpoints": []})

        c.request("continue", {"threadId": thread})
        stop = c.wait_event("stopped")["body"]
        assert stop["reason"] == "exception", stop
        assert stop["text"] == "index 3 out of bounds (length 3)", stop
        frames = top(c, thread)
        assert [(f["name"], f["line"]) for f in frames][0] == ("loudest", fail), frames
        vs = locals_of(c, frames[0]["id"])
        assert vs["i"]["value"] == "3", vs
        name = c.request("threads")["body"]["threads"][0]["name"]
        assert name == f"main: in loudest, line {fail}", name
        info = c.request("exceptionInfo", {"threadId": thread})["body"]
        assert info["description"] == stop["text"], info
        r = c.request("evaluate", {"expression": "i", "frameId": frames[0]["id"], "context": "watch"})
        assert r["body"]["result"] == "3", r
        c.request("disconnect", {"terminateDebuggee": True})
    finally:
        watchdog.cancel()
        c.close()


def check_values(exe, workdir):
    """Values the generated C cannot describe, read by their Goose types, and the
    console's expressions over them."""
    src = Path(workdir) / "dap_values.goose"
    src.write_text((HERE / "dap" / "values.goose").read_text())
    lines = src.read_text().split("\n")
    line = next(i for i, l in enumerate(lines) if l.strip().startswith("print(yes")) + 1
    c = Client(exe)
    watchdog = threading.Timer(90, c.close)
    watchdog.start()
    try:
        c.request("initialize", {"adapterID": "goose"})
        launch = c.send("launch", {"program": str(src)})
        c.wait_event("initialized")
        c.request("setBreakpoints", {"source": {"path": str(src)}, "breakpoints": [{"line": line}]})
        c.request("configurationDone")
        c.wait_response(launch)
        thread = c.wait_event("stopped")["body"]["threadId"]
        frame = top(c, thread)[0]["id"]
        vs = {k: v["value"] for k, v in locals_of(c, frame).items()}
        assert vs["yes"] == "true", vs
        assert vs["f"] == "{alive: false, id: 7}", vs
        assert vs["word"] == '"honk"', vs
        assert vs["bag"] == "{id: 3, items: [10, 20]}", vs
        assert vs["people"] == '[{name: "Wren", age: 4}, {name: "Gander", age: 9}]', vs
        assert vs["notes"] == '[Note.Plain {text: "hi"}, Note.Count {n: 300}, Note.Empty]', vs
        assert vs["i"] == "1", vs
        assert "gone" not in vs, "a local shows past the end of its block"
        for expr, want in [("nums[i]", "6"), ("people[1].age + 1", "10"), ("bag.items[1] * 2", "40"),
                           ("!f.alive && yes", "true"), ("word.len", "4"), ("i + 1 == 2", "true"),
                           ("people[1]", '{name: "Gander", age: 9}')]:
            r = c.request("evaluate", {"expression": expr, "frameId": frame, "context": "repl"})
            assert r["success"] and r["body"]["result"] == want, (expr, r)
        r = c.request("evaluate", {"expression": "nums[9]", "frameId": frame, "context": "repl"})
        assert not r["success"] and "out of bounds" in r["message"], r
    finally:
        watchdog.cancel()
        c.close()


def check_threads(exe, workdir):
    """Workers and typed queues, as a stop in main sees them."""
    src = Path(workdir) / "dap_threads.goose"
    src.write_text((HERE / "dap" / "threads.goose").read_text())
    lines = src.read_text().split("\n")
    line = next(i for i, l in enumerate(lines) if "all reported" in l) + 1
    c = Client(exe)
    watchdog = threading.Timer(90, c.close)
    watchdog.start()
    try:
        c.request("initialize", {"adapterID": "goose"})
        launch = c.send("launch", {"program": str(src)})
        c.wait_event("initialized")
        c.request("setBreakpoints", {"source": {"path": str(src)}, "breakpoints": [{"line": line}]})
        c.request("configurationDone")
        c.wait_response(launch)
        stop = c.wait_event("stopped")["body"]
        threads = c.request("threads")["body"]["threads"]
        names = [t["name"] for t in threads]
        assert len(names) == 3 and names[0].startswith("main: line"), names
        assert all(n.startswith("scout") for n in names[1:]), names
        worker = next(t for t in threads if t["id"] != stop["threadId"])
        frames = top(c, worker["id"])
        assert frames[0]["name"] == "scout", frames
        vs = locals_of(c, frames[0]["id"])
        assert vs["speed"]["value"] in ("40", "25"), vs
        frame = top(c, stop["threadId"])[0]["id"]
        scopes = c.request("scopes", {"frameId": frame})["body"]["scopes"]
        queues = next(s for s in scopes if s["name"] == "Queues")
        rows = {v["name"]: v for v in c.request("variables", {"variablesReference": queues["variablesReference"]})["body"]["variables"]}
        assert rows["Note"]["value"] == "[{n: 9}]", rows
    finally:
        watchdog.cancel()
        c.close()


def check_unwind(exe, workdir):
    """Each frame shows its own locals, stepping keeps to source order, and a step
    over `return ... from` lands past every frame it unwinds."""
    src = Path(workdir) / "dap_unwind.goose"
    src.write_text((HERE / "dap" / "unwind.goose").read_text())
    lines = src.read_text().split("\n")
    at = lambda text: next(i for i, l in enumerate(lines) if text in l) + 1
    c = Client(exe)
    watchdog = threading.Timer(90, c.close)
    watchdog.start()
    try:
        c.request("initialize", {"adapterID": "goose"})
        launch = c.send("launch", {"program": str(src)})
        c.wait_event("initialized")
        c.request("setBreakpoints", {"source": {"path": str(src)}, "breakpoints": [{"line": at("let hit =")}]})
        c.request("configurationDone")
        c.wait_response(launch)
        thread = c.wait_event("stopped")["body"]["threadId"]
        frames = top(c, thread)
        assert [f["name"] for f in frames] == ["scan", "scan", "scan", "lookup", "main"], frames
        assert locals_of(c, frames[0]["id"])["i"]["value"] == "2"
        assert locals_of(c, frames[1]["id"])["i"]["value"] == "1"
        assert locals_of(c, frames[2]["id"])["i"]["value"] == "0"
        assert "want" in locals_of(c, frames[3]["id"])
        assert "nums" in locals_of(c, frames[4]["id"])
        seen = []
        for _ in range(3):
            c.request("next", {"threadId": thread})
            c.wait_event("stopped")
            frames = top(c, thread)
            seen.append((frames[0]["name"], frames[0]["line"]))
        assert seen == [("scan", at("print(hit)")), ("scan", at("return hit from")), ("main", at("let found ="))], seen
    finally:
        watchdog.cancel()
        c.close()


def check_mixed(exe, workdir):
    """A program's own C stands in the stack between its Goose frames, shows lldb's
    values and evaluates as C; a step crosses the boundary each way, through an
    `export fn`'s wrapper; and a signal in C stops on the C line."""
    for name in ("mixed.goose", "mixed_c.c", "mixed.build"):
        (Path(workdir) / name).write_text((HERE / "dap" / name).read_text())
    src = Path(workdir).resolve() / "mixed.goose"
    csrc = Path(workdir).resolve() / "mixed_c.c"
    clines = csrc.read_text().split("\n")
    at = lambda text: next(i for i, l in enumerate(clines) if text in l) + 1
    c = Client(exe)
    watchdog = threading.Timer(90, c.close)
    watchdog.start()
    try:
        c.request("initialize", {"adapterID": "goose"})
        launch = c.send("launch", {"program": str(src)})
        c.wait_event("initialized")
        c.request("setBreakpoints", {"source": {"path": str(csrc)}, "breakpoints": [{"line": at("twice = goose_twice(")}]})
        c.request("configurationDone")
        c.wait_response(launch)
        thread = c.wait_event("stopped")["body"]["threadId"]
        frames = top(c, thread)
        assert [(f["name"], Path(f["source"]["path"]).name) for f in frames] == [("walk", "mixed_c.c"), ("main", "mixed.goose")], frames
        vs = locals_of(c, frames[0]["id"])
        assert vs["n"]["type"] == "int32_t" and vs["n"]["value"] == "21", vs
        assert locals_of(c, frames[1]["id"])["base"]["type"] == "i32", frames
        r = c.request("evaluate", {"expression": "n + 1", "frameId": frames[0]["id"], "context": "repl"})
        assert r["body"]["result"] == "22", r
        r = c.request("evaluate", {"expression": "base", "frameId": frames[1]["id"], "context": "repl"})
        assert r["body"]["result"] == "21", r
        assert c.request("threads")["body"]["threads"][0]["name"].startswith("main: in walk, line"), "thread name"

        c.request("setBreakpoints", {"source": {"path": str(csrc)}, "breakpoints": []})
        c.request("stepIn", {"threadId": thread})
        c.wait_event("stopped")
        assert [f["name"] for f in top(c, thread)] == ["twice", "walk", "main"], "into Goose from C"
        c.request("stepOut", {"threadId": thread})
        c.wait_event("stopped")
        assert [f["name"] for f in top(c, thread)] == ["walk", "main"], "back into C"

        c.request("continue", {"threadId": thread})
        stop = c.wait_event("stopped")["body"]
        assert stop["reason"] == "exception", stop
        frames = top(c, thread)
        assert (frames[0]["name"], frames[0]["line"]) == ("crash", at("return *nowhere")), frames
        assert frames[-1]["name"] == "main", frames
        assert c.request("exceptionInfo", {"threadId": thread})["body"]["description"], "signal text"
    finally:
        watchdog.cancel()
        c.close()


if __name__ == "__main__":
    import sys
    check_dap_protocol(sys.argv[1], sys.argv[2])
    check_values(sys.argv[1], sys.argv[2])
    check_threads(sys.argv[1], sys.argv[2])
    check_unwind(sys.argv[1], sys.argv[2])
    check_mixed(sys.argv[1], sys.argv[2])
    print("ok")
