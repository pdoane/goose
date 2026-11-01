// Goose compiler — `goose dap`, the Debug Adapter Protocol for Goose programs.
// A debugger client talks to this process on stdin and stdout and sees Goose:
// source lines, source names, values as the language writes them. Underneath is
// lldb-dap, which debugs the C that `goose build --debug` compiled; requests the
// adapter has no view on pass through it unchanged.
//
// lldb-dap runs the program, places breakpoints, unwinds and reads memory. What
// the program's values mean comes from the compiler: `goose build --debug` writes a
// map of each function's locals and each type's layout (codegen_debugmap.h), and
// the adapter reads values out of memory by their Goose types.
//
// C the program brings along (`c_sources`) is debugged as itself: its frames stand
// in the stack between the Goose frames, with lldb's view of their values, and a
// step crosses the boundary in either direction.
#pragma once

namespace goose {

inline void JsonWrite(JsonOut &out, const Json &v) {
    switch (v.kind) {
        case Json::ARRAY:
            out.BeginArray();
            for (auto &e : v.arr) JsonWrite(out, e);
            out.EndArray();
            return;
        case Json::OBJECT:
            out.BeginObject();
            for (auto &[k, e] : v.obj) { out.Key(k); JsonWrite(out, e); }
            out.EndObject();
            return;
        default:
            out.Raw(JsonText(v));
    }
}

inline void JsonSet(Json &o, const string &key, Json v) {
    for (auto &[k, e] : o.obj) if (k == key) { e = std::move(v); return; }
    o.obj.emplace_back(key, std::move(v));
}
inline Json JsonOfStr(string s) { Json j; j.kind = Json::STRING; j.str = std::move(s); return j; }
inline Json JsonOfInt(int64_t n) { Json j; j.kind = Json::NUMBER; j.num = (double)n; return j; }
inline Json JsonOfBool(bool b) { Json j; j.kind = Json::BOOL; j.b = b; return j; }
inline int64_t JsonInt(const Json &o, const string &key) {
    auto v = o.Get(key);
    return v && v->kind == Json::NUMBER ? (int64_t)v->num : 0;
}

// --- the generated C's spellings ---------------------------------------------

inline bool EndsWith(string_view s, string_view e) {
    return s.size() >= e.size() && s.substr(s.size() - e.size()) == e;
}
inline bool StartsWith(string_view s, string_view p) { return s.substr(0, p.size()) == p; }

// The integer an adapter's rendering of a scalar holds; a uint8_t reads '\x03'.
inline uint64_t DapScalar(string_view v) {
    if (v.size() >= 3 && v[0] == '\'') {
        auto in = v.substr(1, v.size() - 2);
        if (in[0] != '\\') return (uint8_t)in[0];
        if (in.size() < 2) return 0;
        switch (in[1]) {
            case 'x': return strtoull(string(in.substr(2)).c_str(), nullptr, 16);
            case 'a': return 7;
            case 'b': return 8;
            case 't': return 9;
            case 'n': return 10;
            case 'v': return 11;
            case 'f': return 12;
            case 'r': return 13;
            case '0': return 0;
            default: return (uint8_t)in[1];
        }
    }
    return strtoull(string(v).c_str(), nullptr, 10);
}

// The text between the outer quotes of a rendering of bytes, cut to `len` bytes.
inline string DapBytes(string_view v, size_t len) {
    auto first = v.find('"');
    auto last = v.rfind('"');
    if (first == string_view::npos || last <= first) return "";
    return string(v.substr(first + 1, min(len, last - first - 1)));
}

// The address in an adapter's rendering of a pointer.
inline uint64_t DapAddress(string_view v) {
    auto at = v.rfind("0x");
    return at == string_view::npos ? 0 : strtoull(string(v.substr(at)).c_str(), nullptr, 16);
}

inline void DapQuoteBytes(string &out, string_view bytes) {
    out += '"';
    for (auto c : bytes) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c == '\n') out += "\\n";
        else if ((uint8_t)c < 32 || (uint8_t)c == 127) {
            char buf[8];
            snprintf(buf, sizeof buf, "\\x%02x", (unsigned)(uint8_t)c);
            out += buf;
        } else out += c;
    }
    out += '"';
}

// --- the adapter -------------------------------------------------------------

struct DapAdapter {
    string exe;
    PipedProcess child;

    struct Message { bool fromclient; string body; };
    mutex mtx;
    condition_variable cv;
    deque<Message> arrived;
    deque<Message> deferred;      // What came while the adapter waited for a reply of its own.

    int64_t childseq = 1000000, clientseq = 1;
    map<int64_t, Json> forwarded; // The client's requests in flight, by the seq they went to lldb under.

    string programdir;            // The Goose source's directory: C under it is the program's own.

    // This stop.
    string cwd;
    int64_t failedthread = 0;
    string failuretext, failurefile;
    int failureline = 0;
    struct Frame { string source; int line = 0; string function; bool mainthread = false; };
    map<int64_t, Frame> frames;
    set<int64_t> cframes;                 // The frames in the program's own C: lldb describes them.
    map<int64_t, int64_t> scopeframes;    // A locals scope's reference: its frame.
    map<int64_t, vector<Json>> rendered;  // What a reference expands to.
    struct Lazy { uint64_t addr; int type; string path; };
    map<int64_t, Lazy> lazy;              // A reference past the depth read ahead.
    int64_t nextref = 1000000000;
    int64_t currentframe = 0;
    bool unwinding = false;
    int64_t globalsframe = 0;
    static constexpr int64_t globalsref = 999999999, queuesref = 999999998;
    map<uint64_t, string> pages;          // Memory read at this stop.

    Json debugmap;
    const Json *Functions() const { return debugmap.Get("functions"); }
    const Json *Function(const string &cname) const { auto f = Functions(); return f ? f->Get(cname) : nullptr; }
    const Json &TypeOf(int id) const { return debugmap.Arr("types")[id]; }

    static constexpr int readdepth = 4;
    static constexpr size_t readelements = 64, summaryelements = 12;
    static constexpr uint64_t pagesize = 4096;

    void Reader(FILE *in, bool fromclient) {
        string body;
        while (Transport::ReadMessage(in, body)) {
            lock_guard<mutex> lock(mtx);
            arrived.push_back({ fromclient, std::move(body) });
            cv.notify_one();
            body.clear();
        }
        lock_guard<mutex> lock(mtx);
        arrived.push_back({ fromclient, string() });
        cv.notify_one();
    }

    Message Next() {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [&] { return !arrived.empty(); });
        auto m = std::move(arrived.front());
        arrived.pop_front();
        return m;
    }

    static void Write(FILE *out, const string &body) {
        fprintf(out, "Content-Length: %zu\r\n\r\n", body.size());
        fwrite(body.data(), 1, body.size(), out);
        fflush(out);
    }

    void ToClient(Json msg) {
        JsonSet(msg, "seq", JsonOfInt(clientseq++));
        JsonOut out;
        JsonWrite(out, msg);
        Write(stdout, out.s);
    }

    void Respond(const Json &req, bool success, Json body, const string &message = "") {
        Json r;
        r.kind = Json::OBJECT;
        JsonSet(r, "type", JsonOfStr("response"));
        JsonSet(r, "request_seq", JsonOfInt(JsonInt(req, "seq")));
        JsonSet(r, "command", JsonOfStr(req.Str("command")));
        JsonSet(r, "success", JsonOfBool(success));
        if (!message.empty()) JsonSet(r, "message", JsonOfStr(message));
        JsonSet(r, "body", std::move(body));
        ToClient(std::move(r));
    }

    int64_t SendToChild(const string &command, const Json &args) {
        auto seq = childseq++;
        JsonOut out;
        out.BeginObject();
        out.Member("seq", seq);
        out.Member("type", "request");
        out.Member("command", command);
        out.Key("arguments");
        JsonWrite(out, args);
        out.EndObject();
        Write(child.in, out.s);
        return seq;
    }

    // Asks lldb and waits for its answer; what else arrives keeps its order for
    // the main loop.
    Json Call(const string &command, const Json &args) {
        auto seq = SendToChild(command, args);
        for (;;) {
            auto m = Next();
            // lldb gone, it will never answer.
            if (!m.fromclient && m.body.empty()) return Object();
            if (m.fromclient || m.body.empty()) { deferred.push_back(std::move(m)); continue; }
            auto j = ParseJson(m.body);
            if (j.Str("type") == "response" && JsonInt(j, "request_seq") == seq) return j;
            deferred.push_back(std::move(m));
        }
    }

    static Json Object() { Json j; j.kind = Json::OBJECT; return j; }
    static Json Array() { Json j; j.kind = Json::ARRAY; return j; }

    static bool Succeeded(const Json &r) { auto s = r.Get("success"); return s && s->b; }

    bool Evaluate(const string &expr, int64_t frame, string &result, int64_t &ref) {
        auto args = Object();
        JsonSet(args, "expression", JsonOfStr(expr));
        JsonSet(args, "context", JsonOfStr("watch"));
        JsonSet(args, "frameId", JsonOfInt(frame));
        auto r = Call("evaluate", args);
        auto body = r.Get("body");
        if (!Succeeded(r) || !body) return false;
        result = body->Str("result");
        ref = JsonInt(*body, "variablesReference");
        return true;
    }

    // --- memory ---------------------------------------------------------

    static string Base64(const string &in) {
        string out;
        int bits = 0, have = 0;
        for (auto c : in) {
            int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26
                    : c >= '0' && c <= '9' ? c - '0' + 52 : c == '+' ? 62 : c == '/' ? 63 : -1;
            if (v < 0) continue;
            bits = (bits << 6) | v;
            have += 6;
            if (have >= 8) { have -= 8; out += (char)((bits >> have) & 255); }
        }
        return out;
    }

    // `n` bytes at `addr`; false when any of them cannot be read.
    bool Read(uint64_t addr, void *dst, size_t n) {
        auto out = (uint8_t *)dst;
        while (n) {
            auto base = addr & ~(pagesize - 1);
            auto it = pages.find(base);
            if (it == pages.end()) {
                auto args = Object();
                JsonSet(args, "memoryReference", JsonOfStr(cat("0x", Hex(base))));
                JsonSet(args, "count", JsonOfInt((int64_t)pagesize));
                auto r = Call("readMemory", args);
                auto body = r.Get("body");
                it = pages.emplace(base, body ? Base64(body->Str("data")) : string()).first;
            }
            auto off = addr - base;
            auto take = min<uint64_t>(n, pagesize - off);
            if (off + take > it->second.size()) return false;
            memcpy(out, it->second.data() + off, take);
            out += take;
            addr += take;
            n -= take;
        }
        return true;
    }
    static string Hex(uint64_t v) { char b[32]; snprintf(b, sizeof b, "%llx", (unsigned long long)v); return b; }
    uint64_t ReadInt(uint64_t addr, int size) {
        uint64_t v = 0;
        return Read(addr, &v, (size_t)size) ? v : 0;
    }
    uint64_t ReadUleb(uint64_t addr, int64_t &size) {
        uint64_t v = 0;
        size = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            auto b = ReadInt(addr + size++, 1);
            v |= (b & 127) << shift;
            if (!(b & 128)) break;
        }
        return v;
    }

    // --- values -----------------------------------------------------------

    // A value's rendering: its one line, and what it expands to.
    struct Value { string text; vector<Json> children; int64_t ref = 0; int64_t size = 0; };

    static Json Var(const string &name, const string &type, const string &path, Value &v) {
        auto j = Object();
        JsonSet(j, "name", JsonOfStr(name));
        JsonSet(j, "value", JsonOfStr(v.text));
        JsonSet(j, "type", JsonOfStr(type));
        JsonSet(j, "evaluateName", JsonOfStr(path));
        JsonSet(j, "variablesReference", JsonOfInt(v.ref));
        return j;
    }

    int64_t Expansion(vector<Json> children) {
        if (children.empty()) return 0;
        auto ref = nextref++;
        rendered[ref] = std::move(children);
        return ref;
    }

    Value ReadArray(uint64_t data, int64_t count, const Json &sub, int subid, const string &path, int depth) {
        Value v;
        auto issub8 = sub.Str("kind") == "int" && sub.Str("name") == "u8";
        if (issub8) {
            string bytes((size_t)min<int64_t>(count, 4096), 0);
            if (!Read(data, bytes.data(), bytes.size())) { v.text = "<unreadable>"; return v; }
            DapQuoteBytes(v.text, bytes);
            v.size = count;
            return v;
        }
        v.text = "[";
        auto at = data;
        for (int64_t i = 0; i < count; i++) {
            auto elempath = cat(path, "[", i, "]");
            if (i < (int64_t)readelements || !sub.Get("size")) {
                auto e = ReadValue(at, subid, elempath, depth + 1);
                if (i < (int64_t)summaryelements) v.text += cat(i ? ", " : "", e.text);
                else if (i == (int64_t)summaryelements) v.text += ", ...";
                if (i < (int64_t)readelements) v.children.push_back(Var(cat("[", i, "]"), sub.Str("name"), elempath, e));
                at += e.size;
            } else {
                at = data + (uint64_t)count * (uint64_t)JsonInt(sub, "size");
                break;
            }
        }
        v.text += "]";
        v.size = (int64_t)(at - data);
        v.ref = Expansion(v.children);
        return v;
    }

    void ReadFields(Value &v, uint64_t addr, const vector<Json> &fields, const string &path, int depth, int64_t tailoffset = -1) {
        v.text += "{";
        auto first = true;
        auto at = addr;
        for (size_t i = 0; i < fields.size(); i++) {
            auto &f = fields[i];
            if (f.Get("pad")) { auto n = JsonInt(f, "pad"); if (n > 0) at += n; continue; }
            auto name = f.Str("name");
            auto typeid_ = (int)JsonInt(f, "type");
            if (tailoffset >= 0 && i + 1 == fields.size()) at = addr + (uint64_t)tailoffset;
            auto fp = cat(path, ".", name);
            auto e = ReadValue(at, typeid_, fp, depth + 1);
            v.text += cat(first ? "" : ", ", name, ": ", e.text);
            first = false;
            v.children.push_back(Var(name, TypeOf(typeid_).Str("name"), fp, e));
            at += e.size;
        }
        v.text += "}";
        v.size = (int64_t)(at - addr);
        v.ref = Expansion(v.children);
    }

    Value ReadValue(uint64_t addr, int id, const string &path, int depth) {
        auto &t = TypeOf(id);
        auto kind = t.Str("kind");
        Value v;
        v.size = JsonInt(t, "size");
        if (depth > readdepth && (kind == "struct" || kind == "enum" || kind == "array" || kind == "ref" || kind == "slice")) {
            // Past what a one-line form shows, and a reference may go round in a
            // cycle: read when the client opens it.
            v.text = "{...}";
            v.ref = nextref++;
            lazy[v.ref] = { addr, id, path };
            return v;
        }
        if (kind == "int") {
            if (t.Get("varint")->b) {
                // A value is zigzag over the ULEB128 a length uses plainly.
                auto u = ReadUleb(addr, v.size);
                v.text = to_string((int64_t)(u >> 1) ^ -(int64_t)(u & 1));
                return v;
            }
            auto raw = ReadInt(addr, (int)v.size);
            if (t.Get("signed")->b) {
                auto shift = 64 - 8 * (int)v.size;
                v.text = to_string((int64_t)(raw << shift) >> shift);
            } else {
                v.text = to_string(raw);
            }
        } else if (kind == "float") {
            char buf[64];
            if (v.size == 4) { float f = 0; Read(addr, &f, 4); snprintf(buf, sizeof buf, "%.7g", f); }
            else { double d = 0; Read(addr, &d, 8); snprintf(buf, sizeof buf, "%.15g", d); }
            v.text = buf;
        } else if (kind == "bool") {
            v.text = ReadInt(addr, 1) ? "true" : "false";
        } else if (kind == "struct") {
            auto fixed = t.Get("size") != nullptr;
            int64_t tail = -1;
            if (t.Get("frameobject")) {
                // Its tail's header is the last thing in a C struct the C compiler laid out.
                string size;
                int64_t unused = 0;
                if (Evaluate(cat("sizeof(", t.Str("frameobject"), ")"), currentframe, size, unused))
                    tail = (int64_t)DapScalar(size) - 16;
            }
            auto size = v.size;
            ReadFields(v, addr, t.Arr("fields"), path, depth, tail);
            if (fixed) v.size = size;
        } else if (kind == "enum") {
            auto ts = (int)JsonInt(t, "tagsize");
            auto tag = ReadInt(addr, ts);
            auto &variants = t.Arr("variants");
            if (tag >= variants.size()) {
                v.text = cat("<bad tag ", (int64_t)tag, ">");
                return v;
            }
            auto &var = variants[tag];
            auto label = cat(t.Str("enum"), ".", var.Str("name"));
            auto &payload = TypeOf((int)JsonInt(var, "type"));
            auto size = v.size;
            if (payload.Arr("fields").empty()) {
                v.text = label;
                if (!t.Get("size")) v.size = ts;
            } else {
                v.text = label + " ";
                ReadFields(v, addr + ts, payload.Arr("fields"), path, depth);
                if (t.Get("size")) v.size = size;
                else v.size += ts;
            }
        } else if (kind == "slice") {
            auto subid = (int)JsonInt(t, "sub");
            auto r = ReadArray(ReadInt(addr, 8), (int64_t)ReadInt(addr + 8, 8), TypeOf(subid), subid, path, depth);
            r.size = 16;
            return r;
        } else if (kind == "array") {
            auto subid = (int)JsonInt(t, "sub");
            auto &sub = TypeOf(subid);
            auto shape = t.Str("shape");
            Value r;
            if (shape == "fixed") {
                r = ReadArray(addr, JsonInt(t, "count"), sub, subid, path, depth);
            } else if (shape == "limited") {
                // A length, then the capacity's slots, in the size the compiler gave.
                auto cap = JsonInt(t, "count");
                auto lensize = (int)(v.size - cap * JsonInt(sub, "size"));
                r = ReadArray(addr + lensize, (int64_t)ReadInt(addr, lensize), sub, subid, path, depth);
            } else if (shape == "limited_runtime") {
                auto cap = (int64_t)ReadInt(addr, 4);
                r = ReadArray(addr + 8, (int64_t)ReadInt(addr + 4, 4), sub, subid, path, depth);
                r.size = 8 + cap * JsonInt(sub, "size");
                return r;
            } else if (shape == "var") {
                auto ls = (int)JsonInt(t, "lensize");
                int64_t prefix = ls;
                auto n = ls ? ReadInt(addr, ls) : ReadUleb(addr, prefix);
                r = ReadArray(addr + prefix, (int64_t)n, sub, subid, path, depth);
                r.size += prefix;
                return r;
            } else {
                // A resizable array is its header: the elements' base, and a length.
                r = ReadArray(ReadInt(addr, 8), (int64_t)ReadInt(addr + 8, 8), sub, subid, path, depth);
                r.size = 16;
                return r;
            }
            r.size = v.size;
            return r;
        } else if (kind == "ref") {
            auto subid = (int)JsonInt(t, "sub");
            auto size = v.size ? v.size : 8;
            uint64_t target = 0;
            if (auto rel = t.Get("relative"); rel && rel->b) {
                int64_t width = size;
                auto off = t.Get("size") ? ReadInt(addr, (int)size) : ReadUleb(addr, width);
                if (!t.Get("size")) size = width;
                if (t.Get("pool")->b) {
                    v.text = cat("@", (int64_t)off);
                    v.size = size;
                    return v;
                }
                target = off ? addr + off : 0;
            } else {
                target = ReadInt(addr, 8);
            }
            if (!target) {
                v.text = "null";
                v.size = size;
                return v;
            }
            auto r = ReadValue(target, subid, path, depth + 1);
            r.size = size;
            return r;
        } else {
            v.text = "<opaque>";
        }
        return v;
    }

    // The local visible at a frame's line by the name source gives it: the latest
    // declared, as an inner one shadows an outer.
    const Json *FindLocal(const Frame &frame, const string &name) const {
        auto f = Function(frame.function);
        if (!f) return nullptr;
        const Json *best = nullptr;
        for (auto &l : f->Arr("locals")) {
            if (l.Str("name") != name || !Visible(l, frame.line)) continue;
            if (!best || JsonInt(l, "line") > JsonInt(*best, "line")) best = &l;
        }
        if (best) return best;
        for (auto &g : debugmap.Arr("globals")) if (g.Str("name") == name) return &g;
        return nullptr;
    }
    static bool Visible(const Json &local, int line) {
        auto p = local.Get("param");
        if (p && p->b) return true;
        auto end = JsonInt(local, "end");
        return JsonInt(local, "line") < line && (end == 0 || line <= end);
    }

    // A local or global as a value: from memory, from the map when it is a named
    // constant, or a field at a time for a root struct.
    bool LocalValue(const Json &local, int64_t frame, const string &name, Value &v) {
        if (local.Get("value")) { v.text = local.Str("value"); return true; }
        if (local.Get("components")) {
            v.text = "{";
            auto first = true;
            for (auto &c : local.Arr("components")) {
                Value part;
                if (!LocalValue(c, frame, cat(name, ".", c.Str("name")), part)) continue;
                v.text += cat(first ? "" : ", ", c.Str("name"), ": ", part.text);
                first = false;
                v.children.push_back(Var(c.Str("name"), TypeOf((int)JsonInt(c, "type")).Str("name"), cat(name, ".", c.Str("name")), part));
            }
            v.text += "}";
            v.ref = Expansion(v.children);
            return true;
        }
        uint64_t addr = 0;
        if (!LocalAddress(local, frame, addr)) return false;
        v = ReadValue(addr, (int)JsonInt(local, "type"), name, 0);
        return true;
    }

    bool LocalAddress(const Json &local, int64_t frame, uint64_t &addr) {
        string where;
        int64_t unused = 0;
        if (!Evaluate(cat("&", local.Str("c")), frame, where, unused)) return false;
        addr = DapAddress(where);
        if (!addr) return false;
        // A captured variable arrives in a nested function as its address.
        if (auto p = local.Get("pointer"); p && p->b) addr = ReadInt(addr, 8);
        // A variable-size value lives on a data stack; the C local points at it.
        auto &t = TypeOf((int)JsonInt(local, "type"));
        if (!t.Get("size") && t.Str("shape") != "resizable" && !t.Get("frameobject")) addr = ReadInt(addr, 8);
        return addr != 0;
    }

    void ClearStop() {
        failedthread = 0;
        failuretext.clear();
        failurefile.clear();
        failureline = 0;
        scopeframes.clear();
        frames.clear();
        cframes.clear();
        rendered.clear();
        lazy.clear();
        pages.clear();
    }

    // --- requests ---------------------------------------------------------

    void Forward(const Json &req) {
        auto args = req.Get("arguments");
        auto seq = SendToChild(req.Str("command"), args ? *args : Object());
        forwarded[seq] = req;
    }

    string FindLldbDap() {
        if (auto env = getenv("GOOSE_LLDB_DAP")) return env;
        #ifdef __APPLE__
            auto r = RunProcess({ "xcrun", "-f", "lldb-dap" });
            if (r.started && r.code == 0) {
                auto p = r.output;
                while (!p.empty() && isspace((uint8_t)p.back())) p.pop_back();
                if (!p.empty()) return p;
            }
        #endif
        return FindOnPath("lldb-dap");
    }

    void Launch(const Json &req) {
        auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
        auto program = args.Str("program");
        cwd = args.Str("cwd");
        if (EndsWith(program, ".goose")) {
            auto src = CanonicalPath(program);
            programdir = DirOf(src);
            auto r = RunProcess({ exe, "build", "--debug", "-O0", src }, DirOf(src));
            if (!r.started || r.code != 0) {
                Respond(req, false, Object(), cat("goose build failed:\n", r.output));
                return;
            }
            JsonSet(args, "program", JsonOfStr(src.substr(0, src.size() - 6)));
            string map;
            if (LoadFile(cat(ProgramBuildDir(src), "/", PathStem(src), ".debug.json"), map)) debugmap = ParseJson(map);
            if (cwd.empty()) cwd = DirOf(src);
        }
        forwarded[SendToChild("launch", args)] = req;
    }

    void StackTrace(const Json &req) {
        auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
        auto r = Call("stackTrace", args);
        auto body = r.Get("body");
        if (!Succeeded(r) || !body) { Respond(req, false, Object(), r.Str("message")); return; }
        auto raw = body->Arr("stackFrames");
        auto thread = JsonInt(args, "threadId");
        // A failed check never returns, so the frame that called it has no line of
        // its own; the check named the source line it failed on.
        auto failed = thread == failedthread && !failurefile.empty();
        auto out = Array();
        for (auto &f : raw) {
            auto frame = f;
            auto src = f.Get("source") ? f.Get("source")->Str("path") : string();
            auto cname = f.Str("name");
            auto fn = Function(cname);
            if (!fn) {
                // The program's own C keeps its name, its source and lldb's view of
                // its values; what the compiler wrote stays out of sight.
                if (ProgramC(src)) {
                    cframes.insert(JsonInt(f, "id"));
                    out.arr.push_back(std::move(frame));
                }
                continue;
            }
            if (failed && !EndsWith(src, ".goose")) {
                auto s = Object();
                JsonSet(s, "name", JsonOfStr(failurefile.substr(failurefile.rfind('/') + 1)));
                JsonSet(s, "path", JsonOfStr(failurefile));
                JsonSet(frame, "source", std::move(s));
                JsonSet(frame, "line", JsonOfInt(failureline));
                src = failurefile;
                failed = false;
            }
            JsonSet(frame, "name", JsonOfStr(fn->Str("name")));
            frames[JsonInt(f, "id")] = { src, (int)JsonInt(frame, "line"), cname };
            out.arr.push_back(std::move(frame));
        }
        // A worker has its own copies of only the globals it uses.
        auto onmain = false;
        for (auto &f : raw) onmain |= f.Str("name") == "main_g";
        for (auto &f : raw) if (auto it = frames.find(JsonInt(f, "id")); it != frames.end()) it->second.mainthread = onmain;
        auto reply = Object();
        JsonSet(reply, "totalFrames", JsonOfInt((int64_t)out.arr.size()));
        JsonSet(reply, "stackFrames", std::move(out));
        Respond(req, true, std::move(reply));
    }

    // Threads by what they are in Goose: a worker's function, where it is, and what
    // it waits on. A thread running none of the program's code yet keeps its name.
    void Threads(const Json &req) {
        // lldb-dap's first answer after a stop can hold only the stopped thread.
        Call("threads", Object());
        auto r = Call("threads", Object());
        auto body = r.Get("body");
        if (!Succeeded(r) || !body) { Respond(req, false, Object(), r.Str("message")); return; }
        auto out = Array();
        map<string, int> seen;
        for (auto &t : body->Arr("threads")) {
            auto thread = t;
            auto args = Object();
            JsonSet(args, "threadId", JsonOfInt(JsonInt(t, "id")));
            auto trace = Call("stackTrace", args);
            auto tb = trace.Get("body");
            string root, inner, waits;
            int line = 0;
            for (auto &f : tb ? tb->Arr("stackFrames") : vector<Json>{}) {
                auto cname = f.Str("name");
                if (cname == "gs_qget" || cname == "gs_qwait") waits = "waiting in qget";
                if (cname == "gs_thread_wait") waits = "waiting in thread_wait";
                if (inner.empty() && ProgramC(f.Get("source") ? f.Get("source")->Str("path") : string())) {
                    inner = cname;
                    line = (int)JsonInt(f, "line");
                }
                if (auto fn = Function(cname)) {
                    if (inner.empty()) {
                        inner = fn->Str("name");
                        line = (int)JsonInt(f, "line");
                        // A failed check's caller has no line of its own (StackTrace).
                        auto src = f.Get("source") ? f.Get("source")->Str("path") : string();
                        if (JsonInt(t, "id") == failedthread && !EndsWith(src, ".goose")) line = failureline;
                    }
                    root = fn->Str("name");
                }
            }
            // A worker before its first Goose line, or after its last.
            if (root.empty()) {
                for (auto &f : tb ? tb->Arr("stackFrames") : vector<Json>{}) {
                    auto cname = f.Str("name");
                    if (StartsWith(cname, "gs_tmain_")) {
                        auto fn = Function(cname.substr(9));
                        JsonSet(thread, "name", JsonOfStr(cat(fn ? fn->Str("name") : string("worker"), ": starting or ending")));
                    }
                }
            }
            if (!root.empty()) {
                auto n = ++seen[root];
                auto name = n > 1 ? cat(root, " ", n) : root;
                if (!waits.empty()) name += cat(": ", waits, ", line ", line);
                else if (inner != root) name += cat(": in ", inner, ", line ", line);
                else name += cat(": line ", line);
                JsonSet(thread, "name", JsonOfStr(name));
            }
            out.arr.push_back(std::move(thread));
        }
        auto reply = Object();
        JsonSet(reply, "threads", std::move(out));
        Respond(req, true, std::move(reply));
    }

    // A typed queue's messages, oldest first: a list of nodes, each a header ahead
    // of the value's bytes (runtime_threads.h).
    void QueueRows(vector<Json> &rows) {
        for (auto &q : debugmap.Arr("queues")) {
            string where;
            int64_t unused = 0;
            if (!Evaluate(cat("&", q.Str("c")), globalsframe, where, unused)) continue;
            auto state = ReadInt(DapAddress(where), 8);
            auto id = (int)JsonInt(q, "type");
            auto name = TypeOf(id).Str("name");
            Value v;
            v.text = "[";
            auto count = 0;
            for (auto node = state ? ReadInt(state, 8) : 0; node && count < 64; node = ReadInt(node, 8), count++) {
                auto e = ReadValue(node + 16, id, cat(name, "[", (int64_t)count, "]"), 1);
                v.text += cat(count ? ", " : "", e.text);
                v.children.push_back(Var(cat("[", (int64_t)count, "]"), name, "", e));
            }
            v.text += "]";
            v.ref = Expansion(v.children);
            rows.push_back(Var(name, cat("queue of ", (int64_t)count), "", v));
        }
    }

    void Scopes(const Json &req) {
        auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
        auto id = JsonInt(args, "frameId");
        auto r = Call("scopes", args);
        auto body = r.Get("body");
        if (!Succeeded(r) || !body) { Respond(req, false, Object(), r.Str("message")); return; }
        auto goose = frames.count(id) != 0;
        auto out = Array();
        for (auto &s : body->Arr("scopes")) {
            if (goose && s.Str("name") != "Locals") continue;
            if (goose) {
                // lldb-dap uses one reference for a scope in every frame.
                auto scope = s;
                auto ref = nextref++;
                JsonSet(scope, "variablesReference", JsonOfInt(ref));
                scopeframes[ref] = id;
                out.arr.push_back(std::move(scope));
                continue;
            }
            out.arr.push_back(s);
        }
        if (goose && frames[id].mainthread && !debugmap.Arr("globals").empty()) {
            globalsframe = id;
            auto g = Object();
            JsonSet(g, "name", JsonOfStr("Globals"));
            JsonSet(g, "variablesReference", JsonOfInt(globalsref));
            JsonSet(g, "expensive", JsonOfBool(false));
            out.arr.push_back(std::move(g));
        }
        if (goose && !debugmap.Arr("queues").empty()) {
            globalsframe = id;
            auto g = Object();
            JsonSet(g, "name", JsonOfStr("Queues"));
            JsonSet(g, "variablesReference", JsonOfInt(queuesref));
            JsonSet(g, "expensive", JsonOfBool(false));
            out.arr.push_back(std::move(g));
        }
        auto reply = Object();
        JsonSet(reply, "scopes", std::move(out));
        Respond(req, true, std::move(reply));
    }

    void Variables(const Json &req) {
        auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
        auto ref = JsonInt(args, "variablesReference");
        auto reply = Object();
        auto list = Array();
        auto cached = rendered.find(ref);
        if (ref == queuesref) {
            QueueRows(list.arr);
        } else if (ref == globalsref) {
            currentframe = globalsframe;
            for (auto &g : debugmap.Arr("globals")) {
                Value v;
                if (!LocalValue(g, currentframe, g.Str("name"), v)) continue;
                list.arr.push_back(Var(g.Str("name"), TypeOf((int)JsonInt(g, "type")).Str("name"), g.Str("name"), v));
            }
        } else if (cached != rendered.end()) {
            list.arr = cached->second;
        } else if (auto later = lazy.find(ref); later != lazy.end()) {
            auto l = later->second;
            auto v = ReadValue(l.addr, l.type, l.path, 0);
            if (v.ref && rendered.count(v.ref)) list.arr = rendered[v.ref];
        } else if (auto scope = scopeframes.find(ref); scope != scopeframes.end()) {
            currentframe = scope->second;
            auto &frame = frames[currentframe];
            map<string, const Json *> shown;
            vector<const Json *> order;
            if (auto fn = Function(frame.function)) {
                for (auto &l : fn->Arr("locals")) {
                    if (!Visible(l, frame.line)) continue;
                    auto &slot = shown[l.Str("name")];
                    if (!slot) order.push_back(&l);
                    if (slot && JsonInt(l, "line") == JsonInt(*slot, "line")) {
                        // A `const for` copies its body, all on one line: every copy
                        // shows, since a stop does not say which is running.
                        order.push_back(&l);
                        continue;
                    }
                    if (!slot || JsonInt(l, "line") > JsonInt(*slot, "line")) slot = &l;
                }
            }
            sort(order.begin(), order.end(), [](const Json *a, const Json *b) {
                auto pa = a->Get("param") && a->Get("param")->b, pb = b->Get("param") && b->Get("param")->b;
                if (pa != pb) return pa;
                return JsonInt(*a, "line") < JsonInt(*b, "line");
            });
            map<string, int> copies;
            for (auto first : order) {
                auto name = first->Str("name");
                auto local = JsonInt(*first, "line") == JsonInt(*shown[name], "line") ? first : shown[name];
                Value v;
                if (!LocalValue(*local, currentframe, name, v)) continue;
                auto copy = ++copies[name];
                auto label = copy == 1 ? name : cat(name, " (", copy, ")");
                list.arr.push_back(Var(label, TypeOf((int)JsonInt(*local, "type")).Str("name"), name, v));
            }
            rendered[ref] = list.arr;
        } else {
            Forward(req);
            return;
        }
        JsonSet(reply, "variables", std::move(list));
        Respond(req, true, std::move(reply));
    }

    // --- expressions in the console ---------------------------------------

    struct Place { uint64_t addr = 0; int type = -1; string path; };
    struct Result {
        enum Kind { INT, FLOAT, BOOL, PLACE } kind = INT;
        int64_t n = 0;
        double f = 0;
        Place place;
    };
    struct ExprError { string message; };

    const string *expr = nullptr;
    size_t pos = 0;
    Frame *exprframe = nullptr;
    int64_t exprframeid = 0;

    void Spaces() { while (pos < expr->size() && isspace((uint8_t)(*expr)[pos])) pos++; }
    bool Eat(string_view tok) {
        Spaces();
        if (expr->compare(pos, tok.size(), tok) != 0) return false;
        pos += tok.size();
        return true;
    }
    bool WordAt() {
        Spaces();
        return pos < expr->size() && (isalpha((uint8_t)(*expr)[pos]) || (*expr)[pos] == '_');
    }
    string Word() {
        auto start = pos;
        while (pos < expr->size() && (isalnum((uint8_t)(*expr)[pos]) || (*expr)[pos] == '_')) pos++;
        return expr->substr(start, pos - start);
    }

    // Through references, to what they name.
    void Through(Place &p) {
        for (;;) {
            auto &t = TypeOf(p.type);
            if (t.Str("kind") != "ref") return;
            auto sub = (int)JsonInt(t, "sub");
            uint64_t target = 0;
            if (auto rel = t.Get("relative"); rel && rel->b) {
                if (t.Get("pool")->b) throw ExprError { "a pool reference needs its pool" };
                int64_t width = JsonInt(t, "size");
                auto off = width ? ReadInt(p.addr, (int)width) : ReadUleb(p.addr, width);
                target = off ? p.addr + off : 0;
            } else {
                target = ReadInt(p.addr, 8);
            }
            if (!target) throw ExprError { cat(p.path, " is null") };
            p.addr = target;
            p.type = sub;
        }
    }

    // An array's elements, count and element type, whatever holds them.
    bool Elements(const Place &p, uint64_t &data, int64_t &count, int &sub) {
        auto &t = TypeOf(p.type);
        if (t.Str("kind") == "slice") {
            data = ReadInt(p.addr, 8);
            count = (int64_t)ReadInt(p.addr + 8, 8);
            sub = (int)JsonInt(t, "sub");
            return true;
        }
        if (t.Str("kind") != "array") return false;
        sub = (int)JsonInt(t, "sub");
        auto shape = t.Str("shape");
        if (shape == "fixed") { data = p.addr; count = JsonInt(t, "count"); }
        else if (shape == "resizable") { data = ReadInt(p.addr, 8); count = (int64_t)ReadInt(p.addr + 8, 8); }
        else if (shape == "limited") {
            auto lensize = (int)(JsonInt(t, "size") - JsonInt(t, "count") * JsonInt(TypeOf(sub), "size"));
            data = p.addr + lensize;
            count = (int64_t)ReadInt(p.addr, lensize);
        } else if (shape == "limited_runtime") { data = p.addr + 8; count = (int64_t)ReadInt(p.addr + 4, 4); }
        else if (shape == "var") {
            auto ls = (int)JsonInt(t, "lensize");
            int64_t prefix = ls;
            count = (int64_t)(ls ? ReadInt(p.addr, ls) : ReadUleb(p.addr, prefix));
            data = p.addr + prefix;
        } else return false;
        return true;
    }

    Place FieldOf(Place p, const string &name) {
        Through(p);
        auto &t = TypeOf(p.type);
        if (t.Str("kind") != "struct") throw ExprError { cat(p.path, " has no fields") };
        auto &fields = t.Arr("fields");
        auto at = p.addr;
        for (size_t i = 0; i < fields.size(); i++) {
            auto &f = fields[i];
            if (f.Get("pad")) { if (JsonInt(f, "pad") > 0) at += JsonInt(f, "pad"); continue; }
            auto ft = (int)JsonInt(f, "type");
            if (t.Get("frameobject") && i + 1 == fields.size()) {
                string size;
                int64_t unused = 0;
                if (Evaluate(cat("sizeof(", t.Str("frameobject"), ")"), currentframe, size, unused))
                    at = p.addr + DapScalar(size) - 16;
            }
            if (f.Str("name") == name) return { at, ft, cat(p.path, ".", name) };
            at += ReadValue(at, ft, "", 0).size;
        }
        throw ExprError { cat(p.path, " has no field ", name) };
    }

    Place IndexOf(Place p, int64_t n) {
        Through(p);
        uint64_t data = 0;
        int64_t count = 0;
        int sub = 0;
        if (!Elements(p, data, count, sub)) throw ExprError { cat(p.path, " is not an array") };
        if (n < 0 || n >= count) throw ExprError { cat("index ", n, " out of bounds (length ", count, ")") };
        auto &st = TypeOf(sub);
        auto at = data;
        if (st.Get("size")) at += (uint64_t)n * (uint64_t)JsonInt(st, "size");
        else for (int64_t i = 0; i < n; i++) at += ReadValue(at, sub, "", 0).size;
        return { at, sub, cat(p.path, "[", n, "]") };
    }

    // A number or a truth read out of memory takes part in arithmetic as itself.
    Result Scalar(Result r) {
        auto &p = r.place;
        auto &t = TypeOf(p.type);
        auto kind = t.Str("kind");
        if (kind != "int" && kind != "float" && kind != "bool") return r;
        auto text = ReadValue(p.addr, p.type, p.path, 0).text;
        if (kind == "bool") { r.kind = Result::BOOL; r.n = text == "true"; }
        else if (kind == "float") { r.kind = Result::FLOAT; r.f = strtod(text.c_str(), nullptr); }
        else { r.kind = Result::INT; r.n = strtoll(text.c_str(), nullptr, 10); }
        return r;
    }

    Result Primary() {
        Spaces();
        Result r;
        if (Eat("(")) {
            r = Or();
            if (!Eat(")")) throw ExprError { "expected )" };
            return r;
        }
        if (pos < expr->size() && isdigit((uint8_t)(*expr)[pos])) {
            char *end = nullptr;
            auto c = expr->c_str() + pos;
            auto v = strtod(c, &end);
            auto isfloat = false;
            for (auto q = c; q < end; q++) isfloat |= *q == '.' || *q == 'e' || *q == 'E';
            if (isfloat) { r.kind = Result::FLOAT; r.f = v; }
            else r.n = strtoll(c, nullptr, 10);
            pos += (size_t)(end - c);
            return r;
        }
        if (!WordAt()) throw ExprError { "expected a value" };
        auto name = Word();
        if (name == "true" || name == "false") { r.kind = Result::BOOL; r.n = name == "true"; return r; }
        auto local = FindLocal(*exprframe, name);
        uint64_t addr = 0;
        if (!local) throw ExprError { cat("no ", name, " here") };
        if (local->Get("value")) { r.n = strtoll(local->Str("value").c_str(), nullptr, 10); return r; }
        if (local->Get("components")) {
            if (!Eat(".")) throw ExprError { cat(name, " is a root struct: name a field") };
            auto field = Word();
            for (auto &c : local->Arr("components")) if (c.Str("name") == field) local = &c;
            if (local->Get("components")) throw ExprError { cat(name, " has no field ", field) };
            name += "." + field;
        }
        if (!LocalAddress(*local, exprframeid, addr)) throw ExprError { cat("cannot read ", name) };
        Place p { addr, (int)JsonInt(*local, "type"), name };
        for (;;) {
            if (Eat(".")) {
                if (!WordAt()) throw ExprError { "expected a field" };
                auto field = Word();
                if (field == "len") {
                    auto q = p;
                    Through(q);
                    uint64_t data = 0;
                    int64_t count = 0;
                    int sub = 0;
                    if (Elements(q, data, count, sub)) { r.n = count; r.kind = Result::INT; return r; }
                }
                p = FieldOf(p, field);
            } else if (Eat("[")) {
                auto i = Number(Or());
                if (!Eat("]")) throw ExprError { "expected ]" };
                p = IndexOf(p, (int64_t)i);
            } else if (Eat("(")) {
                throw ExprError { "the console reads the stopped program; it cannot call functions" };
            } else {
                break;
            }
        }
        r.kind = Result::PLACE;
        r.place = p;
        return Scalar(r);
    }

    double Number(const Result &r) {
        if (r.kind == Result::FLOAT) return r.f;
        if (r.kind == Result::PLACE) throw ExprError { cat(r.place.path, " is not a number") };
        return (double)r.n;
    }
    static Result Int(int64_t n) { Result r; r.n = n; return r; }
    static Result Bool(bool b) { Result r; r.kind = Result::BOOL; r.n = b; return r; }

    Result Arith(const Result &a, const Result &b, char op) {
        auto x = Number(a), y = Number(b);
        if (a.kind == Result::FLOAT || b.kind == Result::FLOAT) {
            Result r;
            r.kind = Result::FLOAT;
            r.f = op == '+' ? x + y : op == '-' ? x - y : op == '*' ? x * y : x / y;
            return r;
        }
        if ((op == '/' || op == '%') && b.n == 0) throw ExprError { "division by zero" };
        // `%` is Euclidean in Goose.
        if (op == '%') { auto m = a.n % b.n; return Int(m < 0 ? m + (b.n < 0 ? -b.n : b.n) : m); }
        return Int(op == '+' ? a.n + b.n : op == '-' ? a.n - b.n : op == '*' ? a.n * b.n : a.n / b.n);
    }

    Result Unary() {
        Spaces();
        if (pos + 1 < expr->size() && (*expr)[pos] == '!' && (*expr)[pos + 1] != '=') {
            pos++;
            auto v = Unary();
            if (v.kind != Result::BOOL) throw ExprError { "! takes a bool" };
            return Bool(!v.n);
        }
        if (pos < expr->size() && (*expr)[pos] == '-') {
            pos++;
            auto v = Unary();
            Number(v);
            if (v.kind == Result::FLOAT) { v.f = -v.f; return v; }
            return Int(-v.n);
        }
        return Primary();
    }
    Result Mul() {
        auto a = Unary();
        for (;;) {
            Spaces();
            auto c = pos < expr->size() ? (*expr)[pos] : 0;
            if (c != '*' && c != '/' && c != '%') return a;
            pos++;
            a = Arith(a, Unary(), c);
        }
    }
    Result Add() {
        auto a = Mul();
        for (;;) {
            Spaces();
            auto c = pos < expr->size() ? (*expr)[pos] : 0;
            if (c != '+' && c != '-') return a;
            pos++;
            a = Arith(a, Mul(), c);
        }
    }
    Result Cmp() {
        auto a = Add();
        const char *ops[] = { "==", "!=", "<=", ">=", "<", ">" };
        for (auto op : ops) {
            if (!Eat(op)) continue;
            auto b = Add();
            string o = op;
            if (a.kind == Result::BOOL && b.kind == Result::BOOL) {
                if (o == "==") return Bool(a.n == b.n);
                if (o == "!=") return Bool(a.n != b.n);
                throw ExprError { "bools have no order" };
            }
            auto x = Number(a), y = Number(b);
            return Bool(o == "==" ? x == y : o == "!=" ? x != y : o == "<=" ? x <= y : o == ">=" ? x >= y : o == "<" ? x < y : x > y);
        }
        return a;
    }
    Result And() {
        auto a = Cmp();
        while (Eat("&&")) { auto b = Cmp(); a = Bool(a.n && b.n); }
        return a;
    }
    Result Or() {
        auto a = And();
        while (Eat("||")) { auto b = And(); a = Bool(a.n || b.n); }
        return a;
    }

    void EvaluateRequest(const Json &req) {
        auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
        auto id = JsonInt(args, "frameId");
        if (cframes.count(id)) {
            // A frame in C reads C. A client's console asks in "repl", which lldb-dap
            // would take for a debugger command.
            auto forward = req;
            JsonSet(args, "context", JsonOfStr("watch"));
            JsonSet(forward, "arguments", std::move(args));
            Forward(forward);
            return;
        }
        auto it = frames.find(id);
        if (it == frames.end()) {
            if (Functions()) Respond(req, false, Object(), "that frame is not part of the current stop");
            else Forward(req);
            return;
        }
        auto text = args.Str("expression");
        expr = &text;
        pos = 0;
        exprframe = &it->second;
        exprframeid = id;
        currentframe = id;
        auto reply = Object();
        try {
            auto r = Or();
            Spaces();
            if (pos != text.size()) throw ExprError { cat("cannot read past ", text.substr(0, pos)) };
            string value;
            int type = -1;
            int64_t ref = 0;
            if (r.kind == Result::PLACE) {
                auto v = ReadValue(r.place.addr, r.place.type, r.place.path, 0);
                value = v.text;
                ref = v.ref;
                type = r.place.type;
            } else if (r.kind == Result::BOOL) {
                value = r.n ? "true" : "false";
            } else if (r.kind == Result::FLOAT) {
                char buf[64];
                snprintf(buf, sizeof buf, "%.15g", r.f);
                value = buf;
            } else {
                value = to_string(r.n);
            }
            JsonSet(reply, "result", JsonOfStr(value));
            if (type >= 0) JsonSet(reply, "type", JsonOfStr(TypeOf(type).Str("name")));
            JsonSet(reply, "variablesReference", JsonOfInt(ref));
            Respond(req, true, std::move(reply));
        } catch (const ExprError &e) {
            Respond(req, false, Object(), e.message);
        }
    }

    void SetFunctionBreakpoints(const Json &req) {
        auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
        auto list = Array();
        if (auto b = args.Get("breakpoints")) list = *b;
        auto clientcount = list.arr.size();
        auto check = Object();
        JsonSet(check, "name", JsonOfStr("gs_fail"));
        list.arr.push_back(std::move(check));
        JsonSet(args, "breakpoints", std::move(list));
        auto r = Call("setFunctionBreakpoints", args);
        auto reply = Object();
        auto bps = Array();
        if (auto body = r.Get("body")) {
            bps.arr = body->Arr("breakpoints");
            if (bps.arr.size() > clientcount) bps.arr.resize(clientcount);
        }
        JsonSet(reply, "breakpoints", std::move(bps));
        Respond(req, Succeeded(r), std::move(reply));
    }

    void ExceptionInfo(const Json &req) {
        // No check failed, so the stop is lldb's: a signal, usually in C.
        if (failuretext.empty()) { Forward(req); return; }
        auto reply = Object();
        JsonSet(reply, "exceptionId", JsonOfStr("runtime check"));
        JsonSet(reply, "description", JsonOfStr(failuretext));
        JsonSet(reply, "breakMode", JsonOfStr("always"));
        Respond(req, true, std::move(reply));
    }

    void OnClient(const Json &req) {
        auto cmd = req.Str("command");
        if (req.Str("type") != "request") {
            // A reply to something lldb asked of the client.
            JsonOut out;
            JsonWrite(out, req);
            Write(child.in, out.s);
            return;
        }
        if (cmd == "initialize") {
            // What the protocol defaults, stated: lldb-dap refuses a launch without.
            auto init = req;
            auto args = req.Get("arguments") ? *req.Get("arguments") : Object();
            if (!args.Get("linesStartAt1")) JsonSet(args, "linesStartAt1", JsonOfBool(true));
            if (!args.Get("columnsStartAt1")) JsonSet(args, "columnsStartAt1", JsonOfBool(true));
            if (!args.Get("pathFormat")) JsonSet(args, "pathFormat", JsonOfStr("path"));
            JsonSet(init, "arguments", std::move(args));
            Forward(init);
        }
        else if (cmd == "launch") Launch(req);
        else if (cmd == "stackTrace") StackTrace(req);
        else if (cmd == "scopes") Scopes(req);
        else if (cmd == "threads") Threads(req);
        else if (cmd == "variables") Variables(req);
        else if (cmd == "evaluate") EvaluateRequest(req);
        else if (cmd == "exceptionInfo") ExceptionInfo(req);
        else if (cmd == "setFunctionBreakpoints") SetFunctionBreakpoints(req);
        else if (cmd == "configurationDone") {
            SetFunctionBreakpointsForCheck();
            Forward(req);
        } else {
            if (cmd == "continue" || cmd == "next" || cmd == "stepIn" || cmd == "stepOut" || cmd == "pause")
                ClearStop();
            Forward(req);
        }
    }

    // A client that names no function breakpoints still stops on a failed check.
    void SetFunctionBreakpointsForCheck() {
        auto args = Object();
        auto list = Array();
        auto check = Object();
        JsonSet(check, "name", JsonOfStr("gs_fail"));
        list.arr.push_back(std::move(check));
        JsonSet(args, "breakpoints", std::move(list));
        Call("setFunctionBreakpoints", args);
    }

    bool UserFrame(const string &path) const {
        if (!EndsWith(path, ".goose")) return false;
        return cwd.empty() || DirOf(path) == (EndsWith(cwd, "/") ? cwd : cwd + "/");
    }

    // C the program brought along, beside its source: not the C the compiler
    // generated, nor a package's, nor the runtime's.
    bool ProgramC(const string &path) const {
        if (programdir.empty() || !StartsWith(path, programdir)) return false;
        return EndsWith(path, ".c") || EndsWith(path, ".h") || EndsWith(path, ".cc") || EndsWith(path, ".cpp");
    }

    // A stop is the client's to see unless it is inside what the compiler wrote: a
    // step that lands in generated C or the standard library carries on. The
    // program's own C is the client's too, and a step from it into an `export fn`'s
    // wrapper carries on inward, to the Goose the wrapper calls.
    void OnStopped(Json ev) {
        auto body = ev.Get("body") ? *ev.Get("body") : Object();
        auto thread = JsonInt(body, "threadId");
        auto reason = body.Str("reason");
        auto args = Object();
        JsonSet(args, "threadId", JsonOfInt(thread));
        auto trace = Call("stackTrace", args);
        auto tb = trace.Get("body");
        auto raw = tb ? tb->Arr("stackFrames") : vector<Json>{};
        auto hasgoose = false;
        for (auto &f : raw) hasgoose |= Function(f.Str("name")) != nullptr;
        if (!raw.empty()) {
            auto s = raw[0].Get("source");
            // A breakpoint on a function's own line lands in the prologue the
            // compiler wrote, ahead of anything in the source.
            if (Function(raw[0].Str("name")) && !(s && EndsWith(s->Str("path"), ".goose"))) {
                Call("stepIn", args);
                return;
            }
        }
        // A `return ... from` is on its way out through these frames, each checking
        // the flag: a step ends where it lands.
        if (!raw.empty() && reason == "step" && debugmap.Get("longreturn") && debugmap.Get("longreturn")->b) {
            string inflight;
            int64_t unused = 0;
            if (Evaluate("gs_rf", JsonInt(raw[0], "id"), inflight, unused) && DapScalar(inflight) != 0) {
                unwinding = true;
                Call("next", args);
                return;
            }
            // The flag cleared in the function being returned from: finish its return.
            if (unwinding) {
                unwinding = false;
                Call("stepOut", args);
                return;
            }
        }
        if (!raw.empty() && reason == "step") {
            auto s = raw[0].Get("source");
            auto path = s ? s->Str("path") : string();
            if (!UserFrame(path) && !ProgramC(path)) {
                auto caller = raw.size() > 1 && raw[1].Get("source") ? raw[1].Get("source")->Str("path") : string();
                auto inward = (Function(raw[0].Str("name")) && !EndsWith(path, ".goose")) || ProgramC(caller);
                auto onward = !hasgoose ? "continue" : inward ? "stepIn" : "stepOut";
                Call(onward, args);
                return;
            }
        }
        if (!raw.empty() && raw[0].Str("name") == "gs_fail") {
            auto top = JsonInt(raw[0], "id");
            string msg, len, file, line;
            int64_t ref = 0;
            if (Evaluate("(const char*)msg", top, msg, ref) && Evaluate("len", top, len, ref))
                failuretext = DapBytes(msg, DapScalar(len));
            if (Evaluate("file", top, file, ref) && Evaluate("line", top, line, ref)) {
                auto name = DapBytes(file, (size_t)-1);
                failurefile = !name.empty() && name[0] != '/' && !cwd.empty() ? cat(cwd, "/", name) : name;
                failureline = (int)DapScalar(line);
                failedthread = thread;
            }
            JsonSet(body, "reason", JsonOfStr("exception"));
            JsonSet(body, "description", JsonOfStr("Runtime check failed"));
            JsonSet(body, "text", JsonOfStr(failuretext));
            JsonSet(ev, "body", std::move(body));
        }
        ToClient(std::move(ev));
    }

    void OnChild(Json msg) {
        auto type = msg.Str("type");
        if (type == "event") {
            auto event = msg.Str("event");
            if (event == "stopped") { OnStopped(std::move(msg)); return; }
            if (event == "continued") ClearStop();
            ToClient(std::move(msg));
            return;
        }
        if (type == "response") {
            auto it = forwarded.find(JsonInt(msg, "request_seq"));
            if (it == forwarded.end()) return;
            auto req = std::move(it->second);
            forwarded.erase(it);
            JsonSet(msg, "request_seq", JsonOfInt(JsonInt(req, "seq")));
            if (req.Str("command") == "initialize") {
                auto body = msg.Get("body") ? *msg.Get("body") : Object();
                JsonSet(body, "supportsExceptionInfoRequest", JsonOfBool(true));
                JsonSet(msg, "body", std::move(body));
            }
            ToClient(std::move(msg));
            return;
        }
        ToClient(std::move(msg));
    }

    int Run(const string &lldb) {
        if (!child.Start({ lldb })) {
            fprintf(stderr, "goose dap: cannot start %s\n", lldb.c_str());
            return 1;
        }
        thread([this] { Reader(stdin, true); }).detach();
        thread([this] { Reader(child.out, false); }).detach();
        for (;;) {
            Message m;
            if (!deferred.empty()) {
                m = std::move(deferred.front());
                deferred.pop_front();
            } else {
                m = Next();
            }
            if (m.body.empty()) {
                // A client gone without a disconnect takes its debuggee with it.
                if (m.fromclient) {
                    auto args = Object();
                    JsonSet(args, "terminateDebuggee", JsonOfBool(true));
                    Call("disconnect", args);
                }
                return 0;
            }
            auto j = ParseJson(m.body);
            if (m.fromclient) OnClient(j);
            else OnChild(std::move(j));
        }
    }
};

inline int RunDap(const string &exe, const vector<string> &args) {
    string lldb;
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--lldb-dap" && i + 1 < args.size()) { lldb = args[++i]; continue; }
        fprintf(stderr, "goose dap takes --lldb-dap path, not %s\n", args[i].c_str());
        return 1;
    }
    SetBinaryStdio();
    DapAdapter adapter;
    adapter.exe = exe;
    if (lldb.empty()) lldb = adapter.FindLldbDap();
    if (lldb.empty()) {
        fprintf(stderr, "goose dap needs lldb-dap: put it on PATH, or name it with --lldb-dap or GOOSE_LLDB_DAP\n");
        return 1;
    }
    auto code = adapter.Run(lldb);
    fflush(stdout);
    _Exit(code);
}

}  // namespace goose
