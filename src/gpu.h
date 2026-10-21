// Goose compiler — GPU functions (docs/design/gpu.md §5, §6). After
// typechecking, every entry point a gpu_entry call names is compiled as a
// program of its own: its call graph is held to the GPU subset as it is
// emitted, once as HLSL and once as MSL, and the text goes to the
// platform's compilers, DXC for SPIR-V and DXIL and the Metal toolchain for
// a metallib. The result is a shader library (src/gpu_library.h) that the
// gpu_entry call embeds as static data, as embed_shader embeds its blobs.
//
// The emitters work on the checked, unoptimized specializations, which the
// optimizer never sees (a gpu_entry call does not make its entry live): the
// emitted text keeps the functions, names and control flow of the source,
// and the GPU compilers inline and fold it themselves.
#pragma once

#include "gpu_library.h"

namespace goose {

enum GpuDialect { GD_HLSL_DXIL, GD_HLSL_SPIRV, GD_MSL };

// An error in a GPU function: reported at `where`, with the call chain from
// the entry point that reached it.
struct GpuError {
    Line where;
    string msg;
};

// The names neither HLSL nor MSL lets a declaration take, which a Goose name
// gets an underscore after. Not every keyword of each: those a Goose name
// could plausibly be.
inline bool GpuReservedName(string_view n) {
    static const unordered_set<string_view> words = {
        "auto", "bool", "break", "buffer", "case", "cbuffer", "centroid", "char", "class",
        "column_major", "compile", "const", "constant", "constexpr", "continue", "default",
        "delete", "device", "discard", "do", "double", "dword", "else", "enum", "explicit",
        "export", "extern", "false", "float", "for", "fragment", "goto", "groupshared",
        "half", "if", "in", "inline", "inout", "int", "kernel", "line", "linear", "long",
        "main", "matrix", "min16float", "min16int", "namespace", "new", "nointerpolation",
        "noperspective", "operator", "out", "packoffset", "point", "precise", "private",
        "public", "register", "return", "row_major", "sample", "sampler", "shared", "short",
        "signed", "sizeof", "snorm", "static", "string", "struct", "switch", "template",
        "texture", "this", "thread", "threadgroup", "triangle", "true", "typedef", "uint",
        "uniform", "union", "unorm", "unsigned", "using", "vector", "vertex", "virtual",
        "void", "volatile", "while", "abs", "all", "any", "clamp", "cross", "dot", "floor",
        "length", "lerp", "max", "min", "mix", "normalize", "pow", "round", "select", "sign",
        "sqrt", "step",
    };
    return words.count(n) || n.starts_with("gs_");
}

// FNV-1a: the shader cache's key (docs/design/gpu.md §6).
inline uint64_t GpuHash(string_view s, uint64_t h = 1469598103934665603ull) {
    for (auto c : s) h = (h ^ (uint8_t)c) * 1099511628211ull;
    return h;
}

// The emitter for one entry point in one dialect.
struct GpuEmitter {
    Ast &ast;
    GpuDialect dia;
    FnSpec *entry;
    bool msl() const { return dia == GD_MSL; }
    bool hlsl() const { return dia != GD_MSL; }

    string structdefs;                      // Struct declarations, in dependency order.
    string fndefs;                          // Function definitions, callees first.
    unordered_map<StructInst *, string> structnames;
    unordered_map<FnSpec *, string> fnnames;
    set<string> globalnames;                // Every top-level name taken.
    set<FnSpec *> inprogress;               // The functions on the emitter's stack.
    vector<FnSpec *> chain;                 // The same, in call order, for errors.
    vector<Line> callsites;                 // Where each of `chain` after the first was called.

    // The function being emitted.
    struct FnState {
        FnSpec *spec = nullptr;
        string out;
        int ind = 1;
        unordered_map<VarDef *, string> locals;
        set<string> names;
        int temps = 0;
        // The destination a value-giving loop's breaks assign, innermost last.
        vector<string> breakdsts;
    };
    FnState *fs = nullptr;

    GpuEmitter(Ast &_ast, GpuDialect _dia, FnSpec *_entry) : ast(_ast), dia(_dia), entry(_entry) {}

    [[noreturn]] void Fail(Line l, const string &msg) { throw GpuError { l, msg }; }
    [[noreturn]] void Fail(const Node *n, const string &msg) { Fail(n->line, msg); }

    void L(const string &s) {
        fs->out += string(fs->ind * 4, ' ');
        fs->out += s;
        fs->out += "\n";
    }

    // `#line` before a statement, so that a message the GPU compiler gives
    // names the Goose source.
    void LineDirective(Line l) {
        if (l.fileidx < 0 || l.fileidx >= (int)ast.sources.size()) return;
        string path = ast.sources[l.fileidx].first;
        for (auto &c : path) if (c == '\\') c = '/';
        fs->out += cat("#line ", l.line, " \"", path, "\"\n");
    }

    // ------------------------------------------------------------------
    // Types.

    static bool IsScalar(const TypeExpr *t) {
        return (t->kind == TY_INT && (t->intstorage == IS_I32 || t->intstorage == IS_U32)) ||
               (t->kind == TY_FLT && t->fltstorage == FS_F32) || t->kind == TY_BOOL;
    }

    // vec2/vec3/vec4 of a scalar (stdlib vec): its width, else 0.
    static int VecWidth(const TypeExpr *t) {
        if (t->kind != TY_STRUCT || !t->struc->st || !t->struc->st->ns.empty()) return 0;
        auto n = t->struc->st->name;
        auto w = n == "vec2" ? 2 : n == "vec3" ? 3 : n == "vec4" ? 4 : 0;
        if (!w || t->struc->args.size() != 1 || !IsScalar(t->struc->args[0])) return 0;
        return w;
    }

    static TypeExpr *VecElem(const TypeExpr *t) { return t->struc->args[0]; }

    string ScalarName(const TypeExpr *t, Line l) {
        if (t->kind == TY_BOOL) return "bool";
        if (t->kind == TY_FLT) {
            if (t->fltstorage != FS_F32)
                Fail(l, "f64 does not run on the GPU: MSL has no doubles. A float computed from "
                        "literals is an f64 where nothing gives it a type, so write `: f32` on "
                        "the let (docs/design/gpu.md §5.2)");
            return "float";
        }
        if (t->kind == TY_INT) {
            if (t->intstorage == IS_I32) return "int";
            if (t->intstorage == IS_U32) return "uint";
            Fail(l, cat(TypeStr(t), " does not run on the GPU yet, which computes in i32, u32 and "
                        "f32: write the type, or `as` it (docs/design/gpu.md §5.2)"));
        }
        Fail(l, cat(TypeStr(t), " is not a scalar"));
    }

    static string TypeStr(const TypeExpr *t) {
        string s;
        t->Dump(s);
        return s;
    }

    // A type's name in the dialect. `memory` asks for its form in a buffer
    // or a struct, which MSL spells with packed vectors so that a struct
    // lays out as Goose packs it (§5.3).
    string TypeName(TypeExpr *t, Line l, bool memory = false) {
        if (IsScalar(t) || t->kind == TY_INT || t->kind == TY_FLT) return ScalarName(t, l);
        if (t->kind == TY_SLICE)
            Fail(l, cat("slices do not run on the GPU, strings among them: ", TypeStr(t)));
        if (auto w = VecWidth(t)) {
            auto e = ScalarName(VecElem(t), l);
            if (memory && msl()) {
                if (e == "bool") Fail(l, "a vector of bool has no layout on the GPU");
                return cat("packed_", e, w);
            }
            return cat(e, w);
        }
        if (t->kind == TY_STRUCT) return StructName(t, l);
        if (t->kind == TY_REF)
            Fail(l, cat("references do not run on the GPU, except an entry point's uniform "
                        "block: ", TypeStr(t)));
        Fail(l, cat(TypeStr(t), " does not run on the GPU (docs/design/gpu.md §5.2)"));
    }

    string Mangle(const TypeExpr *t) {
        string s;
        t->Dump(s);
        string r;
        for (auto c : s) r += isalnum((unsigned char)c) ? c : '_';
        while (!r.empty() && r.back() == '_') r.pop_back();
        return r;
    }

    string FreshGlobal(string base) {
        if (GpuReservedName(base)) base += "_";
        auto n = base;
        for (int k = 2; globalnames.count(n); k++) n = cat(base, "_", k);
        globalnames.insert(n);
        return n;
    }

    // The field of a struct a GPU function uses, as a name the dialect
    // takes.
    static string FieldName(string_view n) {
        return GpuReservedName(n) ? cat(n, "_") : string(n);
    }

    // Goose's packed size of a GPU type, the stride of a buffer of it.
    int64_t PackedSize(TypeExpr *t, Line l) {
        if (t->kind == TY_BOOL) Fail(l, "bool has no layout on the GPU: use u32");
        if (IsScalar(t)) return 4;
        if (auto w = VecWidth(t)) return PackedSize(VecElem(t), l) * w;
        if (t->kind == TY_STRUCT) {
            auto si = Inst(t, l);
            int64_t size = 0;
            for (size_t i = 0; i < si->st->fields.size(); i++) {
                auto &f = si->st->fields[i];
                if (f.ispad) size += f.padsize >= 0 ? f.padsize : (4 - size % 4) % 4;
                else size += PackedSize(si->ftypes[i], f.line);
            }
            return size;
        }
        TypeName(t, l);  // Fails, saying why.
        return 0;
    }

    StructInst *Inst(TypeExpr *t, Line l) {
        auto si = t->struc->inst;
        if (!si) Fail(l, cat("internal: no instance for ", TypeStr(t)));
        return si;
    }

    string StructName(TypeExpr *t, Line l) {
        auto si = Inst(t, l);
        if (auto it = structnames.find(si); it != structnames.end()) return it->second;
        string base = string(si->st->qname);
        for (auto &c : base) if (c == ':') c = '_';
        while (base.find("__") != string::npos) base.replace(base.find("__"), 2, "_");
        for (auto a : si->args) base += "_" + Mangle(a);
        auto name = FreshGlobal(base);
        structnames[si] = name;
        // The fields first, so that a struct follows the ones it holds.
        string body;
        int64_t off = 0;
        int pads = 0;
        for (size_t i = 0; i < si->st->fields.size(); i++) {
            auto &f = si->st->fields[i];
            if (f.ispad) {
                auto n = f.padsize >= 0 ? f.padsize : (4 - off % 4) % 4;
                if (n % 4)
                    Fail(f.line, cat("a struct on the GPU pads in whole 4-byte words, not ", n,
                                     " bytes"));
                if (n) body += cat("    uint gs_pad", pads++, "[", n / 4, "];\n");
                off += n;
                continue;
            }
            auto ft = si->ftypes[i];
            if (ft->kind == TY_BOOL)
                Fail(f.line, cat("field ", f.name, " is a bool, which has no layout on the GPU: "
                                 "use u32"));
            body += cat("    ", TypeName(ft, f.line, true), " ", FieldName(f.name), ";\n");
            off += PackedSize(ft, f.line);
        }
        if (body.empty()) body = "    uint gs_empty;\n";
        structdefs += cat("struct ", name, " {\n", body, "};\n\n");
        return name;
    }

    // ------------------------------------------------------------------
    // Locals and temporaries.

    string LocalName(VarDef *v) {
        if (auto it = fs->locals.find(v); it != fs->locals.end()) return it->second;
        string base(v->name);
        if (base.empty() || GpuReservedName(base)) base += "_";
        auto n = base;
        for (int k = 2; fs->names.count(n) || globalnames.count(n); k++) n = cat(base, "_", k);
        fs->names.insert(n);
        fs->locals[v] = n;
        return n;
    }

    string Temp() {
        for (;;) {
            auto n = cat("t", fs->temps++);
            if (!fs->names.count(n) && !globalnames.count(n)) {
                fs->names.insert(n);
                return n;
            }
        }
    }

    // ------------------------------------------------------------------
    // Literals.

    string IntText(int64_t v, TypeExpr *t, Line l) {
        if (t->kind == TY_FLT) return FltText((double)v, t, l);
        auto s = ScalarName(t, l);
        if (s == "uint") return cat((uint64_t)(uint32_t)v, "u");
        if (v == INT32_MIN) return "(-2147483647 - 1)";
        return cat(v);
    }

    string FltText(double v, TypeExpr *t, Line l) {
        ScalarName(t, l);
        auto f = (float)v;
        if (std::isnan(f)) return hlsl() ? "asfloat(0x7fc00000u)" : "as_type<float>(0x7fc00000u)";
        if (std::isinf(f))
            return f > 0 ? (hlsl() ? "asfloat(0x7f800000u)" : "INFINITY")
                         : (hlsl() ? "asfloat(0xff800000u)" : "(-INFINITY)");
        char buf[64];
        snprintf(buf, sizeof buf, "%.9g", (double)f);
        string s = buf;
        if (s.find_first_of(".e") == string::npos) s += ".0";
        return s + "f";
    }

    // A float made of literals alone that nothing gave a type is an f64
    // (spec §6.3), converted where it meets an f32. The GPU has no f64, so
    // it takes the value the CPU would convert: folded at f64 here, and
    // emitted as one f32 literal.
    TypeExpr *LitFloatType(Node *n) {
        auto t = n->exprtype;
        return t && t->kind == TY_FLT && t->fltstorage == FS_F64 ? ast.flttypes[FS_F32] : t;
    }

    bool FoldLitFloat(Node *n, double &out) {
        if (auto f = Is<FltLit>(n)) { out = f->val; return true; }
        if (auto i = Is<IntLit>(n)) { out = (double)i->val; return true; }
        if (auto u = Is<Unary>(n)) {
            if (u->op != T_MINUS || !FoldLitFloat(u->child, out)) return false;
            out = -out;
            return true;
        }
        auto b = Is<Binary>(n);
        double l, r;
        if (!b || !FoldLitFloat(b->left, l) || !FoldLitFloat(b->right, r)) return false;
        switch (b->op) {
            case T_PLUS: out = l + r; return true;
            case T_MINUS: out = l - r; return true;
            case T_MUL: out = l * r; return true;
            case T_DIV: out = l / r; return true;
            default: return false;
        }
    }

    // ------------------------------------------------------------------
    // Expressions. X emits the statements an expression needs before it
    // and returns its value as text; Into routes a value to a destination.

    // `x` of type `from` where type `to` is wanted: scalars convert where
    // the checker adapted a value it did not cast.
    string Convert(const string &x, TypeExpr *from, TypeExpr *to, Line l) {
        if (!from || !to || !IsScalar(from) || !IsScalar(to)) return x;
        auto a = ScalarName(from, l), b = ScalarName(to, l);
        return a == b ? x : cat("((", b, ")(", x, "))");
    }

    string X(Node *n) {
        switch (n->nkind) {
            case NK_IntLit: {
                auto i = (IntLit *)n;
                return IntText(i->val, i->exprtype, n->line);
            }
            case NK_FltLit: return FltText(((FltLit *)n)->val, LitFloatType(n), n->line);
            case NK_BoolLit: return ((BoolLit *)n)->val ? "true" : "false";
            case NK_Ident: return IdentX((Ident *)n);
            case NK_Unary: return UnaryX((Unary *)n);
            case NK_Binary: return BinaryX((Binary *)n);
            case NK_Dot: return DotX((Dot *)n);
            case NK_Call: return CallX((Call *)n);
            case NK_StructLit: return StructLitX((StructLit *)n);
            case NK_AsCast: {
                auto c = (AsCast *)n;
                auto x = X(c->child);
                auto tt = c->totype ? c->totype : c->exprtype;
                return cat("((", TypeName(tt, n->line), ")(", x, "))");
            }
            case NK_IfExpr: case NK_Block: {
                if (n->exprtype->kind == TY_VOID) {
                    Stmt(n);
                    return "";
                }
                auto t = Temp();
                L(cat(TypeName(n->exprtype, n->line), " ", t, ";"));
                Into(n, t);
                return t;
            }
            case NK_LoopExpr: {
                auto t = Temp();
                L(cat(TypeName(n->exprtype, n->line), " ", t, ";"));
                Into(n, t);
                return t;
            }
            case NK_Index:
                Fail(n, "arrays do not run on the GPU yet (docs/design/gpu.md §5.2)");
            case NK_StrLit: Fail(n, "strings do not run on the GPU");
            case NK_NullLit: Fail(n, "optionals do not run on the GPU");
            case NK_ArrayLit: Fail(n, "arrays do not run on the GPU yet");
            case NK_SliceExpr: Fail(n, "slices do not run on the GPU");
            case NK_MatchExpr: Fail(n, "match does not run on the GPU yet");
            case NK_EarlyBlock: Fail(n, "`block` does not run on the GPU yet");
            case NK_FunVal: Fail(n, "function values do not run on the GPU yet");
            default: Fail(n, "this construct does not run on the GPU (docs/design/gpu.md §5.2)");
        }
    }

    string IdentX(Ident *id) {
        auto v = id->vdef;
        if (!v) Fail(id, cat("function values do not run on the GPU yet: ", id->name));
        if (v->isglobal) {
            // A named constant reads as its value (§5.2): no other global
            // exists on the GPU.
            if (v->constlit) {
                auto t = id->exprtype ? id->exprtype : v->type;
                return IntText(v->constval, t, id->line);
            }
            if (v->constflt) return FltText(v->constfval, id->exprtype ? id->exprtype : v->type,
                                            id->line);
            // A let or const global initialized with literals reads as its
            // initializer, built where it is used.
            if (auto init = PureGlobalInit(v))
                return Convert(X(init), init->exprtype, id->exprtype, id->line);
            Fail(id, cat("global ", v->name, " is not a constant, and the GPU reads no other "
                         "global (docs/design/gpu.md §5.2)"));
        }
        auto it = fs->locals.find(v);
        if (it == fs->locals.end()) Fail(id, cat("internal: unbound variable ", v->name));
        return Convert(it->second, v->type, id->exprtype, id->line);
    }

    // The initializer of a global no one assigns, when it is built from
    // literals alone (and other such globals): the GPU's constants.
    Node *PureGlobalInit(VarDef *v) {
        if (v->isvar) return nullptr;
        for (auto g : ast.globals)
            for (size_t i = 0; i < g->defs.size() && i < g->inits.size(); i++)
                if (g->defs[i] == v && !g->isvar) return PureNode(g->inits[i]) ? g->inits[i] : nullptr;
        return nullptr;
    }

    bool PureNode(Node *n) {
        if (Is<IntLit>(n) || Is<FltLit>(n) || Is<BoolLit>(n)) return true;
        if (auto u = Is<Unary>(n)) return u->op == T_MINUS && PureNode(u->child);
        if (auto c = Is<AsCast>(n)) return PureNode(c->child);
        if (auto id = Is<Ident>(n))
            return id->vdef && id->vdef->isglobal &&
                   (id->vdef->constlit || id->vdef->constflt || PureGlobalInit(id->vdef));
        if (auto sl = Is<StructLit>(n)) {
            if (sl->einst || !sl->sinst) return false;
            for (size_t i = 0; i < sl->sinst->st->fields.size(); i++) {
                if (sl->sinst->st->fields[i].ispad) continue;
                auto init = sl->InitFor((int)i);
                if (!init || !PureNode(init)) return false;
            }
            return true;
        }
        return false;
    }

    string UnaryX(Unary *u) {
        double folded;
        if (u->litfloat && FoldLitFloat(u, folded)) return FltText(folded, LitFloatType(u), u->line);
        if (u->op == T_BITAND) {
            if (u->exprtype && u->exprtype->kind != TY_REF) return X(u->child);
            Fail(u, "references do not run on the GPU");
        }
        auto x = X(u->child);
        switch (u->op) {
            case T_MINUS: return cat("(-", x, ")");
            case T_NOT: return cat("(!", x, ")");
            case T_BITNOT: return cat("(~", x, ")");
            default: Fail(u, "internal: unary operator");
        }
    }

    // `%` is Euclidean on signed integers (spec §6.2): a helper per type.
    set<string> helpers;
    string helperdefs;

    string EuclidMod(const string &a, const string &b, const string &t) {
        auto name = cat("gs_emod_", t);
        if (!helpers.count(name)) {
            helpers.insert(name);
            helperdefs += cat(t, " ", name, "(", t, " a, ", t, " b) {\n",
                              "    ", t, " r = a % b;\n",
                              "    return r < 0 ? r + abs(b) : r;\n}\n\n");
        }
        return cat(name, "(", a, ", ", b, ")");
    }

    // A shift count, any integer type in Goose (spec §6.1): masked to the
    // 32-bit width the GPU shifts at, a literal folded.
    string ShiftCount(Node *n) {
        if (auto i = Is<IntLit>(n)) return cat(i->val & 31);
        auto t = n->exprtype;
        if (t->kind == TY_INT && (t->intstorage == IS_I32 || t->intstorage == IS_U32))
            return cat("(", X(n), " & 31)");
        Fail(n, cat("a shift count on the GPU is an i32, a u32 or a literal, not ", TypeStr(t)));
    }

    string BinaryX(Binary *b) {
        double folded;
        if (b->litfloat && FoldLitFloat(b, folded)) return FltText(folded, LitFloatType(b), b->line);
        if (b->op == T_DOTEQ || b->op == T_DOTNEQ) Fail(b, "references do not run on the GPU");
        auto lt = b->left->exprtype, rt = b->right->exprtype;
        auto l = X(b->left);
        auto shift = b->op == T_SHL || b->op == T_SHR;
        auto r = shift ? ShiftCount(b->right) : X(b->right);
        auto lw = VecWidth(lt), rw = VecWidth(rt);
        if ((lt->kind == TY_STRUCT && !lw) || (rt->kind == TY_STRUCT && !rw))
            Fail(b, "elementwise arithmetic runs on the GPU on vectors only, not other structs");
        auto vec = lw || rw;
        switch (b->op) {
            case T_PLUS: case T_MINUS: case T_MUL: case T_DIV: {
                static const char *ops[] = { "+", "-", "*", "/" };
                auto k = b->op == T_PLUS ? 0 : b->op == T_MINUS ? 1 : b->op == T_MUL ? 2 : 3;
                return cat("(", l, " ", ops[k], " ", r, ")");
            }
            case T_MOD: {
                auto et = lw ? VecElem(lt) : lt;
                if (et->kind == TY_FLT) return cat("fmod(", l, ", ", r, ")");
                if (vec) Fail(b, "integer `%` on vectors does not run on the GPU yet");
                if (et->kind == TY_INT && et->intstorage == IS_I32) return EuclidMod(l, r, "int");
                return cat("(", l, " % ", r, ")");
            }
            case T_EQ: case T_NEQ: {
                auto op = b->op == T_EQ ? "==" : "!=";
                if (vec) return cat(b->op == T_EQ ? "all(" : "any(", l, " ", op, " ", r, ")");
                return cat("(", l, " ", op, " ", r, ")");
            }
            case T_LT: case T_GT: case T_LTEQ: case T_GTEQ: {
                if (vec) Fail(b, "vectors are not ordered");
                auto op = b->op == T_LT ? "<" : b->op == T_GT ? ">" : b->op == T_LTEQ ? "<=" : ">=";
                return cat("(", l, " ", op, " ", r, ")");
            }
            case T_ANDAND: return cat("(", l, " && ", r, ")");
            case T_OROR: return cat("(", l, " || ", r, ")");
            case T_BITAND: return cat("(", l, " & ", r, ")");
            case T_BITOR: return cat("(", l, " | ", r, ")");
            case T_XOR: return cat("(", l, " ^ ", r, ")");
            case T_SHL: case T_SHR: {
                // Shift counts are masked to the width (spec §6.2), which
                // MSL and SPIR-V leave undefined beyond it.
                if (vec) Fail(b, "shifts on vectors do not run on the GPU yet");
                return cat("(", l, b->op == T_SHL ? " << " : " >> ", r, ")");
            }
            default: break;
        }
        Fail(b, cat("operator ", TName(b->op), " does not run on the GPU"));
    }

    string DotX(Dot *d) {
        if (d->variantconst) Fail(d, "enums do not run on the GPU yet");
        if (d->member >= 0) Fail(d, cat(".", d->name, " does not run on the GPU"));
        auto ot = d->obj->exprtype;
        if (ot->kind == TY_REF) ot = ot->ref->sub;
        auto o = X(d->obj);
        if (VecWidth(ot)) return cat(o, ".", d->name);
        if (ot->kind != TY_STRUCT) Fail(d, "internal: field of a non-struct");
        auto x = cat(o, ".", FieldName(d->name));
        // A struct's vector fields are packed in MSL (TypeName).
        if (msl() && VecWidth(d->exprtype)) return cat(TypeName(d->exprtype, d->line), "(", x, ")");
        return x;
    }

    // An assignment's target: fields as they are stored, where reading one
    // would unpack MSL's packed vectors (DotX).
    string LValue(Node *n) {
        if (auto d = Is<Dot>(n); d && d->IsField()) {
            auto ot = d->obj->exprtype;
            if (ot->kind == TY_REF) ot = ot->ref->sub;
            return cat(LValue(d->obj), ".", VecWidth(ot) ? string(d->name) : FieldName(d->name));
        }
        return X(n);
    }

    string StructLitX(StructLit *s) {
        if (s->einst || s->variant) Fail(s, "enums do not run on the GPU yet");
        auto t = s->exprtype;
        auto tn = TypeName(t, s->line);
        auto si = Inst(t, s->line);
        vector<string> vals;
        for (size_t i = 0; i < si->st->fields.size(); i++) {
            if (si->st->fields[i].ispad) continue;
            auto init = s->InitFor((int)i);
            if (!init) Fail(s, "internal: a struct literal without a field's value");
            vals.push_back(Convert(X(init), init->exprtype, si->ftypes[i], init->line));
        }
        if (VecWidth(t)) {
            string r = cat(tn, "(");
            for (size_t i = 0; i < vals.size(); i++) r += cat(i ? ", " : "", vals[i]);
            return r + ")";
        }
        // HLSL has no struct expression, so the value is built in a local.
        auto tv = Temp();
        L(cat(tn, " ", tv, ";"));
        size_t k = 0;
        int pads = 0;
        for (size_t i = 0; i < si->st->fields.size(); i++) {
            auto &f = si->st->fields[i];
            if (f.ispad) {
                auto n = f.padsize >= 0 ? f.padsize : 0;
                for (int64_t w = 0; w < n / 4; w++) L(cat(tv, ".gs_pad", pads, "[", w, "] = 0u;"));
                if (n) pads++;
                continue;
            }
            L(cat(tv, ".", FieldName(f.name), " = ", vals[k++], ";"));
        }
        return tv;
    }

    // The default value of a type, which default<T>() and an omitted field
    // build.
    string ZeroOf(TypeExpr *t, Line l) {
        auto tn = TypeName(t, l);
        return hlsl() ? cat("((", tn, ")0)") : cat(tn, "{}");
    }

    // Element `i` of the buffer at heap index `index`, as the dialect
    // reaches it: in MSL the element itself, an lvalue; in HLSL the buffer
    // and the element's byte offset, for Load and Store.
    pair<string, string> BufferElem(const string &index, const string &i, TypeExpr *et, Line l,
                                    bool write) {
        auto stride = PackedSize(et, l);
        if (msl())
            return { cat("reinterpret_cast<device ", TypeName(et, l, true), " *>(gs.heap[",
                         index, "])[", i, "]"), "" };
        // A load through the read-only type, which SPIR-V marks NonWritable,
        // as a vertex stage's storage must be.
        auto buf = Temp();
        L(cat(write ? "RWByteAddressBuffer " : "ByteAddressBuffer ", buf,
              " = ResourceDescriptorHeap[", index, "];"));
        return { buf, cat("(", i, ") * ", stride, "u") };
    }

    string CallX(Call *c) {
        if (c->fvbody || c->trailing) Fail(c, "function values do not run on the GPU yet");
        if (!c->dispatch.empty()) Fail(c, "case functions do not run on the GPU yet");
        if (c->rettypes.size() > 1)
            Fail(c, "a call returning several values does not run on the GPU yet");
        auto an = c->ArgNodes();
        if (c->builtin >= 0) {
            switch (c->builtin) {
                case B_GPU_LOAD: {
                    auto et = c->rettypes[0];
                    auto index = X(an[0]);
                    auto i = X(an[1]);
                    auto [e, off] = BufferElem(index, i, et, c->line, false);
                    if (msl()) return VecWidth(et) ? cat(TypeName(et, c->line), "(", e, ")") : e;
                    return cat(e, ".Load<", TypeName(et, c->line), ">(", off, ")");
                }
                case B_GPU_STORE: {
                    auto et = an[2]->exprtype;
                    auto index = X(an[0]);
                    auto i = X(an[1]);
                    auto v = X(an[2]);
                    auto [e, off] = BufferElem(index, i, et, c->line, true);
                    if (msl()) L(cat(e, " = ", v, ";"));
                    else L(cat(e, ".Store<", TypeName(et, c->line), ">(", off, ", ", v, ");"));
                    return "";
                }
                case B_GPU_SAMPLE: case B_GPU_SAMPLE_LEVEL: case B_GPU_TEXEL:
                case B_GPU_STORE_TEXEL:
                    return TextureX(c, an);
                case B_DEFAULT: return ZeroOf(c->exprtype, c->line);
                default:
                    Fail(c, cat(BuiltinByKind(c->builtin).name, " does not run on the GPU"));
            }
        }
        auto sp = c->spec;
        if (!sp) Fail(c, "internal: an unresolved call");
        if (sp->sf->isextern) return ExternX(c, sp, an);
        vector<string> args;
        if (msl()) args.push_back("gs");
        for (size_t i = 0; i < an.size(); i++) {
            auto pt = i < sp->argtypes.size() ? sp->argtypes[i] : nullptr;
            args.push_back(Convert(X(an[i]), an[i]->exprtype, pt, an[i]->line));
        }
        auto name = EnsureFn(sp, c->line);
        string r = cat(name, "(");
        for (size_t i = 0; i < args.size(); i++) r += cat(i ? ", " : "", args[i]);
        r += ")";
        if (sp->rets.empty()) {
            L(r + ";");
            return "";
        }
        return r;
    }

    // An extern on the GPU: a system value (stdlib/gpu_shader.goose), or
    // one of math's functions, which both dialects have by its C name
    // without the runtime's prefix and the f32 suffix.
    string ExternX(Call *c, FnSpec *sp, const vector<Node *> &an) {
        auto cn = string_view(sp->sf->cname);
        if (cn.starts_with("gs_shader_")) {
            auto what = cn.substr(10);
            struct Sys { const char *name; int stage; const char *var; };
            static const Sys sys[] = {
                { "dispatch_thread_id", GPU_COMPUTE, "dispatch_thread_id" },
                { "group_id", GPU_COMPUTE, "group_id" },
                { "group_thread_id", GPU_COMPUTE, "group_thread_id" },
                { "group_index", GPU_COMPUTE, "group_index" },
                { "vertex_index", GPU_VERTEX, "vertex_index" },
                { "instance_index", GPU_VERTEX, "instance_index" },
                { "frag_coord", GPU_FRAGMENT, "frag_coord" },
                { "front_facing", GPU_FRAGMENT, "front_facing" },
            };
            if (what == "trace" || what == "trace_any") return TraceX(c, sp, an, what == "trace");
            if (what == "discard") {
                RequireStage(c, GPU_FRAGMENT, "discard()");
                L(hlsl() ? "discard;" : "discard_fragment();");
                return "";
            }
            if (what == "ddx" || what == "ddy" || what == "fwidth") {
                RequireStage(c, GPU_FRAGMENT, cat(what, "()"));
                auto fn = msl() && what != "fwidth" ? (what == "ddx" ? "dfdx" : "dfdy")
                                                    : string(what);
                return cat(fn, "(", X(an[0]), ")");
            }
            for (auto &s : sys)
                if (what == s.name) {
                    if (entry->sf->gpustage != s.stage)
                        Fail(c, cat(sp->sf->name, "() exists in a ", GpuStageName(s.stage),
                                    " entry point only, and this is reached from ",
                                    GpuStageName(entry->sf->gpustage), " fn ", entry->sf->name));
                    return msl() ? cat("gs.", s.var) : cat("gs_", s.var);
                }
            Fail(c, cat("internal: unknown GPU intrinsic ", cn));
        }
        // libm under the runtime's names (stdlib/math.goose).
        static const unordered_set<string_view> math = {
            "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log", "log2",
            "log10", "pow", "floor", "ceil", "trunc",
        };
        if (cn.starts_with("gs_") && cn.ends_with("f")) {
            auto base = cn.substr(3, cn.size() - 4);
            if (math.count(base)) {
                string r = cat(base, "(");
                for (size_t i = 0; i < an.size(); i++) r += cat(i ? ", " : "", X(an[i]));
                return r + ")";
            }
        }
        Fail(c, cat("extern fn ", sp->sf->name, " runs on the CPU only"));
    }

    // ------------------------------------------------------------------
    // Statements.

    void Into(Node *n, const string &dst) {
        switch (n->nkind) {
            case NK_IfExpr: {
                auto i = (IfExpr *)n;
                auto c = X(i->cond);
                L(cat("if (", c, ") {"));
                fs->ind++;
                Into(i->thenb, dst);
                fs->ind--;
                if (i->elseb) {
                    L("} else {");
                    fs->ind++;
                    Into(i->elseb, dst);
                    fs->ind--;
                }
                L("}");
                return;
            }
            case NK_Block: {
                auto b = (Block *)n;
                for (auto s : b->stmts) Stmt(s);
                if (b->tail) Into(b->tail, dst);
                return;
            }
            case NK_LoopExpr: {
                auto lp = (LoopExpr *)n;
                fs->breakdsts.push_back(dst);
                L("while (true) {");
                fs->ind++;
                Stmt(lp->body);
                fs->ind--;
                L("}");
                fs->breakdsts.pop_back();
                return;
            }
            default:
                if (Is<Return>(n) || Is<Break>(n) || Is<Continue>(n)) {
                    Stmt(n);
                    return;
                }
                L(cat(dst, " = ", X(n), ";"));
        }
    }

    void Stmt(Node *n) {
        if (!n) return;
        if (!Is<Block>(n)) LineDirective(n->line);
        switch (n->nkind) {
            case NK_Block: {
                auto b = (Block *)n;
                for (auto s : b->stmts) Stmt(s);
                if (b->tail) Stmt(b->tail);
                return;
            }
            case NK_VarDecl: {
                auto vd = (VarDecl *)n;
                if (vd->defs.size() != 1)
                    Fail(n, "a declaration of several values does not run on the GPU yet");
                auto v = vd->defs[0];
                auto tn = TypeName(v->type, n->line);
                string init;
                if (!vd->inits.empty())
                    init = Convert(X(vd->inits[0]), vd->inits[0]->exprtype, v->type, n->line);
                auto name = LocalName(v);
                if (init.empty()) L(cat(tn, " ", name, ";"));
                else L(cat(vd->isvar ? "" : "const ", tn, " ", name, " = ", init, ";"));
                return;
            }
            case NK_Assign: {
                auto a = (Assign *)n;
                if (a->pointee) Fail(n, "references do not run on the GPU");
                auto lv = LValue(a->lval);
                if (a->op == T_SHLEQ || a->op == T_SHREQ) {
                    L(cat(lv, a->op == T_SHLEQ ? " <<= " : " >>= ", ShiftCount(a->rhs), ";"));
                    return;
                }
                auto rhs = Convert(X(a->rhs), a->rhs->exprtype, a->lval->exprtype, n->line);
                if (a->op == T_ASSIGN) {
                    L(cat(lv, " = ", rhs, ";"));
                    return;
                }
                auto lt = a->lval->exprtype;
                if (a->op == T_MODEQ && lt->kind == TY_INT && lt->intstorage == IS_I32) {
                    L(cat(lv, " = ", EuclidMod(lv, rhs, "int"), ";"));
                    return;
                }
                L(cat(lv, " ", TName(a->op), " ", rhs, ";"));
                return;
            }
            case NK_IncDec: {
                auto i = (IncDec *)n;
                L(cat(LValue(i->lval), i->op == T_INC ? "++;" : "--;"));
                return;
            }
            case NK_IfExpr: {
                auto i = (IfExpr *)n;
                auto c = X(i->cond);
                L(cat("if (", c, ") {"));
                fs->ind++;
                Stmt(i->thenb);
                fs->ind--;
                if (i->elseb) {
                    L("} else {");
                    fs->ind++;
                    Stmt(i->elseb);
                    fs->ind--;
                }
                L("}");
                return;
            }
            case NK_While: {
                auto w = (While *)n;
                // A condition that needs statements of its own is tested at
                // the top of an endless loop.
                auto saved = fs->out;
                fs->out.clear();
                auto c = X(w->cond);
                auto pre = fs->out;
                fs->out = saved;
                fs->breakdsts.push_back("");
                if (pre.empty()) {
                    L(cat("while (", c, ") {"));
                    fs->ind++;
                } else {
                    L("while (true) {");
                    fs->ind++;
                    fs->out += pre;
                    L(cat("if (!(", c, ")) break;"));
                }
                Stmt(w->body);
                fs->ind--;
                L("}");
                fs->breakdsts.pop_back();
                return;
            }
            case NK_LoopExpr: {
                auto lp = (LoopExpr *)n;
                fs->breakdsts.push_back("");
                L("while (true) {");
                fs->ind++;
                Stmt(lp->body);
                fs->ind--;
                L("}");
                fs->breakdsts.pop_back();
                return;
            }
            case NK_ForLoop: {
                auto f = (ForLoop *)n;
                if (f->iterkind != IK_RANGE && f->iterkind != IK_COUNT)
                    Fail(n, "a for over an array does not run on the GPU yet");
                if (f->idxdef) Fail(n, "a for with an index binding does not run on the GPU yet");
                auto r = Is<RangeExpr>(f->iter);
                auto ct = r ? r->exprtype : f->iter->exprtype;
                auto ctn = TypeName(ct, n->line);
                string lo = r ? X(r->lo) : IntText(0, ct, n->line);
                auto hi = Temp();
                L(cat("const ", ctn, " ", hi, " = ", X(r ? r->hi : f->iter), ";"));
                auto iv = LocalName(f->vdef);
                auto wide = TypeName(f->vdef->type, n->line) != ctn;
                auto ctr = wide ? Temp() : iv;
                fs->breakdsts.push_back("");
                L(cat("for (", ctn, " ", ctr, " = ", lo, "; ", ctr, " < ", hi, "; ", ctr, "++) {"));
                fs->ind++;
                if (wide)
                    L(cat(TypeName(f->vdef->type, n->line), " ", iv, " = (",
                          TypeName(f->vdef->type, n->line), ")", ctr, ";"));
                Stmt(f->body);
                fs->ind--;
                L("}");
                fs->breakdsts.pop_back();
                return;
            }
            case NK_Return: {
                auto r = (Return *)n;
                if (fs->spec == entry && entry->sf->gpustage != GPU_COMPUTE && !r->vals.empty()) {
                    EntryReturn(r->vals, n->line);
                    return;
                }
                if (r->target && r->target != fs->spec->sf)
                    Fail(n, "`return ... from` does not run on the GPU");
                if (r->vals.empty()) {
                    L("return;");
                    return;
                }
                if (r->vals.size() > 1)
                    Fail(n, "returning several values does not run on the GPU yet");
                auto rt = fs->spec->rets.empty() ? nullptr : fs->spec->rets[0];
                L(cat("return ", Convert(X(r->vals[0]), r->vals[0]->exprtype, rt, n->line), ";"));
                return;
            }
            case NK_Break: {
                auto b = (Break *)n;
                if (b->val) {
                    if (fs->breakdsts.empty() || fs->breakdsts.back().empty())
                        Fail(n, "internal: a break with a value outside a value loop");
                    L(cat(fs->breakdsts.back(), " = ", X(b->val), ";"));
                }
                L("break;");
                return;
            }
            case NK_Continue: L("continue;"); return;
            case NK_FnDecl: Fail(n, "nested functions do not run on the GPU yet");
            case NK_Call: {
                auto x = X(n);
                if (!x.empty()) L(cat(x, ";"));
                return;
            }
            default: {
                auto x = X(n);
                if (!x.empty()) L(cat("(void)(", x, ");"));
                return;
            }
        }
    }

    // ------------------------------------------------------------------
    // Functions.

    string EnsureFn(FnSpec *sp, Line callsite) {
        if (auto it = fnnames.find(sp); it != fnnames.end()) return it->second;
        if (inprogress.count(sp))
            Fail(callsite, cat("recursion does not run on the GPU: ", sp->sf->name,
                               " calls itself"));
        if (!sp->body) Fail(callsite, cat("internal: no body for ", sp->sf->name));
        auto base = string(sp->sf->qname);
        for (auto &c : base) if (c == ':') c = '_';
        while (base.find("__") != string::npos) base.replace(base.find("__"), 2, "_");
        auto name = FreshGlobal(base);
        inprogress.insert(sp);
        chain.push_back(sp);
        callsites.push_back(callsite);
        FnState st;
        st.spec = sp;
        auto outer = fs;
        fs = &st;
        string params;
        if (msl()) params = "thread GsCtx &gs";
        for (size_t i = 0; i < sp->params.size(); i++) {
            auto v = sp->params[i];
            if (!params.empty()) params += ", ";
            params += cat(TypeName(v->type, v->line), " ", LocalName(v));
        }
        if (sp->rets.size() > 1)
            Fail(sp->sf->line, cat(sp->sf->name, " returns several values, which does not run "
                                   "on the GPU yet"));
        auto ret = sp->rets.empty() ? string("void") : TypeName(sp->rets[0], sp->sf->line);
        Body(sp);
        fs = outer;
        chain.pop_back();
        callsites.pop_back();
        inprogress.erase(sp);
        fnnames[sp] = name;
        fndefs += cat(ret, " ", name, "(", params, ") {\n", st.out, "}\n\n");
        return name;
    }

    void Body(FnSpec *sp) {
        auto b = sp->body;
        for (auto s : b->stmts) Stmt(s);
        if (!b->tail) return;
        if (sp->rets.empty() || b->tail->exprtype->kind == TY_VOID) {
            Stmt(b->tail);
            return;
        }
        if (Is<Return>(b->tail)) {
            Stmt(b->tail);
            return;
        }
        LineDirective(b->tail->line);
        if (sp == entry) {
            EntryReturn({ b->tail }, b->tail->line);
            return;
        }
        auto x = X(b->tail);
        L(cat("return ", Convert(x, b->tail->exprtype, sp->rets[0], b->tail->line), ";"));
    }

    // ------------------------------------------------------------------
    // The entry point, and the program around it.

    // An entry's parameters (§5.1): a fragment entry's stage input first,
    // then the push constants, a value, then the uniform block, a const T&.
    VarDef *stagein = nullptr, *push = nullptr, *uniform = nullptr;

    void ClassifyParams() {
        size_t i = 0;
        auto &ps = entry->params;
        if (entry->sf->gpustage == GPU_FRAGMENT) {
            if (ps.empty() || ps[0]->type->kind == TY_REF)
                Fail(entry->sf->line, cat("fragment fn ", entry->sf->name, " takes its vertex "
                                          "stage's varyings first, an empty struct for none"));
            stagein = ps[i++];
        }
        if (i < ps.size() && ps[i]->type->kind != TY_REF) push = ps[i++];
        if (i < ps.size() && ps[i]->type->kind == TY_REF) uniform = ps[i++];
        if (i < ps.size())
            Fail(ps[i]->line, cat("an entry point takes ",
                                  entry->sf->gpustage == GPU_FRAGMENT ? "its stage input, " : "",
                                  "push constants and a uniform block, in that order, each at "
                                  "most once: ", ps[i]->name, " is another"));
        if (push && (push->type->kind != TY_STRUCT || VecWidth(push->type)))
            Fail(push->line, cat("an entry point's push constants are a struct, not ",
                                 TypeStr(push->type)));
        if (uniform) {
            auto ut = uniform->type->ref->sub;
            if (ut->kind != TY_STRUCT || VecWidth(ut))
                Fail(uniform->line, cat("an entry point's uniform block is a struct, not ",
                                        TypeStr(ut)));
        }
    }

    // A struct's leaves, the scalars it holds at their Goose offsets: what a
    // constant buffer is checked against and read from (§5.3).
    struct Leaf { string path; TypeExpr *type; int64_t off; };
    void Leaves(TypeExpr *t, const string &path, int64_t off, vector<Leaf> &out, Line l) {
        if (IsScalar(t)) {
            out.push_back({ path, t, off });
            return;
        }
        if (auto w = VecWidth(t)) {
            static const char *comp[] = { ".x", ".y", ".z", ".w" };
            for (int k = 0; k < w; k++) out.push_back({ path + comp[k], VecElem(t), off + 4 * k });
            return;
        }
        auto si = Inst(t, l);
        for (size_t i = 0; i < si->st->fields.size(); i++) {
            auto &f = si->st->fields[i];
            if (f.ispad) {
                off += f.padsize >= 0 ? f.padsize : (4 - off % 4) % 4;
                continue;
            }
            Leaves(si->ftypes[i], cat(path, ".", FieldName(f.name)), off, out, f.line);
            off += PackedSize(si->ftypes[i], f.line);
        }
    }

    // Whether Direct3D 12's constant-buffer packing puts every field of `t`
    // where Goose does: 16-byte registers that a vector may not straddle,
    // a nested struct starting one.
    bool CbufferAgrees(TypeExpr *t, int64_t &cboff, int64_t gooseoff, Line l) {
        if (IsScalar(t)) {
            if (cboff != gooseoff) return false;
            cboff += 4;
            return true;
        }
        if (auto w = VecWidth(t)) {
            auto size = 4 * w;
            if (cboff / 16 != (cboff + size - 1) / 16) cboff = (cboff + 15) / 16 * 16;
            if (cboff != gooseoff) return false;
            cboff += size;
            return true;
        }
        auto si = Inst(t, l);
        cboff = (cboff + 15) / 16 * 16;
        for (size_t i = 0; i < si->st->fields.size(); i++) {
            auto &f = si->st->fields[i];
            if (f.ispad) {
                auto n = f.padsize >= 0 ? f.padsize : (4 - gooseoff % 4) % 4;
                gooseoff += n;
                cboff += n;
                continue;
            }
            if (!CbufferAgrees(si->ftypes[i], cboff, gooseoff, f.line)) return false;
            gooseoff += PackedSize(si->ftypes[i], f.line);
        }
        return true;
    }

    // A block of push constants (k 0) or the uniform buffer (k 1): its
    // declaration, and the statement reading it into the entry's local.
    int64_t pushsize = 0, uniformsize = 0;
    pair<string, string> BlockDecl(VarDef *v, int k) {
        auto pt = k ? v->type->ref->sub : v->type;
        auto tn = TypeName(pt, v->line, true);
        auto size = PackedSize(pt, v->line);
        auto limit = k ? 65536 : 128;
        if (size > limit)
            Fail(v->line, cat(k ? "a uniform block is at most 64 KiB" : "push constants are at "
                              "most 128 bytes", ", and ", TypeStr(pt), " is ", size));
        (k ? uniformsize : pushsize) = size;
        auto name = LocalName(v);
        auto var = k ? "gs_uniforms" : "gs_push";
        int64_t cboff = 0;
        if (msl() || dia == GD_HLSL_SPIRV || CbufferAgrees(pt, cboff, 0, v->line)) {
            string decl;
            if (dia == GD_HLSL_SPIRV)
                decl = k ? cat("[[vk::binding(0, 1)]] ConstantBuffer<", tn, "> ", var, ";\n")
                         : cat("[[vk::push_constant]] ConstantBuffer<", tn, "> ", var, ";\n");
            else if (hlsl())
                decl = cat("ConstantBuffer<", tn, "> ", var, " : register(b", k, ");\n");
            return { decl, cat(tn, " ", name, " = ", var, ";") };
        }
        // Flattened (§5.3): the registers as words, each field read from its
        // Goose offset.
        auto words = (size + 15) / 16;
        string decl = cat("cbuffer ", var, "_words : register(b", k, ") { uint4 ", var, "[", words,
                          "]; };\n\n");
        vector<Leaf> leaves;
        Leaves(pt, "", 0, leaves, v->line);
        decl += cat(tn, " gs_load", var + 2, "() {\n    ", tn, " r;\n");
        static const char *comp[] = { ".x", ".y", ".z", ".w" };
        for (auto &lf : leaves) {
            auto w = cat(var, "[", lf.off / 16, "]", comp[(lf.off % 16) / 4]);
            auto sn = ScalarName(lf.type, v->line);
            auto val = sn == "float" ? cat("asfloat(", w, ")") : sn == "int" ? cat("asint(", w, ")")
                                                                            : w;
            decl += cat("    r", lf.path, " = ", val, ";\n");
        }
        decl += "    return r;\n}\n";
        return { decl, cat(tn, " ", name, " = gs_load", var + 2, "();") };
    }

    // The fields a vertex entry passes its fragment entry, from its
    // varyings struct: scalars and vectors only, flat when integers.
    struct Varying { string name; TypeExpr *type; bool flat; };
    vector<Varying> Varyings(TypeExpr *t, Line l) {
        vector<Varying> out;
        auto si = Inst(t, l);
        for (size_t i = 0; i < si->st->fields.size(); i++) {
            auto &f = si->st->fields[i];
            if (f.ispad) continue;
            auto ft = si->ftypes[i];
            auto et = VecWidth(ft) ? VecElem(ft) : ft;
            if (!IsScalar(et) || et->kind == TY_BOOL)
                Fail(f.line, cat("a varying is a number or a vector of them, not ", TypeStr(ft),
                                 ": ", f.name));
            out.push_back({ FieldName(f.name), ft, et->kind == TY_INT });
        }
        return out;
    }

    // `obj.f` of a struct value, as a value: MSL's packed vector fields
    // unpacked (TypeName).
    string FieldRead(const string &obj, const string &f, TypeExpr *ft, Line l) {
        auto x = cat(obj, ".", f);
        return msl() && VecWidth(ft) ? cat(TypeName(ft, l), "(", x, ")") : x;
    }

    // A vertex or fragment entry's return, its outputs: the clip position
    // and varyings, or the colors.
    void EntryReturn(const vector<Node *> &vals, Line l) {
        auto stage = entry->sf->gpustage;
        if (stage == GPU_VERTEX) {
            auto clip = X(vals[0]);
            L("gs_vertex_out gs_out;");
            L(cat("gs_out.position = ", clip, ";"));
            if (vals.size() > 1) {
                auto v = X(vals[1]);
                if (!Is<Ident>(vals[1])) {
                    auto t = Temp();
                    L(cat(TypeName(vals[1]->exprtype, l), " ", t, " = ", v, ";"));
                    v = t;
                }
                for (auto &vy : Varyings(vals[1]->exprtype, l))
                    L(cat("gs_out.", vy.name, " = ", FieldRead(v, vy.name, vy.type, l), ";"));
            }
            L("return gs_out;");
            return;
        }
        if (vals.size() == 1) {
            L(cat("return ", Convert(X(vals[0]), vals[0]->exprtype, entry->rets[0], l), ";"));
            return;
        }
        L("gs_fragment_out gs_out;");
        for (size_t k = 0; k < vals.size(); k++)
            L(cat("gs_out.color", k, " = ", X(vals[k]), ";"));
        L("return gs_out;");
    }

    // A texture's element type T: the scalar its texels hold, and how many
    // of the four channels T takes.
    pair<string, int> TexElem(TypeExpr *t, Line l) {
        auto w = VecWidth(t);
        auto et = w ? VecElem(t) : t;
        if (!IsScalar(et) || et->kind == TY_BOOL)
            Fail(l, cat("a texture holds numbers or vectors of them, not ", TypeStr(t)));
        return { ScalarName(et, l), w ? w : 1 };
    }

    static string Narrow(const string &x, int w) {
        static const char *sw[] = { "", ".x", ".xy", ".xyz", "" };
        return w == 4 ? x : cat("(", x, ")", sw[w]);
    }

    static string Widen(const string &x, const string &scalar, int w) {
        static const char *pad[] = { "", ", 0, 0, 0", ", 0, 0", ", 0", "" };
        return w == 4 ? x : cat(scalar, "4(", x, pad[w], ")");
    }

    // The argument-buffer structs MSL reaches the heap's textures and
    // samplers through, each declared on first use.
    set<string> mslheaptypes;
    string mslheapdefs;
    string MslHeapType(const string &kind, const string &scalar) {
        auto name = cat("GsHeap", kind, "_", scalar);
        if (mslheaptypes.insert(name).second) {
            auto member = kind == "Sampler" ? string("sampler")
                        : kind == "Accel" ? string("raytracing::instance_acceleration_structure")
                        : kind == "RWTexture2D" ? cat("texture2d<", scalar, ", access::read_write>")
                                                : cat("texture2d<", scalar, ">");
            mslheapdefs += cat("struct ", name, " { ", member, " r; };\n");
        }
        return name;
    }

    // The texture at heap index `index` (`kind` Texture2D or RWTexture2D),
    // as an expression the dialect calls methods on.
    string TextureRef(const string &kind, const string &scalar, const string &index) {
        if (msl())
            return cat("reinterpret_cast<device const ", MslHeapType(kind, scalar),
                       " *>(gs.heap)[", index, "].r");
        auto t = Temp();
        L(cat(kind, "<", scalar, "4> ", t, " = ResourceDescriptorHeap[", index, "];"));
        return t;
    }

    string SamplerRef(const string &index) {
        if (msl())
            return cat("reinterpret_cast<device const ", MslHeapType("Sampler", "s"),
                       " *>(gs.samplers)[", index, "].r");
        auto t = Temp();
        L(cat("SamplerState ", t, " = SamplerDescriptorHeap[", index, "];"));
        return t;
    }

    void RequireStage(Call *c, int stage, const string &what) {
        if (entry->sf->gpustage != stage)
            Fail(c, cat(what, " exists in a ", GpuStageName(stage), " entry point only, and this "
                        "is reached from ", GpuStageName(entry->sf->gpustage), " fn ",
                        entry->sf->name));
    }

    // An inline ray query (§7), through a helper per dialect, defined on
    // first use: the closest opaque triangle, or any.
    int caps = 0;
    bool raytracing = false;
    string TraceX(Call *c, FnSpec *sp, const vector<Node *> &an, bool closest) {
        caps |= GS_GPU_CAP_RAY_QUERY;
        raytracing = true;
        auto ray = TypeName(sp->argtypes[1], c->line);
        auto f = [&](const char *n) { return FieldName(n); };
        auto vec3 = [&](const string &x) { return msl() ? cat("float3(", x, ")") : x; };
        auto name = closest ? "gs_trace" : "gs_trace_any";
        if (helpers.insert(name).second) {
            auto hit = closest ? TypeName(sp->rets[0], c->line) : string("bool");
            string body = cat(hit, " ", name, "(", msl() ? "thread GsCtx &gs, " : "",
                              "uint accel, ", ray, " r, uint mask) {\n");
            if (hlsl()) {
                body += cat("    RaytracingAccelerationStructure scene = ResourceDescriptorHeap[accel];\n"
                            "    RayDesc d;\n    d.Origin = r.", f("origin"), ";\n    d.TMin = r.",
                            f("t_min"), ";\n    d.Direction = r.", f("dir"), ";\n    d.TMax = r.",
                            f("t_max"), ";\n");
                body += closest ? "    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;\n"
                                : "    RayQuery<RAY_FLAG_FORCE_OPAQUE | "
                                  "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;\n";
                body += "    q.TraceRayInline(scene, RAY_FLAG_NONE, mask, d);\n    q.Proceed();\n";
                if (!closest) {
                    body += "    return q.CommittedStatus() != COMMITTED_NOTHING;\n}\n\n";
                } else {
                    body += cat("    ", hit, " h = (", hit, ")0;\n"
                                "    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {\n"
                                "        h.", f("kind"), " = 1u;\n"
                                "        h.", f("t"), " = q.CommittedRayT();\n"
                                "        h.", f("instance"), " = q.CommittedInstanceIndex();\n"
                                "        h.", f("id"), " = q.CommittedInstanceID();\n"
                                "        h.", f("primitive"), " = q.CommittedPrimitiveIndex();\n"
                                "        h.", f("front_face"), " = q.CommittedTriangleFrontFace() ? 1u : 0u;\n"
                                "        h.", f("barycentrics"), " = q.CommittedTriangleBarycentrics();\n"
                                "    }\n    return h;\n}\n\n");
                }
            } else {
                auto heap = MslHeapType("Accel", "scene");
                body += cat("    raytracing::ray rr(", vec3(cat("r.", f("origin"))), ", ",
                            vec3(cat("r.", f("dir"))), ", r.", f("t_min"), ", r.", f("t_max"),
                            ");\n"
                            "    raytracing::intersector<raytracing::triangle_data, "
                            "raytracing::instancing> i;\n"
                            "    i.assume_geometry_type(raytracing::geometry_type::triangle);\n"
                            "    i.force_opacity(raytracing::forced_opacity::opaque);\n");
                if (!closest) body += "    i.accept_any_intersection(true);\n";
                body += cat("    auto res = i.intersect(rr, reinterpret_cast<device const ", heap,
                            " *>(gs.heap)[accel].r, mask);\n");
                if (!closest) {
                    body += "    return res.type != raytracing::intersection_type::none;\n}\n\n";
                } else {
                    body += cat("    ", hit, " h{};\n"
                                "    if (res.type == raytracing::intersection_type::triangle) {\n"
                                "        h.", f("kind"), " = 1u;\n"
                                "        h.", f("t"), " = res.distance;\n"
                                "        h.", f("instance"), " = res.instance_id;\n"
                                "        h.", f("id"), " = res.user_instance_id;\n"
                                "        h.", f("primitive"), " = res.primitive_id;\n"
                                "        h.", f("front_face"), " = res.triangle_front_facing ? 1u : 0u;\n"
                                "        h.", f("barycentrics"), " = res.triangle_barycentric_coord;\n"
                                "    }\n    return h;\n}\n\n");
                }
            }
            helperdefs += body;
        }
        auto accel = X(an[0]);
        auto r = X(an[1]);
        auto mask = X(an[2]);
        return cat(name, "(", msl() ? "gs, " : "", "(", accel, ").index, ", r, ", ", mask, ")");
    }

    // A texture builtin (stdlib/gpu_shader.goose): the coordinate of the
    // right shape, and the texel narrowed to T.
    string TextureX(Call *c, const vector<Node *> &an) {
        auto check = [&](Node *n, int width, bool flt, const char *what) {
            auto t = n->exprtype;
            auto et = VecWidth(t) ? VecElem(t) : t;
            if (VecWidth(t) != width || (flt ? et->kind != TY_FLT : et->kind != TY_INT))
                Fail(n, cat(what, " is a", flt ? " float2" : "n int2 or uint2", ", not ",
                            TypeStr(t)));
        };
        switch (c->builtin) {
            case B_GPU_SAMPLE: case B_GPU_SAMPLE_LEVEL: {
                if (c->builtin == B_GPU_SAMPLE) RequireStage(c, GPU_FRAGMENT, "sample");
                auto [sc, w] = TexElem(c->rettypes[0], c->line);
                check(an[2], 2, true, "a texture coordinate");
                auto tex = TextureRef("Texture2D", sc, X(an[0]));
                auto smp = SamplerRef(X(an[1]));
                auto uv = X(an[2]);
                string r;
                if (c->builtin == B_GPU_SAMPLE)
                    r = hlsl() ? cat(tex, ".Sample(", smp, ", ", uv, ")")
                               : cat(tex, ".sample(", smp, ", ", uv, ")");
                else {
                    auto lod = X(an[3]);
                    r = hlsl() ? cat(tex, ".SampleLevel(", smp, ", ", uv, ", ", lod, ")")
                               : cat(tex, ".sample(", smp, ", ", uv, ", level(", lod, "))");
                }
                return Narrow(r, w);
            }
            case B_GPU_TEXEL: {
                auto [sc, w] = TexElem(c->rettypes[0], c->line);
                check(an[1], 2, false, "a texel coordinate");
                auto tex = TextureRef("Texture2D", sc, X(an[0]));
                auto xy = X(an[1]);
                auto mip = X(an[2]);
                return Narrow(hlsl() ? cat(tex, ".Load(int3(", xy, ", ", mip, "))")
                                     : cat(tex, ".read(uint2(", xy, "), uint(", mip, "))"), w);
            }
            case B_GPU_STORE_TEXEL: {
                auto [sc, w] = TexElem(an[2]->exprtype, c->line);
                check(an[1], 2, false, "a texel coordinate");
                auto tex = TextureRef("RWTexture2D", sc, X(an[0]));
                auto xy = X(an[1]);
                auto v = Widen(X(an[2]), sc, w);
                if (hlsl()) L(cat(tex, "[uint2(", xy, ")] = ", v, ";"));
                else L(cat(tex, ".write(", v, ", uint2(", xy, "));"));
                return "";
            }
            default: break;
        }
        Fail(c, "internal: texture builtin");
    }

    // The entry point's own text: its signature, and the body, with the
    // block declarations and the program's structs and functions it needs
    // before it.
    string Emit() {
        auto sf = entry->sf;
        auto stage = sf->gpustage;
        FnState st;
        st.spec = entry;
        fs = &st;
        chain.push_back(entry);
        callsites.push_back(sf->line);
        globalnames.insert(string(sf->name));
        for (auto n : { "gs_vertex_out", "gs_fragment_in", "gs_fragment_out", "GsCtx" })
            globalnames.insert(n);
        ClassifyParams();
        if (stage == GPU_VERTEX && (entry->rets.empty() || entry->rets.size() > 2 ||
                                    VecWidth(entry->rets[0]) != 4))
            Fail(sf->line, cat("vertex fn ", sf->name, " returns the clip position, a float4, "
                               "then its varyings, if any"));
        if (stage == GPU_FRAGMENT && (entry->rets.empty() || entry->rets.size() > 8))
            Fail(sf->line, cat("fragment fn ", sf->name, " returns one to eight colors"));
        string decls, loads;
        for (int k = 0; k < 2; k++) {
            auto v = k ? uniform : push;
            if (!v) continue;
            auto [d, ld] = BlockDecl(v, k);
            decls += d;
            if (msl()) loads += cat(ld, "\n");
            else loads += cat(ld, "\n");
        }
        for (auto line = loads.find('\n'), from = (size_t)0; line != string::npos;
             from = line + 1, line = loads.find('\n', from))
            L(loads.substr(from, line - from));
        // A fragment entry's stage input, built from what the rasterizer
        // interpolated.
        vector<Varying> ins, outs;
        if (stagein) {
            ins = Varyings(stagein->type, stagein->line);
            auto tn = TypeName(stagein->type, stagein->line);
            auto name = LocalName(stagein);
            L(cat(tn, " ", name, ";"));
            for (auto &vy : ins) L(cat(name, ".", vy.name, " = gs_in.", vy.name, ";"));
        }
        if (stage == GPU_VERTEX && entry->rets.size() > 1)
            outs = Varyings(entry->rets[1], sf->line);
        Body(entry);
        fs = nullptr;

        auto name = string(sf->name);
        string out = cat("// Generated by the Goose compiler from ", GpuStageName(stage), " fn ",
                         sf->name, ".\n\n");
        auto interp = [&](const Varying &vy, int k) {
            if (hlsl())
                return cat("    ", vy.flat ? "nointerpolation " : "", TypeName(vy.type, sf->line),
                           " ", vy.name, " : TEXCOORD", k, ";\n");
            return cat("    ", TypeName(vy.type, sf->line), " ", vy.name, " [[user(locn", k, ")",
                       vy.flat ? ", flat" : "", "]];\n");
        };
        string iface;
        if (stage == GPU_VERTEX) {
            iface = hlsl() ? "struct gs_vertex_out {\n    float4 position : SV_Position;\n"
                           : "struct gs_vertex_out {\n    float4 position [[position, invariant]];\n";
            for (size_t k = 0; k < outs.size(); k++) iface += interp(outs[k], (int)k);
            iface += "};\n\n";
        } else if (stage == GPU_FRAGMENT) {
            iface = hlsl() ? "struct gs_fragment_in {\n    float4 position : SV_Position;\n"
                           : "struct gs_fragment_in {\n    float4 position [[position]];\n";
            for (size_t k = 0; k < ins.size(); k++) iface += interp(ins[k], (int)k);
            if (hlsl()) iface += "    bool front_facing : SV_IsFrontFace;\n";
            iface += "};\n\n";
            if (entry->rets.size() > 1) {
                iface += "struct gs_fragment_out {\n";
                for (size_t k = 0; k < entry->rets.size(); k++)
                    iface += cat("    ", TypeName(entry->rets[k], sf->line), " color", k,
                                 hlsl() ? cat(" : SV_Target", k) : cat(" [[color(", k, ")]]"),
                                 ";\n");
                iface += "};\n\n";
            }
        }
        string ret = stage == GPU_COMPUTE ? "void" : stage == GPU_VERTEX ? "gs_vertex_out"
                   : entry->rets.size() > 1 ? "gs_fragment_out" : TypeName(entry->rets[0], sf->line);
        if (hlsl()) {
            out += structdefs;
            out += iface;
            out += decls;
            out += "\nstatic uint3 gs_dispatch_thread_id;\nstatic uint3 gs_group_id;\n"
                   "static uint3 gs_group_thread_id;\nstatic uint gs_group_index;\n"
                   "static uint gs_vertex_index;\nstatic uint gs_instance_index;\n"
                   "static float4 gs_frag_coord;\nstatic bool gs_front_facing;\n\n";
            out += helperdefs;
            out += fndefs;
            string sig, setup;
            if (stage == GPU_COMPUTE) {
                auto &g = sf->groupsize;
                out += cat("[numthreads(", g[0], ", ", g[1], ", ", g[2], ")]\n");
                sig = "uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID, "
                      "uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex";
                setup = "    gs_dispatch_thread_id = dtid;\n    gs_group_id = gid;\n"
                        "    gs_group_thread_id = gtid;\n    gs_group_index = gi;\n";
            } else if (stage == GPU_VERTEX) {
                sig = "uint vid : SV_VertexID, uint iid : SV_InstanceID";
                setup = "    gs_vertex_index = vid;\n    gs_instance_index = iid;\n";
            } else {
                sig = "gs_fragment_in gs_in";
                setup = "    gs_frag_coord = gs_in.position;\n"
                        "    gs_front_facing = gs_in.front_facing;\n";
            }
            out += cat(ret, " ", name, "(", sig, ")",
                       stage == GPU_FRAGMENT && entry->rets.size() == 1 ? " : SV_Target0" : "",
                       " {\n", setup, st.out, "}\n");
        } else {
            out += raytracing ? "#include <metal_stdlib>\n#include <metal_raytracing>\n"
                                "using namespace metal;\n\n"
                              : "#include <metal_stdlib>\nusing namespace metal;\n\n";
            out += structdefs;
            out += iface;
            out += mslheapdefs;
            if (!mslheapdefs.empty()) out += "\n";
            out += "struct GsCtx {\n    device const ulong *heap;\n    device const ulong *samplers;\n"
                   "    uint3 dispatch_thread_id;\n    uint3 group_id;\n    uint3 group_thread_id;\n"
                   "    uint group_index;\n    uint vertex_index;\n    uint instance_index;\n"
                   "    float4 frag_coord;\n    bool front_facing;\n};\n\n";
            out += helperdefs;
            out += fndefs;
            string params;
            if (push) params += cat("constant ", TypeName(push->type, push->line, true),
                                    " &gs_push [[buffer(0)]], ");
            if (uniform) params += cat("constant ", TypeName(uniform->type->ref->sub, uniform->line,
                                                             true), " &gs_uniforms [[buffer(1)]], ");
            params += "device const ulong *gs_heap [[buffer(2)]], "
                      "device const ulong *gs_samplers [[buffer(3)]]";
            string setup = "    GsCtx gs;\n    gs.heap = gs_heap;\n    gs.samplers = gs_samplers;\n";
            string kind;
            if (stage == GPU_COMPUTE) {
                kind = "kernel";
                params += ", uint3 dtid [[thread_position_in_grid]], "
                          "uint3 gid [[threadgroup_position_in_grid]], "
                          "uint3 gtid [[thread_position_in_threadgroup]], "
                          "uint gi [[thread_index_in_threadgroup]]";
                setup += "    gs.dispatch_thread_id = dtid;\n    gs.group_id = gid;\n"
                         "    gs.group_thread_id = gtid;\n    gs.group_index = gi;\n";
            } else if (stage == GPU_VERTEX) {
                kind = "vertex";
                params += ", uint vid [[vertex_id]], uint iid [[instance_id]]";
                setup += "    gs.vertex_index = vid;\n    gs.instance_index = iid;\n";
            } else {
                kind = "fragment";
                params = cat("gs_fragment_in gs_in [[stage_in]], bool ff [[front_facing]], ",
                             params);
                setup += "    gs.frag_coord = gs_in.position;\n    gs.front_facing = ff;\n";
            }
            out += cat(kind, " ", ret, " ", name, "(", params, ") {\n", setup, st.out, "}\n");
        }
        return out;
    }
};

// ------------------------------------------------------------------
// Compiling the emitted text, and the library around the binaries.

struct GpuCompileOptions {
    bool compile = true;        // False under --check: emit and hold to the subset only.
    string emitdir;             // --emit-shaders: where the text goes, "" for nowhere.
};

// A tool's output for `text`, from the cache under GOOSE_HOME when the same
// text was compiled before by the same tool and arguments. `run` writes the
// output to the path it is given.
inline string GpuCached(const string &text, const string &key, const string &ext,
                        const function<void(const string &src, const string &dst)> &run) {
    auto h = GpuHash(key, GpuHash(text));
    char hex[17];
    snprintf(hex, sizeof hex, "%016llx", (unsigned long long)h);
    auto dir = cat(GooseHome(), "/build/shaders");
    auto dst = cat(dir, "/", hex, ext);
    string bytes;
    if (LoadFile(dst, bytes)) return bytes;
    auto src = cat(dir, "/", hex, ext == ".metallib" ? ".metal" : ".hlsl");
    if (!WriteFileMakingDirs(src, text))
        throw CompileError { cat("cannot write ", src) };
    auto tmp = cat(dst, ".tmp");
    run(src, tmp);
    if (!LoadFile(tmp, bytes)) throw CompileError { cat("internal: no output in ", tmp) };
    RenamePath(tmp, dst);
    return bytes;
}

inline string GpuToolFailure(const vector<string> &argv, const ProcessResult &r) {
    return cat("the GPU compiler failed on code the Goose compiler emitted, which is a bug "
               "in Goose:\n", CommandLineText(argv), "\n", r.output);
}

inline string FindDxc() {
    auto t = FindTool("dxc");
    if (!t.path.empty()) return t.path;
    if (auto sdk = getenv("VULKAN_SDK")) {
        auto p = cat(sdk, "/bin/dxc");
        if (PathExists(p)) return p;
    }
    throw CompileError { "compiling a GPU function needs DXC, which is not on PATH: install the "
                         "Vulkan SDK (vulkan.lunarg.com), or DirectXShaderCompiler" };
}

inline const char *DxcProfile(int stage) {
    return stage == GPU_VERTEX ? "vs_6_6" : stage == GPU_FRAGMENT ? "ps_6_6" : "cs_6_6";
}

inline string GpuDxc(const string &hlsl, const string &entry, int stage, bool spirv) {
    auto dxc = FindDxc();
    vector<string> flags = { "-T", DxcProfile(stage), "-E", entry, "-HV", "2021", "-O3" };
    if (spirv)
        for (auto f : { "-spirv", "-fspv-target-env=vulkan1.2", "-fvk-use-scalar-layout",
                        "-fvk-bind-resource-heap", "0", "0", "-fvk-bind-sampler-heap", "1", "0",
                        "-fvk-invert-y" })
            flags.push_back(f);
    if (spirv && stage != GPU_VERTEX) flags.pop_back();   // -fvk-invert-y: vertex only.
    return GpuCached(hlsl, cat(spirv ? "spirv\n" : "dxil\n", dxc, "\n", CommandLineText(flags)),
                     spirv ? ".spv" : ".dxil", [&](const string &src, const string &dst) {
        vector<string> argv = { dxc };
        for (auto &f : flags) argv.push_back(f);
        argv.push_back("-Fo");
        argv.push_back(dst);
        argv.push_back(src);
        auto r = RunProcess(argv);
        if (!r.started || r.code != 0) throw CompileError { GpuToolFailure(argv, r) };
    });
}

inline string GpuMetallib(const string &msl) {
    vector<string> flags = { "-std=metal3.1", "-fno-fast-math", "-fpreserve-invariance" };
    return GpuCached(msl, cat("metallib\n", CommandLineText(flags)), ".metallib",
                     [&](const string &src, const string &dst) {
        auto air = cat(dst, ".air");
        vector<string> argv = { "xcrun", "-sdk", "macosx", "metal" };
        for (auto &f : flags) argv.push_back(f);
        argv.push_back("-c");
        argv.push_back(src);
        argv.push_back("-o");
        argv.push_back(air);
        auto r = RunProcess(argv);
        if (!r.started)
            throw CompileError { "compiling a GPU function for Metal needs the Metal toolchain: "
                                 "install Xcode, and its Metal Toolchain component" };
        if (r.code != 0) throw CompileError { GpuToolFailure(argv, r) };
        vector<string> link = { "xcrun", "-sdk", "macosx", "metallib", air, "-o", dst };
        r = RunProcess(link);
        if (!r.started || r.code != 0) throw CompileError { GpuToolFailure(link, r) };
        std::error_code ec;
        filesystem::remove(filesystem::path(air), ec);
    });
}

// One entry point, emitted and compiled.
struct GpuCompiled {
    FnSpec *entry;
    string name;
    string codes[GS_GPU_FORMAT_COUNT];
    int64_t pushsize = 0, uniformsize = 0;
    uint32_t caps = 0;
};

// The library of a pipeline's entry points (src/gpu_library.h).
inline string GpuLibrary(const vector<GpuCompiled> &entries) {
    gs_gpu_lib_header h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, GS_GPU_LIB_MAGIC, 4);
    h.version = GS_GPU_LIB_VERSION;
    h.entry_count = (uint16_t)entries.size();
    for (auto &e : entries) {
        h.push_size = max(h.push_size, (uint32_t)e.pushsize);
        h.uniform_size = max(h.uniform_size, (uint32_t)e.uniformsize);
        h.caps |= e.caps;
        if (e.entry->sf->gpustage == GPU_FRAGMENT) h.color_outputs = (uint32_t)e.entry->rets.size();
    }
    auto headsize = sizeof h + entries.size() * sizeof(gs_gpu_lib_entry);
    string tail;
    auto add = [&](const string &bytes) {
        gs_gpu_lib_range r { (uint32_t)(headsize + tail.size()), (uint32_t)bytes.size() };
        tail += bytes;
        return r;
    };
    string out((const char *)&h, sizeof h);
    for (auto &e : entries) {
        gs_gpu_lib_entry le;
        memset(&le, 0, sizeof le);
        le.stage = (uint8_t)e.entry->sf->gpustage;
        for (int k = 0; k < 3; k++) le.group_size[k] = (uint32_t)e.entry->sf->groupsize[k];
        le.name = add(e.name);
        for (int f = 0; f < GS_GPU_FORMAT_COUNT; f++) le.code[f] = add(e.codes[f]);
        out += string((const char *)&le, sizeof le);
    }
    return out + tail;
}

// Whether two entry points of one pipeline agree on what passes between
// them (§5.1): the vertex entry's varyings are the fragment entry's stage
// input, and both take the same push constants and uniform block, or one
// takes none.
inline void GpuCheckInterface(FnSpec *vs, FnSpec *fs, Line l) {
    auto inst = [](TypeExpr *t) { return t && t->kind == TY_STRUCT ? t->struc->inst : nullptr; };
    auto vout = vs->rets.size() > 1 ? inst(vs->rets[1]) : nullptr;
    auto fin = fs->params.empty() ? nullptr : inst(fs->params[0]->type);
    auto empty = [](StructInst *si) {
        if (!si) return true;
        for (auto &f : si->st->fields) if (!f.ispad) return false;
        return true;
    };
    if (vout != fin && !(empty(vout) && empty(fin)))
        throw GpuError { l, cat("fragment fn ", fs->sf->name, " takes other varyings than vertex "
                                "fn ", vs->sf->name, " passes it") };
    auto block = [&](FnSpec *sp, bool uniform) -> StructInst * {
        for (size_t i = sp->sf->gpustage == GPU_FRAGMENT ? 1 : 0; i < sp->params.size(); i++) {
            auto t = sp->params[i]->type;
            if ((t->kind == TY_REF) == uniform) return inst(uniform ? t->ref->sub : t);
        }
        return nullptr;
    };
    for (auto uniform : { false, true }) {
        auto a = block(vs, uniform), b = block(fs, uniform);
        if (a && b && a != b)
            throw GpuError { l, cat("vertex fn ", vs->sf->name, " and fragment fn ", fs->sf->name,
                                    " take different ", uniform ? "uniform blocks" : "push constants") };
    }
}

// Compiles the entry points the gpu_entry calls name, and gives each call
// its library. Errors are reported, each call's on its own.
inline void CompileGpuEntries(Ast &ast, const GpuCompileOptions &opts) {
    map<vector<FnSpec *>, const string *> done;
    for (auto c : ast.gpucalls) {
        vector<FnSpec *> entries = { c->gpuspec };
        if (c->gpuspec2) entries.push_back(c->gpuspec2);
        auto failed = false;
        for (auto e : entries) failed = failed || !e || e->failed;
        if (failed) continue;
        if (auto it = done.find(entries); it != done.end()) {
            c->shaderblob = it->second;
            continue;
        }
        vector<GpuCompiled> compiled;
        GpuEmitter *current = nullptr;
        try {
            if (entries.size() == 2) {
                if (entries[0]->sf->gpustage != GPU_VERTEX || entries[1]->sf->gpustage != GPU_FRAGMENT)
                    throw GpuError { c->line, "gpu_entry(vs, fs) takes a vertex fn and then a "
                                              "fragment fn" };
                GpuCheckInterface(entries[0], entries[1], c->line);
            } else if (entries[0]->sf->gpustage != GPU_COMPUTE) {
                throw GpuError { c->line, cat("gpu_entry(", entries[0]->sf->name, ") makes a "
                                              "compute pipeline: a ", GpuStageName(entries[0]->sf->gpustage),
                                              " entry point needs its other stage, gpu_entry(vs, fs)") };
            }
            for (auto e : entries) {
                GpuCompiled out;
                out.entry = e;
                out.name = string(e->sf->name);
                string texts[3];
                GpuEmitter dxil(ast, GD_HLSL_DXIL, e), spirv(ast, GD_HLSL_SPIRV, e),
                           metal(ast, GD_MSL, e);
                GpuEmitter *em[3] = { &dxil, &spirv, &metal };
                for (int k = 0; k < 3; k++) {
                    current = em[k];
                    texts[k] = em[k]->Emit();
                }
                current = nullptr;
                out.pushsize = dxil.pushsize;
                out.caps = (uint32_t)dxil.caps;
                out.uniformsize = dxil.uniformsize;
                if (!opts.emitdir.empty()) {
                    WriteFileMakingDirs(cat(opts.emitdir, "/", out.name, ".dxil.hlsl"), texts[0]);
                    WriteFileMakingDirs(cat(opts.emitdir, "/", out.name, ".spirv.hlsl"), texts[1]);
                    WriteFileMakingDirs(cat(opts.emitdir, "/", out.name, ".metal"), texts[2]);
                }
                if (opts.compile) {
                    auto stage = e->sf->gpustage;
                    out.codes[GS_GPU_FORMAT_SPIRV] = GpuDxc(texts[1], out.name, stage, true);
#ifdef __APPLE__
                    out.codes[GS_GPU_FORMAT_METALLIB] = GpuMetallib(texts[2]);
#endif
#ifdef _WIN32
                    out.codes[GS_GPU_FORMAT_DXIL] = GpuDxc(texts[0], out.name, stage, false);
#endif
                }
                compiled.push_back(std::move(out));
            }
        } catch (GpuError &e) {
            Diagnostic d;
            d.where = LocOf(e.where);
            d.message = e.msg;
            if (current)
                for (size_t i = current->chain.size(); i-- > 0;) {
                    auto sp = current->chain[i];
                    auto what = sp->sf->gpustage >= 0
                                    ? cat("in ", GpuStageName(sp->sf->gpustage), " fn ", sp->sf->name)
                                    : cat("in fn ", sp->sf->name, ", called from here");
                    d.related.push_back({ LocOf(i ? current->callsites[i] : sp->sf->line), what });
                }
            ast.Report(d);
            done[entries] = nullptr;
            continue;
        }
        string key = "gpu";
        for (auto e : entries) key += cat("\n", e->sf->qname);
        ast.shaders[key] = GpuLibrary(compiled);
        done[entries] = &ast.shaders[key];
        c->shaderblob = done[entries];
    }
}

}  // namespace goose
