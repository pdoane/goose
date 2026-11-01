// Goose compiler — what `goose build --debug` tells the debug adapter (dap.h): each
// function's locals by source name and C name, and each type's layout, written
// beside the generated C. A private contract between this compiler and its own
// adapter, so both change together.
#pragma once

namespace goose {

inline int CodeGen::DebugTypeId(TypeExpr *t) {
    auto it = debugtypeids.find(t);
    if (it != debugtypeids.end()) return it->second;
    string key;
    t->Dump(key);
    if (t->kind == TY_STRUCT || t->kind == TY_ENUM || t->kind == TY_VARIANT) key += cat("#", Mangle(t));
    auto byname = debugtypekeys.find(key);
    if (byname != debugtypekeys.end()) return debugtypeids[t] = byname->second;
    auto id = (int)debugtypes.size();
    debugtypeids[t] = id;
    debugtypekeys[key] = id;
    debugtypes.emplace_back();

    JsonOut o;
    o.BeginObject();
    o.Member("name", key.substr(0, key.find('#')));
    if (IsFix(t)) o.Member("size", FixedSize(t));
    auto fields = [&](const char *member) {
        o.Key(member);
        o.BeginArray();
        for (auto &run : FieldRuns(t)) {
            for (size_t i = 0; i < run.fields->size(); i++) {
                auto &f = (*run.fields)[i];
                o.BeginObject();
                o.Member("name", string(f.name));
                if (f.ispad) o.Member("pad", f.padsize);
                else o.Member("type", DebugTypeId((*run.ftypes)[i]));
                o.EndObject();
            }
        }
        o.EndArray();
    };
    switch (t->kind) {
        case TY_INT:
            o.Member("kind", "int");
            o.Member("signed", t->intstorage == IS_I8 || t->intstorage == IS_I16 || t->intstorage == IS_I32 || t->intstorage == IS_I64);
            o.Member("varint", t->intstorage == IS_VARINT);
            break;
        case TY_FLT: o.Member("kind", "float"); break;
        case TY_BOOL: o.Member("kind", "bool"); break;
        case TY_STRUCT:
            o.Member("kind", "struct");
            if (IsFrameObj(t)) o.Member("frameobject", CT(t));
            fields("fields");
            break;
        case TY_VARIANT:
            o.Member("kind", "struct");
            fields("fields");
            break;
        case TY_ENUM: {
            auto ei = EIOf(t);
            o.Member("kind", "enum");
            o.Member("tagsize", (int64_t)TagSize(ei->en));
            o.Member("enum", string(ei->en->name));
            o.Key("variants");
            o.BeginArray();
            for (size_t vi = 0; vi < ei->en->variants.size(); vi++) {
                o.BeginObject();
                o.Member("name", string(ei->en->variants[vi].name));
                o.Member("type", DebugTypeId(VariantType(t, (int)vi)));
                o.EndObject();
            }
            o.EndArray();
            break;
        }
        case TY_SLICE:
            o.Member("kind", "slice");
            o.Member("sub", DebugTypeId(t->sub));
            break;
        case TY_ARRAY: {
            auto &a = *t->arr;
            o.Member("kind", "array");
            o.Member("sub", DebugTypeId(a.sub));
            switch (a.akind) {
                case A_FIXED:
                    o.Member("shape", "fixed");
                    o.Member("count", a.size);
                    break;
                case A_LIMITED:
                    o.Member("shape", IsFix(t) ? "limited" : "limited_runtime");
                    if (IsFix(t)) o.Member("count", a.size);
                    break;
                case A_VAR: {
                    auto ls = LenStore(&a);
                    o.Member("shape", "var");
                    o.Member("lensize", ls == IS_VARINT ? (int64_t)0 : (int64_t)IntSize(ls));
                    break;
                }
                default: o.Member("shape", "resizable"); break;
            }
            break;
        }
        case TY_REF: {
            auto &r = *t->ref;
            o.Member("kind", "ref");
            o.Member("sub", DebugTypeId(r.sub));
            o.Member("optional", r.optional);
            o.Member("fat", IsFatRef(t));
            if (r.lenstorage >= 0) {
                o.Member("relative", true);
                o.Member("pool", r.pool != nullptr);
            }
            break;
        }
        default: o.Member("kind", "opaque"); break;
    }
    o.EndObject();
    debugtypes[id] = o.s;
    return id;
}

// The locals of the function just emitted, as vnames has them.
inline void CodeGen::RecordDebugFunction(FnSpec *sp, const string &cname) {
    JsonOut o;
    o.BeginObject();
    o.Member("name", string(sp->sf->name));
    o.Key("locals");
    o.BeginArray();
    for (auto &[vd, c] : vnames) {
        if (vd->istemp || vd->componentowner || vd->resulttemp || vd->name.empty() || !vd->type) continue;
        // A stack-held local's lvalue is an expression over the data stacks.
        if (c.find('(') != string::npos) continue;
        o.BeginObject();
        o.Member("name", string(vd->name));
        o.Member("c", c);
        o.Member("line", (int64_t)vd->line.line);
        o.Member("param", vd->isparam);
        if (auto end = debugscopeends.find(vd); end != debugscopeends.end() && end->second > 0)
            o.Member("end", (int64_t)end->second);
        o.Member("type", DebugTypeId(vd->type));
        if (fvptr.count(vd)) o.Member("pointer", true);
        // A root struct's fields each have storage of their own.
        if (IsRoot(vd->type) && !vd->components.empty()) {
            auto &fields = SI(vd->type)->st->fields;
            o.Key("components");
            o.BeginArray();
            for (size_t i = 0; i < vd->components.size() && i < fields.size(); i++) {
                auto comp = vd->components[i];
                if (!comp || !comp->type) continue;
                auto named = vnames.find(comp);
                if (named == vnames.end()) continue;
                o.BeginObject();
                o.Member("name", string(fields[i].name));
                o.Member("c", named->second);
                o.Member("type", DebugTypeId(comp->type));
                o.EndObject();
            }
            o.EndArray();
        }
        o.EndObject();
    }
    for (auto vd : debugconstants) {
        // One that got storage after all is read as a variable, above.
        if (vd->name.empty() || !vd->type || vnames.count(vd)) continue;
        o.BeginObject();
        o.Member("name", string(vd->name));
        o.Member("line", (int64_t)vd->line.line);
        o.Member("param", false);
        if (auto end = debugscopeends.find(vd); end != debugscopeends.end() && end->second > 0)
            o.Member("end", (int64_t)end->second);
        o.Member("type", DebugTypeId(vd->type));
        o.Member("value", to_string(vd->constval));
        o.EndObject();
    }
    debugconstants.clear();
    o.EndArray();
    o.EndObject();
    debugfunctions[cname] = o.s;
}

inline bool StartsWithGlobalBlock(const string &lvalue) { return lvalue.rfind("GS_GL->", 0) == 0; }

inline string CodeGen::DebugMap() {
    string s = cat("{\"longreturn\":", fromids.empty() ? "false" : "true", ",\"functions\":{");
    auto first = true;
    for (auto &[name, f] : debugfunctions) {
        if (!first) s += ',';
        first = false;
        JsonQuote(s, name);
        s += ':';
        s += f;
    }
    s += "},\"globals\":[";
    JsonOut g;
    g.BeginArray();
    for (auto decl : ast.globals) {
        for (auto d : decl->defs) {
            if (!d || !d->type || d->name.empty()) continue;
            // The program's own, not those of what it imports.
            if (d->line.fileidx != 0) continue;
            auto named = gnames.find(d);
            if (d->constlit) {
                g.BeginObject();
                g.Member("name", string(d->name));
                g.Member("type", DebugTypeId(d->type));
                g.Member("value", to_string(d->constval));
                g.EndObject();
            } else if (named != gnames.end() && StartsWithGlobalBlock(named->second)) {
                g.BeginObject();
                g.Member("name", string(d->name));
                g.Member("c", cat("((gs_globals_t *)gs_gl)->", named->second.substr(7)));
                g.Member("type", DebugTypeId(d->type));
                g.EndObject();
            }
        }
    }
    g.EndArray();
    s += g.s.substr(1, g.s.size() - 2);
    s += "],\"queues\":[";
    JsonOut q;
    q.BeginArray();
    for (auto &[t, name] : debugqueues) {
        q.BeginObject();
        q.Member("c", name);
        q.Member("type", DebugTypeId(t));
        q.EndObject();
    }
    q.EndArray();
    s += q.s.substr(1, q.s.size() - 2);
    s += "],\"types\":[";
    for (size_t i = 0; i < debugtypes.size(); i++) {
        if (i) s += ',';
        s += debugtypes[i];
    }
    s += "]}";
    return s;
}

}  // namespace goose
