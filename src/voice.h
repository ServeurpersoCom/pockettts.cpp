#pragma once
// voice.h: voice state, the flow LM KV rows after the voice prompt.
//
// Voices GGUF (convert.py, pocket-tts-<pack>-voices.gguf): per voice,
// <name>.k and <name>.v F32 ne [dim, T, n_layers].
//
// Voice state file (pocket-tts export-voice, safetensors): per layer i
//   transformer.layers.{i}.self_attn/cache   F32 [2, 1, T, H, hd]  K then V
//   transformer.layers.{i}.self_attn/offset  I64 [1]                live rows
//   transformer.layers.{i}.self_attn/pad     I64 [1]                0
// A [T, H, hd] block is exactly T cache rows of C = H * hd floats, the
// flow LM KV cache layout.

#include "gguf-weights.h"
#include "pt-error.h"
#include "utf8.h"
#include "yyjson.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct PTVoice {
    int                n_layers = 0;
    int                dim      = 0;
    int                n        = 0;  // rows
    std::vector<float> k;             // [layer][row][dim]
    std::vector<float> v;
};

static std::string voice_key(int layer, const char * field) {
    return "transformer.layers." + std::to_string(layer) + ".self_attn/" + field;
}

static bool voice_read_file(const char * path, std::vector<uint8_t> & data) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize(n > 0 ? (size_t) n : 0);
    bool ok = n > 0 && fread(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

// Parses a voice state file held in memory; path labels the errors.
static bool voice_parse_safetensors(PTVoice *                    vo,
                                    const std::vector<uint8_t> & data,
                                    const char *                 path,
                                    int                          n_layers,
                                    int                          dim) {
    if (data.size() < 8) {
        pt_set_error("voice: %s is too short", path);
        return false;
    }
    uint64_t hn;
    memcpy(&hn, data.data(), 8);
    if (8 + hn > data.size()) {
        pt_set_error("voice: %s has a truncated header", path);
        return false;
    }
    yyjson_doc * doc = yyjson_read((const char *) data.data() + 8, (size_t) hn, YYJSON_READ_NOFLAG);
    if (!doc) {
        pt_set_error("voice: %s has an invalid header", path);
        return false;
    }
    yyjson_val *    root = yyjson_doc_get_root(doc);
    const uint8_t * base = data.data() + 8 + hn;
    const size_t    size = data.size() - 8 - hn;

    auto tensor = [&](const std::string & key, const char * dtype, std::vector<int64_t> & shape) -> const uint8_t * {
        yyjson_val * t  = yyjson_obj_get(root, key.c_str());
        const char * dt = t ? yyjson_get_str(yyjson_obj_get(t, "dtype")) : nullptr;
        if (!dt || strcmp(dt, dtype) != 0) {
            return nullptr;
        }
        shape.clear();
        yyjson_val * s = yyjson_obj_get(t, "shape");
        size_t       i, mx;
        yyjson_val * e;
        yyjson_arr_foreach(s, i, mx, e) {
            shape.push_back(yyjson_get_int(e));
        }
        yyjson_val * off = yyjson_obj_get(t, "data_offsets");
        uint64_t     a   = yyjson_get_uint(yyjson_arr_get(off, 0));
        uint64_t     b   = yyjson_get_uint(yyjson_arr_get(off, 1));
        return (a <= b && b <= size) ? base + a : nullptr;
    };

    bool ok      = true;
    vo->n_layers = n_layers;
    vo->dim      = dim;
    for (int l = 0; l < n_layers && ok; l++) {
        std::vector<int64_t> shape;
        const uint8_t *      po = tensor(voice_key(l, "offset"), "I64", shape);
        const uint8_t *      pc = tensor(voice_key(l, "cache"), "F32", shape);
        if (!po || !pc || shape.size() != 5 || shape[0] != 2 || shape[1] != 1 || shape[3] * shape[4] != dim) {
            pt_set_error("voice: %s does not match this model (layer %d)", path, l);
            ok = false;
            break;
        }
        int64_t off;
        memcpy(&off, po, 8);
        const int64_t T = shape[2];
        if (off <= 0 || off > T || (l > 0 && off != vo->n)) {
            pt_set_error("voice: %s has an inconsistent offset", path);
            ok = false;
            break;
        }
        if (l == 0) {
            vo->n = (int) off;
            vo->k.resize((size_t) n_layers * off * dim);
            vo->v.resize((size_t) n_layers * off * dim);
        }
        const size_t  rows = (size_t) off * dim;
        const float * c    = (const float *) pc;
        memcpy(vo->k.data() + (size_t) l * rows, c, rows * sizeof(float));
        memcpy(vo->v.data() + (size_t) l * rows, c + (size_t) T * dim, rows * sizeof(float));
    }
    yyjson_doc_free(doc);
    return ok;
}

static bool voice_load_safetensors(PTVoice * vo, const char * path, int n_layers, int dim) {
    std::vector<uint8_t> data;
    if (!voice_read_file(path, data)) {
        pt_set_error("voice: cannot read %s", path);
        return false;
    }
    return voice_parse_safetensors(vo, data, path, n_layers, dim);
}

static bool voice_load_gguf(PTVoice * vo, const char * path, const char * name, int n_layers, int dim) {
    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        pt_set_error("voice: cannot load %s", path);
        return false;
    }
    const std::string kn = std::string(name) + ".k";
    const std::string vn = std::string(name) + ".v";
    ggml_tensor *     kt = ggml_get_tensor(gf.meta, kn.c_str());
    ggml_tensor *     vt = ggml_get_tensor(gf.meta, vn.c_str());
    bool              ok = kt && vt && kt->type == GGML_TYPE_F32 && ggml_are_same_shape(kt, vt) && kt->ne[0] == dim &&
              kt->ne[2] == n_layers;
    if (!ok) {
        pt_set_error("voice: no voice '%s' for this model in %s", name, path);
    } else {
        vo->n_layers = n_layers;
        vo->dim      = dim;
        vo->n        = (int) kt->ne[1];
        vo->k.resize((size_t) ggml_nelements(kt));
        vo->v.resize((size_t) ggml_nelements(vt));
        memcpy(vo->k.data(), gf_get_data(gf, kn.c_str()), ggml_nbytes(kt));
        memcpy(vo->v.data(), gf_get_data(gf, vn.c_str()), ggml_nbytes(vt));
    }
    gf_close(&gf);
    return ok;
}

static bool voice_save_safetensors(const PTVoice & vo, const char * path, int n_heads) {
    const int    hd     = vo.dim / n_heads;
    const size_t cbytes = (size_t) 2 * vo.n * vo.dim * sizeof(float);
    std::string  hdr    = "{";
    size_t       off    = 0;
    for (int l = 0; l < vo.n_layers; l++) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "%s\"%s\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[%zu,%zu]},"
                 "\"%s\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[%zu,%zu]},"
                 "\"%s\":{\"dtype\":\"F32\",\"shape\":[2,1,%d,%d,%d],\"data_offsets\":[%zu,%zu]}",
                 l ? "," : "", voice_key(l, "offset").c_str(), off, off + 8, voice_key(l, "pad").c_str(), off + 8,
                 off + 16, voice_key(l, "cache").c_str(), vo.n, n_heads, hd, off + 16, off + 16 + cbytes);
        hdr += buf;
        off += 16 + cbytes;
    }
    hdr += "}";
    while (hdr.size() % 8) {
        hdr += ' ';
    }

    FILE * f = utf8_fopen(path, "wb");
    if (!f) {
        pt_set_error("voice: cannot write %s", path);
        return false;
    }
    uint64_t     hn   = hdr.size();
    bool         ok   = fwrite(&hn, 8, 1, f) == 1 && fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size();
    const size_t rows = (size_t) vo.n * vo.dim;
    for (int l = 0; l < vo.n_layers && ok; l++) {
        int64_t offset = vo.n;
        int64_t pad    = 0;
        ok             = fwrite(&offset, 8, 1, f) == 1 && fwrite(&pad, 8, 1, f) == 1 &&
             fwrite(vo.k.data() + (size_t) l * rows, sizeof(float), rows, f) == rows &&
             fwrite(vo.v.data() + (size_t) l * rows, sizeof(float), rows, f) == rows;
    }
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
        pt_set_error("voice: write failed on %s", path);
    }
    return ok;
}
