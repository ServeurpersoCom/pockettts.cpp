#pragma once
// tokenizer.h: the tokenizer.json of the packs, a SentencePiece Unigram
// model exported to HuggingFace tokenizers.
//
// Encode: prepend U+2581, replace every space by U+2581, split before each
// U+2581, then per piece the Viterbi path maximizing the summed piece log
// probabilities. A character no single character piece covers takes an
// unknown edge scored min_score - 10 and is emitted as its UTF-8 bytes
// (<0xXX> pieces). Ties keep the first (shortest) candidate.
//
// Decode: U+2581 back to spaces with the leading space of the first piece
// dropped, runs of <0xXX> pieces reassembled into UTF-8.

#include "gguf-weights.h"
#include "pt-error.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

static const char * PT_METASPACE = "\xe2\x96\x81";  // U+2581

struct PTTokenizer {
    std::vector<std::string>             pieces;
    std::vector<float>                   scores;
    std::unordered_map<std::string, int> ids;
    int                                  byte_ids[256];
    int                                  max_len   = 0;
    float                                unk_score = 0.0f;
};

static int utf8_len(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xe ? 3 : (c >> 3) == 0x1e ? 4 : 1;
}

static void tok_load(PTTokenizer * t, const GGUFModel & gf) {
    int64_t kt = gguf_find_key(gf.gguf, "tokenizer.ggml.tokens");
    int64_t ks = gguf_find_key(gf.gguf, "tokenizer.ggml.scores");
    if (kt < 0 || ks < 0) {
        pt_throw("[Tokenizer] vocabulary missing from GGUF");
    }
    const size_t  n  = gguf_get_arr_n(gf.gguf, kt);
    const float * sc = (const float *) gguf_get_arr_data(gf.gguf, ks);
    float         mn = 0.0f;
    for (int i = 0; i < 256; i++) {
        t->byte_ids[i] = -1;
    }
    for (size_t i = 0; i < n; i++) {
        std::string p = gguf_get_arr_str(gf.gguf, kt, i);
        t->pieces.push_back(p);
        t->scores.push_back(sc[i]);
        if (sc[i] < mn) {
            mn = sc[i];
        }
        unsigned int b;
        if (p.size() == 6 && sscanf(p.c_str(), "<0x%02X>", &b) == 1) {
            t->byte_ids[b] = (int) i;
            continue;
        }
        if (p == "<unk>" || p == "<s>" || p == "</s>" || p == "<pad>") {
            continue;
        }
        t->ids.emplace(p, (int) i);
        if ((int) p.size() > t->max_len) {
            t->max_len = (int) p.size();
        }
    }
    t->unk_score = mn - 10.0f;
}

static void tok_encode_piece(const PTTokenizer & t, const std::string & s, std::vector<int> & out) {
    const int          n = (int) s.size();
    std::vector<float> best(n + 1, -1e30f);
    std::vector<int>   from(n + 1, -1);
    std::vector<int>   id(n + 1, -1);
    best[0] = 0.0f;
    for (int i = 0; i < n; i = i + utf8_len((unsigned char) s[i])) {
        if (best[i] <= -1e30f) {
            continue;
        }
        const int cl     = utf8_len((unsigned char) s[i]);
        bool      single = false;
        for (int j = i + cl; j <= n && j - i <= t.max_len;) {
            auto it = t.ids.find(s.substr(i, j - i));
            if (it != t.ids.end()) {
                float sc = best[i] + t.scores[it->second];
                if (from[j] < 0 || sc > best[j]) {
                    best[j] = sc;
                    from[j] = i;
                    id[j]   = it->second;
                }
                if (j - i == cl) {
                    single = true;
                }
            }
            if (j == n) {
                break;
            }
            j += utf8_len((unsigned char) s[j]);
        }
        if (!single && i + cl <= n) {
            float sc = best[i] + t.unk_score;
            if (from[i + cl] < 0 || sc > best[i + cl]) {
                best[i + cl] = sc;
                from[i + cl] = i;
                id[i + cl]   = -1;
            }
        }
    }
    std::vector<std::pair<int, int>> path;
    for (int j = n; j > 0; j = from[j]) {
        path.push_back({ from[j], j });
    }
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        const int j = it->second;
        if (id[j] >= 0) {
            out.push_back(id[j]);
            continue;
        }
        for (int k = it->first; k < j; k++) {
            out.push_back(t.byte_ids[(unsigned char) s[k]]);
        }
    }
}

static std::vector<int> tok_encode(const PTTokenizer & t, const std::string & text) {
    std::string norm = PT_METASPACE;
    for (char c : text) {
        if (c == ' ') {
            norm += PT_METASPACE;
        } else {
            norm += c;
        }
    }
    std::vector<int> out;
    size_t           start = 0;
    while (start < norm.size()) {
        size_t next = norm.find(PT_METASPACE, start + 3);
        if (next == std::string::npos) {
            next = norm.size();
        }
        tok_encode_piece(t, norm.substr(start, next - start), out);
        start = next;
    }
    return out;
}

static std::string tok_decode(const PTTokenizer & t, const std::vector<int> & ids, size_t begin, size_t end) {
    std::string out;
    std::string bytes;
    for (size_t i = begin; i < end; i++) {
        const std::string & p = t.pieces[ids[i]];
        unsigned int        b;
        if (p.size() == 6 && sscanf(p.c_str(), "<0x%02X>", &b) == 1) {
            bytes += (char) b;
            continue;
        }
        out += bytes;
        bytes.clear();
        std::string s;
        for (size_t k = 0; k < p.size();) {
            if (p.compare(k, 3, PT_METASPACE) == 0) {
                s += ' ';
                k += 3;
            } else {
                s += p[k++];
            }
        }
        if (i == begin && !s.empty() && s[0] == ' ') {
            s.erase(0, 1);
        }
        out += s;
    }
    return out + bytes;
}
