#pragma once
// text-prep.h: text preparation and chunking of pocket_tts/models/text_chunking.py.
//
// prepare: strip, per pack character replacement (then whitespace runs
// collapsed and a sentence mark followed by , ; : keeps the mark only),
// newlines to spaces, "  " -> " ", optional ; -> , , first letter
// uppercased, sentence final punctuation ensured, optional 8 space pad for
// short inputs. It also returns the frames after EOS guess: 3 for at most 4
// words, 1 otherwise.
//
// split: the prepared text is tokenized and cut after runs of sentence end
// tokens (a period between two digits does not count), sentences over the
// token budget are cut again after , ; : tokens, and consecutive pieces are
// regrouped greedily into chunks of at most max_tokens tokens.
//
// Strings are UTF-8; the character level work runs on code points.

#include "model.h"
#include "tokenizer.h"

#include <string>
#include <vector>

static std::u32string tp_decode(const std::string & s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char) s[i];
        int           n = utf8_len(c);
        char32_t      cp;
        if (n == 1) {
            cp = c;
        } else if (n == 2) {
            cp = c & 0x1f;
        } else if (n == 3) {
            cp = c & 0x0f;
        } else {
            cp = c & 0x07;
        }
        for (int k = 1; k < n && i + k < s.size(); k++) {
            cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3f);
        }
        out += cp;
        i += n;
    }
    return out;
}

static std::string tp_encode(const std::u32string & s) {
    std::string out;
    for (char32_t cp : s) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xc0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
            out += (char) (0xe0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3f));
            out += (char) (0x80 | (cp & 0x3f));
        } else {
            out += (char) (0xf0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3f));
            out += (char) (0x80 | ((cp >> 6) & 0x3f));
            out += (char) (0x80 | (cp & 0x3f));
        }
    }
    return out;
}

// Python str.isspace
static bool tp_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

static bool tp_in(char32_t c, const char32_t * set) {
    for (; *set; set++) {
        if (*set == c) {
            return true;
        }
    }
    return false;
}

static std::u32string tp_strip(const std::u32string & s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && tp_space(s[a])) {
        a++;
    }
    while (b > a && tp_space(s[b - 1])) {
        b--;
    }
    return s.substr(a, b - a);
}

static std::vector<std::u32string> tp_split(const std::u32string & s) {
    std::vector<std::u32string> out;
    std::u32string              cur;
    for (char32_t c : s) {
        if (tp_space(c)) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

static std::u32string tp_replace(const std::u32string & s, const std::u32string & a, const std::u32string & b) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, a.size(), a) == 0) {
            out += b;
            i += a.size();
        } else {
            out += s[i++];
        }
    }
    return out;
}

// Upper case of the Latin, Greek and Cyrillic lower case letters.
static char32_t tp_upper(char32_t c) {
    if (c >= 'a' && c <= 'z') {
        return c - 32;
    }
    if ((c >= 0xe0 && c <= 0xfe && c != 0xf7)) {
        return c - 32;
    }
    if (c == 0xff) {
        return 0x178;
    }
    if (c >= 0x100 && c <= 0x17f) {
        const bool odd_lower  = (c >= 0x100 && c <= 0x137) || (c >= 0x14a && c <= 0x177);
        const bool even_lower = (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17e);
        if (odd_lower && (c & 1)) {
            return c - 1;
        }
        if (even_lower && !(c & 1)) {
            return c - 1;
        }
        return c;
    }
    if (c >= 0x3b1 && c <= 0x3c9 && c != 0x3c2) {
        return c - 32;
    }
    if (c == 0x3c2) {
        return 0x3a3;
    }
    if (c >= 0x430 && c <= 0x44f) {
        return c - 32;
    }
    if (c >= 0x450 && c <= 0x45f) {
        return c - 80;
    }
    return c;
}

static const char32_t TP_TERMINAL[] = U".!?\u2026";
static const char32_t TP_WEAK[]     = U",;:-\u2013\u2014";
static const char32_t TP_CLOSERS[]  = U"\"'\u201d\u2019)]\u00bb";

static std::u32string tp_terminal_punctuation(const std::u32string & text) {
    size_t e = text.size();
    while (e > 0 && (text[e - 1] == U' ' || tp_in(text[e - 1], TP_CLOSERS))) {
        e--;
    }
    std::u32string core    = text.substr(0, e);
    std::u32string closers = tp_strip(text.substr(e));
    if (core.empty() || tp_in(core.back(), TP_TERMINAL)) {
        return text;
    }
    if (tp_in(core.back(), TP_WEAK)) {
        while (!core.empty() && (core.back() == U' ' || tp_in(core.back(), TP_WEAK))) {
            core.pop_back();
        }
        return core + U"." + closers;
    }
    return text + U".";
}

// Returns false on empty text. *fae_guess receives the frames after EOS guess.
static bool tp_prepare(const PTHparams & hp, const std::string & in, std::string * out, int * fae_guess) {
    std::u32string text = tp_strip(tp_decode(in));
    if (!hp.replace_from.empty()) {
        std::u32string t;
        for (char32_t c : text) {
            bool hit = false;
            for (size_t k = 0; k < hp.replace_from.size(); k++) {
                std::u32string f = tp_decode(hp.replace_from[k]);
                if (f.size() == 1 && f[0] == c) {
                    t += tp_decode(hp.replace_to[k]);
                    hit = true;
                    break;
                }
            }
            if (!hit) {
                t += c;
            }
        }
        std::u32string joined;
        for (const std::u32string & w : tp_split(t)) {
            if (!joined.empty()) {
                joined += U' ';
            }
            joined += w;
        }
        // ([.!?...])\s*[,;:] -> \1
        text.clear();
        for (size_t i = 0; i < joined.size(); i++) {
            text += joined[i];
            if (tp_in(joined[i], TP_TERMINAL)) {
                size_t j = i + 1;
                while (j < joined.size() && tp_space(joined[j])) {
                    j++;
                }
                if (j < joined.size() && tp_in(joined[j], U",;:")) {
                    i = j;
                }
            }
        }
    }
    if (text.empty()) {
        return false;
    }
    for (char32_t & c : text) {
        if (c == U'\n' || c == U'\r') {
            c = U' ';
        }
    }
    text = tp_replace(text, U"  ", U" ");
    if (hp.remove_semicolons) {
        for (char32_t & c : text) {
            if (c == U';') {
                c = U',';
            }
        }
    }
    *fae_guess = tp_split(text).size() <= 4 ? 3 : 1;
    if (hp.capitalize_first_letter) {
        text[0] = tp_upper(text[0]);
    }
    if (hp.append_terminal_punctuation) {
        text = tp_terminal_punctuation(text);
    }
    if (hp.pad_with_spaces && tp_split(text).size() < 5) {
        text = U"        " + text;
    }
    *out = tp_encode(text);
    return true;
}

// Cut points after runs of boundary tokens, [0, ..., n].
static std::vector<size_t> tp_boundaries(const PTTokenizer &      tok,
                                         const std::vector<int> & ids,
                                         const std::vector<int> & bounds,
                                         bool                     skip_decimal) {
    std::vector<size_t> out   = { 0 };
    bool                after = false;
    for (size_t i = 0; i < ids.size(); i++) {
        bool is_b = false;
        for (int b : bounds) {
            is_b |= ids[i] == b;
        }
        if (is_b) {
            after = true;
            continue;
        }
        if (after) {
            bool decimal = false;
            if (skip_decimal) {
                std::string pre = tok_decode(tok, ids, 0, i);
                std::string suf = tok_decode(tok, ids, i, ids.size());
                decimal = pre.size() >= 2 && pre.back() == '.' && isdigit((unsigned char) pre[pre.size() - 2]) &&
                          !suf.empty() && isdigit((unsigned char) suf[0]);
            }
            if (!decimal) {
                out.push_back(i);
            }
        }
        after = false;
    }
    out.push_back(ids.size());
    return out;
}

static std::string tp_trim(const std::string & s) {
    return tp_encode(tp_strip(tp_decode(s)));
}

static std::vector<std::string> tp_chunks(const PTHparams &   hp,
                                          const PTTokenizer & tok,
                                          const std::string & in,
                                          int                 max_tokens) {
    std::vector<std::string> chunks;
    std::string              text;
    int                      guess;
    if (!tp_prepare(hp, in, &text, &guess)) {
        return chunks;
    }
    text = tp_trim(text);

    std::vector<int> ids = tok_encode(tok, text);
    std::vector<int> eos = tok_encode(tok, ".!...?");
    std::vector<int> sub = tok_encode(tok, ",;:");
    eos.erase(eos.begin());
    sub.erase(sub.begin());

    std::vector<std::pair<int, std::string>> segs;
    std::vector<size_t>                      b = tp_boundaries(tok, ids, eos, true);
    for (size_t i = 0; i + 1 < b.size(); i++) {
        const int         n = (int) (b[i + 1] - b[i]);
        const std::string s = tok_decode(tok, ids, b[i], b[i + 1]);
        if (n <= max_tokens) {
            segs.push_back({ n, s });
            continue;
        }
        std::vector<int>    sid = tok_encode(tok, tp_trim(s));
        std::vector<size_t> sb  = tp_boundaries(tok, sid, sub, false);
        if (sb.size() > 2) {
            for (size_t k = 0; k + 1 < sb.size(); k++) {
                segs.push_back({ (int) (sb[k + 1] - sb[k]), tok_decode(tok, sid, sb[k], sb[k + 1]) });
            }
        } else {
            segs.push_back({ n, s });
        }
    }

    std::string cur;
    int         cur_n = 0;
    for (const auto & sg : segs) {
        if (cur.empty()) {
            cur   = sg.second;
            cur_n = sg.first;
            continue;
        }
        if (cur_n + sg.first > max_tokens) {
            chunks.push_back(tp_trim(cur));
            cur   = sg.second;
            cur_n = sg.first;
        } else {
            cur += " " + sg.second;
            cur_n += sg.first;
        }
    }
    if (!cur.empty()) {
        chunks.push_back(tp_trim(cur));
    }
    return chunks;
}
