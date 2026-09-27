// tts-server.cpp: OpenAI-compatible HTTP server backed by the pockettts
// ABI. Loads one language pack once, GPU resident, and serves synthesis
// over POST /v1/audio/speech. The shared core lives in src/tts-server.h;
// this file wires the pt_* ABI into the generic adapter.
//
// Voices: the predefined voices of the --voices GGUF are builtin voices,
// loaded on first use; POST /v1/audio/voices adds registered voices at
// runtime. A request without a voice takes --voice, else the first builtin.

#include "tts-server.h"

#include "gguf.h"
#include "pocket.h"
#include "utf8.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using voice_ptr = std::shared_ptr<pt_voice>;

static pt_context *                     g_ctx = nullptr;
static std::mutex                       g_voices_mutex;
static std::string                      g_voices_path;
static std::vector<std::string>         g_builtin;
static std::map<std::string, voice_ptr> g_loaded;      // builtin voices already loaded
static std::map<std::string, voice_ptr> g_registered;  // runtime voices
static std::string                      g_default_voice;
static float                            g_max_voice_sec = 30.0f;

static void print_usage(const char * prog) {
    fprintf(stderr, "pockettts.cpp %s\n\n", POCKET_VERSION);
    fprintf(stderr,
            "Usage: %s --model <gguf> --voices <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Language pack GGUF (pocket-tts-<pack>-<quant>.gguf)\n\n"
            "Optional:\n"
            "  --voices <gguf>         Predefined voices of the pack (pocket-tts-<pack>-voices.gguf)\n"
            "  --voice <name>          Voice of the requests that name none (default: first voice)\n"
            "  --max-voice-sec <f>     Recording length kept for cloning (default: 30, 0 keeps all)\n"
            "  --alias <name>          Report this model id instead of the GGUF file name\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n",
            prog);
}

static std::string base_name(const char * path) {
    std::string s(path);
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

static voice_ptr make_voice(pt_voice * v) {
    return voice_ptr(v, pt_voice_free);
}

static voice_ptr find_voice(const std::string & req, std::string & err) {
    const std::string           name = req.empty() ? g_default_voice : req;
    std::lock_guard<std::mutex> lk(g_voices_mutex);
    auto                        r = g_registered.find(name);
    if (r != g_registered.end()) {
        return r->second;
    }
    auto l = g_loaded.find(name);
    if (l != g_loaded.end()) {
        return l->second;
    }
    if (std::find(g_builtin.begin(), g_builtin.end(), name) == g_builtin.end()) {
        err = "unknown voice '" + name + "'";
        return nullptr;
    }
    pt_voice * v = pt_voice_named(g_ctx, g_voices_path.c_str(), name.c_str());
    if (!v) {
        err = pt_last_error();
        return nullptr;
    }
    voice_ptr p    = make_voice(v);
    g_loaded[name] = p;
    return p;
}

// Voice names of a voices GGUF: every <name>.k tensor.
static bool list_voices(const char * path) {
    struct gguf_init_params gp = { true, nullptr };
    struct gguf_context *   g  = gguf_init_from_file(path, gp);
    if (!g) {
        return false;
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        std::string t = gguf_get_tensor_name(g, i);
        if (t.size() > 2 && t.compare(t.size() - 2, 2, ".k") == 0) {
            g_builtin.push_back(t.substr(0, t.size() - 2));
        }
    }
    gguf_free(g);
    return true;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    const char *  model  = nullptr;
    const char *  voices = nullptr;
    const char *  alias  = nullptr;
    server_config cfg;
    bool          use_fa     = true;
    bool          clamp_fp16 = false;
    for (int i = 1; i < argc; i++) {
        const char * s   = argv[i];
        const bool   val = i + 1 < argc;
        if (!strcmp(s, "--model") && val) {
            model = argv[++i];
        } else if (!strcmp(s, "--voices") && val) {
            voices = argv[++i];
        } else if (!strcmp(s, "--voice") && val) {
            g_default_voice = tts_voice_name(argv[++i]);
        } else if (!strcmp(s, "--max-voice-sec") && val) {
            g_max_voice_sec = (float) atof(argv[++i]);
        } else if (!strcmp(s, "--no-fa")) {
            use_fa = false;
        } else if (!strcmp(s, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!strcmp(s, "--alias") && val) {
            alias = argv[++i];
        } else if (!strcmp(s, "--host") && val) {
            cfg.host = argv[++i];
        } else if (!strcmp(s, "--port") && val) {
            cfg.port = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (!model) {
        print_usage(argv[0]);
        return 1;
    }

    pt_init_params ip;
    pt_init_default_params(&ip);
    ip.model_path = model;
    ip.use_fa     = use_fa;
    ip.clamp_fp16 = clamp_fp16;
    g_ctx         = pt_init(&ip);
    if (!g_ctx) {
        fprintf(stderr, "[Server] FATAL: %s\n", pt_last_error());
        return 1;
    }
    if (voices) {
        g_voices_path = voices;
        if (!list_voices(voices)) {
            fprintf(stderr, "[Server] FATAL: cannot read %s\n", voices);
            pt_free(g_ctx);
            return 1;
        }
    }
    if (g_default_voice.empty() && !g_builtin.empty()) {
        g_default_voice = g_builtin.front();
    }
    pt_log(PT_LOG_INFO, "[Server] %zu voices, default '%s'", g_builtin.size(), g_default_voice.c_str());

    tts_backend be;
    be.model_id    = alias ? alias : base_name(model);
    be.sample_rate = pt_sample_rate(g_ctx);
    be.voices      = g_builtin;

    be.synthesize = [](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        voice_ptr v = find_voice(req.voice, err);
        if (!v) {
            return PT_STATUS_INVALID_PARAMS;
        }
        pt_tts_params p;
        pt_tts_default_params(&p);
        p.text  = req.input.c_str();
        p.voice = v.get();
        if (req.seed >= 0) {
            p.seed = (uint32_t) req.seed;
        }
        if (!std::isnan(req.temperature)) {
            p.temperature = req.temperature;
        }
        p.on_chunk = [](const float * s, int n, void * ud) -> bool {
            return (*(const tts_sink *) ud)(s, n);
        };
        p.user_data = (void *) &sink;
        int rc      = pt_synthesize(g_ctx, &p, nullptr);
        if (rc != PT_STATUS_OK) {
            err = pt_last_error();
        }
        return rc;
    };

    be.register_voice = [](const tts_voice_upload & up, std::string & err) -> bool {
        pt_voice * v = nullptr;
        if (!up.state.empty()) {
            v = pt_voice_from_state(g_ctx, up.state.data(), up.state.size());
        } else {
            int     T   = 0;
            int     sr  = 0;
            float * raw = audio_io_read_wav_buf((const uint8_t *) up.wav.data(), up.wav.size(), &T, &sr);
            int     n   = 0;
            float * pcm = raw ? audio_mono_from_planar(raw, T, sr, sr, &n) : nullptr;
            if (!pcm) {
                err = "invalid WAV payload";
                return false;
            }
            if (g_max_voice_sec > 0.0f && n > (int) (g_max_voice_sec * (float) sr)) {
                n = (int) (g_max_voice_sec * (float) sr);
            }
            v = pt_voice_from_audio(g_ctx, pcm, n, sr);
            free(pcm);
        }
        if (!v) {
            err = pt_last_error();
            return false;
        }
        std::lock_guard<std::mutex> lk(g_voices_mutex);
        g_registered[up.name] = make_voice(v);
        return true;
    };

    be.remove_voice = [](const std::string & name) -> bool {
        std::lock_guard<std::mutex> lk(g_voices_mutex);
        return g_registered.erase(name) > 0;
    };

    be.registered_voices = []() {
        std::lock_guard<std::mutex> lk(g_voices_mutex);
        std::vector<std::string>    out;
        for (const auto & kv : g_registered) {
            out.push_back(kv.first);
        }
        return out;
    };

    int rc = tts_server_run(be, cfg);
    {
        std::lock_guard<std::mutex> lk(g_voices_mutex);
        g_loaded.clear();
        g_registered.clear();
    }
    pt_free(g_ctx);
    return rc;
}
