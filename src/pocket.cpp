// pocket.cpp: public ABI and synthesis pipeline.
//
// Per synthesis: the text is prepared and cut into chunks of at most
// max_chunk_tokens tokens. Each chunk starts from the voice state (the
// voice rows are written to the KV cache, the text is prefilled after
// them) and runs the AR loop: one flow LM step per 80 ms latent frame
// until EOS (ignored on the first frames) plus frames_after_eos, or until
// the length budget of the chunk. Latents stream into the Mimi decoder,
// whose state restarts at every chunk, in blocks growing 1, 2, 4 ... 16
// frames so the first audio leaves after one frame.
//
// The noise of every step comes from a torch compatible generator seeded
// once per synthesis; each text prefill draws one discarded noise vector
// like the reference forward does, so a seed gives the reference noise.

#include "pocket.h"

#include "audio-io.h"
#include "backend.h"
#include "debug.h"
#include "flow-lm.h"
#include "ggml-alloc.h"
#include "gguf-weights.h"
#include "graph-arena.h"
#include "mimi.h"
#include "model.h"
#include "pt-error.h"
#include "text-prep.h"
#include "timer.h"
#include "tokenizer.h"
#include "torch-rng.h"
#include "version.h"
#include "voice.h"

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static const int PT_MAX_NODES       = 16384;
static const int PT_EOS_MIN_FRAMES  = 6;
static const int PT_DEC_MAX_FRAMES  = 16;
static const int PT_STEP_KV_PADDING = 256;

static thread_local std::string g_last_error;

void pt_set_error(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (n < 0) {
        g_last_error = "pt_set_error: vsnprintf failed";
    } else {
        g_last_error.resize((size_t) n);
        std::vsnprintf(g_last_error.data(), (size_t) n + 1, fmt, ap);
    }
    va_end(ap);
}

void pt_throw(const char * fmt, ...) {
    char    buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    throw std::runtime_error(buf);
}

static std::atomic<pt_log_cb> g_log_cb{ nullptr };
static void *                 g_log_user = nullptr;

void pt_log(pt_log_level level, const char * fmt, ...) {
    char        stackbuf[512];
    std::string heap;
    char *      buf = stackbuf;
    va_list     ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap2);
    va_end(ap2);
    if (n < 0) {
        va_end(ap);
        return;
    }
    if ((size_t) n >= sizeof(stackbuf)) {
        heap.resize((size_t) n);
        std::vsnprintf(heap.data(), (size_t) n + 1, fmt, ap);
        buf = heap.data();
    }
    va_end(ap);
    pt_log_cb cb = g_log_cb.load(std::memory_order_acquire);
    if (cb) {
        cb(level, buf, g_log_user);
    } else {
        std::fprintf(stderr, "%s\n", buf);
    }
}

struct pt_voice {
    PTVoice st;
    int     n_heads;
};

struct pt_context {
    ggml_backend_t     backend = nullptr;
    PTHparams          hp;
    PTOpts             opts;
    PTWeights          w;
    PTTokenizer        tok;
    FlowKV             kv;
    MimiState          ms;
    std::vector<float> bos_voice;

    GraphArena     ar_prefill, ar_step, ar_dec, ar_enc;
    ggml_gallocr_t ga_prefill = nullptr;
    ggml_gallocr_t ga_step    = nullptr;
    ggml_gallocr_t ga_dec     = nullptr;
    ggml_gallocr_t ga_enc     = nullptr;

    std::mutex mu;
};

static ggml_cgraph * pt_graph(GraphArena & a, ggml_context ** ctx) {
    *ctx = graph_arena_begin(&a);
    return ggml_new_graph_custom(*ctx, PT_MAX_NODES, false);
}

static bool pt_alloc(ggml_gallocr_t ga, ggml_cgraph * gf, const char * what) {
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        pt_set_error("%s: graph allocation failed", what);
        return false;
    }
    return true;
}

static bool pt_compute(pt_context * c, ggml_cgraph * gf, const char * what) {
    if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS) {
        pt_set_error("%s: graph compute failed on %s", what, ggml_backend_name(c->backend));
        return false;
    }
    return true;
}

// Prefill T positions from n_past: host embeddings emb [dim, T], or text
// token ids through the lookup table. emb_out, when set, receives the
// input embeddings [dim, T].
static bool pt_prefill(pt_context *                 c,
                       const float *                emb,
                       const std::vector<int32_t> * ids,
                       int                          T,
                       int                          n_past,
                       float *                      emb_out) {
    ggml_context * ctx;
    ggml_cgraph *  gf   = pt_graph(c->ar_prefill, &ctx);
    const int      n_kv = n_past + T;
    FlowInputs     in   = flow_inputs(ctx, T, n_kv);
    ggml_tensor *  src;
    ggml_tensor *  x;
    if (ids) {
        src = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(src);
        x = ggml_get_rows(ctx, c->w.embed, src);
    } else {
        src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->hp.dim, T);
        ggml_set_input(src);
        x = src;
    }
    ggml_tensor * x_in = x;
    ggml_set_output(x_in);
    x = flow_backbone(ctx, gf, c->w, c->hp, c->kv, x, in, n_kv, c->opts);
    ggml_build_forward_expand(gf, x);
    if (!pt_alloc(c->ga_prefill, gf, "prefill")) {
        return false;
    }
    flow_fill_inputs(in, n_past, T, n_kv);
    if (ids) {
        ggml_backend_tensor_set(src, ids->data(), 0, (size_t) T * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(src, emb, 0, (size_t) c->hp.dim * T * sizeof(float));
    }
    if (!pt_compute(c, gf, "prefill")) {
        return false;
    }
    if (emb_out) {
        ggml_backend_tensor_get(x_in, emb_out, 0, (size_t) c->hp.dim * T * sizeof(float));
    }
    return true;
}

// hidden, when set, receives out_norm(h) [dim].
static bool pt_step(pt_context *  c,
                    const float * latent,
                    const float * noise,
                    int           n_steps,
                    int           n_past,
                    float *       eos,
                    float *       out,
                    float *       hidden) {
    ggml_context * ctx;
    ggml_cgraph *  gf   = pt_graph(c->ar_step, &ctx);
    int            n_kv = (n_past + 1 + PT_STEP_KV_PADDING - 1) / PT_STEP_KV_PADDING * PT_STEP_KV_PADDING;
    if (n_kv > c->kv.capacity) {
        n_kv = c->kv.capacity;
    }
    FlowStep st = flow_build_step(ctx, gf, c->w, c->hp, c->kv, n_kv, n_steps, c->opts);
    if (!pt_alloc(c->ga_step, gf, "step")) {
        return false;
    }
    flow_fill_inputs(st.in, n_past, 1, n_kv);
    ggml_backend_tensor_set(st.latent, latent, 0, (size_t) c->hp.ldim * sizeof(float));
    ggml_backend_tensor_set(st.noise, noise, 0, (size_t) c->hp.ldim * sizeof(float));
    if (!pt_compute(c, gf, "step")) {
        return false;
    }
    ggml_backend_tensor_get(st.eos, eos, 0, sizeof(float));
    ggml_backend_tensor_get(st.out, out, 0, (size_t) c->hp.ldim * sizeof(float));
    if (hidden) {
        ggml_backend_tensor_get(st.hidden, hidden, 0, (size_t) c->hp.dim * sizeof(float));
    }
    return true;
}

static bool pt_decode(pt_context * c, const float * latents, int n, std::vector<float> & audio) {
    ggml_context * ctx;
    ggml_cgraph *  gf = pt_graph(c->ar_dec, &ctx);
    MimiDec        d  = mimi_build_decode(ctx, gf, c->w, c->hp, c->ms, n, c->opts);
    if (!pt_alloc(c->ga_dec, gf, "decode")) {
        return false;
    }
    mimi_fill_decode(d, c->hp, c->ms, latents, n);
    if (!pt_compute(c, gf, "decode")) {
        return false;
    }
    audio.resize((size_t) ggml_nelements(d.audio));
    ggml_backend_tensor_get(d.audio, audio.data(), 0, audio.size() * sizeof(float));
    c->ms.pos += n * c->hp.upsample_stride;
    return true;
}

static void pt_write_voice(pt_context * c, const PTVoice & vo) {
    const size_t rows = (size_t) vo.n * vo.dim;
    for (int l = 0; l < vo.n_layers; l++) {
        ggml_backend_tensor_set(c->kv.k[l], vo.k.data() + (size_t) l * rows, 0, rows * sizeof(float));
        ggml_backend_tensor_set(c->kv.v[l], vo.v.data() + (size_t) l * rows, 0, rows * sizeof(float));
    }
}

extern "C" {

const char * pt_version(void) {
    return POCKET_VERSION;
}

const char * pt_last_error(void) {
    return g_last_error.c_str();
}

void pt_log_set(pt_log_cb cb, void * user_data) {
    g_log_user = user_data;
    g_log_cb.store(cb, std::memory_order_release);
}

void pt_audio_free(struct pt_audio * a) {
    if (a) {
        free(a->samples);
        *a = {};
    }
}

void pt_init_default_params(struct pt_init_params * p) {
    *p             = {};
    p->abi_version = PT_ABI_VERSION;
    p->use_fa      = true;
}

void pt_free(struct pt_context * c) {
    if (!c) {
        return;
    }
    ggml_gallocr_free(c->ga_prefill);
    ggml_gallocr_free(c->ga_step);
    ggml_gallocr_free(c->ga_dec);
    ggml_gallocr_free(c->ga_enc);
    graph_arena_free(&c->ar_prefill);
    graph_arena_free(&c->ar_step);
    graph_arena_free(&c->ar_dec);
    graph_arena_free(&c->ar_enc);
    flow_kv_free(&c->kv);
    mimi_state_free(&c->ms);
    pt_free_weights(&c->w);
    if (c->backend) {
        ggml_backend_free(c->backend);
    }
    delete c;
}

struct pt_context * pt_init(const struct pt_init_params * params) {
    if (!params || !params->model_path) {
        pt_set_error("pt_init: params or model_path is NULL");
        return nullptr;
    }
    if (params->abi_version < PT_ABI_MIN_VERSION || params->abi_version > PT_ABI_VERSION) {
        pt_set_error("pt_init: abi_version %d outside [%d, %d]", params->abi_version, PT_ABI_MIN_VERSION,
                     PT_ABI_VERSION);
        return nullptr;
    }
    pt_context * c  = new pt_context();
    GGUFModel    gf = {};
    try {
        Timer t;
        c->backend = backend_init();
        if (!c->backend) {
            pt_throw("pt_init: no backend");
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(c->backend);
        c->opts.fa             = params->use_fa && dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
        c->opts.clamp_fp16     = params->clamp_fp16;
        if (!gf_load(&gf, params->model_path)) {
            pt_throw("pt_init: cannot load %s", params->model_path);
        }
        pt_load_hparams(&c->hp, gf);
        tok_load(&c->tok, gf);
        pt_load_weights(&c->w, c->hp, gf, c->backend);
        if (c->hp.bos_before_voice) {
            c->bos_voice = pt_host_f32(gf, "flow_lm.bos_before_voice");
        }
        gf_close(&gf);
        if (!mimi_state_init(&c->ms, c->w, c->hp, c->backend)) {
            pt_throw("pt_init: Mimi state allocation failed");
        }
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(c->backend);
        c->ga_prefill                 = ggml_gallocr_new(bt);
        c->ga_step                    = ggml_gallocr_new(bt);
        c->ga_dec                     = ggml_gallocr_new(bt);
        c->ga_enc                     = ggml_gallocr_new(bt);
        if (!graph_arena_init(&c->ar_prefill, PT_MAX_NODES) || !graph_arena_init(&c->ar_step, PT_MAX_NODES) ||
            !graph_arena_init(&c->ar_dec, PT_MAX_NODES) || !graph_arena_init(&c->ar_enc, PT_MAX_NODES)) {
            pt_throw("pt_init: graph arena allocation failed");
        }
        const PTHparams & hp = c->hp;
        pt_log(PT_LOG_INFO, "[Load] %s: flow %d layers x %d, head %d x %d, %d time conds, Mimi %d Hz / %.1f Hz",
               params->model_path, hp.n_layers, hp.dim, hp.head_depth, hp.head_dim, hp.n_time_conds, hp.sample_rate,
               hp.frame_rate);
        pt_log(PT_LOG_INFO, "[Load] Flash attention: %s, FP16 clamp: %s", c->opts.fa ? "on" : "off",
               c->opts.clamp_fp16 ? "on" : "off");
        pt_log(PT_LOG_INFO, "[Load] Ready in %.0f ms", t.ms());
        return c;
    } catch (const std::exception & e) {
        pt_set_error("%s", e.what());
        pt_log(PT_LOG_ERROR, "%s", e.what());
        gf_close(&gf);
        pt_free(c);
        return nullptr;
    }
}

int pt_sample_rate(const struct pt_context * c) {
    return c ? c->hp.sample_rate : 0;
}

struct pt_voice * pt_voice_from_audio(struct pt_context * c, const float * samples, int n_samples, int sample_rate) {
    if (!c || !samples || n_samples <= 0 || sample_rate <= 0) {
        pt_set_error("pt_voice_from_audio: invalid arguments");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(c->mu);
    const PTHparams &           hp = c->hp;
    Timer                       t;

    std::vector<float> pcm;
    if (sample_rate != hp.sample_rate) {
        int     n   = 0;
        float * res = audio_resample(samples, n_samples, sample_rate, hp.sample_rate, 1, &n);
        if (!res) {
            pt_set_error("pt_voice_from_audio: resample failed");
            return nullptr;
        }
        pcm.assign(res, res + n);
        free(res);
    } else {
        pcm.assign(samples, samples + n_samples);
    }
    const int n_pad = (int) ((pcm.size() + hp.frame_size - 1) / hp.frame_size * hp.frame_size);
    pcm.resize((size_t) n_pad, 0.0f);

    ggml_context * ctx;
    ggml_cgraph *  gf = pt_graph(c->ar_enc, &ctx);
    MimiEnc        e  = mimi_build_encode(ctx, gf, c->w, hp, n_pad, c->opts);
    if (!pt_alloc(c->ga_enc, gf, "voice encode")) {
        return nullptr;
    }
    mimi_fill_encode(e, hp, pcm.data(), n_pad);
    if (!pt_compute(c, gf, "voice encode")) {
        return nullptr;
    }
    const int          Tf = (int) e.out->ne[1];
    const int          L  = Tf + (hp.bos_before_voice ? 1 : 0);
    std::vector<float> prompt((size_t) hp.dim * L);
    if (hp.bos_before_voice) {
        memcpy(prompt.data(), c->bos_voice.data(), (size_t) hp.dim * sizeof(float));
    }
    ggml_backend_tensor_get(e.out, prompt.data() + (L - Tf) * (size_t) hp.dim, 0, (size_t) hp.dim * Tf * sizeof(float));
    const double t_enc = t.ms();

    if (!flow_kv_ensure(&c->kv, hp, c->backend, L)) {
        pt_set_error("pt_voice_from_audio: KV cache allocation failed");
        return nullptr;
    }
    if (!pt_prefill(c, prompt.data(), nullptr, L, 0, nullptr)) {
        return nullptr;
    }
    pt_voice * v      = new pt_voice();
    v->n_heads        = hp.n_heads;
    v->st.n_layers    = hp.n_layers;
    v->st.dim         = hp.dim;
    v->st.n           = L;
    const size_t rows = (size_t) L * hp.dim;
    v->st.k.resize(rows * hp.n_layers);
    v->st.v.resize(rows * hp.n_layers);
    for (int l = 0; l < hp.n_layers; l++) {
        ggml_backend_tensor_get(c->kv.k[l], v->st.k.data() + l * rows, 0, rows * sizeof(float));
        ggml_backend_tensor_get(c->kv.v[l], v->st.v.data() + l * rows, 0, rows * sizeof(float));
    }
    pt_log(PT_LOG_INFO, "[Voice] %.2f s of audio -> %d prompt frames (encode %.0f ms, prefill %.0f ms)",
           (double) n_samples / sample_rate, L, t_enc, t.ms() - t_enc);
    return v;
}

struct pt_voice * pt_voice_load(struct pt_context * c, const char * path, float max_sec) {
    if (!c || !path) {
        pt_set_error("pt_voice_load: invalid arguments");
        return nullptr;
    }
    const size_t n = strlen(path);
    if (n > 12 && strcmp(path + n - 12, ".safetensors") == 0) {
        pt_voice * v = new pt_voice();
        v->n_heads   = c->hp.n_heads;
        if (!voice_load_safetensors(&v->st, path, c->hp.n_layers, c->hp.dim)) {
            delete v;
            return nullptr;
        }
        pt_log(PT_LOG_INFO, "[Voice] %s: %d prompt frames", path, v->st.n);
        return v;
    }
    int     T   = 0;
    int     sr  = 0;
    float * raw = audio_read(path, &T, &sr);
    if (!raw) {
        pt_set_error("pt_voice_load: cannot read %s", path);
        return nullptr;
    }
    int     nm   = 0;
    float * mono = audio_mono_from_planar(raw, T, sr, sr, &nm);
    if (!mono) {
        pt_set_error("pt_voice_load: cannot convert %s", path);
        return nullptr;
    }
    if (max_sec > 0.0f && nm > (int) (max_sec * (float) sr)) {
        nm = (int) (max_sec * (float) sr);
    }
    pt_voice * v = pt_voice_from_audio(c, mono, nm, sr);
    free(mono);
    return v;
}

struct pt_voice * pt_voice_named(struct pt_context * c, const char * voices_path, const char * name) {
    if (!c || !voices_path || !name) {
        pt_set_error("pt_voice_named: invalid arguments");
        return nullptr;
    }
    pt_voice * v = new pt_voice();
    v->n_heads   = c->hp.n_heads;
    if (!voice_load_gguf(&v->st, voices_path, name, c->hp.n_layers, c->hp.dim)) {
        delete v;
        return nullptr;
    }
    pt_log(PT_LOG_INFO, "[Voice] %s: %d prompt frames", name, v->st.n);
    return v;
}

struct pt_voice * pt_voice_from_state(struct pt_context * c, const void * data, size_t size) {
    if (!c || !data) {
        pt_set_error("pt_voice_from_state: invalid arguments");
        return nullptr;
    }
    std::vector<uint8_t> buf((const uint8_t *) data, (const uint8_t *) data + size);
    pt_voice *           v = new pt_voice();
    v->n_heads             = c->hp.n_heads;
    if (!voice_parse_safetensors(&v->st, buf, "voice state", c->hp.n_layers, c->hp.dim)) {
        delete v;
        return nullptr;
    }
    return v;
}

int pt_voice_save(const struct pt_voice * v, const char * path) {
    if (!v || !path) {
        pt_set_error("pt_voice_save: invalid arguments");
        return PT_STATUS_INVALID_PARAMS;
    }
    return voice_save_safetensors(v->st, path, v->n_heads) ? PT_STATUS_OK : PT_STATUS_GENERATE_FAILED;
}

void pt_voice_free(struct pt_voice * v) {
    delete v;
}

void pt_tts_default_params(struct pt_tts_params * p) {
    *p                  = {};
    p->abi_version      = PT_ABI_VERSION;
    p->temperature      = -1.0f;
    p->lsd_steps        = 1;
    p->eos_threshold    = -4.0f;
    p->frames_after_eos = -1;
    p->max_chunk_tokens = 50;
    p->seed             = UINT32_MAX;
}

int pt_synthesize(struct pt_context * c, const struct pt_tts_params * p, struct pt_audio * out) {
    if (!c || !p || !p->text || !p->voice || (!out && !p->on_chunk)) {
        pt_set_error("pt_synthesize: missing context, params, text, voice or output");
        return PT_STATUS_INVALID_PARAMS;
    }
    if (p->abi_version < PT_ABI_MIN_VERSION || p->abi_version > PT_ABI_VERSION) {
        pt_set_error("pt_synthesize: abi_version %d outside [%d, %d]", p->abi_version, PT_ABI_MIN_VERSION,
                     PT_ABI_VERSION);
        return PT_STATUS_INVALID_PARAMS;
    }
    const PTHparams & hp = c->hp;
    const PTVoice &   vo = p->voice->st;
    if (vo.n_layers != hp.n_layers || vo.dim != hp.dim) {
        pt_set_error("pt_synthesize: voice built for another model");
        return PT_STATUS_INVALID_PARAMS;
    }
    if (p->lsd_steps < 1 || p->max_chunk_tokens < 1) {
        pt_set_error("pt_synthesize: lsd_steps and max_chunk_tokens must be >= 1");
        return PT_STATUS_INVALID_PARAMS;
    }
    if (out) {
        *out = {};
    }

    std::lock_guard<std::mutex> lock(c->mu);
    DebugDumper                 dump;
    debug_init(&dump, p->dump_dir);
    std::vector<float> dump_eos;
    std::vector<float> dump_latents;
    std::vector<float> dump_audio;
    if (dump.enabled) {
        const size_t rows = (size_t) vo.n * vo.dim;
        debug_dump_2d(&dump, "voice-k0", vo.k.data(), vo.n, vo.dim);
        debug_dump_2d(&dump, "voice-v-last", vo.v.data() + (size_t) (vo.n_layers - 1) * rows, vo.n, vo.dim);
    }
    const float    temp = p->temperature < 0.0f ? hp.temperature : p->temperature;
    const float    stdv = sqrtf(temp);
    const uint32_t seed = p->seed == UINT32_MAX ? std::random_device{}() : p->seed;
    TorchRng       rng(seed);

    std::vector<std::string> chunks = tp_chunks(hp, c->tok, p->text, p->max_chunk_tokens);
    if (chunks.empty()) {
        pt_set_error("pt_synthesize: empty text");
        return PT_STATUS_INVALID_PARAMS;
    }

    std::vector<float> all;
    std::vector<float> noise(hp.ldim);
    std::vector<float> latent(hp.ldim);
    std::vector<float> next(hp.ldim);
    std::vector<float> pending;
    std::vector<float> audio;
    int                frames_total = 0;
    double             t_steps      = 0.0;
    double             t_dec        = 0.0;
    Timer              t_all;
    bool               cancelled = false;

    auto emit = [&](const std::vector<float> & a) -> bool {
        if (out) {
            all.insert(all.end(), a.begin(), a.end());
        }
        if (dump.enabled) {
            dump_audio.insert(dump_audio.end(), a.begin(), a.end());
        }
        return !p->on_chunk || p->on_chunk(a.data(), (int) a.size(), p->user_data);
    };

    for (size_t ci = 0; ci < chunks.size() && !cancelled; ci++) {
        std::string text;
        int         guess = 1;
        tp_prepare(hp, chunks[ci], &text, &guess);
        std::vector<int>     ids = tok_encode(c->tok, text);
        std::vector<int32_t> ids32(ids.begin(), ids.end());
        const int            ntok    = (int) ids.size();
        const int            fae     = p->frames_after_eos >= 0 ? p->frames_after_eos :
                                       hp.frames_after_eos >= 0 ? hp.frames_after_eos :
                                                                  guess + 2;
        const int            max_gen = (int) std::ceil(((double) ntok / 3.0 + 2.0) * (double) hp.frame_rate);

        if (!flow_kv_ensure(&c->kv, hp, c->backend, vo.n + ntok + max_gen + 1)) {
            pt_set_error("pt_synthesize: KV cache allocation failed");
            return PT_STATUS_OOM;
        }
        pt_write_voice(c, vo);
        Timer              t_pre;
        const bool         dump0 = dump.enabled && ci == 0;
        std::vector<float> emb(dump0 ? (size_t) hp.dim * ntok : 0);
        if (!pt_prefill(c, nullptr, &ids32, ntok, vo.n, dump0 ? emb.data() : nullptr)) {
            return PT_STATUS_GENERATE_FAILED;
        }
        if (dump0) {
            debug_dump_i32_as_f32(&dump, "prompt-ids", ids32.data(), &ntok, 1);
            debug_dump_2d(&dump, "text-embed", emb.data(), ntok, hp.dim);
        }
        rng.normal(noise.data(), hp.ldim, stdv);
        const double ms_pre = t_pre.ms();

        mimi_state_reset(&c->ms);
        latent.assign(c->w.bos_emb.begin(), c->w.bos_emb.end());
        pending.clear();
        int n_past   = vo.n + ntok;
        int eos_step = -1;
        int block    = 1;
        int step     = 0;
        int frames   = 0;
        for (; step < max_gen; step++) {
            rng.normal(noise.data(), hp.ldim, stdv);
            float              eos = 0.0f;
            std::vector<float> hidden(dump0 && step == 0 ? hp.dim : 0);
            Timer              ts;
            if (!pt_step(c, latent.data(), noise.data(), p->lsd_steps, n_past, &eos, next.data(),
                         hidden.empty() ? nullptr : hidden.data())) {
                return PT_STATUS_GENERATE_FAILED;
            }
            t_steps += ts.ms();
            if (!hidden.empty()) {
                debug_dump_1d(&dump, "step0-hidden", hidden.data(), hp.dim);
                debug_dump_1d(&dump, "step0-latent", next.data(), hp.ldim);
            }
            if (dump.enabled) {
                dump_eos.push_back(eos);
            }
            n_past++;
            if (eos > p->eos_threshold && eos_step < 0 && step >= PT_EOS_MIN_FRAMES) {
                eos_step = step;
            }
            if (eos_step >= 0 && step >= eos_step + fae) {
                break;
            }
            pending.insert(pending.end(), next.begin(), next.end());
            if (dump.enabled) {
                dump_latents.insert(dump_latents.end(), next.begin(), next.end());
            }
            latent = next;
            frames++;
            if ((int) (pending.size() / hp.ldim) >= block) {
                Timer td;
                if (!pt_decode(c, pending.data(), block, audio)) {
                    return PT_STATUS_GENERATE_FAILED;
                }
                t_dec += td.ms();
                pending.clear();
                block = block * 2 > PT_DEC_MAX_FRAMES ? PT_DEC_MAX_FRAMES : block * 2;
                if (!emit(audio)) {
                    cancelled = true;
                    break;
                }
            }
        }
        if (step == max_gen) {
            pt_log(PT_LOG_WARN, "[TTS] chunk %zu reached %d frames without EOS", ci, max_gen);
        }
        if (!cancelled && !pending.empty()) {
            Timer td;
            if (!pt_decode(c, pending.data(), (int) (pending.size() / hp.ldim), audio)) {
                return PT_STATUS_GENERATE_FAILED;
            }
            t_dec += td.ms();
            if (!emit(audio)) {
                cancelled = true;
            }
        }
        frames_total += frames;
        pt_log(PT_LOG_INFO, "[TTS] chunk %zu/%zu: %d tokens, prefill %.1f ms, %d frames, EOS at %d", ci + 1,
               chunks.size(), ntok, ms_pre, frames, eos_step);
    }

    const double secs = (double) frames_total / hp.frame_rate;
    const double wall = t_all.ms();
    pt_log(PT_LOG_INFO,
           "[TTS] %.2f s of audio in %.0f ms (%.1fx real time), step %.2f ms/frame, decode %.2f ms/frame, seed %u",
           secs, wall, wall > 0.0 ? secs * 1000.0 / wall : 0.0, frames_total ? t_steps / frames_total : 0.0,
           frames_total ? t_dec / frames_total : 0.0, seed);

    if (dump.enabled) {
        debug_dump_1d(&dump, "eos", dump_eos.data(), (int) dump_eos.size());
        debug_dump_2d(&dump, "latents", dump_latents.data(), (int) (dump_latents.size() / hp.ldim), hp.ldim);
        debug_dump_1d(&dump, "output-audio", dump_audio.data(), (int) dump_audio.size());
    }
    if (cancelled) {
        pt_set_error("pt_synthesize: cancelled");
        return PT_STATUS_CANCELLED;
    }
    if (out) {
        out->samples = (float *) malloc(all.size() * sizeof(float));
        if (!out->samples && !all.empty()) {
            pt_set_error("pt_synthesize: out of memory");
            return PT_STATUS_OOM;
        }
        memcpy(out->samples, all.data(), all.size() * sizeof(float));
        out->n_samples   = (int) all.size();
        out->sample_rate = hp.sample_rate;
        out->channels    = 1;
    }
    return PT_STATUS_OK;
}

}  // extern "C"
