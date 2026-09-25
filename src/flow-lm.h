#pragma once
// flow-lm.h: flow LM graphs and KV cache.
//
// The sequence is [voice prompt, text, latents]. The voice prompt (raw
// embeddings from the speaker projection, or a KV state loaded from file)
// and the text (lookup table embeddings) are prefilled; each AR step feeds
// one latent through input_linear and produces:
//   eos     out_eos(out_norm(h)), compared to the EOS threshold on the host
//   latent  flow head decode of Gaussian noise conditioned on out_norm(h)
//
// Flow head (SimpleMLPAdaLN), decoded in graph for a fixed step count n:
//   LSD           x += head(c, s = i/n, t = (i+1)/n, x) / n, two time embeddings averaged
//   flow matching x += head(c, t = i/n, x) / n, one time embedding
// TimestepEmbedder: [cos, sin](t * freqs) -> linear, SiLU, linear -> RMSNorm
// with the unbiased variance of the reference (x * alpha / sqrt(var + eps)).
//
// The KV cache holds K (rotated) and V as rows [C] per position, one pair of
// [C, capacity] tensors per layer. Voice state files store the same rows.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "model.h"
#include "transformer.h"

#include <vector>

struct FlowKV {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    std::vector<ggml_tensor *> k;
    std::vector<ggml_tensor *> v;
    int                        capacity = 0;
};

static void flow_kv_free(FlowKV * kv) {
    if (kv->buf) {
        ggml_backend_buffer_free(kv->buf);
    }
    if (kv->ctx) {
        ggml_free(kv->ctx);
    }
    *kv = FlowKV();
}

// Grows the cache to at least n rows (rounded up to 256). Contents are lost
// on growth: the caller rewrites the voice rows before every synthesis.
static bool flow_kv_ensure(FlowKV * kv, const PTHparams & hp, ggml_backend_t backend, int n) {
    if (n <= kv->capacity) {
        return true;
    }
    flow_kv_free(kv);
    const int               cap = (n + 255) / 256 * 256;
    struct ggml_init_params gp  = { ggml_tensor_overhead() * (size_t) (2 * hp.n_layers + 2), nullptr, true };
    kv->ctx                     = ggml_init(gp);
    for (int l = 0; l < hp.n_layers; l++) {
        kv->k.push_back(ggml_new_tensor_2d(kv->ctx, GGML_TYPE_F32, hp.dim, cap));
        kv->v.push_back(ggml_new_tensor_2d(kv->ctx, GGML_TYPE_F32, hp.dim, cap));
    }
    kv->buf = ggml_backend_alloc_ctx_tensors(kv->ctx, backend);
    if (!kv->buf) {
        pt_log(PT_LOG_ERROR, "[FlowLM] KV cache allocation failed (%d rows)", cap);
        flow_kv_free(kv);
        return false;
    }
    ggml_backend_buffer_clear(kv->buf, 0);
    kv->capacity = cap;
    pt_log(PT_LOG_INFO, "[FlowLM] KV cache: %d rows, %.1f MB", cap,
           (double) ggml_backend_buffer_get_size(kv->buf) / (1024.0 * 1024.0));
    return true;
}

// Inputs of one backbone pass over T positions starting at n_past.
struct FlowInputs {
    ggml_tensor * pos;   // I32 [T]
    ggml_tensor * idx;   // I64 [T]
    ggml_tensor * mask;  // F32 [n_kv, T]
};

static FlowInputs flow_inputs(ggml_context * ctx, int T, int n_kv) {
    FlowInputs in;
    in.pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    in.idx  = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    in.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T);
    ggml_set_name(in.pos, "pos");
    ggml_set_name(in.idx, "idx");
    ggml_set_name(in.mask, "mask");
    ggml_set_input(in.pos);
    ggml_set_input(in.idx);
    ggml_set_input(in.mask);
    return in;
}

// Causal mask of T new positions [n_past, n_past + T) over n_kv cache rows.
static void flow_fill_inputs(const FlowInputs & in, int n_past, int T, int n_kv) {
    std::vector<int32_t> pos(T);
    std::vector<int64_t> idx(T);
    std::vector<float>   mask((size_t) n_kv * T);
    for (int i = 0; i < T; i++) {
        pos[i] = n_past + i;
        idx[i] = n_past + i;
        for (int j = 0; j < n_kv; j++) {
            mask[(size_t) i * n_kv + j] = j <= n_past + i ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(in.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in.idx, idx.data(), 0, idx.size() * sizeof(int64_t));
    ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(float));
}

static ggml_tensor * flow_backbone(ggml_context *     ctx,
                                   ggml_cgraph *      gf,
                                   const PTWeights &  w,
                                   const PTHparams &  hp,
                                   const FlowKV &     kv,
                                   ggml_tensor *      x,
                                   const FlowInputs & in,
                                   int                n_kv) {
    for (int l = 0; l < hp.n_layers; l++) {
        x = pt_tfm_layer_cached(ctx, gf, w.layers[l], x, hp.n_heads, hp.max_period, in.pos, kv.k[l], kv.v[l], in.idx,
                                n_kv, in.mask);
    }
    return x;
}

static ggml_tensor * flow_time_embed(ggml_context * ctx, const TimeEmbed & te, float t) {
    ggml_tensor * args = ggml_scale(ctx, te.freqs, t);
    ggml_tensor * e    = ggml_concat(ctx, ggml_cos(ctx, args), ggml_sin(ctx, args), 0);
    e                  = pt_linear(ctx, te.l0_w, te.l0_b, e);
    e                  = pt_linear(ctx, te.l2_w, te.l2_b, ggml_silu(ctx, e));
    const int64_t n    = e->ne[0];
    ggml_tensor * d    = ggml_sub(ctx, e, ggml_mean(ctx, e));
    ggml_tensor * var  = ggml_scale_bias(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, d)), 1.0f / (float) (n - 1), 1e-5f);
    return ggml_div(ctx, ggml_mul(ctx, e, te.alpha), ggml_sqrt(ctx, var));
}

// x * (1 + scale) + shift
static ggml_tensor * flow_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
    return ggml_add(ctx, ggml_add(ctx, x, ggml_mul(ctx, x, scale)), shift);
}

static ggml_tensor * flow_head(ggml_context *    ctx,
                               const PTWeights & w,
                               const PTHparams & hp,
                               ggml_tensor *     c,
                               ggml_tensor *     x,
                               int               n_steps) {
    const int64_t D  = hp.head_dim;
    const size_t  es = sizeof(float);
    ggml_tensor * yc = pt_linear(ctx, w.cond_w, w.cond_b, c);
    for (int i = 0; i < n_steps; i++) {
        const float   s = (float) i / (float) n_steps;
        const float   t = (float) (i + 1) / (float) n_steps;
        ggml_tensor * y = yc;
        if (hp.n_time_conds == 2) {
            ggml_tensor * te = ggml_add(ctx, flow_time_embed(ctx, w.temb[0], s), flow_time_embed(ctx, w.temb[1], t));
            y                = ggml_add(ctx, y, ggml_scale(ctx, te, 0.5f));
        } else if (hp.n_time_conds == 1) {
            y = ggml_add(ctx, y, flow_time_embed(ctx, w.temb[0], s));
        }
        ggml_tensor * ys = ggml_silu(ctx, y);

        ggml_tensor * h = pt_linear(ctx, w.in_w, w.in_b, x);
        for (const ResBlock & b : w.blocks) {
            ggml_tensor * ada   = pt_linear(ctx, b.ada_w, b.ada_b, ys);
            ggml_tensor * shift = ggml_view_1d(ctx, ada, D, 0);
            ggml_tensor * scale = ggml_view_1d(ctx, ada, D, D * es);
            ggml_tensor * gate  = ggml_view_1d(ctx, ada, D, 2 * D * es);
            ggml_tensor * u     = flow_modulate(ctx, pt_layer_norm(ctx, h, b.ln_w, b.ln_b, 1e-6f), shift, scale);
            u                   = pt_linear(ctx, b.mlp0_w, b.mlp0_b, u);
            u                   = pt_linear(ctx, b.mlp2_w, b.mlp2_b, ggml_silu(ctx, u));
            h                   = ggml_add(ctx, h, ggml_mul(ctx, u, gate));
        }
        ggml_tensor * ada   = pt_linear(ctx, w.final_ada_w, w.final_ada_b, ys);
        ggml_tensor * shift = ggml_view_1d(ctx, ada, D, 0);
        ggml_tensor * scale = ggml_view_1d(ctx, ada, D, D * es);
        ggml_tensor * u     = flow_modulate(ctx, pt_layer_norm(ctx, h, nullptr, nullptr, 1e-6f), shift, scale);
        u                   = pt_linear(ctx, w.final_w, w.final_b, u);
        x                   = ggml_add(ctx, x, ggml_scale(ctx, u, 1.0f / (float) n_steps));
    }
    return x;
}

// One AR step: latent [ldim, 1] and noise [ldim, 1] in, the conditioning
// out_norm(h) [dim], eos [1] and the next latent [ldim] out.
struct FlowStep {
    ggml_tensor * latent;
    ggml_tensor * noise;
    FlowInputs    in;
    ggml_tensor * hidden;
    ggml_tensor * eos;
    ggml_tensor * out;
};

static FlowStep flow_build_step(ggml_context *    ctx,
                                ggml_cgraph *     gf,
                                const PTWeights & w,
                                const PTHparams & hp,
                                const FlowKV &    kv,
                                int               n_kv,
                                int               n_steps) {
    FlowStep st;
    st.latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.ldim, 1);
    st.noise  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.ldim, 1);
    ggml_set_name(st.latent, "latent");
    ggml_set_name(st.noise, "noise");
    ggml_set_input(st.latent);
    ggml_set_input(st.noise);
    st.in = flow_inputs(ctx, 1, n_kv);

    ggml_tensor * x = ggml_mul_mat(ctx, w.input_linear, st.latent);
    x               = flow_backbone(ctx, gf, w, hp, kv, x, st.in, n_kv);
    ggml_tensor * c = pt_layer_norm(ctx, x, w.out_norm_w, w.out_norm_b, 1e-5f);

    st.hidden = c;
    st.eos    = pt_linear(ctx, w.out_eos_w, w.out_eos_b, c);
    st.out    = flow_head(ctx, w, hp, c, st.noise, n_steps);
    ggml_set_output(st.hidden);
    ggml_set_output(st.eos);
    ggml_set_output(st.out);
    ggml_build_forward_expand(gf, st.eos);
    ggml_build_forward_expand(gf, st.out);
    return st;
}
