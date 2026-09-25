#pragma once
// mimi.h: Mimi codec graphs.
//
// Decoder (streaming, decode_from_latent): n latent frames per call,
//   latent * emb_std + emb_mean -> quantizer projection [outer, n]
//   -> depthwise transposed conv upsample (x upsample_stride)
//   -> windowed transformer -> SEANet decoder -> PCM [n * frame_size]
// Every causal conv, transposed conv and attention window carries its
// state across calls in a persistent buffer, so any split of a latent
// sequence into calls decodes to the same audio. The state is cleared at
// stream start.
//
// Encoder (offline, encode_to_latent + speaker projection): PCM padded to a
// multiple of frame_size -> SEANet encoder -> windowed transformer run in
// blocks of `context` positions -> replicate padded strided conv downsample
// -> speaker projection [dim, n_frames].

#include "causal-conv.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "model.h"
#include "transformer.h"

#include <vector>

struct MimiState {
    ggml_context *             ctx = nullptr;
    ggml_backend_buffer_t      buf = nullptr;
    ggml_tensor *              up_carry;
    std::vector<ggml_tensor *> kc;
    std::vector<ggml_tensor *> vc;
    ggml_tensor *              in_state;
    std::vector<ggml_tensor *> tr_carry;   // one per stage
    std::vector<ggml_tensor *> res_state;  // one per residual unit, stage major
    ggml_tensor *              out_state;
    int                        pos = 0;    // positions decoded at the encoder frame rate
};

static ggml_tensor * mimi_state_conv(ggml_context * ctx, const Conv & c) {
    const int pad = pt_conv_pad(c);
    return pad > 0 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pad, c.ic, 1) : nullptr;
}

static bool mimi_state_init(MimiState * s, const PTWeights & w, const PTHparams & hp, ggml_backend_t backend) {
    struct ggml_init_params gp = { ggml_tensor_overhead() * 64, nullptr, true };
    s->ctx                     = ggml_init(gp);
    const int K_up             = (int) w.upsample_w->ne[0];
    s->up_carry                = ggml_new_tensor_3d(s->ctx, GGML_TYPE_F32, K_up - hp.upsample_stride, hp.outer_dim, 1);
    for (int l = 0; l < hp.tfm_n_layers; l++) {
        s->kc.push_back(ggml_new_tensor_2d(s->ctx, GGML_TYPE_F32, hp.tfm_dim, hp.tfm_context));
        s->vc.push_back(ggml_new_tensor_2d(s->ctx, GGML_TYPE_F32, hp.tfm_dim, hp.tfm_context));
    }
    s->in_state = mimi_state_conv(s->ctx, w.dec_in);
    for (const SeanetStage & st : w.dec_stages) {
        s->tr_carry.push_back(
            ggml_new_tensor_3d(s->ctx, GGML_TYPE_F32, st.resample.k - st.resample.stride, st.resample.oc, 1));
        for (const SeanetRes & r : st.res) {
            s->res_state.push_back(mimi_state_conv(s->ctx, r.c1));
            if (pt_conv_pad(r.c2) > 0) {
                pt_throw("[Mimi] residual 1x1 conv with kernel %d", r.c2.k);
            }
        }
    }
    s->out_state = mimi_state_conv(s->ctx, w.dec_out);
    s->buf       = ggml_backend_alloc_ctx_tensors(s->ctx, backend);
    if (!s->buf) {
        pt_log(PT_LOG_ERROR, "[Mimi] decoder state allocation failed");
        return false;
    }
    ggml_backend_buffer_clear(s->buf, 0);
    s->pos = 0;
    return true;
}

static void mimi_state_reset(MimiState * s) {
    ggml_backend_buffer_clear(s->buf, 0);
    s->pos = 0;
}

static void mimi_state_free(MimiState * s) {
    if (s->buf) {
        ggml_backend_buffer_free(s->buf);
    }
    if (s->ctx) {
        ggml_free(s->ctx);
    }
    *s = MimiState();
}

struct MimiDec {
    ggml_tensor * latent;  // F32 [ldim, n]
    ggml_tensor * pos;     // I32 [n * upsample_stride]
    ggml_tensor * mask;    // F32 [context + n * upsample_stride, n * upsample_stride]
    ggml_tensor * audio;   // F32 [n * frame_size]
};

static MimiDec mimi_build_decode(ggml_context *    ctx,
                                 ggml_cgraph *     gf,
                                 const PTWeights & w,
                                 const PTHparams & hp,
                                 const MimiState & s,
                                 int               n) {
    const int T = n * hp.upsample_stride;
    MimiDec   d;
    d.latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.ldim, n);
    d.pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    d.mask   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.tfm_context + T, T);
    ggml_set_input(d.latent);
    ggml_set_input(d.pos);
    ggml_set_input(d.mask);

    ggml_tensor * x  = ggml_add(ctx, ggml_mul(ctx, d.latent, w.emb_std), w.emb_mean);
    x                = ggml_mul_mat(ctx, w.quant_proj, x);                                               // [outer, n]
    ggml_tensor * uw = ggml_reshape_3d(ctx, w.upsample_w, 1, w.upsample_w->ne[0], w.upsample_w->ne[2]);  // [1, K, C]
    x                = pt_convtr_dw_stream(ctx, gf, uw, x, hp.upsample_stride, s.up_carry);              // [T, C, 1]
    x                = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, T, hp.outer_dim)));    // [C, T]

    for (int l = 0; l < hp.tfm_n_layers; l++) {
        ggml_tensor * tk;
        ggml_tensor * tv;
        x = pt_tfm_layer_window(ctx, w.dec_layers[l], x, hp.tfm_n_heads, hp.tfm_max_period, d.pos, s.kc[l], s.vc[l],
                                d.mask, hp.tfm_context, &tk, &tv);
        ggml_build_forward_expand(gf, x);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, tk, s.kc[l]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, tv, s.vc[l]));
    }

    x         = ggml_cont(ctx, ggml_transpose(ctx, x));  // [T, C]
    x         = ggml_reshape_3d(ctx, x, T, hp.tfm_dim, 1);
    x         = pt_conv1d_stream(ctx, gf, w.dec_in, x, s.in_state);
    size_t ri = 0;
    for (size_t i = 0; i < w.dec_stages.size(); i++) {
        const SeanetStage & st = w.dec_stages[i];
        x                      = pt_convtr_stream(ctx, gf, st.resample, ggml_elu(ctx, x), s.tr_carry[i]);
        for (const SeanetRes & r : st.res) {
            ggml_tensor * v = pt_conv1d_stream(ctx, gf, r.c1, ggml_elu(ctx, x), s.res_state[ri++]);
            v               = pt_conv1d_stream(ctx, gf, r.c2, ggml_elu(ctx, v), nullptr);
            x               = ggml_add(ctx, x, v);
        }
    }
    x       = pt_conv1d_stream(ctx, gf, w.dec_out, ggml_elu(ctx, x), s.out_state);
    d.audio = ggml_reshape_1d(ctx, x, x->ne[0]);
    ggml_set_output(d.audio);
    ggml_build_forward_expand(gf, d.audio);
    return d;
}

static void mimi_fill_decode(const MimiDec &   d,
                             const PTHparams & hp,
                             const MimiState & s,
                             const float *     latents,
                             int               n) {
    const int T = n * hp.upsample_stride;
    ggml_backend_tensor_set(d.latent, latents, 0, (size_t) hp.ldim * n * sizeof(float));
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; i++) {
        pos[i] = s.pos + i;
    }
    ggml_backend_tensor_set(d.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<float> mask((size_t) (hp.tfm_context + T) * T);
    pt_window_mask(mask.data(), hp.tfm_context, T, s.pos, hp.tfm_context);
    ggml_backend_tensor_set(d.mask, mask.data(), 0, mask.size() * sizeof(float));
}

// Voice encoder: n_samples (a multiple of frame_size) -> [dim, n_samples / frame_size].
struct MimiEnc {
    ggml_tensor *              audio;
    ggml_tensor *              pos;
    std::vector<ggml_tensor *> masks;
    ggml_tensor *              out;
};

static MimiEnc mimi_build_encode(ggml_context *    ctx,
                                 ggml_cgraph *     gf,
                                 const PTWeights & w,
                                 const PTHparams & hp,
                                 int               n_samples) {
    MimiEnc e;
    e.audio = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_samples, 1, 1);
    ggml_set_input(e.audio);

    ggml_tensor * x = pt_conv1d(ctx, w.enc_in, e.audio, PT_PAD_ZERO);
    for (const SeanetStage & st : w.enc_stages) {
        for (const SeanetRes & r : st.res) {
            ggml_tensor * v = pt_conv1d(ctx, r.c1, ggml_elu(ctx, x), PT_PAD_ZERO);
            v               = pt_conv1d(ctx, r.c2, ggml_elu(ctx, v), PT_PAD_ZERO);
            x               = ggml_add(ctx, x, v);
        }
        x = pt_conv1d(ctx, st.resample, ggml_elu(ctx, x), PT_PAD_ZERO);
    }
    x            = pt_conv1d(ctx, w.enc_out, ggml_elu(ctx, x), PT_PAD_ZERO);                      // [Te, C, 1]
    const int Te = (int) x->ne[0];
    x            = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, Te, hp.tfm_dim)));  // [C, Te]

    e.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Te);
    ggml_set_input(e.pos);

    const int                  W = hp.tfm_context;
    std::vector<ggml_tensor *> pk(hp.tfm_n_layers, nullptr);
    std::vector<ggml_tensor *> pv(hp.tfm_n_layers, nullptr);
    ggml_tensor *              y = nullptr;
    for (int b0 = 0; b0 < Te; b0 += W) {
        const int     T = Te - b0 < W ? Te - b0 : W;
        const int     P = b0 == 0 ? 0 : W;
        ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P + T, T);
        ggml_set_input(m);
        e.masks.push_back(m);
        ggml_tensor * xb = ggml_view_2d(ctx, x, hp.tfm_dim, T, x->nb[1], (size_t) b0 * x->nb[1]);
        ggml_tensor * pb = ggml_view_1d(ctx, e.pos, T, (size_t) b0 * sizeof(int32_t));
        for (int l = 0; l < hp.tfm_n_layers; l++) {
            ggml_tensor * tk;
            ggml_tensor * tv;
            xb = pt_tfm_layer_window(ctx, w.enc_layers[l], xb, hp.tfm_n_heads, hp.tfm_max_period, pb, pk[l], pv[l], m,
                                     W, &tk, &tv);
            pk[l] = tk;
            pv[l] = tv;
        }
        y = y ? ggml_concat(ctx, y, xb, 1) : xb;
    }

    y            = ggml_cont(ctx, ggml_transpose(ctx, y));                                          // [Te, C]
    y            = ggml_reshape_3d(ctx, y, Te, hp.tfm_dim, 1);
    y            = pt_conv1d(ctx, w.downsample, y, PT_PAD_REPLICATE);                               // [Tf, inner, 1]
    const int Tf = (int) y->ne[0];
    y            = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, y, Tf, hp.inner_dim)));  // [inner, Tf]
    e.out        = ggml_mul_mat(ctx, w.speaker_proj, y);                                            // [dim, Tf]
    ggml_set_output(e.out);
    ggml_build_forward_expand(gf, e.out);
    return e;
}

static void mimi_fill_encode(const MimiEnc & e, const PTHparams & hp, const float * audio, int n_samples) {
    ggml_backend_tensor_set(e.audio, audio, 0, (size_t) n_samples * sizeof(float));
    const int            Te = (int) e.pos->ne[0];
    std::vector<int32_t> pos(Te);
    for (int i = 0; i < Te; i++) {
        pos[i] = i;
    }
    ggml_backend_tensor_set(e.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    const int W = hp.tfm_context;
    for (size_t b = 0; b < e.masks.size(); b++) {
        const int          P = (int) e.masks[b]->ne[0] - (int) e.masks[b]->ne[1];
        const int          T = (int) e.masks[b]->ne[1];
        std::vector<float> m((size_t) (P + T) * T);
        pt_window_mask(m.data(), P, T, (int) b * W, W);
        ggml_backend_tensor_set(e.masks[b], m.data(), 0, m.size() * sizeof(float));
    }
}
