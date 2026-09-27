#pragma once
// model.h: Pocket TTS hyperparameters and weights, loaded from one GGUF.
//
// Three parts share the file:
//   flow LM    causal transformer over [voice prompt, text, latents] with a
//              lookup table text conditioner, an EOS head and the flow head
//              (SimpleMLPAdaLN) that turns each hidden state into the next
//              continuous latent
//   Mimi dec   latent -> quantizer projection -> depthwise transposed conv
//              upsample -> windowed transformer -> SEANet decoder -> PCM
//   Mimi enc   PCM -> SEANet encoder -> windowed transformer -> strided conv
//              downsample -> speaker projection, for voice cloning
//
// Tensor names are the reference state_dict names. Matmul weights keep
// their GGUF type (quantizable), vectors load as F32, conv kernels as F16
// (the ggml im2col path), transposed conv kernels as F32 pre-permuted for
// the GEMM + col2im formulation.

#include "gguf-weights.h"
#include "pt-error.h"
#include "weight-ctx.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

// Graph options: fused flash attention (GPU backends) and FP16 range clamps
// on V and on the residual stream (CUDA before Ampere accumulates in FP16).
struct PTOpts {
    bool fa;
    bool clamp_fp16;
};

struct PTHparams {
    // flow LM
    int   dim;
    int   n_layers;
    int   n_heads;
    int   ffn_dim;
    float max_period;
    int   ldim;
    int   n_bins;
    bool  bos_before_voice;

    // flow head
    int head_dim;
    int head_depth;
    int n_time_conds;

    // Mimi
    int              sample_rate;
    float            frame_rate;
    int              mimi_dim;
    int              n_filters;
    int              n_residual_layers;
    std::vector<int> ratios;
    int              kernel_size;
    int              residual_kernel_size;
    int              last_kernel_size;
    int              dilation_base;
    int              compress;
    int              inner_dim;
    int              outer_dim;
    int              tfm_dim;
    int              tfm_n_layers;
    int              tfm_n_heads;
    int              tfm_ffn_dim;
    int              tfm_context;
    float            tfm_max_period;

    // text preparation and generation defaults of the pack
    bool                     pad_with_spaces;
    bool                     remove_semicolons;
    bool                     append_terminal_punctuation;
    bool                     capitalize_first_letter;
    std::vector<std::string> replace_from;
    std::vector<std::string> replace_to;
    int                      frames_after_eos;
    float                    temperature;

    // derived
    int hop_length;       // product of the SEANet ratios, samples per encoder frame
    int upsample_stride;  // encoder frames per latent frame
    int frame_size;       // samples per latent frame
};

// Transformer layer shared by the flow LM and both Mimi transformers:
// pre LayerNorm, fused qkv attention, GELU MLP, optional layer scales.
struct TfmLayer {
    ggml_tensor * norm1_w;
    ggml_tensor * norm1_b;
    ggml_tensor * norm2_w;
    ggml_tensor * norm2_b;
    ggml_tensor * in_proj;
    ggml_tensor * out_proj;
    ggml_tensor * linear1;
    ggml_tensor * linear2;
    ggml_tensor * ls1;
    ggml_tensor * ls2;
};

struct TimeEmbed {
    ggml_tensor * freqs;
    ggml_tensor * l0_w;
    ggml_tensor * l0_b;
    ggml_tensor * l2_w;
    ggml_tensor * l2_b;
    ggml_tensor * alpha;
};

struct ResBlock {
    ggml_tensor * ln_w;
    ggml_tensor * ln_b;
    ggml_tensor * mlp0_w;
    ggml_tensor * mlp0_b;
    ggml_tensor * mlp2_w;
    ggml_tensor * mlp2_b;
    ggml_tensor * ada_w;
    ggml_tensor * ada_b;
};

// Conv1d (w [K, IC, OC] F16) or ConvTranspose1d (w [IC, K*OC] F32).
struct Conv {
    ggml_tensor * w;
    ggml_tensor * b;
    int           k;
    int           stride;
    int           dilation;
    int           ic;
    int           oc;
};

// SEANet residual unit: ELU, conv k, ELU, conv 1, residual add.
struct SeanetRes {
    Conv c1;
    Conv c2;
};

// One SEANet stage: the strided conv (encoder) or transposed conv
// (decoder) and its residual units.
struct SeanetStage {
    Conv                   resample;
    std::vector<SeanetRes> res;
};

struct PTWeights {
    // flow LM
    ggml_tensor *         embed;
    ggml_tensor *         input_linear;
    ggml_tensor *         bos_before_voice;
    ggml_tensor *         speaker_proj;
    ggml_tensor *         emb_mean;
    ggml_tensor *         emb_std;
    ggml_tensor *         out_norm_w;
    ggml_tensor *         out_norm_b;
    ggml_tensor *         out_eos_w;
    ggml_tensor *         out_eos_b;
    std::vector<TfmLayer> layers;

    // flow head
    ggml_tensor *          cond_w;
    ggml_tensor *          cond_b;
    ggml_tensor *          in_w;
    ggml_tensor *          in_b;
    std::vector<TimeEmbed> temb;
    std::vector<ResBlock>  blocks;
    ggml_tensor *          final_ada_w;
    ggml_tensor *          final_ada_b;
    ggml_tensor *          final_w;
    ggml_tensor *          final_b;

    // Mimi decoder
    ggml_tensor *            quant_proj;
    ggml_tensor *            upsample_w;  // [1, K, C] depthwise transposed conv kernel
    std::vector<TfmLayer>    dec_layers;
    Conv                     dec_in;
    std::vector<SeanetStage> dec_stages;
    Conv                     dec_out;

    // Mimi encoder
    Conv                     enc_in;
    std::vector<SeanetStage> enc_stages;
    Conv                     enc_out;
    std::vector<TfmLayer>    enc_layers;
    Conv                     downsample;

    // host copies of the small vectors the pipeline reads
    std::vector<float> bos_emb;

    WeightCtx wctx;
};

static uint32_t pt_req_u32(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        pt_throw("[Model] missing GGUF key %s", key);
    }
    return gguf_get_val_u32(gf.gguf, idx);
}

static float pt_req_f32(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        pt_throw("[Model] missing GGUF key %s", key);
    }
    return gguf_get_val_f32(gf.gguf, idx);
}

static bool pt_req_bool(const GGUFModel & gf, const char * key) {
    int64_t idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        pt_throw("[Model] missing GGUF key %s", key);
    }
    return gguf_get_val_bool(gf.gguf, idx);
}

static std::vector<std::string> pt_str_array(const GGUFModel & gf, const char * key) {
    std::vector<std::string> out;
    int64_t                  idx = gguf_find_key(gf.gguf, key);
    if (idx < 0) {
        return out;
    }
    size_t n = gguf_get_arr_n(gf.gguf, idx);
    for (size_t i = 0; i < n; i++) {
        out.push_back(gguf_get_arr_str(gf.gguf, idx, i));
    }
    return out;
}

static void pt_load_hparams(PTHparams * hp, const GGUFModel & gf) {
    hp->dim              = (int) pt_req_u32(gf, "pocket.flow.dim");
    hp->n_layers         = (int) pt_req_u32(gf, "pocket.flow.n_layers");
    hp->n_heads          = (int) pt_req_u32(gf, "pocket.flow.n_heads");
    hp->ffn_dim          = (int) pt_req_u32(gf, "pocket.flow.ffn_dim");
    hp->max_period       = pt_req_f32(gf, "pocket.flow.max_period");
    hp->ldim             = (int) pt_req_u32(gf, "pocket.flow.ldim");
    hp->n_bins           = (int) pt_req_u32(gf, "pocket.flow.n_bins");
    hp->bos_before_voice = pt_req_bool(gf, "pocket.flow.bos_before_voice");
    hp->head_dim         = (int) pt_req_u32(gf, "pocket.head.dim");
    hp->head_depth       = (int) pt_req_u32(gf, "pocket.head.depth");
    hp->n_time_conds     = (int) pt_req_u32(gf, "pocket.head.n_time_conds");

    hp->sample_rate              = (int) pt_req_u32(gf, "pocket.mimi.sample_rate");
    hp->frame_rate               = pt_req_f32(gf, "pocket.mimi.frame_rate");
    hp->mimi_dim                 = (int) pt_req_u32(gf, "pocket.mimi.dimension");
    hp->n_filters                = (int) pt_req_u32(gf, "pocket.mimi.n_filters");
    hp->n_residual_layers        = (int) pt_req_u32(gf, "pocket.mimi.n_residual_layers");
    hp->kernel_size              = (int) pt_req_u32(gf, "pocket.mimi.kernel_size");
    hp->residual_kernel_size     = (int) pt_req_u32(gf, "pocket.mimi.residual_kernel_size");
    hp->last_kernel_size         = (int) pt_req_u32(gf, "pocket.mimi.last_kernel_size");
    hp->dilation_base            = (int) pt_req_u32(gf, "pocket.mimi.dilation_base");
    hp->compress                 = (int) pt_req_u32(gf, "pocket.mimi.compress");
    hp->inner_dim                = (int) pt_req_u32(gf, "pocket.mimi.inner_dim");
    hp->outer_dim                = (int) pt_req_u32(gf, "pocket.mimi.outer_dim");
    hp->tfm_dim                  = (int) pt_req_u32(gf, "pocket.mimi.tfm.dim");
    hp->tfm_n_layers             = (int) pt_req_u32(gf, "pocket.mimi.tfm.n_layers");
    hp->tfm_n_heads              = (int) pt_req_u32(gf, "pocket.mimi.tfm.n_heads");
    hp->tfm_ffn_dim              = (int) pt_req_u32(gf, "pocket.mimi.tfm.ffn_dim");
    hp->tfm_context              = (int) pt_req_u32(gf, "pocket.mimi.tfm.context");
    hp->tfm_max_period           = pt_req_f32(gf, "pocket.mimi.tfm.max_period");
    std::vector<uint32_t> ratios = gf_get_array_u32(gf, "pocket.mimi.ratios");
    if (ratios.empty()) {
        pt_throw("[Model] missing GGUF key pocket.mimi.ratios");
    }
    hp->ratios.assign(ratios.begin(), ratios.end());

    hp->pad_with_spaces             = pt_req_bool(gf, "pocket.text.pad_with_spaces");
    hp->remove_semicolons           = pt_req_bool(gf, "pocket.text.remove_semicolons");
    hp->append_terminal_punctuation = pt_req_bool(gf, "pocket.text.append_terminal_punctuation");
    hp->capitalize_first_letter     = pt_req_bool(gf, "pocket.text.capitalize_first_letter");
    hp->replace_from                = pt_str_array(gf, "pocket.text.replace_from");
    hp->replace_to                  = pt_str_array(gf, "pocket.text.replace_to");
    int64_t fae                     = gguf_find_key(gf.gguf, "pocket.gen.frames_after_eos");
    hp->frames_after_eos            = fae < 0 ? -1 : gguf_get_val_i32(gf.gguf, fae);
    hp->temperature                 = pt_req_f32(gf, "pocket.gen.temperature");

    hp->hop_length = 1;
    for (int r : hp->ratios) {
        hp->hop_length *= r;
    }
    hp->upsample_stride = (int) std::lround((float) hp->sample_rate / (float) hp->hop_length / hp->frame_rate);
    hp->frame_size      = hp->hop_length * hp->upsample_stride;
    if (hp->replace_from.size() != hp->replace_to.size()) {
        pt_throw("[Model] replace_from and replace_to differ in size");
    }
}

// ConvTranspose1d kernel, PyTorch (IC, OC, K) = ne [K, OC, IC], permuted to
// [IC, K*OC] with k fastest inside K*OC so mul_mat(w, x[IC, T]) yields the
// col2im column matrix.
static ggml_tensor * pt_load_convtr(WeightCtx * wctx, const GGUFModel & gf, const std::string & name) {
    ggml_tensor * src = ggml_get_tensor(gf.meta, name.c_str());
    if (!src) {
        pt_throw("[Model] tensor '%s' not found", name.c_str());
    }
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16) {
        pt_throw("[Model] '%s' expected F32, F16 or BF16, got %s", name.c_str(), ggml_type_name(src->type));
    }
    const int K  = (int) src->ne[0];
    const int OC = (int) src->ne[1];
    const int IC = (int) src->ne[2];

    ggml_tensor * dst = ggml_new_tensor_2d(wctx->ctx, GGML_TYPE_F32, IC, (int64_t) K * OC);
    ggml_set_name(dst, name.c_str());

    const void * raw = gf_get_data(gf, name.c_str());
    const size_t n   = (size_t) IC * K * OC;
    auto         buf = std::make_unique<float[]>(n);
    float *      out = buf.get();
    for (int ic = 0; ic < IC; ic++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                size_t i                             = ((size_t) ic * OC + oc) * K + k;
                float  v                             = src->type == GGML_TYPE_F32 ? ((const float *) raw)[i] :
                                                       src->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) raw)[i]) :
                                                                                    ggml_bf16_to_fp32(((const ggml_bf16_t *) raw)[i]);
                out[((size_t) oc * K + k) * IC + ic] = v;
            }
        }
    }
    wctx->pending.push_back({ dst, out, n * sizeof(float), 0 });
    wctx->staging.push_back(std::move(buf));
    return dst;
}

static Conv pt_load_conv(WeightCtx *         wctx,
                         const GGUFModel &   gf,
                         const std::string & prefix,
                         int                 stride,
                         int                 dilation,
                         bool                bias) {
    Conv c     = {};
    c.w        = gf_load_conv(wctx, gf, prefix + ".weight");
    c.b        = bias ? gf_load_tensor_f32(wctx, gf, prefix + ".bias") : nullptr;
    c.k        = (int) c.w->ne[0];
    c.ic       = (int) c.w->ne[1];
    c.oc       = (int) c.w->ne[2];
    c.stride   = stride;
    c.dilation = dilation;
    return c;
}

static Conv pt_load_convtr_layer(WeightCtx * wctx, const GGUFModel & gf, const std::string & prefix, int stride) {
    ggml_tensor * src = ggml_get_tensor(gf.meta, (prefix + ".weight").c_str());
    if (!src) {
        pt_throw("[Model] tensor '%s.weight' not found", prefix.c_str());
    }
    Conv c     = {};
    c.k        = (int) src->ne[0];
    c.oc       = (int) src->ne[1];
    c.ic       = (int) src->ne[2];
    c.stride   = stride;
    c.dilation = 1;
    c.w        = pt_load_convtr(wctx, gf, prefix + ".weight");
    c.b        = gf_load_tensor_f32(wctx, gf, prefix + ".bias");
    return c;
}

static TfmLayer pt_load_tfm_layer(WeightCtx * wctx, const GGUFModel & gf, const std::string & p, bool layer_scale) {
    TfmLayer l = {};
    l.norm1_w  = gf_load_tensor_f32(wctx, gf, p + ".norm1.weight");
    l.norm1_b  = gf_load_tensor_f32(wctx, gf, p + ".norm1.bias");
    l.norm2_w  = gf_load_tensor_f32(wctx, gf, p + ".norm2.weight");
    l.norm2_b  = gf_load_tensor_f32(wctx, gf, p + ".norm2.bias");
    l.in_proj  = gf_load_tensor(wctx, gf, p + ".self_attn.in_proj.weight");
    l.out_proj = gf_load_tensor(wctx, gf, p + ".self_attn.out_proj.weight");
    l.linear1  = gf_load_tensor(wctx, gf, p + ".linear1.weight");
    l.linear2  = gf_load_tensor(wctx, gf, p + ".linear2.weight");
    if (layer_scale) {
        l.ls1 = gf_load_tensor_f32(wctx, gf, p + ".layer_scale_1.scale");
        l.ls2 = gf_load_tensor_f32(wctx, gf, p + ".layer_scale_2.scale");
    }
    return l;
}

static SeanetRes pt_load_res(WeightCtx * wctx, const GGUFModel & gf, const std::string & p, int dilation) {
    SeanetRes r = {};
    r.c1        = pt_load_conv(wctx, gf, p + ".block.1.conv", 1, dilation, true);
    r.c2        = pt_load_conv(wctx, gf, p + ".block.3.conv", 1, 1, true);
    return r;
}

static std::vector<float> pt_host_f32(const GGUFModel & gf, const char * name) {
    ggml_tensor * t   = ggml_get_tensor(gf.meta, name);
    const void *  raw = gf_get_data(gf, name);
    if (!t || !raw) {
        pt_throw("[Model] tensor '%s' not found", name);
    }
    std::vector<float> out((size_t) ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw, out.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw, out.data(), (int64_t) out.size());
    } else if (t->type == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw, out.data(), (int64_t) out.size());
    } else {
        pt_throw("[Model] tensor '%s' has unsupported host type %s", name, ggml_type_name(t->type));
    }
    return out;
}

// SEANet module indices follow the reference nn.ModuleList: the decoder
// is [conv, (ELU, convtr, res * n)* per ratio, ELU, conv], the encoder
// [conv, (res * n, ELU, conv)* per reversed ratio, ELU, conv].
static void pt_load_weights(PTWeights * w, const PTHparams & hp, const GGUFModel & gf, ggml_backend_t backend) {
    wctx_init(&w->wctx, (int) gguf_get_n_tensors(gf.gguf) + 16);
    WeightCtx * wc = &w->wctx;

    w->embed            = gf_load_tensor(wc, gf, "flow_lm.conditioner.embed.weight");
    w->input_linear     = gf_load_tensor(wc, gf, "flow_lm.input_linear.weight");
    w->bos_before_voice = hp.bos_before_voice ? gf_load_tensor_f32(wc, gf, "flow_lm.bos_before_voice") : nullptr;
    w->speaker_proj     = gf_load_tensor(wc, gf, "flow_lm.speaker_proj_weight");
    w->emb_mean         = gf_load_tensor_f32(wc, gf, "flow_lm.emb_mean");
    w->emb_std          = gf_load_tensor_f32(wc, gf, "flow_lm.emb_std");
    w->out_norm_w       = gf_load_tensor_f32(wc, gf, "flow_lm.out_norm.weight");
    w->out_norm_b       = gf_load_tensor_f32(wc, gf, "flow_lm.out_norm.bias");
    w->out_eos_w        = gf_load_tensor_f32(wc, gf, "flow_lm.out_eos.weight");
    w->out_eos_b        = gf_load_tensor_f32(wc, gf, "flow_lm.out_eos.bias");
    for (int i = 0; i < hp.n_layers; i++) {
        w->layers.push_back(pt_load_tfm_layer(wc, gf, "flow_lm.transformer.layers." + std::to_string(i), false));
    }

    const std::string fh = "flow_lm.flow_net.";
    w->cond_w            = gf_load_tensor(wc, gf, fh + "cond_embed.weight");
    w->cond_b            = gf_load_tensor_f32(wc, gf, fh + "cond_embed.bias");
    w->in_w              = gf_load_tensor(wc, gf, fh + "input_proj.weight");
    w->in_b              = gf_load_tensor_f32(wc, gf, fh + "input_proj.bias");
    for (int i = 0; i < hp.n_time_conds; i++) {
        const std::string p = fh + "time_embed." + std::to_string(i);
        TimeEmbed         t = {};
        t.freqs             = gf_load_tensor_f32(wc, gf, p + ".freqs");
        t.l0_w              = gf_load_tensor(wc, gf, p + ".mlp.0.weight");
        t.l0_b              = gf_load_tensor_f32(wc, gf, p + ".mlp.0.bias");
        t.l2_w              = gf_load_tensor(wc, gf, p + ".mlp.2.weight");
        t.l2_b              = gf_load_tensor_f32(wc, gf, p + ".mlp.2.bias");
        t.alpha             = gf_load_tensor_f32(wc, gf, p + ".mlp.3.alpha");
        w->temb.push_back(t);
    }
    for (int i = 0; i < hp.head_depth; i++) {
        const std::string p = fh + "res_blocks." + std::to_string(i);
        ResBlock          b = {};
        b.ln_w              = gf_load_tensor_f32(wc, gf, p + ".in_ln.weight");
        b.ln_b              = gf_load_tensor_f32(wc, gf, p + ".in_ln.bias");
        b.mlp0_w            = gf_load_tensor(wc, gf, p + ".mlp.0.weight");
        b.mlp0_b            = gf_load_tensor_f32(wc, gf, p + ".mlp.0.bias");
        b.mlp2_w            = gf_load_tensor(wc, gf, p + ".mlp.2.weight");
        b.mlp2_b            = gf_load_tensor_f32(wc, gf, p + ".mlp.2.bias");
        b.ada_w             = gf_load_tensor(wc, gf, p + ".adaLN_modulation.1.weight");
        b.ada_b             = gf_load_tensor_f32(wc, gf, p + ".adaLN_modulation.1.bias");
        w->blocks.push_back(b);
    }
    w->final_ada_w = gf_load_tensor(wc, gf, fh + "final_layer.adaLN_modulation.1.weight");
    w->final_ada_b = gf_load_tensor_f32(wc, gf, fh + "final_layer.adaLN_modulation.1.bias");
    w->final_w     = gf_load_tensor(wc, gf, fh + "final_layer.linear.weight");
    w->final_b     = gf_load_tensor_f32(wc, gf, fh + "final_layer.linear.bias");

    // Mimi decoder
    w->quant_proj = gf_load_tensor(wc, gf, "mimi.quantizer.output_proj.weight");
    w->upsample_w = gf_load_tensor_f32(wc, gf, "mimi.upsample.convtr.convtr.weight");
    for (int i = 0; i < hp.tfm_n_layers; i++) {
        w->dec_layers.push_back(
            pt_load_tfm_layer(wc, gf, "mimi.decoder_transformer.transformer.layers." + std::to_string(i), true));
    }
    int idx   = 0;
    w->dec_in = pt_load_conv(wc, gf, "mimi.decoder.model." + std::to_string(idx++) + ".conv", 1, 1, true);
    for (int r : hp.ratios) {
        idx++;  // ELU
        SeanetStage st = {};
        st.resample    = pt_load_convtr_layer(wc, gf, "mimi.decoder.model." + std::to_string(idx++) + ".convtr", r);
        for (int j = 0; j < hp.n_residual_layers; j++) {
            int d = 1;
            for (int e = 0; e < j; e++) {
                d *= hp.dilation_base;
            }
            st.res.push_back(pt_load_res(wc, gf, "mimi.decoder.model." + std::to_string(idx++), d));
        }
        w->dec_stages.push_back(st);
    }
    idx++;  // ELU
    w->dec_out = pt_load_conv(wc, gf, "mimi.decoder.model." + std::to_string(idx) + ".conv", 1, 1, true);

    // Mimi encoder
    idx       = 0;
    w->enc_in = pt_load_conv(wc, gf, "mimi.encoder.model." + std::to_string(idx++) + ".conv", 1, 1, true);
    for (auto it = hp.ratios.rbegin(); it != hp.ratios.rend(); ++it) {
        SeanetStage st = {};
        for (int j = 0; j < hp.n_residual_layers; j++) {
            int d = 1;
            for (int e = 0; e < j; e++) {
                d *= hp.dilation_base;
            }
            st.res.push_back(pt_load_res(wc, gf, "mimi.encoder.model." + std::to_string(idx++), d));
        }
        idx++;  // ELU
        st.resample = pt_load_conv(wc, gf, "mimi.encoder.model." + std::to_string(idx++) + ".conv", *it, 1, true);
        w->enc_stages.push_back(st);
    }
    idx++;  // ELU
    w->enc_out = pt_load_conv(wc, gf, "mimi.encoder.model." + std::to_string(idx) + ".conv", 1, 1, true);
    for (int i = 0; i < hp.tfm_n_layers; i++) {
        w->enc_layers.push_back(
            pt_load_tfm_layer(wc, gf, "mimi.encoder_transformer.transformer.layers." + std::to_string(i), true));
    }
    w->downsample = pt_load_conv(wc, gf, "mimi.downsample.conv.conv", hp.upsample_stride, 1, false);

    w->bos_emb = pt_host_f32(gf, "flow_lm.bos_emb");

    if (!wctx_alloc(wc, backend)) {
        pt_throw("[Model] weight allocation failed");
    }
}

static void pt_free_weights(PTWeights * w) {
    wctx_free(&w->wctx);
}
