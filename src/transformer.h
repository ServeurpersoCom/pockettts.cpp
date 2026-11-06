#pragma once
// transformer.h: StreamingTransformerLayer graph builders.
//
// x = x + ls1 * out_proj(attn(norm1(x)))
// x = x + ls2 * linear2(gelu_tanh(linear1(norm2(x))))
//
// Activations are channels-first [C, T]. in_proj packs q, k, v along the
// output dim: rows [0, attn_dim) hold queries (n_heads of hd), then
// [attn_dim, attn_dim+kv_dim) keys and values (num_kv_heads of hd each).
// With no GQA, attn_dim == kv_dim == C = n_heads * hd. RoPE rotates
// interleaved pairs (GGML_ROPE_TYPE_NORMAL) at absolute positions. Attention
// is ggml_flash_attn_ext with an F32 accumulator (native GQA), or the
// explicit F32 chain (mul_mat, masked softmax, mul_mat) on CPU and with
// --no-fa; the explicit path tiles K/V from num_kv_heads to n_heads.
// Masks are host built F16 [n_kv, T], 0 for allowed and -INF for masked
// keys. With clamp_fp16, the qkv projection and the residual stream after
// each add are clamped to the FP16 range (the qkv output is contiguous,
// the V views taken from it are not).
//
// Two key sources:
//   flow LM  linear KV cache [kv_dim, capacity] per layer, the new rows
//            written with set_rows at host given indices, attention over
//            the first n_kv rows (padded, masked past the live length)
//   Mimi     sliding window: keys = [previous tail, new rows], the tail of
//            the concatenation (last `window` rows) feeds the next call

#include "ggml-backend.h"
#include "ggml.h"
#include "model.h"

#include <cmath>
#include <vector>

static ggml_tensor * pt_layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w) {
        x = ggml_mul(ctx, x, w);
    }
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

static ggml_tensor * pt_clamp(ggml_context * ctx, ggml_tensor * x, const PTOpts & o) {
    return o.clamp_fp16 ? ggml_clamp(ctx, x, -65504.0f, 65504.0f) : x;
}

static ggml_tensor * pt_mask_new(ggml_context * ctx, int64_t n_kv, int64_t T) {
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T);
    ggml_set_input(m);
    return m;
}

static void pt_mask_upload(ggml_tensor * m, const float * mask) {
    std::vector<ggml_fp16_t> h((size_t) ggml_nelements(m));
    ggml_fp32_to_fp16_row(mask, h.data(), (int64_t) h.size());
    ggml_backend_tensor_set(m, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
}

static ggml_tensor * pt_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    x = ggml_mul_mat(ctx, w, x);
    if (b) {
        x = ggml_add(ctx, x, b);
    }
    return x;
}

struct PTQKV {
    ggml_tensor * q;  // [hd, T, n_heads] rotated, permuted view
    ggml_tensor * k;  // [kv_dim, T] rotated (== dim when no GQA)
    ggml_tensor * v;  // [kv_dim, T] view with the in_proj row stride
};

static PTQKV pt_qkv(ggml_context *   ctx,
                    const TfmLayer & L,
                    ggml_tensor *    h,
                    int              n_heads,
                    int              num_kv_heads,  // == n_heads when no GQA
                    int              attn_dim,     // total query head space
                    int              kv_dim,       // total KV head space
                    float            max_period,
                    ggml_tensor *    pos,
                    const PTOpts &   o) {
    const int64_t T = h->ne[1];
    const int64_t hd = attn_dim / n_heads;  // per-head dim (== kv_dim/num_kv_heads for GQA)
    const size_t es = ggml_element_size(h);

    // in_proj output: [attn_dim + 2*kv_dim, T]
    ggml_tensor * proj = pt_clamp(ctx, ggml_mul_mat(ctx, L.in_proj, h), o);
    // q: rows [0, attn_dim) -> [hd, n_heads, T]
    ggml_tensor * q = ggml_view_3d(ctx, proj, hd, n_heads, T, hd * es, proj->nb[1], 0);
    // k: rows [attn_dim, attn_dim+kv_dim) -> [hd, num_kv_heads, T]
    ggml_tensor * k = ggml_view_3d(ctx, proj, hd, num_kv_heads, T, hd * es, proj->nb[1], (int64_t) attn_dim * es);

    q = ggml_rope_ext(ctx, q, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, max_period, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, (int) hd, GGML_ROPE_TYPE_NORMAL, 0, max_period, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    PTQKV r;
    r.q = ggml_permute(ctx, q, 0, 2, 1, 3);            // [hd, T, n_heads]
    r.k = ggml_reshape_2d(ctx, k, kv_dim, T);          // [kv_dim, T]
    r.v = ggml_view_2d(ctx, proj, kv_dim, T, proj->nb[1], (int64_t)(attn_dim + kv_dim) * es);
    return r;
}

// q [hd, T, n_heads], keys and values [kv_dim, S] (rows of all KV heads), mask [S, T].
// GQA: num_kv_heads <= n_heads; query head i uses KV head i / (n_heads / num_kv_heads)
// (consecutive groups share a KV head, matching Python's repeat_interleave).
// Returns [attn_dim, T].

// CUDA FA kernels exist only for these per-head dims (ggml-cuda/fattn.cu).
// Head dims outside this set must use the explicit path.
static bool pt_fa_hd_supported(int64_t hd) {
    switch (hd) {
        case 40: case 64: case 72: case 80: case 96:
        case 112: case 128: case 192: case 256: case 320:
        case 512: case 576:
            return true;
        default:
            return false;
    }
}

static ggml_tensor * pt_attn(ggml_context * ctx,
                             ggml_tensor *  q,
                             ggml_tensor *  keys,
                             ggml_tensor *  vals,
                             int64_t        n_kv,
                             ggml_tensor *  mask,
                             int            n_heads,
                             int            num_kv_heads,
                             int            attn_dim,
                             const PTOpts & o) {
    const int64_t hd = q->ne[0];
    const int64_t T  = q->ne[1];
    const size_t es = ggml_element_size(keys);

    // View keys/vals as [hd, S, num_kv_heads]
    ggml_tensor * k = ggml_view_3d(ctx, keys, hd, n_kv, num_kv_heads, keys->nb[1], hd * es, 0);
    ggml_tensor * v = ggml_view_3d(ctx, vals, hd, n_kv, num_kv_heads, vals->nb[1], hd * es, 0);

    if (o.fa && pt_fa_hd_supported(hd)) {
        // Flash attention handles GQA natively (n_q != n_kv).
        ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, a, attn_dim, T);                     // [hd, n_heads, T]
    }

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                           // [S, T, H]
    kq               = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [S, hd, H]
    ggml_tensor * y  = ggml_mul_mat(ctx, vt, kq);                         // [hd, T, H]
    if (T > 1) {
        y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));             // [hd, H, T]
    }
    return ggml_reshape_2d(ctx, y, attn_dim, T);
}

static ggml_tensor * pt_ffn(ggml_context * ctx, const TfmLayer & L, ggml_tensor * x, const PTOpts & o) {
    ggml_tensor * h = pt_layer_norm(ctx, x, L.norm2_w, L.norm2_b, 1e-5f);
    h               = ggml_gelu(ctx, ggml_mul_mat(ctx, L.linear1, h));
    h               = ggml_mul_mat(ctx, L.linear2, h);
    if (L.ls2) {
        h = ggml_mul(ctx, h, L.ls2);
    }
    return pt_clamp(ctx, ggml_add(ctx, x, h), o);
}

// Flow LM layer over a linear KV cache. idx [T] I64 are the cache rows of
// the new positions, n_kv the attended prefix of the cache.
static ggml_tensor * pt_tfm_layer_cached(ggml_context *   ctx,
                                         ggml_cgraph *    gf,
                                         const TfmLayer & L,
                                         ggml_tensor *    x,
                                         int              n_heads,
                                         int              num_kv_heads,
                                         int              attn_dim,
                                         int              kv_dim,
                                         float            max_period,
                                         ggml_tensor *    pos,
                                         ggml_tensor *    cache_k,
                                         ggml_tensor *    cache_v,
                                         ggml_tensor *    idx,
                                         int64_t          n_kv,
                                         ggml_tensor *    mask,
                                         const PTOpts &   o) {
    ggml_tensor * h   = pt_layer_norm(ctx, x, L.norm1_w, L.norm1_b, 1e-5f);
    PTQKV         qkv = pt_qkv(ctx, L, h, n_heads, num_kv_heads, attn_dim, kv_dim, max_period, pos, o);

    ggml_tensor * ks = ggml_set_rows(ctx, cache_k, qkv.k, idx);
    ggml_tensor * vs = ggml_set_rows(ctx, cache_v, qkv.v, idx);
    ggml_build_forward_expand(gf, ks);
    ggml_build_forward_expand(gf, vs);

    ggml_tensor * a = pt_attn(ctx, qkv.q, ks, vs, n_kv, mask, n_heads, num_kv_heads, attn_dim, o);
    a               = ggml_mul_mat(ctx, L.out_proj, a);
    if (L.ls1) {
        a = ggml_mul(ctx, a, L.ls1);
    }
    x = pt_clamp(ctx, ggml_add(ctx, x, a), o);
    return pt_ffn(ctx, L, x, o);
}

// Mimi layer over a sliding window. prev_k / prev_v [C, P] are the rows of
// the previous positions (NULL when there are none); *tail_k / *tail_v
// receive the last `window` rows of [prev, new] (fewer when shorter).
static ggml_tensor * pt_tfm_layer_window(ggml_context *   ctx,
                                         const TfmLayer & L,
                                         ggml_tensor *    x,
                                         int              n_heads,
                                         int              num_kv_heads,
                                         int              attn_dim,
                                         int              kv_dim,
                                         float            max_period,
                                         ggml_tensor *    pos,
                                         ggml_tensor *    prev_k,
                                         ggml_tensor *    prev_v,
                                         ggml_tensor *    mask,
                                         int              window,
                                         ggml_tensor **   tail_k,
                                         ggml_tensor **   tail_v,
                                         const PTOpts &   o) {
    ggml_tensor * h   = pt_layer_norm(ctx, x, L.norm1_w, L.norm1_b, 1e-5f);
    PTQKV         qkv = pt_qkv(ctx, L, h, n_heads, num_kv_heads, attn_dim, kv_dim, max_period, pos, o);

    ggml_tensor * ks = prev_k ? ggml_concat(ctx, prev_k, qkv.k, 1) : qkv.k;
    ggml_tensor * vc = ggml_cont(ctx, qkv.v);
    ggml_tensor * vs = prev_v ? ggml_concat(ctx, prev_v, vc, 1) : vc;

    const int64_t S    = ks->ne[1];
    const int64_t keep = S < window ? S : window;
    *tail_k            = ggml_view_2d(ctx, ks, ks->ne[0], keep, ks->nb[1], (size_t) (S - keep) * ks->nb[1]);
    *tail_v            = ggml_view_2d(ctx, vs, vs->ne[0], keep, vs->nb[1], (size_t) (S - keep) * vs->nb[1]);

    ggml_tensor * a = pt_attn(ctx, qkv.q, ks, vs, S, mask, n_heads, num_kv_heads, attn_dim, o);
    a               = ggml_mul_mat(ctx, L.out_proj, a);
    if (L.ls1) {
        a = ggml_mul(ctx, a, L.ls1);
    }
    x = pt_clamp(ctx, ggml_add(ctx, x, a), o);
    return pt_ffn(ctx, L, x, o);
}

// Sliding window mask for T queries at absolute positions [p0, p0 + T)
// against P previous rows (positions [p0 - P, p0)) and the T new rows:
// key allowed when its position is >= 0 and 0 <= q - k < window.
static void pt_window_mask(float * mask, int P, int T, int p0, int window) {
    const int S = P + T;
    for (int i = 0; i < T; i++) {
        const int qp = p0 + i;
        for (int j = 0; j < S; j++) {
            const int kp             = p0 - P + j;
            const int d              = qp - kp;
            mask[(size_t) i * S + j] = (kp >= 0 && d >= 0 && d < window) ? 0.0f : -INFINITY;
        }
    }
}
