#pragma once
// causal-conv.h: causal Conv1d and ConvTranspose1d graph primitives of the
// Mimi SEANet (StreamingConv1d / StreamingConvTranspose1d).
//
// Activations are T-first [T, C, 1] (the ggml_conv_1d input layout).
//
// Conv1d: left padding of (k - 1) * d + 1 - stride frames, zeros or the
// replicated first frame. The offline form pads in graph; the streaming
// form (stride 1) concats a persistent state holding the previous input
// tail and refreshes it in graph.
//
// ConvTranspose1d: GEMM + col2im. The column matrix [K*OC, T] (k fastest)
// scatters to (T - 1) * stride + K rows; the first K - stride rows receive
// the tail carried from the previous call, the last K - stride rows become
// the next carry (bias free), and T * stride rows are emitted.

#include "ggml.h"
#include "model.h"

enum PTPadMode {
    PT_PAD_ZERO      = 0,
    PT_PAD_REPLICATE = 1,
};

static int pt_conv_pad(const Conv & c) {
    return (c.k - 1) * c.dilation + 1 - c.stride;
}

// Offline causal conv, T a multiple of c.stride. Returns [T / stride, OC, 1].
static ggml_tensor * pt_conv1d(ggml_context * ctx, const Conv & c, ggml_tensor * x, PTPadMode mode) {
    const int pad = pt_conv_pad(c);
    if (pad > 0) {
        if (mode == PT_PAD_REPLICATE) {
            ggml_tensor * first = ggml_view_3d(ctx, x, 1, x->ne[1], 1, x->nb[1], x->nb[2], 0);
            ggml_tensor * shape = ggml_new_tensor_3d(ctx, x->type, pad, x->ne[1], 1);
            x                   = ggml_concat(ctx, ggml_repeat(ctx, first, shape), x, 0);
        } else {
            x = ggml_pad_ext(ctx, x, pad, 0, 0, 0, 0, 0, 0, 0);
        }
    }
    ggml_tensor * y = ggml_conv_1d(ctx, c.w, x, c.stride, 0, c.dilation);
    if (c.b) {
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, c.b, 1, c.oc, 1));
    }
    return y;
}

// Streaming causal conv, stride 1. state [pad, IC, 1] holds the last pad
// input rows of the previous call (zero at stream start). Returns [T, OC, 1].
static ggml_tensor * pt_conv1d_stream(ggml_context * ctx,
                                      ggml_cgraph *  gf,
                                      const Conv &   c,
                                      ggml_tensor *  x,
                                      ggml_tensor *  state) {
    ggml_tensor * xe = x;
    if (state) {
        const int64_t L = state->ne[0];
        xe              = ggml_concat(ctx, state, x, 0);
        ggml_tensor * tail =
            ggml_view_3d(ctx, xe, L, xe->ne[1], 1, xe->nb[1], xe->nb[2], (size_t) (xe->ne[0] - L) * xe->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, state));
    }
    ggml_tensor * y = ggml_conv_1d(ctx, c.w, xe, 1, 0, c.dilation);
    if (c.b) {
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, c.b, 1, c.oc, 1));
    }
    return y;
}

// Streaming transposed conv from its column matrix col [K*OC, T].
// carry [K - stride, OC, 1]. Returns [T * stride, OC, 1].
static ggml_tensor * pt_convtr_stream_col(ggml_context * ctx,
                                          ggml_cgraph *  gf,
                                          ggml_tensor *  col,
                                          int            stride,
                                          int            oc,
                                          ggml_tensor *  bias,
                                          ggml_tensor *  carry) {
    const int64_t T    = col->ne[1];
    const int64_t trim = carry->ne[0];
    const int64_t emit = T * stride;

    ggml_tensor * raw = ggml_col2im_1d(ctx, col, stride, oc, 0);  // [emit + trim, OC]
    raw               = ggml_reshape_3d(ctx, raw, raw->ne[0], oc, 1);

    ggml_tensor * head = ggml_view_3d(ctx, raw, trim, oc, 1, raw->nb[1], raw->nb[2], 0);
    ggml_tensor * y    = ggml_add(ctx, head, carry);
    if (emit > trim) {
        ggml_tensor * mid =
            ggml_view_3d(ctx, raw, emit - trim, oc, 1, raw->nb[1], raw->nb[2], (size_t) trim * raw->nb[0]);
        y = ggml_concat(ctx, y, mid, 0);
    }
    ggml_build_forward_expand(gf, y);

    ggml_tensor * tail = ggml_view_3d(ctx, raw, trim, oc, 1, raw->nb[1], raw->nb[2], (size_t) emit * raw->nb[0]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tail, carry));

    if (bias) {
        y = ggml_add(ctx, y, ggml_reshape_3d(ctx, bias, 1, oc, 1));
    }
    return y;
}

// Streaming ConvTranspose1d on T-first x [T, IC, 1].
static ggml_tensor * pt_convtr_stream(ggml_context * ctx,
                                      ggml_cgraph *  gf,
                                      const Conv &   c,
                                      ggml_tensor *  x,
                                      ggml_tensor *  carry) {
    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1])));  // [IC, T]
    ggml_tensor * col = ggml_mul_mat(ctx, c.w, xt);                                                        // [K*OC, T]
    return pt_convtr_stream_col(ctx, gf, col, c.stride, c.oc, c.b, carry);
}

// Streaming depthwise ConvTranspose1d (groups = channels, no bias) on
// channels-first x [C, T]. w [1, K, C]: one batched outer product per
// channel gives [K, T, C], reordered to the col2im layout [K*C, T].
static ggml_tensor * pt_convtr_dw_stream(ggml_context * ctx,
                                         ggml_cgraph *  gf,
                                         ggml_tensor *  w,
                                         ggml_tensor *  x,
                                         int            stride,
                                         ggml_tensor *  carry) {
    const int64_t C   = x->ne[0];
    const int64_t T   = x->ne[1];
    const int64_t K   = w->ne[1];
    ggml_tensor * xtc = ggml_cont(ctx, ggml_transpose(ctx, x));                    // [T, C]
    ggml_tensor * col = ggml_mul_mat(ctx, w, ggml_reshape_3d(ctx, xtc, 1, T, C));  // [K, T, C]
    col               = ggml_cont(ctx, ggml_permute(ctx, col, 0, 2, 1, 3));        // [K, C, T]
    col               = ggml_reshape_2d(ctx, col, K * C, T);
    return pt_convtr_stream_col(ctx, gf, col, stride, (int) C, nullptr, carry);
}
