# pockettts.cpp architecture

## Model

Pocket TTS generates continuous latents instead of codec tokens. One
latent (32 values) covers 80 ms of audio (12.5 Hz).

```
voice prompt ----\
text tokens -----+--> flow LM (causal transformer, 6 or 24 layers, d 1024)
latent t-1 ------/        |
                          +--> out_norm --> out_eos --> EOS logit
                          +--> out_norm --> flow head (SimpleMLPAdaLN) + noise --> latent t
latents --> Mimi decoder --> 24 kHz PCM
```

- Sequence: `[voice prompt, text, latents]` as plain prefixes of one
  causal sequence, no cross attention. The voice prompt is the Mimi
  encoding of a recording through `speaker_proj`, after a learned BOS
  (`bos_before_voice`, all packs but english_2026-01). Text goes through
  a lookup table (4001 x 1024). The first AR input is `bos_emb`.
- Flow LM layer: pre LayerNorm (eps 1e-5), fused qkv, RoPE on
  interleaved pairs (theta 10000), full causal attention, GELU (tanh)
  MLP. No layer scale, no sliding window.
- Flow head, LSD decode with n steps (1 for the released packs):
  `x += head(c, s = i/n, t = (i+1)/n, x) / n`, starting from Gaussian
  noise of variance `temperature`. Two TimestepEmbedders averaged, each
  `[cos, sin](t * freqs) -> linear, SiLU, linear -> RMSNorm` with the
  unbiased variance of the reference. Six AdaLN residual blocks
  (LayerNorm eps 1e-6, shift / scale / gate), AdaLN final layer. A
  flow matching head (one time embedding, Euler) is supported as well.
- EOS: `out_eos(c) > -4`, ignored on the first 6 frames; generation
  stops `frames_after_eos` frames later (pack value, else 3 for at most
  4 words and 1 otherwise, plus 2).
- Mimi decoder: `latent * emb_std + emb_mean`, quantizer projection
  (32 -> 512), depthwise transposed conv upsample x16 (200 Hz), 2 layer
  transformer with a 250 position sliding window and layer scales,
  SEANet decoder (ratios 6, 5, 4, ELU, causal convs, one residual unit
  per stage).
- Mimi encoder (voice cloning): SEANet encoder (ratios 4, 5, 6), the
  windowed transformer, a replicate padded strided conv downsample x16,
  `speaker_proj` into the flow LM width.

## GGUF layout

`convert.py` reads the `kyutai/pocket-tts` snapshot and writes two GGUFs
per pack. The model, `general.architecture = pocket-tts`, keeps the
reference state_dict names verbatim; `flow_lm.bos_before_voice` is
squeezed to [1024] and the k=1 quantizer projection to 2D. Shapes come
from the tensors; the fixed Mimi settings and the text rules of each
language (from the reference configs) are tables in convert.py. The KV:

| key | content |
| --- | --- |
| `pocket.flow.*` | dim, n_layers, n_heads, ffn_dim, max_period, ldim, n_bins, bos_before_voice |
| `pocket.head.*` | dim, depth, n_time_conds (2 LSD, 1 flow matching) |
| `pocket.mimi.*` | sample_rate, frame_rate, SEANet shape, inner/outer dim, transformer shape and window |
| `pocket.text.*` | pad_with_spaces, remove_semicolons, append_terminal_punctuation, capitalize_first_letter, replace_from / replace_to |
| `pocket.gen.*` | frames_after_eos (-1 derived from the text), temperature |
| `tokenizer.ggml.*` | unigram pieces and log probabilities of tokenizer.json |

`quantize` only touches 2D matmul weights; conv kernels, the EOS row
and every vector stay F32.

The voices GGUF, `general.architecture = pocket-tts-voices`, holds per
predefined voice `<name>.k` and `<name>.v` F32 ne [dim, T, n_layers]: the
KV rows of the reference voice state file.

## Inference

- One backend per context (best device or `GGML_BACKEND`), no
  scheduler: every graph runs whole on it or fails.
- Attention: `ggml_flash_attn_ext` with an F32 accumulator on GPU
  backends, the explicit F32 chain on CPU and with `--no-fa`; masks are
  F16. `--clamp-fp16` clamps the qkv projection and the residual stream
  after each add to the FP16 range; the clamp runs on contiguous tensors
  since the CUDA clamp kernel ignores view strides.
- Weights: matmul weights keep their GGUF type, vectors load as F32,
  conv kernels as F16 (the ggml im2col path), transposed conv kernels
  as F32 permuted for GEMM + col2im.
- Flow LM KV cache: K (rotated) and V as rows of 1024 floats per
  position, one pair of [1024, capacity] tensors per layer. The voices
  GGUF and the voice state files store exactly these rows, so loading a
  voice is a copy.
- Synthesis: the text is prepared and chunked (<= 50 tokens). Each
  chunk writes the voice rows, prefills its text after them, then runs
  one step graph per frame. The step graph (input_linear, backbone,
  EOS, flow head) keeps a static shape across frames: the attended
  cache prefix is padded to 256 rows and masked, positions, cache
  indices and mask are inputs, and the graph is rebuilt in the same
  arena so CUDA graph replay applies.
- Mimi decoder: a persistent state buffer (conv tails, transposed conv
  carries, transformer windows) cleared at each chunk; latents decode
  in blocks of 1, 2, 4, 8 then 16 frames. The audio of the first frame of
  every chunk is dropped: the first latent after BOS carries an onset
  transient ahead of the speech, a click once the chunk follows audio.
- Noise: `torch-rng.h` reproduces `torch.manual_seed` + `normal_` on a
  CPU float tensor (MT19937, 24 bit uniforms, Box-Muller over blocks of
  16). Each text prefill draws one discarded noise vector like the
  reference forward.

## Validation

`tests/debug-tts-cossim.sh` (predefined voice) and
`tests/debug-clone-cossim.sh` (cloning `examples/freeman.wav`) run the
reference and `pocket-tts --dump` on the same text, voice and seed for
every backend and quant, and log one cossim per stage in
`tests/<case>-<backend>-<quant>.log`: prompt ids, text embeddings,
voice KV rows, the first step (hidden state and latent), then the EOS
logits, latents and audio over the whole utterance.

Against the reference Python on the same weights, same text, same seed:

- Text preparation, chunking and token ids: identical on 30 edge cases
  (curly quotes, French guillemets, decimals, ellipsis, emoji through
  byte fallback, long sentences split on commas) for 4 packs.
- Noise: within 1 ulp of torch.
- english_2026-09, CPU, F32: latents within 1e-3 of the reference
  frame by frame, same frame count and length. With an exact tanh GELU
  in place of the F16 table of ggml-cpu, 4e-6 and 65 dB of SNR on the
  waveform: the port is exact up to float rounding.
- CUDA runs the F32 GEMMs of the prefills in TF32, Vulkan in its own
  precision: about 1e-4 relative per frame. The AR loop is chaotic,
  trajectories drift apart after tens of frames (a few frames on some
  6 layer packs, whose reference itself loses words on the same seeds);
  frame counts usually match.
- Voice cloning: voice state within 1.6e-3 (CPU) and 3.6e-3 (CUDA) of
  the reference, dominated by the F16 conv kernels of the SEANet
  encoder.
- ASR (qwenasr 1.7B) on french_24l: F32 and BF16 transcripts identical
  to the reference transcript; Q8_0 and Q4_K_M lose a few words.

Speed on an RTX PRO 6000 (warm, one synthesis, real time factors):

| pack | backend | type | step ms/frame | x real time |
| --- | --- | --- | --- | --- |
| english_2026-09 (6L) | CUDA | BF16 | 0.55 | ~40 |
| english_2026-09 (6L) | Vulkan | F32 | 0.85 | ~73 |
| french_24l (24L) | CUDA | Q8_0 | 1.4 | ~40 |
| english_2026-09 (6L) | CPU 16 threads | Q8_0 | 1.4 | ~18 |

## Not converted

`languages/english_drifting_26-09` of the snapshot has no config in the
reference repository and no time embeddings in its flow head; the
reference cannot run it and neither does this port.
