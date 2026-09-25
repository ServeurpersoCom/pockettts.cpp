#!/usr/bin/env python3
# convert.py: Kyutai Pocket TTS checkpoint -> GGUF.
#
# Reads checkpoints/pocket-tts (the kyutai/pocket-tts snapshot) and writes,
# for every language pack under languages/:
#   models/pocket-tts-<pack>-F32.gguf     flow LM, flow head, Mimi, tokenizer
#   models/pocket-tts-<pack>-voices.gguf  the predefined voices of the pack
#
# Tensor names are the reference state_dict names verbatim; gguf reverses
# the numpy order into ggml ne (Linear [out, in] -> [in, out], Conv1d
# [OC, IC, K] -> [K, IC, OC], ConvTranspose1d [IC, OC, K] -> [K, OC, IC]).
# The voice BOS [1, 1, D] is squeezed to [D] and the k=1 quantizer
# projection [OC, IC, 1] to [OC, IC].
#
# Shapes come from the tensors. The few values they do not carry are the
# fixed Mimi settings and the text rules of each language, both taken from
# the reference configs (pocket_tts/config/*.yaml).
#
# A voice is the flow LM KV state after its prompt: per voice, <name>.k and
# <name>.v [n_layers, T, dim] F32, the rows of the reference voice state.

import json
import os
import re
import struct
import sys

import numpy as np

import gguf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "pocket-tts")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

HEAD_DIM = 64  # flow LM and Mimi attention heads
MAX_PERIOD = 10000.0
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
MIMI_CONTEXT = 250
TEMPERATURE = 0.3

# Text rules of pocket_tts/config/<pack>.yaml, keyed by language.
QUOTES = {'"': "", "\u201c": "", "\u201d": "", "\u201e": "", "\u00ab": "", "\u00bb": "", "\u2019": "'", "\u2018": "'"}
BRACKETS = {"(": "", ")": "", "[": "", "]": ""}
TEXT_RULES = {
    "english":    dict(remove_semicolons=False, replace={}),
    "french":     dict(remove_semicolons=True,  replace={**QUOTES, ":": ",", **BRACKETS}),
    "german":     dict(remove_semicolons=True,  replace={**QUOTES, **BRACKETS}),
    "dutch":      dict(remove_semicolons=False, replace={**QUOTES, **BRACKETS}),
    "italian":    dict(remove_semicolons=False, replace={**QUOTES, **BRACKETS}),
    "portuguese": dict(remove_semicolons=False, replace={**QUOTES, **BRACKETS}),
    "spanish":    dict(remove_semicolons=False, replace={**QUOTES, **BRACKETS, "\u00a1": "", "\u00bf": ""}),
}
PAD_WITH_SPACES = {"english_2026-01"}


def read_safetensors(path):
    # numpy has no bfloat16: BF16 widens to F32 through the high half of a u32.
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        raw = data[a:b]
        dt = info["dtype"]
        if dt == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "I64":
            arr = np.frombuffer(raw, dtype=np.int64)
        else:
            raise ValueError(f"{path}: unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(info["shape"]).copy()
    return out


def count(tensors, pattern):
    return len({m.group(1) for k in tensors for m in [re.match(pattern, k)] if m})


def convert_model(pack, pack_dir, out_path):
    t = read_safetensors(os.path.join(pack_dir, "model.safetensors"))
    n_time_conds = count(t, r"flow_lm\.flow_net\.time_embed\.(\d+)\.freqs")
    if n_time_conds != 2:
        print(f"[Convert] skip {pack}: flow head with {n_time_conds} time embeddings has no reference decode")
        return 0
    if not np.any(t["mimi.encoder.model.0.conv.weight"]):
        print(f"[Convert] FATAL {pack}: zeroed Mimi encoder (voice cloning stripped mirror)")
        return 1

    tok = json.load(open(os.path.join(pack_dir, "tokenizer.json"), encoding="utf-8"))["model"]
    assert tok["type"] == "Unigram" and tok.get("byte_fallback"), f"{pack}: expected byte fallback Unigram"

    lang = pack.split("_")[0]
    rules = TEXT_RULES[lang]
    dim = t["flow_lm.out_norm.weight"].shape[0]
    tfm_dim = t["mimi.decoder_transformer.transformer.layers.0.norm1.weight"].shape[0]
    n_dec = count(t, r"mimi\.decoder\.model\.(\d+)\.convtr\.weight")
    ratios = [t[f"mimi.decoder.model.{2 + 3 * i}.convtr.weight"].shape[2] // 2 for i in range(n_dec)]

    w = gguf.GGUFWriter(out_path, "pocket-tts")
    w.add_name(f"Pocket TTS {pack}")
    w.add_uint32("pocket.flow.dim", dim)
    w.add_uint32("pocket.flow.n_layers", count(t, r"flow_lm\.transformer\.layers\.(\d+)\."))
    w.add_uint32("pocket.flow.n_heads", dim // HEAD_DIM)
    w.add_uint32("pocket.flow.ffn_dim", t["flow_lm.transformer.layers.0.linear1.weight"].shape[0])
    w.add_float32("pocket.flow.max_period", MAX_PERIOD)
    w.add_uint32("pocket.flow.ldim", t["flow_lm.emb_mean"].shape[0])
    w.add_uint32("pocket.flow.n_bins", len(tok["vocab"]))
    w.add_bool("pocket.flow.bos_before_voice", "flow_lm.bos_before_voice" in t)
    w.add_uint32("pocket.head.dim", t["flow_lm.flow_net.cond_embed.weight"].shape[0])
    w.add_uint32("pocket.head.depth", count(t, r"flow_lm\.flow_net\.res_blocks\.(\d+)\."))
    w.add_uint32("pocket.head.n_time_conds", n_time_conds)

    w.add_uint32("pocket.mimi.sample_rate", SAMPLE_RATE)
    w.add_float32("pocket.mimi.frame_rate", FRAME_RATE)
    w.add_uint32("pocket.mimi.dimension", t["mimi.decoder.model.0.conv.weight"].shape[1])
    w.add_uint32("pocket.mimi.n_filters", t["mimi.encoder.model.0.conv.weight"].shape[0])
    w.add_uint32("pocket.mimi.n_residual_layers", 1)
    w.add_array("pocket.mimi.ratios", ratios)
    w.add_uint32("pocket.mimi.kernel_size", t["mimi.decoder.model.0.conv.weight"].shape[2])
    w.add_uint32("pocket.mimi.residual_kernel_size", t["mimi.decoder.model.3.block.1.conv.weight"].shape[2])
    w.add_uint32("pocket.mimi.last_kernel_size", t[f"mimi.decoder.model.{3 * n_dec + 2}.conv.weight"].shape[2])
    w.add_uint32("pocket.mimi.dilation_base", 2)
    w.add_uint32("pocket.mimi.compress", 2)
    w.add_uint32("pocket.mimi.inner_dim", t["mimi.downsample.conv.conv.weight"].shape[0])
    w.add_uint32("pocket.mimi.outer_dim", t["mimi.upsample.convtr.convtr.weight"].shape[0])
    w.add_uint32("pocket.mimi.tfm.dim", tfm_dim)
    w.add_uint32("pocket.mimi.tfm.n_layers", count(t, r"mimi\.decoder_transformer\.transformer\.layers\.(\d+)\."))
    w.add_uint32("pocket.mimi.tfm.n_heads", tfm_dim // HEAD_DIM)
    w.add_uint32("pocket.mimi.tfm.ffn_dim", t["mimi.decoder_transformer.transformer.layers.0.linear1.weight"].shape[0])
    w.add_uint32("pocket.mimi.tfm.context", MIMI_CONTEXT)
    w.add_float32("pocket.mimi.tfm.max_period", MAX_PERIOD)

    w.add_bool("pocket.text.pad_with_spaces", pack in PAD_WITH_SPACES)
    w.add_bool("pocket.text.remove_semicolons", rules["remove_semicolons"])
    w.add_bool("pocket.text.append_terminal_punctuation", True)
    w.add_bool("pocket.text.capitalize_first_letter", True)
    if rules["replace"]:
        w.add_array("pocket.text.replace_from", list(rules["replace"].keys()))
        w.add_array("pocket.text.replace_to", list(rules["replace"].values()))
    w.add_int32("pocket.gen.frames_after_eos", -1)
    w.add_float32("pocket.gen.temperature", TEMPERATURE)

    w.add_tokenizer_model("unigram")
    w.add_token_list([p for p, _ in tok["vocab"]])
    w.add_token_scores([float(s) for _, s in tok["vocab"]])
    w.add_unk_token_id(tok.get("unk_id", 0))

    for name in sorted(t):
        arr = t[name]
        if name == "flow_lm.bos_before_voice":
            arr = arr.reshape(-1)
        elif name == "mimi.quantizer.output_proj.weight":
            arr = arr.reshape(arr.shape[0], arr.shape[1])
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[Convert] {os.path.basename(out_path)}: {len(t)} tensors")
    return 0


def convert_voices(pack, pack_dir, out_path):
    vdir = os.path.join(pack_dir, "embeddings")
    w = gguf.GGUFWriter(out_path, "pocket-tts-voices")
    w.add_name(f"Pocket TTS {pack} voices")
    names = sorted(f[:-12] for f in os.listdir(vdir) if f.endswith(".safetensors"))
    for name in names:
        s = read_safetensors(os.path.join(vdir, name + ".safetensors"))
        n_layers = count(s, r"transformer\.layers\.(\d+)\.")
        k, v = [], []
        for l in range(n_layers):
            p = f"transformer.layers.{l}.self_attn/"
            n = int(s[p + "offset"][0])
            c = s[p + "cache"]  # [2, 1, T, H, hd]
            k.append(c[0, 0, :n].reshape(n, -1))
            v.append(c[1, 0, :n].reshape(n, -1))
        w.add_tensor(name + ".k", np.ascontiguousarray(np.stack(k)))
        w.add_tensor(name + ".v", np.ascontiguousarray(np.stack(v)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[Convert] {os.path.basename(out_path)}: {len(names)} voices")
    return 0


def main():
    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"[Convert] FATAL: {CHECKPOINT_DIR}/ not found, run ./checkpoints.sh first")
        return 1
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rc = 0
    root = os.path.join(CHECKPOINT_DIR, "languages")
    for pack in sorted(os.listdir(root)):
        pack_dir = os.path.join(root, pack)
        model = os.path.join(OUTPUT_DIR, f"pocket-tts-{pack}-F32.gguf")
        voices = os.path.join(OUTPUT_DIR, f"pocket-tts-{pack}-voices.gguf")
        if os.path.exists(model):
            print(f"[Convert] skip {os.path.basename(model)}: exists")
        else:
            rc |= convert_model(pack, pack_dir, model)
        if not os.path.exists(model):
            continue
        if os.path.exists(voices):
            print(f"[Convert] skip {os.path.basename(voices)}: exists")
        else:
            rc |= convert_voices(pack, pack_dir, voices)
    return rc


if __name__ == "__main__":
    sys.exit(main())
