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
# <name>.v [n_layers, T, kv_dim] F32, the rows of the reference voice state.
# kv_dim = (attn_dim / n_heads) * num_kv_heads; equals dim when no GQA.

import argparse
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

HEAD_DIM = 64  # flow LM and Mimi attention heads (legacy; GQA uses attn_dim)
MAX_PERIOD = 10000.0
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
MIMI_CONTEXT = 250
TEMPERATURE = 0.3
EOS_THRESHOLD = -4.0

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


def read_pt(path):
    """Load a PyTorch checkpoint into an F32 numpy tensor dict.

    Handles the Pocket TTS int8 delivery format (fmt 'pocket-tts-zh-6l-int8-v1'):
    top-level {'fmt', 'flow_lm': {...}, 'mimi': {...}} where transformer linears
    are stored qint8 per-tensor as '<base>._packed_params._packed_params'
    (a tuple whose first element is the qint8 tensor) with '<base>.scale'/
    '.zero_point' companions. Those are dequantized to canonical '<base>.weight';
    all other tensors pass through as float32. The result matches the
    safetensors key layout expected by derive_arch().
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "flow_lm" not in ckpt:
        raise ValueError(f"{path}: unexpected checkpoint format (no 'flow_lm' group)")

    out = {}
    for group in ("flow_lm", "mimi"):
        entries = ckpt.get(group)
        if not isinstance(entries, dict):
            continue
        # Quantized linears: the qint8 tensor sits at '<base>._packed_params.
        # _packed_params' with '<base>.scale'/'.zero_point' companions (stored
        # as separate keys, so the packed tensor carries no attached quant
        # state). Dequantize manually to F32 under the canonical '<base>.weight'
        # name; drop all companion keys. Plain tensors pass through as F32.
        qbases = {k.split("._packed_params")[0] for k in entries
                  if "._packed_params" in k}
        for key, value in entries.items():
            if key.endswith("._packed_params._packed_params"):
                base = key[: -len("._packed_params._packed_params")]
                out[f"{group}.{base}.weight"] = \
                    _dequant_linear(entries, base).to(torch.float32).numpy()
            elif "._packed_params" in key:
                continue  # dtype companion of a quant linear
            elif any(key == b + ".scale" or key == b + ".zero_point"
                     for b in qbases):
                continue  # scale / zero_point companions of a quant linear
            elif isinstance(value, torch.Tensor):
                out[f"{group}.{key}"] = value.to(torch.float32).numpy()
    return out


def _dequant_linear(entries, base):
    """Reconstruct an F32 weight from an int8-quantized linear's parts.

    Prefers the attached quant state (tensor.dequantize()); otherwise falls back
    to manual affine dequantization from the separate '<base>.scale'/'.zero_point'
    keys (this is how the pocket-tts-zh-6l-int8-v1 format stores them).
    """
    import torch

    packed = entries[f"{base}._packed_params._packed_params"]
    packed = packed[0] if isinstance(packed, tuple) else packed
    if getattr(packed, "is_quantized", False):
        return packed.dequantize()
    scale = entries[f"{base}.scale"].to(torch.float32)
    zp_key = f"{base}.zero_point"
    zp = entries[zp_key].to(torch.int32) if zp_key in entries else 0
    q = packed.to(torch.int32)
    return (q - zp).to(torch.float32) * scale


def count(tensors, pattern):
    return len({m.group(1) for k in tensors for m in [re.match(pattern, k)] if m})


# --- config sourcing -------------------------------------------------------
# Most shapes come from the tensors. A few values they do not carry uniquely
# (the flow type when the head has no time conditions -- drifting and
# pointwise are both zero -- and the grouped-query attention geometry of the
# small models) live in the reference YAML configs, which ship with the Python
# package, not the HuggingFace model repo. load_pack_config resolves a config
# for a pack from (in order): an explicit path, <pack>.yaml beside the pack,
# or a directory of <pack>.yaml files. It returns the parsed mapping or None.

def _load_yaml(path):
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            f"[Convert] {path} is required for this model but PyYAML is not installed; "
            "run `pip install pyyaml` and retry"
        ) from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pack_config(pack, pack_dir, config_dir):
    cands = []
    if config_dir:
        cands.append(os.path.join(config_dir, pack + ".yaml"))
    beside = os.path.join(pack_dir, "config.yaml")
    if not os.path.exists(beside):
        beside = os.path.join(pack_dir, pack + ".yaml")
    cands.append(beside)
    for c in cands:
        if os.path.isfile(c):
            return _load_yaml(c)
    return None


# --- architecture derivation ------------------------------------------------

def derive_arch(t, cfg, pack):
    """Combine tensor shapes with the optional config into the GGUF flow keys.

    Returns a dict of scalar params plus n_time_conds. Raises SystemExit when a
    value is neither in the tensors nor supplied by a config.
    """
    dim = t["flow_lm.out_norm.weight"].shape[0]
    n_layers = count(t, r"flow_lm\.transformer\.layers\.(\d+)\.")
    ffn_dim = t["flow_lm.transformer.layers.0.linear1.weight"].shape[0]

    # flow type / time conditions
    n_time_conds = count(t, r"flow_lm\.flow_net\.time_embed\.(\d+)\.freqs")
    flow_type = None
    if cfg:
        flow_type = (cfg.get("flow_lm", {}).get("flow", {}) or {}).get("type")
    if n_time_conds == 2:
        flow_type = flow_type or "lsd"
    elif n_time_conds == 1:
        flow_type = flow_type or "flow_matching"
    else:  # drifting vs pointwise are indistinguishable from tensors alone
        if not flow_type:
            raise SystemExit(
                f"[Convert] FATAL {pack}: zero-time-condition head needs a config to choose "
                "drifting vs pointwise (pass --config-dir)"
            )
    assert flow_type in ("lsd", "flow_matching", "drifting", "pointwise"), \
        f"{pack}: unsupported flow type {flow_type}"

    # grouped-query attention geometry
    attn_dim = t["flow_lm.transformer.layers.0.self_attn.out_proj.weight"].shape[1]
    n_heads, num_kv_heads = None, None
    if cfg:
        tr = cfg.get("flow_lm", {}).get("transformer", {}) or {}
        n_heads = tr.get("num_heads")
        # null means "same as num_heads" (no GQA)
        num_kv_heads = tr.get("num_kv_heads") or n_heads
    # Legacy (no GQA) packs: square heads of HEAD_DIM, kv == q.
    if attn_dim == dim and n_heads is None:
        n_heads = dim // HEAD_DIM
        num_kv_heads = n_heads
    if n_heads is None or num_kv_heads is None:
        raise SystemExit(
            f"[Convert] FATAL {pack}: grouped-query attention needs num_heads/num_kv_heads "
            "from a config (pass --config-dir)"
        )
    kv_dim = attn_dim // n_heads * num_kv_heads
    in_proj_rows = t["flow_lm.transformer.layers.0.self_attn.in_proj.weight"].shape[0]
    if attn_dim + 2 * kv_dim != in_proj_rows:
        raise SystemExit(
            f"[Convert] FATAL {pack}: in_proj rows {in_proj_rows} != attn_dim + 2*kv_dim "
            f"({attn_dim} + 2*{kv_dim}); config/tensor mismatch"
        )

    # low-rank text embedding (proj present => rank-wide table)
    embed_rank = t["flow_lm.conditioner.embed.weight"].shape[1]
    if "flow_lm.conditioner.proj.weight" in t:
        proj_in = t["flow_lm.conditioner.proj.weight"].shape[1]
        if proj_in != embed_rank or t["flow_lm.conditioner.proj.weight"].shape[0] != dim:
            raise SystemExit(f"[Convert] FATAL {pack}: low-rank proj shape inconsistent")

    # z projection (present => backbone output is projected before eos/head)
    has_z_proj = "flow_lm.z_proj.weight" in t
    if cfg and not has_z_proj:
        z_dim_cfg = (cfg.get("flow_lm", {}) or {}).get("z_dim")
        if z_dim_cfg is not None:
            raise SystemExit(f"[Convert] FATAL {pack}: config sets z_dim but tensor missing")

    # generation defaults from the config when present
    temperature = float(cfg.get("default_temperature", TEMPERATURE)) if cfg else TEMPERATURE
    eos_threshold = float(cfg.get("default_eos_threshold", EOS_THRESHOLD)) if cfg else EOS_THRESHOLD

    return dict(
        dim=dim, n_layers=n_layers, ffn_dim=ffn_dim, flow_type=flow_type,
        n_time_conds=n_time_conds, attn_dim=attn_dim, n_heads=n_heads,
        num_kv_heads=num_kv_heads, kv_dim=kv_dim, embed_rank=embed_rank,
        has_z_proj=has_z_proj, temperature=temperature, eos_threshold=eos_threshold,
    )


def load_unigram_tokenizer(path):
    """Return (tokens, scores, unk_id) from a HuggingFace tokenizer.json or a
    SentencePiece .model file. Official packs ship the former; community repos
    usually ship the latter under an arbitrary filename."""
    if path.endswith(".json"):
        tok = json.load(open(path, encoding="utf-8"))["model"]
        assert tok["type"] == "Unigram" and tok.get("byte_fallback"), \
            f"{path}: expected byte fallback Unigram"
        tokens = [p for p, _ in tok["vocab"]]
        scores = [float(s) for _, s in tok["vocab"]]
        unk_id = int(tok.get("unk_id", 0))
    else:
        import sentencepiece as spm
        m = spm.SentencePieceProcessor()
        m.load(path)
        n = m.vocab_size()
        tokens = [m.id_to_piece(i) for i in range(n)]
        scores = [float(m.get_score(i)) for i in range(n)]
        unk_id = int(m.unk_id())
    return tokens, scores, unk_id


def convert_model(pack, pack_dir, out_path, config_dir=None, model_file=None,
                  tokenizer_path=None, config=None):
    src = model_file or os.path.join(pack_dir, "model.safetensors")
    # PyTorch int8 delivery format (e.g. PocketTTS-ZH-6L) is dequantized to F32;
    # everything else is read as safetensors.
    t = read_pt(src) if src.lower().endswith(".pt") else read_safetensors(src)
    if not np.any(t["mimi.encoder.model.0.conv.weight"]):
        print(f"[Convert] FATAL {pack}: zeroed Mimi encoder (voice cloning stripped mirror)")
        return 1

    cfg = config if config is not None else load_pack_config(pack, pack_dir, config_dir)
    arch = derive_arch(t, cfg, pack)

    tokens, scores, unk_id = load_unigram_tokenizer(
        tokenizer_path or os.path.join(pack_dir, "tokenizer.json"))

    lang = pack.split("_")[0]
    rules = TEXT_RULES.get(lang)
    if rules is None:  # community packs may use an unseen language key
        rules = dict(remove_semicolons=False, replace={})
    dim = arch["dim"]
    tfm_dim = t["mimi.decoder_transformer.transformer.layers.0.norm1.weight"].shape[0]
    n_dec = count(t, r"mimi\.decoder\.model\.(\d+)\.convtr\.weight")
    ratios = [t[f"mimi.decoder.model.{2 + 3 * i}.convtr.weight"].shape[2] // 2 for i in range(n_dec)]

    w = gguf.GGUFWriter(out_path, "pocket-tts")
    w.add_name(f"Pocket TTS {pack}")
    w.add_string("pocket.flow.type", arch["flow_type"])
    w.add_uint32("pocket.flow.dim", dim)
    w.add_uint32("pocket.flow.n_layers", arch["n_layers"])
    w.add_uint32("pocket.flow.n_heads", arch["n_heads"])
    w.add_uint32("pocket.flow.attn_dim", arch["attn_dim"])
    w.add_uint32("pocket.flow.num_kv_heads", arch["num_kv_heads"])
    w.add_uint32("pocket.flow.kv_dim", arch["kv_dim"])
    w.add_uint32("pocket.flow.ffn_dim", arch["ffn_dim"])
    w.add_float32("pocket.flow.max_period", MAX_PERIOD)
    w.add_uint32("pocket.flow.ldim", t["flow_lm.emb_mean"].shape[0])
    w.add_uint32("pocket.flow.n_bins", len(tokens))
    w.add_bool("pocket.flow.bos_before_voice", "flow_lm.bos_before_voice" in t)
    w.add_uint32("pocket.head.dim", t["flow_lm.flow_net.cond_embed.weight"].shape[0])
    w.add_uint32("pocket.head.depth", count(t, r"flow_lm\.flow_net\.res_blocks\.(\d+)\."))
    w.add_uint32("pocket.head.n_time_conds", arch["n_time_conds"])

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
    # Transformer head counts: the legacy assumption is HEAD_DIM = 64, which holds
    # for every encoder and for symmetric (full) models. Asymmetric decoder packs
    # (d_model 128 / num_heads 4 -> head dim 32) must take their counts from config.
    mimi_cfg = (cfg or {}).get("mimi", {}) or {}
    dec_tr_cfg = mimi_cfg.get("decoder_transformer") or {}
    enc_tr_cfg = mimi_cfg.get("transformer") or {}
    tfm_n_heads = int(dec_tr_cfg["num_heads"]) if "num_heads" in dec_tr_cfg else tfm_dim // HEAD_DIM
    enc_dim = t["mimi.encoder_transformer.transformer.layers.0.norm1.weight"].shape[0]
    enc_n_heads = int(enc_tr_cfg["num_heads"]) if "num_heads" in enc_tr_cfg else (enc_dim // HEAD_DIM)
    w.add_uint32("pocket.mimi.tfm.dim", tfm_dim)
    w.add_uint32("pocket.mimi.tfm.n_layers", count(t, r"mimi\.decoder_transformer\.transformer\.layers\.(\d+)\."))
    w.add_uint32("pocket.mimi.tfm.n_heads", tfm_n_heads)
    w.add_uint32("pocket.mimi.enc_tfm.n_heads", enc_n_heads)
    w.add_uint32("pocket.mimi.tfm.ffn_dim", t["mimi.decoder_transformer.transformer.layers.0.linear1.weight"].shape[0])
    w.add_uint32("pocket.mimi.tfm.context", MIMI_CONTEXT)
    # Decoder transformer context (asymmetric Mimi: decoder may use a smaller window).
    dec_ctx = int(dec_tr_cfg["context"]) if "context" in dec_tr_cfg else MIMI_CONTEXT
    w.add_uint32("pocket.mimi.dec_tfm.context", dec_ctx)
    w.add_float32("pocket.mimi.tfm.max_period", MAX_PERIOD)

    if cfg is not None:
        text = {
            "pad_with_spaces": bool(cfg.get("pad_with_spaces_for_short_inputs", False)),
            "remove_semicolons": bool(cfg.get("remove_semicolons", False)),
            "append_terminal_punctuation": bool(cfg.get("append_terminal_punctuation", True)),
            "capitalize_first_letter": bool(cfg.get("capitalize_first_letter", True)),
        }
        replace = cfg.get("replace_characters") or {}
    else:
        text = {
            "pad_with_spaces": pack in PAD_WITH_SPACES,
            "remove_semicolons": rules["remove_semicolons"],
            "append_terminal_punctuation": True,
            "capitalize_first_letter": True,
        }
        replace = rules["replace"]
    w.add_bool("pocket.text.pad_with_spaces", text["pad_with_spaces"])
    w.add_bool("pocket.text.remove_semicolons", text["remove_semicolons"])
    w.add_bool("pocket.text.append_terminal_punctuation", text["append_terminal_punctuation"])
    w.add_bool("pocket.text.capitalize_first_letter", text["capitalize_first_letter"])
    if replace:
        w.add_array("pocket.text.replace_from", list(replace.keys()))
        w.add_array("pocket.text.replace_to", list(replace.values()))
    fae = (cfg.get("model_recommended_frames_after_eos") if cfg else None)
    w.add_int32("pocket.gen.frames_after_eos", -1 if fae is None else int(fae))
    w.add_float32("pocket.gen.temperature", arch["temperature"])
    w.add_float32("pocket.gen.eos_threshold", arch["eos_threshold"])

    w.add_tokenizer_model("unigram")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_unk_token_id(unk_id)

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
    print(f"[Convert] {os.path.basename(out_path)}: {len(t)} tensors "
          f"({arch['flow_type']}, kv_dim={arch['kv_dim']}{'/z' if arch['has_z_proj'] else ''})")
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
    parser = argparse.ArgumentParser(description="Convert Pocket TTS safetensors to GGUF.")
    parser.add_argument("--config-dir", default=None,
                        help="Directory containing <pack>.yaml architecture configs. "
                             "Default: None (no config directory is selected automatically). "
                             "A config.yaml beside the checkpoint is used if present; "
                             "otherwise configs are optional and tensor-derived.")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR,
                        help="Directory containing checkpoint packs (default: %(default)s)")
    args = parser.parse_args()
    config_dir      = args.config_dir
    checkpoint_dir  = args.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        print(f"[Convert] FATAL: {checkpoint_dir}/ not found, run ./checkpoints.sh first")
        return 1
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rc = 0
    root = os.path.join(checkpoint_dir, "languages")
    for pack in sorted(os.listdir(root)):
        pack_dir = os.path.join(root, pack)
        model = os.path.join(OUTPUT_DIR, f"pocket-tts-{pack}-F32.gguf")
        voices = os.path.join(OUTPUT_DIR, f"pocket-tts-{pack}-voices.gguf")
        if os.path.exists(model):
            print(f"[Convert] skip {os.path.basename(model)}: exists")
        else:
            rc |= convert_model(pack, pack_dir, model, config_dir)
        if not os.path.exists(model):
            continue
        if os.path.exists(voices):
            print(f"[Convert] skip {os.path.basename(voices)}: exists")
        else:
            rc |= convert_voices(pack, pack_dir, voices)
    return rc


if __name__ == "__main__":
    sys.exit(main())
