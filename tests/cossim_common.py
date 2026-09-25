"""Shared helpers for the pockettts.cpp cossim debug scripts.

Provides the reference loader (kyutai-labs/pocket-tts reading the local
checkpoints/pocket-tts snapshot), the hooks that dump the reference
intermediates, dump load and save, and the metric helpers used by the
debug-tts / debug-clone cossim scripts.

Importing this module forces TF32 off on every torch CUDA matmul path so
Python results stay bit comparable across runs and machines.
"""

import os
import pathlib
import struct
import sys

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32                             = False
torch.backends.cudnn.allow_tf32                                   = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision("highest")

UPSTREAM_ROOT = "../../pocket-tts"
CKPT          = "../checkpoints/pocket-tts"
BIN           = "../build/pocket-tts"
sys.path.insert(0, UPSTREAM_ROOT)

import pocket_tts.models.tts_model as tts_model
import pocket_tts.utils.utils as utils
from pocket_tts import TTSModel


def local_path(uri, *args, **kwargs):
    """hf://<org>/<repo>/<path>@<rev> -> CKPT/<path>: the gated repo and the
    no voice cloning mirror share one tree layout."""
    s = str(uri)
    if s.startswith("hf://"):
        s = s.split("@", 1)[0][len("hf://"):]
        return pathlib.Path(CKPT, *s.split("/")[2:])
    return pathlib.Path(s)


utils.download_if_necessary      = local_path
tts_model.download_if_necessary  = local_path


def load_reference(pack):
    return TTSModel.load_model(language=pack)


def install_hooks(model):
    """Record what the C++ side dumps: the text tokens and their lookup
    embeddings (first non empty conditioner call), every out_norm output
    (the text prefill, then one per step), every EOS logit and every
    generated latent."""
    rec = {"ids": None, "embed": None, "hidden": [], "eos": [], "latents": []}
    fl  = model.flow_lm

    cond = fl.conditioner.forward
    def conditioner(tokens):
        out = cond(tokens)
        if rec["ids"] is None and tokens.numel() > 0:
            rec["ids"]   = tokens[0].clone()
            rec["embed"] = out[0].detach().clone()
        return out
    fl.conditioner.forward = conditioner

    norm = fl.out_norm.forward
    def out_norm(x):
        out = norm(x)
        rec["hidden"].append(out[0, -1].detach().clone())
        return out
    fl.out_norm.forward = out_norm

    eos = fl.out_eos.forward
    def out_eos(x):
        out = eos(x)
        rec["eos"].append(float(out.reshape(-1)[-1]))
        return out
    fl.out_eos.forward = out_eos

    gen = model._autoregressive_generation
    def autoregressive(model_state, max_gen_len, frames_after_eos, queue, stop):
        class Tee:
            def put(self, x):
                if x is not None:
                    rec["latents"].append(x.reshape(-1).detach().clone())
                queue.put(x)
        return gen(model_state, max_gen_len, frames_after_eos, Tee(), stop)
    model._autoregressive_generation = autoregressive
    return rec


def voice_rows(state, layer, which):
    """KV rows [T, dim] of one flow LM layer of a voice state (0 = K, 1 = V)."""
    s = state[f"transformer.layers.{layer}.self_attn"]
    n = int(s["offset"].view(-1)[0])
    c = s["cache"][which, 0, :n]
    return c.reshape(n, -1)


def dump_reference(rec, state, n_layers, audio, dump_dir):
    """Write the recorded intermediates under the C++ dump names. The first
    out_norm and EOS calls belong to the text prefill."""
    save_dump_i32(os.path.join(dump_dir, "prompt-ids.bin"), rec["ids"])
    save_dump(os.path.join(dump_dir, "text-embed.bin"), rec["embed"])
    save_dump(os.path.join(dump_dir, "voice-k0.bin"), voice_rows(state, 0, 0))
    save_dump(os.path.join(dump_dir, "voice-v-last.bin"), voice_rows(state, n_layers - 1, 1))
    save_dump(os.path.join(dump_dir, "step0-hidden.bin"), rec["hidden"][1])
    save_dump(os.path.join(dump_dir, "step0-latent.bin"), rec["latents"][0])
    save_dump(os.path.join(dump_dir, "eos.bin"), np.array(rec["eos"][1:], dtype=np.float32))
    save_dump(os.path.join(dump_dir, "latents.bin"), torch.stack(rec["latents"]))
    save_dump(os.path.join(dump_dir, "output-audio.bin"), audio)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_dump(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.float32).cpu().numpy()
    data  = np.ascontiguousarray(data.astype(np.float32))
    shape = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(data.tobytes())

def save_dump_i32(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.int64).cpu().numpy()
    data  = np.ascontiguousarray(data.astype(np.int64))
    shape = data.shape
    fdata = data.astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(fdata.tobytes())

def load_dump(path):
    raw   = np.fromfile(path, dtype=np.uint8)
    ndim  = int(np.frombuffer(raw[0:4], dtype=np.int32)[0])
    shape = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * ndim], dtype=np.int32))
    body  = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.float32)
    return body.reshape(shape), shape

def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def stft_cos(a, b, win=2048, hop=512):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    window = np.hanning(win)
    frames = (n - win) // hop + 1
    if frames <= 0:
        return 0.0
    sa = np.zeros((frames, win // 2 + 1))
    sb = np.zeros((frames, win // 2 + 1))
    for i in range(frames):
        s = i * hop
        sa[i] = np.abs(np.fft.rfft(a[s:s + win] * window))
        sb[i] = np.abs(np.fft.rfft(b[s:s + win] * window))
    return cos(sa.ravel(), sb.ravel())

def pair(name, dump_cpp, dump_pt):
    a, _ = load_dump(os.path.join(dump_cpp, name))
    b, _ = load_dump(os.path.join(dump_pt,  name))
    return a, b

def metric(a, b):
    n     = min(a.size, b.size)
    af    = a.astype(np.float64).ravel()[:n]
    bf    = b.astype(np.float64).ravel()[:n]
    d     = np.abs(af - bf)
    nrm_a = float(np.linalg.norm(af))
    nrm_b = float(np.linalg.norm(bf))
    c     = float(np.dot(af, bf) / (nrm_a * nrm_b)) if nrm_a > 1e-10 and nrm_b > 1e-10 else 0.0
    return c, float(d.max()), float(d.mean())

def compare_stages(stages, dump_cpp, dump_pt):
    """Iterate the stages list and print one line per pair. Skips silently
    when a dump file is missing."""
    for label, name in stages:
        try:
            a, b = pair(name, dump_cpp, dump_pt)
        except FileNotFoundError:
            print(f"[Cossim] {label} skipped (missing dump)")
            continue
        c, mx, mn = metric(a, b)
        print(f"[Cossim] {label} cos: {c:.6f} max: {mx:.4e} mean: {mn:.4e}")

def compare_exact_i32(name, dump_cpp, dump_pt, label):
    """Compare two int dumps stored as f32 (save_dump_i32 in Python,
    debug_dump_i32_as_f32 in C++). Prints an exact match percentage."""
    a, b = pair(name, dump_cpp, dump_pt)
    ai   = a.astype(np.int64).ravel()
    bi   = b.astype(np.int64).ravel()
    n    = min(ai.size, bi.size)
    pct  = 100.0 * float(np.mean(ai[:n] == bi[:n]))
    print(f"[Cossim] {label} exact: {pct:.2f}% ({n} values)")
    return pct

# Stages dumped by both sides, in pipeline order.
STAGES = [
    ("TextEmbed",   "text-embed.bin"),
    ("VoiceK0",     "voice-k0.bin"),
    ("VoiceVLast",  "voice-v-last.bin"),
    ("Step0Hidden", "step0-hidden.bin"),
    ("Step0Latent", "step0-latent.bin"),
    ("Eos",         "eos.bin"),
    ("Latents",     "latents.bin"),
]


def run(case, voice_ref, voice_args):
    """One cossim case: the reference then the C++ CLI on the same text,
    pack, voice and seed, both dumping, then one line per stage.
    voice_ref is what get_state_for_audio_prompt takes, voice_args maps the
    pack to the CLI voice flags."""
    import argparse
    import subprocess

    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="../examples/prompt.txt")
    ap.add_argument("--pack",   default="english_2026-09")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--quant",  default="F32", help="GGUF quantization suffix (F32, BF16, Q8_0, Q4_K_M)")
    args = ap.parse_args()

    dump_pt  = f"python/{case}"
    dump_cpp = f"cpp/{case}"
    ensure_dir(dump_pt)
    ensure_dir(dump_cpp)
    out_pt  = os.path.join(dump_pt, f"{case}-python.wav")
    out_cpp = os.path.join(dump_cpp, f"{case}-cpp.wav")

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] Pack: {args.pack} Voice: {voice_ref} Seed: {args.seed}")

    model = load_reference(args.pack)
    state = model.get_state_for_audio_prompt(voice_ref, truncate=True)
    rec   = install_hooks(model)
    torch.manual_seed(args.seed)
    audio_pt = model.generate_audio(state, text).numpy().astype(np.float32)
    sf.write(out_pt, audio_pt, model.sample_rate, subtype="FLOAT")
    n_layers = model.config.flow_lm.transformer.num_layers
    dump_reference(rec, state, n_layers, audio_pt, dump_pt)
    print(f"[Python] Frames: {len(rec['latents'])}")
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {model.sample_rate} Hz "
          f"{audio_pt.shape[0] / model.sample_rate:.2f}s -> {out_pt}")

    if not os.path.isfile(BIN):
        print(f"[Cossim] FATAL: {BIN} not found, build pocket-tts first")
        sys.exit(1)
    gguf = f"../models/pocket-tts-{args.pack}-{args.quant}.gguf"
    if not os.path.isfile(gguf):
        print(f"[Cossim] FATAL: GGUF not found: {gguf}")
        sys.exit(1)
    print(f"[Quant] {args.quant} -> {gguf}")

    cmd = [BIN, "--model", gguf, *voice_args(args.pack), "--seed", str(args.seed),
           "--dump", dump_cpp, "--format", "wav32", "-o", out_cpp]
    print(f"[GGML] Cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, input=text, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
    audio_cpp, sr = sf.read(out_cpp, dtype="float32")
    lat_cpp, _ = load_dump(os.path.join(dump_cpp, "latents.bin"))
    print(f"[GGML] Frames: {lat_cpp.shape[0]}")
    print(f"[GGML] Audio: {audio_cpp.shape[0]} samples {sr} Hz {audio_cpp.shape[0] / sr:.2f}s -> {out_cpp}")

    compare_exact_i32("prompt-ids.bin", dump_cpp, dump_pt, "PromptIDs")
    compare_stages(STAGES, dump_cpp, dump_pt)
    aa, ab = pair("output-audio.bin", dump_cpp, dump_pt)
    print(f"[Cossim] Audio cos: {cos(aa, ab):.6f}")
    n = min(aa.size, ab.size)
    print(f"[Cossim] WAV stft_cos: {stft_cos(aa.ravel()[:n], ab.ravel()[:n]):.6f} samples: {n}")
