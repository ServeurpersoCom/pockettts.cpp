#!/usr/bin/env python3
# convert_community.py: Convert community-trained Pocket TTS repos to GGUF.
#
# Scans checkpoints/community/ for downloaded repos, discovers architecture
# configs and weight/tokenizer paths from each YAML, and calls convert_model()
# from convert.py for every variant found.
#
# Handles the layout variations across community repos:
#   - root-level model.safetensors + tokenizer.model (most common)
#   - per-variant subdirs (6l/, 24l/, japanese/)
#   - named weight files (teacher.safetensors)
#   - non-standard tokenizer filenames (pl_bpe4000.model)
#   - multi-variant repos (one YAML per variant)
#
# Output: models/community/<repo>/<config_stem>-F32.gguf
#
# Usage: .venv/Scripts/python.exe convert_community.py

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from convert import convert_model, _load_yaml


def hf_relpath(url):
    """Extract relative file path from an hf:// URL (strips @commit)."""
    url = url.split("@")[0]
    if url.startswith("hf://"):
        parts = url[5:].split("/", 2)
        return parts[2] if len(parts) > 2 else ""
    # bare path or local reference
    return os.path.basename(url)


def _discover_weights(repo_dir):
    """Fallback weight discovery for configs without weights_path.

    Prefers model.safetensors; otherwise accepts a single root-level checkpoint
    file (e.g. PocketTTS-ZH-6L ships model_int8.pt, the int8 delivery format).
    Returns '' if nothing usable is found.
    """
    if os.path.isfile(os.path.join(repo_dir, "model.safetensors")):
        return "model.safetensors"
    candidates = sorted(f for f in os.listdir(repo_dir)
                        if f.endswith((".safetensors", ".pt")))
    return candidates[0] if len(candidates) == 1 else ""


def discover_variants(repo_dir):
    """Find all (yaml_stem, cfg_dict, model_relpath, tok_relpath) in a repo."""
    variants = []
    for f in sorted(os.listdir(repo_dir)):
        if not f.endswith(".yaml"):
            continue
        cfg_path = os.path.join(repo_dir, f)
        try:
            cfg = _load_yaml(cfg_path)
        except Exception as e:
            print(f"  [Skip] {f}: {e}")
            continue
        if not isinstance(cfg, dict):
            continue
        # Must be an architecture config (has flow_lm or mimi section)
        if "flow_lm" not in cfg and "mimi" not in cfg:
            continue

        stem = f[:-5]  # strip .yaml

        # Model weights path from top-level weights_path
        model_rel = ""
        wp = cfg.get("weights_path", "")
        if isinstance(wp, str) and wp:
            model_rel = hf_relpath(wp)

        # Tokenizer path from flow_lm.lookup_table.tokenizer_path
        tok_rel = "tokenizer.model"  # default
        lt = (cfg.get("flow_lm") or {}).get("lookup_table") or {}
        tp = lt.get("tokenizer_path", "")
        if isinstance(tp, str) and tp:
            tok_rel = hf_relpath(tp)

        variants.append((stem, cfg, model_rel, tok_rel))
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Convert community Pocket TTS repos to GGUF.")
    parser.add_argument("--repos-dir",
                        default=os.path.join(SCRIPT_DIR, "checkpoints", "community"),
                        help="Directory containing downloaded community repos")
    parser.add_argument("--output-dir",
                        default=os.path.join(SCRIPT_DIR, "models", "community"),
                        help="Output directory for GGUF files (default: %(default)s)")
    parser.add_argument("repos", nargs="*", default=[],
                        help="Only convert these repos (default: all)")
    args = parser.parse_args()

    if not os.path.isdir(args.repos_dir):
        print(f"[Community] FATAL: {args.repos_dir}/ not found. "
              "Run download_community.sh first.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    rc = 0
    for repo_name in sorted(os.listdir(args.repos_dir)):
        if args.repos and repo_name not in args.repos:
            continue
        repo_dir = os.path.join(args.repos_dir, repo_name)
        if not os.path.isdir(repo_dir):
            continue

        variants = discover_variants(repo_dir)
        if not variants:
            print(f"[Community] {repo_name}: no architecture configs found")
            continue

        out_repo = os.path.join(args.output_dir, repo_name)
        os.makedirs(out_repo, exist_ok=True)

        for stem, cfg, model_rel, tok_rel in variants:
            out_path = os.path.join(out_repo, f"{stem}-F32.gguf")
            if os.path.exists(out_path):
                print(f"[Community] skip {repo_name}/{os.path.basename(out_path)}: exists")
                continue

            # Configs without weights_path (e.g. PocketTTS-ZH-6L) fall back to
            # discovering a root-level checkpoint file.
            if not model_rel:
                model_rel = _discover_weights(repo_dir)

            model_path = os.path.join(repo_dir, model_rel) if model_rel else None
            tok_path = os.path.join(repo_dir, tok_rel) if tok_rel else None

            if not model_path or not os.path.isfile(model_path):
                print(f"[Community] {repo_name}/{stem}: model file not found "
                      f"({model_rel}), skipping")
                continue
            if not tok_path or not os.path.isfile(tok_path):
                print(f"[Community] {repo_name}/{stem}: tokenizer not found "
                      f"({tok_rel}), skipping")
                continue

            print(f"[Community] {repo_name}/{stem} -> {os.path.basename(out_path)}")
            rc |= convert_model(stem, repo_dir, out_path,
                                config=cfg, model_file=model_path,
                                tokenizer_path=tok_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
