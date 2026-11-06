#!/bin/bash
# Download community-trained Pocket TTS checkpoints from HuggingFace into
# checkpoints/community/<repo>. Only the files needed for conversion are pulled:
# architecture YAMLs, F32 safetensors (root and per-variant subdirs), and the
# tokenizer (tokenizer.json / *.model). Samples, onnx/, benchmarks/ etc. are skipped.
# Usage: ./download_community.sh [repo ...]   (no args = all repos below)

set -eu

DIR="checkpoints/community"
mkdir -p "$DIR"

INCLUDES=(
    "*.yaml"
    "*.safetensors"
    "*/*.safetensors"
    "*.pt"
    "tokenizer.json"
    "*.model"
)

dl_repo() {
    local name="$1" repo="$2"
    local target="$DIR/$name"
    if [ -d "$target" ] && { ls "$target"/**/*.safetensors >/dev/null 2>&1 \
            || ls "$target"/*.pt >/dev/null 2>&1; }; then
        echo "[OK] $name (present)"
        return
    fi
    echo "[Download] $name <- $repo"
    hf download --quiet "$repo" \
        --local-dir "$target" \
        --include "${INCLUDES[@]}"
}

REPOS=(
    "vvolhejn/pocket-tts-czech"
    "saryps-labs/pocket-tts-hindi"
    "seastar105/pocket-tts-korean-300m"
    "mehdi-hf/pocket-tts-farsi"
    "anak10thn/pocket-tts-indonesian"
    "cbentes/pocket-tts-estonian"
    "EryriLabs/pocket-tts-cymraeg"
    "alperiox/pocket-tts-turkish"
    "shefowl/pocket-tts-polish-6l"
    "myned-ai/pocket-tts-greek"
    "takuma104/pocket-tts-ja"
    "Thanushttk/pocket-tts-multilingual"
    "hustfyb/PocketTTS-ZH-6L"
)

if [ "$#" -gt 0 ]; then
    for repo in "$@"; do dl_repo "${repo##*/}" "$repo"; done
else
    for repo in "${REPOS[@]}"; do dl_repo "${repo##*/}" "$repo"; done
fi
