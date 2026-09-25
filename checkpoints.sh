#!/bin/bash
# Download the Pocket TTS checkpoint from HuggingFace. The repo is gated:
# accept its terms on the model page and log in (hf auth login) first.
# Usage: ./checkpoints.sh

set -eu

DIR="checkpoints"
mkdir -p "$DIR"

dl_repo() {
    local name="$1" repo="$2"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls "$target"/languages/*/model.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $repo"
    hf download --quiet "$repo" --local-dir "$target"
}

dl_repo "pocket-tts" "kyutai/pocket-tts"
