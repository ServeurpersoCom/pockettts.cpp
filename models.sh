#!/bin/bash
# Download pre-quantized pockettts.cpp GGUF models from HuggingFace.
# Usage: ./models.sh

set -eu

REPO="Serveurperso/pockettts.cpp-GGUF"
DIR="models"
mkdir -p "$DIR"

dl() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file"
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
}

dl "pocket-tts-english_2026-09-Q8_0.gguf"
dl "pocket-tts-english_2026-09-voices.gguf"
dl "pocket-tts-french_24l-Q8_0.gguf"
dl "pocket-tts-french_24l-voices.gguf"
