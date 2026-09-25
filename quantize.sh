#!/bin/bash
# Derive lighter GGUFs from the F32 source-of-truth produced by convert.py.
# Each F32 model under models/ is quantized to BF16, Q8_0 and Q4_K_M.
#
# Q8_0 is the default: BF16 keeps the reference weights exactly (every
# released pack ships BF16), Q4_K_M is the smallest variant. The policy
# lives in tools/quantize.cpp should_quantize: only 2D matmul weights are
# quantized, conv kernels and vectors keep F32.

set -eu

Q="./build/quantize"

quantize() {
    local src="$1" type="$2"
    local out="${src/-F32.gguf/-${type}.gguf}"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$src" "$out" "$type"
    fi
}

for src in models/pocket-tts-*-F32.gguf; do
    [ -f "$src" ] || continue
    quantize "$src" BF16
    quantize "$src" Q8_0
    quantize "$src" Q4_K_M
done
