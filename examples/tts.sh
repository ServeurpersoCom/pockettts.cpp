#!/bin/bash

set -eu

../build/pocket-tts \
    --model ../models/pocket-tts-english_2026-09-Q8_0.gguf \
    --voices ../models/pocket-tts-english_2026-09-voices.gguf \
    --voice alba \
    -o tts.wav < prompt.txt
