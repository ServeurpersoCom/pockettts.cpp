#!/bin/bash
# Call the pockettts OpenAI-compatible TTS server.
# Default response_format is pcm: audio streams chunked as it is generated.

host="${1:-127.0.0.1}"
port="${2:-8080}"

# Streaming pcm, piped straight into a player as it arrives (real time).
# s16le mono 24 kHz. ffplay reads the raw stream from stdin.
curl -s -X POST "http://${host}:${port}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"input":"The quick brown fox jumps over the lazy dog.","voice":"alba"}' \
    | ffplay -f s16le -ar 24000 -ch_layout mono -nodisp -autoexit -i -

# One-shot wav written to a file.
curl -s -X POST "http://${host}:${port}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d '{"input":"This one is written to a file.","voice":"alba","response_format":"wav"}' \
    --output out.wav
echo "wrote out.wav"

# Register the voice of a recording, then speak with it.
printf '{"name":"mine","wav_b64":"%s"}' "$(base64 -w0 out.wav)" | curl -s -X POST \
    "http://${host}:${port}/v1/audio/voices" -H "Content-Type: application/json" --data-binary @-
echo
curl -s "http://${host}:${port}/v1/audio/voices"
echo
