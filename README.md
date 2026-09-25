# pockettts.cpp

Local AI text-to-speech with voice cloning, powered by GGML. C++17 port
of Pocket TTS (Kyutai): a flow language model over continuous Mimi
latents, 7 languages in 6 and 24 layer packs, 24 kHz mono output, runs
on CPU, CUDA, Metal, Vulkan.

## Features

- Every released language pack: English (2026-01, 2026-04, 2026-09, 6
  and 24 layers), French, German, Italian, Portuguese, Spanish, Dutch
  (6 and 24 layers)
- The predefined voices of each pack and zero shot voice cloning from
  any recording
- Voice states saved to and loaded from `.safetensors` in the reference
  layout, interchangeable with the Python implementation
- Streaming synthesis: the flow LM runs one 80 ms latent per step and
  the stateful Mimi decoder emits audio after the first frame; any split
  of the latents decodes to the same audio
- Text preparation, sentence chunking and unigram tokenization identical
  to the reference, and a torch compatible noise generator: a seed gives
  the reference trajectory
- BF16, Q8_0 and Q4_K_M quantisation of the matmul weights, conv
  kernels and vectors kept at full precision
- Two tools: `pocket-tts` (text -> WAV) and `tts-server`
  (OpenAI-compatible HTTP server with a voice directory and a runtime
  voice registry)

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/pockettts.cpp.git
cd pockettts.cpp
./buildcuda.sh                   # NVIDIA GPU
./buildvulkan.sh                 # AMD/Intel GPU (Vulkan)
./buildcpu.sh                    # CPU only
./buildall.sh                    # all backends, runtime DL loading
NVCC_CCBIN=g++-13 ./buildcuda.sh # rolling release distros (Arch w/ GCC 16, etc.)
```

`-DGGML_SOURCE_DIR=<path>` swaps the ggml submodule for another tree (upstream ggml, llama.cpp/ggml).

## Model conversion

Pre-converted GGUFs are available on Hugging Face:

  https://huggingface.co/Serveurperso/pockettts.cpp-GGUF

```
./models.sh           # english_2026-09 and french_24l, Q8_0, with their voices -> models/
```

To convert from the original checkpoint (`kyutai/pocket-tts` is gated:
accept its terms on Hugging Face and log in with `hf auth login`):

```
./checkpoints.sh      # hf download kyutai/pocket-tts -> checkpoints/
./convert.py          # every pack: F32 model + voices -> models/
./quantize.sh         # BF16 / Q8_0 / Q4_K_M
```

Two GGUFs per language pack: the model
(`pocket-tts-{pack}-{variant}.gguf`: flow LM, flow head, Mimi encoder
and decoder, tokenizer, text rules) and its predefined voices
(`pocket-tts-{pack}-voices.gguf`). Packs are `english`,
`english_2026-01`, `english_2026-04`, `english_2026-09`, `french`,
`german`, `italian`, `portuguese`, `spanish`, `dutch`, and the 24 layer
`*_24l` variants. A voice belongs to the pack it was computed with.

## Quick start

Each block is the command run by the matching script in `examples/`.

Predefined voice (`tts.sh`):

```
./build/pocket-tts \
    --model models/pocket-tts-english_2026-09-Q8_0.gguf \
    --voices models/pocket-tts-english_2026-09-voices.gguf \
    --voice alba -o out.wav < prompt.txt
```

Voice cloning (`clone.sh`): any WAV works, the first 30 seconds are
kept (`--max-voice-sec`). `--export-voice ref.safetensors` writes the
voice state, reloadable with `--voice ref.safetensors` and by the
reference implementation:

```
./build/pocket-tts \
    --model models/pocket-tts-english_2026-09-Q8_0.gguf \
    --voice freeman.wav -o out.wav < prompt.txt
```

Generation options: `--seed`, `--temp` (flow noise temperature, pack
default 0.3, 0 is noise free), `--lsd-steps`, `--eos-threshold`,
`--frames-after-eos`, `--max-chunk-tokens`. `-o -` streams the WAV to
stdout.

OpenAI-compatible server (`tts-server`): `response_format` "pcm"
streams s16le as it is generated, "wav" returns a one-shot file. The
voices of `--voices` are served by name; more voices register over HTTP,
a recording encoded server side or a voice state file taken verbatim:

```
./build/tts-server \
    --model models/pocket-tts-english_2026-09-Q8_0.gguf \
    --voices models/pocket-tts-english_2026-09-voices.gguf \
    --voice alba --port 8080

curl -X POST localhost:8080/v1/audio/voices -H "Content-Type: application/json" \
    -d "{\"name\":\"freeman\",\"wav_b64\":\"$(base64 -w0 freeman.wav)\"}"

curl -X POST localhost:8080/v1/audio/speech -H "Content-Type: application/json" \
    -d '{"input":"Hello world.","voice":"freeman","response_format":"wav",
         "seed":42,"temperature":0.3}' -o out.wav
```

`GET /v1/audio/voices` lists the voices, `DELETE /v1/audio/voices/<name>`
drops a registered one.

## Embedding the library

The CLI tools are thin wrappers over a public ABI. Single-header,
single-name-prefix, plain C linkage so that C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way.

```c
#include "pocket.h"

struct pt_init_params iparams;
pt_init_default_params(&iparams);
iparams.model_path = "models/pocket-tts-english_2026-09-Q8_0.gguf";

struct pt_context * ctx   = pt_init(&iparams);
struct pt_voice *   voice = pt_voice_named(ctx, "models/pocket-tts-english_2026-09-voices.gguf", "alba");

struct pt_tts_params params;
pt_tts_default_params(&params);
params.text  = "Hello world.";
params.voice = voice;

struct pt_audio audio = { 0 };
pt_synthesize(ctx, &params, &audio);
/* audio.samples, audio.n_samples, audio.sample_rate, audio.channels */
pt_audio_free(&audio);
pt_voice_free(voice);
pt_free(ctx);
```

`pt_voice_load` builds a voice from a WAV recording or a voice state
file, `pt_voice_from_audio` from PCM in memory,
`pt_voice_from_state` from the bytes of a voice state file,
`pt_voice_save` writes one. With `params.on_chunk` set, audio streams
through the callback. `pt_synthesize` is thread safe; concurrent calls
on one context run one after the other.

`tests/abi-c.c` is built with `-std=c99 -Wall -Werror -pedantic` on
every build (the `test-abi-c` target). For a binding-friendly shared
library, configure with `cmake -DPOCKET_SHARED=ON ...`: it exports only
the `pt_*` symbols.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the model, the
GGUF layout, the inference pipeline and the validation results.

## License

MIT. See [LICENSE](LICENSE).

Upstream model: Pocket TTS by Kyutai, CC-BY-4.0, with its prohibited
use terms (no voice cloning without consent, no deceptive content).
