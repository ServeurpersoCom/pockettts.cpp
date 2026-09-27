#pragma once
// pocket.h: public ABI for pockettts.cpp.
//
// Single-header public API. Pure C99, consumable from C and C++ alike.
// Style follows whisper.h / llama.h / qwen.h: extern "C" linkage on every
// entry, POD structs only, const char * UTF-8 strings, pt_status enum
// returns.
//
// One pt_context holds one Pocket TTS language pack (flow LM, flow head,
// Mimi codec, tokenizer) resident on one GGML backend. A pt_voice holds
// the flow LM state after the voice prompt: it is built once, either from
// a reference recording or from a voice state file in the reference
// safetensors format, and reused by every synthesis on the same context.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(POCKET_STATIC)
#        define PT_API
#    elif defined(POCKET_BUILD)
#        define PT_API __declspec(dllexport)
#    else
#        define PT_API __declspec(dllimport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define PT_API __attribute__((visibility("default")))
#else
#    define PT_API
#endif

// Struct ABI version. Callers fill `.abi_version = PT_ABI_VERSION` (or let
// the pt_*_default_params helpers set it); an entry rejects any struct
// outside [PT_ABI_MIN_VERSION, PT_ABI_VERSION].
#define PT_ABI_VERSION     1
#define PT_ABI_MIN_VERSION 1

// "<git-hash> (<date>)" of the commit this binary was built from.
PT_API const char * pt_version(void);

enum pt_status {
    PT_STATUS_OK              = 0,
    PT_STATUS_INVALID_PARAMS  = -1,
    PT_STATUS_GENERATE_FAILED = -3,
    PT_STATUS_OOM             = -4,
    PT_STATUS_CANCELLED       = -5,
};

// Last error message produced on the calling thread by a failing pt_*
// entry. Thread local storage, valid until the next failing entry.
PT_API const char * pt_last_error(void);

// Log severity, numerically ordered for threshold filtering.
enum pt_log_level {
    PT_LOG_DEBUG = 0,
    PT_LOG_INFO  = 1,
    PT_LOG_WARN  = 2,
    PT_LOG_ERROR = 3,
};

// Log callback, msg has no trailing newline. NULL restores stderr output.
typedef void (*pt_log_cb)(enum pt_log_level level, const char * msg, void * user_data);
PT_API void pt_log_set(pt_log_cb cb, void * user_data);

// Mono PCM owned by the struct, released by pt_audio_free. Zero
// initialise before first use.
struct pt_audio {
    float * samples;
    int     n_samples;
    int     sample_rate;
    int     channels;
};

PT_API void pt_audio_free(struct pt_audio * a);

struct pt_context;
struct pt_voice;

// use_fa enables fused flash attention on GPU backends (CPU always runs
// the F32 manual chain); clamp_fp16 clamps V and the residual stream to the
// FP16 range, for CUDA targets before Ampere that accumulate in FP16.
struct pt_init_params {
    int          abi_version;
    const char * model_path;  // pocket-tts-<pack>-<type>.gguf
    bool         use_fa;
    bool         clamp_fp16;
};

// Defaults: model_path NULL, use_fa true, clamp_fp16 false.
PT_API void                pt_init_default_params(struct pt_init_params * p);
PT_API struct pt_context * pt_init(const struct pt_init_params * params);
PT_API void                pt_free(struct pt_context * ctx);

// Sample rate of the synthesized audio (24000 for the released packs).
PT_API int pt_sample_rate(const struct pt_context * ctx);

// Voice from a file: a .safetensors voice state in the reference layout
// (the predefined voices of the pack, or one written by pt_voice_save),
// or any WAV recording encoded through the Mimi encoder. max_sec > 0
// keeps only the first max_sec seconds of a recording. NULL on failure.
PT_API struct pt_voice * pt_voice_load(struct pt_context * ctx, const char * path, float max_sec);

// Predefined voice `name` of a voices GGUF (pocket-tts-<pack>-voices.gguf).
PT_API struct pt_voice * pt_voice_named(struct pt_context * ctx, const char * voices_path, const char * name);

// Voice from the bytes of a voice state .safetensors file.
PT_API struct pt_voice * pt_voice_from_state(struct pt_context * ctx, const void * data, size_t size);

// Voice from mono float PCM at any sample rate.
PT_API struct pt_voice * pt_voice_from_audio(struct pt_context * ctx,
                                             const float *       samples,
                                             int                 n_samples,
                                             int                 sample_rate);

// Voice state as a .safetensors file readable by pt_voice_load and by the
// reference implementation (pocket-tts export-voice layout).
PT_API int pt_voice_save(const struct pt_voice * voice, const char * path);

PT_API void pt_voice_free(struct pt_voice * voice);

// Streaming sink: n_samples of mono audio at pt_sample_rate. Returning
// false cancels the synthesis.
typedef bool (*pt_chunk_cb)(const float * samples, int n_samples, void * user_data);

struct pt_tts_params {
    int                     abi_version;
    const char *            text;
    const struct pt_voice * voice;

    float    temperature;       // noise variance of the flow head, < 0 keeps the pack default
    int      lsd_steps;         // flow integration steps, 1 for the released packs
    float    eos_threshold;     // EOS logit threshold
    int      frames_after_eos;  // frames kept after EOS, < 0 derives it from the text
    int      max_chunk_tokens;  // text chunk budget in tokens
    uint32_t seed;              // UINT32_MAX draws a random seed

    pt_chunk_cb on_chunk;       // NULL: audio returned in one buffer
    void *      user_data;

    const char * dump_dir;  // intermediate tensors for the cossim tests, NULL disables
};

// Reference defaults: temperature -1 (pack default), lsd_steps 1,
// eos_threshold -4, frames_after_eos -1, max_chunk_tokens 50, seed
// UINT32_MAX, no streaming.
PT_API void pt_tts_default_params(struct pt_tts_params * p);

// Synthesize text with a voice. Thread safe: concurrent calls on one
// context run one after the other. With on_chunk set, audio streams
// through the callback and out may be NULL; otherwise out receives the
// whole utterance.
PT_API int pt_synthesize(struct pt_context * ctx, const struct pt_tts_params * params, struct pt_audio * out);

#ifdef __cplusplus
}
#endif
