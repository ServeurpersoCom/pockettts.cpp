/* tests/abi-c.c: link-only ABI smoke test for pocket.h.
 *
 * Compiled in pure C99 with -Wall -Werror -pedantic, loads no model. It
 * guarantees at every build that pocket.h parses as C, that every public
 * pt_* symbol links from a C translation unit, that the structs are POD
 * and zero-initialisable, and that NULL arguments and out of range
 * abi_version values are rejected.
 */

#include "pocket.h"

#include <stdio.h>
#include <string.h>

static void stub_log(enum pt_log_level level, const char * msg, void * user_data) {
    (void) level;
    (void) msg;
    (void) user_data;
}

static bool stub_on_chunk(const float * samples, int n_samples, void * user_data) {
    (void) samples;
    (void) n_samples;
    (void) user_data;
    return true;
}

int main(void) {
    struct pt_init_params ip;
    struct pt_tts_params  tp;
    struct pt_audio       audio = { 0 };
    int                   fails = 0;

    if (!pt_version() || !pt_version()[0]) {
        fprintf(stderr, "FAIL: pt_version empty\n");
        fails++;
    }

    pt_init_default_params(&ip);
    if (ip.abi_version != PT_ABI_VERSION || ip.model_path) {
        fprintf(stderr, "FAIL: pt_init_default_params\n");
        fails++;
    }
    pt_tts_default_params(&tp);
    if (tp.abi_version != PT_ABI_VERSION || tp.lsd_steps != 1 || tp.max_chunk_tokens != 50 || tp.seed != UINT32_MAX) {
        fprintf(stderr, "FAIL: pt_tts_default_params\n");
        fails++;
    }
    tp.on_chunk = stub_on_chunk;

    pt_log_set(stub_log, NULL);
    ip.abi_version = PT_ABI_VERSION + 1;
    ip.model_path  = "missing.gguf";
    if (pt_init(&ip) || !strstr(pt_last_error(), "abi_version")) {
        fprintf(stderr, "FAIL: future abi_version accepted\n");
        fails++;
    }
    ip.abi_version = PT_ABI_VERSION;
    ip.model_path  = NULL;
    if (pt_init(&ip) || !pt_last_error()[0]) {
        fprintf(stderr, "FAIL: NULL model_path accepted\n");
        fails++;
    }
    if (pt_synthesize(NULL, &tp, &audio) != PT_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "FAIL: pt_synthesize without context\n");
        fails++;
    }
    if (pt_voice_load(NULL, "x.wav", 0.0f) || pt_voice_from_state(NULL, NULL, 0) ||
        pt_voice_from_audio(NULL, NULL, 0, 0) || pt_voice_save(NULL, "x") != PT_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "FAIL: voice entries without context\n");
        fails++;
    }
    pt_audio_free(&audio);
    pt_voice_free(NULL);
    pt_free(NULL);
    if (pt_sample_rate(NULL) != 0) {
        fprintf(stderr, "FAIL: pt_sample_rate(NULL)\n");
        fails++;
    }
    pt_log_set(NULL, NULL);

    if (fails) {
        return 1;
    }
    printf("abi-c: OK (%s)\n", pt_version());
    return 0;
}
