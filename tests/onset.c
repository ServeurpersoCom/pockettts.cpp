/* tests/onset.c: every generated frame reaches the output.
 *
 * Usage: test-onset <model.gguf> <voices.gguf> <voice>
 *
 * Synthesizes a two chunk text with a fixed seed and checks that the
 * output holds one 80 ms frame of audio per generated latent (frame
 * counts from the [TTS] chunk log lines), that the first streamed block
 * is the first frame alone, and that every chunk starts at 0, the start
 * of its 5 ms fade-in. Speech can begin in the first frame of a chunk,
 * so a dropped frame clips the first word.
 */

#include "pocket.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CHUNKS 16

struct run {
    float * samples;
    int     n;
    int     first_block;
    int     frames[MAX_CHUNKS];
    int     n_chunks;
};

static void on_log(enum pt_log_level level, const char * msg, void * user_data) {
    struct run * r = (struct run *) user_data;
    int          ci, nc, tokens, frames;
    float        ms;
    (void) level;
    if (sscanf(msg, "[TTS] chunk %d/%d: %d tokens, prefill %f ms, %d frames", &ci, &nc, &tokens, &ms, &frames) == 5 &&
        r->n_chunks < MAX_CHUNKS) {
        r->frames[r->n_chunks++] = frames;
    }
}

static bool on_chunk(const float * samples, int n_samples, void * user_data) {
    struct run * r = (struct run *) user_data;
    if (r->first_block == 0) {
        r->first_block = n_samples;
    }
    r->samples = (float *) realloc(r->samples, (size_t) (r->n + n_samples) * sizeof(float));
    memcpy(r->samples + r->n, samples, (size_t) n_samples * sizeof(float));
    r->n += n_samples;
    return true;
}

int main(int argc, char ** argv) {
    struct pt_init_params ip;
    struct pt_tts_params  tp;
    struct pt_context *   ctx;
    struct pt_voice *     voice;
    struct run            r = { 0 };
    int                   frame, total = 0, start = 0, i, fails = 0;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <voices.gguf> <voice>\n", argv[0]);
        return 2;
    }
    pt_log_set(on_log, &r);
    pt_init_default_params(&ip);
    ip.model_path = argv[1];
    ctx           = pt_init(&ip);
    if (!ctx) {
        fprintf(stderr, "FAIL: %s\n", pt_last_error());
        return 1;
    }
    voice = pt_voice_named(ctx, argv[2], argv[3]);
    if (!voice) {
        fprintf(stderr, "FAIL: %s\n", pt_last_error());
        return 1;
    }
    frame = pt_sample_rate(ctx) * 2 / 25;

    pt_tts_default_params(&tp);
    tp.text             = "I found the bug. I moved the reset into the speech start handler.";
    tp.voice            = voice;
    tp.seed             = 42;
    tp.max_chunk_tokens = 8;
    tp.on_chunk         = on_chunk;
    tp.user_data        = &r;
    if (pt_synthesize(ctx, &tp, NULL) != PT_STATUS_OK) {
        fprintf(stderr, "FAIL: %s\n", pt_last_error());
        return 1;
    }

    if (r.n_chunks < 2) {
        fprintf(stderr, "FAIL: %d chunk(s), the text needs two\n", r.n_chunks);
        fails++;
    }
    for (i = 0; i < r.n_chunks; i++) {
        total += r.frames[i];
    }
    if (r.n != total * frame) {
        fprintf(stderr, "FAIL: %d samples for %d frames, expected %d\n", r.n, total, total * frame);
        fails++;
    }
    if (r.first_block != frame) {
        fprintf(stderr, "FAIL: first block of %d samples, expected one frame (%d)\n", r.first_block, frame);
        fails++;
    }
    for (i = 0; i < r.n_chunks && start < r.n; i++) {
        if (r.samples[start] != 0.0f) {
            fprintf(stderr, "FAIL: chunk %d starts at %g, expected 0 (fade-in)\n", i + 1, r.samples[start]);
            fails++;
        }
        start += r.frames[i] * frame;
    }

    free(r.samples);
    pt_voice_free(voice);
    pt_free(ctx);
    pt_log_set(NULL, NULL);
    if (fails) {
        return 1;
    }
    printf("OK: %d chunks, %d frames, %d samples\n", r.n_chunks, total, r.n);
    return 0;
}
