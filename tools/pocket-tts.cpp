// pocket-tts.cpp: CLI wrapper around the pockettts.cpp public ABI.
// Reads the text from stdin, loads the voice (a predefined voice of the
// voices GGUF, a WAV recording or a voice state .safetensors), synthesizes
// and writes a WAV file, or streams it to stdout with -o -, one utterance
// per stdin line with --stream-by-line. --export-voice writes the voice
// state for later runs and for the reference.

#include "audio-io.h"
#include "pocket.h"
#include "utf8.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#    include <fcntl.h>
#    include <io.h>
#endif

static void print_usage(const char * prog) {
    fprintf(stderr, "pockettts.cpp %s\n\n", pt_version());
    fprintf(stderr,
            "Usage: %s --model <gguf> --voice <name|path> [options] -o <out.wav> < text.txt\n\n"
            "Required:\n"
            "  --model <gguf>            Language pack GGUF (pocket-tts-<pack>-<quant>.gguf)\n"
            "  --voice <name|path>       Voice of --voices, recording (.wav) or voice state (.safetensors)\n\n"
            "Optional:\n"
            "  --voices <gguf>           Predefined voices of the pack (pocket-tts-<pack>-voices.gguf)\n"
            "  -o <path>                 Output WAV, '-' streams to stdout (default: out.wav)\n"
            "  --format <fmt>            wav16, wav24 or wav32 (default: wav16)\n"
            "  --export-voice <path>     Write the voice state to a .safetensors file\n"
            "  --max-voice-sec <f>       Recording length kept for cloning (default: 30, 0 keeps all)\n"
            "  --stream-by-line          Synthesize each stdin line as it arrives, one WAV header per line (-o '-')\n\n"
            "Generation:\n"
            "  --seed <n>                Noise seed (default: random)\n"
            "  --temp <f>                Flow noise temperature (default: pack value)\n"
            "  --lsd-steps <n>           Flow integration steps (default: 1)\n"
            "  --eos-threshold <f>       EOS logit threshold (default: -4)\n"
            "  --frames-after-eos <n>    Frames kept after EOS (default: from the text)\n"
            "  --max-chunk-tokens <n>    Text chunk budget in tokens (default: 50)\n\n"
            "Debug:\n"
            "  --no-fa                   Disable flash attention\n"
            "  --clamp-fp16              Clamp hidden states to FP16 range\n"
            "  --dump <dir>              Dump intermediate tensors (f32) to <dir>\n",
            prog);
}

struct Args {
    const char *  model          = nullptr;
    const char *  voice          = nullptr;
    const char *  voices         = nullptr;
    const char *  out            = "out.wav";
    const char *  format         = "wav16";
    const char *  export_voice   = nullptr;
    float         max_voice_sec  = 30.0f;
    bool          stream_by_line = false;
    bool          use_fa         = true;
    bool          clamp_fp16     = false;
    pt_tts_params tts;
};

static bool parse_args(int argc, char ** argv, Args & a) {
    pt_tts_default_params(&a.tts);
    for (int i = 1; i < argc; i++) {
        const char * s   = argv[i];
        const bool   val = i + 1 < argc;
        if (!strcmp(s, "--model") && val) {
            a.model = argv[++i];
        } else if (!strcmp(s, "--voice") && val) {
            a.voice = argv[++i];
        } else if (!strcmp(s, "--voices") && val) {
            a.voices = argv[++i];
        } else if (!strcmp(s, "--stream-by-line")) {
            a.stream_by_line = true;
        } else if (!strcmp(s, "--no-fa")) {
            a.use_fa = false;
        } else if (!strcmp(s, "--clamp-fp16")) {
            a.clamp_fp16 = true;
        } else if (!strcmp(s, "--dump") && val) {
            a.tts.dump_dir = argv[++i];
        } else if (!strcmp(s, "-o") && val) {
            a.out = argv[++i];
        } else if (!strcmp(s, "--format") && val) {
            a.format = argv[++i];
        } else if (!strcmp(s, "--export-voice") && val) {
            a.export_voice = argv[++i];
        } else if (!strcmp(s, "--max-voice-sec") && val) {
            a.max_voice_sec = (float) atof(argv[++i]);
        } else if (!strcmp(s, "--seed") && val) {
            a.tts.seed = (uint32_t) strtoul(argv[++i], nullptr, 10);
        } else if (!strcmp(s, "--temp") && val) {
            a.tts.temperature = (float) atof(argv[++i]);
        } else if (!strcmp(s, "--lsd-steps") && val) {
            a.tts.lsd_steps = atoi(argv[++i]);
        } else if (!strcmp(s, "--eos-threshold") && val) {
            a.tts.eos_threshold = (float) atof(argv[++i]);
        } else if (!strcmp(s, "--frames-after-eos") && val) {
            a.tts.frames_after_eos = atoi(argv[++i]);
        } else if (!strcmp(s, "--max-chunk-tokens") && val) {
            a.tts.max_chunk_tokens = atoi(argv[++i]);
        } else {
            return false;
        }
    }
    return a.model && a.voice;
}

static std::string read_stdin_text() {
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    std::string s = ss.str();
    utf8_normalize(s);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage(argv[0]);
        return 1;
    }
    WavFormat fmt;
    if (!audio_parse_format(a.format, fmt)) {
        fprintf(stderr, "[CLI] ERROR: invalid --format '%s' (expected wav16, wav24, wav32)\n", a.format);
        return 1;
    }

    pt_init_params ip;
    pt_init_default_params(&ip);
    ip.model_path  = a.model;
    ip.use_fa      = a.use_fa;
    ip.clamp_fp16  = a.clamp_fp16;
    pt_context * c = pt_init(&ip);
    if (!c) {
        fprintf(stderr, "[CLI] ERROR: %s\n", pt_last_error());
        return 1;
    }
    // A name without a file extension is a predefined voice of --voices.
    const char * ext  = strrchr(a.voice, '.');
    const bool   file = ext && (!strcmp(ext, ".wav") || !strcmp(ext, ".safetensors"));
    if (!file && !a.voices) {
        fprintf(stderr, "[CLI] ERROR: --voice %s needs --voices\n", a.voice);
        pt_free(c);
        return 1;
    }
    pt_voice * v = file ? pt_voice_load(c, a.voice, a.max_voice_sec) : pt_voice_named(c, a.voices, a.voice);
    if (!v) {
        fprintf(stderr, "[CLI] ERROR: %s\n", pt_last_error());
        pt_free(c);
        return 1;
    }
    if (a.export_voice) {
        if (pt_voice_save(v, a.export_voice) != PT_STATUS_OK) {
            fprintf(stderr, "[CLI] ERROR: %s\n", pt_last_error());
            pt_voice_free(v);
            pt_free(c);
            return 1;
        }
        fprintf(stderr, "[CLI] Voice state -> %s\n", a.export_voice);
    }

    const bool  stream    = !strcmp(a.out, "-");
    const bool  line_mode = a.stream_by_line && stream;
    std::string text;
    if (!line_mode) {
        text = read_stdin_text();
        if (text.empty()) {
            fprintf(stderr, "[CLI] ERROR: stdin is empty, nothing to synthesize\n");
            pt_voice_free(v);
            pt_free(c);
            return 1;
        }
    }
    a.tts.text  = text.c_str();
    a.tts.voice = v;

    int rc = 0;
    if (stream) {
        wav_stream ws = {};
        if (!wav_stream_open_stdout(&ws, pt_sample_rate(c), fmt)) {
            rc = 1;
        } else {
            a.tts.on_chunk = [](const float * s, int n, void * ud) -> bool {
                return wav_stream_write((wav_stream *) ud, s, n);
            };
            a.tts.user_data = &ws;
            if (line_mode) {
                // One utterance per line; every line after the first opens
                // with a fresh RIFF header on the same stream.
#if defined(_WIN32)
                _setmode(_fileno(stdin), _O_BINARY);
#endif
                char        buf[4096];
                std::string line;
                bool        need_header = false;
                bool        eof         = false;
                while (!eof && rc == 0) {
                    eof = fgets(buf, sizeof(buf), stdin) == nullptr;
                    if (!eof) {
                        line += buf;
                    }
                    if ((line.empty() || line.back() != '\n') && !eof) {
                        continue;
                    }
                    utf8_normalize(line);
                    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                        line.pop_back();
                    }
                    if (!line.empty()) {
                        if (need_header && !wav_stream_write_header(&ws)) {
                            rc = 1;
                            break;
                        }
                        a.tts.text = line.c_str();
                        if (pt_synthesize(c, &a.tts, nullptr) != PT_STATUS_OK) {
                            fprintf(stderr, "[CLI] ERROR: %s\n", pt_last_error());
                            rc = 1;
                        }
                        need_header = true;
                    }
                    line.clear();
                }
            } else if (pt_synthesize(c, &a.tts, nullptr) != PT_STATUS_OK) {
                fprintf(stderr, "[CLI] ERROR: %s\n", pt_last_error());
                rc = 1;
            }
            wav_stream_close(&ws);
        }
    } else {
        pt_audio audio = {};
        if (pt_synthesize(c, &a.tts, &audio) != PT_STATUS_OK) {
            fprintf(stderr, "[CLI] ERROR: %s\n", pt_last_error());
            rc = 1;
        } else if (!audio_write_wav(a.out, audio.samples, audio.n_samples, audio.sample_rate, fmt)) {
            fprintf(stderr, "[CLI] ERROR: cannot write %s\n", a.out);
            rc = 1;
        } else {
            fprintf(stderr, "[CLI] Wrote %d samples (%.2f s) -> %s\n", audio.n_samples,
                    (double) audio.n_samples / audio.sample_rate, a.out);
        }
        pt_audio_free(&audio);
    }
    pt_voice_free(v);
    pt_free(c);
    return rc;
}
