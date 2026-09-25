#!/usr/bin/env python3
"""Cossim debug: C++ pocket-tts vs Python Pocket TTS with a predefined voice.

Inputs (relative to CWD = tests/):
    ../examples/prompt.txt              target text fed to both pipelines
    ../checkpoints/pocket-tts           reference weights and voice states
    ../models/pocket-tts-<pack>-*.gguf  C++ model and voices

Both sides draw the flow noise from the same seed (torch.manual_seed on
the reference, the torch compatible generator on the C++ side).

Dumps land in cpp/tts/ (C++) and python/tts/ (Python).
"""

import cossim_common as cc

VOICE = "alba"

if __name__ == "__main__":
    cc.run("tts", VOICE, lambda pack: ["--voices", f"../models/pocket-tts-{pack}-voices.gguf", "--voice", VOICE])
