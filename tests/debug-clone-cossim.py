#!/usr/bin/env python3
"""Cossim debug: C++ pocket-tts vs Python Pocket TTS cloning a recording.

Inputs (relative to CWD = tests/):
    ../examples/prompt.txt              target text fed to both pipelines
    ../examples/freeman.wav             voice recording, encoded by both Mimi encoders
    ../checkpoints/pocket-tts           reference weights
    ../models/pocket-tts-<pack>-*.gguf  C++ model

The recording is 22.05 kHz: the reference resamples with scipy
resample_poly, the C++ side with a windowed sinc, so the voice stages carry
the resampler difference on top of the encoder.

Dumps land in cpp/clone/ (C++) and python/clone/ (Python).
"""

import pathlib

import cossim_common as cc

VOICE = "../examples/freeman.wav"

if __name__ == "__main__":
    cc.run("clone", pathlib.Path(VOICE), lambda pack: ["--voice", VOICE])
