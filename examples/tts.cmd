@echo off

set PATH=%~dp0..\build\Release;%PATH%

pocket-tts.exe ^
    --model ..\models\pocket-tts-english_2026-09-Q8_0.gguf ^
    --voices ..\models\pocket-tts-english_2026-09-voices.gguf ^
    --voice alba ^
    -o tts.wav < prompt.txt

pause
