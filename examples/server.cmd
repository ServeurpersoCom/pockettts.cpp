@echo off

set PATH=%~dp0..\build\Release;%PATH%

rem Multi-GPU: add --device <name> below (CUDA0, CUDA1, Vulkan0, CPU)

tts-server.exe ^
    --model ..\models\pocket-tts-english_2026-09-Q8_0.gguf ^
    --voices ..\models\pocket-tts-english_2026-09-voices.gguf ^
    --voice alba ^
    --alias pocket-tts ^
    --host 127.0.0.1 --port 8080

pause
