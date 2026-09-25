@echo off

set PATH=%~dp0..\build\Release;%PATH%

rem set GGML_BACKEND=Vulkan0

tts-server.exe ^
    --model ..\models\pocket-tts-english_2026-09-Q8_0.gguf ^
    --voices ..\models\pocket-tts-english_2026-09-voices.gguf ^
    --voice alba ^
    --alias pocket-tts ^
    --host 127.0.0.1 --port 8080

pause
