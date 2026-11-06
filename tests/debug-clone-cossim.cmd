@echo off
setlocal

set POCKETTTS_UPSTREAM_ROOT=..\..\..\pocket-tts
set POCKETTTS_CKPT=..\..\..\pocket-tts\models\pocket-tts
set POCKETTTS_BIN=..\build\Release\pocket-tts.exe

if not exist %POCKETTTS_UPSTREAM_ROOT% (
    echo POCKETTTS_UPSTREAM_ROOT not found
    exit /b 1
)
if not exist %POCKETTTS_CKPT% (
    echo POCKETTTS_CKPT not found
    exit /b 1
)
if not exist %POCKETTTS_BIN% (
    echo pocket-tts.exe not found
    exit /b 1
)

for %%b in (CUDA0 Vulkan0 CPU) do (
    for %%q in (F32 BF16 Q8_0 Q4_K_M) do (
        echo Running with device %%b and quant %%q
        ..\.venv\Scripts\python.exe ./debug-clone-cossim.py --device %%b --quant %%q > clone-%%b-%%q.log 2>&1
    )
)

endlocal
