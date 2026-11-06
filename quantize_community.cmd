@echo off
REM Derive lighter GGUFs from the F32 source-of-truth produced by convert.py.
REM Each F32 model under models/ is quantized to BF16, Q8_0 and Q4_K_M.

REM Q8_0 is the default: BF16 keeps the reference weights exactly (every
REM released pack ships BF16), Q4_K_M is the smallest variant. The policy
REM lives in tools/quantize.cpp should_quantize: only 2D matmul weights are
REM quantized, conv kernels and vectors keep F32.
REM // Types: Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0

set Q=.\build\Release\quantize

for /D %%I in ("models\community\pocket-tts-*" "models\community\PocketTTS-ZH*" "models\community\xVibePocketTTS") do (
    for %%f in ("%%I\*-F32.gguf") do (
        echo [Processing] %%f
        if not exist "%%f" (
            continue
        )    
        call :quantize "%%f" BF16
        call :quantize "%%f" Q8_0
        call :quantize "%%f" Q4_K_M
    )
)
goto :eof

:quantize
set src=%1
set type=%2

call set out=%%src:-F32.gguf=-%type%.gguf%%

if exist "%out%" (
    echo [Skip] %out%
) else (
    echo [Quantize] %src% to %out% with type %type%
    %Q% %src% %out% %type%
)

:eof