@echo off

set pocket=build\Release\pocket-tts.exe

REM F32 BF16 Q8_0 Q4_K_M
set type=F32
for %%b in (F32 BF16 Q8_0 Q4_K_M) do (
    call :process_type %%b
)
goto :eof

:process_type
set type=%1
for %%f in ("models\pocket-tts-english_2m*-%type%.gguf" "models\pocket-tts-english_7m*-%type%.gguf" "models\pocket-tts-english_18m*-%type%.gguf" "models\pocket-tts-english_2026-*-%type%.gguf") do (
REM for %%f in ("models\pocket-tts-english_*-%type%.gguf") do (
    echo [Processing] %%f
    if not exist "%%f" (
        continue
    )    
    call :clone %%f %type%

)
goto :eof

:clone
set model=%1
set type=%2
call set outname=%~nx1
call set out=%%outname:-%type%.gguf=-%type%.wav%%
echo on

    %pocket% ^
        --device cuda0 ^
        --temp 0.3 ^
        --model %model% ^
        --voice examples\freeman.wav ^
        -o tests\output\cuda\%out%_3_cuda.wav < examples\prompt.txt


@echo off

:eof
