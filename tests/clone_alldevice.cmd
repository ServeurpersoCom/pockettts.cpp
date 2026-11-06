@echo off
setlocal enabledelayedexpansion

set pocket=build\Release\pocket-tts.exe

REM F32 BF16 Q8_0 Q4_K_M
set type=F32
for %%b in (F32 BF16 Q8_0 Q4_K_M) do (
  call :process_type %%b
)
goto :eof

:process_type
set type=%1
for %%f in ("models\pocket-tts-english_*-%type%.gguf") do (
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
set model_parent=%~n1


call set outname=%~nx1
call set out=%%outname:-%type%.gguf=-%type%.wav%%
call :StripLastPart "%model_parent%" model_parent_short
set out_folder=tests\output\%model_parent_short%


REM cpu cuda0 vulkan0 vulkan1
for %%d in (cpu cuda0 vulkan0 vulkan1) do (
  if not exist "%out_folder%\%%d" (
    mkdir "%out_folder%\%%d"
  )
  echo [Processing %out_name% using %type%]
  echo on
  
  %pocket% ^
  --device %%d ^
  --temp 0.3 ^
  --model %model% ^
  --voice examples\freeman.wav ^
  -o %out_folder%\%%d\%out% < examples\prompt.txt
  
  @echo off
)
goto :eof


:StripLastPart
set "str=%~1"
set "out_var=%2"
set "result="
set "last="
:: Split by hyphens
for %%A in ("%str:-=" "%") do (
  if defined last (
    if defined result (
      set "result=!result!-!last!"
      ) else (
      set "result=!last!"
    )
  )
  set "last=%%~A"
)
endlocal & set "%out_var%=%result%"
goto :eof

:eof
