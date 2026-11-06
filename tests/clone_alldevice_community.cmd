@echo off

set pocket=build\Release\pocket-tts.exe

REM F32 BF16 Q8_0 Q4_K_M
for %%q in (F32 BF16 Q8_0 Q4_K_M) do (
  call :process_type %%q
)
goto :eof

:process_type
set type=%1
for /d %%c in (models\community\*) do (
  for %%f in ("%%c\*-%type%.gguf") do (
    if not exist "%%f" (
      continue
    )
    call :clone %%f %type%
    
  )
)
goto :eof

:clone
set model=%1
set type=%2
for %%A in ("%model%") do for %%B in ("%%~dpA.") do set "model_parent=%%~nxB"


call set outname=%~nx1
call set out=%%outname:-%type%.gguf=-%type%.wav%%

REM cpu cuda0 vulkan0 vulkan1
for %%d in (cpu cuda0 vulkan0 vulkan1) do (
  if not exist "tests\output\community\%model_parent%\%%d" (
    mkdir "tests\output\community\%model_parent%\%%d"
  )
  echo [Processing %model% using %type% on %%d]
  echo on
  %pocket% ^
  --device %%d ^
  --temp 0.3 ^
  --model %model% ^
  --voice examples\freeman.wav ^
  -o tests\output\community\%model_parent%\%%d\%out%_%%d.wav < examples\prompt.txt
  
  @echo off
)

:eof
