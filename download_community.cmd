@echo off
rem Download community-trained Pocket TTS checkpoints from HuggingFace into
rem checkpoints\community\<repo>. Only conversion inputs are pulled (YAMLs, F32
rem safetensors at root and one level down, tokenizer.json / *.model).
rem Usage: download_community.cmd [repo ...]   (no args = all repos below)

setlocal enabledelayedexpansion
rem Anchor to the script's own folder so relative paths work from any CWD.
cd /d "%~dp0"
set "DIR=checkpoints\community"
rem Prefer a venv-local hf.exe ; fall back to hf.
if exist ".venv\Scripts\hf.exe" (set "HF=.venv\Scripts\hf.exe") else (set "HF=hf")
echo [hf] %HF%
if not exist "%DIR%" mkdir "%DIR%"

REM Repositories to download from HuggingFace
REM from the list of community-trained models in the README from https://github.com/kyutai-labs/pocket-tts
REM https://huggingface.co/vvolhejn/pocket-tts-czech Czech
REM https://huggingface.co/saryps-labs/pocket-tts-hindi Hindi
REM https://huggingface.co/seastar105/pocket-tts-korean-300m Korean
REM https://huggingface.co/mehdi-hf/pocket-tts-farsi  Persian script input
REM https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2  Updated Farsi model requiring phonemes
REM https://huggingface.co/anak10thn/pocket-tts-indonesian Indonesian
REM https://huggingface.co/cbentes/pocket-tts-estonian Estonian
REM https://huggingface.co/EryriLabs/pocket-tts-cymraeg Welsh
REM https://huggingface.co/alperiox/pocket-tts-turkish Turkish
REM https://huggingface.co/shefowl/pocket-tts-polish-6l Polish
REM https://huggingface.co/myned-ai/pocket-tts-greek Greek
REM https://huggingface.co/takuma104/pocket-tts-ja Japanese
REM https://huggingface.co/Thanushttk/pocket-tts-multilingual English, Hindi, Marathi, Kannada, Tamil, Malayalam, and Telugu
REM https://huggingface.co/hustfyb/PocketTTS-ZH-6L Chinese / Mandarin
REM https://huggingface.co/genvoice/xVibePocketTTS Russian - needs text stress processing "RUAccent"
REM https://huggingface.co/amekhrishvili/pocket-tts-ru Russian - needs text stress processing "silero_stress"

call :download "vvolhejn/pocket-tts-czech"
REM call :download "saryps-labs/pocket-tts-hindi" skipping for multilingual
call :download "seastar105/pocket-tts-korean-300m"
call :download "mehdi-hf/pocket-tts-farsi"
REM call :download "mehdi-hf/pocket-tts-farsi-v2" update for farsi but requires phonemes
call :download "anak10thn/pocket-tts-indonesian"
call :download "cbentes/pocket-tts-estonian"
call :download "EryriLabs/pocket-tts-cymraeg"
call :download "alperiox/pocket-tts-turkish"
call :download "shefowl/pocket-tts-polish-6l"
call :download "myned-ai/pocket-tts-greek"
call :download "takuma104/pocket-tts-ja"
call :download "Thanushttk/pocket-tts-multilingual"
call :download "hustfyb/PocketTTS-ZH-6L"
call :download "genvoice/xVibePocketTTS"
call :download "amekhrishvili/pocket-tts-ru"
goto :eof

:download
set "REPO=%~1"
for /f "tokens=*" %%a in ('echo %REPO%') do set "NAME=%%~na"
set "TARGET=%DIR%\%NAME%"
echo [Info] Processing %NAME%
set "HAVE_WEIGHTS="
if exist "%TARGET%\*.safetensors" set "HAVE_WEIGHTS=1"
if exist "%TARGET%\*.pt" set "HAVE_WEIGHTS=1"
if defined HAVE_WEIGHTS (
    echo   [OK] %NAME% -present
    exit /b 0
)
echo   [Download] %NAME% ^<-%REPO%
"%HF%" download %REPO% --local-dir "%TARGET%" --quiet
exit /b 0

 
:eof