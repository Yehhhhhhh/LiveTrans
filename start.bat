@echo off
chcp 65001 >nul
rem NOTE: keep this file pure ASCII with CRLF line endings. cmd.exe misparses
rem UTF-8 multibyte text in .bat files (runs fragments of comments as commands).
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [LiveTrans] .venv not found. Please run install.bat first.
    pause
    exit /b 1
)

rem Mainland China: uncomment to download models via the HuggingFace mirror
rem set HF_ENDPOINT=https://hf-mirror.com

rem Optional: change model (e.g. large-v3 / medium) or port
rem set LIVETRANS_MODEL=large-v3-turbo
rem set LIVETRANS_PORT=8765

rem Optional: save each session's raw audio to recordings\ (for debugging)
rem set LIVETRANS_RECORD=1

set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set LIVETRANS_OPEN_BROWSER=1
.venv\Scripts\python backend\server.py
pause
