@echo off
rem NOTE: keep this file pure ASCII with CRLF line endings. cmd.exe misparses
rem UTF-8 multibyte text in .bat files (runs fragments of comments as commands).
cd /d "%~dp0"

echo [LiveTrans] Creating Python virtual environment .venv ...
if not exist .venv (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo [LiveTrans] Failed to create the virtual environment. Make sure Python 3.12 is installed.
    pause
    exit /b 1
)

echo [LiveTrans] Installing dependencies (incl. CUDA/cuDNN runtime, about 1.5GB) ...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [LiveTrans] Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo [LiveTrans] Done. Run start.bat to launch (first launch downloads the ~1.6GB Whisper model).
pause
