@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo 请先运行 install.bat
    pause
    exit /b 1
)

rem 访问 HuggingFace 困难时取消下一行注释使用镜像
rem set HF_ENDPOINT=https://hf-mirror.com

rem 可选：更换模型（如 large-v3 / medium）或端口
rem set LIVETRANS_MODEL=large-v3-turbo
rem set LIVETRANS_PORT=8765

rem 可选：把每次采集的原始音频保存到 recordings\ 目录，用于排查识别问题
rem set LIVETRANS_RECORD=1

set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set LIVETRANS_OPEN_BROWSER=1
.venv\Scripts\python backend\server.py
pause
