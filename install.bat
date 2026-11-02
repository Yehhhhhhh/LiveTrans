@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [LiveTrans] 创建 Python 虚拟环境 .venv ...
if not exist .venv (
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
    echo 创建虚拟环境失败，请确认已安装 Python 3.12
    pause
    exit /b 1
)

echo [LiveTrans] 安装依赖（含 CUDA/cuDNN 运行库，约 1.5GB）...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r backend\requirements.txt
if errorlevel 1 (
    echo 依赖安装失败
    pause
    exit /b 1
)

echo.
echo 安装完成。运行 start.bat 启动（首次启动会下载约 1.6GB 的 Whisper 模型）。
pause
