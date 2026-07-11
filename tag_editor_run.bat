@echo off
chcp 65001 > nul
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1

echo Verifying virtual environment...
if exist "venv\Scripts\python.exe" goto :activate
echo Creating virtual environment...
python -m venv venv

:activate
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Activation failed.
    pause
    exit /b 1
)

fc /b requirements.txt "venv\.reqs_installed" >nul 2>&1
if not errorlevel 1 goto :check_quick

echo [REPAIR] Installing/updating dependencies...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt --only-binary :all: -q
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
copy /y requirements.txt "venv\.reqs_installed" >nul
echo [REPAIR] Setup complete!

:check_quick
python -c "import onnxruntime, onnx, PyQt6, PIL, huggingface_hub, numpy, requests" 2>nul
if not errorlevel 1 goto :start_app

echo [INFO] Core packages missing, installing dependencies...
python -m pip install -r requirements.txt --only-binary :all: -q

:start_app
echo Starting AI Tag Editor...
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed.
    pause
)
