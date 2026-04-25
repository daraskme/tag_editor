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

if exist "venv\.repaired_v5" goto :check_deps

echo [REPAIR] Starting environment fix (First time only - DirectML transition)...
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision torchaudio onnxruntime onnxruntime-gpu onnxruntime-directml transformers tokenizers dghs-imgutils
echo [REPAIR] Installing Torch with CUDA 12.1...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo [REPAIR] Installing stable ONNX Runtime (DirectML)...
python -m pip install onnxruntime-directml
echo [REPAIR] Installing remaining dependencies...
python -m pip install -r requirements.txt
echo done > "venv\.repaired_v5"
echo [REPAIR] Setup complete!

:check_deps
echo [INFO] Checking dependencies...
python -m pip install --upgrade pip -q
python -m pip install "numpy>=2.0.0" --only-binary :all: --no-cache-dir -q
python -m pip install PyQt6 Pillow huggingface-hub einops timm opencv-python onnxruntime-directml --only-binary :all: -q
python -m pip install dghs-imgutils -q
python -m pip install torch --only-binary :all: -q
python -m pip install transformers -q

:start_app
echo Starting AI Tag Editor...
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed.
    pause
)
