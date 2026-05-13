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

if exist "venv\.repaired_v7" goto :check_quick

echo [REPAIR] Starting environment fix (First time only)...
python -m pip install --upgrade pip -q
python -m pip uninstall -y torch torchvision torchaudio onnxruntime onnxruntime-gpu onnxruntime-directml transformers tokenizers dghs-imgutils 2>nul
echo [REPAIR] Installing ONNX Runtime (DirectML)...
python -m pip install onnxruntime-directml --only-binary :all: -q
echo [REPAIR] Installing core dependencies...
python -m pip install "numpy>=2.0.0" --only-binary :all: -q
python -m pip install PyQt6 Pillow huggingface-hub opencv-python --only-binary :all: -q
python -m pip install dghs-imgutils --no-deps -q
python -m pip install hbutils hfutils cheeseshop deprecation requests tqdm -q
python -m pip install pandas scikit-learn scipy shapely --only-binary :all: -q
python -m pip install "emoji<2.12,>=2.5.0" piexif pyrfc6266 urlobject -q
python -m pip install pilmoji pyclipper bchlib --only-binary :all: -q 2>nul
python -m pip install opencv-contrib-python --only-binary :all: -q
python -m pip install torch torchvision --only-binary :all: -q
echo done > "venv\.repaired_v7"
echo [REPAIR] Setup complete!

:check_quick
python -c "import torch, onnxruntime, imgutils, PyQt6, PIL, huggingface_hub" 2>nul
if not errorlevel 1 goto :start_app

echo [INFO] Core packages missing, installing dependencies...
python -m pip install "numpy>=2.0.0" --only-binary :all: -q
python -m pip install PyQt6 Pillow huggingface-hub opencv-python onnxruntime-directml --only-binary :all: -q
python -m pip install dghs-imgutils --no-deps -q
python -m pip install hbutils hfutils cheeseshop deprecation requests tqdm -q
python -m pip install pandas scikit-learn scipy shapely --only-binary :all: -q
python -m pip install "emoji<2.12,>=2.5.0" piexif pyrfc6266 urlobject -q
python -m pip install pilmoji pyclipper bchlib --only-binary :all: -q 2>nul
python -m pip install opencv-contrib-python --only-binary :all: -q
python -m pip install torch torchvision --only-binary :all: -q

:start_app
echo Starting AI Tag Editor...
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed.
    pause
)
