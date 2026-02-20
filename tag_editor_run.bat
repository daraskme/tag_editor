@echo off
echo Setting up AI Tag Editor...
if not exist "venv\" (
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing dependencies... (This may take a while)
pip install -r requirements.txt
echo Starting application...
python main.py
