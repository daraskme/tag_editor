#!/bin/bash
echo "Setting up AI Tag Editor..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "Installing dependencies... (This may take a while)"
# We need to install the correct torch version for CUDA if available, but for generic Linux/Windows we use standard pip
pip install -r requirements.txt
echo "Starting application..."
python main.py
