#!/bin/bash
echo "Setting up AI Tag Editor..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
echo "Installing dependencies... (This may take a while)"
pip install --upgrade pip -q
pip install -r requirements.txt
echo "Starting application..."
python main.py
