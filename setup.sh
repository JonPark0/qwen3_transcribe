#!/bin/bash

echo "Setting up Qwen3 Transcriber..."
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core requirements..."
pip install -r requirements.txt

echo ""
echo "Flash Attention 2 is optional and can be installed separately for GPU acceleration."
echo "To install: pip install ninja psutil && pip install flash-attn --no-build-isolation"
echo "Note: This requires a compatible GPU and may take 5-10 minutes to compile."
echo ""

echo ""
echo "Setup complete!"
echo ""
echo "To use the transcriber:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the transcriber: python3 convert.py -i input.mp3 -o output_folder/"
echo ""
echo "For help: python3 convert.py --help"