#!/bin/bash
set -e

echo "====================================="
echo "   EgoHOS VM Installation Script     "
echo "====================================="

# 1. Clone EgoHOS
if [ ! -d "EgoHOS" ]; then
    echo "Cloning EgoHOS repository..."
    git clone https://github.com/owenzlz/EgoHOS
else
    echo "EgoHOS directory already exists, skipping clone."
fi

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install -r EgoHOS/requirements.txt
pip install -U openmim
mim install mmcv-full==1.6.0

# 3. Install MMSegmentation
echo "Installing MMSegmentation..."
cd EgoHOS/mmsegmentation
pip install -v -e .
cd ../..

# 4. Download Checkpoints (Weights)
echo "Downloading EgoHOS Model Weights..."
cd EgoHOS
bash download_checkpoints.sh
cd ..

echo "====================================="
echo " Installation Complete! "
echo " EgoHOS and all weights are ready."
echo "====================================="
