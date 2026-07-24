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

# 2. Install PyTorch explicitly first (to match MMCV 1.6.0 constraints)
echo "Installing compatible PyTorch version..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Install Python dependencies (with fixed numpy version to avoid np.bool crash)
echo "Installing Python dependencies..."
pip install -r EgoHOS/requirements.txt
pip install "numpy<1.24.0"  # CRITICAL FIX for mmsegmentation np.bool errors
pip install -U openmim
mim install mmcv-full==1.6.0

# 4. Install MMSegmentation
echo "Installing MMSegmentation..."
cd EgoHOS/mmsegmentation
pip install -v -e .
cd ../..

# 5. Download Checkpoints (Weights)
echo "Downloading EgoHOS Model Weights..."
cd EgoHOS
bash download_checkpoints.sh
cd ..

echo "====================================="
echo " Installation Complete! "
echo " EgoHOS and all weights are ready."
echo "====================================="
