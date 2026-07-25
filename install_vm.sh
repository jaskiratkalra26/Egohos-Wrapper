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

# 2. Install PyTorch explicitly first (upgraded to 1.13.1+cu117 for Ada Lovelace / L4 GPU support)
echo "Installing compatible PyTorch version..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 3. Install Python dependencies (with fixed numpy version to avoid np.bool crash)
echo "Installing Python dependencies..."
pip install -r EgoHOS/requirements.txt
pip install -U openmim
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# 4. Install MMSegmentation
echo "Installing MMSegmentation..."
cd EgoHOS/mmsegmentation
# Patch the hardcoded mmcv max version to allow 1.7.0
sed -i "s/mmcv_maximum_version.*/mmcv_maximum_version = '1.8.0'/g" mmseg/__init__.py
pip install -v -e .
cd ../..

# CRITICAL: Force numpy downgrade at the very end so other packages don't sneakily upgrade it
pip install "numpy<1.24.0"

# 5. Download Checkpoints (Weights)
echo "Downloading EgoHOS Model Weights..."
cd EgoHOS
bash download_checkpoints.sh
cd ..

echo "====================================="
echo " Installation Complete! "
echo " EgoHOS and all weights are ready."
echo "====================================="
