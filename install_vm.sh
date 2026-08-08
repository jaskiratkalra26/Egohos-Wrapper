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

# 2. Check PyTorch version
echo "Checking PyTorch..."
python -c "import torch" 2>/dev/null || pip install torch torchvision

# 3. Install Python dependencies (with fixed numpy version to avoid np.bool crash)
echo "Installing Python dependencies..."
sed -i -E '/^(torch|torchvision)==/d' EgoHOS/requirements.txt
pip install -r EgoHOS/requirements.txt
pip install -U openmim
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html || pip install mmcv-full==1.7.0 || pip install mmcv

# 4. Install MMSegmentation
echo "Installing MMSegmentation..."
cd EgoHOS/mmsegmentation
# Patch the hardcoded mmcv max version to allow 1.7.0 and import torch first
sed -i "s/MMCV_MAX = .*/MMCV_MAX = '1.8.0'/g" mmseg/__init__.py
sed -i "1s/^/import torch\n/" predict_image.py
pip install -v -e .
cd ../..
cp predict_all.py EgoHOS/mmsegmentation/predict_all.py 2>/dev/null || true

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
