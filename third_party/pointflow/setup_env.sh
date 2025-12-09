#!/bin/bash

# 1. Delete existing environment
echo "Deleting existing PointFlow environment..."
conda deactivate || true
conda env remove -n PointFlow -y

# 2. Create new environment with Python 3.9
echo "Creating new PointFlow environment..."
conda create -n PointFlow python=3.9 -y

# 3. Install PyTorch (Latest Stable) with CUDA 12.1 (Compatible with A100/H100 and TACC modules)
# We use -n PointFlow to explicitly target the environment, avoiding activation issues
echo "Installing PyTorch and dependencies..."
conda install -n PointFlow pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install other dependencies
conda install -n PointFlow matplotlib tqdm scikit-learn scipy numpy -y

# Use conda run to execute pip inside the environment
echo "Installing pip packages..."
conda run -n PointFlow pip install pillow tensorboardX torchdiffeq

# 5. Compile Custom CUDA Kernels
echo "Compiling Custom CUDA Kernels..."
cd metrics/pytorch_structural_losses/
make clean
# Run make inside the conda environment so 'python' refers to the right one
# We also need to load a newer GCC (9+) for PyTorch 2.0+ compatibility
module load gcc/9.4.0
conda run -n PointFlow make

echo "Setup complete!"
