#!/bin/bash
#SBATCH --job-name=matmul_benchmark
#SBATCH --output=matmul_benchmark.out
#SBATCH --error=matmul_benchmark.err
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Load required modules
module load nvidia/cuda/12.2.0
module load gcc/11.3.0
module load conda/miniforge/23.1.0

# Activate conda and environment
source ~/.bashrc
conda activate myenv || conda create -n myenv python=3.10 -y && conda activate myenv

# Install Python packages (once)
pip install --upgrade pip
pip install torch torchvision numpy

# Confirm environment
python -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"

# Install dependencies (optional if already installed)
pip install --upgrade pip
pip install torch numpy

# Build CUDA extension
python setup.py build_ext --inplace

# Run the benchmark
python matmul_bench.py

