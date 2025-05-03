#!/bin/bash
#SBATCH --job-name=cpu_benchmark
#SBATCH --output=cpu_benchmark_%j.out
#SBATCH --error=cpu_benchmark_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=instruction

# Load modules
module load nvidia/cuda/12.2.0
module load gcc/11.3.0
module load conda/miniforge/23.1.0

source ~/.bashrc
conda activate myenv || conda create -n myenv python=3.10 -y && conda activate myenv

# Install Python packages (once)
pip install --upgrade pip
pip install torch torchvision numpy

# Confirm environment
python -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"

# Build CUDA extension
python setup.py build_ext --inplace


# Run the benchmark
python benchmark_pytorch_vs_custom.py 