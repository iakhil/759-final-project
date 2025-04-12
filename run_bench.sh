#!/bin/bash
#SBATCH --job-name=mnist_benchmark
#SBATCH --output=mnist_benchmark.out
#SBATCH --error=mnist_benchmark.err
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Create and activate virtual environment if it doesn't exist
VENV_DIR=venv
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  source $VENV_DIR/bin/activate
  pip install --upgrade pip

  # Install minimal dependencies
  pip3 install -r requirements.txt
else
  source $VENV_DIR/bin/activate
fi

# Fix: Set CUDA_HOME (adjust this if needed)
export CUDA_HOME=/usr/local/cuda

# Confirm PyTorch and NumPy
python -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"

# Build the CUDA extension
python setup.py build_ext --inplace

# Run your benchmark
python benchmark_layer.py
