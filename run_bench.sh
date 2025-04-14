#!/bin/bash
#SBATCH --job-name=mnist_benchmark
#SBATCH --output=mnist_benchmark.out
#SBATCH --error=mnist_benchmark.err
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

module load nvidia/cuda/11.8.0
VENV_DIR=$HOME/759-final-project/.venv
PYTHON=$VENV_DIR/bin/python
PIP=$VENV_DIR/bin/pip

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  $VENV_DIR/bin/pip install --upgrade pip
  $VENV_DIR/bin/pip install -r requirements.txt
fi

# Fix: Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda

# Confirm libraries
$PYTHON -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"

# Build CUDA extension
$PYTHON setup.py build_ext --inplace

# Run benchmark
$PYTHON benchmark_layer.py
