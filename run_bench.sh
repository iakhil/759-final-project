#!/bin/bash
#SBATCH --job-name=mnist_benchmark
#SBATCH --output=mnist_benchmark.out
#SBATCH --error=mnist_benchmark.err
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Clean any conflicting variables
unset CUDA_VISIBLE_DEVICES
unset CUDA_HOME

# Load correct CUDA
module purge
module load nvidia/cuda/12.2.0

# Define env
VENV_DIR=$HOME/759-final-project/.venv
PYTHON=$VENV_DIR/bin/python
PIP=$VENV_DIR/bin/pip

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc path: $(which nvcc)"
nvidia-smi

# Setup venv
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  $PIP install --upgrade pip
  $PIP install -r requirements.txt
fi

# Confirm torch sees GPU
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Build extension
$PYTHON setup.py build_ext --inplace

# Run code
$PYTHON benchmark_layer.py
