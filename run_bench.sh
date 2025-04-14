#!/bin/bash
#SBATCH --job-name=mnist_benchmark
#SBATCH --output=mnist_benchmark.out
#SBATCH --error=mnist_benchmark.err
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

module load nvidia/nvhpc-hpcx-cuda12/24.5


VENV_DIR=$HOME/759-final-project/.venv
PYTHON=$VENV_DIR/bin/python
PIP=$VENV_DIR/bin/pip

# Create virtualenv if needed
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  $PIP install --upgrade pip
  $PIP install -r requirements.txt
fi

# Print CUDA info
echo "nvcc path: $(which nvcc)"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME: $CUDA_HOME"

# Confirm torch + CUDA
$PYTHON -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Build extension
$PYTHON setup.py build_ext --inplace

# Run benchmark
$PYTHON benchmark_layer.py
