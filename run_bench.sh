#!/bin/bash
#SBATCH --job-name=mnist_benchmark
#SBATCH --output=mnist_benchmark.out
#SBATCH --error=mnist_benchmark.err
#SBATCH --partition=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Load a CUDA version compatible with PyTorch (12.2 works with 12.4 PyTorch build)
module purge
module load nvidia/cuda/12.2.0

# Define paths
VENV_DIR=$HOME/759-final-project/.venv
PYTHON=$VENV_DIR/bin/python
PIP=$VENV_DIR/bin/pip

# Set CUDA paths
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "Using CUDA from: $CUDA_HOME"
echo "nvcc path: $(which nvcc)"

# Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  $PIP install --upgrade pip
  $PIP install -r requirements.txt
fi

# Confirm libraries and CUDA availability
$PYTHON -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Build CUDA extension
$PYTHON setup.py build_ext --inplace

# Run benchmark
$PYTHON benchmark_layer.py
