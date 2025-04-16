#!/bin/bash
#SBATCH --job-name=mnist_benchmark
#SBATCH --output=mnist_benchmark.out
#SBATCH --error=mnist_benchmark.err
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G

# Clean any conflicting variables
unset CUDA_VISIBLE_DEVICES
unset CUDA_HOME

# Load compatible CUDA version
module purge
module load nvidia/cuda/12.2.0

# Load compatible GCC version for CUDA 12.2
module load gcc/11.2.0
export CC=gcc
export CXX=g++

# Define virtual environment paths
VENV_DIR=$HOME/759-final-project/.venv
PYTHON=$VENV_DIR/bin/python
PIP=$VENV_DIR/bin/pip

# Set CUDA-related environment variables
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"  # Update if needed

# Ensure Python headers are found
PYTHON_INCLUDE_DIR=$($PYTHON -c "from sysconfig import get_paths; print(get_paths()['include'])")
export CFLAGS="-I$PYTHON_INCLUDE_DIR"

# Avoid using ninja (switch to Makefiles)
export CMAKE_GENERATOR="Unix Makefiles"

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc path: $(which nvcc)"
nvidia-smi

# Setup virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv $VENV_DIR
  $PIP install --upgrade pip

  # Install PyTorch with CUDA 12.2
  $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

  # Install other requirements
  $PIP install -r requirements.txt
fi

# Confirm torch sees GPU
$PYTHON -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Clean and build C++/CUDA extension
$PYTHON setup.py clean
$PYTHON setup.py build_ext --inplace --verbose

# Run your code
$PYTHON benchmark_layer.py
