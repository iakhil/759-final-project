#!/bin/bash
#SBATCH --job-name=train_benchmark
#SBATCH --output=train_benchmark_%j.out
#SBATCH --error=train_benchmark_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1

# Load modules
module load nvidia/cuda/12.2.0
module load gcc/11.3.0
module load conda/miniforge/23.1.0

source ~/.bashrc
conda activate myenv || conda create -n myenv python=3.10 -y && conda activate myenv

# Install Python packages (once)
pip install --upgrade pip
pip install torch torchvision numpy matplotlib

# Confirm environment and CUDA availability
python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count())"

# Run the benchmark comparing PyTorch on CPU vs Custom on GPU - training metrics
python benchmark_inference_cpu.py 