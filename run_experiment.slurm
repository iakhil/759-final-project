#!/bin/bash
#SBATCH --job-name=mlp_cpu_experiment
#SBATCH --output=mlp_cpu_results.out
#SBATCH --error=mlp_cpu_errors.err
#SBATCH --time=00:30:00
#SBATCH --partition=instruction
#SBATCH --mem=8G

# Print node information
echo "Running on host: $(hostname)"

# Load required modules
echo "Loading modules..."
module purge
module load gcc/11.3.0
module load conda/miniforge/23.1.0

# Activate conda environment without relying on .bashrc
echo "Setting up Conda environment..."
eval "$(conda shell.bash hook)"
conda activate myenv 2>/dev/null || conda create -n myenv python=3.10 -y && conda activate myenv

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch torchvision numpy pandas scikit-learn

# Confirm environment
python -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"

# Run standard MLP
echo "Running standard MLP on CPU..."
python mlp.py --epochs 5 --batch_size 64 --device cpu > standard_mlp_cpu_output.txt

# Summary
echo "===== CPU EXPERIMENT SUMMARY ====="
echo "Standard MLP results (CPU):"
grep "STANDARD MLP PERFORMANCE" -A 5 standard_mlp_cpu_output.txt || echo "No results"

echo "Completed at $(date)"
