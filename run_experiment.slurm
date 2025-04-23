#!/bin/bash
#SBATCH --job-name=mlp_tuner
#SBATCH --output=mlp_tuner_results.out
#SBATCH --error=mlp_tuner_errors.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Print node information
echo "Running on host: $(hostname)"
echo "GPU information:"
nvidia-smi

# Set up environment
echo "Setting up Python environment..."
module purge  # Clear any existing modules
module load python/3.8  # Adjust as needed for your cluster
module load cuda/11.7.0  # Adjust version as needed for your cluster

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Build CUDA extension
echo "Building CUDA extension..."
python setup.py install

# Run prediction experiment
echo "Running prediction experiment..."
python run_prediction_experiments.py --epochs 2 --batch_size 64

echo "Job completed at $(date)"
