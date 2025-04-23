#!/bin/bash
#SBATCH --job-name=mlp_tuner_experiment
#SBATCH --output=mlp_tuner_results.out
#SBATCH --error=mlp_tuner_errors.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Print node information
echo "Running on host: $(hostname)"
echo "CUDA devices:"
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

# Now run both models:
# 1. First predict optimal block sizes using the tuner model
echo "Predicting optimal block sizes..."
python run_prediction.py > block_prediction.txt

# Extract predicted block sizes
BLOCK_X=$(grep "block_x=" block_prediction.txt | tail -1 | awk '{print $3}' | cut -d'=' -f2 | cut -d',' -f1)
BLOCK_Y=$(grep "block_y=" block_prediction.txt | tail -1 | awk '{print $4}' | cut -d'=' -f2)

echo "Using block sizes: $BLOCK_X, $BLOCK_Y"

# 2. Run the standard MLP model
echo "Running standard MLP model..."
python mlp.py --epochs 5 --batch_size 64 > standard_mlp_output.txt

# 3. Run the custom MLP model with predicted block sizes
echo "Running custom MLP model with predicted block sizes..."
python mlp_with_custom_layer.py --epochs 5 --batch_size 64 --block_x $BLOCK_X --block_y $BLOCK_Y > custom_mlp_output.txt

# Print summary of results
echo "===== EXPERIMENT SUMMARY ====="
echo "Standard MLP results:"
grep "STANDARD MLP PERFORMANCE" -A 5 standard_mlp_output.txt

echo "Custom MLP results:"
grep "CUSTOM MLP PERFORMANCE" -A 5 custom_mlp_output.txt

echo "Experiment completed at $(date)"
