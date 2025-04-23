#!/bin/bash
#SBATCH --job-name=mlp_tuner_experiment
#SBATCH --output=mlp_tuner_results.out
#SBATCH --error=mlp_tuner_errors.err
#SBATCH --time=00:30:00
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Load required modules
echo "Loading modules..."
module purge
module load nvidia/cuda/12.2.0
time module load gcc/11.3.0
module load conda/miniforge/23.1.0

# Activate conda environment
echo "Activating Conda environment..."
source ~/.bashrc
conda activate myenv || (conda create -n myenv python=3.10 -y && conda activate myenv)

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Confirm environment
python -c "import torch, numpy; print('Torch:', torch.__version__, '| NumPy:', numpy.__version__)"

# Build CUDA extension
echo "Building CUDA extension..."
cd "Single layer Working model"
python setup.py build_ext --inplace
cd ..

# Ensure Python can find compiled extension
export PYTHONPATH="$PYTHONPATH:$(pwd)/Single layer Working model"

# 1. Predict optimal block sizes
echo "Predicting optimal block sizes..."
python run_prediction.py 2>&1 | tee prediction_output.txt

# Set defaults
BLOCK_X=16
BLOCK_Y=16
if grep -q "PREDICTION_RESULT:" prediction_output.txt; then
  PREDICTION_LINE=$(grep "PREDICTION_RESULT:" prediction_output.txt | tail -1)
  BLOCK_X=$(echo $PREDICTION_LINE | grep -o 'block_x=[0-9]*' | cut -d'=' -f2)
  BLOCK_Y=$(echo $PREDICTION_LINE | grep -o 'block_y=[0-9]*' | cut -d'=' -f2)
  echo "Extracted block sizes: $BLOCK_X, $BLOCK_Y"
else
  echo "Using default block sizes: $BLOCK_X, $BLOCK_Y"
fi

echo "Using block sizes: BLOCK_X=$BLOCK_X, BLOCK_Y=$BLOCK_Y"

# 2. Run standard MLP
echo "Running standard MLP..."
python mlp.py --epochs 5 --batch_size 64 > standard_mlp_output.txt

# 3. Run custom MLP
echo "Running custom MLP with block_x=$BLOCK_X, block_y=$BLOCK_Y..."
python mlp_with_custom_layer.py --epochs 5 --batch_size 64 --block_x $BLOCK_X --block_y $BLOCK_Y > custom_mlp_output.txt

# Summary
echo "===== EXPERIMENT SUMMARY ====="
echo "Block sizes: BLOCK_X=$BLOCK_X, BLOCK_Y=$BLOCK_Y"
echo "Standard MLP results:"
grep "STANDARD MLP PERFORMANCE" -A 5 standard_mlp_output.txt || echo "No results"

echo "Custom MLP results:"
grep "CUSTOM MLP PERFORMANCE" -A 5 custom_mlp_output.txt || echo "No results"

echo "Completed at $(date)"
