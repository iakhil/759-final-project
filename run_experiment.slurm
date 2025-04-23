#!/bin/bash
#SBATCH --job-name=mlp_tuner_experiment
#SBATCH --output=mlp_tuner_results.out
#SBATCH --error=mlp_tuner_errors.err
#SBATCH --time=00:30:00
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# Print node information
echo "Running on host: $(hostname)"
echo "CUDA devices:"
nvidia-smi

# Set up environment
echo "Setting up Python environment..."
module purge  # Clear any existing modules
module load python/3.8  
module load cuda/11.7.0  

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
# Run the prediction script and save its complete output
python run_prediction.py 2>&1 | tee prediction_output.txt

# Default block sizes in case extraction fails
BLOCK_X=16
BLOCK_Y=16

# Extract block sizes if possible - look for the PREDICTION_RESULT format
if grep -q "PREDICTION_RESULT:" prediction_output.txt; then
    PREDICTION_LINE=$(grep "PREDICTION_RESULT:" prediction_output.txt | tail -1)
    BLOCK_X=$(echo $PREDICTION_LINE | grep -o 'block_x=[0-9]*' | cut -d'=' -f2)
    BLOCK_Y=$(echo $PREDICTION_LINE | grep -o 'block_y=[0-9]*' | cut -d'=' -f2)
    echo "Successfully extracted block sizes from prediction output"
else
    echo "Could not extract block sizes from prediction output, using defaults"
fi

echo "Using block sizes: BLOCK_X=$BLOCK_X, BLOCK_Y=$BLOCK_Y"

# 2. Run the standard MLP model
echo "Running standard MLP model..."
python mlp.py --epochs 5 --batch_size 64 > standard_mlp_output.txt

# 3. Run the custom MLP model with predicted block sizes
echo "Running custom MLP model with predicted block sizes..."
python mlp_with_custom_layer.py --epochs 5 --batch_size 64 --block_x $BLOCK_X --block_y $BLOCK_Y > custom_mlp_output.txt

# Print summary of results
echo "===== EXPERIMENT SUMMARY ====="
echo "Block sizes used: BLOCK_X=$BLOCK_X, BLOCK_Y=$BLOCK_Y"
echo "Standard MLP results:"
grep "STANDARD MLP PERFORMANCE" -A 5 standard_mlp_output.txt

echo "Custom MLP results:"
grep "CUSTOM MLP PERFORMANCE" -A 5 custom_mlp_output.txt

echo "Experiment completed at $(date)"
